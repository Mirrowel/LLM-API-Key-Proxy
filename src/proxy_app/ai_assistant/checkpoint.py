"""
Checkpoint management for the AI Assistant.

Provides undo capability through a hybrid snapshot/delta storage system
with temp file persistence for crash recovery.
"""

import copy
import json
import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .context import apply_delta, compute_delta
from .tools import ToolCallSummary

logger = logging.getLogger(__name__)

# Full snapshot is stored every N checkpoints
SNAPSHOT_INTERVAL = 10


@dataclass
class Checkpoint:
    """Represents a point-in-time snapshot of the window state."""

    id: str  # UUID
    timestamp: datetime
    description: str  # Auto-generated from tools
    tool_calls: List[ToolCallSummary]  # What tools were called
    message_index: int  # Conversation position at checkpoint time

    # One of these will be populated:
    full_state: Optional[Dict[str, Any]] = None  # Full snapshot (every Nth)
    delta: Optional[Dict[str, Any]] = None  # Changes from previous

    is_full_snapshot: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "message_index": self.message_index,
            "full_state": self.full_state,
            "delta": self.delta,
            "is_full_snapshot": self.is_full_snapshot,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            description=data["description"],
            tool_calls=[
                ToolCallSummary(
                    name=tc["name"],
                    arguments=tc["arguments"],
                    success=tc["success"],
                    message=tc["message"],
                )
                for tc in data["tool_calls"]
            ],
            message_index=data["message_index"],
            full_state=data.get("full_state"),
            delta=data.get("delta"),
            is_full_snapshot=data.get("is_full_snapshot", False),
        )

    def get_display_text(self) -> str:
        """Get human-readable display text for UI."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        if self.tool_calls:
            first_call = self.tool_calls[0]
            tool_text = f'{first_call.name}("{list(first_call.arguments.values())[0] if first_call.arguments else ""}")'
            if len(self.tool_calls) > 1:
                tool_text += f" +{len(self.tool_calls) - 1} more"
            return f"{time_str} - {tool_text}"
        return f"{time_str} - {self.description}"


class CheckpointManager:
    """
    Manages checkpoints for undo capability.

    Uses a hybrid snapshot/delta approach:
    - Full snapshot stored every SNAPSHOT_INTERVAL checkpoints
    - Deltas stored between snapshots
    - All checkpoints persisted to temp file for crash recovery
    """

    def __init__(self, session_id: str):
        """
        Initialize the checkpoint manager.

        Args:
            session_id: Unique identifier for this session
        """
        self.session_id = session_id
        self._checkpoints: List[Checkpoint] = []
        self._current_position: int = (
            -1
        )  # Index of current checkpoint (-1 = no checkpoints)
        self._temp_file: Optional[Path] = None
        self._checkpoint_created_for_response: bool = False

        # Initialize temp file for crash recovery
        self._init_temp_file()

    def _init_temp_file(self) -> None:
        """Initialize the temp file for checkpoint persistence."""
        try:
            temp_dir = Path(tempfile.gettempdir()) / "ai_assistant_checkpoints"
            temp_dir.mkdir(exist_ok=True)
            self._temp_file = temp_dir / f"{self.session_id}.json"
            logger.info(f"Checkpoint temp file: {self._temp_file}")
        except Exception as e:
            logger.error(f"Failed to initialize temp file: {e}")
            self._temp_file = None

    def _save_to_temp_file(self) -> None:
        """Save checkpoints to temp file."""
        if not self._temp_file:
            return

        try:
            data = {
                "session_id": self.session_id,
                "current_position": self._current_position,
                "checkpoints": [cp.to_dict() for cp in self._checkpoints],
            }
            self._temp_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save checkpoints to temp file: {e}")

    def load_from_temp_file(self) -> bool:
        """
        Load checkpoints from temp file (for crash recovery).

        Returns:
            True if checkpoints were loaded, False otherwise
        """
        if not self._temp_file or not self._temp_file.exists():
            return False

        try:
            data = json.loads(self._temp_file.read_text())
            if data.get("session_id") != self.session_id:
                return False

            self._checkpoints = [Checkpoint.from_dict(cp) for cp in data["checkpoints"]]
            self._current_position = data["current_position"]
            logger.info(f"Loaded {len(self._checkpoints)} checkpoints from temp file")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoints from temp file: {e}")
            return False

    def clear_temp_file(self) -> None:
        """Delete the temp file."""
        if self._temp_file and self._temp_file.exists():
            try:
                self._temp_file.unlink()
            except Exception as e:
                logger.error(f"Failed to delete temp file: {e}")

    def start_response(self) -> None:
        """
        Called when a new LLM response begins.

        Resets the flag tracking whether a checkpoint was created for this response.
        """
        self._checkpoint_created_for_response = False

    def should_create_checkpoint(self) -> bool:
        """
        Check if a checkpoint should be created for this response.

        Returns:
            True if no checkpoint has been created for this response yet
        """
        return not self._checkpoint_created_for_response

    def create_checkpoint(
        self,
        state: Dict[str, Any],
        tool_calls: List[ToolCallSummary],
        message_index: int,
        description: Optional[str] = None,
    ) -> Checkpoint:
        """
        Create a new checkpoint.

        Args:
            state: The current window state to snapshot
            tool_calls: Summary of tool calls that led to this checkpoint
            message_index: Current position in conversation history
            description: Optional description (auto-generated if not provided)

        Returns:
            The created Checkpoint
        """
        checkpoint_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now()

        # Generate description if not provided
        if description is None:
            if tool_calls:
                first_call = tool_calls[0]
                description = f"{first_call.name} executed"
                if len(tool_calls) > 1:
                    description += f" (+{len(tool_calls) - 1} more)"
            else:
                description = "State saved"

        # Determine if this should be a full snapshot
        is_full = len(self._checkpoints) % SNAPSHOT_INTERVAL == 0

        if is_full:
            # Full snapshot
            checkpoint = Checkpoint(
                id=checkpoint_id,
                timestamp=timestamp,
                description=description,
                tool_calls=tool_calls,
                message_index=message_index,
                full_state=copy.deepcopy(state),
                is_full_snapshot=True,
            )
        else:
            # Delta from previous state
            previous_state = self._reconstruct_state(len(self._checkpoints) - 1)
            delta = compute_delta(previous_state, state) if previous_state else None

            checkpoint = Checkpoint(
                id=checkpoint_id,
                timestamp=timestamp,
                description=description,
                tool_calls=tool_calls,
                message_index=message_index,
                delta=delta if delta else {"added": [], "removed": [], "modified": []},
                is_full_snapshot=False,
            )

        self._checkpoints.append(checkpoint)
        self._current_position = len(self._checkpoints) - 1
        self._checkpoint_created_for_response = True

        # Persist to temp file
        self._save_to_temp_file()

        logger.info(
            f"Created checkpoint {checkpoint_id}: {description} "
            f"(full={is_full}, position={self._current_position})"
        )

        return checkpoint

    def _find_nearest_snapshot_before(self, index: int) -> int:
        """Find the nearest full snapshot at or before the given index."""
        for i in range(index, -1, -1):
            if self._checkpoints[i].is_full_snapshot:
                return i
        return -1

    def _reconstruct_state(self, target_index: int) -> Optional[Dict[str, Any]]:
        """
        Reconstruct the state at a given checkpoint index.

        Args:
            target_index: Index of the checkpoint to reconstruct

        Returns:
            The reconstructed state, or None if reconstruction fails
        """
        if target_index < 0 or target_index >= len(self._checkpoints):
            return None

        # Find nearest full snapshot
        snapshot_index = self._find_nearest_snapshot_before(target_index)
        if snapshot_index < 0:
            logger.error("No full snapshot found - cannot reconstruct state")
            return None

        # Start with the full snapshot
        state = copy.deepcopy(self._checkpoints[snapshot_index].full_state)
        if state is None:
            logger.error(
                f"Checkpoint {snapshot_index} marked as snapshot but has no state"
            )
            return None

        # Apply deltas from snapshot to target
        for i in range(snapshot_index + 1, target_index + 1):
            checkpoint = self._checkpoints[i]
            if checkpoint.delta:
                state = apply_delta(state, checkpoint.delta)

        return state

    def get_state_at(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the state at a specific checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint

        Returns:
            The state at that checkpoint, or None if not found
        """
        index = self._find_checkpoint_index(checkpoint_id)
        if index < 0:
            return None
        return self._reconstruct_state(index)

    def _find_checkpoint_index(self, checkpoint_id: str) -> int:
        """Find the index of a checkpoint by ID."""
        for i, cp in enumerate(self._checkpoints):
            if cp.id == checkpoint_id:
                return i
        return -1

    def rollback_to(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Rollback to a specific checkpoint.

        This truncates all checkpoints after the target and returns the state
        that should be applied to the window.

        Args:
            checkpoint_id: ID of the checkpoint to rollback to

        Returns:
            The state to apply, or None if rollback fails
        """
        target_index = self._find_checkpoint_index(checkpoint_id)
        if target_index < 0:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return None

        # Reconstruct the state
        state = self._reconstruct_state(target_index)
        if state is None:
            return None

        # Truncate checkpoints after target
        self._checkpoints = self._checkpoints[: target_index + 1]
        self._current_position = target_index

        # Persist the truncated list
        self._save_to_temp_file()

        logger.info(
            f"Rolled back to checkpoint {checkpoint_id} (position={target_index})"
        )

        return state

    def get_checkpoints(self) -> List[Checkpoint]:
        """Get all checkpoints."""
        return self._checkpoints.copy()

    def get_current_checkpoint(self) -> Optional[Checkpoint]:
        """Get the current checkpoint, if any."""
        if 0 <= self._current_position < len(self._checkpoints):
            return self._checkpoints[self._current_position]
        return None

    def get_message_index_at(self, checkpoint_id: str) -> Optional[int]:
        """
        Get the message index at a specific checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint

        Returns:
            The message index, or None if checkpoint not found
        """
        index = self._find_checkpoint_index(checkpoint_id)
        if index < 0:
            return None
        return self._checkpoints[index].message_index

    def clear(self) -> None:
        """Clear all checkpoints (for new session)."""
        self._checkpoints.clear()
        self._current_position = -1
        self._checkpoint_created_for_response = False
        self._save_to_temp_file()
        logger.info("Cleared all checkpoints")

    @property
    def checkpoint_count(self) -> int:
        """Get the number of checkpoints."""
        return len(self._checkpoints)

    @property
    def current_position(self) -> int:
        """Get the current checkpoint position."""
        return self._current_position
