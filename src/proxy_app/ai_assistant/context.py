"""
Window context adapter abstract base class and utilities.

Each GUI window that wants to use the AI assistant must implement
the WindowContextAdapter interface.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .tools import ToolDefinition


class WindowContextAdapter(ABC):
    """
    Abstract base class that each window must implement to connect to the AI assistant.

    The adapter provides:
    - Full context extraction for the LLM
    - Window-specific system prompts
    - Tool definitions for the window
    - State application for checkpoint rollback
    """

    @abstractmethod
    def get_full_context(self) -> Dict[str, Any]:
        """
        Get the complete structured state of the window.

        This should include all relevant information the AI needs to understand
        the current state and make decisions. The returned dictionary will be
        serialized and included in the LLM context.

        Returns:
            Dictionary containing the full window state
        """
        pass

    @abstractmethod
    def get_window_system_prompt(self) -> str:
        """
        Get window-specific instructions for the AI.

        This prompt is appended to the base assistant prompt to provide
        domain-specific knowledge and guidelines.

        Returns:
            String containing the window-specific system prompt
        """
        pass

    @abstractmethod
    def get_tools(self) -> List[ToolDefinition]:
        """
        Get the list of tools available for this window.

        Returns:
            List of ToolDefinition objects
        """
        pass

    @abstractmethod
    def apply_state(self, state: Dict[str, Any]) -> None:
        """
        Restore the window to a given state (for checkpoint rollback).

        This method should atomically restore the window state. If restoration
        fails partway through, the window should remain in its pre-restore state.

        Args:
            state: The state dictionary to restore (from a checkpoint)
        """
        pass

    def get_state_hash(self) -> str:
        """
        Get a quick hash of the current state for change detection.

        Override this for more efficient change detection if needed.
        Default implementation hashes the full context.

        Returns:
            String hash of the current state
        """
        context = self.get_full_context()
        context_str = json.dumps(context, sort_keys=True, default=str)
        return hashlib.md5(context_str.encode()).hexdigest()

    def lock_window(self) -> None:
        """
        Lock the window to prevent user interaction during AI execution.

        Override this to implement window locking (gray out widgets, change cursor, etc.)
        Default implementation does nothing.
        """
        pass

    def unlock_window(self) -> None:
        """
        Unlock the window after AI execution completes.

        Override this to restore window interactivity.
        Default implementation does nothing.
        """
        pass

    def on_ai_started(self) -> None:
        """
        Called when the AI starts processing a request.

        Override to perform any setup needed (e.g., start tracking changes).
        Default implementation just locks the window.
        """
        self.lock_window()

    def on_ai_completed(self) -> None:
        """
        Called when the AI finishes processing (success or failure).

        Override to perform any cleanup needed (e.g., stop tracking changes).
        Default implementation just unlocks the window.
        """
        self.unlock_window()


def compute_context_diff(
    old_context: Dict[str, Any], new_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute the differences between two context snapshots.

    Args:
        old_context: The previous context snapshot
        new_context: The current context snapshot

    Returns:
        Dictionary with 'added', 'removed', and 'modified' keys
    """

    def _diff_recursive(
        old: Any, new: Any, path: str = ""
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Recursively diff two values."""
        result: Dict[str, List[Dict[str, Any]]] = {
            "added": [],
            "removed": [],
            "modified": [],
        }

        if type(old) != type(new):
            result["modified"].append({"path": path, "old": old, "new": new})
            return result

        if isinstance(old, dict) and isinstance(new, dict):
            all_keys = set(old.keys()) | set(new.keys())
            for key in all_keys:
                sub_path = f"{path}.{key}" if path else key
                if key not in old:
                    result["added"].append({"path": sub_path, "value": new[key]})
                elif key not in new:
                    result["removed"].append({"path": sub_path, "value": old[key]})
                else:
                    sub_diff = _diff_recursive(old[key], new[key], sub_path)
                    result["added"].extend(sub_diff["added"])
                    result["removed"].extend(sub_diff["removed"])
                    result["modified"].extend(sub_diff["modified"])

        elif isinstance(old, list) and isinstance(new, list):
            if old != new:
                result["modified"].append({"path": path, "old": old, "new": new})

        elif old != new:
            result["modified"].append({"path": path, "old": old, "new": new})

        return result

    return _diff_recursive(old_context, new_context)


def compute_delta(
    old_state: Dict[str, Any], new_state: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute a delta that can be applied to old_state to produce new_state.

    Used for checkpoint storage - stores only differences between states.

    Args:
        old_state: The previous state
        new_state: The new state

    Returns:
        Delta dictionary with 'added', 'removed', 'modified' keys
    """
    return compute_context_diff(old_state, new_state)


def apply_delta(state: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply a delta to a state to produce a new state.

    Args:
        state: The base state
        delta: The delta to apply

    Returns:
        New state with delta applied
    """
    import copy

    result = copy.deepcopy(state)

    def _set_nested(d: Dict, path: str, value: Any) -> None:
        """Set a nested value by path."""
        keys = path.split(".")
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    def _delete_nested(d: Dict, path: str) -> None:
        """Delete a nested value by path."""
        keys = path.split(".")
        for key in keys[:-1]:
            if key not in d:
                return
            d = d[key]
        if keys[-1] in d:
            del d[keys[-1]]

    # Apply additions
    for item in delta.get("added", []):
        _set_nested(result, item["path"], item["value"])

    # Apply modifications
    for item in delta.get("modified", []):
        _set_nested(result, item["path"], item["new"])

    # Apply removals (do this last)
    for item in delta.get("removed", []):
        _delete_nested(result, item["path"])

    return result
