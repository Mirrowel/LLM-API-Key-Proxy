"""
Cross-provider fallback group management.

Allows pooling credentials from multiple providers for equivalent models,
with tier-aware rotation that respects entry priority ordering.
"""

import json
import logging
import os
from typing import Dict, List, Optional

lib_logger = logging.getLogger("rotator_library")


class FallbackGroupManager:
    """
    Manages cross-provider fallback groups for models.

    Configuration format (JSON array of arrays):
    [
        ["2/gemini-3", "3/gemini-3", "4/gemini-2.5-pro"],
        ["antigravity/claude-sonnet-4.5", "openrouter/claude-3.5-sonnet"]
    ]

    Each inner array is a fallback group. When a request matches any entry
    in a group, all entries in that group become fallback candidates.

    Order matters:
    - Entries earlier in the list have higher priority
    - The requested entry is always promoted to first position
    - Same provider can appear multiple times with different models
    """

    def __init__(self, config: Optional[List[List[str]]] = None):
        """
        Initialize the FallbackGroupManager.

        Args:
            config: List of fallback groups. If None, loads from
                    MODEL_FALLBACK_GROUPS environment variable.
        """
        self._groups: List[List[str]] = []
        self._entry_to_group_index: Dict[str, int] = {}

        if config is not None:
            self._load_from_config(config)
        else:
            self._load_from_env()

        self._build_index()

    def _load_from_config(self, config: List[List[str]]) -> None:
        """Load groups from provided config."""
        if not isinstance(config, list):
            lib_logger.warning(
                f"model_fallback_groups must be a list, got {type(config).__name__}"
            )
            return

        for i, group in enumerate(config):
            if not isinstance(group, list):
                lib_logger.warning(
                    f"Fallback group {i} must be a list, got {type(group).__name__}"
                )
                continue

            # Validate entries
            valid_entries = []
            for entry in group:
                if not isinstance(entry, str):
                    lib_logger.warning(
                        f"Fallback entry must be a string, got {type(entry).__name__}: {entry}"
                    )
                    continue
                if "/" not in entry:
                    lib_logger.warning(
                        f"Fallback entry must be 'provider/model' format: {entry}"
                    )
                    continue
                valid_entries.append(entry)

            if len(valid_entries) >= 2:
                self._groups.append(valid_entries)
                lib_logger.info(
                    f"Loaded fallback group with {len(valid_entries)} entries: "
                    f"{', '.join(valid_entries[:3])}{'...' if len(valid_entries) > 3 else ''}"
                )
            elif valid_entries:
                lib_logger.warning(
                    f"Fallback group needs at least 2 entries, got {len(valid_entries)}: {valid_entries}"
                )

    def _load_from_env(self) -> None:
        """Load groups from MODEL_FALLBACK_GROUPS environment variable."""
        env_value = os.environ.get("MODEL_FALLBACK_GROUPS", "").strip()
        if not env_value:
            return

        try:
            config = json.loads(env_value)
            self._load_from_config(config)
        except json.JSONDecodeError as e:
            lib_logger.warning(f"Invalid JSON in MODEL_FALLBACK_GROUPS: {e}")

    def _build_index(self) -> None:
        """Build lookup index from entry to group index."""
        self._entry_to_group_index.clear()
        for group_idx, group in enumerate(self._groups):
            for entry in group:
                if entry in self._entry_to_group_index:
                    lib_logger.warning(
                        f"Entry '{entry}' appears in multiple fallback groups. "
                        f"Using first occurrence (group {self._entry_to_group_index[entry]})."
                    )
                    continue
                self._entry_to_group_index[entry] = group_idx

    def get_fallback_group(self, provider_model: str) -> Optional[List[str]]:
        """
        Get fallback group for a provider/model, with target promoted to first.

        Args:
            provider_model: Full "provider/model" string (e.g., "3/gemini-3")

        Returns:
            Reordered list with target first, or None if not in any group.
            Example: ["3/gemini-3", "2/gemini-3", "4/gemini-2.5-pro"]
        """
        group_idx = self._entry_to_group_index.get(provider_model)
        if group_idx is None:
            return None

        group = self._groups[group_idx]

        # Move target to front, preserve relative order of others
        reordered = [provider_model]
        for entry in group:
            if entry != provider_model:
                reordered.append(entry)

        return reordered

    def is_in_fallback_group(self, provider_model: str) -> bool:
        """Check if provider/model is part of any fallback group."""
        return provider_model in self._entry_to_group_index

    def get_all_groups(self) -> List[List[str]]:
        """Get all configured fallback groups (for debugging/logging)."""
        return list(self._groups)

    def __bool__(self) -> bool:
        """Return True if any fallback groups are configured."""
        return len(self._groups) > 0

    def __len__(self) -> int:
        """Return number of configured fallback groups."""
        return len(self._groups)
