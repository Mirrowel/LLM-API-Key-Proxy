# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Sequential rotation strategy.

Uses one credential until exhausted, then moves to the next.
Good for providers that benefit from request caching.
"""

import logging
from typing import Dict, List, Optional

from ...types import CredentialState, SelectionContext, RotationMode
from ....error_handler import mask_credential
from ....error_handler import mask_credential

lib_logger = logging.getLogger("rotator_library")


class SequentialStrategy:
    """
    Sequential credential rotation strategy.

    Sticks to one credential until it's exhausted (rate limited,
    quota exceeded, etc.), then moves to the next in priority order.

    This is useful for providers where repeated requests to the same
    credential benefit from caching (e.g., context caching in LLMs).
    """

    def __init__(self, fallback_multiplier: int = 1):
        """
        Initialize sequential strategy.

        Args:
            fallback_multiplier: Default concurrent slots per priority
                when not explicitly configured
        """
        self.fallback_multiplier = fallback_multiplier
        # Track current "sticky" credential per (provider, model_group)
        self._current: Dict[tuple, str] = {}

    @property
    def name(self) -> str:
        return "sequential"

    @property
    def mode(self) -> RotationMode:
        return RotationMode.SEQUENTIAL

    def select(
        self,
        context: SelectionContext,
        states: Dict[str, CredentialState],
    ) -> Optional[str]:
        """
        Select a credential using sequential/sticky selection.

        Prefers the currently active credential if it's still available.
        Otherwise, selects the first available by priority.

        Args:
            context: Selection context with candidates and usage info
            states: Dict of stable_id -> CredentialState

        Returns:
            Selected stable_id, or None if no candidates
        """
        if not context.candidates:
            return None

        if len(context.candidates) == 1:
            return context.candidates[0]

        key = (context.provider, context.quota_group or context.model)

        # Check if current sticky credential is still available
        current = self._current.get(key)
        if current and current in context.candidates:
            return current

        # Current not available - select new one by priority
        selected = self._select_by_priority(context.candidates, context.priorities)

        # Make it sticky
        if selected:
            self._current[key] = selected
            masked = (
                mask_credential(states[selected].accessor, style="full")
                if selected in states
                else mask_credential(selected, style="full")
            )
            lib_logger.debug(f"Sequential: switched to credential {masked} for {key}")

        return selected

    def mark_exhausted(self, provider: str, model_or_group: str) -> None:
        """
        Mark current credential as exhausted, forcing rotation.

        Args:
            provider: Provider name
            model_or_group: Model or quota group
        """
        key = (provider, model_or_group)
        if key in self._current:
            old = self._current[key]
            del self._current[key]
            lib_logger.debug(
                f"Sequential: marked {mask_credential(old, style='full')} exhausted for {key}"
            )

    def get_current(self, provider: str, model_or_group: str) -> Optional[str]:
        """
        Get the currently sticky credential.

        Args:
            provider: Provider name
            model_or_group: Model or quota group

        Returns:
            Current sticky credential stable_id, or None
        """
        key = (provider, model_or_group)
        return self._current.get(key)

    def _select_by_priority(
        self,
        candidates: List[str],
        priorities: Dict[str, int],
    ) -> Optional[str]:
        """Select the highest priority (lowest number) candidate."""
        if not candidates:
            return None

        # Sort by priority (lower = higher priority)
        sorted_candidates = sorted(candidates, key=lambda c: priorities.get(c, 999))

        return sorted_candidates[0]

    def clear_sticky(self, provider: Optional[str] = None) -> None:
        """
        Clear sticky credential state.

        Args:
            provider: If specified, only clear for this provider
        """
        if provider:
            keys_to_remove = [k for k in self._current if k[0] == provider]
            for key in keys_to_remove:
                del self._current[key]
        else:
            self._current.clear()
