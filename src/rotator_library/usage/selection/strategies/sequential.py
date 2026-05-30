# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Sequential rotation strategy.

Uses one credential until exhausted, then moves to the next.
Good for providers that benefit from request caching.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from ...types import CredentialState, SelectionContext, RotationMode
from ....error_handler import mask_credential

lib_logger = logging.getLogger("rotator_library")


@dataclass
class _StickyEntry:
    credential: str
    last_seen: float


class SequentialStrategy:
    """
    Sequential credential rotation strategy.

    Sticks to one credential until it's exhausted (rate limited,
    quota exceeded, etc.), then moves to the next in priority order.

    This is useful for providers where repeated requests to the same
    credential benefit from caching (e.g., context caching in LLMs).
    """

    def __init__(
        self,
        fallback_multiplier: int = 1,
        sticky_entry_ttl_seconds: int = 3600,
        max_sticky_entries: int = 10000,
    ):
        """
        Initialize sequential strategy.

        Args:
            fallback_multiplier: Default concurrent slots per priority
                when not explicitly configured
        """
        self.fallback_multiplier = fallback_multiplier
        self.sticky_entry_ttl_seconds = max(1, sticky_entry_ttl_seconds)
        self.max_sticky_entries = max(100, max_sticky_entries)
        # Track current "sticky" credential per model session or model-group fallback.
        self._current: Dict[tuple, _StickyEntry] = {}

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

        now = time.time()
        self._prune_sticky(now)

        if len(context.candidates) == 1:
            return context.candidates[0]

        scope = context.model if context.session_id else (context.quota_group or context.model)
        key = (context.provider, scope, context.session_id or "__default__")

        # Check if current sticky credential is still available
        current = self._current.get(key)
        if current and current.credential in context.candidates:
            current.last_seen = now
            return current.credential

        if context.session_id:
            selected = self._select_initial_for_session(
                context.session_affinity_key or context.session_id,
                context.candidates,
                context.priorities,
            )
        else:
            # Current not available - select new one by tier -> usage -> recency
            selected = self._select_by_priority(
                context.candidates,
                context.priorities,
                context.usage_counts,
                states,
            )

        # Make it sticky
        if selected:
            self._current[key] = _StickyEntry(selected, now)
            self._trim_sticky()
            masked = (
                mask_credential(states[selected].accessor, style="full")
                if selected in states
                else mask_credential(selected, style="full")
            )
            lib_logger.debug(f"Sequential: switched to credential {masked} for {key}")

        return selected

    def _select_initial_for_session(
        self,
        affinity_seed: str,
        candidates: List[str],
        priorities: Dict[str, int],
    ) -> Optional[str]:
        """Pick a stable first credential for a new session.

        Sequential mode still honors tier priority first, but spreads new
        sessions across equal-priority candidates to improve cache locality. The
        seed may be a deterministic affinity key when tracking evidence is strong
        enough, otherwise it falls back to the live session id.
        """
        if not candidates:
            return None

        best_priority = min(priorities.get(candidate, 999) for candidate in candidates)
        eligible = sorted(
            candidate
            for candidate in candidates
            if priorities.get(candidate, 999) == best_priority
        )
        if not eligible:
            return None

        digest = hashlib.sha256(affinity_seed.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % len(eligible)
        return eligible[index]

    def mark_exhausted(self, provider: str, model_or_group: str) -> None:
        """
        Mark current credential as exhausted, forcing rotation.

        Args:
            provider: Provider name
            model_or_group: Model or quota group
        """
        keys_to_remove = [
            key for key in self._current if key[0] == provider and key[1] == model_or_group
        ]
        for key in keys_to_remove:
            old = self._current[key].credential
            del self._current[key]
            lib_logger.debug(
                f"Sequential: marked {mask_credential(old, style='full')} exhausted for {key}"
            )

    def get_current(
        self,
        provider: str,
        model_or_group: str,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get the currently sticky credential.

        Args:
            provider: Provider name
            model_or_group: Model or quota group

        Returns:
            Current sticky credential stable_id, or None
        """
        key = (provider, model_or_group, session_id or "__default__")
        entry = self._current.get(key)
        if not entry:
            return None
        if time.time() - entry.last_seen > self.sticky_entry_ttl_seconds:
            del self._current[key]
            return None
        return entry.credential

    def _prune_sticky(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._current.items()
            if now - entry.last_seen > self.sticky_entry_ttl_seconds
        ]
        for key in expired:
            del self._current[key]

    def _trim_sticky(self) -> None:
        if len(self._current) <= self.max_sticky_entries:
            return
        overage = len(self._current) - self.max_sticky_entries
        for key, _entry in sorted(self._current.items(), key=lambda item: item[1].last_seen)[:overage]:
            del self._current[key]

    def _select_by_priority(
        self,
        candidates: List[str],
        priorities: Dict[str, int],
        usage_counts: Optional[Dict[str, int]] = None,
        states: Optional[Dict[str, CredentialState]] = None,
    ) -> Optional[str]:
        """
        Select credential by: tier (priority) -> usage (highest) -> recency (most recent).

        Sequential mode prefers most-used credentials within the window to maximize
        cache hits. When selecting a new sticky credential:
        1. Highest tier (lowest priority number) first
        2. Within same tier, prefer highest usage count
        3. Within same usage, prefer most recently used

        Args:
            candidates: List of available credential stable_ids
            priorities: Dict of stable_id -> priority (lower = higher tier)
            usage_counts: Dict of stable_id -> request count for relevant window
            states: Dict of stable_id -> CredentialState for recency lookup

        Returns:
            Selected stable_id, or None if no candidates
        """
        if not candidates:
            return None

        usage_counts = usage_counts or {}
        states = states or {}

        def sort_key(c: str):
            # 1. Priority/tier (lower number = higher tier = preferred)
            priority = priorities.get(c, 999)

            # 2. Usage count (higher = preferred, so negate for ascending sort)
            usage = -(usage_counts.get(c, 0))

            # 3. Recency (more recent = preferred, so negate for ascending sort)
            state = states.get(c)
            last_used = -(state.totals.last_used_at or 0) if state else 0

            return (priority, usage, last_used)

        sorted_candidates = sorted(candidates, key=sort_key)
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
