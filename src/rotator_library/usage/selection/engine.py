# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Selection engine for credential selection.

Central component that orchestrates limit checking, modifiers,
and rotation strategies to select the best credential.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Set, Union

from ..types import (
    CredentialState,
    SelectionContext,
    RotationMode,
    LimitCheckResult,
)
from ..config import ProviderUsageConfig
from ..limits.engine import LimitEngine
from .strategies.balanced import BalancedStrategy
from .strategies.sequential import SequentialStrategy

lib_logger = logging.getLogger("rotator_library")


class SelectionEngine:
    """
    Central engine for credential selection.

    Orchestrates:
    1. Limit checking (filter unavailable credentials)
    2. Fair cycle modifiers (filter exhausted credentials)
    3. Rotation strategy (select from available)
    """

    def __init__(
        self,
        config: ProviderUsageConfig,
        limit_engine: LimitEngine,
    ):
        """
        Initialize selection engine.

        Args:
            config: Provider usage configuration
            limit_engine: LimitEngine for availability checks
        """
        self._config = config
        self._limits = limit_engine

        # Initialize strategies
        self._balanced = BalancedStrategy(config.rotation_tolerance)
        self._sequential = SequentialStrategy(config.sequential_fallback_multiplier)

        # Current strategy
        if config.rotation_mode == RotationMode.SEQUENTIAL:
            self._strategy = self._sequential
        else:
            self._strategy = self._balanced

    def select(
        self,
        provider: str,
        model: str,
        states: Dict[str, CredentialState],
        quota_group: Optional[str] = None,
        exclude: Optional[Set[str]] = None,
        priorities: Optional[Dict[str, int]] = None,
        deadline: float = 0.0,
    ) -> Optional[str]:
        """
        Select the best available credential.

        Args:
            provider: Provider name
            model: Model being requested
            states: Dict of stable_id -> CredentialState
            quota_group: Quota group for this model
            exclude: Set of stable_ids to exclude
            priorities: Override priorities (stable_id -> priority)
            deadline: Request deadline timestamp

        Returns:
            Selected stable_id, or None if none available
        """
        exclude = exclude or set()

        # Step 1: Get all candidates (not excluded)
        candidates = [sid for sid in states.keys() if sid not in exclude]

        if not candidates:
            return None

        # Step 2: Filter by limits
        available = []
        for stable_id in candidates:
            state = states[stable_id]
            result = self._limits.check_all(state, model, quota_group)
            if result.allowed:
                available.append(stable_id)

        if not available:
            lib_logger.debug(
                f"No available credentials for {provider}/{model} "
                f"(all {len(candidates)} blocked by limits)"
            )
            return None

        # Step 3: Build selection context
        # Get usage counts for weighting
        usage_counts = {}
        for stable_id in available:
            state = states[stable_id]
            usage_counts[stable_id] = self._get_usage_count(state)

        # Build priorities map
        if priorities is None:
            priorities = {}
            for stable_id in available:
                priorities[stable_id] = states[stable_id].priority

        context = SelectionContext(
            provider=provider,
            model=model,
            quota_group=quota_group,
            candidates=available,
            priorities=priorities,
            usage_counts=usage_counts,
            rotation_mode=self._config.rotation_mode,
            rotation_tolerance=self._config.rotation_tolerance,
            deadline=deadline or (time.time() + 120),
        )

        # Step 4: Apply rotation strategy
        selected = self._strategy.select(context, states)

        if selected:
            lib_logger.debug(
                f"Selected credential {selected} for {provider}/{model} "
                f"(from {len(available)} available)"
            )

        return selected

    def select_with_retry(
        self,
        provider: str,
        model: str,
        states: Dict[str, CredentialState],
        quota_group: Optional[str] = None,
        tried: Optional[Set[str]] = None,
        priorities: Optional[Dict[str, int]] = None,
        deadline: float = 0.0,
    ) -> Optional[str]:
        """
        Select a credential for retry, excluding already-tried ones.

        Convenience method for retry loops.

        Args:
            provider: Provider name
            model: Model being requested
            states: Dict of stable_id -> CredentialState
            quota_group: Quota group for this model
            tried: Set of already-tried stable_ids
            priorities: Override priorities
            deadline: Request deadline timestamp

        Returns:
            Selected stable_id, or None if none available
        """
        return self.select(
            provider=provider,
            model=model,
            states=states,
            quota_group=quota_group,
            exclude=tried,
            priorities=priorities,
            deadline=deadline,
        )

    def get_availability_stats(
        self,
        provider: str,
        model: str,
        states: Dict[str, CredentialState],
        quota_group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get availability statistics for credentials.

        Useful for status reporting and debugging.

        Args:
            provider: Provider name
            model: Model being requested
            states: Dict of stable_id -> CredentialState
            quota_group: Quota group for this model

        Returns:
            Dict with availability stats
        """
        total = len(states)
        available = 0
        blocked_by = {
            "cooldowns": 0,
            "window_limits": 0,
            "custom_caps": 0,
            "fair_cycle": 0,
            "concurrent": 0,
        }

        for stable_id, state in states.items():
            blocking = self._limits.get_blocking_info(state, model, quota_group)

            is_available = True
            for checker_name, result in blocking.items():
                if not result.allowed:
                    is_available = False
                    if checker_name in blocked_by:
                        blocked_by[checker_name] += 1
                    break

            if is_available:
                available += 1

        return {
            "total": total,
            "available": available,
            "blocked": total - available,
            "blocked_by": blocked_by,
            "rotation_mode": self._config.rotation_mode.value,
        }

    def set_rotation_mode(self, mode: RotationMode) -> None:
        """
        Change the rotation mode.

        Args:
            mode: New rotation mode
        """
        self._config.rotation_mode = mode
        if mode == RotationMode.SEQUENTIAL:
            self._strategy = self._sequential
        else:
            self._strategy = self._balanced

        lib_logger.info(f"Rotation mode changed to {mode.value}")

    def mark_exhausted(self, provider: str, model_or_group: str) -> None:
        """
        Mark current credential as exhausted (for sequential mode).

        Args:
            provider: Provider name
            model_or_group: Model or quota group
        """
        if isinstance(self._strategy, SequentialStrategy):
            self._strategy.mark_exhausted(provider, model_or_group)

    @property
    def balanced_strategy(self) -> BalancedStrategy:
        """Get the balanced strategy instance."""
        return self._balanced

    @property
    def sequential_strategy(self) -> SequentialStrategy:
        """Get the sequential strategy instance."""
        return self._sequential

    def _get_usage_count(self, state: CredentialState) -> int:
        """Get the relevant usage count for rotation weighting."""
        # Use primary window if available, otherwise total
        for window_name, window in state.usage.windows.items():
            # Look for primary window (usually "5h")
            return window.request_count

        return state.usage.total_requests
