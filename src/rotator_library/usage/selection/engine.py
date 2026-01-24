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
from ..tracking.windows import WindowManager
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
        window_manager: WindowManager,
    ):
        """
        Initialize selection engine.

        Args:
            config: Provider usage configuration
            limit_engine: LimitEngine for availability checks
        """
        self._config = config
        self._limits = limit_engine
        self._windows = window_manager

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
            # Check if we should reset fair cycle
            if self._config.fair_cycle.enabled:
                reset_performed = self._try_fair_cycle_reset(
                    provider,
                    model,
                    quota_group,
                    states,
                    candidates,
                    priorities,
                )
                if reset_performed:
                    # Retry selection after reset
                    return self.select(
                        provider,
                        model,
                        states,
                        quota_group,
                        exclude,
                        priorities,
                        deadline,
                    )

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
            usage_counts[stable_id] = self._get_usage_count(state, model, quota_group)

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

    def _get_usage_count(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str],
    ) -> int:
        """Get the relevant usage count for rotation weighting."""
        primary_def = self._windows.get_primary_definition()
        if primary_def:
            windows = None

            if primary_def.applies_to == "model":
                model_stats = state.get_model_stats(model, create=False)
                if model_stats:
                    windows = model_stats.windows
            elif primary_def.applies_to == "group":
                group_key = quota_group or model
                group_stats = state.get_group_stats(group_key, create=False)
                if group_stats:
                    windows = group_stats.windows

            if windows:
                window = self._windows.get_active_window(windows, primary_def.name)
                if window:
                    return window.request_count

        return state.totals.request_count

    def _get_shortest_cooldown(
        self,
        states: List[CredentialState],
        group_key: str,
    ) -> tuple:
        """
        Find the shortest remaining cooldown among the given credentials.

        Args:
            states: List of credential states to check
            group_key: Model or quota group key for cooldown lookup

        Returns:
            Tuple of (has_short_cooldown, stable_id, remaining_seconds)
            where has_short_cooldown is True if any cooldown is under the threshold
        """
        import time

        now = time.time()
        threshold = self._config.fair_cycle.reset_cooldown_threshold
        shortest_remaining = float("inf")
        shortest_cred_id = None

        for state in states:
            # Check group-specific cooldown
            cooldown = state.cooldowns.get(group_key)
            if cooldown and cooldown.until > now:
                remaining = cooldown.until - now
                if remaining < shortest_remaining:
                    shortest_remaining = remaining
                    shortest_cred_id = state.stable_id

            # Also check global cooldown
            global_cooldown = state.cooldowns.get("_global_")
            if global_cooldown and global_cooldown.until > now:
                remaining = global_cooldown.until - now
                if remaining < shortest_remaining:
                    shortest_remaining = remaining
                    shortest_cred_id = state.stable_id

        if shortest_remaining < threshold:
            return (True, shortest_cred_id, shortest_remaining)
        return (False, None, shortest_remaining)

    def _try_fair_cycle_reset(
        self,
        provider: str,
        model: str,
        quota_group: Optional[str],
        states: Dict[str, CredentialState],
        candidates: List[str],
        priorities: Optional[Dict[str, int]],
    ) -> bool:
        """
        Try to reset fair cycle if all credentials are exhausted.

        Tier-aware: If cross_tier is disabled, checks each tier separately.

        Args:
            provider: Provider name
            model: Model being requested
            quota_group: Quota group for this model
            states: All credential states
            candidates: Candidate stable_ids

        Returns:
            True if reset was performed, False otherwise
        """
        from ..types import LimitResult

        group_key = quota_group or model
        fair_cycle_checker = self._limits.fair_cycle_checker
        tracking_key = fair_cycle_checker.get_tracking_key(model, quota_group)

        # Check if all candidates are blocked by fair cycle
        all_fair_cycle_blocked = True
        fair_cycle_blocked_count = 0

        for stable_id in candidates:
            state = states[stable_id]
            result = self._limits.check_all(state, model, quota_group)

            if result.allowed:
                # Some credential is available - no need to reset
                return False

            if result.result == LimitResult.BLOCKED_FAIR_CYCLE:
                fair_cycle_blocked_count += 1
            else:
                # Blocked by something other than fair cycle
                all_fair_cycle_blocked = False

        # If no credentials blocked by fair cycle, can't help
        if fair_cycle_blocked_count == 0:
            return False

        # Get all candidate states for reset
        candidate_states = [states[sid] for sid in candidates]
        priority_map = priorities or {sid: states[sid].priority for sid in candidates}

        # Tier-aware reset
        if self._config.fair_cycle.cross_tier:
            # Cross-tier: reset all at once
            if fair_cycle_checker.check_all_exhausted(
                provider, tracking_key, candidate_states, priorities=priority_map
            ):
                # Before resetting, check if any credential has a short cooldown
                # that will expire soon - if so, wait instead of resetting
                has_short, cred_id, remaining = self._get_shortest_cooldown(
                    candidate_states, group_key
                )
                if has_short:
                    lib_logger.debug(
                        f"Skipping fair cycle reset for {provider}/{model}: "
                        f"credential {cred_id} has short cooldown ({remaining:.0f}s remaining)"
                    )
                    return False

                lib_logger.info(
                    f"All credentials fair-cycle exhausted for {provider}/{model} "
                    f"(cross-tier), resetting cycle"
                )
                fair_cycle_checker.reset_cycle(provider, tracking_key, candidate_states)
                return True
        else:
            # Per-tier: group by priority and check each tier
            tier_groups: Dict[int, List[CredentialState]] = {}
            for state in candidate_states:
                priority = state.priority
                tier_groups.setdefault(priority, []).append(state)

            reset_any = False
            for priority, tier_states in tier_groups.items():
                # Check if all in this tier are exhausted
                all_tier_exhausted = all(
                    state.is_fair_cycle_exhausted(tracking_key) for state in tier_states
                )

                if all_tier_exhausted:
                    # Before resetting, check if any credential has a short cooldown
                    # that will expire soon - if so, wait instead of resetting
                    has_short, cred_id, remaining = self._get_shortest_cooldown(
                        tier_states, group_key
                    )
                    if has_short:
                        lib_logger.debug(
                            f"Skipping fair cycle reset for {provider}/{model} tier {priority}: "
                            f"credential {cred_id} has short cooldown ({remaining:.0f}s remaining)"
                        )
                        continue

                    lib_logger.info(
                        f"All credentials fair-cycle exhausted for {provider}/{model} "
                        f"in tier {priority}, resetting tier cycle"
                    )
                    fair_cycle_checker.reset_cycle(provider, tracking_key, tier_states)
                    reset_any = True

            return reset_any

        return False
