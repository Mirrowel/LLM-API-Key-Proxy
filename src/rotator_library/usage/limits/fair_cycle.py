# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Fair cycle limit checker.

Ensures credentials are used fairly by blocking exhausted ones
until all credentials in the pool are exhausted.
"""

import time
import logging
from typing import Dict, List, Optional, Set

from ..types import (
    CredentialState,
    LimitCheckResult,
    LimitResult,
    FairCycleState,
    GlobalFairCycleState,
    TrackingMode,
    FAIR_CYCLE_GLOBAL_KEY,
)
from ..config import FairCycleConfig
from .base import LimitChecker

lib_logger = logging.getLogger("rotator_library")


class FairCycleChecker(LimitChecker):
    """
    Checks fair cycle constraints.

    Blocks credentials that have been "exhausted" (quota used or long cooldown)
    until all credentials in the pool have been exhausted, then resets the cycle.
    """

    def __init__(self, config: FairCycleConfig):
        """
        Initialize fair cycle checker.

        Args:
            config: Fair cycle configuration
        """
        self._config = config
        # Global cycle state per provider
        self._global_state: Dict[str, Dict[str, GlobalFairCycleState]] = {}

    @property
    def name(self) -> str:
        return "fair_cycle"

    def check(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> LimitCheckResult:
        """
        Check if credential is blocked by fair cycle.

        Args:
            state: Credential state to check
            model: Model being requested
            quota_group: Quota group for this model

        Returns:
            LimitCheckResult indicating pass/fail
        """
        if not self._config.enabled:
            return LimitCheckResult.ok()

        group_key = self._resolve_tracking_key(model, quota_group)
        fc_state = state.fair_cycle.get(group_key)

        # Not exhausted = allowed
        if fc_state is None or not fc_state.exhausted:
            return LimitCheckResult.ok()

        # Exhausted - check if cycle should reset
        provider = state.provider
        global_state = self._get_global_state(provider, group_key)

        # Check if cycle has expired
        if self._should_reset_cycle(global_state):
            # Don't block - cycle will be reset
            return LimitCheckResult.ok()

        # Still blocked by fair cycle
        return LimitCheckResult.blocked(
            result=LimitResult.BLOCKED_FAIR_CYCLE,
            reason=f"Fair cycle: exhausted for '{group_key}' - waiting for other credentials",
            blocked_until=None,  # Depends on other credentials
        )

    def reset(
        self,
        state: CredentialState,
        model: Optional[str] = None,
        quota_group: Optional[str] = None,
    ) -> None:
        """
        Reset fair cycle state for a credential.

        Args:
            state: Credential state
            model: Optional model scope
            quota_group: Optional quota group scope
        """
        group_key = self._resolve_tracking_key(model or "", quota_group)

        if quota_group or model:
            if group_key in state.fair_cycle:
                fc_state = state.fair_cycle[group_key]
                fc_state.exhausted = False
                fc_state.exhausted_at = None
                fc_state.exhausted_reason = None
                fc_state.cycle_request_count = 0
        else:
            # Reset all
            for fc_state in state.fair_cycle.values():
                fc_state.exhausted = False
                fc_state.exhausted_at = None
                fc_state.exhausted_reason = None
                fc_state.cycle_request_count = 0

    def check_all_exhausted(
        self,
        provider: str,
        group_key: str,
        all_states: List[CredentialState],
        priorities: Optional[Dict[str, int]] = None,
    ) -> bool:
        """
        Check if all credentials in the pool are exhausted.

        Args:
            provider: Provider name
            group_key: Model or quota group
            all_states: All credential states for this provider
            priorities: Optional priority filter

        Returns:
            True if all are exhausted
        """
        # Filter by tier if not cross-tier
        if priorities and not self._config.cross_tier:
            # Group by priority tier
            priority_groups: Dict[int, List[CredentialState]] = {}
            for state in all_states:
                p = priorities.get(state.stable_id, 999)
                priority_groups.setdefault(p, []).append(state)

            # Check each priority group separately
            for priority, group_states in priority_groups.items():
                if not self._all_exhausted_in_group(group_states, group_key):
                    return False
            return True
        else:
            return self._all_exhausted_in_group(all_states, group_key)

    def reset_cycle(
        self,
        provider: str,
        group_key: str,
        all_states: List[CredentialState],
    ) -> None:
        """
        Reset the fair cycle for all credentials.

        Args:
            provider: Provider name
            group_key: Model or quota group
            all_states: All credential states to reset
        """
        now = time.time()

        for state in all_states:
            if group_key in state.fair_cycle:
                fc_state = state.fair_cycle[group_key]
                fc_state.exhausted = False
                fc_state.exhausted_at = None
                fc_state.exhausted_reason = None
                fc_state.cycle_request_count = 0

        # Update global state
        global_state = self._get_global_state(provider, group_key)
        global_state.cycle_start = now
        global_state.all_exhausted_at = None
        global_state.cycle_count += 1

        lib_logger.info(
            f"Fair cycle reset for {provider}/{group_key}, cycle #{global_state.cycle_count}"
        )

    def mark_all_exhausted(
        self,
        provider: str,
        group_key: str,
    ) -> None:
        """
        Record that all credentials are now exhausted.

        Args:
            provider: Provider name
            group_key: Model or quota group
        """
        global_state = self._get_global_state(provider, group_key)
        global_state.all_exhausted_at = time.time()

        lib_logger.info(f"All credentials exhausted for {provider}/{group_key}")

    def get_tracking_key(self, model: str, quota_group: Optional[str]) -> str:
        """Get the fair cycle tracking key for a request."""
        return self._resolve_tracking_key(model, quota_group)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _get_global_state(
        self,
        provider: str,
        group_key: str,
    ) -> GlobalFairCycleState:
        """Get or create global fair cycle state."""
        if provider not in self._global_state:
            self._global_state[provider] = {}

        if group_key not in self._global_state[provider]:
            self._global_state[provider][group_key] = GlobalFairCycleState(
                cycle_start=time.time()
            )

        return self._global_state[provider][group_key]

    def _resolve_tracking_key(
        self,
        model: str,
        quota_group: Optional[str],
    ) -> str:
        """Resolve tracking key based on fair cycle mode."""
        if self._config.tracking_mode == TrackingMode.CREDENTIAL:
            return FAIR_CYCLE_GLOBAL_KEY
        return quota_group or model

    def _should_reset_cycle(self, global_state: GlobalFairCycleState) -> bool:
        """Check if cycle duration has expired."""
        now = time.time()
        return now >= global_state.cycle_start + self._config.duration

    def _all_exhausted_in_group(
        self,
        states: List[CredentialState],
        group_key: str,
    ) -> bool:
        """Check if all credentials in a group are exhausted."""
        if not states:
            return True

        for state in states:
            fc_state = state.fair_cycle.get(group_key)
            if fc_state is None or not fc_state.exhausted:
                return False

        return True

    def get_global_state_dict(self) -> Dict[str, Dict[str, Dict]]:
        """
        Get global state for serialization.

        Returns:
            Dict suitable for JSON serialization
        """
        result = {}
        for provider, groups in self._global_state.items():
            result[provider] = {}
            for group_key, state in groups.items():
                result[provider][group_key] = {
                    "cycle_start": state.cycle_start,
                    "all_exhausted_at": state.all_exhausted_at,
                    "cycle_count": state.cycle_count,
                }
        return result

    def load_global_state_dict(self, data: Dict[str, Dict[str, Dict]]) -> None:
        """
        Load global state from serialized data.

        Args:
            data: Dict from get_global_state_dict()
        """
        self._global_state.clear()
        for provider, groups in data.items():
            self._global_state[provider] = {}
            for group_key, state_data in groups.items():
                self._global_state[provider][group_key] = GlobalFairCycleState(
                    cycle_start=state_data.get("cycle_start", 0),
                    all_exhausted_at=state_data.get("all_exhausted_at"),
                    cycle_count=state_data.get("cycle_count", 0),
                )
