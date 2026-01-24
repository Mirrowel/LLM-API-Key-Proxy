# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Custom cap limit checker.

Enforces user-defined limits on API usage.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple

from ..types import CredentialState, LimitCheckResult, LimitResult, WindowStats
from ..config import CustomCapConfig, CooldownMode, CapMode, CapMode, CapMode
from ..tracking.windows import WindowManager
from .base import LimitChecker

lib_logger = logging.getLogger("rotator_library")


# Scope constants for cap application
SCOPE_MODEL = "model"
SCOPE_GROUP = "group"


class CustomCapChecker(LimitChecker):
    """
    Checks custom cap limits.

    Custom caps allow users to set custom usage limits per tier/model/group.
    Limits can be absolute numbers or percentages of API limits.

    Caps are checked independently for both model AND group scopes:
    - A model cap being exceeded blocks only that model
    - A group cap being exceeded blocks the entire group
    - Both caps can exist and are checked separately (first blocked wins)
    """

    def __init__(
        self,
        caps: List[CustomCapConfig],
        window_manager: WindowManager,
    ):
        """
        Initialize custom cap checker.

        Args:
            caps: List of custom cap configurations
            window_manager: WindowManager for checking window usage
        """
        self._caps = caps
        self._windows = window_manager
        # Index caps by (tier_key, model_or_group) for fast lookup
        self._cap_index: Dict[tuple, CustomCapConfig] = {}
        for cap in caps:
            self._cap_index[(cap.tier_key, cap.model_or_group)] = cap

    @property
    def name(self) -> str:
        return "custom_caps"

    def check(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> LimitCheckResult:
        """
        Check if any custom cap is exceeded.

        Checks both model and group caps independently. If both exist,
        each is checked against its respective usage scope. First blocked
        cap wins.

        Args:
            state: Credential state to check
            model: Model being requested
            quota_group: Quota group for this model

        Returns:
            LimitCheckResult indicating pass/fail
        """
        if not self._caps:
            return LimitCheckResult.ok()

        primary_def = self._windows.get_primary_definition()
        if primary_def is None:
            return LimitCheckResult.ok()

        priority = state.priority

        # Find all applicable caps (model + group separately)
        all_caps = self._find_all_caps(str(priority), model, quota_group)
        if not all_caps:
            return LimitCheckResult.ok()

        # Check each cap against its proper scope
        for cap, scope, scope_key in all_caps:
            result = self._check_single_cap(
                state, cap, scope, scope_key, model, quota_group
            )
            if not result.allowed:
                return result

        return LimitCheckResult.ok()

    def get_cap_for(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> Optional[CustomCapConfig]:
        """
        Get the first applicable custom cap for a credential/model.

        Args:
            state: Credential state
            model: Model name
            quota_group: Quota group

        Returns:
            CustomCapConfig if one applies, None otherwise
        """
        priority = state.priority
        all_caps = self._find_all_caps(str(priority), model, quota_group)
        if all_caps:
            return all_caps[0][0]  # Return first cap
        return None

    def get_all_caps_for(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> List[Tuple[CustomCapConfig, str, str]]:
        """
        Get all applicable custom caps for a credential/model.

        Args:
            state: Credential state
            model: Model name
            quota_group: Quota group

        Returns:
            List of (cap, scope, scope_key) tuples
        """
        priority = state.priority
        return self._find_all_caps(str(priority), model, quota_group)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _find_all_caps(
        self,
        priority_key: str,
        model: str,
        quota_group: Optional[str],
    ) -> List[Tuple[CustomCapConfig, str, str]]:
        """
        Find all applicable caps for a request.

        Returns caps for both model AND group scopes (if they exist).
        Each cap is returned with its scope type and scope key.

        Args:
            priority_key: Priority level as string
            model: Model name
            quota_group: Quota group (optional)

        Returns:
            List of (cap, scope, scope_key) tuples where:
            - cap: The CustomCapConfig
            - scope: SCOPE_MODEL or SCOPE_GROUP
            - scope_key: The model name or group name
        """
        result: List[Tuple[CustomCapConfig, str, str]] = []

        # Check model cap (priority-specific, then default)
        model_cap = self._cap_index.get((priority_key, model)) or self._cap_index.get(
            ("default", model)
        )
        if model_cap:
            result.append((model_cap, SCOPE_MODEL, model))

        # Check group cap (priority-specific, then default) - only if group differs from model
        if quota_group and quota_group != model:
            group_cap = self._cap_index.get(
                (priority_key, quota_group)
            ) or self._cap_index.get(("default", quota_group))
            if group_cap:
                result.append((group_cap, SCOPE_GROUP, quota_group))

        return result

    def _check_single_cap(
        self,
        state: CredentialState,
        cap: CustomCapConfig,
        scope: str,
        scope_key: str,
        model: str,
        quota_group: Optional[str],
    ) -> LimitCheckResult:
        """
        Check a single cap against its appropriate usage scope.

        Args:
            state: Credential state
            cap: The cap configuration to check
            scope: SCOPE_MODEL or SCOPE_GROUP
            scope_key: The model name or group name
            model: Original model name (for fallback)
            quota_group: Original quota group (for fallback)

        Returns:
            LimitCheckResult for this specific cap
        """
        # Get windows based on scope
        windows = None

        if scope == SCOPE_GROUP:
            group_stats = state.get_group_stats(scope_key, create=False)
            if group_stats:
                windows = group_stats.windows
        else:  # SCOPE_MODEL
            model_stats = state.get_model_stats(scope_key, create=False)
            if model_stats:
                windows = model_stats.windows

        if windows is None:
            return LimitCheckResult.ok()

        # Get usage from primary window
        primary_window = self._windows.get_primary_window(windows)
        if primary_window is None:
            return LimitCheckResult.ok()

        current_usage = primary_window.request_count
        max_requests = self._resolve_max_requests(cap, primary_window.limit)

        if current_usage >= max_requests:
            # Calculate cooldown end
            cooldown_until = self._calculate_cooldown_until(cap, primary_window)

            # Build descriptive reason with scope info
            scope_desc = "model" if scope == SCOPE_MODEL else "group"
            reason = (
                f"Custom cap for {scope_desc} '{scope_key}' exceeded "
                f"({current_usage}/{max_requests})"
            )

            return LimitCheckResult.blocked(
                result=LimitResult.BLOCKED_CUSTOM_CAP,
                reason=reason,
                blocked_until=cooldown_until,
            )

        return LimitCheckResult.ok()

    def _find_cap(
        self,
        priority_key: str,
        group_key: str,
        model: str,
    ) -> Optional[CustomCapConfig]:
        """
        Find the most specific applicable cap (legacy method for compatibility).

        Deprecated: Use _find_all_caps() for layered cap checking.
        """
        # Try exact matches first
        # Priority + group
        cap = self._cap_index.get((priority_key, group_key))
        if cap:
            return cap

        # Priority + model (if different from group)
        if model != group_key:
            cap = self._cap_index.get((priority_key, model))
            if cap:
                return cap

        # Default tier + group
        cap = self._cap_index.get(("default", group_key))
        if cap:
            return cap

        # Default tier + model
        if model != group_key:
            cap = self._cap_index.get(("default", model))
            if cap:
                return cap

        return None

    def _resolve_max_requests(
        self,
        cap: CustomCapConfig,
        window_limit: Optional[int],
    ) -> int:
        """
        Resolve max requests based on mode.

        Modes:
        - ABSOLUTE: Use value as-is (e.g., 130 → 130)
        - OFFSET: Add/subtract from window limit (e.g., -130 → max - 130)
        - PERCENTAGE: Percentage of window limit (e.g., 80 → 80% of max)

        Always clamps result to >= 0.
        """
        if cap.max_requests_mode == CapMode.ABSOLUTE:
            return max(0, cap.max_requests)

        # For OFFSET and PERCENTAGE, we need window_limit
        if window_limit is None:
            # No limit known - fallback behavior
            if cap.max_requests_mode == CapMode.OFFSET:
                # Can't apply offset without knowing the max
                # Use absolute value as fallback
                return max(0, abs(cap.max_requests))
            # PERCENTAGE with no limit - use safe default
            return 1000

        if cap.max_requests_mode == CapMode.OFFSET:
            # +130 means max + 130, -130 means max - 130
            return max(0, window_limit + cap.max_requests)

        if cap.max_requests_mode == CapMode.PERCENTAGE:
            return max(0, int(window_limit * cap.max_requests / 100))

        # Fallback (shouldn't happen)
        return max(0, cap.max_requests)

    def _calculate_cooldown_until(
        self,
        cap: CustomCapConfig,
        window: WindowStats,
    ) -> Optional[float]:
        """
        Calculate when the custom cap cooldown ends.

        Modes:
        - QUOTA_RESET: Wait until natural window reset
        - OFFSET: Add/subtract offset from natural reset (clamped to >= reset)
        - FIXED: Fixed duration from now
        """
        now = time.time()
        natural_reset = window.reset_at

        if cap.cooldown_mode == CooldownMode.QUOTA_RESET:
            # Wait until window resets
            return natural_reset

        elif cap.cooldown_mode == CooldownMode.OFFSET:
            # Offset from natural reset time
            # Positive offset = wait AFTER reset
            # Negative offset = wait BEFORE reset (clamped to >= reset for safety)
            if natural_reset:
                calculated = natural_reset + cap.cooldown_value
                # Always clamp to at least natural_reset (can't end before quota resets)
                return max(calculated, natural_reset)
            else:
                # No natural reset known, use absolute offset from now
                return now + abs(cap.cooldown_value)

        elif cap.cooldown_mode == CooldownMode.FIXED:
            # Fixed duration from now
            calculated = now + cap.cooldown_value
            return calculated

        return None
