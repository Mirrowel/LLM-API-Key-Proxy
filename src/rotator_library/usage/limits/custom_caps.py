# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Custom cap limit checker.

Enforces user-defined limits that are stricter than API limits.
"""

import time
import logging
from typing import Dict, List, Optional

from ..types import CredentialState, LimitCheckResult, LimitResult
from ..config import CustomCapConfig, CooldownMode
from ..tracking.windows import WindowManager
from .base import LimitChecker

lib_logger = logging.getLogger("rotator_library")


class CustomCapChecker(LimitChecker):
    """
    Checks custom cap limits.

    Custom caps allow users to set limits more restrictive than
    what the API allows, for cost control or other reasons.
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
        Check if custom cap is exceeded.

        Args:
            state: Credential state to check
            model: Model being requested
            quota_group: Quota group for this model

        Returns:
            LimitCheckResult indicating pass/fail
        """
        if not self._caps:
            return LimitCheckResult.ok()

        group_key = quota_group or model
        priority = state.priority

        # Find applicable cap
        cap = self._find_cap(str(priority), group_key, model)
        if cap is None:
            return LimitCheckResult.ok()

        # Get usage from primary window
        primary_window = self._windows.get_primary_window(state.usage.windows)
        if primary_window is None:
            return LimitCheckResult.ok()

        current_usage = primary_window.request_count
        max_requests = self._resolve_max_requests(cap, primary_window.limit)

        if current_usage >= max_requests:
            # Calculate cooldown end
            cooldown_until = self._calculate_cooldown_until(cap, primary_window)

            return LimitCheckResult.blocked(
                result=LimitResult.BLOCKED_CUSTOM_CAP,
                reason=f"Custom cap for '{group_key}' exceeded ({current_usage}/{max_requests})",
                blocked_until=cooldown_until,
            )

        return LimitCheckResult.ok()

    def get_cap_for(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> Optional[CustomCapConfig]:
        """
        Get the applicable custom cap for a credential/model.

        Args:
            state: Credential state
            model: Model name
            quota_group: Quota group

        Returns:
            CustomCapConfig if one applies, None otherwise
        """
        group_key = quota_group or model
        priority = state.priority
        return self._find_cap(str(priority), group_key, model)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _find_cap(
        self,
        priority_key: str,
        group_key: str,
        model: str,
    ) -> Optional[CustomCapConfig]:
        """Find the most specific applicable cap."""
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
        Resolve max requests, handling percentage values.

        Custom caps can only be MORE restrictive than API limits,
        so the result is clamped to window_limit if available.
        """
        if cap.max_requests >= 0:
            # Absolute value - clamp to window limit if known
            if window_limit is not None:
                return min(cap.max_requests, window_limit)
            return cap.max_requests

        # Negative value indicates percentage
        if window_limit is None:
            # No window limit known, use a high default
            return 1000

        percentage = -cap.max_requests
        calculated = int(window_limit * percentage / 100)
        # Clamp to window limit (already is <= since percentage < 100 typically)
        return min(calculated, window_limit)

    def _calculate_cooldown_until(
        self,
        cap: CustomCapConfig,
        window: "WindowStats",
    ) -> Optional[float]:
        """Calculate when the custom cap cooldown ends."""
        now = time.time()

        if cap.cooldown_mode == CooldownMode.QUOTA_RESET:
            # Wait until window resets
            return window.reset_at

        elif cap.cooldown_mode == CooldownMode.OFFSET:
            # Add offset to current time
            return now + cap.cooldown_value

        elif cap.cooldown_mode == CooldownMode.FIXED:
            # Fixed duration
            return now + cap.cooldown_value

        return None
