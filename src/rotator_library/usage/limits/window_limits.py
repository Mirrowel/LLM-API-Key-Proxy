# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Window limit checker.

Checks if a credential has exceeded its request quota for a window.
"""

from typing import List, Optional

from ..types import CredentialState, LimitCheckResult, LimitResult, WindowStats
from ..tracking.windows import WindowManager
from .base import LimitChecker


class WindowLimitChecker(LimitChecker):
    """
    Checks window-based request limits.

    Blocks credentials that have exhausted their quota in any
    tracked window.
    """

    def __init__(self, window_manager: WindowManager):
        """
        Initialize window limit checker.

        Args:
            window_manager: WindowManager instance for window operations
        """
        self._windows = window_manager

    @property
    def name(self) -> str:
        return "window_limits"

    def check(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> LimitCheckResult:
        """
        Check if any window limit is exceeded.

        Args:
            state: Credential state to check
            model: Model being requested
            quota_group: Quota group for this model

        Returns:
            LimitCheckResult indicating pass/fail
        """
        # Check all windows
        for window_name, window in state.usage.windows.items():
            # Skip if no limit defined
            if window.limit is None:
                continue

            # Check if window is still active
            active = self._windows.get_active_window(state.usage.windows, window_name)
            if active is None:
                continue  # Window expired, will be reset

            # Check if limit exceeded
            if active.request_count >= active.limit:
                return LimitCheckResult.blocked(
                    result=LimitResult.BLOCKED_WINDOW,
                    reason=f"Window '{window_name}' exhausted ({active.request_count}/{active.limit})",
                    blocked_until=active.reset_at,
                )

        return LimitCheckResult.ok()

    def get_remaining(
        self,
        state: CredentialState,
        window_name: str,
    ) -> Optional[int]:
        """
        Get remaining requests in a specific window.

        Args:
            state: Credential state
            window_name: Name of window to check

        Returns:
            Remaining requests, or None if unlimited/unknown
        """
        return self._windows.get_window_remaining(state.usage.windows, window_name)

    def get_all_remaining(
        self,
        state: CredentialState,
    ) -> dict[str, Optional[int]]:
        """
        Get remaining requests for all windows.

        Args:
            state: Credential state

        Returns:
            Dict of window_name -> remaining (None if unlimited)
        """
        result = {}
        for window_name in state.usage.windows:
            result[window_name] = self.get_remaining(state, window_name)
        return result
