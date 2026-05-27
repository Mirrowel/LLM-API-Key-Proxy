# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Window limit checker.

Checks if a credential has exceeded its request quota for a window.
"""

from typing import Dict, List, Optional

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
        group_key = quota_group or model

        # Check all configured windows
        for definition in self._windows.definitions.values():
            windows = None

            if definition.applies_to == "model":
                model_stats = state.get_model_stats(model, create=False)
                if model_stats:
                    windows = model_stats.windows
            elif definition.applies_to == "group":
                group_stats = state.get_group_stats(group_key, create=False)
                if group_stats:
                    windows = group_stats.windows

            if windows is None:
                continue

            window = windows.get(definition.name)
            if window is None or window.limit is None:
                continue

            active = self._windows.get_active_window(windows, definition.name)
            if active is None:
                continue

            if active.request_count >= active.limit:
                return LimitCheckResult.blocked(
                    result=LimitResult.BLOCKED_WINDOW,
                    reason=(
                        f"Window '{definition.name}' exhausted "
                        f"({active.request_count}/{active.limit})"
                    ),
                    blocked_until=active.reset_at,
                )

        return LimitCheckResult.ok()

    def get_remaining(
        self,
        state: CredentialState,
        window_name: str,
        model: Optional[str] = None,
        quota_group: Optional[str] = None,
    ) -> Optional[int]:
        """
        Get remaining requests in a specific window.

        Args:
            state: Credential state
            window_name: Name of window to check
            model: Model to check
            quota_group: Quota group to check

        Returns:
            Remaining requests, or None if unlimited/unknown
        """
        group_key = quota_group or model or ""
        definition = self._windows.definitions.get(window_name)

        windows = None
        if definition:
            if definition.applies_to == "model" and model:
                model_stats = state.get_model_stats(model, create=False)
                if model_stats:
                    windows = model_stats.windows
            elif definition.applies_to == "group":
                group_stats = state.get_group_stats(group_key, create=False)
                if group_stats:
                    windows = group_stats.windows

        if windows is None:
            return None

        return self._windows.get_window_remaining(windows, window_name)

    def get_all_remaining(
        self,
        state: CredentialState,
        model: Optional[str] = None,
        quota_group: Optional[str] = None,
    ) -> Dict[str, Optional[int]]:
        """
        Get remaining requests for all windows.

        Args:
            state: Credential state
            model: Model to check
            quota_group: Quota group to check

        Returns:
            Dict of window_name -> remaining (None if unlimited)
        """
        result = {}
        for definition in self._windows.definitions.values():
            result[definition.name] = self.get_remaining(
                state,
                definition.name,
                model=model,
                quota_group=quota_group,
            )
        return result
