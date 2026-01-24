# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Limit engine for orchestrating limit checks.

Central component that runs all limit checkers and determines
if a credential is available for use.
"""

import logging
from typing import Dict, List, Optional

from ..types import CredentialState, LimitCheckResult, LimitResult
from ..config import ProviderUsageConfig
from ..tracking.windows import WindowManager
from .base import LimitChecker
from .concurrent import ConcurrentLimitChecker
from .window_limits import WindowLimitChecker
from .cooldowns import CooldownChecker
from .fair_cycle import FairCycleChecker
from .custom_caps import CustomCapChecker
from ...error_handler import mask_credential
from ...error_handler import mask_credential

lib_logger = logging.getLogger("rotator_library")


class LimitEngine:
    """
    Central engine for limit checking.

    Orchestrates all limit checkers and provides a single entry point
    for determining credential availability.
    """

    def __init__(
        self,
        config: ProviderUsageConfig,
        window_manager: WindowManager,
    ):
        """
        Initialize limit engine.

        Args:
            config: Provider usage configuration
            window_manager: WindowManager for window-based checks
        """
        self._config = config
        self._window_manager = window_manager

        # Initialize all limit checkers
        # Order matters: concurrent first (fast check), then others
        # Note: WindowLimitChecker is optional - only included if window_limits_enabled
        self._checkers: List[LimitChecker] = [
            ConcurrentLimitChecker(),
            CooldownChecker(),
        ]

        # Window limit checker - kept as reference for info purposes,
        # only added to blocking checkers if explicitly enabled
        self._window_checker = WindowLimitChecker(window_manager)
        if config.window_limits_enabled:
            self._checkers.append(self._window_checker)

        # Custom caps and fair cycle always active
        self._custom_cap_checker = CustomCapChecker(config.custom_caps, window_manager)
        self._fair_cycle_checker = FairCycleChecker(config.fair_cycle, window_manager)
        self._checkers.append(self._custom_cap_checker)
        self._checkers.append(self._fair_cycle_checker)

        # Quick access to specific checkers
        self._concurrent_checker = self._checkers[0]
        self._cooldown_checker = self._checkers[1]

    def check_all(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> LimitCheckResult:
        """
        Check all limits for a credential.

        Runs all limit checkers in order and returns the first failure,
        or success if all pass.

        Args:
            state: Credential state to check
            model: Model being requested
            quota_group: Quota group for this model

        Returns:
            LimitCheckResult indicating overall pass/fail
        """
        for checker in self._checkers:
            result = checker.check(state, model, quota_group)
            if not result.allowed:
                lib_logger.debug(
                    f"Credential {mask_credential(state.accessor, style='full')} blocked by {checker.name}: {result.reason}"
                )
                return result

        return LimitCheckResult.ok()

    def check_specific(
        self,
        checker_name: str,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> LimitCheckResult:
        """
        Check a specific limit type.

        Args:
            checker_name: Name of the checker ("cooldowns", "window_limits", etc.)
            state: Credential state to check
            model: Model being requested
            quota_group: Quota group for this model

        Returns:
            LimitCheckResult from the specified checker
        """
        for checker in self._checkers:
            if checker.name == checker_name:
                return checker.check(state, model, quota_group)

        # Unknown checker - return ok
        return LimitCheckResult.ok()

    def get_available_candidates(
        self,
        states: List[CredentialState],
        model: str,
        quota_group: Optional[str] = None,
    ) -> List[CredentialState]:
        """
        Filter credentials to only those passing all limits.

        Args:
            states: List of credential states to check
            model: Model being requested
            quota_group: Quota group for this model

        Returns:
            List of available credential states
        """
        available = []
        for state in states:
            result = self.check_all(state, model, quota_group)
            if result.allowed:
                available.append(state)

        return available

    def get_blocking_info(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> Dict[str, LimitCheckResult]:
        """
        Get detailed blocking info for each limit type.

        Useful for debugging and status reporting.

        Args:
            state: Credential state to check
            model: Model being requested
            quota_group: Quota group for this model

        Returns:
            Dict mapping checker name to its result
        """
        results = {}
        for checker in self._checkers:
            results[checker.name] = checker.check(state, model, quota_group)
        return results

    def reset_all(
        self,
        state: CredentialState,
        model: Optional[str] = None,
        quota_group: Optional[str] = None,
    ) -> None:
        """
        Reset all limits for a credential.

        Args:
            state: Credential state
            model: Optional model scope
            quota_group: Optional quota group scope
        """
        for checker in self._checkers:
            checker.reset(state, model, quota_group)

    @property
    def concurrent_checker(self) -> ConcurrentLimitChecker:
        """Get the concurrent limit checker."""
        return self._concurrent_checker

    @property
    def cooldown_checker(self) -> CooldownChecker:
        """Get the cooldown checker."""
        return self._cooldown_checker

    @property
    def window_checker(self) -> WindowLimitChecker:
        """Get the window limit checker."""
        return self._window_checker

    @property
    def custom_cap_checker(self) -> CustomCapChecker:
        """Get the custom cap checker."""
        return self._custom_cap_checker

    @property
    def fair_cycle_checker(self) -> FairCycleChecker:
        """Get the fair cycle checker."""
        return self._fair_cycle_checker

    def add_checker(self, checker: LimitChecker) -> None:
        """
        Add a custom limit checker.

        Allows extending the limit system with custom logic.

        Args:
            checker: LimitChecker implementation to add
        """
        self._checkers.append(checker)

    def remove_checker(self, name: str) -> bool:
        """
        Remove a limit checker by name.

        Args:
            name: Name of the checker to remove

        Returns:
            True if removed, False if not found
        """
        for i, checker in enumerate(self._checkers):
            if checker.name == name:
                del self._checkers[i]
                return True
        return False
