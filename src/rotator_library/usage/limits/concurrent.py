# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Concurrent request limit checker.

Blocks credentials that have reached their max_concurrent limit.
"""

from typing import Optional

from ..types import CredentialState, LimitCheckResult, LimitResult
from .base import LimitChecker


class ConcurrentLimitChecker(LimitChecker):
    """
    Checks concurrent request limits.

    Blocks credentials that have active_requests >= max_concurrent.
    This ensures we don't overload any single credential.
    """

    @property
    def name(self) -> str:
        return "concurrent"

    def check(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> LimitCheckResult:
        """
        Check if credential is at max concurrent.

        Args:
            state: Credential state to check
            model: Model being requested
            quota_group: Quota group for this model

        Returns:
            LimitCheckResult indicating pass/fail
        """
        # If no limit set, always allow
        if state.max_concurrent is None:
            return LimitCheckResult.ok()

        # Check if at or above limit
        if state.active_requests >= state.max_concurrent:
            return LimitCheckResult.blocked(
                result=LimitResult.BLOCKED_CONCURRENT,
                reason=f"At max concurrent: {state.active_requests}/{state.max_concurrent}",
                blocked_until=None,  # No specific time - depends on request completion
            )

        return LimitCheckResult.ok()

    def reset(
        self,
        state: CredentialState,
        model: Optional[str] = None,
        quota_group: Optional[str] = None,
    ) -> None:
        """
        Reset concurrent count.

        Note: This is rarely needed as active_requests is
        managed by acquire/release, not limit checking.
        """
        # Typically don't reset active_requests via limit system
        pass
