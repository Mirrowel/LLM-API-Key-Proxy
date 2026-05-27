# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Base interface for limit checkers.

All limit types implement this interface for consistent behavior.
"""

from abc import ABC, abstractmethod
from typing import Optional

from ..types import CredentialState, LimitCheckResult, LimitResult


class LimitChecker(ABC):
    """
    Abstract base class for limit checkers.

    Each limit type (window, cooldown, fair cycle, custom cap)
    implements this interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this limit checker."""
        ...

    @abstractmethod
    def check(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
    ) -> LimitCheckResult:
        """
        Check if a credential passes this limit.

        Args:
            state: Credential state to check
            model: Model being requested
            quota_group: Quota group for this model

        Returns:
            LimitCheckResult indicating pass/fail and reason
        """
        ...

    def reset(
        self,
        state: CredentialState,
        model: Optional[str] = None,
        quota_group: Optional[str] = None,
    ) -> None:
        """
        Reset this limit for a credential.

        Default implementation does nothing - override if needed.

        Args:
            state: Credential state to reset
            model: Optional model scope
            quota_group: Optional quota group scope
        """
        pass
