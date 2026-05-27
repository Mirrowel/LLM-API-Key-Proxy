# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Client-specific type definitions.

Types that are only used within the client package.
Shared types are in core/types.py.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class AvailabilityStats:
    """
    Statistics about credential availability for a model.

    Used for logging and monitoring credential pool status.
    """

    available: int  # Credentials not on cooldown and not exhausted
    on_cooldown: int  # Credentials on cooldown
    fair_cycle_excluded: int  # Credentials excluded by fair cycle
    total: int  # Total credentials for provider

    @property
    def usable(self) -> int:
        """Return count of usable credentials."""
        return self.available

    def __str__(self) -> str:
        parts = [f"{self.available}/{self.total}"]
        if self.on_cooldown > 0:
            parts.append(f"cd:{self.on_cooldown}")
        if self.fair_cycle_excluded > 0:
            parts.append(f"fc:{self.fair_cycle_excluded}")
        return ",".join(parts)


@dataclass
class RetryState:
    """
    State tracking for a retry loop.

    Used by RequestExecutor to track retry attempts and errors.
    """

    tried_credentials: Set[str] = field(default_factory=set)
    last_exception: Optional[Exception] = None
    consecutive_quota_failures: int = 0

    def record_attempt(self, credential: str) -> None:
        """Record that a credential was tried."""
        self.tried_credentials.add(credential)

    def reset_quota_failures(self) -> None:
        """Reset quota failure counter (called after non-quota error)."""
        self.consecutive_quota_failures = 0

    def increment_quota_failures(self) -> None:
        """Increment quota failure counter."""
        self.consecutive_quota_failures += 1


@dataclass
class ExecutionResult:
    """
    Result of executing a request.

    Returned by RequestExecutor to indicate outcome.
    """

    success: bool
    response: Optional[Any] = None
    error: Optional[Exception] = None
    should_rotate: bool = False
    should_fail: bool = False
