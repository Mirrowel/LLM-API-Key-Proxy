# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Type definitions for the usage tracking package.

This module contains dataclasses and type definitions specific to
usage tracking, limits, and credential selection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union


# =============================================================================
# ENUMS
# =============================================================================


FAIR_CYCLE_GLOBAL_KEY = "_credential_"


class ResetMode(str, Enum):
    """How a usage window resets."""

    ROLLING = "rolling"  # Continuous rolling window
    FIXED_DAILY = "fixed_daily"  # Reset at specific time each day
    CALENDAR_WEEKLY = "calendar_weekly"  # Reset at start of week
    CALENDAR_MONTHLY = "calendar_monthly"  # Reset at start of month
    API_AUTHORITATIVE = "api_authoritative"  # Provider API determines reset


class LimitResult(str, Enum):
    """Result of a limit check."""

    ALLOWED = "allowed"
    BLOCKED_WINDOW = "blocked_window"
    BLOCKED_COOLDOWN = "blocked_cooldown"
    BLOCKED_FAIR_CYCLE = "blocked_fair_cycle"
    BLOCKED_CUSTOM_CAP = "blocked_custom_cap"
    BLOCKED_CONCURRENT = "blocked_concurrent"


class RotationMode(str, Enum):
    """How credentials are rotated."""

    BALANCED = "balanced"  # Weighted random selection
    SEQUENTIAL = "sequential"  # Sticky until exhausted


class TrackingMode(str, Enum):
    """How fair cycle tracks exhaustion."""

    MODEL_GROUP = "model_group"  # Track per quota group or model
    CREDENTIAL = "credential"  # Track per credential globally


class CooldownMode(str, Enum):
    """How custom cap cooldowns are calculated."""

    QUOTA_RESET = "quota_reset"  # Wait until quota window resets
    OFFSET = "offset"  # Add offset seconds to current time
    FIXED = "fixed"  # Use fixed duration


# =============================================================================
# WINDOW TYPES
# =============================================================================


@dataclass
class WindowStats:
    """
    Statistics for a single usage window.

    Tracks usage within a specific time window (e.g., 5-hour, daily).
    """

    name: str  # Window identifier (e.g., "5h", "daily")
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0
    prompt_tokens_cache_read: int = 0
    prompt_tokens_cache_write: int = 0
    approx_cost: float = 0.0
    started_at: Optional[float] = None  # Timestamp when window started
    reset_at: Optional[float] = None  # Timestamp when window resets
    limit: Optional[int] = None  # Max requests allowed (None = unlimited)

    @property
    def remaining(self) -> Optional[int]:
        """Remaining requests in this window, or None if unlimited."""
        if self.limit is None:
            return None
        return max(0, self.limit - self.request_count)

    @property
    def is_exhausted(self) -> bool:
        """True if limit reached."""
        if self.limit is None:
            return False
        return self.request_count >= self.limit


@dataclass
class UsageStats:
    """
    Aggregated usage statistics for a credential.

    Contains both per-window and global (all-time) statistics.
    """

    windows: Dict[str, WindowStats] = field(default_factory=dict)
    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_thinking_tokens: int = 0
    total_output_tokens: int = 0
    total_prompt_tokens_cache_read: int = 0
    total_prompt_tokens_cache_write: int = 0
    total_approx_cost: float = 0.0
    first_used_at: Optional[float] = None
    last_used_at: Optional[float] = None

    # Per-model request counts (for quota group synchronization)
    # Key: normalized model name, Value: request count
    model_request_counts: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# COOLDOWN TYPES
# =============================================================================


@dataclass
class CooldownInfo:
    """
    Information about a cooldown period.

    Cooldowns temporarily block a credential from being used.
    """

    reason: str  # Why the cooldown was applied
    until: float  # Timestamp when cooldown ends
    started_at: float  # Timestamp when cooldown started
    source: str = "system"  # "system", "custom_cap", "rate_limit", "provider_hook"
    model_or_group: Optional[str] = None  # Scope of cooldown (None = credential-wide)
    backoff_count: int = 0  # Number of consecutive cooldowns

    @property
    def remaining_seconds(self) -> float:
        """Seconds remaining in cooldown."""
        import time

        return max(0.0, self.until - time.time())

    @property
    def is_active(self) -> bool:
        """True if cooldown is still in effect."""
        import time

        return time.time() < self.until


# =============================================================================
# FAIR CYCLE TYPES
# =============================================================================


@dataclass
class FairCycleState:
    """
    Fair cycle state for a credential.

    Tracks whether a credential has been exhausted in the current cycle.
    """

    exhausted: bool = False
    exhausted_at: Optional[float] = None
    exhausted_reason: Optional[str] = None
    cycle_request_count: int = 0  # Requests in current cycle
    model_or_group: Optional[str] = None  # Scope of exhaustion


@dataclass
class GlobalFairCycleState:
    """
    Global fair cycle state for a provider.

    Tracks the overall cycle across all credentials.
    """

    cycle_start: float = 0.0  # Timestamp when current cycle started
    all_exhausted_at: Optional[float] = None  # When all credentials exhausted
    cycle_count: int = 0  # How many full cycles completed


# =============================================================================
# CREDENTIAL STATE
# =============================================================================


@dataclass
class CredentialState:
    """
    Complete state for a single credential.

    This is the primary storage unit for credential data.
    """

    # Identity
    stable_id: str  # Email (OAuth) or hash (API key)
    provider: str
    accessor: str  # Current file path or API key
    display_name: Optional[str] = None
    tier: Optional[str] = None
    priority: int = 999  # Lower = higher priority

    # Usage stats
    usage: UsageStats = field(default_factory=UsageStats)

    # Per-model usage stats (for per-model windows)
    model_usage: Dict[str, UsageStats] = field(default_factory=dict)

    # Per-quota-group usage stats (for shared quota windows)
    group_usage: Dict[str, UsageStats] = field(default_factory=dict)

    # Cooldowns (keyed by model/group or "_global_")
    cooldowns: Dict[str, CooldownInfo] = field(default_factory=dict)

    # Fair cycle state (keyed by model/group)
    fair_cycle: Dict[str, FairCycleState] = field(default_factory=dict)

    # Active requests (for concurrent request limiting)
    active_requests: int = 0
    max_concurrent: Optional[int] = None

    # Metadata
    created_at: Optional[float] = None
    last_updated: Optional[float] = None

    def get_cooldown(
        self, model_or_group: Optional[str] = None
    ) -> Optional[CooldownInfo]:
        """Get active cooldown for given scope."""
        import time

        now = time.time()

        # Check specific cooldown
        if model_or_group:
            cooldown = self.cooldowns.get(model_or_group)
            if cooldown and cooldown.until > now:
                return cooldown

        # Check global cooldown
        global_cooldown = self.cooldowns.get("_global_")
        if global_cooldown and global_cooldown.until > now:
            return global_cooldown

        return None

    def is_fair_cycle_exhausted(self, model_or_group: str) -> bool:
        """Check if exhausted for fair cycle purposes."""
        state = self.fair_cycle.get(model_or_group)
        return state.exhausted if state else False

    def get_usage_for_scope(
        self,
        scope: str,
        key: Optional[str] = None,
        create: bool = True,
    ) -> Optional[UsageStats]:
        """Get usage stats for a given scope."""
        if scope == "credential":
            return self.usage
        if scope == "model":
            if not key:
                return self.usage
            if create:
                return self.model_usage.setdefault(key, UsageStats())
            return self.model_usage.get(key)
        if scope == "group":
            if not key:
                return self.usage
            if create:
                return self.group_usage.setdefault(key, UsageStats())
            return self.group_usage.get(key)
        return self.usage


# =============================================================================
# SELECTION TYPES
# =============================================================================


@dataclass
class SelectionContext:
    """
    Context passed to rotation strategies during credential selection.

    Contains all information needed to make a selection decision.
    """

    provider: str
    model: str
    quota_group: Optional[str]  # Quota group for this model
    candidates: List[str]  # Stable IDs of available candidates
    priorities: Dict[str, int]  # stable_id -> priority
    usage_counts: Dict[str, int]  # stable_id -> request count for relevant window
    rotation_mode: RotationMode
    rotation_tolerance: float
    deadline: float


@dataclass
class LimitCheckResult:
    """
    Result of checking all limits for a credential.

    Used by LimitEngine to report why a credential was blocked.
    """

    allowed: bool
    result: LimitResult = LimitResult.ALLOWED
    reason: Optional[str] = None
    blocked_until: Optional[float] = None  # When the block expires

    @classmethod
    def ok(cls) -> "LimitCheckResult":
        """Create an allowed result."""
        return cls(allowed=True, result=LimitResult.ALLOWED)

    @classmethod
    def blocked(
        cls,
        result: LimitResult,
        reason: str,
        blocked_until: Optional[float] = None,
    ) -> "LimitCheckResult":
        """Create a blocked result."""
        return cls(
            allowed=False,
            result=result,
            reason=reason,
            blocked_until=blocked_until,
        )


# =============================================================================
# STORAGE TYPES
# =============================================================================


@dataclass
class StorageSchema:
    """
    Root schema for usage.json storage file.
    """

    schema_version: int = 2
    updated_at: Optional[str] = None  # ISO format
    credentials: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    accessor_index: Dict[str, str] = field(
        default_factory=dict
    )  # accessor -> stable_id
    fair_cycle_global: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )  # provider -> GlobalFairCycleState
