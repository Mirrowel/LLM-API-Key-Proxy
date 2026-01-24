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
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple, Union

if TYPE_CHECKING:
    from .config import WindowDefinition


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


class CapMode(str, Enum):
    """How custom cap max_requests values are interpreted."""

    ABSOLUTE = "absolute"  # e.g., 130 → exactly 130 requests
    OFFSET = "offset"  # e.g., -130 → max - 130, +130 → max + 130
    PERCENTAGE = "percentage"  # e.g., 80% → 80% of max


# =============================================================================
# WINDOW STATS
# =============================================================================


@dataclass
class WindowStats:
    """
    Statistics for a single time-based usage window (e.g., 5h, daily).

    Tracks usage within a specific time window for quota management.
    """

    name: str  # Window identifier (e.g., "5h", "daily")
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # Token stats
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0  # completion + thinking
    prompt_tokens_cache_read: int = 0
    prompt_tokens_cache_write: int = 0
    total_tokens: int = 0

    approx_cost: float = 0.0

    # Window timing
    started_at: Optional[float] = None  # When window period started
    reset_at: Optional[float] = None  # When window resets
    limit: Optional[int] = None  # Max requests allowed (None = unlimited)

    # Historical max tracking (persists across window resets)
    max_recorded_requests: Optional[int] = (
        None  # Highest request_count ever in any window period
    )
    max_recorded_at: Optional[float] = None  # When the max was recorded

    # Usage timing (for smart selection)
    first_used_at: Optional[float] = None  # First request in this window
    last_used_at: Optional[float] = None  # Last request in this window

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


# =============================================================================
# TOTAL STATS
# =============================================================================


@dataclass
class TotalStats:
    """
    All-time totals for a model, group, or credential.

    Tracks cumulative usage across all time (never resets).
    """

    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # Token stats
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    output_tokens: int = 0  # completion + thinking
    prompt_tokens_cache_read: int = 0
    prompt_tokens_cache_write: int = 0
    total_tokens: int = 0

    approx_cost: float = 0.0

    # Timestamps
    first_used_at: Optional[float] = None  # All-time first use
    last_used_at: Optional[float] = None  # All-time last use


# =============================================================================
# MODEL & GROUP STATS CONTAINERS
# =============================================================================


@dataclass
class ModelStats:
    """
    Stats for a single model (own usage only).

    Contains time-based windows and all-time totals.
    Each model only tracks its own usage, not shared quota.
    """

    windows: Dict[str, WindowStats] = field(default_factory=dict)
    totals: TotalStats = field(default_factory=TotalStats)


@dataclass
class GroupStats:
    """
    Stats for a quota group (shared usage).

    Contains time-based windows and all-time totals.
    Updated when ANY model in the group is used.
    """

    windows: Dict[str, WindowStats] = field(default_factory=dict)
    totals: TotalStats = field(default_factory=TotalStats)


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
# USAGE UPDATE (for consolidated tracking)
# =============================================================================


@dataclass
class UsageUpdate:
    """
    All data for a single usage update.

    Used by TrackingEngine.record_usage() to apply updates atomically.
    """

    request_count: int = 1
    success: bool = True

    # Tokens (optional)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking_tokens: int = 0
    prompt_tokens_cache_read: int = 0
    prompt_tokens_cache_write: int = 0
    approx_cost: float = 0.0


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

    # Window definitions for this credential's tier
    # Populated during initialization based on tier/priority
    window_definitions: List["WindowDefinition"] = field(default_factory=list)

    # Stats - source of truth
    model_usage: Dict[str, ModelStats] = field(default_factory=dict)
    group_usage: Dict[str, GroupStats] = field(default_factory=dict)
    totals: TotalStats = field(default_factory=TotalStats)  # Credential-level totals

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

    def get_model_stats(self, model: str, create: bool = True) -> Optional[ModelStats]:
        """Get model stats, optionally creating if not exists."""
        if create:
            return self.model_usage.setdefault(model, ModelStats())
        return self.model_usage.get(model)

    def get_group_stats(self, group: str, create: bool = True) -> Optional[GroupStats]:
        """Get group stats, optionally creating if not exists."""
        if create:
            return self.group_usage.setdefault(group, GroupStats())
        return self.group_usage.get(group)

    def get_window_for_model(
        self, model: str, window_name: str
    ) -> Optional[WindowStats]:
        """Get a specific window for a model."""
        model_stats = self.model_usage.get(model)
        if model_stats:
            return model_stats.windows.get(window_name)
        return None

    def get_window_for_group(
        self, group: str, window_name: str
    ) -> Optional[WindowStats]:
        """Get a specific window for a group."""
        group_stats = self.group_usage.get(group)
        if group_stats:
            return group_stats.windows.get(window_name)
        return None


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
