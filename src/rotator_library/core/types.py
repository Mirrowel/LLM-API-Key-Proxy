# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Shared type definitions for the rotator library.

This module contains dataclasses and type definitions used across
both the client and usage manager packages.
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)


# =============================================================================
# CREDENTIAL TYPES
# =============================================================================


@dataclass
class CredentialInfo:
    """
    Information about a credential.

    Used for passing credential metadata between components.
    """

    accessor: str  # File path or API key
    stable_id: str  # Email (OAuth) or hash (API key)
    provider: str
    tier: Optional[str] = None
    priority: int = 999  # Lower = higher priority
    display_name: Optional[str] = None


# =============================================================================
# REQUEST TYPES
# =============================================================================


@dataclass
class RequestContext:
    """
    Context for a request being processed.

    Contains all information needed to execute a request with
    retry/rotation logic.
    """

    model: str
    provider: str
    kwargs: Dict[str, Any]
    streaming: bool
    credentials: List[str]
    deadline: float
    request: Optional[Any] = None  # FastAPI Request object
    pre_request_callback: Optional[Callable] = None
    transaction_logger: Optional[Any] = None


@dataclass
class ProcessedChunk:
    """
    Result of processing a streaming chunk.

    Used by StreamingHandler to return processed chunk data.
    """

    sse_string: str  # The SSE-formatted string to yield
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    has_tool_calls: bool = False


# =============================================================================
# FILTER TYPES
# =============================================================================


@dataclass
class FilterResult:
    """
    Result of credential filtering.

    Contains categorized credentials after filtering by tier compatibility.
    """

    compatible: List[str] = field(default_factory=list)  # Known compatible
    unknown: List[str] = field(default_factory=list)  # Unknown tier
    incompatible: List[str] = field(default_factory=list)  # Known incompatible
    priorities: Dict[str, int] = field(default_factory=dict)  # credential -> priority
    tier_names: Dict[str, str] = field(default_factory=dict)  # credential -> tier name

    @property
    def all_usable(self) -> List[str]:
        """Return all usable credentials (compatible + unknown)."""
        return self.compatible + self.unknown


# =============================================================================
# CONFIGURATION TYPES
# =============================================================================


@dataclass
class FairCycleConfig:
    """
    Fair cycle rotation configuration for a provider.

    Fair cycle ensures each credential is used at least once before
    any credential is reused.
    """

    enabled: Optional[bool] = None  # None = derive from rotation mode
    tracking_mode: str = "model_group"  # "model_group" or "credential"
    cross_tier: bool = False  # Track across all tiers
    duration: int = 604800  # 7 days in seconds


@dataclass
class CustomCapConfig:
    """
    Custom cap configuration for a tier/model combination.

    Allows setting usage limits that can be absolute, offset from API limits,
    or percentage of API limits.
    """

    tier_key: Union[int, Tuple[int, ...], str]  # Priority(s) or "default"
    model_or_group: str  # Model name or quota group name
    max_requests: Union[int, str]  # Absolute value, offset, or percentage
    max_requests_mode: str = "absolute"  # "absolute", "offset", "percentage"
    cooldown_mode: str = "quota_reset"  # "quota_reset", "offset", "fixed"
    cooldown_value: int = 0  # Seconds for offset/fixed modes


@dataclass
class WindowConfig:
    """
    Quota window configuration.

    Defines how usage is tracked and reset for a credential.
    """

    name: str  # e.g., "5h", "daily", "weekly"
    duration_seconds: Optional[int]  # None for infinite/total
    reset_mode: str  # "rolling", "fixed_daily", "calendar_weekly", "api_authoritative"
    applies_to: str  # "credential", "group", "model"


@dataclass
class ProviderConfig:
    """
    Complete configuration for a provider.

    Loaded by ConfigLoader and used by both client and usage manager.
    """

    rotation_mode: str = "balanced"  # "balanced" or "sequential"
    rotation_tolerance: float = 3.0
    priority_multipliers: Dict[int, int] = field(default_factory=dict)
    priority_multipliers_by_mode: Dict[str, Dict[int, int]] = field(
        default_factory=dict
    )
    sequential_fallback_multiplier: int = 1
    fair_cycle: FairCycleConfig = field(default_factory=FairCycleConfig)
    custom_caps: List[CustomCapConfig] = field(default_factory=list)
    exhaustion_cooldown_threshold: int = 300  # 5 minutes
    windows: List[WindowConfig] = field(default_factory=list)


# =============================================================================
# HOOK RESULT TYPES
# =============================================================================


@dataclass
class RequestCompleteResult:
    """
    Result from on_request_complete provider hook.

    Allows providers to customize how requests are counted and cooled down.
    """

    count_override: Optional[int] = None  # How many requests to count
    cooldown_override: Optional[float] = None  # Custom cooldown duration
    force_exhausted: bool = False  # Mark for fair cycle


# =============================================================================
# ERROR ACTION ENUM
# =============================================================================


class ErrorAction:
    """
    Actions to take after an error.

    Used by RequestExecutor to determine next steps.
    """

    RETRY_SAME = "retry_same"  # Retry with same credential
    ROTATE = "rotate"  # Try next credential
    FAIL = "fail"  # Fail the request immediately
