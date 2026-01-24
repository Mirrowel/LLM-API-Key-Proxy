# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Example Provider Implementation with Custom Usage Management.

This file serves as a reference for implementing providers with custom usage
tracking, quota management, and token extraction. Copy this file and modify
it for your specific provider.

=============================================================================
ARCHITECTURE OVERVIEW
=============================================================================

The usage management system is per-provider. Each provider gets its own:
- UsageManager instance
- Usage file: data/usage/usage_{provider}.json
- Configuration (ProviderUsageConfig)

Data flows like this:

    Request → Executor → Provider transforms → API call → Response
                ↓                                            ↓
           UsageManager ← TrackingEngine ← Token extraction ←┘
                ↓
           Persistence (usage_{provider}.json)

Providers customize behavior through:
1. Class attributes (declarative configuration)
2. Methods (behavioral overrides)
3. Hooks (request lifecycle callbacks)

=============================================================================
USAGE STATS SCHEMA
=============================================================================

UsageStats (tracked at global/model/group levels):
    total_requests: int              # All requests
    total_successes: int             # Successful requests
    total_failures: int              # Failed requests
    total_tokens: int                # All tokens combined
    total_prompt_tokens: int         # Input tokens
    total_completion_tokens: int     # Output tokens (content only)
    total_thinking_tokens: int       # Reasoning/thinking tokens
    total_output_tokens: int         # completion + thinking
    total_prompt_tokens_cache_read: int   # Cached input tokens read
    total_prompt_tokens_cache_write: int  # Cached input tokens written
    total_approx_cost: float         # Estimated cost
    first_used_at: float             # Timestamp
    last_used_at: float              # Timestamp
    windows: Dict[str, WindowStats]  # Per-window breakdown

WindowStats (per time window: "5h", "daily", "total"):
    request_count: int
    success_count: int
    failure_count: int
    prompt_tokens: int
    completion_tokens: int
    thinking_tokens: int
    output_tokens: int
    prompt_tokens_cache_read: int
    prompt_tokens_cache_write: int
    total_tokens: int
    approx_cost: float
    started_at: float
    reset_at: float
    limit: int | None

=============================================================================
"""

import asyncio
import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .provider_interface import ProviderInterface, QuotaGroupMap

# Alias for clarity in examples
ProviderPlugin = ProviderInterface

# Import these types for hook returns and usage manager access
from ..core.types import RequestCompleteResult
from ..usage import UsageManager, ProviderUsageConfig, WindowDefinition
from ..usage.types import ResetMode, RotationMode, CooldownMode

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# INTERNAL RETRY COUNTING (ContextVar Pattern)
# =============================================================================
#
# When your provider performs internal retries (e.g., for transient errors,
# empty responses, or rate limits), each retry is an API call that should be
# counted for accurate usage tracking.
#
# The challenge: Instance variables (self.count) are shared across concurrent
# requests, so they can't be used safely. ContextVar solves this by giving
# each async task its own isolated value.
#
# Usage pattern:
#   1. Reset to 1 at the start of your retry loop
#   2. Increment before each retry
#   3. Read in on_request_complete() to report the actual count
#
# Example:
#   _attempt_count.set(1)  # Reset
#   for attempt in range(max_attempts):
#       try:
#           result = await api_call()
#           return result
#       except RetryableError:
#           _attempt_count.set(_attempt_count.get() + 1)  # Increment
#           continue
#
# Then on_request_complete returns RequestCompleteResult(count_override=_attempt_count.get())
#
_example_attempt_count: ContextVar[int] = ContextVar(
    "example_provider_attempt_count", default=1
)


# =============================================================================
# EXAMPLE PROVIDER IMPLEMENTATION
# =============================================================================


class ExampleProvider(ProviderPlugin):
    """
    Example provider demonstrating all usage management customization points.

    This provider shows how to:
    - Configure rotation and quota behavior
    - Define model quota groups
    - Extract tokens from provider-specific response formats
    - Override request counting via hooks
    - Run background quota refresh jobs
    - Define custom usage windows
    """

    # =========================================================================
    # REQUIRED: BASIC PROVIDER IDENTITY
    # =========================================================================

    provider_name = "example"  # Used in model prefix: "example/gpt-4"
    provider_env_name = "EXAMPLE"  # For env vars: EXAMPLE_API_KEY, etc.

    # =========================================================================
    # USAGE MANAGEMENT: CLASS ATTRIBUTES (DECLARATIVE)
    # =========================================================================

    # -------------------------------------------------------------------------
    # ROTATION MODE
    # -------------------------------------------------------------------------
    # Controls how credentials are selected for requests.
    #
    # Options:
    #   "balanced"   - Weighted random selection based on usage (default)
    #   "sequential" - Stick to one credential until exhausted, then rotate
    #
    # Sequential mode is better for:
    #   - Providers with per-credential rate limits
    #   - Maximizing cache hits (same credential = same context)
    #   - Providers where switching credentials has overhead
    #
    # Balanced mode is better for:
    #   - Even distribution across credentials
    #   - Providers without per-credential state
    #
    default_rotation_mode = "sequential"

    # -------------------------------------------------------------------------
    # MODEL QUOTA GROUPS
    # -------------------------------------------------------------------------
    # Models in the same group share a quota pool. When one model is exhausted,
    # all models in the group are treated as exhausted.
    #
    # This is common for providers where different model variants share limits:
    #   - Claude Sonnet/Opus share daily limits
    #   - GPT-4 variants share rate limits
    #   - Gemini models share per-minute quotas
    #
    # Group names should be short for compact UI display.
    #
    # Can be overridden via environment:
    #   QUOTA_GROUPS_EXAMPLE_GPT4="gpt-4o,gpt-4o-mini,gpt-4-turbo"
    #
    model_quota_groups: QuotaGroupMap = {
        # GPT-4 variants share quota
        "gpt4": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
        ],
        # Claude models share quota
        "claude": [
            "claude-3-opus",
            "claude-3-sonnet",
            "claude-3-haiku",
        ],
        # Standalone model (no sharing)
        "whisper": [
            "whisper-1",
        ],
    }

    # -------------------------------------------------------------------------
    # PRIORITY MULTIPLIERS (CONCURRENCY)
    # -------------------------------------------------------------------------
    # Higher priority credentials (lower number) can handle more concurrent
    # requests. This is useful for paid vs free tier credentials.
    #
    # Priority is assigned per-credential via:
    #   - .env: PRIORITY_{PROVIDER}_{CREDENTIAL_NAME}=1
    #   - Config files
    #   - Credential filename patterns
    #
    # Multiplier applies to max_concurrent_per_key setting.
    # Example: max_concurrent_per_key=6, priority 1 multiplier=5 → 30 concurrent
    #
    default_priority_multipliers = {
        1: 5,  # Ultra tier: 5x concurrent
        2: 3,  # Standard paid: 3x concurrent
        3: 2,  # Free tier: 2x concurrent
        # Others: Use fallback multiplier
    }

    # For sequential mode, credentials not in priority_multipliers get this.
    # For balanced mode, they get 1x (no multiplier).
    default_sequential_fallback_multiplier = 2

    # -------------------------------------------------------------------------
    # CUSTOM CAPS
    # -------------------------------------------------------------------------
    # Apply stricter limits than the actual API limits. Useful for:
    #   - Reserving quota for critical requests
    #   - Preventing runaway usage
    #   - Testing rotation behavior
    #
    # Structure: {priority: {model_or_group: config}}
    # Or: {(priority1, priority2): {model_or_group: config}} for multiple tiers
    #
    # Config options:
    #   max_requests: int or "80%" (percentage of actual limit)
    #   cooldown_mode: "quota_reset" | "offset" | "fixed"
    #   cooldown_value: seconds for offset/fixed modes
    #
    default_custom_caps = {
        # Tier 3 (free tier) - cap at 50 requests, cooldown until API resets
        3: {
            "gpt4": {
                "max_requests": 50,
                "cooldown_mode": "quota_reset",
            },
            "claude": {
                "max_requests": 30,
                "cooldown_mode": "quota_reset",
            },
        },
        # Tiers 2 and 3 together - cap at 80% of actual limit
        (2, 3): {
            "whisper": {
                "max_requests": "80%",  # 80% of actual API limit
                "cooldown_mode": "offset",
                "cooldown_value": 1800,  # +30 min buffer after hitting cap
            },
        },
        # Default for unknown tiers
        "default": {
            "gpt4": {
                "max_requests": 100,
                "cooldown_mode": "fixed",
                "cooldown_value": 3600,  # 1 hour fixed cooldown
            },
        },
    }

    # -------------------------------------------------------------------------
    # MODEL USAGE WEIGHTS
    # -------------------------------------------------------------------------
    # Some models consume more quota per request. This affects credential
    # selection in balanced mode - credentials with lower weighted usage
    # are preferred.
    #
    # Example: Opus costs 2x what Sonnet does per request
    #
    model_usage_weights = {
        "claude-3-opus": 2,
        "gpt-4-turbo": 2,
        # Default is 1 for unlisted models
    }

    # -------------------------------------------------------------------------
    # FAIR CYCLE CONFIGURATION
    # -------------------------------------------------------------------------
    # Fair cycle ensures all credentials get used before any is reused.
    # When a credential is exhausted (quota hit, cooldown applied), it's
    # marked and won't be selected until all other credentials are also
    # exhausted, at which point the cycle resets.
    #
    # This is enabled by default for sequential mode.
    #
    # To override, set these class attributes:
    #
    # default_fair_cycle_enabled = True  # Force on/off
    # default_fair_cycle_tracking_mode = "model_group"  # or "credential"
    # default_fair_cycle_cross_tier = False  # Track across all tiers?
    # default_fair_cycle_duration = 3600  # Cycle duration in seconds

    # =========================================================================
    # USAGE MANAGEMENT: METHODS (BEHAVIORAL)
    # =========================================================================

    def normalize_model_for_tracking(self, model: str) -> str:
        """
        Normalize internal model names to public-facing names for tracking.

        Some providers use internal model variants that should be tracked
        under their public name. This ensures usage files only contain
        user-facing model names.

        Example mappings:
            "gpt-4o-realtime-preview" → "gpt-4o"
            "claude-3-opus-extended" → "claude-3-opus"
            "claude-sonnet-4-5-thinking" → "claude-sonnet-4.5"

        Args:
            model: Model name (may include provider prefix: "example/gpt-4o")

        Returns:
            Normalized model name (preserves prefix if present)
        """
        has_prefix = "/" in model
        if has_prefix:
            provider, clean_model = model.split("/", 1)
        else:
            clean_model = model

        # Define your internal → public mappings
        internal_to_public = {
            "gpt-4o-realtime-preview": "gpt-4o",
            "gpt-4o-realtime": "gpt-4o",
            "claude-3-opus-extended": "claude-3-opus",
        }

        normalized = internal_to_public.get(clean_model, clean_model)

        if has_prefix:
            return f"{provider}/{normalized}"
        return normalized

    def on_request_complete(
        self,
        credential: str,
        model: str,
        success: bool,
        response: Optional[Any],
        error: Optional[Any],
    ) -> Optional[RequestCompleteResult]:
        """
        Hook called after each request completes (success or failure).

        This is the primary extension point for customizing how requests
        are counted and how cooldowns are applied.

        Use cases:
            - Don't count server errors as quota usage
            - Apply custom cooldowns based on error type
            - Force credential exhaustion for fair cycle
            - Count internal retries accurately (see ContextVar pattern below)

        Args:
            credential: The credential accessor (file path or API key)
            model: Model that was called
            success: Whether the request succeeded
            response: Response object (if success=True)
            error: ClassifiedError object (if success=False)

        Returns:
            RequestCompleteResult to override behavior, or None for default.

            RequestCompleteResult fields:
                count_override: int | None
                    - 0 = Don't count this request against quota
                    - N = Count as N requests
                    - None = Use default (1 for success, 1 for countable errors)

                cooldown_override: float | None
                    - Seconds to cool down this credential
                    - Applied in addition to any error-based cooldown

                force_exhausted: bool
                    - True = Mark credential as exhausted for fair cycle
                    - Useful for quota errors even without long cooldown
        """
        # =====================================================================
        # PATTERN: Counting Internal Retries with ContextVar
        # =====================================================================
        # If your provider performs internal retries, report the actual count:
        #
        # 1. At module level, define:
        #    _attempt_count: ContextVar[int] = ContextVar('my_attempt_count', default=1)
        #
        # 2. In your retry loop:
        #    _attempt_count.set(1)  # Reset at start
        #    for attempt in range(max_attempts):
        #        try:
        #            return await api_call()
        #        except RetryableError:
        #            _attempt_count.set(_attempt_count.get() + 1)  # Increment before retry
        #            continue
        #
        # 3. Here, report the count:
        attempt_count = _example_attempt_count.get()
        _example_attempt_count.set(1)  # Reset for safety

        if attempt_count > 1:
            lib_logger.debug(
                f"Request to {model} used {attempt_count} API calls (internal retries)"
            )
            return RequestCompleteResult(count_override=attempt_count)

        # =====================================================================
        # PATTERN: Don't Count Server Errors
        # =====================================================================
        # Server errors (5xx) shouldn't count against quota since they're
        # not the user's fault and don't consume API quota.
        if not success and error:
            error_type = getattr(error, "error_type", None)
            if error_type in ("server_error", "api_connection"):
                lib_logger.debug(
                    f"Not counting {error_type} error against quota for {model}"
                )
                return RequestCompleteResult(count_override=0)

        # =====================================================================
        # PATTERN: Custom Cooldown for Rate Limits
        # =====================================================================
        if not success and error:
            error_type = getattr(error, "error_type", None)
            if error_type == "rate_limit":
                # Check for retry-after header
                retry_after = getattr(error, "retry_after", None)
                if retry_after and retry_after > 60:
                    # Long rate limit - mark as exhausted
                    return RequestCompleteResult(
                        cooldown_override=retry_after,
                        force_exhausted=True,
                    )
                elif retry_after:
                    # Short rate limit - just cooldown
                    return RequestCompleteResult(cooldown_override=retry_after)

        # =====================================================================
        # PATTERN: Force Exhaustion on Quota Exceeded
        # =====================================================================
        if not success and error:
            error_type = getattr(error, "error_type", None)
            if error_type == "quota_exceeded":
                return RequestCompleteResult(
                    force_exhausted=True,
                    cooldown_override=3600.0,  # Default 1 hour if no reset time
                )

        # Default behavior
        return None

    # =========================================================================
    # BACKGROUND JOBS
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Configure periodic background tasks.

        Common use cases:
            - Refresh quota baselines from API
            - Clean up expired cache entries
            - Preemptively refresh OAuth tokens

        Returns:
            None if no background job, otherwise:
            {
                "interval": 300,        # Seconds between runs
                "name": "quota_refresh", # For logging
                "run_on_start": True,   # Run immediately at startup?
            }
        """
        return {
            "interval": 600,  # Every 10 minutes
            "name": "quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: UsageManager,
        credentials: List[str],
    ) -> None:
        """
        Periodic background task execution.

        Called by BackgroundRefresher at the interval specified in
        get_background_job_config().

        Common tasks:
            - Fetch current quota from API and update usage manager
            - Clean up stale cache entries
            - Refresh tokens proactively

        Args:
            usage_manager: The UsageManager for this provider
            credentials: List of credential accessors (file paths or keys)
        """
        lib_logger.debug(f"Running background job for {self.provider_name}")

        for cred in credentials:
            try:
                # Example: Fetch quota from provider API
                quota_info = await self._fetch_quota_from_api(cred)

                if quota_info:
                    for model, info in quota_info.items():
                        # Update usage manager with fresh quota data
                        await usage_manager.update_quota_baseline(
                            accessor=cred,
                            model=model,
                            quota_max_requests=info.get("limit"),
                            quota_reset_ts=info.get("reset_ts"),
                            quota_used=info.get("used"),
                            quota_group=info.get("group"),
                        )

            except Exception as e:
                lib_logger.warning(f"Quota refresh failed for {cred}: {e}")

    async def _fetch_quota_from_api(
        self,
        credential: str,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Fetch current quota information from provider API.

        Override this with actual API calls for your provider.

        Returns:
            Dict mapping model names to quota info:
            {
                "gpt-4o": {
                    "limit": 500,
                    "used": 123,
                    "reset_ts": 1735689600.0,
                    "group": "gpt4",  # Optional
                },
                ...
            }
        """
        # Placeholder - implement actual API call
        return None

    # =========================================================================
    # TOKEN EXTRACTION
    # =========================================================================

    def _build_usage_from_response(
        self,
        response: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Build standardized usage dict from provider-specific response.

        The usage manager expects a standardized format. If your provider
        returns a different format, convert it here.

        Standard format:
        {
            "prompt_tokens": int,           # Input tokens
            "completion_tokens": int,       # Output tokens (content + thinking)
            "total_tokens": int,            # All tokens

            # Optional: Input breakdown
            "prompt_tokens_details": {
                "cached_tokens": int,       # Cache read tokens
                "cache_creation_tokens": int, # Cache write tokens
            },

            # Optional: Output breakdown
            "completion_tokens_details": {
                "reasoning_tokens": int,    # Thinking/reasoning tokens
            },

            # Alternative top-level fields (some APIs use these)
            "cache_read_tokens": int,
            "cache_creation_tokens": int,
        }

        Args:
            response: Raw response from provider API

        Returns:
            Standardized usage dict, or None if no usage data
        """
        if not hasattr(response, "usage") or not response.usage:
            return None

        # Example: Provider returns Gemini-style metadata
        # Adapt this to your provider's format
        usage = response.usage

        # Standard fields
        result = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

        # Example: Extract cached tokens from details
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        if prompt_details:
            if isinstance(prompt_details, dict):
                cached = prompt_details.get("cached_tokens", 0)
                cache_write = prompt_details.get("cache_creation_tokens", 0)
            else:
                cached = getattr(prompt_details, "cached_tokens", 0)
                cache_write = getattr(prompt_details, "cache_creation_tokens", 0)

            if cached or cache_write:
                result["prompt_tokens_details"] = {}
                if cached:
                    result["prompt_tokens_details"]["cached_tokens"] = cached
                if cache_write:
                    result["prompt_tokens_details"]["cache_creation_tokens"] = (
                        cache_write
                    )

        # Example: Extract thinking tokens from details
        completion_details = getattr(usage, "completion_tokens_details", None)
        if completion_details:
            if isinstance(completion_details, dict):
                reasoning = completion_details.get("reasoning_tokens", 0)
            else:
                reasoning = getattr(completion_details, "reasoning_tokens", 0)

            if reasoning:
                result["completion_tokens_details"] = {"reasoning_tokens": reasoning}

        return result


# =============================================================================
# CUSTOM WINDOWS
# =============================================================================
#
# To add custom usage windows, you have two options:
#
# OPTION 1: Override windows via provider config (recommended)
# ------------------------------------------------------------
# Add class attribute to your provider:
#
#     default_windows = [
#         WindowDefinition.rolling("1h", 3600, is_primary=False),
#         WindowDefinition.rolling("6h", 21600, is_primary=True),
#         WindowDefinition.daily("daily"),
#         WindowDefinition.total("total"),
#     ]
#
# WindowDefinition options:
#   - name: str - Window identifier (e.g., "1h", "daily")
#   - duration_seconds: int | None - Window duration (None for "total")
#   - reset_mode: ResetMode - How window resets
#       - ROLLING: Continuous sliding window
#       - FIXED_DAILY: Reset at specific UTC time
#       - CALENDAR_WEEKLY: Reset at week start
#       - CALENDAR_MONTHLY: Reset at month start
#       - API_AUTHORITATIVE: Provider determines reset
#   - is_primary: bool - Used for rotation decisions
#   - applies_to: str - Scope of window
#       - "credential": Global per-credential
#       - "model": Per-model per-credential
#       - "group": Per-quota-group per-credential
#
# OPTION 2: Build config manually in RotatingClient
# -------------------------------------------------
# In your client initialization:
#
#     from rotator_library.usage.config import (
#         ProviderUsageConfig,
#         WindowDefinition,
#         FairCycleConfig,
#     )
#     from rotator_library.usage.types import RotationMode, ResetMode
#
#     config = ProviderUsageConfig(
#         rotation_mode=RotationMode.SEQUENTIAL,
#         windows=[
#             WindowDefinition(
#                 name="1h",
#                 duration_seconds=3600,
#                 reset_mode=ResetMode.ROLLING,
#                 is_primary=False,
#                 applies_to="model",
#             ),
#             WindowDefinition(
#                 name="6h",
#                 duration_seconds=21600,
#                 reset_mode=ResetMode.ROLLING,
#                 is_primary=True,  # Primary for rotation
#                 applies_to="group",  # Track per quota group
#             ),
#         ],
#         fair_cycle=FairCycleConfig(
#             enabled=True,
#             tracking_mode=TrackingMode.MODEL_GROUP,
#         ),
#     )
#
#     manager = UsageManager(
#         provider="example",
#         config=config,
#         file_path="usage_example.json",
#     )
#
# =============================================================================


# =============================================================================
# REGISTERING YOUR PROVIDER
# =============================================================================
#
# To register your provider with the system:
#
# 1. Add to PROVIDER_PLUGINS dict in src/rotator_library/providers/__init__.py:
#
#     from .example_provider import ExampleProvider
#
#     PROVIDER_PLUGINS = {
#         ...
#         "example": ExampleProvider,
#     }
#
# 2. Add credential discovery in RotatingClient if using OAuth:
#
#     # In _discover_oauth_credentials:
#     if provider == "example":
#         creds = self._discover_example_credentials()
#
# 3. Configure via environment variables:
#
#     # API key credentials
#     EXAMPLE_API_KEY=sk-xxx
#     EXAMPLE_API_KEY_2=sk-yyy
#
#     # OAuth credential paths
#     EXAMPLE_OAUTH_PATHS=./creds/example_*.json
#
#     # Priority/tier assignment
#     PRIORITY_EXAMPLE_CRED1=1
#     TIER_EXAMPLE_CRED2=standard-tier
#
#     # Quota group overrides
#     QUOTA_GROUPS_EXAMPLE_GPT4=gpt-4o,gpt-4o-mini,gpt-4-turbo
#
# =============================================================================


# =============================================================================
# ACCESSING USAGE DATA
# =============================================================================
#
# The usage manager exposes data through several methods:
#
# 1. Get availability stats (for UI/monitoring):
#
#     stats = await usage_manager.get_availability_stats(model, quota_group)
#     # Returns: {
#     #     "total": 10,
#     #     "available": 7,
#     #     "blocked_by": {"cooldowns": 2, "fair_cycle": 1},
#     #     "rotation_mode": "sequential",
#     # }
#
# 2. Get comprehensive stats (for quota-stats endpoint):
#
#     stats = await usage_manager.get_stats_for_endpoint()
#     # Returns full credential/model/group breakdown
#
# 3. Direct state access (for advanced use):
#
#     # Get credential state
#     state = usage_manager.states.get(stable_id)
#
#     # Access usage at different scopes
#     global_usage = state.usage
#     model_usage = state.model_usage.get("gpt-4o")
#     group_usage = state.group_usage.get("gpt4")
#
#     # Check cooldowns
#     cooldown = state.get_cooldown("gpt4")
#     if cooldown and cooldown.is_active:
#         print(f"Cooldown remaining: {cooldown.remaining_seconds}s")
#
#     # Check fair cycle
#     fc = state.fair_cycle.get("gpt4")
#     if fc and fc.exhausted:
#         print(f"Exhausted at: {fc.exhausted_at}")
#
# 4. Update quota baseline (from API response):
#
#     await usage_manager.update_quota_baseline(
#         accessor=credential,
#         model="gpt-4o",
#         quota_max_requests=500,
#         quota_reset_ts=time.time() + 3600,
#         quota_used=123,
#         quota_group="gpt4",
#     )
#
# =============================================================================
