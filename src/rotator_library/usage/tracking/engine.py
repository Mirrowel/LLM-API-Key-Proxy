# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Tracking engine for usage recording.

Central component for recording requests, successes, and failures.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

from ..types import (
    UsageStats,
    WindowStats,
    CredentialState,
    CooldownInfo,
    FairCycleState,
    TrackingMode,
    FAIR_CYCLE_GLOBAL_KEY,
)
from ..config import WindowDefinition, ProviderUsageConfig
from .windows import WindowManager

lib_logger = logging.getLogger("rotator_library")


class TrackingEngine:
    """
    Central engine for usage tracking.

    Responsibilities:
    - Recording request successes and failures
    - Managing usage windows
    - Updating global statistics
    - Managing cooldowns
    - Tracking fair cycle state
    """

    def __init__(
        self,
        window_manager: WindowManager,
        config: ProviderUsageConfig,
    ):
        """
        Initialize tracking engine.

        Args:
            window_manager: WindowManager instance for window operations
            config: Provider usage configuration
        """
        self._windows = window_manager
        self._config = config
        self._lock = asyncio.Lock()

    async def record_success(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        prompt_tokens_cached: int = 0,
        approx_cost: float = 0.0,
        request_count: int = 1,
        response_headers: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a successful request.

        Args:
            state: Credential state to update
            model: Model that was used
            quota_group: Quota group for this model (None = use model name)
            prompt_tokens: Prompt tokens used
            completion_tokens: Completion tokens used
            prompt_tokens_cached: Cached prompt tokens (e.g., from Claude)
            response_headers: Optional response headers with rate limit info
        """
        async with self._lock:
            now = time.time()
            group_key = quota_group or model
            fair_cycle_key = self._resolve_fair_cycle_key(group_key)

            # Update usage stats
            usage = state.usage
            usage.total_requests += request_count
            usage.total_successes += request_count
            usage.total_tokens += (
                prompt_tokens + completion_tokens + prompt_tokens_cached
            )
            usage.total_prompt_tokens_cached += prompt_tokens_cached
            usage.total_approx_cost += approx_cost
            usage.last_used_at = now
            if usage.first_used_at is None:
                usage.first_used_at = now

            self._update_scoped_usage(
                state,
                scope="model",
                key=model,
                now=now,
                request_count=request_count,
                success_count=request_count,
                failure_count=0,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                prompt_tokens_cached=prompt_tokens_cached,
                approx_cost=approx_cost,
            )
            if group_key != model:
                self._update_scoped_usage(
                    state,
                    scope="group",
                    key=group_key,
                    now=now,
                    request_count=request_count,
                    success_count=request_count,
                    failure_count=0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    prompt_tokens_cached=prompt_tokens_cached,
                    approx_cost=approx_cost,
                )

            # Update per-model request count (for quota group sync)
            usage.model_request_counts[model] = (
                usage.model_request_counts.get(model, 0) + request_count
            )

            # Record in all windows
            for window_def in self._config.windows:
                scoped_usage = self._get_usage_for_window(
                    state, window_def, model, group_key
                )
                if scoped_usage is None:
                    continue
                window = self._windows.record_request(
                    scoped_usage.windows,
                    window_def.name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    prompt_tokens_cached=prompt_tokens_cached,
                    approx_cost=approx_cost,
                    request_count=request_count,
                )
                if window.limit is not None and window.request_count >= window.limit:
                    if self._config.fair_cycle.enabled:
                        self._mark_exhausted(state, fair_cycle_key, "window_limit")

            # Update from response headers if provided
            if response_headers:
                self._update_from_headers(state, response_headers, model, group_key)

            # Update fair cycle request count
            if self._config.fair_cycle.enabled:
                fc_state = state.fair_cycle.get(fair_cycle_key)
                if not fc_state:
                    fc_state = FairCycleState(model_or_group=fair_cycle_key)
                    state.fair_cycle[fair_cycle_key] = fc_state
                fc_state.cycle_request_count += request_count

            state.last_updated = now

    async def record_failure(
        self,
        state: CredentialState,
        model: str,
        error_type: str,
        quota_group: Optional[str] = None,
        cooldown_duration: Optional[float] = None,
        quota_reset_timestamp: Optional[float] = None,
        mark_exhausted: bool = False,
        request_count: int = 1,
    ) -> None:
        """
        Record a failed request.

        Args:
            state: Credential state to update
            model: Model that was used
            error_type: Type of error (quota_exceeded, rate_limit, etc.)
            quota_group: Quota group for this model
            cooldown_duration: How long to cool down (if applicable)
            quota_reset_timestamp: When quota resets (from API)
            mark_exhausted: Whether to mark as exhausted for fair cycle
        """
        async with self._lock:
            now = time.time()
            group_key = quota_group or model
            fair_cycle_key = self._resolve_fair_cycle_key(group_key)

            # Update failure stats
            state.usage.total_requests += request_count
            state.usage.total_failures += request_count
            state.usage.last_used_at = now

            self._update_scoped_usage(
                state,
                scope="model",
                key=model,
                now=now,
                request_count=request_count,
                success_count=0,
                failure_count=request_count,
                prompt_tokens=0,
                completion_tokens=0,
                prompt_tokens_cached=0,
                approx_cost=0.0,
            )
            if group_key != model:
                self._update_scoped_usage(
                    state,
                    scope="group",
                    key=group_key,
                    now=now,
                    request_count=request_count,
                    success_count=0,
                    failure_count=request_count,
                    prompt_tokens=0,
                    completion_tokens=0,
                    prompt_tokens_cached=0,
                    approx_cost=0.0,
                )

            # Update per-model request count (for quota group sync)
            state.usage.model_request_counts[model] = (
                state.usage.model_request_counts.get(model, 0) + request_count
            )

            # Record failure in windows (counts against quota)
            for window_def in self._config.windows:
                scoped_usage = self._get_usage_for_window(
                    state, window_def, model, group_key
                )
                if scoped_usage is None:
                    continue
                window = self._windows.record_request(
                    scoped_usage.windows,
                    window_def.name,
                    request_count=request_count,
                )
                if window.limit is not None and window.request_count >= window.limit:
                    if self._config.fair_cycle.enabled:
                        self._mark_exhausted(state, fair_cycle_key, "window_limit")

            # Apply cooldown if specified
            if cooldown_duration is not None and cooldown_duration > 0:
                self._apply_cooldown(
                    state=state,
                    reason=error_type,
                    duration=cooldown_duration,
                    model_or_group=group_key,
                    source="error",
                )

            # Use quota reset timestamp if provided
            if quota_reset_timestamp is not None:
                self._apply_cooldown(
                    state=state,
                    reason=error_type,
                    until=quota_reset_timestamp,
                    model_or_group=group_key,
                    source="api_quota",
                )

            # Mark exhausted for fair cycle if requested
            if mark_exhausted:
                self._mark_exhausted(state, fair_cycle_key, error_type)

            if self._config.fair_cycle.enabled:
                fc_state = state.fair_cycle.get(fair_cycle_key)
                if not fc_state:
                    fc_state = FairCycleState(model_or_group=fair_cycle_key)
                    state.fair_cycle[fair_cycle_key] = fc_state
                fc_state.cycle_request_count += request_count

            state.last_updated = now

    async def acquire(
        self,
        state: CredentialState,
        model: str,
    ) -> bool:
        """
        Acquire a credential for a request (increment active count).

        Args:
            state: Credential state
            model: Model being used

        Returns:
            True if acquired, False if at max concurrent
        """
        async with self._lock:
            # Check concurrent limit
            if state.max_concurrent is not None:
                if state.active_requests >= state.max_concurrent:
                    return False

            state.active_requests += 1
            return True

    async def release(
        self,
        state: CredentialState,
        model: str,
    ) -> None:
        """
        Release a credential after request completes.

        Args:
            state: Credential state
            model: Model that was used
        """
        async with self._lock:
            return

    async def apply_cooldown(
        self,
        state: CredentialState,
        reason: str,
        duration: Optional[float] = None,
        until: Optional[float] = None,
        model_or_group: Optional[str] = None,
        source: str = "system",
    ) -> None:
        """
        Apply a cooldown to a credential.

        Args:
            state: Credential state
            reason: Why the cooldown was applied
            duration: Cooldown duration in seconds (if not using 'until')
            until: Timestamp when cooldown ends (if not using 'duration')
            model_or_group: Scope of cooldown (None = credential-wide)
            source: Source of cooldown (system, custom_cap, rate_limit, etc.)
        """
        async with self._lock:
            self._apply_cooldown(
                state=state,
                reason=reason,
                duration=duration,
                until=until,
                model_or_group=model_or_group,
                source=source,
            )

    async def clear_cooldown(
        self,
        state: CredentialState,
        model_or_group: Optional[str] = None,
    ) -> None:
        """
        Clear a cooldown from a credential.

        Args:
            state: Credential state
            model_or_group: Scope of cooldown to clear (None = global)
        """
        async with self._lock:
            key = model_or_group or "_global_"
            if key in state.cooldowns:
                del state.cooldowns[key]

    async def mark_exhausted(
        self,
        state: CredentialState,
        model_or_group: str,
        reason: str,
    ) -> None:
        """
        Mark a credential as exhausted for fair cycle.

        Args:
            state: Credential state
            model_or_group: Scope of exhaustion
            reason: Why credential was exhausted
        """
        async with self._lock:
            self._mark_exhausted(state, model_or_group, reason)

    async def reset_fair_cycle(
        self,
        state: CredentialState,
        model_or_group: str,
    ) -> None:
        """
        Reset fair cycle state for a credential.

        Args:
            state: Credential state
            model_or_group: Scope to reset
        """
        async with self._lock:
            if model_or_group in state.fair_cycle:
                fc_state = state.fair_cycle[model_or_group]
                fc_state.exhausted = False
                fc_state.exhausted_at = None
                fc_state.exhausted_reason = None
                fc_state.cycle_request_count = 0

    def get_window_usage(
        self,
        state: CredentialState,
        window_name: str,
    ) -> int:
        """
        Get request count for a specific window.

        Args:
            state: Credential state
            window_name: Name of window

        Returns:
            Request count (0 if window doesn't exist)
        """
        window = self._windows.get_active_window(state.usage.windows, window_name)
        return window.request_count if window else 0

    def get_primary_window_usage(self, state: CredentialState) -> int:
        """
        Get request count for the primary window.

        Args:
            state: Credential state

        Returns:
            Request count (0 if no primary window)
        """
        window = self._windows.get_primary_window(state.usage.windows)
        return window.request_count if window else 0

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _apply_cooldown(
        self,
        state: CredentialState,
        reason: str,
        duration: Optional[float] = None,
        until: Optional[float] = None,
        model_or_group: Optional[str] = None,
        source: str = "system",
    ) -> None:
        """Internal cooldown application (no lock)."""
        now = time.time()

        if until is not None:
            cooldown_until = until
        elif duration is not None:
            cooldown_until = now + duration
        else:
            return  # No cooldown specified

        key = model_or_group or "_global_"

        # Check for existing cooldown
        existing = state.cooldowns.get(key)
        backoff_count = 0
        if existing and existing.is_active:
            backoff_count = existing.backoff_count + 1

        state.cooldowns[key] = CooldownInfo(
            reason=reason,
            until=cooldown_until,
            started_at=now,
            source=source,
            model_or_group=model_or_group,
            backoff_count=backoff_count,
        )

        # Check if cooldown qualifies as exhaustion
        cooldown_duration = cooldown_until - now
        if cooldown_duration >= self._config.exhaustion_cooldown_threshold:
            if self._config.fair_cycle.enabled and model_or_group:
                fair_cycle_key = self._resolve_fair_cycle_key(model_or_group)
                self._mark_exhausted(state, fair_cycle_key, f"cooldown_{reason}")

    def _mark_exhausted(
        self,
        state: CredentialState,
        model_or_group: str,
        reason: str,
    ) -> None:
        """Internal exhaustion marking (no lock)."""
        now = time.time()

        if model_or_group not in state.fair_cycle:
            state.fair_cycle[model_or_group] = FairCycleState(
                model_or_group=model_or_group
            )

        fc_state = state.fair_cycle[model_or_group]
        fc_state.exhausted = True
        fc_state.exhausted_at = now
        fc_state.exhausted_reason = reason

        lib_logger.debug(
            f"Credential {state.stable_id} marked exhausted for {model_or_group}: {reason}"
        )

    def _resolve_fair_cycle_key(self, group_key: str) -> str:
        """Resolve fair cycle tracking key based on config."""
        if self._config.fair_cycle.tracking_mode == TrackingMode.CREDENTIAL:
            return FAIR_CYCLE_GLOBAL_KEY
        return group_key

    def _get_usage_for_window(
        self,
        state: CredentialState,
        window_def: WindowDefinition,
        model: str,
        group_key: str,
    ) -> Optional[UsageStats]:
        """Get usage stats for a window definition's scope."""
        scope_key = None
        if window_def.applies_to == "model":
            scope_key = model
        elif window_def.applies_to == "group":
            scope_key = group_key
        return state.get_usage_for_scope(window_def.applies_to, scope_key)

    def _update_scoped_usage(
        self,
        state: CredentialState,
        scope: str,
        key: Optional[str],
        now: float,
        request_count: int,
        success_count: int,
        failure_count: int,
        prompt_tokens: int,
        completion_tokens: int,
        prompt_tokens_cached: int,
        approx_cost: float,
    ) -> None:
        """Update scoped usage stats."""
        usage = state.get_usage_for_scope(scope, key)
        if not usage:
            return
        usage.total_requests += request_count
        usage.total_successes += success_count
        usage.total_failures += failure_count
        usage.total_tokens += prompt_tokens + completion_tokens + prompt_tokens_cached
        usage.total_prompt_tokens_cached += prompt_tokens_cached
        usage.total_approx_cost += approx_cost
        usage.last_used_at = now
        if usage.first_used_at is None:
            usage.first_used_at = now

    def _update_from_headers(
        self,
        state: CredentialState,
        headers: Dict[str, Any],
        model: str,
        group_key: str,
    ) -> None:
        """Update state from API response headers."""
        # Common header patterns for rate limiting
        # X-RateLimit-Remaining, X-RateLimit-Reset, etc.
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        limit = headers.get("x-ratelimit-limit")

        # Update primary window if we have limit info
        primary_def = self._windows.get_primary_definition()
        if primary_def is None:
            return

        scope_key = None
        if primary_def.applies_to == "model":
            scope_key = model
        elif primary_def.applies_to == "group":
            scope_key = group_key

        usage = state.get_usage_for_scope(
            primary_def.applies_to, scope_key, create=False
        )
        if usage is None:
            return

        if limit is not None:
            try:
                limit_int = int(limit)
                primary = self._windows.get_primary_window(usage.windows)
                if primary:
                    primary.limit = limit_int
            except (ValueError, TypeError):
                pass

        if reset is not None:
            try:
                reset_float = float(reset)
                # If reset is in the past, it might be a Unix timestamp
                # If it's a small number, it might be seconds until reset
                if reset_float < 1000000000:  # Less than ~2001, probably relative
                    reset_float = time.time() + reset_float
                primary = self._windows.get_primary_window(usage.windows)
                if primary:
                    primary.reset_at = reset_float
            except (ValueError, TypeError):
                pass
