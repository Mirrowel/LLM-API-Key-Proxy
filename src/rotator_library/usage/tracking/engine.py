# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Tracking engine for usage recording.

Central component for recording requests, successes, and failures.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from ..config import WindowDefinition

from ..types import (
    WindowStats,
    TotalStats,
    ModelStats,
    GroupStats,
    CredentialState,
    CooldownInfo,
    FairCycleState,
    TrackingMode,
    UsageUpdate,
    FAIR_CYCLE_GLOBAL_KEY,
)
from ..config import WindowDefinition, ProviderUsageConfig
from .windows import WindowManager
from ...error_handler import mask_credential
from ...error_handler import mask_credential

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

    async def record_usage(
        self,
        state: CredentialState,
        model: str,
        update: UsageUpdate,
        group: Optional[str] = None,
        response_headers: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record usage for a request (consolidated function).

        Updates:
        - model_usage[model].windows[*] + totals
        - group_usage[group].windows[*] + totals (if group provided)
        - credential.totals

        Args:
            state: Credential state to update
            model: Model that was used
            update: UsageUpdate with all metrics
            group: Quota group for this model (None = no group tracking)
            response_headers: Optional response headers with rate limit info
        """
        async with self._lock:
            now = time.time()
            fair_cycle_key = self._resolve_fair_cycle_key(group or model)

            # Calculate derived values
            output_tokens = update.completion_tokens + update.thinking_tokens
            total_tokens = (
                update.prompt_tokens
                + update.completion_tokens
                + update.thinking_tokens
                + update.prompt_tokens_cache_read
                + update.prompt_tokens_cache_write
            )

            # 1. Update model stats
            model_stats = state.get_model_stats(model)
            self._apply_to_windows(
                model_stats.windows,
                update,
                now,
                total_tokens,
                output_tokens,
                window_definitions=state.window_definitions or None,
            )
            self._apply_to_totals(
                model_stats.totals, update, now, total_tokens, output_tokens
            )

            # 2. Update group stats (if applicable)
            if group:
                group_stats = state.get_group_stats(group)
                self._apply_to_windows(
                    group_stats.windows,
                    update,
                    now,
                    total_tokens,
                    output_tokens,
                    window_definitions=state.window_definitions or None,
                )
                self._apply_to_totals(
                    group_stats.totals, update, now, total_tokens, output_tokens
                )

                # Sync model window timing from group (group is authoritative)
                # All models in a quota group share the same started_at/reset_at
                self._sync_window_timing_from_group(
                    model_stats.windows, group_stats.windows
                )

            # 3. Update credential totals
            self._apply_to_totals(
                state.totals, update, now, total_tokens, output_tokens
            )

            # 4. Update fair cycle request count
            if self._config.fair_cycle.enabled:
                fc_state = state.fair_cycle.get(fair_cycle_key)
                if not fc_state:
                    fc_state = FairCycleState(model_or_group=fair_cycle_key)
                    state.fair_cycle[fair_cycle_key] = fc_state
                fc_state.cycle_request_count += update.request_count

            # 5. Update from response headers if provided
            if response_headers:
                self._update_from_headers(state, response_headers, model, group)

            state.last_updated = now

    async def record_success(
        self,
        state: CredentialState,
        model: str,
        quota_group: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        prompt_tokens_cache_read: int = 0,
        prompt_tokens_cache_write: int = 0,
        thinking_tokens: int = 0,
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
            prompt_tokens_cache_read: Cached prompt tokens read
            prompt_tokens_cache_write: Cached prompt tokens written
            thinking_tokens: Thinking tokens used
            approx_cost: Approximate cost
            request_count: Number of requests to record
            response_headers: Optional response headers with rate limit info
        """
        update = UsageUpdate(
            request_count=request_count,
            success=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=thinking_tokens,
            prompt_tokens_cache_read=prompt_tokens_cache_read,
            prompt_tokens_cache_write=prompt_tokens_cache_write,
            approx_cost=approx_cost,
        )
        await self.record_usage(
            state=state,
            model=model,
            update=update,
            group=quota_group,
            response_headers=response_headers,
        )

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
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        thinking_tokens: int = 0,
        prompt_tokens_cache_read: int = 0,
        prompt_tokens_cache_write: int = 0,
        approx_cost: float = 0.0,
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
            request_count: Number of requests to record
            prompt_tokens: Prompt tokens used
            completion_tokens: Completion tokens used
            thinking_tokens: Thinking tokens used
            prompt_tokens_cache_read: Cached prompt tokens read
            prompt_tokens_cache_write: Cached prompt tokens written
            approx_cost: Approximate cost
        """
        update = UsageUpdate(
            request_count=request_count,
            success=False,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=thinking_tokens,
            prompt_tokens_cache_read=prompt_tokens_cache_read,
            prompt_tokens_cache_write=prompt_tokens_cache_write,
            approx_cost=approx_cost,
        )

        # Record the usage
        await self.record_usage(
            state=state,
            model=model,
            update=update,
            group=quota_group,
        )

        async with self._lock:
            group_key = quota_group or model
            fair_cycle_key = self._resolve_fair_cycle_key(group_key)

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
        model: Optional[str] = None,
        group: Optional[str] = None,
    ) -> int:
        """
        Get request count for a specific window.

        Args:
            state: Credential state
            window_name: Name of window
            model: Model to check (optional)
            group: Group to check (optional)

        Returns:
            Request count (0 if window doesn't exist)
        """
        # Check group first if provided
        if group:
            group_stats = state.group_usage.get(group)
            if group_stats:
                window = self._windows.get_active_window(
                    group_stats.windows, window_name
                )
                if window:
                    return window.request_count

        # Check model if provided
        if model:
            model_stats = state.model_usage.get(model)
            if model_stats:
                window = self._windows.get_active_window(
                    model_stats.windows, window_name
                )
                if window:
                    return window.request_count

        return 0

    def get_primary_window_usage(
        self,
        state: CredentialState,
        model: Optional[str] = None,
        group: Optional[str] = None,
    ) -> int:
        """
        Get request count for the primary window.

        Args:
            state: Credential state
            model: Model to check (optional)
            group: Group to check (optional)

        Returns:
            Request count (0 if no primary window)
        """
        primary_def = self._windows.get_primary_definition()
        if primary_def is None:
            return 0
        return self.get_window_usage(state, primary_def.name, model, group)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _apply_to_windows(
        self,
        windows: Dict[str, WindowStats],
        update: UsageUpdate,
        now: float,
        total_tokens: int,
        output_tokens: int,
        window_definitions: Optional[List["WindowDefinition"]] = None,
    ) -> None:
        """Apply update to all configured windows."""
        # Use credential's window definitions if provided, otherwise fall back to config
        defs = window_definitions if window_definitions else self._config.windows
        for window_def in defs:
            window = self._windows.get_or_create_window(windows, window_def.name)
            self._apply_to_window(
                window, update, now, total_tokens, output_tokens, window_def
            )

    def _apply_to_window(
        self,
        window: WindowStats,
        update: UsageUpdate,
        now: float,
        total_tokens: int,
        output_tokens: int,
        window_def: Optional["WindowDefinition"] = None,
    ) -> None:
        """Apply update to a single window."""
        window.request_count += update.request_count
        if update.success:
            window.success_count += update.request_count
        else:
            window.failure_count += update.request_count

        window.prompt_tokens += update.prompt_tokens
        window.completion_tokens += update.completion_tokens
        window.thinking_tokens += update.thinking_tokens
        window.output_tokens += output_tokens
        window.prompt_tokens_cache_read += update.prompt_tokens_cache_read
        window.prompt_tokens_cache_write += update.prompt_tokens_cache_write
        window.total_tokens += total_tokens
        window.approx_cost += update.approx_cost

        window.last_used_at = now
        if window.first_used_at is None:
            window.first_used_at = now

        # Set started_at on first usage and calculate reset_at
        if window.started_at is None:
            window.started_at = now
            # Calculate reset_at based on window definition
            # Use passed window_def first, then fall back to shared definitions
            effective_def = window_def or self._windows.definitions.get(window.name)
            if effective_def and window.reset_at is None:
                window.reset_at = self._windows._calculate_reset_time(
                    effective_def, now
                )

        # Update max recorded requests (historical high-water mark)
        if (
            window.max_recorded_requests is None
            or window.request_count > window.max_recorded_requests
        ):
            window.max_recorded_requests = window.request_count
            window.max_recorded_at = now

    def _apply_to_totals(
        self,
        totals: TotalStats,
        update: UsageUpdate,
        now: float,
        total_tokens: int,
        output_tokens: int,
    ) -> None:
        """Apply update to totals."""
        totals.request_count += update.request_count
        if update.success:
            totals.success_count += update.request_count
        else:
            totals.failure_count += update.request_count

        totals.prompt_tokens += update.prompt_tokens
        totals.completion_tokens += update.completion_tokens
        totals.thinking_tokens += update.thinking_tokens
        totals.output_tokens += output_tokens
        totals.prompt_tokens_cache_read += update.prompt_tokens_cache_read
        totals.prompt_tokens_cache_write += update.prompt_tokens_cache_write
        totals.total_tokens += total_tokens
        totals.approx_cost += update.approx_cost

        totals.last_used_at = now
        if totals.first_used_at is None:
            totals.first_used_at = now

    def _sync_window_timing_from_group(
        self,
        model_windows: Dict[str, WindowStats],
        group_windows: Dict[str, WindowStats],
    ) -> None:
        """
        Sync timing fields from group windows to model windows.

        Group window is authoritative for started_at and reset_at.
        All models in a quota group share the same timing to ensure
        consistent quota tracking and window resets.

        Args:
            model_windows: The model's windows dict to update
            group_windows: The group's windows dict (authoritative)
        """
        for window_name, group_window in group_windows.items():
            model_window = model_windows.get(window_name)
            if model_window:
                model_window.started_at = group_window.started_at
                model_window.reset_at = group_window.reset_at

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
            # Preserve original reason/source/started_at - cooldown reason should
            # reflect why it was originally set, not subsequent updates
            # Time (until) is updated to the new value as API is authoritative
            backoff_count = existing.backoff_count + 1
            reason = existing.reason
            source = existing.source
            started_at = existing.started_at
        else:
            started_at = now

        state.cooldowns[key] = CooldownInfo(
            reason=reason,
            until=cooldown_until,
            started_at=started_at,
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

        # Idempotency check: skip if already exhausted (avoid duplicate logging)
        if fc_state.exhausted:
            return

        fc_state.exhausted = True
        fc_state.exhausted_at = now
        fc_state.exhausted_reason = reason

        lib_logger.info(
            f"Credential {mask_credential(state.accessor, style='full')} marked fair-cycle exhausted "
            f"for {model_or_group}: {reason}"
        )

    def _resolve_fair_cycle_key(self, group_key: str) -> str:
        """Resolve fair cycle tracking key based on config."""
        if self._config.fair_cycle.tracking_mode == TrackingMode.CREDENTIAL:
            return FAIR_CYCLE_GLOBAL_KEY
        return group_key

    def _update_from_headers(
        self,
        state: CredentialState,
        headers: Dict[str, Any],
        model: str,
        group: Optional[str],
    ) -> None:
        """Update state from API response headers."""
        # Common header patterns for rate limiting
        # X-RateLimit-Remaining, X-RateLimit-Reset, etc.
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        limit = headers.get("x-ratelimit-limit")

        primary_def = self._windows.get_primary_definition()
        if primary_def is None:
            return

        # Update group windows if group is provided
        if group:
            group_stats = state.get_group_stats(group, create=False)
            if group_stats:
                window = group_stats.windows.get(primary_def.name)
                if window:
                    self._apply_header_updates(window, limit, reset)

        # Update model windows
        model_stats = state.get_model_stats(model, create=False)
        if model_stats:
            window = model_stats.windows.get(primary_def.name)
            if window:
                self._apply_header_updates(window, limit, reset)

    def _apply_header_updates(
        self,
        window: WindowStats,
        limit: Optional[Any],
        reset: Optional[Any],
    ) -> None:
        """Apply header updates to a window."""
        if limit is not None:
            try:
                window.limit = int(limit)
            except (ValueError, TypeError):
                pass

        if reset is not None:
            try:
                reset_float = float(reset)
                # If reset is in the past, it might be a Unix timestamp
                # If it's a small number, it might be seconds until reset
                if reset_float < 1000000000:  # Less than ~2001, probably relative
                    reset_float = time.time() + reset_float
                window.reset_at = reset_float
            except (ValueError, TypeError):
                pass
