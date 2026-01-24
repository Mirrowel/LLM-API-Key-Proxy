# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
UsageManager facade and CredentialContext.

This is the main public API for the usage tracking system.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

from ..core.types import CredentialInfo, RequestCompleteResult
from ..error_handler import ClassifiedError, classify_error, mask_credential

from .types import (
    WindowStats,
    TotalStats,
    ModelStats,
    GroupStats,
    CredentialState,
    LimitCheckResult,
    RotationMode,
    LimitResult,
    FAIR_CYCLE_GLOBAL_KEY,
    TrackingMode,
    ResetMode,
)
from .config import (
    ProviderUsageConfig,
    load_provider_usage_config,
    get_default_windows,
    CapMode,
)
from .identity.registry import CredentialRegistry
from .tracking.engine import TrackingEngine
from .tracking.windows import WindowManager
from .limits.engine import LimitEngine
from .selection.engine import SelectionEngine
from .persistence.storage import UsageStorage
from .integration.hooks import HookDispatcher
from .integration.api import UsageAPI

lib_logger = logging.getLogger("rotator_library")


class CredentialContext:
    """
    Context manager for credential lifecycle.

    Handles:
    - Automatic release on exit
    - Success/failure recording
    - Usage tracking

    Usage:
        async with usage_manager.acquire_credential(provider, model) as ctx:
            response = await make_request(ctx.credential)
            ctx.mark_success(response)
    """

    def __init__(
        self,
        manager: "UsageManager",
        credential: str,
        stable_id: str,
        model: str,
        quota_group: Optional[str] = None,
    ):
        self._manager = manager
        self.credential = credential  # The accessor (path or key)
        self.stable_id = stable_id
        self.model = model
        self.quota_group = quota_group
        self._acquired_at = time.time()
        self._result: Optional[Literal["success", "failure"]] = None
        self._response: Optional[Any] = None
        self._response_headers: Optional[Dict[str, Any]] = None
        self._error: Optional[ClassifiedError] = None
        self._tokens: Dict[str, int] = {}
        self._approx_cost: float = 0.0

    async def __aenter__(self) -> "CredentialContext":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Always release the credential
        await self._manager._release_credential(self.stable_id, self.model)

        success = False
        error = self._error
        response = self._response

        if self._result == "success":
            success = True
        elif self._result == "failure":
            success = False
        elif exc_val is not None:
            error = classify_error(exc_val)
            success = False
        else:
            success = True

        await self._manager._handle_request_complete(
            stable_id=self.stable_id,
            model=self.model,
            quota_group=self.quota_group,
            success=success,
            response=response,
            response_headers=self._response_headers,
            error=error,
            prompt_tokens=self._tokens.get("prompt", 0),
            completion_tokens=self._tokens.get("completion", 0),
            thinking_tokens=self._tokens.get("thinking", 0),
            prompt_tokens_cache_read=self._tokens.get("prompt_cached", 0),
            prompt_tokens_cache_write=self._tokens.get("prompt_cache_write", 0),
            approx_cost=self._approx_cost,
        )

        return False  # Don't suppress exceptions

    def mark_success(
        self,
        response: Any = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        thinking_tokens: int = 0,
        prompt_tokens_cache_read: int = 0,
        prompt_tokens_cache_write: int = 0,
        approx_cost: float = 0.0,
        response_headers: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark request as successful."""
        self._result = "success"
        self._response = response
        self._response_headers = response_headers
        self._tokens = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "thinking": thinking_tokens,
            "prompt_cached": prompt_tokens_cache_read,
            "prompt_cache_write": prompt_tokens_cache_write,
        }
        self._approx_cost = approx_cost

    def mark_failure(self, error: ClassifiedError) -> None:
        """Mark request as failed."""
        self._result = "failure"
        self._error = error


class UsageManager:
    """
    Main facade for usage tracking and credential selection.

    This class provides the primary interface for:
    - Acquiring credentials for requests (with context manager)
    - Recording usage and failures
    - Selecting the best available credential
    - Managing cooldowns and limits

    Example:
        manager = UsageManager(provider="gemini", file_path="usage.json")
        await manager.initialize(credentials)

        async with manager.acquire_credential(model="gemini-pro") as ctx:
            response = await make_request(ctx.credential)
            ctx.mark_success(response, prompt_tokens=100, completion_tokens=50)
    """

    def __init__(
        self,
        provider: str,
        file_path: Optional[Union[str, Path]] = None,
        provider_plugins: Optional[Dict[str, Any]] = None,
        config: Optional[ProviderUsageConfig] = None,
        max_concurrent_per_key: Optional[int] = None,
    ):
        """
        Initialize UsageManager.

        Args:
            provider: Provider name (e.g., "gemini", "openai")
            file_path: Path to usage.json file
            provider_plugins: Dict of provider plugin classes
            config: Optional pre-built configuration
            max_concurrent_per_key: Max concurrent requests per credential
        """
        self.provider = provider
        self._provider_plugins = provider_plugins or {}
        self._max_concurrent_per_key = max_concurrent_per_key

        # Load configuration
        if config:
            self._config = config
        else:
            self._config = load_provider_usage_config(provider, self._provider_plugins)

        # Initialize components
        self._registry = CredentialRegistry()
        self._window_manager = WindowManager(
            window_definitions=self._config.windows or get_default_windows()
        )
        self._tracking = TrackingEngine(self._window_manager, self._config)
        self._limits = LimitEngine(self._config, self._window_manager)
        self._selection = SelectionEngine(
            self._config, self._limits, self._window_manager
        )
        self._hooks = HookDispatcher(self._provider_plugins)
        self._api = UsageAPI(self)

        # Storage
        if file_path:
            self._storage = UsageStorage(file_path)
        else:
            self._storage = None

        # State
        self._states: Dict[str, CredentialState] = {}
        self._initialized = False
        self._lock = asyncio.Lock()
        self._loaded_from_storage = False
        self._loaded_count = 0
        self._quota_exhausted_summary: Dict[str, Dict[str, float]] = {}
        self._quota_exhausted_task: Optional[asyncio.Task] = None
        self._quota_exhausted_lock = asyncio.Lock()
        self._save_task: Optional[asyncio.Task] = None
        self._save_lock = asyncio.Lock()

        # Concurrency control: per-credential locks and conditions for waiting
        self._key_locks: Dict[str, asyncio.Lock] = {}
        self._key_conditions: Dict[str, asyncio.Condition] = {}

        # Track which credentials are currently active in the proxy session
        # (vs. historical data loaded from storage)
        self._active_stable_ids: Set[str] = set()

    async def initialize(
        self,
        credentials: List[str],
        priorities: Optional[Dict[str, int]] = None,
        tiers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize with credentials.

        Args:
            credentials: List of credential accessors (paths or keys)
            priorities: Optional priority overrides (accessor -> priority)
            tiers: Optional tier overrides (accessor -> tier name)
        """
        async with self._lock:
            if self._initialized:
                return
            # Load persisted state
            if self._storage:
                (
                    self._states,
                    fair_cycle_global,
                    loaded_from_storage,
                ) = await self._storage.load()
                self._loaded_from_storage = loaded_from_storage
                self._loaded_count = len(self._states)
                if fair_cycle_global:
                    self._limits.fair_cycle_checker.load_global_state_dict(
                        fair_cycle_global
                    )

            # Register credentials and track active ones
            self._active_stable_ids.clear()
            for accessor in credentials:
                stable_id = self._registry.get_stable_id(accessor, self.provider)
                self._active_stable_ids.add(stable_id)

                # Create or update state
                if stable_id not in self._states:
                    self._states[stable_id] = CredentialState(
                        stable_id=stable_id,
                        provider=self.provider,
                        accessor=accessor,
                        created_at=time.time(),
                    )
                else:
                    # Update accessor in case it changed
                    self._states[stable_id].accessor = accessor

                # Apply overrides
                if priorities and accessor in priorities:
                    self._states[stable_id].priority = priorities[accessor]
                if tiers and accessor in tiers:
                    self._states[stable_id].tier = tiers[accessor]

                # Debug: Log state before max_concurrent calculation
                old_max_concurrent = self._states[stable_id].max_concurrent

                # Always set max concurrent, applying priority multiplier
                # Uses configured value or defaults to 1 if not set
                base_concurrent = (
                    self._max_concurrent_per_key
                    if self._max_concurrent_per_key is not None
                    else 1
                )
                priority = self._states[stable_id].priority
                multiplier = self._config.get_effective_multiplier(priority)
                effective_concurrent = base_concurrent * multiplier
                self._states[stable_id].max_concurrent = effective_concurrent

            # Clean up stale windows from tier changes
            # This handles the case where a credential's tier changed and now has
            # windows from the old tier that should be removed
            # Also populate window_definitions for each credential based on tier
            total_removed = 0
            for stable_id, state in self._states.items():
                # Populate window definitions for this credential's tier
                state.window_definitions = self._get_window_definitions_for_state(state)

                valid_windows = self._get_valid_window_names_for_state(state)
                removed = self._cleanup_stale_windows_for_state(state, valid_windows)
                total_removed += removed

            if total_removed > 0:
                lib_logger.info(
                    f"Cleaned up {total_removed} stale window(s) for {self.provider}"
                )
                # Mark storage dirty so changes get saved
                if self._storage:
                    self._storage.mark_dirty()

            self._initialized = True
            lib_logger.debug(
                f"UsageManager initialized for {self.provider} with {len(credentials)} credentials"
            )

    async def acquire_credential(
        self,
        model: str,
        quota_group: Optional[str] = None,
        exclude: Optional[Set[str]] = None,
        candidates: Optional[List[str]] = None,
        priorities: Optional[Dict[str, int]] = None,
        deadline: float = 0.0,
    ) -> CredentialContext:
        """
        Acquire a credential for a request.

        Returns a context manager that automatically releases
        the credential and records success/failure.

        This method will wait for credentials to become available if all are
        currently busy (at max_concurrent), up until the deadline.

        Args:
            model: Model to use
            quota_group: Optional quota group (uses model name if None)
            exclude: Set of stable_ids to exclude (by accessor)
            candidates: Optional list of credential accessors to consider.
                       If provided, only these will be considered for selection.
            priorities: Optional priority overrides (accessor -> priority).
                       If provided, overrides the stored priorities.
            deadline: Request deadline timestamp

        Returns:
            CredentialContext for use with async with

        Raises:
            NoAvailableKeysError: If no credentials available within deadline
        """
        from ..error_handler import NoAvailableKeysError

        # Convert accessor-based exclude to stable_id-based
        exclude_ids = set()
        if exclude:
            for accessor in exclude:
                stable_id = self._registry.get_stable_id(accessor, self.provider)
                exclude_ids.add(stable_id)

        # Filter states to only candidates if provided
        if candidates is not None:
            candidate_ids = set()
            for accessor in candidates:
                stable_id = self._registry.get_stable_id(accessor, self.provider)
                candidate_ids.add(stable_id)
            states_to_check = {
                sid: state
                for sid, state in self._states.items()
                if sid in candidate_ids
            }
        else:
            states_to_check = self._states

        # Convert accessor-based priorities to stable_id-based
        priority_overrides = None
        if priorities:
            priority_overrides = {}
            for accessor, priority in priorities.items():
                stable_id = self._registry.get_stable_id(accessor, self.provider)
                priority_overrides[stable_id] = priority

        # Normalize model name for consistent tracking and selection
        normalized_model = self._normalize_model(model)

        # Ensure key conditions exist for all candidates
        for stable_id in states_to_check:
            if stable_id not in self._key_conditions:
                self._key_conditions[stable_id] = asyncio.Condition()
                self._key_locks[stable_id] = asyncio.Lock()

        # Main acquisition loop - continues until deadline
        while time.time() < deadline:
            # Try to select a credential
            stable_id = self._selection.select(
                provider=self.provider,
                model=normalized_model,
                states=states_to_check,
                quota_group=quota_group,
                exclude=exclude_ids,
                priorities=priority_overrides,
                deadline=deadline,
            )

            if stable_id is not None:
                state = self._states[stable_id]
                lock = self._key_locks.get(stable_id)

                if lock:
                    async with lock:
                        # Double-check availability after acquiring lock
                        if (
                            state.max_concurrent is None
                            or state.active_requests < state.max_concurrent
                        ):
                            state.active_requests += 1
                            lib_logger.debug(
                                f"Acquired credential {mask_credential(state.accessor, style='full')} "
                                f"for {model} (active: {state.active_requests}"
                                f"{f'/{state.max_concurrent}' if state.max_concurrent else ''})"
                            )
                            return CredentialContext(
                                manager=self,
                                credential=state.accessor,
                                stable_id=stable_id,
                                model=normalized_model,
                                quota_group=quota_group,
                            )
                else:
                    # No lock configured, just increment
                    state.active_requests += 1
                    return CredentialContext(
                        manager=self,
                        credential=state.accessor,
                        stable_id=stable_id,
                        model=normalized_model,
                        quota_group=quota_group,
                    )

            # No credential available - need to wait
            # Find the best credential to wait for (prefer lowest usage)
            best_wait_id = None
            best_usage = float("inf")

            for sid, state in states_to_check.items():
                if sid in exclude_ids:
                    continue
                if (
                    state.max_concurrent is not None
                    and state.active_requests >= state.max_concurrent
                ):
                    # This one is busy but might become free
                    usage = state.totals.request_count
                    if usage < best_usage:
                        best_usage = usage
                        best_wait_id = sid

            if best_wait_id is None:
                # All credentials blocked by cooldown or limits, not just concurrency
                # Check if waiting for cooldown makes sense
                soonest_cooldown = self._get_soonest_cooldown_end(
                    states_to_check, normalized_model, quota_group
                )

                if soonest_cooldown is not None:
                    remaining_budget = deadline - time.time()
                    wait_needed = soonest_cooldown - time.time()

                    if wait_needed > remaining_budget:
                        # No credential will be available in time
                        lib_logger.warning(
                            f"All credentials on cooldown. Soonest in {wait_needed:.1f}s, "
                            f"budget {remaining_budget:.1f}s. Failing fast."
                        )
                        break

                    # Wait for cooldown to expire
                    lib_logger.info(
                        f"All credentials on cooldown. Waiting {wait_needed:.1f}s..."
                    )
                    await asyncio.sleep(min(wait_needed + 0.1, remaining_budget))
                    continue

                # No cooldowns and no busy keys - truly no keys available
                break

            # Wait on the best credential's condition
            condition = self._key_conditions.get(best_wait_id)
            if condition:
                lib_logger.debug(
                    f"All credentials busy. Waiting for {mask_credential(self._states[best_wait_id].accessor, style='full')}..."
                )
                try:
                    async with condition:
                        remaining_budget = deadline - time.time()
                        if remaining_budget <= 0:
                            break
                        # Wait for notification or timeout (max 1 second to re-check)
                        await asyncio.wait_for(
                            condition.wait(),
                            timeout=min(1.0, remaining_budget),
                        )
                    lib_logger.debug("Credential released. Re-evaluating...")
                except asyncio.TimeoutError:
                    # Timeout is normal, just retry the loop
                    lib_logger.debug("Wait timed out. Re-evaluating...")
            else:
                # No condition, just sleep briefly and retry
                await asyncio.sleep(0.1)

        # Deadline exceeded
        raise NoAvailableKeysError(
            f"Could not acquire a credential for {self.provider}/{model} "
            f"within the time budget."
        )

    def _get_soonest_cooldown_end(
        self,
        states: Dict[str, CredentialState],
        model: str,
        quota_group: Optional[str],
    ) -> Optional[float]:
        """Get the soonest cooldown end time across all credentials."""
        soonest = None
        now = time.time()
        group_key = quota_group or model

        for state in states.values():
            # Check model-specific cooldown
            cooldown = state.get_cooldown(group_key)
            if cooldown and cooldown.until > now:
                if soonest is None or cooldown.until < soonest:
                    soonest = cooldown.until

            # Check global cooldown
            global_cooldown = state.get_cooldown()
            if global_cooldown and global_cooldown.until > now:
                if soonest is None or global_cooldown.until < soonest:
                    soonest = global_cooldown.until

        return soonest

    async def get_best_credential(
        self,
        model: str,
        quota_group: Optional[str] = None,
        exclude: Optional[Set[str]] = None,
        deadline: float = 0.0,
    ) -> Optional[str]:
        """
        Get the best available credential without acquiring.

        Useful for checking availability or manual acquisition.

        Args:
            model: Model to use
            quota_group: Optional quota group
            exclude: Set of accessors to exclude
            deadline: Request deadline

        Returns:
            Credential accessor, or None if none available
        """
        # Convert exclude from accessors to stable_ids
        exclude_ids = set()
        if exclude:
            for accessor in exclude:
                stable_id = self._registry.get_stable_id(accessor, self.provider)
                exclude_ids.add(stable_id)

        # Normalize model name for consistent selection
        normalized_model = self._normalize_model(model)

        stable_id = self._selection.select(
            provider=self.provider,
            model=normalized_model,
            states=self._states,
            quota_group=quota_group,
            exclude=exclude_ids,
            deadline=deadline,
        )

        if stable_id is None:
            return None

        return self._states[stable_id].accessor

    async def record_usage(
        self,
        accessor: str,
        model: str,
        success: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        thinking_tokens: int = 0,
        prompt_tokens_cache_read: int = 0,
        prompt_tokens_cache_write: int = 0,
        approx_cost: float = 0.0,
        error: Optional[ClassifiedError] = None,
        quota_group: Optional[str] = None,
    ) -> None:
        """
        Record usage for a credential (manual recording).

        Use this for manual tracking outside of context manager.

        Args:
            accessor: Credential accessor
            model: Model used
            success: Whether request succeeded
            prompt_tokens: Prompt tokens used
            completion_tokens: Completion tokens used
            thinking_tokens: Thinking tokens used
            prompt_tokens_cache_read: Cached prompt tokens read
            prompt_tokens_cache_write: Cached prompt tokens written
            approx_cost: Approximate cost
            error: Classified error if failed
            quota_group: Quota group
        """
        stable_id = self._registry.get_stable_id(accessor, self.provider)

        if success:
            await self._record_success(
                stable_id,
                model,
                quota_group,
                prompt_tokens,
                completion_tokens,
                thinking_tokens,
                prompt_tokens_cache_read,
                prompt_tokens_cache_write,
                approx_cost,
            )
        else:
            await self._record_failure(
                stable_id,
                model,
                quota_group,
                error,
                request_count=1,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                thinking_tokens=thinking_tokens,
                prompt_tokens_cache_read=prompt_tokens_cache_read,
                prompt_tokens_cache_write=prompt_tokens_cache_write,
                approx_cost=approx_cost,
            )

    async def _handle_request_complete(
        self,
        stable_id: str,
        model: str,
        quota_group: Optional[str],
        success: bool,
        response: Optional[Any],
        response_headers: Optional[Dict[str, Any]],
        error: Optional[ClassifiedError],
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        thinking_tokens: int = 0,
        prompt_tokens_cache_read: int = 0,
        prompt_tokens_cache_write: int = 0,
        approx_cost: float = 0.0,
    ) -> None:
        """Handle provider hooks and record request outcome."""
        state = self._states.get(stable_id)
        if not state:
            return

        normalized_model = self._normalize_model(model)
        group_key = quota_group or self._get_model_quota_group(normalized_model)

        hook_result: Optional[RequestCompleteResult] = None
        if self._hooks:
            hook_result = await self._hooks.dispatch_request_complete(
                provider=self.provider,
                credential=state.accessor,
                model=normalized_model,
                success=success,
                response=response,
                error=error,
            )

        request_count = 1
        cooldown_override = None
        force_exhausted = False

        if hook_result:
            if hook_result.count_override is not None:
                request_count = max(0, hook_result.count_override)
            cooldown_override = hook_result.cooldown_override
            force_exhausted = hook_result.force_exhausted

        if not success and error and hook_result is None:
            if error.error_type in {"server_error", "api_connection"}:
                request_count = 0

        if request_count == 0:
            prompt_tokens = 0
            completion_tokens = 0
            thinking_tokens = 0
            prompt_tokens_cache_read = 0
            prompt_tokens_cache_write = 0
            approx_cost = 0.0

        if cooldown_override:
            await self._tracking.apply_cooldown(
                state=state,
                reason="provider_hook",
                duration=cooldown_override,
                model_or_group=group_key,
                source="provider_hook",
            )

        if force_exhausted:
            await self._tracking.mark_exhausted(
                state=state,
                model_or_group=self._resolve_fair_cycle_key(
                    group_key or normalized_model
                ),
                reason="provider_hook",
            )

        if success:
            await self._record_success(
                stable_id,
                normalized_model,
                quota_group,
                prompt_tokens,
                completion_tokens,
                thinking_tokens,
                prompt_tokens_cache_read,
                prompt_tokens_cache_write,
                approx_cost,
                response_headers=response_headers,
                request_count=request_count,
            )
        else:
            await self._record_failure(
                stable_id,
                normalized_model,
                quota_group,
                error,
                request_count=request_count,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                thinking_tokens=thinking_tokens,
                prompt_tokens_cache_read=prompt_tokens_cache_read,
                prompt_tokens_cache_write=prompt_tokens_cache_write,
                approx_cost=approx_cost,
            )

    async def apply_cooldown(
        self,
        accessor: str,
        duration: float,
        reason: str = "manual",
        model_or_group: Optional[str] = None,
    ) -> None:
        """
        Apply a cooldown to a credential.

        Args:
            accessor: Credential accessor
            duration: Cooldown duration in seconds
            reason: Reason for cooldown
            model_or_group: Scope of cooldown
        """
        stable_id = self._registry.get_stable_id(accessor, self.provider)
        state = self._states.get(stable_id)
        if state:
            await self._tracking.apply_cooldown(
                state=state,
                reason=reason,
                duration=duration,
                model_or_group=model_or_group,
            )
            await self._save_if_needed()

    async def get_availability_stats(
        self,
        model: str,
        quota_group: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get availability statistics for credentials.

        Args:
            model: Model to check
            quota_group: Quota group

        Returns:
            Dict with availability info
        """
        return self._selection.get_availability_stats(
            provider=self.provider,
            model=model,
            states=self._states,
            quota_group=quota_group,
        )

    async def get_stats_for_endpoint(
        self,
        model_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get comprehensive stats suitable for status endpoints.

        Returns credential states, usage windows, cooldowns, and fair cycle state.

        Args:
            model_filter: Optional model to filter stats for

        Returns:
            Dict with comprehensive statistics
        """
        stats = {
            "provider": self.provider,
            "credential_count": len(self._active_stable_ids),
            "rotation_mode": self._config.rotation_mode.value,
            "credentials": {},
        }

        stats.update(
            {
                "active_count": 0,
                "exhausted_count": 0,
                "total_requests": 0,
                "tokens": {
                    "input_cached": 0,
                    "input_uncached": 0,
                    "input_cache_pct": 0,
                    "output": 0,
                },
                "approx_cost": None,
                "quota_groups": {},
            }
        )

        for stable_id, state in self._states.items():
            # Skip credentials not currently active in the proxy
            if stable_id not in self._active_stable_ids:
                continue

            now = time.time()

            # Determine credential status with proper granularity
            status = "active"
            has_global_cooldown = False
            has_group_cooldown = False
            fc_exhausted_groups = []

            # Check cooldowns (global vs per-group)
            for key, cooldown in state.cooldowns.items():
                if cooldown.until > now:
                    if key == "_global_":
                        has_global_cooldown = True
                    else:
                        has_group_cooldown = True

            # Check fair cycle per group
            for group_key, fc_state in state.fair_cycle.items():
                if fc_state.exhausted:
                    fc_exhausted_groups.append(group_key)

            # Determine final status
            known_groups = set(state.group_usage.keys()) if state.group_usage else set()

            if has_global_cooldown:
                status = "cooldown"
            elif fc_exhausted_groups:
                # Check if ALL known groups are exhausted
                if known_groups and set(fc_exhausted_groups) >= known_groups:
                    status = "exhausted"
                else:
                    status = "mixed"  # Some groups available
            elif has_group_cooldown:
                status = "cooldown"

            cred_stats = {
                "stable_id": stable_id,
                "accessor_masked": mask_credential(state.accessor, style="full"),
                "full_path": state.accessor,
                "identifier": mask_credential(state.accessor, style="full"),
                "email": state.display_name,
                "tier": state.tier,
                "priority": state.priority,
                "active_requests": state.active_requests,
                "status": status,
                "totals": {
                    "request_count": state.totals.request_count,
                    "success_count": state.totals.success_count,
                    "failure_count": state.totals.failure_count,
                    "prompt_tokens": state.totals.prompt_tokens,
                    "completion_tokens": state.totals.completion_tokens,
                    "thinking_tokens": state.totals.thinking_tokens,
                    "output_tokens": state.totals.output_tokens,
                    "prompt_tokens_cache_read": state.totals.prompt_tokens_cache_read,
                    "prompt_tokens_cache_write": state.totals.prompt_tokens_cache_write,
                    "total_tokens": state.totals.total_tokens,
                    "approx_cost": state.totals.approx_cost,
                    "first_used_at": state.totals.first_used_at,
                    "last_used_at": state.totals.last_used_at,
                },
                "model_usage": {},
                "group_usage": {},
                "cooldowns": {},
                "fair_cycle": {},
            }

            stats["total_requests"] += state.totals.request_count
            stats["tokens"]["output"] += state.totals.output_tokens
            stats["tokens"]["input_cached"] += state.totals.prompt_tokens_cache_read
            # prompt_tokens in LiteLLM = uncached tokens (not total input)
            # Total input = prompt_tokens + prompt_tokens_cache_read
            stats["tokens"]["input_uncached"] += state.totals.prompt_tokens
            if state.totals.approx_cost:
                stats["approx_cost"] = (
                    stats["approx_cost"] or 0.0
                ) + state.totals.approx_cost

            if status == "active":
                stats["active_count"] += 1
            elif status == "exhausted":
                stats["exhausted_count"] += 1

            # Add model usage stats
            for model_key, model_stats in state.model_usage.items():
                model_windows = {}
                for window_name, window in model_stats.windows.items():
                    model_windows[window_name] = {
                        "request_count": window.request_count,
                        "success_count": window.success_count,
                        "failure_count": window.failure_count,
                        "prompt_tokens": window.prompt_tokens,
                        "completion_tokens": window.completion_tokens,
                        "thinking_tokens": window.thinking_tokens,
                        "output_tokens": window.output_tokens,
                        "prompt_tokens_cache_read": window.prompt_tokens_cache_read,
                        "prompt_tokens_cache_write": window.prompt_tokens_cache_write,
                        "total_tokens": window.total_tokens,
                        "limit": window.limit,
                        "remaining": window.remaining,
                        "max_recorded_requests": window.max_recorded_requests,
                        "max_recorded_at": window.max_recorded_at,
                        "reset_at": window.reset_at,
                        "approx_cost": window.approx_cost,
                        "first_used_at": window.first_used_at,
                        "last_used_at": window.last_used_at,
                    }
                cred_stats["model_usage"][model_key] = {
                    "windows": model_windows,
                    "totals": {
                        "request_count": model_stats.totals.request_count,
                        "success_count": model_stats.totals.success_count,
                        "failure_count": model_stats.totals.failure_count,
                        "prompt_tokens": model_stats.totals.prompt_tokens,
                        "completion_tokens": model_stats.totals.completion_tokens,
                        "thinking_tokens": model_stats.totals.thinking_tokens,
                        "output_tokens": model_stats.totals.output_tokens,
                        "prompt_tokens_cache_read": model_stats.totals.prompt_tokens_cache_read,
                        "prompt_tokens_cache_write": model_stats.totals.prompt_tokens_cache_write,
                        "total_tokens": model_stats.totals.total_tokens,
                        "approx_cost": model_stats.totals.approx_cost,
                        "first_used_at": model_stats.totals.first_used_at,
                        "last_used_at": model_stats.totals.last_used_at,
                    },
                }

            # Add group usage stats
            for group_key, group_stats in state.group_usage.items():
                group_windows = {}
                for window_name, window in group_stats.windows.items():
                    group_windows[window_name] = {
                        "request_count": window.request_count,
                        "success_count": window.success_count,
                        "failure_count": window.failure_count,
                        "prompt_tokens": window.prompt_tokens,
                        "completion_tokens": window.completion_tokens,
                        "thinking_tokens": window.thinking_tokens,
                        "output_tokens": window.output_tokens,
                        "prompt_tokens_cache_read": window.prompt_tokens_cache_read,
                        "prompt_tokens_cache_write": window.prompt_tokens_cache_write,
                        "total_tokens": window.total_tokens,
                        "limit": window.limit,
                        "remaining": window.remaining,
                        "max_recorded_requests": window.max_recorded_requests,
                        "max_recorded_at": window.max_recorded_at,
                        "reset_at": window.reset_at,
                        "approx_cost": window.approx_cost,
                        "first_used_at": window.first_used_at,
                        "last_used_at": window.last_used_at,
                    }
                cred_stats["group_usage"][group_key] = {
                    "windows": group_windows,
                    "totals": {
                        "request_count": group_stats.totals.request_count,
                        "success_count": group_stats.totals.success_count,
                        "failure_count": group_stats.totals.failure_count,
                        "prompt_tokens": group_stats.totals.prompt_tokens,
                        "completion_tokens": group_stats.totals.completion_tokens,
                        "thinking_tokens": group_stats.totals.thinking_tokens,
                        "output_tokens": group_stats.totals.output_tokens,
                        "prompt_tokens_cache_read": group_stats.totals.prompt_tokens_cache_read,
                        "prompt_tokens_cache_write": group_stats.totals.prompt_tokens_cache_write,
                        "total_tokens": group_stats.totals.total_tokens,
                        "approx_cost": group_stats.totals.approx_cost,
                        "first_used_at": group_stats.totals.first_used_at,
                        "last_used_at": group_stats.totals.last_used_at,
                    },
                }

                # Add per-group status info for this credential
                group_data = cred_stats["group_usage"][group_key]

                # Fair cycle status for this group
                fc_state = state.fair_cycle.get(group_key)
                group_data["fair_cycle_exhausted"] = (
                    fc_state.exhausted if fc_state else False
                )
                group_data["fair_cycle_reason"] = (
                    fc_state.exhausted_reason
                    if fc_state and fc_state.exhausted
                    else None
                )

                # Group-specific cooldown
                group_cooldown = state.cooldowns.get(group_key)
                if group_cooldown and group_cooldown.is_active:
                    group_data["cooldown_remaining"] = int(
                        group_cooldown.remaining_seconds
                    )
                    group_data["cooldown_source"] = group_cooldown.source
                else:
                    group_data["cooldown_remaining"] = None
                    group_data["cooldown_source"] = None

                # Custom cap info for this group
                cap = self._limits.custom_cap_checker.get_cap_for(
                    state, group_key, group_key
                )
                if cap:
                    # Get usage from primary window
                    primary_window = group_windows.get(
                        self._window_manager.get_primary_definition().name
                        if self._window_manager.get_primary_definition()
                        else "5h"
                    )
                    cap_used = (
                        primary_window.get("request_count", 0) if primary_window else 0
                    )
                    cap_limit = cap.max_requests
                    # Resolve cap limit based on mode
                    api_limit = primary_window.get("limit") if primary_window else None
                    if cap.max_requests_mode == CapMode.OFFSET:
                        if api_limit:
                            cap_limit = max(0, api_limit + cap.max_requests)
                        else:
                            cap_limit = max(0, abs(cap.max_requests))
                    elif cap.max_requests_mode == CapMode.PERCENTAGE:
                        if api_limit:
                            cap_limit = max(0, int(api_limit * cap.max_requests / 100))
                        else:
                            cap_limit = 0
                    else:  # ABSOLUTE
                        cap_limit = max(0, cap.max_requests)
                    group_data["custom_cap"] = {
                        "limit": cap_limit,
                        "used": cap_used,
                        "remaining": max(0, cap_limit - cap_used),
                    }
                else:
                    group_data["custom_cap"] = None

                # Aggregate quota group stats with per-window breakdown
                group_agg = stats["quota_groups"].setdefault(
                    group_key,
                    {
                        "tiers": {},  # Credential tier counts (provider-level)
                        "windows": {},  # Per-window aggregated stats
                        "fair_cycle_summary": {  # FC status across all credentials
                            "exhausted_count": 0,
                            "total_count": 0,
                        },
                    },
                )

                # Update fair cycle summary for this group
                group_agg["fair_cycle_summary"]["total_count"] += 1
                if group_data.get("fair_cycle_exhausted"):
                    group_agg["fair_cycle_summary"]["exhausted_count"] += 1

                # Add credential to tier count (provider-level, not per-window)
                tier_key = state.tier or "unknown"
                tier_stats = group_agg["tiers"].setdefault(
                    tier_key,
                    {"priority": state.priority or 0, "total": 0},
                )
                tier_stats["total"] += 1

                # Aggregate per-window stats
                for window_name, window in group_windows.items():
                    window_agg = group_agg[
                        "windows"
                    ].setdefault(
                        window_name,
                        {
                            "total_used": 0,
                            "total_remaining": 0,
                            "total_max": 0,
                            "remaining_pct": None,
                            "tier_availability": {},  # Per-window credential availability
                        },
                    )

                    # Track tier availability for this window
                    tier_avail = window_agg["tier_availability"].setdefault(
                        tier_key,
                        {"total": 0, "available": 0},
                    )
                    tier_avail["total"] += 1

                    # Check if this credential has quota remaining in this window
                    limit = window.get("limit")
                    if limit is not None:
                        used = window["request_count"]
                        remaining = max(0, limit - used)
                        window_agg["total_used"] += used
                        window_agg["total_remaining"] += remaining
                        window_agg["total_max"] += limit

                        # Credential has availability if remaining > 0
                        if remaining > 0:
                            tier_avail["available"] += 1
                    else:
                        # No limit = unlimited = always available
                        tier_avail["available"] += 1

            # Add active cooldowns
            for key, cooldown in state.cooldowns.items():
                if cooldown.is_active:
                    cred_stats["cooldowns"][key] = {
                        "reason": cooldown.reason,
                        "remaining_seconds": cooldown.remaining_seconds,
                        "source": cooldown.source,
                    }

            # Add fair cycle state
            for key, fc_state in state.fair_cycle.items():
                if model_filter and key != model_filter:
                    continue
                cred_stats["fair_cycle"][key] = {
                    "exhausted": fc_state.exhausted,
                    "cycle_request_count": fc_state.cycle_request_count,
                }

            # Sort group_usage by quota limit (lowest first), then alphabetically
            # This ensures detail view matches the global summary sort order
            def group_sort_key(item):
                group_name, group_data = item
                windows = group_data.get("windows", {})
                if not windows:
                    return (float("inf"), group_name)  # No windows = sort last

                # Find minimum limit across windows
                min_limit = float("inf")
                for window_data in windows.values():
                    limit = window_data.get("limit")
                    if limit is not None and limit > 0:
                        min_limit = min(min_limit, limit)

                return (min_limit, group_name)

            sorted_group_usage = dict(
                sorted(cred_stats["group_usage"].items(), key=group_sort_key)
            )
            cred_stats["group_usage"] = sorted_group_usage

            stats["credentials"][stable_id] = cred_stats

        # Calculate remaining percentages for each window in each quota group
        for group_stats in stats["quota_groups"].values():
            for window_stats in group_stats.get("windows", {}).values():
                if window_stats["total_max"] > 0:
                    window_stats["remaining_pct"] = round(
                        window_stats["total_remaining"]
                        / window_stats["total_max"]
                        * 100,
                        1,
                    )

        total_input = (
            stats["tokens"]["input_cached"] + stats["tokens"]["input_uncached"]
        )
        stats["tokens"]["input_cache_pct"] = (
            round(stats["tokens"]["input_cached"] / total_input * 100, 1)
            if total_input > 0
            else 0
        )

        return stats

    def _get_provider_plugin_instance(self) -> Optional[Any]:
        """Get provider plugin instance for the current provider."""
        if not self._provider_plugins:
            return None

        # Provider plugins dict maps provider name -> plugin class or instance
        plugin = self._provider_plugins.get(self.provider)
        if plugin is None:
            return None

        # If it's a class, instantiate it (singleton via metaclass); if already an instance, use directly
        if isinstance(plugin, type):
            return plugin()
        return plugin

    def _normalize_model(self, model: str) -> str:
        """
        Normalize model name using provider's mapping.

        Converts internal model names (e.g., claude-sonnet-4-5-thinking) to
        public-facing names (e.g., claude-sonnet-4.5) for consistent storage
        and tracking.

        Args:
            model: Model name (with or without provider prefix)

        Returns:
            Normalized model name (provider prefix preserved if present)
        """
        plugin_instance = self._get_provider_plugin_instance()

        if plugin_instance and hasattr(plugin_instance, "normalize_model_for_tracking"):
            return plugin_instance.normalize_model_for_tracking(model)

        return model

    def _get_model_quota_group(self, model: str) -> Optional[str]:
        """
        Get the quota group for a model, if the provider defines one.

        Models in the same quota group share a single quota pool.
        For example, all Claude models in Antigravity share the same daily quota.

        Args:
            model: Model name (with or without provider prefix)

        Returns:
            Group name (e.g., "claude") or None if not grouped
        """
        plugin_instance = self._get_provider_plugin_instance()

        if plugin_instance and hasattr(plugin_instance, "get_model_quota_group"):
            return plugin_instance.get_model_quota_group(model)

        return None

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """Public helper to get quota group for a model."""
        normalized_model = self._normalize_model(model)
        return self._get_model_quota_group(normalized_model)

    def _get_grouped_models(self, group: str) -> List[str]:
        """
        Get all model names in a quota group (with provider prefix), normalized.

        Returns only public-facing model names, deduplicated. Internal variants
        (e.g., claude-sonnet-4-5-thinking) are normalized to their public name
        (e.g., claude-sonnet-4.5).

        Args:
            group: Group name (e.g., "claude")

        Returns:
            List of normalized, deduplicated model names with provider prefix
            (e.g., ["antigravity/claude-sonnet-4.5", "antigravity/claude-opus-4.5"])
        """
        plugin_instance = self._get_provider_plugin_instance()

        if plugin_instance and hasattr(plugin_instance, "get_models_in_quota_group"):
            models = plugin_instance.get_models_in_quota_group(group)

            # Normalize and deduplicate
            if hasattr(plugin_instance, "normalize_model_for_tracking"):
                seen: Set[str] = set()
                normalized: List[str] = []
                for m in models:
                    prefixed = f"{self.provider}/{m}"
                    norm = plugin_instance.normalize_model_for_tracking(prefixed)
                    if norm not in seen:
                        seen.add(norm)
                        normalized.append(norm)
                return normalized

            # Fallback: just add provider prefix
            return [f"{self.provider}/{m}" for m in models]

        return []

    async def save(self, force: bool = False) -> bool:
        """
        Save usage data to file.

        Args:
            force: Force save even if debounce not elapsed

        Returns:
            True if saved successfully
        """
        if self._storage:
            fair_cycle_global = self._limits.fair_cycle_checker.get_global_state_dict()
            return await self._storage.save(
                self._states, fair_cycle_global, force=force
            )
        return False

    async def get_usage_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a lightweight usage snapshot keyed by accessor.

        Returns:
            Dict mapping accessor -> usage metadata.
        """
        async with self._lock:
            snapshot: Dict[str, Dict[str, Any]] = {}
            for state in self._states.values():
                snapshot[state.accessor] = {
                    "last_used_ts": state.totals.last_used_at or 0,
                }
            return snapshot

    async def shutdown(self) -> None:
        """Shutdown and save any pending data."""
        await self.save(force=True)

    async def reload_from_disk(self) -> None:
        """
        Force reload usage data from disk.

        Useful when wanting fresh stats without making external API calls.
        This reloads persisted state while preserving current credential registrations.
        """
        if not self._storage:
            lib_logger.debug(
                f"reload_from_disk: No storage configured for {self.provider}"
            )
            return

        async with self._lock:
            # Load persisted state
            loaded_states, fair_cycle_global, _ = await self._storage.load()

            # Merge loaded state with current state
            # Keep current accessors but update usage data
            for stable_id, loaded_state in loaded_states.items():
                if stable_id in self._states:
                    # Update usage data from loaded state
                    current = self._states[stable_id]
                    current.model_usage = loaded_state.model_usage
                    current.group_usage = loaded_state.group_usage
                    current.totals = loaded_state.totals
                    current.cooldowns = loaded_state.cooldowns
                    current.fair_cycle = loaded_state.fair_cycle
                    current.last_updated = loaded_state.last_updated
                else:
                    # New credential from disk, add it
                    self._states[stable_id] = loaded_state

            # Reload fair cycle global state
            if fair_cycle_global:
                self._limits.fair_cycle_checker.load_global_state_dict(
                    fair_cycle_global
                )

            lib_logger.info(
                f"Reloaded usage data from disk for {self.provider}: "
                f"{len(self._states)} credentials"
            )

    async def update_quota_baseline(
        self,
        accessor: str,
        model: str,
        quota_max_requests: Optional[int] = None,
        quota_reset_ts: Optional[float] = None,
        quota_used: Optional[int] = None,
        quota_group: Optional[str] = None,
        force: bool = False,
        apply_exhaustion: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Update quota baseline from provider API response.

        Called by provider plugins after receiving rate limit headers or
        quota information from API responses.

        Args:
            accessor: Credential accessor (path or key)
            model: Model name
            quota_max_requests: Max requests allowed in window
            quota_reset_ts: When quota resets (Unix timestamp)
            quota_used: Current used count from API
            quota_group: Optional quota group (uses model if None)
            force: If True, always use API values (for manual refresh).
                If False (default), use max(local, api) to prevent stale
                API data from overwriting accurate local counts during
                background fetches.
            apply_exhaustion: If True, apply cooldown for exhausted quota.
                Provider controls when this is set based on its semantics
                (e.g., Antigravity only on initial fetch, others always
                when remaining == 0).

        Returns:
            Cooldown info dict if cooldown was applied, None otherwise
        """
        stable_id = self._registry.get_stable_id(accessor, self.provider)
        state = self._states.get(stable_id)
        if not state:
            lib_logger.warning(
                f"update_quota_baseline: Unknown credential {accessor[:20]}..."
            )
            return None

        # Normalize model name for consistent tracking
        normalized_model = self._normalize_model(model)
        group_key = quota_group or self._get_model_quota_group(normalized_model)

        primary_def = self._window_manager.get_primary_definition()

        # Update windows based on quota scope
        # If group_key exists, quota is at group level - only update group stats
        # We can't know which model the requests went to from API-level quota
        if group_key:
            group_stats = state.get_group_stats(group_key)
            if primary_def:
                group_window = self._window_manager.get_or_create_window(
                    group_stats.windows, primary_def.name
                )
                self._apply_quota_update(
                    group_window, quota_max_requests, quota_reset_ts, quota_used, force
                )

                # Sync timing to all model windows in this group
                # All models share the same started_at/reset_at/limit as the group
                self._sync_group_timing_to_models(
                    state, group_key, group_window, primary_def.name
                )
        else:
            # No quota group - model IS the quota scope, update model stats
            model_stats = state.get_model_stats(normalized_model)
            if primary_def:
                model_window = self._window_manager.get_or_create_window(
                    model_stats.windows, primary_def.name
                )
                self._apply_quota_update(
                    model_window, quota_max_requests, quota_reset_ts, quota_used, force
                )

        # Mark state as updated
        state.last_updated = time.time()

        # Apply cooldown if provider indicates exhaustion
        # Provider controls when apply_exhaustion is set based on its semantics
        if apply_exhaustion:
            cooldown_target = group_key or normalized_model
            if quota_reset_ts:
                await self._tracking.apply_cooldown(
                    state=state,
                    reason="quota_exhausted",
                    until=quota_reset_ts,
                    model_or_group=cooldown_target,
                    source="api_quota",
                )

                await self._queue_quota_exhausted_log(
                    accessor=accessor,
                    group_key=cooldown_target,
                    quota_reset_ts=quota_reset_ts,
                )

                await self._save_if_needed()

                return {
                    "cooldown_until": quota_reset_ts,
                    "reason": "quota_exhausted",
                    "model": model,
                    "cooldown_hours": max(0.0, (quota_reset_ts - time.time()) / 3600),
                }
            else:
                # ERROR: Provider says exhausted but no reset timestamp!
                lib_logger.error(
                    f"Quota exhausted for {cooldown_target} on "
                    f"{mask_credential(accessor, style='full')} but no reset_timestamp "
                    f"provided by API - cannot apply cooldown"
                )

        await self._save_if_needed()

        return None

    # =========================================================================
    # WINDOW CLEANUP
    # =========================================================================

    def _get_valid_window_names_for_state(self, state: CredentialState) -> Set[str]:
        """
        Get the set of valid window names for a credential based on its tier.

        Uses the provider's usage_reset_configs to determine which window(s)
        should exist for this credential's tier/priority.

        Args:
            state: The credential state

        Returns:
            Set of valid window names (e.g., {"5h"} or {"168h"})
        """
        plugin_class = self._provider_plugins.get(self.provider)
        if not plugin_class:
            # No plugin - use current config windows as valid
            return {w.name for w in self._config.windows}

        # Check if provider defines usage_reset_configs
        usage_reset_configs = getattr(plugin_class, "usage_reset_configs", None)
        if not usage_reset_configs:
            # No tier-specific configs - use current config windows
            return {w.name for w in self._config.windows}

        # Get tier priorities mapping
        tier_priorities = getattr(plugin_class, "tier_priorities", {})
        default_priority = getattr(plugin_class, "default_tier_priority", 10)

        # Resolve credential's priority from tier
        priority = state.priority
        if priority is None and state.tier:
            priority = tier_priorities.get(state.tier, default_priority)
        if priority is None:
            priority = default_priority

        # Find matching usage config for this priority
        matching_config = None
        for key, config in usage_reset_configs.items():
            if isinstance(key, frozenset) and priority in key:
                matching_config = config
                break
        if matching_config is None:
            matching_config = usage_reset_configs.get("default")

        if matching_config is None:
            # No matching config - use current windows
            return {w.name for w in self._config.windows}

        # Generate window name from window_seconds
        window_seconds = matching_config.window_seconds
        if window_seconds == 86400:
            window_name = "daily"
        elif window_seconds % 3600 == 0:
            window_name = f"{window_seconds // 3600}h"
        else:
            window_name = "window"

        return {window_name}

    def _get_window_definitions_for_state(
        self, state: CredentialState
    ) -> List["WindowDefinition"]:
        """
        Get the window definitions for a credential based on its tier.

        Uses the provider's usage_reset_configs to determine which window(s)
        should be used for this credential's tier/priority.

        Args:
            state: The credential state

        Returns:
            List of WindowDefinition objects for this credential's tier
        """
        from .config import WindowDefinition

        plugin_class = self._provider_plugins.get(self.provider)
        if not plugin_class:
            # No plugin - use current config windows
            return list(self._config.windows) if self._config.windows else []

        # Check if provider defines usage_reset_configs
        usage_reset_configs = getattr(plugin_class, "usage_reset_configs", None)
        if not usage_reset_configs:
            # No tier-specific configs - use current config windows
            return list(self._config.windows) if self._config.windows else []

        # Get tier priorities mapping
        tier_priorities = getattr(plugin_class, "tier_priorities", {})
        default_priority = getattr(plugin_class, "default_tier_priority", 10)

        # Resolve credential's priority from tier
        priority = state.priority
        if priority is None and state.tier:
            priority = tier_priorities.get(state.tier, default_priority)
        if priority is None:
            priority = default_priority

        # Find matching usage config for this priority
        matching_config = None
        for key, config in usage_reset_configs.items():
            if isinstance(key, frozenset) and priority in key:
                matching_config = config
                break
        if matching_config is None:
            matching_config = usage_reset_configs.get("default")

        if matching_config is None:
            # No matching config - use current windows
            return list(self._config.windows) if self._config.windows else []

        # Generate window name from window_seconds
        window_seconds = matching_config.window_seconds
        if window_seconds == 86400:
            window_name = "daily"
        elif window_seconds % 3600 == 0:
            window_name = f"{window_seconds // 3600}h"
        else:
            window_name = "window"

        # Create WindowDefinition for this tier
        return [
            WindowDefinition.rolling(
                name=window_name,
                duration_seconds=window_seconds,
                is_primary=True,
                applies_to=matching_config.field_name or "model",
            )
        ]

    def _cleanup_stale_windows_for_state(
        self, state: CredentialState, valid_windows: Set[str]
    ) -> int:
        """
        Remove windows that don't match the credential's current tier config.

        This handles the case where a credential's tier changed and now has
        windows from the old tier that should be cleaned up.

        Args:
            state: The credential state to clean up
            valid_windows: Set of valid window names for this credential

        Returns:
            Number of windows removed
        """
        removed_count = 0

        # Clean up model_usage windows
        for model_name, model_stats in state.model_usage.items():
            windows_to_remove = [
                name for name in model_stats.windows.keys() if name not in valid_windows
            ]
            for window_name in windows_to_remove:
                del model_stats.windows[window_name]
                removed_count += 1
                lib_logger.debug(
                    f"Removed stale window '{window_name}' from model "
                    f"'{model_name}' for {mask_credential(state.accessor, style='full')}"
                )

        # Clean up group_usage windows
        for group_name, group_stats in state.group_usage.items():
            windows_to_remove = [
                name for name in group_stats.windows.keys() if name not in valid_windows
            ]
            for window_name in windows_to_remove:
                del group_stats.windows[window_name]
                removed_count += 1
                lib_logger.debug(
                    f"Removed stale window '{window_name}' from group "
                    f"'{group_name}' for {mask_credential(state.accessor, style='full')}"
                )

        return removed_count

    async def clear_cooldown_if_exists(
        self,
        accessor: str,
        model_or_group: Optional[str] = None,
    ) -> bool:
        """
        Clear a cooldown if one exists for the given scope.

        Used during baseline refresh to clear cooldowns when API
        reports quota is available.

        Args:
            accessor: Credential accessor (path or key)
            model_or_group: Scope of cooldown to clear (None = global)

        Returns:
            True if a cooldown was cleared, False if none existed
        """
        stable_id = self._registry.get_stable_id(accessor, self.provider)
        state = self._states.get(stable_id)
        if not state:
            return False

        key = model_or_group or "_global_"
        cooldown = state.cooldowns.get(key)

        if cooldown and cooldown.is_active:
            await self._tracking.clear_cooldown(state, model_or_group)
            lib_logger.info(
                f"Cleared cooldown for {key} on "
                f"{mask_credential(accessor, style='full')} - API shows quota available "
                f"(was: {cooldown.reason}, source: {cooldown.source})"
            )
            return True

        return False

    def _apply_quota_update(
        self,
        window: WindowStats,
        quota_max_requests: Optional[int],
        quota_reset_ts: Optional[float],
        quota_used: Optional[int],
        force: bool,
    ) -> None:
        """Apply quota update to a window."""
        if quota_max_requests is not None:
            window.limit = quota_max_requests

        # Determine if there's actual usage (either API-reported or local)
        has_usage = (
            quota_used is not None and quota_used > 0
        ) or window.request_count > 0

        # Only set started_at and reset_at if there's actual usage
        # This prevents bogus reset times for unused windows
        if has_usage:
            if quota_reset_ts is not None:
                window.reset_at = quota_reset_ts
            # Set started_at to now if not already set (API shows usage we don't have locally)
            if window.started_at is None:
                window.started_at = time.time()

        if quota_used is not None:
            if force:
                synced_count = quota_used
            else:
                synced_count = max(
                    window.request_count,
                    quota_used,
                    window.success_count + window.failure_count,
                )
            self._reconcile_window_counts(window, synced_count)

    def _reconcile_window_counts(self, window: WindowStats, request_count: int) -> None:
        """Reconcile window counts after quota sync."""
        local_total = window.success_count + window.failure_count
        window.request_count = request_count
        if local_total == 0 and request_count > 0:
            window.success_count = request_count
            window.failure_count = 0
            return

        if request_count < local_total:
            failure_count = min(window.failure_count, request_count)
            success_count = max(0, request_count - failure_count)
            window.success_count = success_count
            window.failure_count = failure_count
            return

        if request_count > local_total:
            window.success_count += request_count - local_total

    def _sync_group_timing_to_models(
        self,
        state: "CredentialState",
        group_key: str,
        group_window: "WindowStats",
        window_name: str,
    ) -> None:
        """
        Sync timing from group window to all model windows in the group.

        Called after updating a group window to ensure all models have
        consistent started_at, reset_at, and limit values. All models
        in a quota group share the same timing since they share API quota.

        Args:
            state: Credential state containing model stats
            group_key: Quota group name
            group_window: The authoritative group window
            window_name: Name of the window to sync (e.g., "5h")
        """
        models_in_group = self._get_grouped_models(group_key)
        for model_name in models_in_group:
            model_stats = state.get_model_stats(model_name, create=False)
            if model_stats:
                model_window = model_stats.windows.get(window_name)
                if model_window:
                    model_window.started_at = group_window.started_at
                    model_window.reset_at = group_window.reset_at
                    if group_window.limit is not None:
                        model_window.limit = group_window.limit

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def config(self) -> ProviderUsageConfig:
        """Get the configuration."""
        return self._config

    @property
    def registry(self) -> CredentialRegistry:
        """Get the credential registry."""
        return self._registry

    @property
    def api(self) -> UsageAPI:
        """Get the usage API facade."""
        return self._api

    @property
    def initialized(self) -> bool:
        """Check if the manager is initialized."""
        return self._initialized

    @property
    def tracking(self) -> TrackingEngine:
        """Get the tracking engine."""
        return self._tracking

    @property
    def limits(self) -> LimitEngine:
        """Get the limit engine."""
        return self._limits

    @property
    def window_manager(self) -> WindowManager:
        """Get the window manager."""
        return self._window_manager

    @property
    def selection(self) -> SelectionEngine:
        """Get the selection engine."""
        return self._selection

    @property
    def states(self) -> Dict[str, CredentialState]:
        """Get all credential states."""
        return self._states

    @property
    def loaded_from_storage(self) -> bool:
        """Whether usage data was loaded from storage."""
        return self._loaded_from_storage

    @property
    def loaded_credentials(self) -> int:
        """Number of credentials loaded from storage."""
        return self._loaded_count

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _resolve_fair_cycle_key(self, group_key: str) -> str:
        """Resolve fair cycle tracking key based on config."""
        if self._config.fair_cycle.tracking_mode == TrackingMode.CREDENTIAL:
            return FAIR_CYCLE_GLOBAL_KEY
        return group_key

    async def _release_credential(self, stable_id: str, model: str) -> None:
        """Release a credential after use and notify waiting tasks."""
        state = self._states.get(stable_id)
        if not state:
            return

        # Decrement active requests
        lock = self._key_locks.get(stable_id)
        if lock:
            async with lock:
                state.active_requests = max(0, state.active_requests - 1)
        else:
            state.active_requests = max(0, state.active_requests - 1)

        # Log release with current state
        remaining = state.active_requests
        max_concurrent = state.max_concurrent
        lib_logger.info(
            f"Released credential {mask_credential(state.accessor, style='full')} "
            f"from {model} (remaining concurrent: {remaining}"
            f"{f'/{max_concurrent}' if max_concurrent else ''})"
        )

        # Notify all tasks waiting on this credential's condition
        condition = self._key_conditions.get(stable_id)
        if condition:
            async with condition:
                condition.notify_all()

    async def _queue_quota_exhausted_log(
        self, accessor: str, group_key: str, quota_reset_ts: float
    ) -> None:
        async with self._quota_exhausted_lock:
            masked = mask_credential(accessor, style="full")
            if masked not in self._quota_exhausted_summary:
                self._quota_exhausted_summary[masked] = {}
            self._quota_exhausted_summary[masked][group_key] = quota_reset_ts

            if self._quota_exhausted_task is None or self._quota_exhausted_task.done():
                self._quota_exhausted_task = asyncio.create_task(
                    self._flush_quota_exhausted_log()
                )

    async def _flush_quota_exhausted_log(self) -> None:
        await asyncio.sleep(0.2)
        async with self._quota_exhausted_lock:
            summary = self._quota_exhausted_summary
            self._quota_exhausted_summary = {}

        if not summary:
            return

        now = time.time()
        parts = []
        for accessor, groups in sorted(summary.items()):
            group_parts = []
            for group, reset_ts in sorted(groups.items()):
                hours = max(0.0, (reset_ts - now) / 3600) if reset_ts else 0.0
                group_parts.append(f"{group} {hours:.1f}h")
            parts.append(f"{accessor}[{', '.join(group_parts)}]")

        lib_logger.info(f"Quota exhausted: {', '.join(parts)}")

    async def _save_if_needed(self) -> None:
        """Persist state if storage is configured."""
        if not self._storage:
            return
        fair_cycle_global = self._limits.fair_cycle_checker.get_global_state_dict()
        saved = await self._storage.save(self._states, fair_cycle_global)
        if not saved:
            await self._schedule_save_flush()

    async def _schedule_save_flush(self) -> None:
        if self._save_task and not self._save_task.done():
            return
        self._save_task = asyncio.create_task(self._flush_save())

    async def _flush_save(self) -> None:
        async with self._save_lock:
            await asyncio.sleep(self._storage.save_debounce_seconds)
            if not self._storage:
                return
            fair_cycle_global = self._limits.fair_cycle_checker.get_global_state_dict()
            await self._storage.save_if_dirty(self._states, fair_cycle_global)

    async def _record_success(
        self,
        stable_id: str,
        model: str,
        quota_group: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        thinking_tokens: int = 0,
        prompt_tokens_cache_read: int = 0,
        prompt_tokens_cache_write: int = 0,
        approx_cost: float = 0.0,
        response_headers: Optional[Dict[str, Any]] = None,
        request_count: int = 1,
    ) -> None:
        """Record a successful request."""
        state = self._states.get(stable_id)
        if state:
            # Normalize model name for consistent tracking
            normalized_model = self._normalize_model(model)
            group_key = quota_group or self._get_model_quota_group(normalized_model)

            await self._tracking.record_success(
                state=state,
                model=normalized_model,
                quota_group=group_key,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                thinking_tokens=thinking_tokens,
                prompt_tokens_cache_read=prompt_tokens_cache_read,
                prompt_tokens_cache_write=prompt_tokens_cache_write,
                approx_cost=approx_cost,
                response_headers=response_headers,
                request_count=request_count,
            )

            # Apply custom cap cooldown if exceeded
            cap_result = self._limits.custom_cap_checker.check(
                state, normalized_model, group_key
            )
            if (
                not cap_result.allowed
                and cap_result.result == LimitResult.BLOCKED_CUSTOM_CAP
                and cap_result.blocked_until
            ):
                await self._tracking.apply_cooldown(
                    state=state,
                    reason="custom_cap",
                    until=cap_result.blocked_until,
                    model_or_group=group_key or normalized_model,
                    source="custom_cap",
                )

            await self._save_if_needed()

    async def _record_failure(
        self,
        stable_id: str,
        model: str,
        quota_group: Optional[str] = None,
        error: Optional[ClassifiedError] = None,
        request_count: int = 1,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        thinking_tokens: int = 0,
        prompt_tokens_cache_read: int = 0,
        prompt_tokens_cache_write: int = 0,
        approx_cost: float = 0.0,
    ) -> None:
        """Record a failed request."""
        state = self._states.get(stable_id)
        if not state:
            return

        # Normalize model name for consistent tracking
        normalized_model = self._normalize_model(model)
        group_key = quota_group or self._get_model_quota_group(normalized_model)

        # Determine cooldown from error
        cooldown_duration = None
        quota_reset = None
        mark_exhausted = False

        if error:
            cooldown_duration = error.retry_after
            quota_reset = error.quota_reset_timestamp

            # Mark exhausted for quota errors with long cooldown
            if error.error_type == "quota_exceeded":
                if (
                    cooldown_duration
                    and cooldown_duration >= self._config.exhaustion_cooldown_threshold
                ):
                    mark_exhausted = True

                    # Log quota exhaustion like legacy system
                    masked = mask_credential(state.accessor, style="full")
                    cooldown_target = group_key or normalized_model

                    if quota_reset:
                        reset_dt = datetime.fromtimestamp(quota_reset, tz=timezone.utc)
                        hours = max(0.0, (quota_reset - time.time()) / 3600)
                        lib_logger.info(
                            f"Quota exhausted for '{cooldown_target}' on {masked}. "
                            f"Resets at {reset_dt.isoformat()} ({hours:.1f}h)"
                        )
                    elif cooldown_duration:
                        hours = cooldown_duration / 3600
                        lib_logger.info(
                            f"Quota exhausted on {masked} for '{cooldown_target}'. "
                            f"Cooldown: {cooldown_duration:.0f}s ({hours:.1f}h)"
                        )

        await self._tracking.record_failure(
            state=state,
            model=normalized_model,
            error_type=error.error_type if error else "unknown",
            quota_group=group_key,
            cooldown_duration=cooldown_duration,
            quota_reset_timestamp=quota_reset,
            mark_exhausted=mark_exhausted,
            request_count=request_count,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=thinking_tokens,
            prompt_tokens_cache_read=prompt_tokens_cache_read,
            prompt_tokens_cache_write=prompt_tokens_cache_write,
            approx_cost=approx_cost,
        )

        # Log fair cycle marking like legacy system
        if mark_exhausted and self._config.fair_cycle.enabled:
            fc_key = group_key or normalized_model
            # Resolve fair cycle tracking key based on config
            if self._config.fair_cycle.tracking_mode == TrackingMode.CREDENTIAL:
                fc_key = FAIR_CYCLE_GLOBAL_KEY
            exhausted_count = sum(
                1
                for s in self._states.values()
                if fc_key in s.fair_cycle and s.fair_cycle[fc_key].exhausted
            )
            masked = mask_credential(state.accessor, style="full")
            lib_logger.info(
                f"Fair cycle: marked {masked} exhausted for {fc_key} ({exhausted_count} total)"
            )

        await self._save_if_needed()
