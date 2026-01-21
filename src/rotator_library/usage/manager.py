# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
UsageManager facade and CredentialContext.

This is the main public API for the usage tracking system.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Union

from ..core.types import CredentialInfo
from ..error_handler import ClassifiedError, classify_error

from .types import (
    UsageStats,
    CredentialState,
    LimitCheckResult,
    RotationMode,
)
from .config import (
    ProviderUsageConfig,
    load_provider_usage_config,
    get_default_windows,
)
from .identity.registry import CredentialRegistry
from .tracking.engine import TrackingEngine
from .tracking.windows import WindowManager
from .limits.engine import LimitEngine
from .selection.engine import SelectionEngine
from .persistence.storage import UsageStorage

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
        self._error: Optional[ClassifiedError] = None
        self._tokens: Dict[str, int] = {}

    async def __aenter__(self) -> "CredentialContext":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Always release the credential
        await self._manager._release_credential(self.stable_id, self.model)

        # Record result
        if self._result == "success":
            await self._manager._record_success(
                self.stable_id,
                self.model,
                self.quota_group,
                self._tokens.get("prompt", 0),
                self._tokens.get("completion", 0),
                self._tokens.get("prompt_cached", 0),
            )
        elif self._result == "failure":
            await self._manager._record_failure(
                self.stable_id,
                self.model,
                self.quota_group,
                self._error,
            )
        elif exc_val is not None:
            # Exception occurred without explicit mark
            error = classify_error(exc_val)
            await self._manager._record_failure(
                self.stable_id,
                self.model,
                self.quota_group,
                error,
            )
        else:
            # No explicit mark and no exception = success without details
            await self._manager._record_success(
                self.stable_id,
                self.model,
                self.quota_group,
            )

        return False  # Don't suppress exceptions

    def mark_success(
        self,
        response: Any = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        prompt_tokens_cached: int = 0,
    ) -> None:
        """Mark request as successful."""
        self._result = "success"
        self._response = response
        self._tokens = {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "prompt_cached": prompt_tokens_cached,
        }

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
        self._selection = SelectionEngine(self._config, self._limits)

        # Storage
        if file_path:
            self._storage = UsageStorage(file_path)
        else:
            self._storage = None

        # State
        self._states: Dict[str, CredentialState] = {}
        self._initialized = False
        self._lock = asyncio.Lock()

        # Concurrency control: per-credential locks and conditions for waiting
        self._key_locks: Dict[str, asyncio.Lock] = {}
        self._key_conditions: Dict[str, asyncio.Condition] = {}

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
            # Load persisted state
            if self._storage:
                self._states = await self._storage.load()

            # Register credentials
            for accessor in credentials:
                stable_id = self._registry.get_stable_id(accessor, self.provider)

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

                # Set max concurrent if configured, applying priority multiplier
                if self._max_concurrent_per_key is not None:
                    base_concurrent = self._max_concurrent_per_key
                    # Apply priority multiplier from config
                    priority = self._states[stable_id].priority
                    multiplier = self._config.get_effective_multiplier(priority)
                    effective_concurrent = base_concurrent * multiplier
                    self._states[stable_id].max_concurrent = effective_concurrent

            self._initialized = True
            lib_logger.info(
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
                                f"Acquired credential {self._mask_accessor(state.accessor)} "
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
                    usage = state.usage.total_requests
                    if usage < best_usage:
                        best_usage = usage
                        best_wait_id = sid

            if best_wait_id is None:
                # All credentials blocked by cooldown or limits, not just concurrency
                # Check if waiting for cooldown makes sense
                soonest_cooldown = self._get_soonest_cooldown_end(
                    states_to_check, normalized_model
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
                    f"All credentials busy. Waiting for {self._mask_accessor(self._states[best_wait_id].accessor)}..."
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
    ) -> Optional[float]:
        """Get the soonest cooldown end time across all credentials."""
        soonest = None
        now = time.time()

        for state in states.values():
            # Check model-specific cooldown
            cooldown = state.get_cooldown(model)
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
        prompt_tokens_cached: int = 0,
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
            prompt_tokens_cached: Cached prompt tokens (e.g., from Claude)
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
                prompt_tokens_cached,
            )
        else:
            await self._record_failure(stable_id, model, quota_group, error)

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
            "credential_count": len(self._states),
            "rotation_mode": self._config.rotation_mode.value,
            "credentials": {},
        }

        for stable_id, state in self._states.items():
            cred_stats = {
                "stable_id": stable_id,
                "accessor_masked": self._mask_accessor(state.accessor),
                "tier": state.tier,
                "priority": state.priority,
                "active_requests": state.active_requests,
                "usage": {
                    "total_requests": state.usage.total_requests,
                    "total_successes": state.usage.total_successes,
                    "total_failures": state.usage.total_failures,
                    "total_tokens": state.usage.total_tokens,
                },
                "windows": {},
                "cooldowns": {},
                "fair_cycle": {},
            }

            # Add window stats
            for window_name, window in state.usage.windows.items():
                cred_stats["windows"][window_name] = {
                    "request_count": window.request_count,
                    "limit": window.limit,
                    "remaining": window.remaining,
                    "reset_at": window.reset_at,
                }

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

            stats["credentials"][stable_id] = cred_stats

        return stats

    def _mask_accessor(self, accessor: str) -> str:
        """Mask an accessor for safe display."""
        if accessor.endswith(".json"):
            # OAuth credential - show filename only
            from pathlib import Path

            return Path(accessor).name
        elif len(accessor) > 12:
            # API key - show first 4 and last 4 chars
            return f"{accessor[:4]}...{accessor[-4:]}"
        else:
            return "***"

    def _get_provider_plugin_instance(self) -> Optional[Any]:
        """Get the provider plugin instance for the current provider."""
        if not self._provider_plugins:
            return None

        # Provider plugins dict maps provider name -> plugin class or instance
        plugin = self._provider_plugins.get(self.provider)
        if plugin is None:
            return None

        # If it's a class, instantiate it; if already an instance, use directly
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

    async def _sync_quota_group_counts(
        self,
        state: CredentialState,
        model: str,
    ) -> None:
        """
        Synchronize request counts across models in the same quota group.

        For providers with shared quota pools (e.g., Antigravity), all models
        in a quota group share the same daily limit. When one model's count
        increases, we sync that count to all other models in the group so
        selection correctly accounts for the shared quota.

        Args:
            state: Credential state that was just updated
            model: The normalized model that was just used
        """
        # Check if this model is in a quota group
        group = self._get_model_quota_group(model)
        if not group:
            return

        # Get all models in the group
        grouped_models = self._get_grouped_models(group)
        if len(grouped_models) <= 1:
            return

        # Get the current request count for the used model
        current_count = state.usage.model_request_counts.get(model, 0)

        # Sync to all other models in the group
        for grouped_model in grouped_models:
            if grouped_model != model:
                state.usage.model_request_counts[grouped_model] = current_count

        lib_logger.debug(
            f"Synced request_count={current_count} across quota group '{group}' "
            f"({len(grouped_models)} models)"
        )

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

    async def shutdown(self) -> None:
        """Shutdown and save any pending data."""
        await self.save(force=True)

    async def update_quota_baseline(
        self,
        accessor: str,
        model: str,
        quota_max_requests: Optional[int] = None,
        quota_reset_ts: Optional[float] = None,
        quota_used: Optional[int] = None,
        quota_group: Optional[str] = None,
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
        group_key = quota_group or normalized_model

        # Get or create primary window
        primary_window = None
        for window in state.usage.windows.values():
            primary_window = window
            break

        if primary_window is None:
            # Create a window if none exists
            from .types import WindowStats

            primary_window = WindowStats(name="api_quota")
            state.usage.windows["api_quota"] = primary_window

        # Update baseline values
        if quota_max_requests is not None:
            primary_window.limit = quota_max_requests
        if quota_reset_ts is not None:
            primary_window.reset_at = quota_reset_ts
        if quota_used is not None:
            primary_window.request_count = quota_used

        # Mark state as updated
        state.last_updated = time.time()

        # Check if we need to apply cooldown (quota exhausted)
        if primary_window.is_exhausted and quota_reset_ts:
            await self._tracking.apply_cooldown(
                state=state,
                reason="quota_exhausted",
                until=quota_reset_ts,
                model_or_group=group_key,
                source="api_quota",
            )

            lib_logger.info(
                f"Quota exhausted for {self._mask_accessor(accessor)}/{model}, "
                f"cooldown until {quota_reset_ts}"
            )

            return {
                "cooldown_until": quota_reset_ts,
                "reason": "quota_exhausted",
                "model": model,
            }

        return None

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
    def tracking(self) -> TrackingEngine:
        """Get the tracking engine."""
        return self._tracking

    @property
    def limits(self) -> LimitEngine:
        """Get the limit engine."""
        return self._limits

    @property
    def selection(self) -> SelectionEngine:
        """Get the selection engine."""
        return self._selection

    @property
    def states(self) -> Dict[str, CredentialState]:
        """Get all credential states."""
        return self._states

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

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
                remaining = state.active_requests
                lib_logger.debug(
                    f"Released credential {self._mask_accessor(state.accessor)} "
                    f"from {model} (remaining: {remaining}"
                    f"{f'/{state.max_concurrent}' if state.max_concurrent else ''})"
                )
        else:
            state.active_requests = max(0, state.active_requests - 1)

        # Release in tracking engine
        await self._tracking.release(state, model)

        # Notify all tasks waiting on this credential's condition
        condition = self._key_conditions.get(stable_id)
        if condition:
            async with condition:
                condition.notify_all()

    async def _record_success(
        self,
        stable_id: str,
        model: str,
        quota_group: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        prompt_tokens_cached: int = 0,
    ) -> None:
        """Record a successful request."""
        state = self._states.get(stable_id)
        if state:
            # Normalize model name for consistent tracking
            normalized_model = self._normalize_model(model)

            await self._tracking.record_success(
                state=state,
                model=normalized_model,
                quota_group=quota_group,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                prompt_tokens_cached=prompt_tokens_cached,
            )

            # Sync request_count across quota group (for shared quota pools)
            await self._sync_quota_group_counts(state, normalized_model)

            if self._storage:
                self._storage.mark_dirty()

    async def _record_failure(
        self,
        stable_id: str,
        model: str,
        quota_group: Optional[str] = None,
        error: Optional[ClassifiedError] = None,
    ) -> None:
        """Record a failed request."""
        state = self._states.get(stable_id)
        if not state:
            return

        # Normalize model name for consistent tracking
        normalized_model = self._normalize_model(model)

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

        await self._tracking.record_failure(
            state=state,
            model=normalized_model,
            error_type=error.error_type if error else "unknown",
            quota_group=quota_group,
            cooldown_duration=cooldown_duration,
            quota_reset_timestamp=quota_reset,
            mark_exhausted=mark_exhausted,
        )

        # Sync request_count across quota group (for shared quota pools)
        await self._sync_quota_group_counts(state, normalized_model)

        if self._storage:
            self._storage.mark_dirty()
