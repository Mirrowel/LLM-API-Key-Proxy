# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Unified request execution with retry and rotation.

This module extracts and unifies the retry logic that was duplicated in:
- _execute_with_retry (lines 1174-1945)
- _streaming_acompletion_with_retry (lines 1947-2780)

The RequestExecutor provides a single code path for all request types,
with streaming vs non-streaming handled as a parameter.
"""

import asyncio
import json
import logging
import os
import random
import time
from copy import deepcopy
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Set,
    TYPE_CHECKING,
    Tuple,
    Union,
)

import httpx
import litellm
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    InternalServerError,
)

from ..core.types import RequestContext, ErrorAction
from ..core.utils import normalize_usage_for_response
from ..core.errors import (
    NoAvailableKeysError,
    PreRequestCallbackError,
    StreamedAPIError,
    ClassifiedError,
    RequestErrorAccumulator,
    classify_error,
    should_rotate_on_error,
    should_retry_same_key,
    mask_credential,
)
from ..core.constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_SMALL_COOLDOWN_RETRY_THRESHOLD,
    DEFAULT_TRANSIENT_RETRY_DELAY,
    DEFAULT_TRANSIENT_RETRY_JITTER,
    DEFAULT_STREAM_RETRY_ON_REASONING_ONLY,
)
from ..request_sanitizer import sanitize_request_payload
from ..transaction_logger import TransactionLogger
from ..failure_logger import log_failure
from ..retry_policy import decide_provider_cooldown, provider_cooldown_env
from ..routing import FallbackPolicy, clone_context_for_target
from ..routing.policy import normalize_route_error_type
from ..routing.types import RouteTarget
from ..native_provider import NativeHTTPTransport, NativeProviderContext, NativeProviderExecutor
from ..field_cache.paths import FieldCachePathError, PathToken, parse_path
from ..transform_trace import REDACTED
from ..usage.accounting import UsageRecord, extract_usage_record
from ..usage.costs import CostBreakdown, CostCalculator

from .types import RetryState, AvailabilityStats
from .filters import CredentialFilter
from .transforms import ProviderTransforms
from .streaming import StreamingHandler
from ..streaming.policy import can_retry_stream_after_error, is_visible_stream_output

if TYPE_CHECKING:
    from ..usage import UsageManager

lib_logger = logging.getLogger("rotator_library")


class RoutingExecutionError(RuntimeError):
    """Internal error used when a routed target cannot use its requested mode."""

    def __init__(self, message: str, error_type: str = "unsupported_operation") -> None:
        super().__init__(message)
        self.error_type = error_type


class RequestExecutor:
    """
    Unified retry/rotation logic for all request types.

    This class handles:
    - Credential rotation across providers
    - Per-credential retry with backoff
    - Error classification and handling
    - Streaming and non-streaming requests
    """

    def __init__(
        self,
        usage_managers: Dict[str, "UsageManager"],
        cooldown_manager: Any,
        credential_filter: CredentialFilter,
        provider_transforms: ProviderTransforms,
        provider_plugins: Dict[str, Any],
        http_client: httpx.AsyncClient,
        max_retries: int = DEFAULT_MAX_RETRIES,
        global_timeout: int = 30,
        abort_on_callback_error: bool = True,
        litellm_provider_params: Optional[Dict[str, Any]] = None,
        litellm_logger_fn: Optional[Any] = None,
        provider_instances: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize RequestExecutor.

        Args:
            usage_managers: Dict mapping provider names to UsageManager instances
            cooldown_manager: CooldownManager instance
            credential_filter: CredentialFilter instance
            provider_transforms: ProviderTransforms instance
            provider_plugins: Dict mapping provider names to plugin classes
            http_client: Shared httpx.AsyncClient for provider requests
            max_retries: Max retries per credential
            global_timeout: Global request timeout in seconds
            abort_on_callback_error: Abort on pre-request callback errors
            litellm_provider_params: Optional dict of provider-specific LiteLLM
                parameters to merge into requests (e.g., custom headers, timeouts)
            litellm_logger_fn: Optional callback function for LiteLLM logging
            provider_instances: Shared dict for caching provider instances.
                If None, creates a new dict (not recommended - leads to duplicate instances).
        """
        self._usage_managers = usage_managers
        self._cooldown = cooldown_manager
        self._filter = credential_filter
        self._transforms = provider_transforms
        self._plugins = provider_plugins
        self._plugin_instances: Dict[str, Any] = (
            provider_instances if provider_instances is not None else {}
        )
        self._http_client = http_client
        self._max_retries = max_retries
        self._global_timeout = global_timeout
        self._abort_on_callback_error = abort_on_callback_error
        self._litellm_provider_params = litellm_provider_params or {}
        self._litellm_logger_fn = litellm_logger_fn
        # StreamingHandler no longer needs usage_manager - we pass cred_context directly
        self._streaming_handler = StreamingHandler()
        self._native_executor = NativeProviderExecutor()

    def _get_transient_retry_delay(self) -> float:
        """Small jittered delay used before transient retries and rotations."""
        try:
            base = float(
                os.environ.get("TRANSIENT_RETRY_DELAY", DEFAULT_TRANSIENT_RETRY_DELAY)
            )
        except (TypeError, ValueError):
            base = DEFAULT_TRANSIENT_RETRY_DELAY
        try:
            jitter = float(
                os.environ.get("TRANSIENT_RETRY_JITTER", DEFAULT_TRANSIENT_RETRY_JITTER)
            )
        except (TypeError, ValueError):
            jitter = DEFAULT_TRANSIENT_RETRY_JITTER
        return max(0.0, base) + random.uniform(0.0, max(0.0, jitter))

    def _is_transient_error(self, classified: ClassifiedError) -> bool:
        return classified.error_type in {"server_error", "api_connection", "rate_limit"}

    def _stream_retry_on_reasoning_only_enabled(self) -> bool:
        value = os.environ.get("STREAM_RETRY_ON_REASONING_ONLY")
        if value is None:
            return DEFAULT_STREAM_RETRY_ON_REASONING_ONLY
        return value.strip().lower() in {"1", "true", "yes", "on"}

    async def _sleep_before_transient_action(
        self,
        delay: float,
        deadline: float,
        reason: str,
    ) -> bool:
        """Sleep if the request deadline has enough budget left."""
        if delay <= 0:
            return True
        remaining = deadline - time.time()
        if delay > remaining:
            lib_logger.info(
                f"Skipping {reason} delay ({delay:.1f}s); only {remaining:.1f}s left"
            )
            return False
        lib_logger.info(f"Waiting {delay:.1f}s before {reason}")
        await asyncio.sleep(delay)
        return True

    def _get_plugin_instance(self, provider: str) -> Optional[Any]:
        """Get or create a plugin instance for a provider."""
        if provider not in self._plugin_instances:
            plugin_class = self._plugins.get(provider)
            if plugin_class:
                if isinstance(plugin_class, type):
                    self._plugin_instances[provider] = plugin_class()
                else:
                    self._plugin_instances[provider] = plugin_class
            else:
                return None
        return self._plugin_instances[provider]

    def _has_tier_support(self, provider: str) -> bool:
        """
        Check if provider has tier/priority configuration.

        Providers with tier support define tier_priorities mapping
        (e.g., GeminiCli, NanoGpt).

        Args:
            provider: Provider name

        Returns:
            True if provider has tier configuration, False otherwise
        """
        plugin = self._get_plugin_instance(provider)
        if not plugin:
            return False
        tier_priorities = getattr(plugin, "tier_priorities", {})
        return bool(tier_priorities)

    def _get_usage_display(
        self,
        state: Any,
        model: str,
        quota_group: Optional[str],
        usage_manager: "UsageManager",
    ) -> int:
        """
        Get usage count from the primary window.

        This returns the same usage count used for credential selection,
        ensuring consistency between what's logged and what's used for rotation.

        Args:
            state: CredentialState object
            model: Model name
            quota_group: Optional quota group name
            usage_manager: UsageManager instance

        Returns:
            Request count from primary window, or 0 if unavailable
        """
        if not state:
            return 0

        window_manager = getattr(usage_manager, "window_manager", None)
        if not window_manager:
            return state.totals.request_count

        primary_def = window_manager.get_primary_definition()
        if not primary_def:
            return state.totals.request_count

        # Get windows based on what the primary definition applies to
        # This mirrors the logic in selection/engine.py:_get_usage_count
        windows = None
        if primary_def.applies_to == "model":
            model_stats = state.get_model_stats(model, create=False)
            if model_stats:
                windows = model_stats.windows
        elif primary_def.applies_to == "group":
            group_key = quota_group or model
            group_stats = state.get_group_stats(group_key, create=False)
            if group_stats:
                windows = group_stats.windows

        if windows:
            window = window_manager.get_active_window(windows, primary_def.name)
            if window:
                return window.request_count

        return state.totals.request_count

    def _get_quota_display(
        self,
        state: Any,
        model: str,
        quota_group: Optional[str],
        usage_manager: "UsageManager",
    ) -> str:
        """
        Get quota display string for logging.

        Checks group stats first (if quota_group provided), then falls back
        to model stats. Returns a formatted string like "5/50 [90%]".

        Args:
            state: CredentialState object
            model: Model name
            quota_group: Optional quota group name
            usage_manager: UsageManager instance

        Returns:
            Formatted quota display string, or "?/?" if unavailable
        """
        if not state:
            return "?/?"

        window_manager = getattr(usage_manager, "window_manager", None)
        if not window_manager:
            return "?/?"

        primary_def = window_manager.get_primary_definition()
        if not primary_def:
            return "?/?"

        window = None
        # Check GROUP first if quota_group provided (shared limits)
        if quota_group:
            group_stats = state.get_group_stats(quota_group, create=False)
            if group_stats:
                window = group_stats.windows.get(primary_def.name)

        # Fall back to MODEL if no group limit found
        if window is None or window.limit is None:
            model_stats = state.get_model_stats(model, create=False)
            if model_stats:
                window = model_stats.windows.get(primary_def.name)

        # Display quota if we found a window with a limit
        if window and window.limit is not None:
            remaining = max(0, window.limit - window.request_count)
            pct = round(remaining / window.limit * 100) if window.limit else 0
            return f"{window.request_count}/{window.limit} [{pct}%]"

        return "?/?"

    def _log_acquiring_credential(
        self,
        model: str,
        tried_count: int,
        availability: Dict[str, Any],
    ) -> None:
        """
        Log credential acquisition attempt with availability info.

        Args:
            model: Model name
            tried_count: Number of credentials already tried
            availability: Availability stats dict from usage manager
        """
        blocked = availability.get("blocked_by", {})
        blocked_parts = []
        if blocked.get("cooldowns"):
            blocked_parts.append(f"cd:{blocked['cooldowns']}")
        if blocked.get("fair_cycle"):
            blocked_parts.append(f"fc:{blocked['fair_cycle']}")
        if blocked.get("custom_caps"):
            blocked_parts.append(f"cap:{blocked['custom_caps']}")
        if blocked.get("window_limits"):
            blocked_parts.append(f"wl:{blocked['window_limits']}")
        if blocked.get("concurrent"):
            blocked_parts.append(f"con:{blocked['concurrent']}")
        blocked_str = f"({', '.join(blocked_parts)})" if blocked_parts else ""
        lib_logger.info(
            f"Acquiring credential for model {model}. Tried: {tried_count}/"
            f"{availability.get('available', 0)}({availability.get('total', 0)}{blocked_str})"
        )

    async def _prepare_request_kwargs(
        self,
        provider: str,
        model: str,
        cred: str,
        context: "RequestContext",
        *,
        credential_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepare request kwargs with transforms, sanitization, and provider params.

        Args:
            provider: Provider name
            model: Model name
            cred: Credential string
            context: Request context

        Returns:
            Prepared kwargs dict for the LiteLLM call
        """
        # Apply transforms
        kwargs = await self._transforms.apply(
            provider,
            model,
            cred,
            context.kwargs.copy(),
            provider_config_override=context.provider_config,
            transaction_logger=context.transaction_logger,
            credential_id=credential_id,
            transport="sse" if context.streaming else "http",
            trace_metadata={
                "session_id": context.session_id,
                "scope_key": context.usage_manager_key,
                "classifier": context.classifier,
            },
        )

        # Sanitize request payload. Some provider compatibility fields are
        # intentionally removed here, so record it as its own transform pass.
        before_sanitize = deepcopy(kwargs) if context.transaction_logger else None
        kwargs = sanitize_request_payload(kwargs, model)
        self._log_executor_trace(
            context,
            "after_request_sanitization",
            kwargs,
            direction="request",
            stage="client",
            credential_id=credential_id,
            changed_from_previous=(before_sanitize != kwargs) if before_sanitize is not None else None,
            metadata={"provider": provider, "model": model},
        )

        # Apply provider-specific LiteLLM params
        before_params = deepcopy(kwargs) if context.transaction_logger else None
        self._apply_litellm_provider_params(provider, kwargs)
        self._log_executor_trace(
            context,
            "after_litellm_provider_params",
            kwargs,
            direction="request",
            stage="client",
            credential_id=credential_id,
            changed_from_previous=(before_params != kwargs) if before_params is not None else None,
            metadata={"provider": provider, "model": model},
        )

        # Add transaction context for provider logging
        if context.transaction_logger:
            kwargs["transaction_context"] = context.transaction_logger.get_context()
            self._log_executor_trace(
                context,
                "after_transaction_context_attached",
                kwargs,
                direction="request",
                stage="client",
                credential_id=credential_id,
                changed_from_previous=True,
                metadata={"provider": provider, "model": model},
            )

        return kwargs

    def _log_acquired_credential(
        self,
        cred: str,
        model: str,
        state: Any,
        quota_group: Optional[str],
        availability: Dict[str, Any],
        usage_manager: "UsageManager",
    ) -> None:
        """
        Log successful credential acquisition.

        Format varies based on provider capabilities:
        - Providers with tier support: (tier, priority, selection, quota)
        - Providers without tiers but with quotas: (selection, quota)
        - Providers without tiers or quotas: (selection, usage)

        Args:
            cred: Credential string
            model: Model name
            state: CredentialState object
            quota_group: Optional quota group
            availability: Availability stats dict
            usage_manager: UsageManager instance
        """
        selection_mode = availability.get("rotation_mode")

        # Extract provider from model (e.g., "nvidia_nim" from "nvidia_nim/deepseek-ai/...")
        provider = model.split("/")[0] if "/" in model else None

        if provider and self._has_tier_support(provider):
            # Full format with tier/priority/quota for providers with tier configuration
            tier = state.tier if state else None
            priority = state.priority if state else None
            quota_display = self._get_quota_display(
                state, model, quota_group, usage_manager
            )
            lib_logger.info(
                f"Acquired key {mask_credential(cred)} for model {model} "
                f"(tier: {tier}, priority: {priority}, selection: {selection_mode}, quota: {quota_display})"
            )
        else:
            # Simple format for providers without tier configuration
            # Check if there's quota info available (limit set on window)
            quota_display = self._get_quota_display(
                state, model, quota_group, usage_manager
            )
            if quota_display != "?/?":
                # Has quota limits - show selection and quota
                lib_logger.info(
                    f"Acquired key {mask_credential(cred)} for model {model} "
                    f"(selection: {selection_mode}, quota: {quota_display})"
                )
            else:
                # No quota limits - show selection and usage from primary window
                usage = self._get_usage_display(
                    state, model, quota_group, usage_manager
                )
                lib_logger.info(
                    f"Acquired key {mask_credential(cred)} for model {model} "
                    f"(selection: {selection_mode}, usage: {usage})"
                )

    async def _run_pre_request_callback(
        self,
        context: "RequestContext",
        kwargs: Dict[str, Any],
    ) -> None:
        """
        Run pre-request callback if configured.

        Args:
            context: Request context
            kwargs: Request kwargs

        Raises:
            PreRequestCallbackError: If callback fails and abort_on_callback_error is True
        """
        if context.pre_request_callback:
            try:
                before = deepcopy(kwargs) if context.transaction_logger else None
                await context.pre_request_callback(context.request, kwargs)
                if before is not None and before != kwargs:
                    self._log_executor_trace(
                        context,
                        "after_pre_request_callback",
                        kwargs,
                        direction="request",
                        stage="client",
                        changed_from_previous=True,
                        snapshot=True,
                    )
            except Exception as e:
                if self._abort_on_callback_error:
                    raise PreRequestCallbackError(str(e)) from e
                lib_logger.warning(f"Pre-request callback failed: {e}")

    async def _execute_provider_request(
        self,
        provider: str,
        model: str,
        plugin: Any,
        credential_secret: str,
        credential_id: str,
        kwargs: Dict[str, Any],
        context: RequestContext,
    ) -> Any:
        """Execute one provider request using routed execution-mode rules."""

        target = _current_route_target(context)
        execution = target.execution if target else "auto"
        self._log_executor_trace(
            context,
            "pre_provider_execution_request",
            kwargs,
            direction="request",
            stage="provider",
            credential_id=credential_id,
            metadata={"execution": execution, "provider": provider, "model": model},
        )
        if execution == "litellm_fallback":
            self._log_routing_trace(
                context,
                "routing_litellm_fallback",
                _target_trace(target) if target else {"provider": provider, "model": model},
            )
            return await self._execute_litellm_request(kwargs, credential_secret, context=context, credential_id=credential_id)

        if execution == "custom" or (execution == "auto" and plugin and plugin.has_custom_logic()):
            if not plugin or not plugin.has_custom_logic():
                raise RoutingExecutionError(f"Provider {provider} does not support custom execution")
            kwargs["credential_identifier"] = credential_secret
            self._log_executor_trace(
                context,
                "provider_execution_request",
                kwargs,
                direction="request",
                stage="provider",
                credential_id=credential_id,
                metadata={"execution": "custom", "provider": provider, "model": model},
            )
            return await plugin.acompletion(self._http_client, **kwargs)

        if execution == "native" or (execution == "auto" and _provider_native_protocol(plugin, model, target)):
            native_context, native_request = self._build_native_provider_context(
                provider,
                model,
                plugin,
                credential_secret,
                credential_id,
                context,
                target,
                raw_request=kwargs,
                return_request=True,
            )
            self._log_routing_trace(
                context,
                "routing_native_execution_selected",
                _target_trace(target) if target else {"provider": provider, "model": model},
                metadata={"protocol": native_context.protocol_name, "operation": native_context.operation},
            )
            return await self._get_native_executor().execute(native_request, native_context, NativeHTTPTransport(self._http_client))

        return await self._execute_litellm_request(kwargs, credential_secret, context=context, credential_id=credential_id)

    def _get_native_executor(self) -> NativeProviderExecutor:
        """Return the shared native executor for process-local field-cache state."""

        native_executor = getattr(self, "_native_executor", None)
        if native_executor is None:
            native_executor = NativeProviderExecutor()
            self._native_executor = native_executor
        return native_executor

    async def _execute_litellm_request(
        self,
        kwargs: Dict[str, Any],
        credential_secret: str,
        *,
        context: Optional[RequestContext] = None,
        credential_id: Optional[str] = None,
    ) -> Any:
        """Execute the existing LiteLLM request path."""

        kwargs["api_key"] = credential_secret
        self._apply_litellm_logger(kwargs)
        kwargs.pop("transaction_context", None)
        if context:
            self._log_executor_trace(
                context,
                "provider_execution_request",
                kwargs,
                direction="request",
                stage="provider",
                credential_id=credential_id,
                metadata={"execution": "litellm", "provider": context.provider, "model": context.model},
            )
        return await litellm.acompletion(**kwargs)

    def _build_native_provider_context(
        self,
        provider: str,
        model: str,
        plugin: Any,
        credential_secret: str,
        credential_id: str,
        context: RequestContext,
        target: Optional[RouteTarget],
        raw_request: Optional[Dict[str, Any]] = None,
        transport: str = "http",
        stream: bool = False,
        return_request: bool = False,
    ) -> NativeProviderContext | tuple[NativeProviderContext, Dict[str, Any]]:
        """Build native provider context from provider declarations."""

        if not plugin:
            raise RoutingExecutionError(f"Provider {provider} has no plugin for native execution")
        protocol_name = _provider_native_protocol(plugin, model, target)
        if not protocol_name:
            raise RoutingExecutionError(f"Provider {provider} has no native protocol declaration")
        if not hasattr(plugin, "get_native_endpoint") or not hasattr(plugin, "get_native_headers"):
            raise RoutingExecutionError(f"Provider {provider} has no native endpoint/header helpers")
        public_model = model
        native_model = plugin.normalize_native_model(model) if hasattr(plugin, "normalize_native_model") else _strip_provider_prefix(model)
        request_payload = _native_request_payload(raw_request or {})
        if native_model:
            request_payload["model"] = native_model
        operation = plugin.get_native_operation(native_model, request_payload, stream=stream) if hasattr(plugin, "get_native_operation") else "chat"
        if hasattr(plugin, "prepare_native_request"):
            prepared = plugin.prepare_native_request(request_payload, model=native_model, operation=operation)
            if prepared is not request_payload:
                request_payload = dict(prepared)
            self._log_executor_trace(
                context,
                "provider_native_request_prepared",
                request_payload,
                direction="request",
                stage="provider",
                credential_id=credential_id,
                metadata={"provider": provider, "model": public_model, "native_model": native_model, "operation": operation},
            )
        endpoint = plugin.get_native_endpoint(model=native_model, operation=operation)
        headers = plugin.get_native_headers(credential_secret, model=native_model, operation=operation)
        native_context = NativeProviderContext(
            provider=provider,
            model=native_model,
            protocol_name=protocol_name,
            endpoint=endpoint,
            operation=operation,
            headers=headers,
            credential_id=credential_id,
            session_id=context.session_id,
            scope_key=context.usage_manager_key,
            classifier=context.classifier,
            transport=transport,
            adapter_names=tuple(plugin.get_adapter_names(native_model) if hasattr(plugin, "get_adapter_names") else ()),
            adapter_config=dict(plugin.get_adapter_config(native_model) if hasattr(plugin, "get_adapter_config") else {}),
            field_cache_rules=_merged_field_cache_rules(provider, public_model, plugin),
            transaction_logger=context.transaction_logger,
            metadata={"public_model": public_model},
        )
        if return_request:
            return native_context, request_payload
        return native_context

    async def execute(
        self,
        context: RequestContext,
    ) -> Union[Any, AsyncGenerator[str, None]]:
        """
        Execute request with retry/rotation.

        This is the main entry point for request execution.

        Args:
            context: RequestContext with all request details

        Returns:
            Response object or async generator for streaming
        """
        if context.streaming and context.routing_targets:
            return self._execute_streaming_with_fallback(context)
        if context.streaming:
            return self._execute_streaming(context)
        elif context.routing_targets:
            return await self._execute_non_streaming_with_fallback(context)
        else:
            return await self._execute_non_streaming(context)

    async def _execute_non_streaming_with_fallback(self, context: RequestContext) -> Any:
        """Execute an ordered non-streaming fallback target chain.

        The normal single-target path remains `_execute_non_streaming()`. This
        wrapper only runs when request building has populated `routing_targets`,
        preserving existing behavior for all current requests.
        """

        targets = tuple(context.routing_targets or ())
        if not targets:
            return await self._execute_non_streaming(context)
        policy = FallbackPolicy()
        last_failure: Any = None
        target_failures: List[Dict[str, Any]] = []
        self._log_routing_trace(
            context,
            "routing_decision",
            {"requested_model": context.model, "target_count": len(targets)},
            metadata={"group": context.routing_group_name, "targets": [_target_trace(target) for target in targets]},
        )
        for index, target in enumerate(targets):
            target_context = clone_context_for_target(
                context,
                target,
                target_index=index,
                credentials=_target_scope_value(target, "credentials", context.credentials),
                usage_manager_key=_target_scope_value(target, "usage_manager_key", target.provider),
                provider_config=_target_scope_value(target, "provider_config", context.provider_config),
                credential_secrets=_target_scope_value(target, "credential_secrets", context.credential_secrets),
            )
            self._log_routing_trace(
                context,
                "routing_target_attempt_started",
                _target_trace(target),
                metadata={"target_index": index, "group": context.routing_group_name},
            )
            try:
                result = await self._execute_non_streaming(target_context)
            except Exception as exc:
                last_failure = exc
                error_type = _route_error_type(exc, target.provider)
                self._log_routing_trace(
                    context,
                    "routing_target_attempt_failed",
                    _target_trace(target),
                    metadata={"target_index": index, "error_type": error_type, "exception": exc.__class__.__name__},
                )
                target_failures.append(_target_failure_summary(target, error_type))
                if index >= len(targets) - 1 or not policy.should_fallback(error_type, group=context.routing_group):
                    self._log_routing_trace(context, "routing_fallback_exhausted", _target_trace(target), metadata={"error_type": error_type, "fallback_targets": target_failures})
                    raise
                self._log_routing_trace(context, "routing_fallback_selected", _target_trace(targets[index + 1]), metadata={"from_target_index": index, "to_target_index": index + 1, "reason": error_type})
                continue

            error_type = _route_error_type_from_response(result)
            if error_type:
                last_failure = result
                self._log_routing_trace(
                    context,
                    "routing_target_attempt_failed",
                    _target_trace(target),
                    metadata={"target_index": index, "error_type": error_type},
                )
                target_failures.append(_target_failure_summary(target, error_type, status_code=_route_status_code_from_response(result)))
                if index < len(targets) - 1 and policy.should_fallback(error_type, group=context.routing_group):
                    self._log_routing_trace(context, "routing_fallback_selected", _target_trace(targets[index + 1]), metadata={"from_target_index": index, "to_target_index": index + 1, "reason": error_type})
                    continue
                self._log_routing_trace(context, "routing_fallback_exhausted", _target_trace(target), metadata={"error_type": error_type, "fallback_targets": target_failures})
                return _with_fallback_summary(result, target_failures)

            self._log_routing_trace(
                context,
                "routing_target_attempt_succeeded",
                _target_trace(target),
                metadata={"target_index": index},
            )
            return result

        if isinstance(last_failure, Exception):
            raise last_failure
        return last_failure

    async def _execute_streaming_with_fallback(self, context: RequestContext) -> AsyncGenerator[str, None]:
        """Execute streaming fallback targets with pre-output-only failover."""

        targets = tuple(context.routing_targets or ())
        if not targets:
            async for chunk in self._execute_streaming(context):
                yield chunk
            return
        policy = FallbackPolicy()
        target_failures: List[Dict[str, Any]] = []
        self._log_routing_trace(
            context,
            "routing_decision",
            {"requested_model": context.model, "target_count": len(targets), "stream": True},
            metadata={"group": context.routing_group_name, "streaming_policy": _group_streaming_policy(context.routing_group), "targets": [_target_trace(target) for target in targets]},
        )
        for index, target in enumerate(targets):
            emitted_output = False
            pending_chunks: List[str] = []
            terminal_error_type: Optional[str] = None
            target_context = clone_context_for_target(
                context,
                target,
                target_index=index,
                credentials=_target_scope_value(target, "credentials", context.credentials),
                usage_manager_key=_target_scope_value(target, "usage_manager_key", target.provider),
                provider_config=_target_scope_value(target, "provider_config", context.provider_config),
                credential_secrets=_target_scope_value(target, "credential_secrets", context.credential_secrets),
            )
            self._log_routing_trace(
                context,
                "routing_stream_target_attempt_started",
                _target_trace(target),
                metadata={"target_index": index, "group": context.routing_group_name},
            )
            try:
                async for chunk in self._execute_streaming(target_context):
                    chunk_error_type = _stream_chunk_error_type(chunk)
                    if chunk_error_type and not emitted_output:
                        terminal_error_type = chunk_error_type
                        pending_chunks.append(chunk)
                        continue
                    if _stream_chunk_is_visible_output(chunk):
                        for pending in pending_chunks:
                            yield pending
                        pending_chunks.clear()
                        emitted_output = True
                        yield chunk
                        continue
                    pending_chunks.append(chunk)
                if terminal_error_type and not emitted_output:
                    error_type = terminal_error_type
                    self._log_routing_trace(
                        context,
                        "routing_stream_target_attempt_failed",
                        _target_trace(target),
                        metadata={"target_index": index, "error_type": error_type, "emitted_output": emitted_output, "terminal_error_frame": True},
                    )
                    target_failures.append(_target_failure_summary(target, error_type))
                    if index < len(targets) - 1 and _streaming_policy_allows_fallback(context.routing_group) and policy.should_fallback(error_type, group=context.routing_group, stream=True, emitted_output=False):
                        self._log_routing_trace(
                            context,
                            "routing_fallback_selected",
                            _target_trace(targets[index + 1]),
                            metadata={"from_target_index": index, "to_target_index": index + 1, "reason": error_type, "stream": True},
                        )
                        continue
                    self._log_routing_trace(context, "routing_fallback_exhausted", _target_trace(target), metadata={"error_type": error_type, "stream": True, "streaming_policy": _group_streaming_policy(context.routing_group), "fallback_targets": target_failures})
                    for pending in pending_chunks:
                        yield pending
                    return
                for pending in pending_chunks:
                    yield pending
                self._log_routing_trace(
                    context,
                    "routing_stream_target_attempt_succeeded",
                    _target_trace(target),
                    metadata={"target_index": index, "emitted_output": emitted_output},
                )
                return
            except Exception as exc:
                error_type = _route_error_type(exc, target.provider)
                self._log_routing_trace(
                    context,
                    "routing_stream_target_attempt_failed",
                    _target_trace(target),
                    metadata={"target_index": index, "error_type": error_type, "emitted_output": emitted_output},
                )
                target_failures.append(_target_failure_summary(target, error_type))
                if emitted_output:
                    self._log_routing_trace(
                        context,
                        "routing_stream_fallback_blocked_after_output",
                        _target_trace(target),
                        metadata={"target_index": index, "error_type": error_type},
                    )
                    raise
                if index < len(targets) - 1 and _streaming_policy_allows_fallback(context.routing_group) and policy.should_fallback(error_type, group=context.routing_group, stream=True, emitted_output=False):
                    self._log_routing_trace(
                        context,
                        "routing_fallback_selected",
                        _target_trace(targets[index + 1]),
                        metadata={"from_target_index": index, "to_target_index": index + 1, "reason": error_type, "stream": True},
                    )
                    continue
                self._log_routing_trace(context, "routing_fallback_exhausted", _target_trace(target), metadata={"error_type": error_type, "stream": True, "streaming_policy": _group_streaming_policy(context.routing_group), "fallback_targets": target_failures})
                raise

    @staticmethod
    def _log_routing_trace(context: RequestContext, pass_name: str, data: Any, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record routing trace entries without affecting request execution."""

        if not context.transaction_logger:
            return
        context.transaction_logger.log_transform_pass(
            pass_name,
            data,
            direction="metadata",
            stage="routing",
            metadata=metadata or {},
            snapshot=False,
        )

    @staticmethod
    def _log_executor_trace(
        context: RequestContext,
        pass_name: str,
        data: Any,
        *,
        direction: str,
        stage: str,
        credential_id: Optional[str] = None,
        changed_from_previous: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        snapshot: bool = True,
    ) -> None:
        """Record live executor trace boundaries without affecting requests."""

        if not context.transaction_logger:
            return
        context.transaction_logger.log_transform_pass(
            pass_name,
            data,
            direction=direction,
            stage=stage,
            credential_id=credential_id,
            transport="sse" if context.streaming else "http",
            changed_from_previous=changed_from_previous,
            metadata={
                "provider": context.provider,
                "model": context.model,
                "session_id": context.session_id,
                "scope_key": context.usage_manager_key,
                "classifier": context.classifier,
                **(metadata or {}),
            },
            snapshot=snapshot,
        )

    def _terminal_stream_error_lines(self, context: RequestContext, error_data: Dict[str, Any]) -> Tuple[str, str]:
        """Return executor-created terminal SSE lines and trace them first."""

        error_line = f"data: {json.dumps(error_data)}\n\n"
        done_line = "data: [DONE]\n\n"
        self._log_executor_trace(
            context,
            "stream_error_event",
            error_data,
            direction="stream",
            stage="client",
            snapshot=False,
        )
        self._log_executor_trace(
            context,
            "stream_done_event",
            {"raw": done_line},
            direction="stream",
            stage="final",
            snapshot=False,
        )
        return error_line, done_line

    async def _prepare_execution(
        self,
        context: RequestContext,
    ) -> Tuple["UsageManager", Any, List[str], Optional[str], Dict[str, Any]]:
        provider = context.provider
        model = context.model

        usage_manager = self._usage_managers.get(context.usage_manager_key or provider)
        if not usage_manager:
            raise NoAvailableKeysError(f"No UsageManager for provider {provider}")

        filter_result = self._filter.filter_by_tier(
            context.credentials, model, provider
        )
        credentials = filter_result.all_usable
        quota_group = usage_manager.get_model_quota_group(model)

        await self._ensure_initialized(usage_manager, context, filter_result)
        await self._validate_request(provider, model, context.kwargs)

        if not credentials:
            raise NoAvailableKeysError(f"No compatible credentials for model {model}")

        request_headers = (
            dict(context.request.headers) if context.request is not None else {}
        )

        return usage_manager, filter_result, credentials, quota_group, request_headers

    async def _execute_non_streaming(
        self,
        context: RequestContext,
    ) -> Any:
        """
        Execute non-streaming request with retry/rotation.

        Args:
            context: RequestContext with all request details

        Returns:
            Response object
        """
        provider = context.provider
        model = context.model
        deadline = context.deadline

        (
            usage_manager,
            filter_result,
            credentials,
            quota_group,
            request_headers,
        ) = await self._prepare_execution(context)

        error_accumulator = RequestErrorAccumulator()
        error_accumulator.model = model
        error_accumulator.provider = provider

        retry_state = RetryState()
        last_exception: Optional[Exception] = None

        while time.time() < deadline:
            # Check for untried credentials
            untried = [c for c in credentials if c not in retry_state.tried_credentials]
            if not untried:
                lib_logger.warning(
                    f"All {len(credentials)} credentials tried for {model}"
                )
                break

            # Wait for provider cooldown
            await self._wait_for_cooldown(provider, deadline)

            # Acquire credential using context manager
            try:
                availability = await usage_manager.get_availability_stats(
                    model, quota_group
                )
                self._log_acquiring_credential(
                    model, len(retry_state.tried_credentials), availability
                )
                async with await usage_manager.acquire_credential(
                    model=model,
                    quota_group=quota_group,
                    session_id=context.session_id,
                    session_affinity_key=context.session_affinity_key,
                    candidates=untried,
                    priorities=filter_result.priorities,
                    deadline=deadline,
                ) as cred_context:
                    cred = cred_context.credential
                    credential_secret = context.credential_secrets.get(cred, cred)
                    retry_state.record_attempt(cred)

                    state = getattr(usage_manager, "states", {}).get(
                        cred_context.stable_id
                    )
                    self._log_acquired_credential(
                        cred, model, state, quota_group, availability, usage_manager
                    )

                    try:
                        # Prepare request kwargs
                        kwargs = await self._prepare_request_kwargs(
                            provider,
                            model,
                            cred,
                            context,
                            credential_id=cred_context.stable_id,
                        )

                        # Log transformed request if it differs from original
                        if context.transaction_logger:
                            context.transaction_logger.log_transformed_request(
                                kwargs,
                                context.kwargs,
                                credential_id=cred_context.stable_id,
                                metadata={
                                    "session_id": context.session_id,
                                    "scope_key": context.usage_manager_key,
                                    "classifier": context.classifier,
                                },
                            )

                        # Get provider plugin
                        plugin = self._get_plugin_instance(provider)

                        # Execute request with retries
                        for attempt in range(self._max_retries):
                            try:
                                lib_logger.info(
                                    f"Attempting call with credential {mask_credential(cred)} "
                                    f"(Attempt {attempt + 1}/{self._max_retries})"
                                )
                                # Pre-request callback
                                await self._run_pre_request_callback(context, kwargs)

                                response = await self._execute_provider_request(
                                    provider,
                                    model,
                                    plugin,
                                    credential_secret,
                                    cred_context.stable_id,
                                    kwargs,
                                    context,
                                )
                                trace_response = _redact_context_field_cache_paths(response, context, "response", plugin)
                                self._log_executor_trace(
                                    context,
                                    "raw_provider_response",
                                    trace_response,
                                    direction="response",
                                    stage="provider",
                                    credential_id=cred_context.stable_id,
                                    metadata={"provider": provider, "model": model},
                                )

                                # Success! Extract token usage if available
                                usage_record, cost_breakdown = self._account_for_response_usage(
                                    provider, model, response, context
                                )
                                prompt_tokens = usage_record.prompt_tokens_for_mark_success
                                completion_tokens = usage_record.completion_tokens
                                prompt_tokens_cached = usage_record.cache_read_tokens
                                prompt_tokens_cache_write = usage_record.cache_write_tokens
                                thinking_tokens = usage_record.reasoning_tokens
                                approx_cost = cost_breakdown.total_cost
                                response_headers = self._extract_response_headers(
                                    response
                                )

                                cred_context.mark_success(
                                    response=response,
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    thinking_tokens=thinking_tokens,
                                    prompt_tokens_cache_read=prompt_tokens_cached,
                                    prompt_tokens_cache_write=prompt_tokens_cache_write,
                                    approx_cost=approx_cost,
                                    response_headers=response_headers,
                                )
                                self._record_session_response(context, response)

                                lib_logger.info(
                                    f"Recorded usage from response object for key {mask_credential(cred)}"
                                )

                                # Log response if transaction logging enabled
                                if context.transaction_logger:
                                    try:
                                        response_data = (
                                            response.model_dump()
                                            if hasattr(response, "model_dump")
                                            else response
                                        )
                                        response_data = _redact_context_field_cache_paths(response_data, context, "response", plugin)
                                        context.transaction_logger.log_response(
                                            response_data
                                        )
                                    except Exception as log_err:
                                        lib_logger.debug(
                                            f"Failed to log response: {log_err}"
                                        )

                                normalized_response = self._normalize_response_usage(response, model)
                                trace_normalized_response = _redact_context_field_cache_paths(normalized_response, context, "response", plugin)
                                self._log_executor_trace(
                                    context,
                                    "post_usage_normalization_response",
                                    trace_normalized_response,
                                    direction="response",
                                    stage="final",
                                    credential_id=cred_context.stable_id,
                                    metadata={"provider": provider, "model": model},
                                )
                                return normalized_response

                            except RoutingExecutionError as e:
                                if e.error_type == "configuration_error":
                                    raise
                                last_exception = e
                                action = await self._handle_error_with_context(
                                    e,
                                    cred_context,
                                    model,
                                    provider,
                                    attempt,
                                    error_accumulator,
                                    retry_state,
                                    request_headers,
                                    context,
                                )
                                if action == ErrorAction.RETRY_SAME:
                                    continue
                                elif action == ErrorAction.ROTATE:
                                    break
                                else:
                                    raise
                            except Exception as e:
                                last_exception = e
                                action = await self._handle_error_with_context(
                                    e,
                                    cred_context,
                                    model,
                                    provider,
                                    attempt,
                                    error_accumulator,
                                    retry_state,
                                    request_headers,
                                    context,
                                )

                                if action == ErrorAction.RETRY_SAME:
                                    continue
                                elif action == ErrorAction.ROTATE:
                                    break  # Try next credential
                                else:  # FAIL
                                    raise

                    except PreRequestCallbackError:
                        raise
                    except RoutingExecutionError as exc:
                        if exc.error_type == "configuration_error":
                            raise
                    except Exception:
                        # Let context manager handle cleanup
                        pass

            except NoAvailableKeysError:
                break

        # All credentials exhausted
        error_accumulator.timeout_occurred = time.time() >= deadline
        if last_exception and not error_accumulator.has_errors():
            raise last_exception

        # Return error response
        return error_accumulator.build_client_error_response()

    async def _execute_streaming(
        self,
        context: RequestContext,
    ) -> AsyncGenerator[str, None]:
        """
        Execute streaming request with retry/rotation.

        This is an async generator that yields SSE-formatted strings.

        Args:
            context: RequestContext with all request details

        Yields:
            SSE-formatted strings
        """
        provider = context.provider
        model = context.model
        deadline = context.deadline

        try:
            (
                usage_manager,
                filter_result,
                credentials,
                quota_group,
                request_headers,
            ) = await self._prepare_execution(context)
        except NoAvailableKeysError as exc:
            error_data = {
                "error": {
                    "message": str(exc),
                    "type": "proxy_error",
                }
            }
            for line in self._terminal_stream_error_lines(context, error_data):
                yield line
            return

        error_accumulator = RequestErrorAccumulator()
        error_accumulator.model = model
        error_accumulator.provider = provider

        retry_state = RetryState()
        last_exception: Optional[Exception] = None

        try:
            while time.time() < deadline:
                # Check for untried credentials
                untried = [
                    c for c in credentials if c not in retry_state.tried_credentials
                ]
                if not untried:
                    lib_logger.warning(
                        f"All {len(credentials)} credentials tried for {model}"
                    )
                    break

                # Wait for provider cooldown
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                await self._wait_for_cooldown(provider, deadline)

                # Acquire credential using context manager
                try:
                    availability = await usage_manager.get_availability_stats(
                        model, quota_group
                    )
                    self._log_acquiring_credential(
                        model, len(retry_state.tried_credentials), availability
                    )
                    async with await usage_manager.acquire_credential(
                        model=model,
                        quota_group=quota_group,
                        session_id=context.session_id,
                        session_affinity_key=context.session_affinity_key,
                        candidates=untried,
                        priorities=filter_result.priorities,
                        deadline=deadline,
                    ) as cred_context:
                        cred = cred_context.credential
                        credential_secret = context.credential_secrets.get(cred, cred)
                        retry_state.record_attempt(cred)

                        state = getattr(usage_manager, "states", {}).get(
                            cred_context.stable_id
                        )
                        self._log_acquired_credential(
                            cred, model, state, quota_group, availability, usage_manager
                        )

                        try:
                            # Prepare request kwargs
                            kwargs = await self._prepare_request_kwargs(
                                provider,
                                model,
                                cred,
                                context,
                                credential_id=cred_context.stable_id,
                            )

                            # Log transformed request if it differs from original
                            if context.transaction_logger:
                                context.transaction_logger.log_transformed_request(
                                    kwargs,
                                    context.kwargs,
                                    credential_id=cred_context.stable_id,
                                    metadata={
                                        "session_id": context.session_id,
                                        "scope_key": context.usage_manager_key,
                                        "classifier": context.classifier,
                                    },
                                )

                            # Add stream usage metadata for active providers.
                            if "stream_options" not in kwargs:
                                kwargs["stream_options"] = {}
                            if "include_usage" not in kwargs["stream_options"]:
                                kwargs["stream_options"]["include_usage"] = True

                            # Get provider plugin
                            plugin = self._get_plugin_instance(provider)
                            skip_cost_calculation = bool(
                                plugin
                                and getattr(plugin, "skip_cost_calculation", False)
                            )

                            # Execute request with retries
                            for attempt in range(self._max_retries):
                                last_streamed_chunk: Optional[str] = None

                                try:
                                    lib_logger.info(
                                        f"Attempting stream with credential {mask_credential(cred)} "
                                        f"(Attempt {attempt + 1}/{self._max_retries})"
                                    )
                                    # Pre-request callback
                                    await self._run_pre_request_callback(
                                        context, kwargs
                                    )

                                    target = _current_route_target(context)
                                    execution = target.execution if target else "auto"

                                    # Make the API call
                                    if _should_use_native_streaming(plugin, model, target, execution, provider):
                                        native_context, native_request = self._build_native_provider_context(
                                            provider,
                                            model,
                                            plugin,
                                            credential_secret,
                                            cred_context.stable_id,
                                            context,
                                            target,
                                            raw_request=kwargs,
                                            transport="sse",
                                            stream=True,
                                            return_request=True,
                                        )
                                        self._log_routing_trace(
                                            context,
                                            "routing_native_stream_execution_selected",
                                            _target_trace(target) if target else {"provider": provider, "model": model},
                                            metadata={"protocol": native_context.protocol_name, "operation": native_context.operation},
                                        )
                                        stream = self._get_native_executor().stream(native_request, native_context, NativeHTTPTransport(self._http_client))
                                    elif plugin and plugin.has_custom_logic():
                                        kwargs["credential_identifier"] = credential_secret
                                        self._log_executor_trace(
                                            context,
                                            "provider_execution_request",
                                            kwargs,
                                            direction="request",
                                            stage="provider",
                                            credential_id=cred_context.stable_id,
                                            metadata={"execution": "custom_stream", "provider": provider, "model": model},
                                        )
                                        stream = await plugin.acompletion(
                                            self._http_client, **kwargs
                                        )
                                    else:
                                        kwargs["api_key"] = credential_secret
                                        kwargs["stream"] = True
                                        self._apply_litellm_logger(kwargs)
                                        # Remove internal context before litellm call
                                        kwargs.pop("transaction_context", None)
                                        self._log_executor_trace(
                                            context,
                                            "provider_execution_request",
                                            kwargs,
                                            direction="request",
                                            stage="provider",
                                            credential_id=cred_context.stable_id,
                                            metadata={"execution": "litellm_stream", "provider": provider, "model": model},
                                        )
                                        stream = await litellm.acompletion(**kwargs)

                                    self._log_executor_trace(
                                        context,
                                        "raw_provider_stream_response",
                                        stream,
                                        direction="response",
                                        stage="provider",
                                        credential_id=cred_context.stable_id,
                                        metadata={"provider": provider, "model": model},
                                        snapshot=False,
                                    )

                                    # Hand off to streaming handler with cred_context
                                    # The handler will call mark_success on completion
                                    base_stream = self._streaming_handler.wrap_stream(
                                        stream,
                                        cred,
                                        model,
                                        context.request,
                                        cred_context,
                                        skip_cost_calculation=skip_cost_calculation,
                                        response_callback=lambda response: self._record_session_response(
                                            context, response
                                        ),
                                        transaction_logger=context.transaction_logger,
                                    )

                                    lib_logger.info(
                                        f"Stream connection established for credential {mask_credential(cred)}. "
                                        "Processing response."
                                    )

                                    # Wrap with transaction logging if enabled
                                    if context.transaction_logger:
                                        async for (
                                            chunk
                                        ) in self._transaction_logging_stream_wrapper(
                                            base_stream,
                                            context.transaction_logger,
                                            context.kwargs,
                                            context=context,
                                            plugin=plugin,
                                        ):
                                            last_streamed_chunk = chunk
                                            yield chunk
                                    else:
                                        async for chunk in base_stream:
                                            last_streamed_chunk = chunk
                                            yield chunk
                                    return

                                except StreamedAPIError as e:
                                    last_exception = e
                                    original = getattr(e, "data", e)
                                    classified = classify_error(original, provider)
                                    if _can_start_stream_provider_cooldown(
                                        last_streamed_chunk
                                    ):
                                        await self._maybe_start_provider_cooldown(
                                            provider, classified, context=context
                                        )
                                    log_failure(
                                        api_key=cred,
                                        model=model,
                                        attempt=attempt + 1,
                                        error=e,
                                        request_headers=request_headers,
                                    )
                                    error_accumulator.record_error(
                                        cred, classified, str(original)[:150]
                                    )

                                    # Track consecutive quota failures
                                    if classified.error_type == "quota_exceeded":
                                        retry_state.increment_quota_failures()
                                        if retry_state.consecutive_quota_failures >= 3:
                                            lib_logger.error(
                                                "3 consecutive quota errors in streaming - "
                                                "request may be too large"
                                            )
                                            cred_context.mark_failure(classified)
                                            error_data = {
                                                "error": {
                                                    "message": "Request exceeds quota for all credentials",
                                                    "type": "quota_exhausted",
                                                }
                                            }
                                            for line in self._terminal_stream_error_lines(context, error_data):
                                                yield line
                                            return
                                    else:
                                        retry_state.reset_quota_failures()

                                    if not should_rotate_on_error(classified):
                                        cred_context.mark_failure(classified)
                                        raise

                                    if not can_retry_stream_after_error(
                                        last_streamed_chunk,
                                        self._stream_retry_on_reasoning_only_enabled(),
                                    ):
                                        cred_context.mark_failure(classified)
                                        error_data = {
                                            "error": {
                                                "message": "Upstream stream failed after output began",
                                                "type": classified.error_type,
                                            }
                                        }
                                        for line in self._terminal_stream_error_lines(context, error_data):
                                            yield line
                                        return

                                    small_cooldown_threshold = int(
                                        os.environ.get(
                                            "SMALL_COOLDOWN_RETRY_THRESHOLD",
                                            DEFAULT_SMALL_COOLDOWN_RETRY_THRESHOLD,
                                        )
                                    )
                                    if (
                                        should_retry_same_key(
                                            classified, small_cooldown_threshold
                                        )
                                        and attempt < self._max_retries - 1
                                    ):
                                        wait_time = (
                                            classified.retry_after
                                            if classified.retry_after is not None
                                            else self._get_transient_retry_delay()
                                        )
                                        if await self._sleep_before_transient_action(
                                            wait_time, deadline, "stream retry"
                                        ):
                                            continue

                                    cred_context.mark_failure(classified)
                                    if self._is_transient_error(classified):
                                        await self._sleep_before_transient_action(
                                            self._get_transient_retry_delay(),
                                            deadline,
                                            "credential rotation",
                                        )
                                    break  # Rotate

                                except (RateLimitError, httpx.HTTPStatusError) as e:
                                    last_exception = e
                                    classified = classify_error(e, provider)
                                    if _can_start_stream_provider_cooldown(
                                        last_streamed_chunk
                                    ):
                                        await self._maybe_start_provider_cooldown(
                                            provider, classified, context=context
                                        )
                                    log_failure(
                                        api_key=cred,
                                        model=model,
                                        attempt=attempt + 1,
                                        error=e,
                                        request_headers=request_headers,
                                    )
                                    error_accumulator.record_error(
                                        cred, classified, str(e)[:150]
                                    )

                                    # Track consecutive quota failures
                                    if classified.error_type == "quota_exceeded":
                                        retry_state.increment_quota_failures()
                                        if retry_state.consecutive_quota_failures >= 3:
                                            lib_logger.error(
                                                "3 consecutive quota errors in streaming - "
                                                "request may be too large"
                                            )
                                            cred_context.mark_failure(classified)
                                            error_data = {
                                                "error": {
                                                    "message": "Request exceeds quota for all credentials",
                                                    "type": "quota_exhausted",
                                                }
                                            }
                                            for line in self._terminal_stream_error_lines(context, error_data):
                                                yield line
                                            return
                                    else:
                                        retry_state.reset_quota_failures()

                                    if not should_rotate_on_error(classified):
                                        cred_context.mark_failure(classified)
                                        raise

                                    # Check for small cooldown - retry same key instead of rotating
                                    small_cooldown_threshold = int(
                                        os.environ.get(
                                            "SMALL_COOLDOWN_RETRY_THRESHOLD",
                                            DEFAULT_SMALL_COOLDOWN_RETRY_THRESHOLD,
                                        )
                                    )
                                    if (
                                        classified.retry_after is not None
                                        and 0
                                        < classified.retry_after
                                        < small_cooldown_threshold
                                        and attempt < self._max_retries - 1
                                    ):
                                        remaining = deadline - time.time()
                                        if classified.retry_after <= remaining:
                                            lib_logger.info(
                                                f"Retrying {mask_credential(cred)} in {classified.retry_after:.1f}s "
                                                f"(small cooldown {classified.retry_after}s < {small_cooldown_threshold}s threshold)"
                                            )
                                            await asyncio.sleep(classified.retry_after)
                                            continue  # Retry same key

                                    cred_context.mark_failure(classified)
                                    if self._is_transient_error(classified):
                                        await self._sleep_before_transient_action(
                                            self._get_transient_retry_delay(),
                                            deadline,
                                            "credential rotation",
                                        )
                                    break  # Rotate

                                except (
                                    APIConnectionError,
                                    InternalServerError,
                                    ServiceUnavailableError,
                                ) as e:
                                    last_exception = e
                                    classified = classify_error(e, provider)
                                    if _can_start_stream_provider_cooldown(
                                        last_streamed_chunk
                                    ):
                                        await self._maybe_start_provider_cooldown(
                                            provider, classified, context=context
                                        )
                                    log_failure(
                                        api_key=cred,
                                        model=model,
                                        attempt=attempt + 1,
                                        error=e,
                                        request_headers=request_headers,
                                    )

                                    if attempt >= self._max_retries - 1:
                                        error_accumulator.record_error(
                                            cred, classified, str(e)[:150]
                                        )
                                        cred_context.mark_failure(classified)
                                        await self._sleep_before_transient_action(
                                            self._get_transient_retry_delay(),
                                            deadline,
                                            "credential rotation",
                                        )
                                        break  # Rotate

                                    # Calculate wait time
                                    wait_time = (
                                        classified.retry_after
                                        if classified.retry_after is not None
                                        else self._get_transient_retry_delay()
                                    )
                                    if not await self._sleep_before_transient_action(
                                        wait_time, deadline, "stream retry"
                                    ):
                                        break  # No time to wait

                                    continue  # Retry

                                except RoutingExecutionError as e:
                                    if e.error_type == "configuration_error":
                                        raise
                                    last_exception = e
                                    classified = classify_error(e, provider)
                                    if _can_start_stream_provider_cooldown(last_streamed_chunk):
                                        await self._maybe_start_provider_cooldown(provider, classified, context=context)
                                    log_failure(api_key=cred, model=model, attempt=attempt + 1, error=e, request_headers=request_headers)
                                    error_accumulator.record_error(cred, classified, str(e)[:150])
                                    if not should_rotate_on_error(classified):
                                        cred_context.mark_failure(classified)
                                        raise
                                    cred_context.mark_failure(classified)
                                    break

                                except Exception as e:
                                    last_exception = e
                                    classified = classify_error(e, provider)
                                    if _can_start_stream_provider_cooldown(
                                        last_streamed_chunk
                                    ):
                                        await self._maybe_start_provider_cooldown(
                                            provider, classified, context=context
                                        )
                                    log_failure(
                                        api_key=cred,
                                        model=model,
                                        attempt=attempt + 1,
                                        error=e,
                                        request_headers=request_headers,
                                    )
                                    error_accumulator.record_error(
                                        cred, classified, str(e)[:150]
                                    )

                                    if not should_rotate_on_error(classified):
                                        cred_context.mark_failure(classified)
                                        raise

                                    small_cooldown_threshold = int(
                                        os.environ.get(
                                            "SMALL_COOLDOWN_RETRY_THRESHOLD",
                                            DEFAULT_SMALL_COOLDOWN_RETRY_THRESHOLD,
                                        )
                                    )
                                    if (
                                        should_retry_same_key(
                                            classified, small_cooldown_threshold
                                        )
                                        and attempt < self._max_retries - 1
                                    ):
                                        wait_time = (
                                            classified.retry_after
                                            if classified.retry_after is not None
                                            else self._get_transient_retry_delay()
                                        )
                                        if await self._sleep_before_transient_action(
                                            wait_time, deadline, "stream retry"
                                        ):
                                            continue

                                    cred_context.mark_failure(classified)
                                    if self._is_transient_error(classified):
                                        await self._sleep_before_transient_action(
                                            self._get_transient_retry_delay(),
                                            deadline,
                                            "credential rotation",
                                        )
                                    break  # Rotate

                        except PreRequestCallbackError:
                            raise
                        except RoutingExecutionError as exc:
                            if exc.error_type == "configuration_error":
                                raise
                        except Exception:
                            # Let context manager handle cleanup
                            pass

                except NoAvailableKeysError:
                    break

            # All credentials exhausted or timeout
            error_accumulator.timeout_occurred = time.time() >= deadline
            error_data = error_accumulator.build_client_error_response()
            for line in self._terminal_stream_error_lines(context, error_data):
                yield line

        except NoAvailableKeysError as e:
            lib_logger.error(f"No keys available: {e}")
            error_data = {"error": {"message": str(e), "type": "proxy_busy"}}
            for line in self._terminal_stream_error_lines(context, error_data):
                yield line

        except Exception as e:
            lib_logger.error(f"Unhandled exception in streaming: {e}", exc_info=True)
            classified = classify_error(e, context.provider)
            error_data = {"error": {"message": "Streaming request failed", "type": classified.error_type, "details": {"status_code": classified.status_code}}}
            for line in self._terminal_stream_error_lines(context, error_data):
                yield line

    def _apply_litellm_provider_params(
        self, provider: str, kwargs: Dict[str, Any]
    ) -> None:
        """Merge provider-specific LiteLLM parameters into request kwargs."""
        params = self._litellm_provider_params.get(provider)
        if not params:
            return
        kwargs["litellm_params"] = {
            **params,
            **kwargs.get("litellm_params", {}),
        }

    def _apply_litellm_logger(self, kwargs: Dict[str, Any]) -> None:
        """Attach LiteLLM logger callback if configured."""
        if self._litellm_logger_fn and "logger_fn" not in kwargs:
            kwargs["logger_fn"] = self._litellm_logger_fn

    def _extract_response_headers(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract response headers from LiteLLM response objects."""
        if hasattr(response, "response") and response.response is not None:
            headers = getattr(response.response, "headers", None)
            if headers is not None:
                return dict(headers)
        headers = getattr(response, "headers", None)
        if headers is not None:
            return dict(headers)
        return None

    async def _wait_for_cooldown(
        self,
        provider: str,
        deadline: float,
    ) -> None:
        """
        Wait for provider-level cooldown to end.

        Args:
            provider: Provider name
            deadline: Request deadline
        """
        if not self._cooldown:
            return

        remaining = await self._cooldown.get_remaining_cooldown(provider)
        if remaining > 0:
            budget = deadline - time.time()
            if remaining > budget:
                lib_logger.warning(
                    f"Provider {provider} cooldown ({remaining:.1f}s) exceeds budget ({budget:.1f}s)"
                )
                return  # Will fail on no keys available
            lib_logger.info(f"Waiting {remaining:.1f}s for {provider} cooldown")
            await asyncio.sleep(remaining)

    async def _handle_error_with_context(
        self,
        error: Exception,
        cred_context: Any,  # CredentialContext
        model: str,
        provider: str,
        attempt: int,
        error_accumulator: RequestErrorAccumulator,
        retry_state: RetryState,
        request_headers: Dict[str, Any],
        context: Optional[RequestContext] = None,
    ) -> str:
        """
        Handle an error and determine next action.

        Args:
            error: The caught exception
            cred_context: CredentialContext for marking failure
            model: Model name
            provider: Provider name
            attempt: Current attempt number
            error_accumulator: Error tracking
            retry_state: Retry state tracking

        Returns:
            ErrorAction indicating what to do next
        """
        classified = classify_error(error, provider)
        error_message = str(error)[:150]
        credential = cred_context.credential

        log_failure(
            api_key=credential,
            model=model,
            attempt=attempt + 1,
            error=error,
            request_headers=request_headers,
        )

        # Check for quota errors
        if classified.error_type == "quota_exceeded":
            retry_state.increment_quota_failures()
            if retry_state.consecutive_quota_failures >= 3:
                # Likely request is too large
                lib_logger.error(
                    f"3 consecutive quota errors - request may be too large"
                )
                error_accumulator.record_error(credential, classified, error_message)
                cred_context.mark_failure(classified)
                return ErrorAction.FAIL
        else:
            retry_state.reset_quota_failures()

        # Check if should rotate
        if not should_rotate_on_error(classified):
            error_accumulator.record_error(credential, classified, error_message)
            cred_context.mark_failure(classified)
            return ErrorAction.FAIL

        await self._maybe_start_provider_cooldown(provider, classified, context=context)

        # Check if should retry same key (including small cooldown auto-retry)
        small_cooldown_threshold = int(
            os.environ.get(
                "SMALL_COOLDOWN_RETRY_THRESHOLD", DEFAULT_SMALL_COOLDOWN_RETRY_THRESHOLD
            )
        )
        is_small_cooldown = (
            classified.retry_after is not None
            and 0 < classified.retry_after < small_cooldown_threshold
        )

        if (
            should_retry_same_key(classified, small_cooldown_threshold)
            and attempt < self._max_retries - 1
        ):
            wait_time = (
                classified.retry_after
                if classified.retry_after is not None
                else self._get_transient_retry_delay()
            )
            retry_reason = (
                f" (small cooldown {classified.retry_after}s < {small_cooldown_threshold}s threshold)"
                if is_small_cooldown
                else ""
            )
            lib_logger.info(
                f"Retrying {mask_credential(credential)} in {wait_time:.1f}s{retry_reason}"
            )
            await asyncio.sleep(wait_time)
            return ErrorAction.RETRY_SAME

        # Record error and rotate
        error_accumulator.record_error(credential, classified, error_message)
        cred_context.mark_failure(classified)
        if self._is_transient_error(classified):
            wait_time = self._get_transient_retry_delay()
            lib_logger.info(
                f"Waiting {wait_time:.1f}s before rotating from {mask_credential(credential)}"
            )
            await asyncio.sleep(wait_time)
        lib_logger.info(
            f"Rotating from {mask_credential(credential)} after {classified.error_type}"
        )
        return ErrorAction.ROTATE

    async def _maybe_start_provider_cooldown(
        self,
        provider: str,
        classified: ClassifiedError,
        *,
        context: Optional[RequestContext],
    ) -> None:
        """Start provider-wide cooldown for large provider-level throttles.

        This is intentionally conservative: small retry-after values stay on the
        same credential path, and quota cooldown is disabled by default because
        most quota errors are per credential or account.
        """

        if not self._cooldown:
            return
        small_cooldown_threshold = int(
            os.environ.get(
                "SMALL_COOLDOWN_RETRY_THRESHOLD", DEFAULT_SMALL_COOLDOWN_RETRY_THRESHOLD
            )
        )
        min_seconds, default_seconds, cooldown_on_quota = provider_cooldown_env()
        decision = decide_provider_cooldown(
            classified,
            small_cooldown_threshold=small_cooldown_threshold,
            provider_cooldown_min_seconds=min_seconds,
            default_duration=default_seconds,
            cooldown_on_quota=cooldown_on_quota,
        )
        if not decision.should_start:
            self._log_provider_cooldown_trace(
                context,
                "provider_cooldown_skipped",
                provider,
                classified,
                decision.duration,
                decision.reason,
            )
            return
        try:
            await self._cooldown.start_cooldown(provider, decision.duration)
            self._log_provider_cooldown_trace(
                context,
                "provider_cooldown_started",
                provider,
                classified,
                decision.duration,
                decision.reason,
            )
        except Exception as exc:
            lib_logger.debug("Failed to start provider cooldown for %s: %s", provider, exc)

    @staticmethod
    def _log_provider_cooldown_trace(
        context: Optional[RequestContext],
        pass_name: str,
        provider: str,
        classified: ClassifiedError,
        duration: int,
        reason: str,
    ) -> None:
        if not context or not context.transaction_logger:
            return
        context.transaction_logger.log_transform_pass(
            pass_name,
            {"provider": provider, "error_type": classified.error_type, "duration": duration},
            direction="metadata",
            stage="retry",
            metadata={
                "provider": provider,
                "duration": duration,
                "error_type": classified.error_type,
                "retry_after_present": classified.retry_after is not None,
                "reason": reason,
            },
            snapshot=False,
        )

    def _record_session_response(self, context: RequestContext, response: Any) -> None:
        """Let the tracker learn anchors emitted by a successful response.

        Response anchors are additive evidence for the next request. Failures are
        ignored because a failed response should not mutate session identity.
        """
        tracker = context.session_tracker
        if not tracker or not context.session_id:
            return
        try:
            tracker.record_response(
                context.session_id,
                provider=context.provider,
                model=context.model,
                scope_key=context.usage_manager_key,
                tracking_namespace=context.session_tracking_namespace,
                response=response,
            )
        except Exception as exc:
            lib_logger.debug("Session response tracking failed: %s", exc)

    async def _ensure_initialized(
        self,
        usage_manager: "UsageManager",
        context: RequestContext,
        filter_result: "FilterResult",
    ) -> None:
        is_scoped_manager = bool(
            context.usage_manager_key and context.usage_manager_key != context.provider
        )
        if usage_manager.initialized and not is_scoped_manager:
            return
        await usage_manager.initialize(
            context.credentials,
            priorities=filter_result.priorities,
            tiers=filter_result.tier_names,
        )

    async def _validate_request(
        self,
        provider: str,
        model: str,
        kwargs: Dict[str, Any],
    ) -> None:
        plugin = self._get_plugin_instance(provider)
        if not plugin or not hasattr(plugin, "validate_request"):
            return

        result = plugin.validate_request(kwargs, model)
        if asyncio.iscoroutine(result):
            result = await result
        if result is False:
            raise ValueError(f"Request validation failed for {provider}/{model}")
        if isinstance(result, str):
            raise ValueError(result)

    def _account_for_response_usage(
        self,
        provider: str,
        model: str,
        response: Any,
        context: RequestContext,
    ) -> tuple[UsageRecord, CostBreakdown]:
        """Normalize usage and advisory cost for one successful response."""

        usage_record = extract_usage_record(
            response,
            provider=provider,
            model=model,
            source="executor_response",
        )
        plugin = self._get_plugin_instance(provider)
        cost_breakdown = CostCalculator(provider_plugin=plugin).calculate(
            usage_record,
            model=model,
            response=response,
        )
        self._trace_usage_accounting(context, usage_record, cost_breakdown)
        return usage_record, cost_breakdown

    @staticmethod
    def _trace_usage_accounting(
        context: RequestContext,
        usage_record: UsageRecord,
        cost_breakdown: CostBreakdown,
    ) -> None:
        """Record normalized usage/cost trace data without affecting requests."""

        if not context.transaction_logger:
            return
        context.transaction_logger.log_transform_pass(
            "usage_accounting_summary",
            {"usage": usage_record.to_dict(), "cost": cost_breakdown.to_dict()},
            direction="metadata",
            stage="final",
            metadata={
                "provider": usage_record.provider,
                "model": usage_record.model,
                "source": usage_record.source,
                "pricing_source": cost_breakdown.pricing_source,
            },
            snapshot=False,
        )

    @staticmethod
    def _normalize_response_usage(response: Any, model: str) -> Any:
        """
        Normalize usage fields on the non-streaming response object.

        Delegates to normalize_usage_for_response which handles both
        dicts (streaming) and pydantic objects (non-streaming).
        Internal tracking values from UsageRecord accounting are unaffected.
        """
        if hasattr(response, "usage") and response.usage:
            normalize_usage_for_response(response.usage, model)
        return response

    async def _transaction_logging_stream_wrapper(
        self,
        stream: AsyncGenerator[str, None],
        transaction_logger: TransactionLogger,
        request_kwargs: Dict[str, Any],
        *,
        context: Optional[RequestContext] = None,
        plugin: Any = None,
    ) -> AsyncGenerator[str, None]:
        """
        Wrap a stream to log chunks and final response to TransactionLogger.

        Yields all chunks unchanged while accumulating them for final logging.

        Args:
            stream: The SSE stream from wrap_stream
            transaction_logger: TransactionLogger instance
            request_kwargs: Original request kwargs for context

        Yields:
            SSE-formatted strings unchanged
        """
        chunks = []

        async for sse_line in stream:
            trace_sse_line = _redact_stream_sse_for_trace(sse_line, context, plugin)
            transaction_logger.log_transform_pass(
                "raw_stream_chunk",
                trace_sse_line,
                direction="stream",
                stage="client",
                transport="sse",
                snapshot=False,
            )
            if sse_line.startswith("data: [DONE]"):
                transaction_logger.log_transform_pass(
                    "stream_done_event",
                    {"raw": trace_sse_line},
                    direction="stream",
                    stage="final",
                    transport="sse",
                    snapshot=False,
                )
            yield sse_line

            # Parse and accumulate for final logging
            if sse_line.startswith("data: ") and not sse_line.startswith(
                "data: [DONE]"
            ):
                try:
                    content = sse_line[6:].strip()
                    if content:
                        chunk_data = json.loads(content)
                        chunks.append(chunk_data)
                        trace_chunk_data = _redact_context_field_cache_paths(chunk_data, context, "stream", plugin) if context else chunk_data
                        transaction_logger.log_stream_chunk(trace_chunk_data)
                        if isinstance(chunk_data, dict) and chunk_data.get("error") is not None:
                            transaction_logger.log_transform_pass(
                                "stream_error_event",
                                trace_chunk_data,
                                direction="stream",
                                stage="client",
                                transport="sse",
                                snapshot=False,
                            )
                except json.JSONDecodeError:
                    lib_logger.debug(
                        f"Failed to parse chunk for logging: {sse_line[:100]}"
                    )

        # Log assembled final response
        if chunks:
            try:
                final_response = TransactionLogger.assemble_streaming_response(chunks)
                trace_final_response = _redact_context_field_cache_paths(final_response, context, "stream", plugin) if context else final_response
                transaction_logger.log_transform_pass(
                    "assembled_stream_response",
                    trace_final_response,
                    direction="response",
                    stage="client",
                    transport="sse",
                )
                transaction_logger.log_response(trace_final_response)
            except Exception as e:
                lib_logger.debug(
                    f"Failed to assemble/log final streaming response: {e}"
                )


def _target_trace(target: RouteTarget) -> Dict[str, Any]:
    """Return non-secret route target metadata for transaction traces."""

    return {
        "name": target.name,
        "provider": target.provider,
        "model": target.prefixed_model,
        "execution": target.execution,
        "protocol": target.protocol,
    }


def _current_route_target(context: RequestContext) -> Optional[RouteTarget]:
    """Return the currently selected route target from context metadata."""

    targets = tuple(context.routing_targets or ())
    if not targets:
        return None
    if context.routing_target_index < 0 or context.routing_target_index >= len(targets):
        return None
    return targets[context.routing_target_index]


def _provider_native_protocol(plugin: Any, model: str, target: Optional[RouteTarget]) -> Optional[str]:
    """Resolve native protocol from target override or provider declaration."""

    if target and target.protocol:
        return target.protocol
    if plugin and hasattr(plugin, "get_protocol_name"):
        return plugin.get_protocol_name(model)
    return None


def _strip_provider_prefix(model: str) -> str:
    """Return model without the proxy-facing provider prefix."""

    return model.split("/", 1)[1] if "/" in model else model


_NATIVE_REQUEST_DROP_KEYS = {
    "api_base",
    "api_key",
    "api_type",
    "api_version",
    "base_url",
    "custom_llm_provider",
    "drop_params",
    "logger_fn",
    "litellm_call_id",
    "mock_response",
    "organization",
    "project",
    "transaction_context",
}


def _native_request_payload(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return provider-visible kwargs for native protocol parsing.

    Full executor calls prepare kwargs for LiteLLM before execution-mode routing.
    Native providers must not receive LiteLLM-only routing, logging, or transport
    controls because protocol adapters preserve unknown fields intentionally.
    """

    return {key: deepcopy(value) for key, value in kwargs.items() if key not in _NATIVE_REQUEST_DROP_KEYS and not key.startswith("litellm_")}


def _provider_supports_native_streaming(plugin: Any, model: str) -> bool:
    """Return whether a provider declares native streaming support."""

    support = getattr(plugin, "supports_native_streaming", None)
    if not callable(support):
        return False
    operation = "chat"
    resolver = getattr(plugin, "get_native_operation", None)
    if callable(resolver):
        try:
            operation = resolver(model, {"model": model, "stream": True}, stream=True)
        except TypeError:
            operation = resolver(model)
    try:
        return bool(support(model=model, operation=operation))
    except TypeError:
        try:
            return bool(support(model))
        except TypeError:
            return bool(support())


def _should_use_native_streaming(plugin: Any, model: str, target: Optional[RouteTarget], execution: str, provider: str) -> bool:
    """Return whether streaming may use the native executor.

    Explicit `@native` routing is still constrained by provider capability.
    Falling through to native streaming when a provider has not opted in is unsafe
    because the generic stream wrapper currently expects LiteLLM-shaped chunks.
    """

    if execution == "native":
        if _provider_supports_native_streaming(plugin, model):
            return True
        raise RoutingExecutionError(
            f"Provider {provider} does not support native streaming for {model}",
            error_type="configuration_error",
        )
    return bool(execution == "auto" and plugin and _provider_native_protocol(plugin, model, target) and _provider_supports_native_streaming(plugin, model))


def _redact_context_field_cache_paths(payload: Any, context: RequestContext, direction: str, plugin: Any) -> Any:
    """Redact configured field-cache paths before executor-level traces.

    Native provider execution already redacts its internal trace passes. The
    client executor also logs returned responses, so it must apply the same
    rule-aware redaction there without mutating the client-facing response.
    """

    if direction not in {"response", "stream"} or not plugin or not _provider_native_protocol(plugin, context.model, _current_route_target(context)):
        return payload
    try:
        rules = _merged_field_cache_rules(context.provider, context.model, plugin)
    except RoutingExecutionError:
        raise
    except Exception:
        return payload
    if not rules:
        return payload
    redacted = deepcopy(payload)
    for rule in rules:
        if direction == "response" and getattr(rule, "source", None) not in {"response", "unified_response"}:
            continue
        if direction == "stream" and getattr(rule, "source", None) not in {"stream_event", "unified_stream_event", "response", "unified_response"}:
            continue
        for path in _trace_redaction_paths((rule.path,), direction=direction):
            try:
                tokens = parse_path(path)
                _redact_trace_path(redacted, tokens)
                _redact_trace_leaf_key(redacted, tokens)
            except (FieldCachePathError, TypeError, ValueError):
                continue
    return redacted


def _trace_redaction_paths(paths: tuple[str, ...] | list[str], *, direction: str) -> list[str]:
    """Return configured paths plus raw-stream envelope fallbacks for traces."""

    expanded: list[str] = []
    for path in paths:
        expanded.append(path)
        if direction == "stream" and path.startswith("raw."):
            expanded.append(path[4:])
    return expanded


def _redact_stream_sse_for_trace(sse_line: str, context: Optional[RequestContext], plugin: Any) -> str:
    """Return a trace-only SSE line with configured native cache paths redacted."""

    if not context or not isinstance(sse_line, str) or not sse_line.startswith("data: ") or sse_line.startswith("data: [DONE]"):
        return sse_line
    try:
        payload = json.loads(sse_line[6:].strip())
    except json.JSONDecodeError:
        return sse_line
    redacted = _redact_context_field_cache_paths(payload, context, "stream", plugin)
    if redacted is payload:
        return sse_line
    return f"data: {json.dumps(redacted, separators=(',', ':'))}\n\n"


def _redact_trace_path(value: Any, tokens: tuple[PathToken, ...]) -> None:
    if not tokens:
        return
    token = tokens[0]
    rest = tokens[1:]
    if token.kind == "key":
        if isinstance(value, dict) and token.value in value:
            if rest:
                _redact_trace_path(value[token.value], rest)
            else:
                value[token.value] = REDACTED
        return
    if token.kind == "index":
        if isinstance(value, list) and value:
            index = int(token.value)
            if -len(value) <= index < len(value):
                if rest:
                    _redact_trace_path(value[index], rest)
                else:
                    value[index] = REDACTED
        return
    if token.kind == "wildcard":
        if isinstance(value, dict):
            for key in list(value.keys()):
                if rest:
                    _redact_trace_path(value[key], rest)
                else:
                    value[key] = REDACTED
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if rest:
                    _redact_trace_path(item, rest)
                else:
                    value[index] = REDACTED


def _redact_trace_leaf_key(value: Any, tokens: tuple[PathToken, ...]) -> None:
    """Redact terminal configured cache keys across duplicated trace envelopes."""

    leaf = next((token.value for token in reversed(tokens) if token.kind == "key"), None)
    if not leaf:
        return
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key == leaf:
                value[key] = REDACTED
            else:
                _redact_trace_leaf_key(item, tokens)
    elif isinstance(value, list):
        for item in value:
            _redact_trace_leaf_key(item, tokens)


def _merged_field_cache_rules(provider: str, model: str, plugin: Any) -> tuple[Any, ...]:
    """Merge provider-declared and JSON-configured field-cache rules.

    Provider declarations are the safe default. Optional JSON config can add or
    replace rules by name so operators can tune protocol-state preservation per
    provider/model without editing provider code. The import is local to keep the
    experimental config layer out of executor module initialization.
    """

    declared = list(plugin.get_field_cache_rules(model) if plugin and hasattr(plugin, "get_field_cache_rules") else ())
    try:
        from ..config.experimental import load_experimental_config, parse_field_cache_rules

        configured = list(parse_field_cache_rules(load_experimental_config(), provider, model))
    except Exception as exc:
        raise RoutingExecutionError(f"Invalid field-cache configuration for {provider}/{model}", error_type="configuration_error") from exc
    if not configured:
        return tuple(declared)
    merged: dict[str, Any] = {getattr(rule, "name", str(index)): rule for index, rule in enumerate(declared)}
    order = [getattr(rule, "name", str(index)) for index, rule in enumerate(declared)]
    for rule in configured:
        name = getattr(rule, "name", "")
        if name and name not in merged:
            order.append(name)
        if name:
            merged[name] = rule
    return tuple(merged[name] for name in order if name in merged)


def _target_scope_value(target: RouteTarget, key: str, default: Any) -> Any:
    """Read request-scope metadata attached by routing resolution."""

    scope = target.metadata.get("request_scope") if isinstance(target.metadata, dict) else None
    if isinstance(scope, dict) and key in scope:
        return scope[key]
    return default


def _route_error_type(error: BaseException, provider: Optional[str] = None) -> str:
    """Map an exception to a fallback-policy error type."""

    if isinstance(error, asyncio.CancelledError):
        return "cancelled"
    explicit = getattr(error, "error_type", None)
    if explicit:
        return normalize_route_error_type(str(explicit))
    classified = classify_error(error, provider)
    return normalize_route_error_type(classified.error_type)


def _route_error_type_from_response(response: Any) -> Optional[str]:
    """Infer retryability from the proxy's structured error response."""

    if not isinstance(response, dict) or not isinstance(response.get("error"), dict):
        return None
    error = response["error"]
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    candidates = _structured_error_candidates(error, details)
    hard_stop = _first_route_error_candidate(candidates, _HARD_STOP_ROUTE_ERRORS)
    if hard_stop:
        return hard_stop
    retryable = _first_route_error_candidate(candidates, _RETRYABLE_ROUTE_ERRORS)
    if retryable:
        return retryable
    normal_summary = str(details.get("normal_error_summary", "")).lower()
    if any(token in normal_summary for token in ("authentication", "forbidden", "invalid_request", "context_window", "credential_reauth", "configuration_error")):
        return _summary_hard_stop_type(normal_summary)
    if any(token in normal_summary for token in ("rate_limit", "quota", "capacity")):
        return "quota_exceeded" if "quota" in normal_summary else "rate_limit"
    if any(token in normal_summary for token in ("server_error", "api_connection", "transient")):
        return "api_connection" if "api_connection" in normal_summary else "server_error"
    error_type = normalize_route_error_type(str(error.get("type", "")))
    if error_type in {"proxy_timeout", "proxy_all_credentials_exhausted"}:
        return "rate_limit"
    return None


def _route_status_code_from_response(response: Any) -> Optional[int]:
    """Return a structured status code without reading raw provider text."""

    if not isinstance(response, dict) or not isinstance(response.get("error"), dict):
        return None
    error = response["error"]
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    for candidate in (details.get("status_code"), details.get("status"), error.get("status_code"), error.get("status"), error.get("code")):
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue
    return None


def _target_failure_summary(target: RouteTarget, error_type: str, *, status_code: Optional[int] = None) -> Dict[str, Any]:
    """Return a client-safe fallback target failure summary."""

    summary = {
        "target": target.name,
        "provider": target.provider,
        "model": target.prefixed_model,
        "execution": target.execution,
        "error_type": normalize_route_error_type(error_type),
        # Provider error text can contain raw upstream payload fragments or
        # credential-like identifiers. Keep the cross-target summary structural;
        # detailed per-credential errors remain in the existing sanitized error
        # accumulator.
        "message": "",
    }
    if status_code is not None:
        summary["status_code"] = status_code
    return summary


def _group_streaming_policy(group: Any) -> str:
    """Return the active streaming fallback policy for trace and decisions."""

    return str(getattr(group, "streaming_policy", "pre_output_only") or "pre_output_only")


def _streaming_policy_allows_fallback(group: Any) -> bool:
    """Return whether a group permits pre-output streaming fallback."""

    return _group_streaming_policy(group) != "never"


_HARD_STOP_ROUTE_ERRORS = {
    "authentication",
    "forbidden",
    "invalid_request",
    "context_window_exceeded",
    "credential_reauth_needed",
    "pre_request_callback_error",
    "cancelled",
    "configuration_error",
}
_RETRYABLE_ROUTE_ERRORS = {"rate_limit", "quota_exceeded", "server_error", "api_connection", "unsupported_operation"}


def _structured_error_candidates(error: Dict[str, Any], details: Dict[str, Any]) -> List[str]:
    """Return normalized structured error hints before reading free-form text."""

    values: List[Any] = [
        error.get("type"),
        error.get("code"),
        error.get("status"),
        details.get("classification"),
        details.get("error_type"),
        details.get("status"),
    ]
    for abnormal in details.get("abnormal_errors") or []:
        if isinstance(abnormal, dict):
            values.append(abnormal.get("error_type"))
            values.append(abnormal.get("status_code"))
    status_code = _route_status_code_from_response({"error": error})
    if status_code == 400:
        values.append("invalid_request")
    elif status_code == 401:
        values.append("authentication")
    elif status_code == 403:
        values.append("forbidden")
    elif status_code == 429:
        values.append("rate_limit")
    elif status_code is not None and status_code >= 500:
        values.append("server_error")
    return [normalize_route_error_type(str(value)) for value in values if value not in (None, "")]


def _first_route_error_candidate(candidates: List[str], allowed: set[str]) -> Optional[str]:
    """Return the first structured candidate in an allowed policy set."""

    for candidate in candidates:
        if candidate in allowed:
            return candidate
    return None


def _summary_hard_stop_type(summary: str) -> str:
    """Map legacy summary text to a hard-stop category without using raw text."""

    if "credential_reauth" in summary:
        return "credential_reauth_needed"
    if "context_window" in summary:
        return "context_window_exceeded"
    if "configuration_error" in summary or "config" in summary:
        return "configuration_error"
    if "forbidden" in summary:
        return "forbidden"
    if "authentication" in summary:
        return "authentication"
    return "invalid_request"


def _with_fallback_summary(response: Any, target_failures: List[Dict[str, Any]]) -> Any:
    """Attach fallback target summaries to a structured error response."""

    if not target_failures or not isinstance(response, dict) or not isinstance(response.get("error"), dict):
        return response
    details = response["error"].setdefault("details", {})
    if isinstance(details, dict):
        details["fallback_targets"] = list(target_failures)
    return response


def _stream_chunk_is_visible_output(chunk: str) -> bool:
    """Return whether a stream chunk should block cross-target fallback.

    Only client-visible model output should lock the route. Empty frames, DONE
    sentinels, and structured error frames are not considered visible output, so
    a provider that fails before producing content can still fall through to the
    next ordered target.
    """

    return is_visible_stream_output(chunk)


def _stream_chunk_error_type(chunk: str) -> Optional[str]:
    """Return a route error type for terminal stream error frames.

    Per-target stream executors can emit a structured error SSE and `[DONE]`
    instead of raising. Fallback wrappers must treat those frames as target
    failures before visible output, while still forwarding them if no fallback is
    available.
    """

    payload = _stream_chunk_payload(chunk)
    if not isinstance(payload, dict):
        return None
    event_type = normalize_route_error_type(str(payload.get("event_type") or payload.get("type") or ""))
    if event_type == "error":
        error = payload.get("error") if isinstance(payload.get("error"), dict) else payload
        return _route_error_type_from_response({"error": error}) or "server_error"
    if event_type == "response.failed":
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {"type": "server_error"}
        return _route_error_type_from_response({"error": error}) or "server_error"
    if isinstance(payload.get("error"), dict):
        return _route_error_type_from_response({"error": payload["error"]}) or "server_error"
    return None


def _stream_chunk_payload(chunk: str) -> Optional[Dict[str, Any]]:
    """Parse a minimal SSE payload for routing decisions only."""

    text = str(chunk or "").strip()
    if not text:
        return None
    event_type = None
    data_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            continue
        if stripped.startswith("event:"):
            event_type = stripped[6:].strip()
            continue
        if stripped.startswith("data:"):
            data_lines.append(stripped[5:].strip())
    if not data_lines:
        return {"event_type": event_type} if event_type else None
    data = "\n".join(data_lines).strip()
    if not data or data == "[DONE]":
        return None
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    if event_type and "event_type" not in parsed:
        parsed["event_type"] = event_type
    return parsed


def _can_start_stream_provider_cooldown(last_streamed_chunk: Optional[str]) -> bool:
    """Return whether a streaming failure occurred before visible output."""

    return last_streamed_chunk is None or not _stream_chunk_is_visible_output(last_streamed_chunk)
