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
import re
import time
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
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

from ..core.types import RequestContext, ErrorAction, FilterResult
from ..core.errors import (
    NoAvailableKeysError,
    PreRequestCallbackError,
    StreamedAPIError,
    TerminalRequestError,
    ProxyExhaustionError,
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
    DEFAULT_RATE_LIMIT_MAX_RETRY_AFTER,
    ENV_PREFIX_MAX_RETRIES,
)
from ..request_sanitizer import sanitize_request_payload
from ..transaction_logger import TransactionLogger
from ..failure_logger import log_failure
from ..core.utils import normalize_usage_for_response

from .types import RetryState
from .filters import CredentialFilter
from .transforms import ProviderTransforms
from .streaming import StreamingHandler
from .stream_retry_policy import can_retry_stream_after_error

if TYPE_CHECKING:
    from ..error_handler import ClassifiedError
    from ..usage import UsageManager

lib_logger = logging.getLogger("rotator_library")


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
        client_pool: Optional[Any] = None,
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
            client_pool: Optional ProxiedClientPool instance
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
        self._client_pool = client_pool
        # Per-provider retry overrides (cached on first lookup)
        self._provider_max_retries: Dict[str, int] = {}
        self._provider_retries_loaded = False
        # StreamingHandler no longer needs usage_manager - we pass cred_context directly
        self._streaming_handler = StreamingHandler()

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

    def _is_transient_error(self, classified: "ClassifiedError") -> bool:
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

    async def _resolve_http_client(
        self, provider: str, credential: str, stable_id: str
    ) -> httpx.AsyncClient:
        """Resolve the httpx client for a request, using proxy pool if available."""
        if self._client_pool:
            return await self._client_pool.get_client(provider, credential, stable_id)
        return self._http_client

    async def _resolve_litellm_client(
        self, provider: str, credential: str, stable_id: str,
        base_url: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[Any]:
        """Build a litellm-compatible client backed by the proxy pool.

        Many providers are routed through an OpenAI-compatible endpoint via
        ``ProviderConfig.convert_for_litellm``, so litellm's OpenAI code
        path expects an ``openai.AsyncOpenAI`` instance.  We create one
        whose underlying ``httpx.AsyncClient`` routes through our SOCKS5 /
        HTTP proxy.

        Args:
            base_url: The API base URL.  Required for the injected
                ``openai.AsyncOpenAI`` client — without it the SDK would
                default to ``api.openai.com``.  For native litellm providers
                (e.g. gemini/) where no ``API_BASE_*`` is configured, this
                will be ``None`` and the method returns ``None``; litellm
                then uses its own handler (without SOCKS proxy).
            extra_headers: Additional default headers to set on the OpenAI
                client (e.g. User-Agent, X-Title).  When we inject our own
                ``openai.AsyncOpenAI`` client, litellm's ``extra_headers``
                kwarg is bypassed, so these must be baked in here.

        Returns None when no proxy applies or base_url is missing.
        """
        if not self._client_pool:
            return None
        spec = self._client_pool.config.resolve(provider, credential, stable_id)
        if spec is None:
            return None
        if not base_url:
            lib_logger.debug(
                f"Proxy configured for {provider}/{stable_id} but no api_base "
                f"available — skipping custom client (litellm will use its "
                f"native handler without SOCKS proxy)"
            )
            return None
        import openai

        proxy_url = spec.url
        proxied_transport = httpx.AsyncHTTPTransport(proxy=proxy_url)
        proxied_httpx = httpx.AsyncClient(
            transport=proxied_transport,
            timeout=httpx.Timeout(300.0, connect=10.0),
            follow_redirects=True,
        )
        return openai.AsyncOpenAI(
            api_key=credential,
            base_url=base_url,
            http_client=proxied_httpx,
            default_headers=extra_headers or {},
        )
    def _get_max_retries(self, provider: str) -> int:
        """Get max retries for a provider.

        Resolution order:
        1. MAX_RETRIES_{PROVIDER} env var (e.g. MAX_RETRIES_CHUTES=5)
        2. Global self._max_retries (from constructor / DEFAULT_MAX_RETRIES)

        Results are cached after first lookup.

        Args:
            provider: Provider name

        Returns:
            Max retry count for this provider
        """
        if provider in self._provider_max_retries:
            return self._provider_max_retries[provider]

        provider_upper = provider.upper()
        env_keys = [f"{ENV_PREFIX_MAX_RETRIES}{provider_upper}"]

        if provider == "gemini":
            env_keys.append(f"{ENV_PREFIX_MAX_RETRIES}GOOGLE")
        elif provider == "google":
            env_keys.append(f"{ENV_PREFIX_MAX_RETRIES}GEMINI")

        env_val = None
        for env_key in env_keys:
            env_val = os.environ.get(env_key)
            if env_val is not None:
                break

        if env_val is not None:
            try:
                retries = int(env_val)
                if retries < 1:
                    lib_logger.warning(
                        f"Invalid {env_key}='{env_val}'. Must be >= 1. Using default ({self._max_retries})."
                    )
                    retries = self._max_retries
                else:
                    lib_logger.info(
                        f"Per-provider max retries: {provider} = {retries} (from {env_key})"
                    )
            except ValueError:
                lib_logger.warning(
                    f"Invalid {env_key}='{env_val}'. Must be integer. Using default ({self._max_retries})."
                )
                retries = self._max_retries
        else:
            retries = self._max_retries

        self._provider_max_retries[provider] = retries
        return retries

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
            request_type=getattr(context, "request_type", "chat"),
        )

        # Sanitize request payload
        kwargs = sanitize_request_payload(kwargs, model)

        # Apply provider-specific LiteLLM params
        self._apply_litellm_provider_params(provider, kwargs)

        # Add transaction context for provider logging
        if context.transaction_logger:
            kwargs["transaction_context"] = context.transaction_logger.get_context()

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
                await context.pre_request_callback(context.request, kwargs)
            except Exception as e:
                if self._abort_on_callback_error:
                    raise PreRequestCallbackError(str(e)) from e
                lib_logger.warning(f"Pre-request callback failed: {e}")

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
        if context.streaming:
            return self._execute_streaming(context)
        else:
            return await self._execute_non_streaming(context)

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
        max_retries = self._get_max_retries(provider)
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

                    if context.transaction_logger:
                        context.transaction_logger.credential_masked = mask_credential(cred)

                    try:
                        # Prepare request kwargs
                        kwargs = await self._prepare_request_kwargs(
                            provider, model, cred, context
                        )

                        # Log transformed request if it differs from original
                        if context.transaction_logger:
                            context.transaction_logger.log_transformed_request(
                                kwargs, context.kwargs
                            )

                        # Get provider plugin
                        plugin = self._get_plugin_instance(provider)

                        # Execute request with retries
                        for attempt in range(max_retries):
                            try:
                                lib_logger.info(
                                    f"Attempting call with credential {mask_credential(cred)} "
                                    f"(Attempt {attempt + 1}/{max_retries})"
                                )
                                # Pre-request callback
                                await self._run_pre_request_callback(context, kwargs)

                                # Resolve proxy-aware HTTP client
                                request_client = await self._resolve_http_client(
                                    provider, cred, cred_context.stable_id
                                )

                                is_embedding = getattr(context, "request_type", "chat") == "embedding"

                                # Make the API call
                                if plugin and plugin.has_custom_logic():
                                    kwargs["credential_identifier"] = credential_secret
                                    call_fn = plugin.aembedding if is_embedding else plugin.acompletion
                                    response = await call_fn(self._http_client, **kwargs)
                                else:
                                    # Standard LiteLLM call
                                    kwargs["api_key"] = credential_secret
                                    kwargs["max_retries"] = 0
                                    self._apply_litellm_logger(kwargs)
                                    # Remove internal context before litellm call
                                    kwargs.pop("transaction_context", None)
                                    litellm_client = await self._resolve_litellm_client(
                                        provider, cred, cred_context.stable_id,
                                        base_url=kwargs.get("api_base"),
                                        extra_headers=kwargs.get("extra_headers"),
                                    )
                                    if litellm_client:
                                        kwargs["client"] = litellm_client
                                    if is_embedding:
                                        response = await litellm.aembedding(**kwargs)
                                    else:
                                        response = await litellm.acompletion(**kwargs)

                                # Success! Extract token usage if available
                                (
                                    prompt_tokens,
                                    completion_tokens,
                                    prompt_tokens_cached,
                                    prompt_tokens_cache_write,
                                    thinking_tokens,
                                ) = self._extract_usage_tokens(response)
                                approx_cost = self._calculate_cost(
                                    provider, model, response
                                )
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
                                        context.transaction_logger.log_response(
                                            response_data
                                        )
                                    except Exception as log_err:
                                        lib_logger.debug(
                                            f"Failed to log response: {log_err}"
                                        )

                                if hasattr(response, "usage") and response.usage:
                                    normalize_usage_for_response(response.usage, model)
                                return self._extract_thought_tags_from_response(response)

                            except Exception as e:
                                last_exception = e
                                action = await self._handle_error_with_context(
                                    e,
                                    cred_context,
                                    model,
                                    provider,
                                    attempt,
                                    max_retries,
                                    error_accumulator,
                                    retry_state,
                                    request_headers,
                                    deadline=deadline,
                                )

                                if action == ErrorAction.RETRY_SAME:
                                    continue
                                elif action == ErrorAction.ROTATE:
                                    break  # Try next credential
                                else:  # FAIL
                                    # Raise as TerminalRequestError so it escapes
                                    # the `except Exception: pass` cleanup block below.
                                    raise TerminalRequestError(e)

                    except PreRequestCallbackError:
                        raise
                    except TerminalRequestError:
                        # Non-rotatable error (e.g. 404 model not found) - propagate immediately.
                        # Must be caught before the bare `except Exception: pass` below.
                        raise
                    except Exception:
                        # Let context manager handle cleanup
                        pass

            except NoAvailableKeysError:
                break
            except TerminalRequestError as terminal:
                # Non-rotatable error (e.g. 404 model not found) — stop immediately.
                # Record in accumulator for a clean error response, then bail out.
                original = terminal.original
                classified = classify_error(original, provider)
                lib_logger.error(
                    f"Non-rotatable error for {model} ({classified.error_type}, "
                    f"HTTP {classified.status_code}): {str(original)[:200]} — skipping rotation"
                )
                # Build an immediate error response and raise with proper HTTP mapping
                from ..error_handler import RequestErrorAccumulator as _RqErrAcc
                acc = _RqErrAcc()
                acc.model = model
                acc.provider = provider
                acc.record_error("(terminal)", classified, str(original)[:200])
                error_response = acc.build_client_error_response()
                raise ProxyExhaustionError(
                    error_response,
                    dominant_code=classified.error_type,
                )

        # All credentials exhausted
        error_accumulator.timeout_occurred = time.time() >= deadline
        if last_exception and not error_accumulator.has_errors():
            raise last_exception

        # Raise ProxyExhaustionError so main.py can map to the correct HTTP status
        error_response = error_accumulator.build_client_error_response()
        raise ProxyExhaustionError(
            error_response,
            dominant_code=error_accumulator.get_dominant_error_type(),
        )

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
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
            return

        error_accumulator = RequestErrorAccumulator()
        error_accumulator.model = model
        error_accumulator.provider = provider

        retry_state = RetryState()
        max_retries = self._get_max_retries(provider)
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

                        if context.transaction_logger:
                            context.transaction_logger.credential_masked = mask_credential(cred)

                        try:
                            # Prepare request kwargs
                            kwargs = await self._prepare_request_kwargs(
                                provider, model, cred, context
                            )

                            # Log transformed request if it differs from original
                            if context.transaction_logger:
                                context.transaction_logger.log_transformed_request(
                                    kwargs, context.kwargs
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
                            # Use plugin's cost calculator if available
                            cost_calculator = None
                            if plugin and hasattr(plugin, "calculate_cost"):
                                cost_calculator = plugin.calculate_cost

                            # Execute request with retries
                            for attempt in range(max_retries):
                                last_streamed_chunk: Optional[str] = None
                                try:
                                    lib_logger.info(
                                        f"Attempting stream with credential {mask_credential(cred)} "
                                        f"(Attempt {attempt + 1}/{max_retries})"
                                    )
                                    # Pre-request callback
                                    await self._run_pre_request_callback(
                                        context, kwargs
                                    )

                                    # Make the API call
                                    if plugin and plugin.has_custom_logic():
                                        kwargs["credential_identifier"] = credential_secret
                                        stream = await plugin.acompletion(
                                            self._http_client, **kwargs
                                        )
                                    else:
                                        kwargs["api_key"] = credential_secret
                                        kwargs["stream"] = True
                                        kwargs["max_retries"] = 0  # Disable litellm internal retries; we handle them
                                        self._apply_litellm_logger(kwargs)
                                        # Remove internal context before litellm call
                                        kwargs.pop("transaction_context", None)
                                        stream = await litellm.acompletion(**kwargs)

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
                                        cost_calculator=cost_calculator,
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
                                        if retry_state.consecutive_quota_failures >= retry_state.quota_failure_threshold:
                                            lib_logger.error(
                                                f"{retry_state.quota_failure_threshold} consecutive quota errors in streaming - "
                                                "request may be too large"
                                            )
                                            cred_context.mark_failure(classified)
                                            error_data = {
                                                "error": {
                                                    "message": "Request exceeds quota for all credentials",
                                                    "type": "quota_exhausted",
                                                    "code": "quota_exceeded",
                                                }
                                            }
                                            yield f"data: {json.dumps(error_data)}\n\n"
                                            yield "data: [DONE]\n\n"
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
                                        yield f"data: {json.dumps(error_data)}\n\n"
                                        yield "data: [DONE]\n\n"
                                        return

                                    small_cooldown_threshold = int(
                                        os.environ.get(
                                            "SMALL_COOLDOWN_RETRY_THRESHOLD",
                                            DEFAULT_SMALL_COOLDOWN_RETRY_THRESHOLD,
                                        )
                                    )
                                    rate_limit_max_retry_after = int(
                                        os.environ.get(
                                            "RATE_LIMIT_MAX_RETRY_AFTER",
                                            DEFAULT_RATE_LIMIT_MAX_RETRY_AFTER,
                                        )
                                    )
                                    is_rate_limit_retry = (
                                        classified.error_type == "rate_limit"
                                        and classified.retry_after is not None
                                        and 0 < classified.retry_after <= rate_limit_max_retry_after
                                    )
                                    if (
                                        (should_retry_same_key(classified, small_cooldown_threshold) or is_rate_limit_retry)
                                        and attempt < max_retries - 1
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
                                        if retry_state.consecutive_quota_failures >= retry_state.quota_failure_threshold:
                                            lib_logger.error(
                                                f"{retry_state.quota_failure_threshold} consecutive quota errors in streaming - "
                                                "request may be too large"
                                            )
                                            cred_context.mark_failure(classified)
                                            error_data = {
                                                "error": {
                                                    "message": "Request exceeds quota for all credentials",
                                                    "type": "quota_exhausted",
                                                    "code": "quota_exceeded",
                                                }
                                            }
                                            yield f"data: {json.dumps(error_data)}\n\n"
                                            yield "data: [DONE]\n\n"
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
                                    rate_limit_max_retry_after = int(
                                        os.environ.get(
                                            "RATE_LIMIT_MAX_RETRY_AFTER",
                                            DEFAULT_RATE_LIMIT_MAX_RETRY_AFTER,
                                        )
                                    )
                                    if (
                                        classified.retry_after is not None
                                        and 0
                                        < classified.retry_after
                                        < small_cooldown_threshold
                                        and attempt < max_retries - 1
                                    ):
                                        remaining = deadline - time.time()
                                        if classified.retry_after <= remaining:
                                            lib_logger.info(
                                                f"Retrying {mask_credential(cred)} in {classified.retry_after:.1f}s "
                                                f"(small cooldown {classified.retry_after}s < {small_cooldown_threshold}s threshold)"
                                            )
                                            await asyncio.sleep(classified.retry_after)
                                            continue  # Retry same key

                                    _retryable_429 = classified.error_type in ("rate_limit", "quota_exceeded")

                                    # For rate_limit/quota_exceeded with retry_after, wait
                                    # and retry same key if within our max threshold.
                                    if (
                                        _retryable_429
                                        and classified.retry_after is not None
                                        and 0 < classified.retry_after <= rate_limit_max_retry_after
                                        and attempt < max_retries - 1
                                    ):
                                        remaining = deadline - time.time()
                                        if classified.retry_after <= remaining:
                                            lib_logger.info(
                                                f"Retrying {mask_credential(cred)} in {classified.retry_after:.1f}s "
                                                f"({classified.error_type} retry_after={classified.retry_after}s <= {rate_limit_max_retry_after}s max)"
                                            )
                                            await asyncio.sleep(classified.retry_after)
                                            continue  # Retry same key

                                    # For rate_limit/quota_exceeded (429) without
                                    # retry_after, retry with exponential backoff —
                                    # transient capacity errors (including Google
                                    # RESOURCE_EXHAUSTED) are better handled by backoff,
                                    # especially with few credentials.
                                    if (
                                        _retryable_429
                                        and attempt < max_retries - 1
                                        and not classified.retry_after
                                    ):
                                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                                        remaining = deadline - time.time()
                                        if wait_time <= remaining:
                                            lib_logger.info(
                                                f"Retrying {mask_credential(cred)} in {wait_time:.1f}s "
                                                f"({classified.error_type} backoff, attempt {attempt + 1}/{max_retries})"
                                            )
                                            await asyncio.sleep(wait_time)
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
                                    log_failure(
                                        api_key=cred,
                                        model=model,
                                        attempt=attempt + 1,
                                        error=e,
                                        request_headers=request_headers,
                                    )

                                    if attempt >= max_retries - 1:
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

                                except Exception as e:
                                    last_exception = e
                                    classified = classify_error(e, provider)
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
                                        # Raise as TerminalRequestError so it escapes
                                        # the `except Exception: pass` cleanup block.
                                        raise TerminalRequestError(e)

                                    small_cooldown_threshold = int(
                                        os.environ.get(
                                            "SMALL_COOLDOWN_RETRY_THRESHOLD",
                                            DEFAULT_SMALL_COOLDOWN_RETRY_THRESHOLD,
                                        )
                                    )
                                    rate_limit_max_retry_after = int(
                                        os.environ.get(
                                            "RATE_LIMIT_MAX_RETRY_AFTER",
                                            DEFAULT_RATE_LIMIT_MAX_RETRY_AFTER,
                                        )
                                    )
                                    is_rate_limit_retry = (
                                        classified.error_type == "rate_limit"
                                        and classified.retry_after is not None
                                        and 0 < classified.retry_after <= rate_limit_max_retry_after
                                    )
                                    if (
                                        (should_retry_same_key(classified, small_cooldown_threshold) or is_rate_limit_retry)
                                        and attempt < max_retries - 1
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
                        except TerminalRequestError:
                            # Non-rotatable error — propagate immediately.
                            raise
                        except Exception:
                            # Let context manager handle cleanup
                            pass

                except NoAvailableKeysError:
                    break

            # All credentials exhausted or timeout
            error_accumulator.timeout_occurred = time.time() >= deadline
            error_data = error_accumulator.build_client_error_response()
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

        except NoAvailableKeysError as e:
            lib_logger.error(f"No keys available: {e}")
            error_data = {"error": {"message": str(e), "type": "proxy_busy"}}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

        except TerminalRequestError as terminal:
            # Non-rotatable error (e.g. 404 model not found) — stop immediately.
            original = terminal.original
            classified = classify_error(original, provider)
            lib_logger.error(
                f"Non-rotatable error for {model} ({classified.error_type}, "
                f"HTTP {classified.status_code}): {str(original)[:200]} — skipping rotation"
            )
            error_data = {
                "error": {
                    "message": (
                        f"Model not available: {str(original)[:300]}"
                        if classified.status_code == 404
                        else f"Request error ({classified.error_type}): {str(original)[:300]}"
                    ),
                    "type": "model_not_available" if classified.status_code == 404 else classified.error_type,
                    "details": {"model": model, "status_code": classified.status_code},
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            lib_logger.error(f"Unhandled exception in streaming: {e}", exc_info=True)
            error_data = {"error": {"message": str(e), "type": "proxy_internal_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

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

    # Gemma-4 and similar models emit reasoning inside <thought>...</thought>
    # tags within the regular content field.  For non-streaming responses we
    # can strip them with a simple regex.
    _THOUGHT_PATTERN = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)

    def _extract_thought_tags_from_response(self, response: Any) -> Any:
        """
        Extract <thought>...</thought> blocks from a non-streaming response.

        Thought content is moved to ``message.reasoning_content`` (OpenAI
        o1-style) and removed from ``message.content``.  Mutates the
        response in-place when possible; falls back to returning a plain
        dict for immutable Pydantic models.
        """
        choices = getattr(response, "choices", None)
        if not choices:
            return response

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            return response

        content = getattr(message, "content", None)
        if not content or not isinstance(content, str):
            return response

        # Extract all reasoning segments and build stripped content.
        reasoning_parts: list[str] = []
        stripped = content
        for match in self._THOUGHT_PATTERN.finditer(content):
            reasoning_parts.append(match.group(1))
            stripped = stripped.replace(match.group(0), "", 1)

        if not reasoning_parts and stripped == content:
            return response

        reasoning_content = "".join(reasoning_parts)

        # Try in-place mutation first (some LiteLLM objects are mutable).
        try:
            message.content = stripped
            message.reasoning_content = reasoning_content
            return response
        except Exception:
            pass

        # Fallback: convert to dict, modify, and return the dict.
        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        elif hasattr(response, "dict"):
            response_dict = response.dict()
        else:
            response_dict = dict(response)

        msg = response_dict.get("choices", [{}])[0].get("message")
        if msg is not None:
            msg["content"] = stripped
            msg["reasoning_content"] = reasoning_content

        return response_dict

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
        max_retries: int,
        error_accumulator: RequestErrorAccumulator,
        retry_state: RetryState,
        request_headers: Dict[str, Any],
        deadline: float,
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
            deadline: Request deadline

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
            if retry_state.consecutive_quota_failures >= retry_state.quota_failure_threshold:
                lib_logger.error(
                    f"{retry_state.quota_failure_threshold} consecutive quota errors - request may be too large"
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

        # Check for rate_limit with retry_after within our max threshold
        rate_limit_max_retry_after = int(
            os.environ.get(
                "RATE_LIMIT_MAX_RETRY_AFTER", DEFAULT_RATE_LIMIT_MAX_RETRY_AFTER
            )
        )
        is_rate_limit_retry = (
            classified.error_type == "rate_limit"
            and classified.retry_after is not None
            and 0 < classified.retry_after <= rate_limit_max_retry_after
        )

        if (
            (should_retry_same_key(classified, small_cooldown_threshold) or is_rate_limit_retry)
            and attempt < max_retries - 1
        ):
            wait_time = (
                classified.retry_after
                if classified.retry_after is not None
                else self._get_transient_retry_delay()
            )
            
            # Check remaining deadline budget
            remaining = deadline - time.time()
            if wait_time <= remaining:
                retry_reason = (
                    f" (small cooldown {classified.retry_after}s < {small_cooldown_threshold}s threshold)"
                    if is_small_cooldown
                    else (
                        f" (rate_limit retry_after={classified.retry_after}s <= {rate_limit_max_retry_after}s max)"
                        if is_rate_limit_retry
                        else ""
                    )
                )
                lib_logger.info(
                    f"Retrying {mask_credential(credential)} in {wait_time:.1f}s{retry_reason}"
                )
                await asyncio.sleep(wait_time)
                return ErrorAction.RETRY_SAME
            else:
                lib_logger.info(
                    f"Skipping retry same key for {mask_credential(credential)} (wait {wait_time:.1f}s > {remaining:.1f}s remaining)"
                )

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

    def _extract_usage_tokens(self, response: Any) -> tuple[int, int, int, int, int]:
        prompt_tokens = 0
        completion_tokens = 0
        cached_tokens = 0
        cache_write_tokens = 0
        thinking_tokens = 0

        if hasattr(response, "usage") and response.usage:
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0

            prompt_details = getattr(response.usage, "prompt_tokens_details", None)
            if prompt_details:
                if isinstance(prompt_details, dict):
                    cached_tokens = prompt_details.get("cached_tokens", 0) or 0
                    cache_write_tokens = (
                        prompt_details.get("cache_creation_tokens", 0) or 0
                    )
                else:
                    cached_tokens = getattr(prompt_details, "cached_tokens", 0) or 0
                    cache_write_tokens = (
                        getattr(prompt_details, "cache_creation_tokens", 0) or 0
                    )

            completion_details = getattr(
                response.usage, "completion_tokens_details", None
            )
            if completion_details:
                if isinstance(completion_details, dict):
                    thinking_tokens = completion_details.get("reasoning_tokens", 0) or 0
                else:
                    thinking_tokens = (
                        getattr(completion_details, "reasoning_tokens", 0) or 0
                    )

            cache_read_tokens = getattr(response.usage, "cache_read_tokens", None)
            if cache_read_tokens is not None:
                cached_tokens = cache_read_tokens or 0
            cache_creation_tokens = getattr(
                response.usage, "cache_creation_tokens", None
            )
            if cache_creation_tokens is not None:
                cache_write_tokens = cache_creation_tokens or 0

            if thinking_tokens and completion_tokens >= thinking_tokens:
                completion_tokens = completion_tokens - thinking_tokens

        uncached_prompt = max(0, prompt_tokens - cached_tokens)
        return (
            uncached_prompt,
            completion_tokens,
            cached_tokens,
            cache_write_tokens,
            thinking_tokens,
        )

    def _calculate_cost(self, provider: str, model: str, response: Any) -> float:
        plugin = self._get_plugin_instance(provider)
        if plugin and getattr(plugin, "skip_cost_calculation", False):
            return 0.0

        # If the plugin provides its own cost calculation (e.g. from provider
        # API pricing data), use it instead of LiteLLM's internal database.
        if plugin and hasattr(plugin, "calculate_cost"):
            try:
                usage = getattr(response, "usage", None)
                if usage:
                    (
                        prompt_tokens,
                        completion_tokens,
                        cache_read,
                        cache_write,
                        thinking_tokens,
                    ) = self._extract_usage_tokens(response)
                    
                    # Try to pass cache info if plugin supports it
                    import inspect
                    sig = inspect.signature(plugin.calculate_cost)
                    if "cache_read_tokens" in sig.parameters:
                        cost = plugin.calculate_cost(
                            model, 
                            prompt_tokens, 
                            completion_tokens + thinking_tokens,
                            cache_read_tokens=cache_read,
                            cache_creation_tokens=cache_write
                        )
                    else:
                        # Fallback for plugins with simple signatures
                        cost = plugin.calculate_cost(model, prompt_tokens, completion_tokens + thinking_tokens)
                        
                    if cost > 0:
                        return cost
            except Exception as exc:
                lib_logger.debug(
                    f"Plugin cost calculation failed for {model}: {exc}"
                )

        try:
            if isinstance(response, litellm.EmbeddingResponse):
                model_info = litellm.get_model_info(model)
                input_cost = model_info.get("input_cost_per_token")
                if input_cost:
                    return (response.usage.prompt_tokens or 0) * input_cost
                return 0.0

            cost = litellm.completion_cost(
                completion_response=response,
                model=model,
            )
            return float(cost) if cost is not None else 0.0
        except Exception as exc:
            lib_logger.debug(f"Cost calculation failed for {model}: {exc}")
            return 0.0

    async def _transaction_logging_stream_wrapper(
        self,
        stream: AsyncGenerator[str, None],
        transaction_logger: TransactionLogger,
        request_kwargs: Dict[str, Any],
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
                        transaction_logger.log_stream_chunk(chunk_data)
                except json.JSONDecodeError:
                    lib_logger.debug(
                        f"Failed to parse chunk for logging: {sse_line[:100]}"
                    )

        # Log assembled final response
        if chunks:
            try:
                final_response = TransactionLogger.assemble_streaming_response(chunks)
                transaction_logger.log_response(final_response)
            except Exception as e:
                lib_logger.debug(
                    f"Failed to assemble/log final streaming response: {e}"
                )
