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
import random
import time
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
from ..core.constants import DEFAULT_MAX_RETRIES
from ..request_sanitizer import sanitize_request_payload
from ..transaction_logger import TransactionLogger
from ..failure_logger import log_failure

from .types import RetryState, AvailabilityStats
from .filters import CredentialFilter
from .transforms import ProviderTransforms
from .streaming import StreamingHandler

if TYPE_CHECKING:
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
        """
        self._usage_managers = usage_managers
        self._cooldown = cooldown_manager
        self._filter = credential_filter
        self._transforms = provider_transforms
        self._plugins = provider_plugins
        self._plugin_instances: Dict[str, Any] = {}
        self._http_client = http_client
        self._max_retries = max_retries
        self._global_timeout = global_timeout
        self._abort_on_callback_error = abort_on_callback_error
        self._litellm_provider_params = litellm_provider_params or {}
        self._litellm_logger_fn = litellm_logger_fn
        # StreamingHandler no longer needs usage_manager - we pass cred_context directly
        self._streaming_handler = StreamingHandler()

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

        usage_manager = self._usage_managers.get(provider)
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

            availability = await usage_manager.get_availability_stats(
                model, quota_group
            )
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
                f"Acquiring key for model {model}. Tried keys: {len(retry_state.tried_credentials)}/"
                f"{availability.get('available', 0)}({availability.get('total', 0)}{blocked_str})"
            )

            # Wait for provider cooldown
            await self._wait_for_cooldown(provider, deadline)

            # Acquire credential using context manager
            try:
                availability = await usage_manager.get_availability_stats(
                    model, quota_group
                )
                async with await usage_manager.acquire_credential(
                    model=model,
                    quota_group=quota_group,
                    candidates=untried,
                    priorities=filter_result.priorities,
                    deadline=deadline,
                ) as cred_context:
                    cred = cred_context.credential
                    retry_state.record_attempt(cred)

                    state = getattr(usage_manager, "states", {}).get(
                        cred_context.stable_id
                    )
                    tier = state.tier if state else None
                    priority = state.priority if state else None
                    selection_mode = availability.get("rotation_mode")
                    quota_display = "?/?"
                    primary_def = None
                    if state and getattr(usage_manager, "window_manager", None):
                        primary_def = (
                            usage_manager.window_manager.get_primary_definition()
                        )
                    if state and primary_def:
                        scope_key = (
                            quota_group if primary_def.applies_to == "group" else model
                        )
                        if primary_def.applies_to == "group":
                            usage = state.get_group_stats(scope_key, create=False)
                        else:
                            usage = state.get_model_stats(scope_key, create=False)
                        if usage:
                            window = usage.windows.get(primary_def.name)
                            if window and window.limit is not None:
                                remaining = max(0, window.limit - window.request_count)
                                pct = (
                                    round(remaining / window.limit * 100)
                                    if window.limit
                                    else 0
                                )
                                quota_display = (
                                    f"{window.request_count}/{window.limit} [{pct}%]"
                                )
                    lib_logger.info(
                        f"Acquired key {mask_credential(cred)} for model {model} "
                        f"(tier: {tier}, priority: {priority}, selection: {selection_mode}, quota: {quota_display})"
                    )

                    try:
                        # Apply transforms
                        kwargs = await self._transforms.apply(
                            provider, model, cred, context.kwargs.copy()
                        )

                        # Sanitize request payload
                        kwargs = sanitize_request_payload(kwargs, model)

                        # Apply provider-specific LiteLLM params
                        self._apply_litellm_provider_params(provider, kwargs)

                        # Get provider plugin
                        plugin = self._get_plugin_instance(provider)

                        # Add transaction context for provider logging
                        if context.transaction_logger:
                            kwargs["transaction_context"] = (
                                context.transaction_logger.get_context()
                            )

                        # Execute request with retries
                        for attempt in range(self._max_retries):
                            try:
                                lib_logger.info(
                                    f"Attempting call with credential {mask_credential(cred)} "
                                    f"(Attempt {attempt + 1}/{self._max_retries})"
                                )
                                # Pre-request callback
                                if context.pre_request_callback:
                                    try:
                                        await context.pre_request_callback(
                                            context.request, kwargs
                                        )
                                    except Exception as e:
                                        if self._abort_on_callback_error:
                                            raise PreRequestCallbackError(str(e)) from e
                                        lib_logger.warning(
                                            f"Pre-request callback failed: {e}"
                                        )

                                # Make the API call
                                if plugin and plugin.has_custom_logic():
                                    kwargs["credential_identifier"] = cred
                                    response = await plugin.acompletion(
                                        self._http_client, **kwargs
                                    )
                                else:
                                    # Standard LiteLLM call
                                    kwargs["api_key"] = cred
                                    self._apply_litellm_logger(kwargs)
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

                                return response

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
                                )

                                if action == ErrorAction.RETRY_SAME:
                                    continue
                                elif action == ErrorAction.ROTATE:
                                    break  # Try next credential
                                else:  # FAIL
                                    raise

                    except PreRequestCallbackError:
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
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"
            return

        error_accumulator = RequestErrorAccumulator()
        error_accumulator.model = model
        error_accumulator.provider = provider

        retry_state = RetryState()
        last_exception: Optional[Exception] = None

        try:
            availability = await usage_manager.get_availability_stats(
                model, quota_group
            )
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
                f"Acquiring credential for model {model}. Tried credentials: {len(retry_state.tried_credentials)}/"
                f"{availability.get('available', 0)}({availability.get('total', 0)}{blocked_str})"
            )

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
                    async with await usage_manager.acquire_credential(
                        model=model,
                        quota_group=quota_group,
                        candidates=untried,
                        priorities=filter_result.priorities,
                        deadline=deadline,
                    ) as cred_context:
                        cred = cred_context.credential
                        retry_state.record_attempt(cred)

                        state = getattr(usage_manager, "states", {}).get(
                            cred_context.stable_id
                        )
                        tier = state.tier if state else None
                        priority = state.priority if state else None
                        selection_mode = availability.get("rotation_mode")
                        quota_display = "?/?"
                        primary_def = None
                        if state and getattr(usage_manager, "window_manager", None):
                            primary_def = (
                                usage_manager.window_manager.get_primary_definition()
                            )
                        if state and primary_def:
                            scope_key = (
                                quota_group
                                if primary_def.applies_to == "group"
                                else model
                            )
                            if primary_def.applies_to == "group":
                                usage = state.get_group_stats(scope_key, create=False)
                            else:
                                usage = state.get_model_stats(scope_key, create=False)
                            if usage:
                                window = usage.windows.get(primary_def.name)
                                if window and window.limit is not None:
                                    remaining = max(
                                        0, window.limit - window.request_count
                                    )
                                    pct = (
                                        round(remaining / window.limit * 100)
                                        if window.limit
                                        else 0
                                    )
                                    quota_display = f"{window.request_count}/{window.limit} [{pct}%]"
                        lib_logger.info(
                            f"Acquired key {mask_credential(cred)} for model {model} "
                            f"(tier: {tier}, priority: {priority}, selection: {selection_mode}, quota: {quota_display})"
                        )

                        try:
                            # Apply transforms
                            kwargs = await self._transforms.apply(
                                provider, model, cred, context.kwargs.copy()
                            )

                            # Sanitize request payload
                            kwargs = sanitize_request_payload(kwargs, model)

                            # Apply provider-specific LiteLLM params
                            self._apply_litellm_provider_params(provider, kwargs)

                            # Add stream options (but not for iflow - it returns 406)
                            if provider != "iflow":
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

                            # Add transaction context for provider logging
                            if context.transaction_logger:
                                kwargs["transaction_context"] = (
                                    context.transaction_logger.get_context()
                                )

                            # Execute request with retries
                            for attempt in range(self._max_retries):
                                try:
                                    lib_logger.info(
                                        f"Attempting stream with credential {mask_credential(cred)} "
                                        f"(Attempt {attempt + 1}/{self._max_retries})"
                                    )
                                    # Pre-request callback
                                    if context.pre_request_callback:
                                        try:
                                            await context.pre_request_callback(
                                                context.request, kwargs
                                            )
                                        except Exception as e:
                                            if self._abort_on_callback_error:
                                                raise PreRequestCallbackError(
                                                    str(e)
                                                ) from e
                                            lib_logger.warning(
                                                f"Pre-request callback failed: {e}"
                                            )

                                    # Make the API call
                                    if plugin and plugin.has_custom_logic():
                                        kwargs["credential_identifier"] = cred
                                        stream = await plugin.acompletion(
                                            self._http_client, **kwargs
                                        )
                                    else:
                                        kwargs["api_key"] = cred
                                        kwargs["stream"] = True
                                        self._apply_litellm_logger(kwargs)
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
                                            yield chunk
                                    else:
                                        async for chunk in base_stream:
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
                                            yield f"data: {json.dumps(error_data)}\n\n"
                                            yield "data: [DONE]\n\n"
                                            return
                                    else:
                                        retry_state.reset_quota_failures()

                                    if not should_rotate_on_error(classified):
                                        cred_context.mark_failure(classified)
                                        raise

                                    cred_context.mark_failure(classified)
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
                                            yield f"data: {json.dumps(error_data)}\n\n"
                                            yield "data: [DONE]\n\n"
                                            return
                                    else:
                                        retry_state.reset_quota_failures()

                                    if not should_rotate_on_error(classified):
                                        cred_context.mark_failure(classified)
                                        raise

                                    cred_context.mark_failure(classified)
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

                                    if attempt >= self._max_retries - 1:
                                        error_accumulator.record_error(
                                            cred, classified, str(e)[:150]
                                        )
                                        cred_context.mark_failure(classified)
                                        break  # Rotate

                                    # Calculate wait time
                                    wait_time = classified.retry_after or (
                                        2**attempt
                                    ) + random.uniform(0, 1)
                                    remaining = deadline - time.time()
                                    if wait_time > remaining:
                                        break  # No time to wait

                                    await asyncio.sleep(wait_time)
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
                                        raise

                                    cred_context.mark_failure(classified)
                                    break  # Rotate

                        except PreRequestCallbackError:
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

        # Check if should retry same key
        if should_retry_same_key(classified) and attempt < self._max_retries - 1:
            wait_time = classified.retry_after or (2**attempt) + random.uniform(0, 1)
            lib_logger.info(
                f"Retrying {mask_credential(credential)} in {wait_time:.1f}s"
            )
            await asyncio.sleep(wait_time)
            return ErrorAction.RETRY_SAME

        # Record error and rotate
        error_accumulator.record_error(credential, classified, error_message)
        cred_context.mark_failure(classified)
        lib_logger.info(
            f"Rotating from {mask_credential(credential)} after {classified.error_type}"
        )
        return ErrorAction.ROTATE

    async def _ensure_initialized(
        self,
        usage_manager: "UsageManager",
        context: RequestContext,
        filter_result: "FilterResult",
    ) -> None:
        if usage_manager.initialized:
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
