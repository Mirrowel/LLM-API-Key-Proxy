# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
NanoGPT Provider

Provider for NanoGPT API (https://nano-gpt.com).
OpenAI-compatible API with subscription-based usage tracking.

Features:
- Dynamic model discovery from configurable endpoint (NANOGPT_MODEL_SOURCE)
- Environment variable model override (NANOGPT_MODELS)
- Subscription usage monitoring via /api/subscription/v1/usage
- Tier-based credential prioritization

Usage units:
NanoGPT tracks "usage units" (successful operations) rather than tokens.
All models share a daily/monthly usage pool at the credential level.
"""

import asyncio
import httpx
import os
import json
import uuid
import time
import logging
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, TYPE_CHECKING

import litellm

if TYPE_CHECKING:
    from ..usage import UsageManager

from .provider_interface import ProviderInterface
from .utilities.nanogpt_quota_tracker import NanoGptQuotaTracker
from .utilities.anthropic_converters import (
    convert_openai_to_anthropic_messages,
    convert_tools_to_anthropic_format,
    TOOL_PREFIX,
)
from ..model_definitions import ModelDefinitions
from ..timeout_config import TimeoutConfig

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

# NanoGPT API base URL
NANOGPT_API_BASE = "https://nano-gpt.com"

# Concurrency limit for parallel quota fetches
QUOTA_FETCH_CONCURRENCY = 5

# Minimum remaining tokens before treating quota as effectively exhausted.
# At < 250K tokens remaining on a 60M weekly budget, most requests will fail
# mid-stream anyway, so proactively mark the credential as exhausted.
NANOGPT_EXHAUSTION_TOKEN_THRESHOLD = int(
    os.getenv("NANOGPT_EXHAUSTION_TOKEN_THRESHOLD", "250000")
)

# Model discovery endpoint mapping
# Controlled by NANOGPT_MODEL_SOURCE env var
# Endpoints support ?detailed=true for full metadata (context_length, pricing, etc.)
NANOGPT_MODEL_SOURCES = {
    "all": "/api/v1/models",
    "personalized": "/api/personalized/v1/models",
    "subscription": "/api/subscription/v1/models",
}

# Auth header style per source: Bearer for /api/v1, x-api-key for others
_MODEL_SOURCE_AUTH_HEADERS = {
    "all": "Bearer",
    "personalized": "x-api-key",
    "subscription": "x-api-key",
}

# Fallback models if API discovery fails and no env override
NANOGPT_FALLBACK_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3.5-sonnet",
    "claude-3.5-haiku",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

# Required headers for NanoGPT Anthropic-compatible requests
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_USER_AGENT = "claude-cli/2.1.2 (external, cli)"
ANTHROPIC_BETA_HEADER = "oauth-2025-04-20,interleaved-thinking-2025-05-14"

# Models that should use the Anthropic endpoint by default on NanoGPT
CLAUDE_MODELS_PREFIX = "claude-"


def _attempt_json_repair(s: str) -> Optional[Any]:
    """Attempt to repair truncated JSON from LLM tool calls.

    Handles the common case where the model generates valid JSON that is cut
    short (missing closing brackets/braces).
    """
    stripped = s.rstrip()
    if not stripped:
        return None

    opener_stack: list = []
    in_string = False
    escape_next = False

    for ch in stripped:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            opener_stack.append("}")
        elif ch == "[":
            opener_stack.append("]")
        elif ch in ("}", "]"):
            if opener_stack and opener_stack[-1] == ch:
                opener_stack.pop()

    if not opener_stack:
        return None

    candidate = stripped.rstrip(",")
    candidate += "".join(reversed(opener_stack))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


class NanoGptProvider(NanoGptQuotaTracker, ProviderInterface):
    """
    Provider for NanoGPT API.

    Supports subscription-based usage tracking with daily/monthly limits.
    All models share the same usage pool at the credential level.
    """

    # Skip cost calculation - NanoGPT uses "usage units", not tokens
    skip_cost_calculation = True

    # =========================================================================
    # PROVIDER CONFIGURATION
    # =========================================================================

    provider_env_name = "nanogpt"

    # Tier priorities based on subscription state
    # Active subscriptions get highest priority
    tier_priorities = {
        "subscription-active": 1,  # Active subscription
        "subscription-grace": 2,  # Grace period (subscription lapsed but still has access)
        "no-subscription": 3,  # No active subscription (pay-as-you-go only)
    }
    default_tier_priority = 3

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse NanoGPT-specific quota/rate-limit errors.

        NanoGPT 429 responses indicate subscription quota exhaustion when
        they mention usage limits, balance, or subscription caps.
        Any 429 from NanoGPT that isn't clearly a short-term rate limit
        (per-minute/per-second) is treated as quota exhaustion since
        NanoGPT's rate limiting is primarily subscription-based.
        """
        body = error_body
        if not body:
            if hasattr(error, "response") and hasattr(error.response, "text"):
                body = error.response.text
            elif hasattr(error, "body"):
                body = str(error.body) if not isinstance(error.body, str) else error.body
            else:
                body = str(error)

        body_lower = body.lower() if body else ""

        status_code = None
        if hasattr(error, "status_code"):
            status_code = error.status_code
        elif hasattr(error, "response") and hasattr(error.response, "status_code"):
            status_code = error.response.status_code

        if status_code != 429 and "429" not in body_lower:
            return None

        quota_keywords = [
            "limit", "balance", "subscription", "exceeded",
            "usage", "insufficient", "cap", "exhausted",
        ]
        per_request_keywords = ["per minute", "per_minute", "per second", "per_second"]

        if any(kw in body_lower for kw in per_request_keywords):
            return {"retry_after": None, "reason": "rate_limit_exceeded"}

        if any(kw in body_lower for kw in quota_keywords):
            return {"retry_after": None, "reason": "subscription_quota_exhausted"}

        return {"retry_after": None, "reason": "quota_exhausted"}

    # Quota groups for tracking weekly input tokens.
    # All real models share the weekly_tokens pool (subscription-level quota).
    # get_model_quota_group() override below returns "weekly_tokens" for all models.
    model_quota_groups = {
        "weekly_tokens": ["_weekly_tokens"],
    }

    def __init__(self):
        self.model_definitions = ModelDefinitions()

        # Quota tracking cache
        self._subscription_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval = int(
            os.getenv("NANOGPT_QUOTA_REFRESH_INTERVAL", "300")
        )

        # Model source filtering (which API endpoint to use for discovery)
        self._model_source = os.getenv("NANOGPT_MODEL_SOURCE", "all").lower()
        if self._model_source not in NANOGPT_MODEL_SOURCES:
            lib_logger.warning(
                f"Invalid NANOGPT_MODEL_SOURCE='{self._model_source}', "
                f"falling back to 'all'. "
                f"Valid options: {list(NANOGPT_MODEL_SOURCES.keys())}"
            )
            self._model_source = "all"

        if self._model_source != "all":
            lib_logger.info(f"NanoGPT model source: {self._model_source}")

        # Tier cache (credential -> tier name)
        self._tier_cache: Dict[str, str] = {}

        # Track discovered models for quota group sync
        self._discovered_models: set = set()

        # Track subscription-only models (subject to daily/monthly limits)
        self._subscription_models: set = set()

    def has_custom_logic(self) -> bool:
        """NanoGPT uses custom logic to route to Anthropic endpoint for Claude models."""
        return True

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle chat completion request for NanoGPT.

        Routes based on original request type:
        - _anthropic_payload: Use NanoGPT's Anthropic endpoint
        - Other models/formats: Use LiteLLM passthrough (OpenAI endpoint)
        """
        model = kwargs.get("model", "")
        # Remove internal context
        credential = kwargs.pop("credential_identifier", "")
        anthropic_payload = kwargs.pop("_anthropic_payload", None)
        stream = kwargs.pop("stream", False)

        # Determine if we should use the Anthropic endpoint
        # Only use it if the request came in via the Anthropic handler.
        use_anthropic_endpoint = anthropic_payload is not None

        if use_anthropic_endpoint:
            return await self._anthropic_completion(
                client, credential, stream, anthropic_payload, **kwargs
            )
        else:
            return await self._openai_completion(
                client, credential=credential, stream=stream, **kwargs
            )

    async def _openai_completion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """Standard OpenAI-compatible path via direct HTTP.

        Uses raw httpx streaming instead of litellm.acompletion to avoid
        litellm/OpenAI SDK choking on DeepSeek's ``malformed_function_call``
        finish reason and broken tool-call JSON.
        """
        credential = kwargs.pop("credential", "")
        stream = kwargs.pop("stream", False)
        model = kwargs.get("model", "")

        # Pop internal fields
        kwargs.pop("transaction_context", None)
        kwargs.pop("litellm_params", None)
        kwargs.pop("api_base", None)
        kwargs.pop("api_key", None)
        kwargs.pop("custom_llm_provider", None)

        # Build the OpenAI-format payload
        payload: Dict[str, Any] = {"model": model.split("/", 1)[-1] if "/" in model else model, "stream": stream}
        for key in ("messages", "tools", "tool_choice", "max_tokens",
                     "temperature", "top_p", "stop", "frequency_penalty",
                     "presence_penalty", "n", "reasoning_effort"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]
        if stream:
            payload["stream_options"] = {"include_usage": True}

        headers = {
            "Authorization": f"Bearer {credential}",
            "Content-Type": "application/json",
        }

        url = f"{NANOGPT_API_BASE}/api/v1/chat/completions"

        if stream:
            return self._stream_openai_response(client, url, headers, payload, model)
        else:
            return await self._non_stream_openai_response(client, url, headers, payload, model)

    async def _stream_openai_response(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Stream from NanoGPT's OpenAI endpoint with malformed tool-call repair.

        DeepSeek models can emit ``finish_reason: "malformed_function_call"``
        with truncated/broken tool-call argument JSON.  The standard litellm
        OpenAI path cannot recover from this.  Here we parse the SSE ourselves,
        attempt JSON repair on tool-call arguments, and yield clean chunks.
        """
        async with client.stream(
            "POST", url, headers=headers, json=payload,
            timeout=TimeoutConfig.streaming(),
        ) as response:
            if response.status_code >= 400:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                lib_logger.error(f"NanoGPT OpenAI API error {response.status_code}: {error_text[:500]}")
                raise httpx.HTTPStatusError(
                    f"NanoGPT OpenAI API error: {response.status_code}",
                    request=response.request, response=response,
                )

            accumulated_tool_args: Dict[int, str] = {}
            accumulated_tool_meta: Dict[int, Dict[str, str]] = {}

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if not data or data == "[DONE]":
                    continue

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    # Usage-only chunk (stream_options)
                    usage_obj = chunk.get("usage")
                    if usage_obj:
                        final = litellm.ModelResponse(
                            id=chunk.get("id", ""),
                            created=chunk.get("created", 0),
                            model=model,
                            object="chat.completion.chunk",
                            choices=[{"index": 0, "delta": {}, "finish_reason": None}],
                        )
                        final.usage = litellm.Usage(
                            prompt_tokens=usage_obj.get("prompt_tokens", 0),
                            completion_tokens=usage_obj.get("completion_tokens", 0),
                            total_tokens=usage_obj.get("total_tokens", 0),
                        )
                        yield final
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                # Accumulate incremental tool-call argument fragments
                tool_calls = delta.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        idx = tc.get("index", 0)
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            accumulated_tool_meta[idx] = {
                                "id": tc.get("id", ""),
                                "name": fn["name"],
                            }
                        if "arguments" in fn:
                            accumulated_tool_args.setdefault(idx, "")
                            accumulated_tool_args[idx] += fn["arguments"]

                # Map DeepSeek-specific finish reasons
                if finish_reason == "malformed_function_call":
                    lib_logger.warning(
                        f"NanoGPT/DeepSeek returned malformed_function_call for {model}, "
                        "attempting tool-call argument repair"
                    )
                    repaired_tool_calls = self._repair_accumulated_tool_calls(
                        accumulated_tool_args, accumulated_tool_meta
                    )
                    if repaired_tool_calls:
                        yield litellm.ModelResponse(
                            id=chunk.get("id", ""),
                            created=chunk.get("created", 0),
                            model=model,
                            object="chat.completion.chunk",
                            choices=[{
                                "index": 0,
                                "delta": {"tool_calls": repaired_tool_calls},
                                "finish_reason": None,
                            }],
                        )
                        finish_reason = "tool_calls"
                    else:
                        finish_reason = "stop"
                    # Emit the final chunk with corrected finish_reason
                    yield litellm.ModelResponse(
                        id=chunk.get("id", ""),
                        created=chunk.get("created", 0),
                        model=model,
                        object="chat.completion.chunk",
                        choices=[{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                    )
                    continue

                # Build output delta, mapping upstream field names
                out_delta: Dict[str, Any] = {}
                if "role" in delta:
                    out_delta["role"] = delta["role"]
                if delta.get("content"):
                    out_delta["content"] = delta["content"]
                # NanoGPT/DeepSeek uses "reasoning"; normalize to "reasoning_content"
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if reasoning:
                    out_delta["reasoning_content"] = reasoning
                if tool_calls:
                    out_delta["tool_calls"] = tool_calls

                resp = litellm.ModelResponse(
                    id=chunk.get("id", ""),
                    created=chunk.get("created", 0),
                    model=model,
                    object="chat.completion.chunk",
                    choices=[{
                        "index": 0,
                        "delta": out_delta,
                        "finish_reason": finish_reason,
                    }],
                )
                # Attach usage from the final chunk (NanoGPT embeds it
                # in the same chunk that carries finish_reason)
                usage_obj = chunk.get("usage")
                if usage_obj:
                    resp.usage = litellm.Usage(
                        prompt_tokens=usage_obj.get("prompt_tokens", 0),
                        completion_tokens=usage_obj.get("completion_tokens", 0),
                        total_tokens=usage_obj.get("total_tokens", 0),
                    )
                yield resp

    async def _non_stream_openai_response(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> litellm.ModelResponse:
        """Non-streaming call to NanoGPT's OpenAI endpoint."""
        resp = await client.post(url, headers=headers, json=payload,
                                 timeout=TimeoutConfig.non_streaming())
        if resp.status_code >= 400:
            lib_logger.error(f"NanoGPT OpenAI API error {resp.status_code}: {resp.text[:500]}")
            raise httpx.HTTPStatusError(
                f"NanoGPT OpenAI API error: {resp.status_code}",
                request=resp.request, response=resp,
            )

        data = resp.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        choice_data: Dict[str, Any] = {
            "index": 0,
            "message": {
                "role": message.get("role", "assistant"),
                "content": message.get("content"),
            },
            "finish_reason": choice.get("finish_reason", "stop"),
        }
        if message.get("tool_calls"):
            choice_data["message"]["tool_calls"] = message["tool_calls"]
            if choice_data["finish_reason"] == "malformed_function_call":
                choice_data["finish_reason"] = "tool_calls"

        return litellm.ModelResponse(
            id=data.get("id", ""),
            created=data.get("created", 0),
            model=model,
            choices=[choice_data],
            usage=litellm.Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    @staticmethod
    def _repair_accumulated_tool_calls(
        args_map: Dict[int, str],
        meta_map: Dict[int, Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Try to repair truncated tool-call JSON arguments.

        Returns a list of OpenAI-format tool_call dicts with repaired
        arguments, or an empty list if repair fails for all calls.
        """
        repaired = []
        for idx in sorted(args_map):
            raw = args_map[idx]
            meta = meta_map.get(idx, {})
            # Try parsing as-is first
            try:
                json.loads(raw)
                repaired_json = raw
            except json.JSONDecodeError:
                repaired_json = _attempt_json_repair(raw)
                if repaired_json is None:
                    lib_logger.warning(
                        f"Could not repair tool call args for index {idx} "
                        f"(tool: {meta.get('name', '?')}): {raw[:200]}"
                    )
                    continue
                repaired_json = json.dumps(repaired_json)
                lib_logger.info(
                    f"Repaired truncated tool call args for {meta.get('name', '?')}"
                )
            repaired.append({
                "index": idx,
                "id": meta.get("id", f"call_{idx}"),
                "type": "function",
                "function": {
                    "name": meta.get("name", ""),
                    "arguments": repaired_json,
                },
            })
        return repaired

    async def _anthropic_completion(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        stream: bool,
        anthropic_payload: Optional[Any] = None,
        **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """Direct call to NanoGPT's Anthropic-compatible Messages API."""
        model = kwargs.get("model", "")
        model_bare = model.split("/")[-1] if "/" in model else model

        # If we have the original Anthropic payload, use it directly
        if anthropic_payload:
            payload = anthropic_payload.model_dump(exclude_none=True)
            payload["model"] = model_bare
        else:
            # Convert from OpenAI format
            messages = kwargs.get("messages", [])
            tools = kwargs.get("tools")
            max_tokens = kwargs.get("max_tokens") or 8192
            temperature = kwargs.get("temperature")
            top_p = kwargs.get("top_p")
            stop = kwargs.get("stop")

            system_prompt, anthropic_messages = convert_openai_to_anthropic_messages(messages)
            anthropic_tools = convert_tools_to_anthropic_format(tools)

            payload = {
                "model": model_bare,
                "max_tokens": max_tokens,
                "messages": anthropic_messages,
                "stream": stream,
            }

            if system_prompt:
                payload["system"] = system_prompt
            if anthropic_tools:
                payload["tools"] = anthropic_tools
            if temperature is not None:
                payload["temperature"] = temperature
            if top_p is not None:
                payload["top_p"] = top_p
            if stop:
                payload["stop_sequences"] = stop if isinstance(stop, list) else [stop]

            # Handle reasoning_effort (thinking)
            if "reasoning_effort" in kwargs:
                effort = kwargs["reasoning_effort"]
                budget = 4096  # Default budget
                if effort == "medium":
                    budget = 8192
                elif effort == "high":
                    budget = 16384
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}

        # Build request headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
            "anthropic-beta": ANTHROPIC_BETA_HEADER,
            "user-agent": ANTHROPIC_USER_AGENT,
        }

        url = f"{NANOGPT_API_BASE}/api/v1/messages"

        lib_logger.debug(f"NanoGPT Anthropic request to {model_bare}: {json.dumps(payload, default=str)[:500]}...")

        if stream:
            return self._stream_anthropic_response(client, url, headers, payload, model)
        else:
            return await self._non_stream_anthropic_response(client, url, headers, payload, model)

    async def _stream_anthropic_response(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Handle streaming response from NanoGPT Anthropic-compatible endpoint."""
        created = int(time.time())
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        thinking_text = ""
        sent_thinking = False
        current_tool_calls: Dict[int, Dict[str, Any]] = {}
        tool_index = 0
        input_tokens = 0
        output_tokens = 0

        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.streaming(),
        ) as response:

            if response.status_code >= 400:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="ignore")
                lib_logger.error(f"NanoGPT Anthropic API error {response.status_code}: {error_text[:500]}")
                raise httpx.HTTPStatusError(
                    f"NanoGPT Anthropic API error: {response.status_code}",
                    request=response.request,
                    response=response,
                )

            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue

                data = line[len("data: "):].strip()
                if not data or data == "[DONE]":
                    continue

                try:
                    evt = json.loads(data)
                except json.JSONDecodeError:
                    continue

                event_type = evt.get("type")

                if event_type == "message_start":
                    msg = evt.get("message", {})
                    if msg.get("id"):
                        response_id = msg["id"]
                    usage = msg.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    continue

                if event_type == "content_block_start":
                    block = evt.get("content_block", {})
                    if block.get("type") == "tool_use":
                        current_tool_calls[tool_index] = {
                            "id": block.get("id", ""),
                            "name": self._strip_tool_prefix(block.get("name", "")),
                            "arguments": "",
                        }
                    continue

                if event_type == "content_block_delta":
                    delta_obj = evt.get("delta", {})
                    delta_type = delta_obj.get("type")

                    if delta_type == "text_delta":
                        text = delta_obj.get("text", "")
                        if text:
                            if not sent_thinking and thinking_text:
                                text = f"<think>{thinking_text}</think>{text}"
                                sent_thinking = True

                            yield litellm.ModelResponse(
                                id=response_id,
                                created=created,
                                model=model,
                                object="chat.completion.chunk",
                                choices=[{
                                    "index": 0,
                                    "delta": {"content": text, "role": "assistant"},
                                    "finish_reason": None,
                                }],
                            )

                    elif delta_type == "thinking_delta":
                        thinking_text += delta_obj.get("thinking", "")

                    elif delta_type == "input_json_delta":
                        partial_json = delta_obj.get("partial_json", "")
                        if tool_index in current_tool_calls:
                            current_tool_calls[tool_index]["arguments"] += partial_json
                    continue

                if event_type == "content_block_stop":
                    if tool_index in current_tool_calls:
                        tc = current_tool_calls[tool_index]
                        yield litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=model,
                            object="chat.completion.chunk",
                            choices=[{
                                "index": 0,
                                "delta": {
                                    "tool_calls": [{
                                        "index": tool_index,
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tc["name"],
                                            "arguments": tc["arguments"],
                                        },
                                    }],
                                },
                                "finish_reason": None,
                            }],
                        )
                        tool_index += 1
                    continue

                if event_type == "message_delta":
                    delta_obj = evt.get("delta", {})
                    stop_reason = delta_obj.get("stop_reason", "end_turn")
                    usage = evt.get("usage", {})
                    output_tokens = usage.get("output_tokens", 0)

                    finish_reason_map = {
                        "end_turn": "stop",
                        "stop_sequence": "stop",
                        "tool_use": "tool_calls",
                        "max_tokens": "length",
                    }
                    finish_reason = finish_reason_map.get(stop_reason, "stop")

                    if not sent_thinking and thinking_text:
                        yield litellm.ModelResponse(
                            id=response_id,
                            created=created,
                            model=model,
                            object="chat.completion.chunk",
                            choices=[{
                                "index": 0,
                                "delta": {"content": f"<think>{thinking_text}</think>", "role": "assistant"},
                                "finish_reason": None,
                            }],
                        )

                    final_chunk = litellm.ModelResponse(
                        id=response_id,
                        created=created,
                        model=model,
                        object="chat.completion.chunk",
                        choices=[{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }],
                    )
                    final_chunk.usage = litellm.Usage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    )
                    yield final_chunk
                    break

    async def _non_stream_anthropic_response(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> litellm.ModelResponse:
        """Handle non-streaming response from NanoGPT Anthropic-compatible endpoint."""
        created = int(time.time())

        response = await client.post(
            url,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.non_streaming(),
        )

        if response.status_code >= 400:
            error_text = response.text
            lib_logger.error(f"NanoGPT Anthropic API error {response.status_code}: {error_text[:500]}")
            raise httpx.HTTPStatusError(
                f"NanoGPT Anthropic API error: {response.status_code}",
                request=response.request,
                response=response,
            )

        data = response.json()
        response_id = data.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}")

        # Extract content
        full_text = ""
        thinking_text = ""
        tool_calls = []

        for block in data.get("content", []):
            block_type = block.get("type")
            if block_type == "text":
                full_text += block.get("text", "")
            elif block_type == "thinking":
                thinking_text += block.get("thinking", "")
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": self._strip_tool_prefix(block.get("name", "")),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        # Build message content
        content = ""
        if thinking_text:
            content += f"<think>{thinking_text}</think>"
        content += full_text

        # Build OpenAI ModelResponse
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        choice_data = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content if content else None,
            },
            "finish_reason": "stop" if data.get("stop_reason") == "end_turn" else data.get("stop_reason"),
        }
        if tool_calls:
            choice_data["message"]["tool_calls"] = tool_calls
            choice_data["finish_reason"] = "tool_calls"

        resp = litellm.ModelResponse(
            id=response_id,
            created=created,
            model=model,
            choices=[choice_data],
            usage=litellm.Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
        )
        return resp

    def _strip_tool_prefix(self, name: str) -> str:
        """Strip mcp_ prefix from tool name if present."""
        if name.startswith(TOOL_PREFIX):
            return name[len(TOOL_PREFIX):]
        return name

    # =========================================================================
    # USAGE TRACKING CONFIGURATION
    # =========================================================================

    def get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Return usage reset configuration for NanoGPT credentials.

        NanoGPT uses per_model mode to track usage at the model level,
        with daily and monthly quotas managed via the background job.

        Args:
            credential: The API key (unused, same config for all)

        Returns:
            Configuration with per_model mode
        """
        return {
            "mode": "per_model",
            "window_seconds": 86400,  # 24 hours (daily quota reset)
        }

    # =========================================================================
    # QUOTA GROUPING
    # =========================================================================

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """
        Get the quota group for a model.

        NanoGPT tracks weekly input tokens (60M/week) as the primary quota.
        All models share the same weekly token pool.

        Args:
            model: Model name

        Returns:
            Quota group name
        """
        return "weekly_tokens"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """
        Get all models belonging to a quota group.

        This is used by UsageManager.update_quota_baseline to sync
        request_count, baseline, and cooldowns across all group members.

        Args:
            group: Quota group identifier

        Returns:
            List of model names in the group
        """
        if group == "weekly_tokens":
            return ["_weekly_tokens"]
        return []

    def get_quota_groups(self) -> List[str]:
        """
        Get the list of quota groups for this provider.

        Returns:
            List of quota group names
        """
        return ["weekly_tokens"]

    # =========================================================================
    # MODEL DISCOVERY
    # =========================================================================

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns NanoGPT models from:
        1. Environment variable (NANOGPT_MODELS) - priority
        2. Dynamic discovery from API (filtered by NANOGPT_MODEL_SOURCE)
        3. Hardcoded fallback list

        NANOGPT_MODEL_SOURCE controls which API endpoint is used for discovery:
        - "all" (default): /api/v1/models (canonical, all models)
        - "personalized": /api/personalized/v1/models (user's visible models)
        - "subscription": /api/subscription/v1/models (subscription-only models)

        Also refreshes subscription usage to determine tier.
        """
        models = []
        seen_ids = set()

        # Source 1: Environment variable models (via NANOGPT_MODELS)
        static_models = self.model_definitions.get_all_provider_models("nanogpt")
        if static_models:
            for model in static_models:
                model_id = model.split("/")[-1] if "/" in model else model
                models.append(model)
                seen_ids.add(model_id)
            lib_logger.debug(f"Loaded {len(static_models)} static models for nanogpt")

        # Source 2: Dynamic discovery from API
        model_endpoint = NANOGPT_MODEL_SOURCES[self._model_source]
        auth_header = _MODEL_SOURCE_AUTH_HEADERS[self._model_source]
        try:
            response = await client.get(
                f"{NANOGPT_API_BASE}{model_endpoint}?detailed=true",
                headers={
                    "Authorization" if auth_header == "Bearer" else "x-api-key": api_key
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            dynamic_count = 0
            for model in data.get("data", []):
                model_id = model.get("id", "")
                if model_id and model_id not in seen_ids:
                    # Skip auto-model variants - these are internal routing models
                    if model_id.startswith("auto-model"):
                        continue
                    models.append(f"nanogpt/{model_id}")
                    seen_ids.add(model_id)
                    dynamic_count += 1
                    # Track for quota group sync
                    self._discovered_models.add(model_id)

            if dynamic_count > 0:
                lib_logger.debug(
                    f"Discovered {dynamic_count} models for nanogpt from API"
                )

        except Exception as e:
            lib_logger.debug(f"Dynamic model discovery failed for nanogpt: {e}")

            # Source 3: Fallback to hardcoded models if nothing discovered
            if not models:
                for model_id in NANOGPT_FALLBACK_MODELS:
                    if model_id not in seen_ids:
                        models.append(f"nanogpt/{model_id}")
                        seen_ids.add(model_id)
                lib_logger.debug(
                    f"Using {len(NANOGPT_FALLBACK_MODELS)} fallback models for nanogpt"
                )
                # Track fallback models for quota group sync
                for model_id in NANOGPT_FALLBACK_MODELS:
                    self._discovered_models.add(model_id)

        # Also track static models for quota group sync
        for model in models:
            model_id = model.split("/")[-1] if "/" in model else model
            self._discovered_models.add(model_id)

        # Fetch subscription-only models for quota tracking
        await self._fetch_subscription_models(api_key, client)

        # Refresh subscription usage to get tier info (only if not already cached)
        if api_key not in self._tier_cache:
            await self._refresh_tier_from_api(api_key)

        return models

    async def _fetch_subscription_models(self, api_key: str, client: httpx.AsyncClient):
        """
        Fetch subscription-only models from NanoGPT API.

        These are the models subject to daily/monthly quota limits.
        Non-subscription (paid) models are pay-as-you-go and not limited.
        """
        try:
            response = await client.get(
                f"{NANOGPT_API_BASE}/api/subscription/v1/models?detailed=true",
                headers={"x-api-key": api_key},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            self._subscription_models.clear()
            for model in data.get("data", []):
                model_id = model.get("id", "")
                if model_id and not model_id.startswith("auto-model"):
                    self._subscription_models.add(model_id)

            lib_logger.debug(
                f"Discovered {len(self._subscription_models)} subscription models for nanogpt"
            )
        except Exception as e:
            lib_logger.debug(f"Subscription model discovery failed for nanogpt: {e}")
            # Fall back to treating all discovered models as subscription
            self._subscription_models = self._discovered_models.copy()

    # =========================================================================
    # TIER MANAGEMENT
    # =========================================================================

    async def _refresh_tier_from_api(self, api_key: str) -> Optional[str]:
        """
        Refresh subscription status and cache the tier.

        Args:
            api_key: NanoGPT API key

        Returns:
            Tier name or None if fetch failed
        """
        usage_data = await self.fetch_subscription_usage(api_key)

        if usage_data.get("status") == "success":
            state = usage_data.get("state", "inactive")
            tier = self.get_tier_from_state(state)
            self._tier_cache[api_key] = tier

            daily = usage_data.get("daily", {})
            limits = usage_data.get("limits", {})
            lib_logger.info(
                f"NanoGPT subscription: state={state}, "
                f"daily={daily.get('remaining', 0)}/{limits.get('daily', 0)}"
            )
            return tier

        return None

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """
        Returns the tier name for a credential.

        Uses cached subscription state from API refresh.

        Args:
            credential: The API key

        Returns:
            Tier name or None if not yet discovered
        """
        return self._tier_cache.get(credential)

    # =========================================================================
    # BACKGROUND JOB CONFIGURATION
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Configure periodic subscription usage refresh.

        Returns:
            Background job configuration
        """
        return {
            "interval": self._quota_refresh_interval,
            "name": "nanogpt_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Refresh subscription usage for all credentials in parallel.

        Uses the mixin's refresh_subscription_usage method to avoid code duplication.

        Args:
            usage_manager: UsageManager instance
            credentials: List of API keys
        """
        semaphore = asyncio.Semaphore(QUOTA_FETCH_CONCURRENCY)

        async def refresh_single_credential(
            api_key: str, client: httpx.AsyncClient
        ) -> None:
            async with semaphore:
                try:
                    # Use mixin method for refresh (handles caching internally)
                    # Pass the shared client to respect concurrency control
                    usage_data = await self.refresh_subscription_usage(
                        api_key, credential_identifier=api_key, client=client
                    )

                    if usage_data.get("status") == "success":
                        # Update tier cache
                        state = usage_data.get("state", "inactive")
                        tier = self.get_tier_from_state(state)
                        self._tier_cache[api_key] = tier

                        # Extract weekly token quota data
                        weekly_token_data = usage_data.get("weekly_input_tokens")
                        limits = usage_data.get("limits", {})

                        # Store weekly token quota baseline
                        if weekly_token_data is not None:
                            weekly_token_limit = limits.get("weekly_input_tokens", 0)
                            weekly_token_remaining = weekly_token_data.get("remaining", 0)
                            weekly_token_reset_ts = weekly_token_data.get("reset_at", 0)
                            weekly_token_used = weekly_token_limit - weekly_token_remaining if weekly_token_limit > 0 else 0

                            effectively_exhausted = (
                                weekly_token_limit > 0
                                and weekly_token_remaining <= NANOGPT_EXHAUSTION_TOKEN_THRESHOLD
                                and weekly_token_reset_ts > 0
                            )

                            await usage_manager.update_quota_baseline(
                                api_key,
                                "nanogpt/_weekly_tokens",
                                quota_max_requests=weekly_token_limit,
                                quota_reset_ts=weekly_token_reset_ts if weekly_token_reset_ts > 0 else None,
                                quota_used=weekly_token_used,
                                apply_exhaustion=effectively_exhausted,
                            )

                            if effectively_exhausted:
                                lib_logger.info(
                                    f"NanoGPT weekly token quota effectively exhausted: "
                                    f"{weekly_token_remaining}/{weekly_token_limit} remaining "
                                    f"(threshold={NANOGPT_EXHAUSTION_TOKEN_THRESHOLD})"
                                )
                            else:
                                lib_logger.debug(
                                    f"Updated NanoGPT quota baseline: "
                                    f"weekly_tokens={weekly_token_remaining}/{weekly_token_limit}"
                                )

                except Exception as e:
                    lib_logger.warning(
                        f"Failed to refresh NanoGPT subscription usage: {e}"
                    )

        # Fetch all credentials in parallel using a shared client
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                refresh_single_credential(api_key, client) for api_key in credentials
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
