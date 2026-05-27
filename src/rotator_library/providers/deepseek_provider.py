# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from __future__ import annotations

import json
import logging
import os
import time
import hashlib
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import litellm
from litellm.exceptions import RateLimitError

from ..timeout_config import TimeoutConfig
from ..transaction_logger import ProviderLogger
from ..utils.paths import get_cache_dir
from .provider_cache import ProviderCache
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())


DEFAULT_API_BASE = "https://api.deepseek.com"
REASONING_PLACEHOLDER = "Reasoning content unavailable."
HARDCODED_MODELS = [
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-chat",
    "deepseek-reasoner",
]

SUPPORTED_PARAMS = {
    "model",
    "messages",
    "thinking",
    "reasoning_effort",
    "max_tokens",
    "response_format",
    "stop",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "logprobs",
    "top_logprobs",
    "presence_penalty",
    "frequency_penalty",
    "n",
}


class DeepseekProvider(ProviderInterface):
    """First-party DeepSeek provider using the native OpenAI-compatible API."""

    V4_MODELS = {"deepseek-v4-pro", "deepseek-v4-flash"}
    V4_EFFORT_MAP = {
        "low": "high",
        "medium": "high",
        "high": "max",
        "max": "max",
        "xhigh": "max",
    }
    DISABLE_VALUES = {"none", "disable", "disabled", "off"}

    skip_cost_calculation = True
    default_rotation_mode: str = "sequential"
    default_max_concurrent_per_key: int = -1

    def __init__(self):
        self._reasoning_cache: Optional[ProviderCache] = None

    def _get_reasoning_cache(self) -> ProviderCache:
        if self._reasoning_cache is None:
            self._reasoning_cache = ProviderCache(
                cache_file=get_cache_dir(subdir="deepseek") / "reasoning_content.json",
                memory_ttl_seconds=self._env_int("DEEPSEEK_REASONING_CACHE_TTL", 3600),
                disk_ttl_seconds=self._env_int(
                    "DEEPSEEK_REASONING_DISK_TTL", 604800
                ),
                env_prefix="DEEPSEEK_REASONING_CACHE",
            )
        return self._reasoning_cache

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch available DeepSeek models, with a conservative fallback list."""
        try:
            response = await client.get(
                self._models_url(),
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=TimeoutConfig.non_streaming(),
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            models = [m.get("id") for m in data if isinstance(m, dict) and m.get("id")]
            if models:
                return [f"deepseek/{model}" for model in models]
        except Exception as e:
            lib_logger.debug(f"Failed to fetch DeepSeek models, using fallback list: {e}")

        return [f"deepseek/{model}" for model in HARDCODED_MODELS]

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[
        litellm.ModelResponse,
        AsyncGenerator[litellm.ModelResponseStream, None],
    ]:
        api_key = kwargs.pop("credential_identifier")
        transaction_context = kwargs.pop("transaction_context", None)
        file_logger = ProviderLogger(transaction_context)

        api_base = kwargs.pop("api_base", None)
        model = kwargs.get("model", "")
        model_name = self._strip_provider_prefix(model)
        payload = await self._build_payload(model_name=model_name, kwargs=kwargs)
        url = self._chat_url(api_base)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if payload.get("stream") else "application/json",
        }

        file_logger.log_request(payload)

        if payload.get("stream"):
            return self._stream_completion(
                client=client,
                url=url,
                headers=headers,
                payload=payload,
                model=model,
                file_logger=file_logger,
            )

        response = await client.post(
            url,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.non_streaming(),
        )
        await self._raise_for_status(response, model)
        response_data = response.json()
        response_data["model"] = model
        file_logger.log_final_response(response_data)
        await self._store_response_reasoning(response_data)
        return litellm.ModelResponse(**response_data)

    async def _build_payload(
        self, model_name: str, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}
        payload["model"] = model_name

        if "max_completion_tokens" in kwargs and "max_tokens" not in payload:
            payload["max_tokens"] = kwargs["max_completion_tokens"]

        if not payload.get("stream"):
            payload.pop("stream_options", None)

        extra_body = kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            payload.update(extra_body)

        if "messages" in payload and isinstance(payload["messages"], list):
            payload["messages"] = await self._inject_reasoning_content(
                payload["messages"]
            )

        self._apply_thinking_config(payload, model_name, kwargs)
        return payload

    def _apply_thinking_config(
        self, payload: Dict[str, Any], model_name: str, kwargs: Dict[str, Any]
    ) -> None:
        reasoning_effort = kwargs.get("reasoning_effort", payload.get("reasoning_effort"))
        thinking = kwargs.get("thinking", payload.get("thinking"))

        if self._is_disabled(reasoning_effort) or self._is_thinking_disabled(thinking):
            payload["thinking"] = {"type": "disabled"}
            payload.pop("reasoning_effort", None)
            lib_logger.info(
                f"DeepSeek '{model_name}' - thinking disabled "
                f"(reasoning_effort='{reasoning_effort}')"
            )
            return

        if not self._is_v4(model_name):
            return

        payload["thinking"] = {"type": "enabled"}
        effort_key = reasoning_effort.lower() if isinstance(reasoning_effort, str) else None
        mapped = self.V4_EFFORT_MAP.get(effort_key, "max")
        payload["reasoning_effort"] = mapped
        lib_logger.info(
            f"DeepSeek V4 '{model_name}' - thinking enabled, "
            f"reasoning_effort='{mapped}' (input: '{reasoning_effort}')"
        )

    async def _stream_completion(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
        file_logger: ProviderLogger,
    ) -> AsyncGenerator[litellm.ModelResponseStream, None]:
        accumulator: Dict[str, Any] = {
            "message": {"role": "assistant"},
            "tool_calls": {},
            "response": None,
            "usage": None,
            "finish_reason": None,
        }

        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=payload,
            timeout=TimeoutConfig.streaming(),
        ) as response:
            await self._raise_for_status(response, model)

            async for line in response.aiter_lines():
                file_logger.log_response_chunk(line)
                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    lib_logger.warning(f"Could not decode JSON from DeepSeek: {line}")
                    continue

                chunk["model"] = model
                self._accumulate_stream_chunk(accumulator, chunk)
                yield litellm.ModelResponseStream(**chunk)

        final_response = self._final_response_from_accumulator(model, accumulator)
        file_logger.log_final_response(final_response)
        await self._store_response_reasoning(final_response)

    async def _raise_for_status(self, response: httpx.Response, model: str) -> None:
        if response.status_code < 400:
            return

        content = await response.aread()
        error_text = content.decode("utf-8", errors="replace") if content else ""
        if response.status_code == 429:
            raise RateLimitError(
                f"DeepSeek rate limit exceeded: {error_text}",
                llm_provider="deepseek",
                model=model,
                response=response,
            )

        raise httpx.HTTPStatusError(
            f"DeepSeek HTTP {response.status_code}: {error_text}",
            request=response.request,
            response=response,
        )

    def _accumulate_stream_chunk(
        self, accumulator: Dict[str, Any], chunk: Dict[str, Any]
    ) -> None:
        accumulator["response"] = chunk
        if chunk.get("usage"):
            accumulator["usage"] = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            return

        choice = choices[0]
        if choice.get("finish_reason"):
            accumulator["finish_reason"] = choice["finish_reason"]

        delta = choice.get("delta") or {}
        message = accumulator["message"]

        if delta.get("role"):
            message["role"] = delta["role"]
        if "content" in delta and delta["content"] is not None:
            message["content"] = message.get("content", "") + delta["content"]
        if "reasoning_content" in delta and delta["reasoning_content"] is not None:
            message["reasoning_content"] = (
                message.get("reasoning_content", "") + delta["reasoning_content"]
            )

        for tool_call_delta in delta.get("tool_calls") or []:
            index = tool_call_delta.get("index", 0)
            tool_call = accumulator["tool_calls"].setdefault(
                index, {"type": "function", "function": {"name": "", "arguments": ""}}
            )
            if tool_call_delta.get("id"):
                tool_call["id"] = tool_call_delta["id"]
            if tool_call_delta.get("type"):
                tool_call["type"] = tool_call_delta["type"]

            function_delta = tool_call_delta.get("function") or {}
            if function_delta.get("name"):
                tool_call["function"]["name"] += function_delta["name"]
            if function_delta.get("arguments"):
                tool_call["function"]["arguments"] += function_delta["arguments"]

    def _final_response_from_accumulator(
        self, model: str, accumulator: Dict[str, Any]
    ) -> Dict[str, Any]:
        last_chunk = accumulator.get("response") or {}
        message = accumulator["message"]
        tool_calls = accumulator.get("tool_calls") or {}
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

        final_response = {
            "id": last_chunk.get("id", f"chatcmpl-deepseek-{time.time()}"),
            "object": "chat.completion",
            "created": last_chunk.get("created", int(time.time())),
            "model": last_chunk.get("model", model),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": accumulator.get("finish_reason") or "stop",
                }
            ],
        }
        if accumulator.get("usage"):
            final_response["usage"] = accumulator["usage"]
        return final_response

    async def _store_response_reasoning(self, response_data: Dict[str, Any]) -> None:
        choices = response_data.get("choices") or []
        if not choices:
            return

        message = choices[0].get("message") or {}
        await self._store_message_reasoning(message)

    async def _inject_reasoning_content(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        result = []
        restored = 0
        placeholdered = 0

        for message in messages:
            if (
                message.get("role") == "assistant"
                and message.get("tool_calls")
                and not message.get("reasoning_content")
            ):
                cached = None
                for cache_key in self._reasoning_cache_keys(message):
                    cached = await self._get_reasoning_cache().retrieve_async(cache_key)
                    if cached:
                        break
                if cached:
                    message = {**message, "reasoning_content": cached}
                    restored += 1
                else:
                    message = {**message, "reasoning_content": REASONING_PLACEHOLDER}
                    placeholdered += 1
            result.append(message)

        if restored:
            lib_logger.debug(
                f"DeepSeek: Restored reasoning_content for {restored} tool-call messages"
            )
        if placeholdered:
            lib_logger.debug(
                f"DeepSeek: Added placeholder reasoning_content for {placeholdered} tool-call messages"
            )
        return result

    async def _store_message_reasoning(self, message: Dict[str, Any]) -> None:
        if not self._should_cache_reasoning(message):
            return

        for cache_key in self._reasoning_cache_keys(message):
            await self._get_reasoning_cache().store_async(
                cache_key, message["reasoning_content"]
            )
            lib_logger.debug(f"DeepSeek: Cached reasoning_content for {cache_key}")

    def _should_cache_reasoning(self, message: Dict[str, Any]) -> bool:
        return bool(
            message.get("role") == "assistant"
            and message.get("reasoning_content")
            and message.get("reasoning_content") != REASONING_PLACEHOLDER
            and message.get("tool_calls")
        )

    def _reasoning_cache_keys(self, message: Dict[str, Any]) -> List[str]:
        keys = []
        for tool_call in message.get("tool_calls") or []:
            if isinstance(tool_call, dict) and tool_call.get("id"):
                keys.append(f"tool:{tool_call['id']}")

        content_key = self._content_cache_key(message)
        if content_key:
            keys.append(content_key)
        return keys

    def _content_cache_key(self, message: Dict[str, Any]) -> Optional[str]:
        content = message.get("content")
        if content is None or content == "":
            return None
        raw = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return f"content:{digest}"

    def _env_int(self, key: str, default: int) -> int:
        value = os.getenv(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            lib_logger.warning(f"Invalid {key}={value!r}; using {default}")
            return default

    def _is_v4(self, model_name: str) -> bool:
        return model_name.lower() in self.V4_MODELS

    def _is_disabled(self, value: Any) -> bool:
        return isinstance(value, str) and value.lower() in self.DISABLE_VALUES

    def _is_thinking_disabled(self, value: Any) -> bool:
        if isinstance(value, dict):
            return value.get("type") in {"disabled", "disable", "off"}
        return self._is_disabled(value)

    def _strip_provider_prefix(self, model: str) -> str:
        return model.split("/", 1)[1] if "/" in model else model

    def _api_base(self, override: Optional[str] = None) -> str:
        return (override or os.getenv("DEEPSEEK_API_BASE") or DEFAULT_API_BASE).rstrip("/")

    def _chat_url(self, api_base: Optional[str] = None) -> str:
        base = self._api_base(api_base)
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _models_url(self) -> str:
        base = self._api_base()
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        return f"{base}/models"
