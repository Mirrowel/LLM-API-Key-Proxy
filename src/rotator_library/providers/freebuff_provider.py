# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Freebuff Provider

Provider for Freebuff (https://freebuff.com) - a free AI model hosting platform.
Implements custom request handling due to Freebuff's unique session/run lifecycle.

Features:
- Free session management (create, poll for active, refresh)
- Agent run lifecycle (start, finish, rotate)
- Model-to-agent mapping (dynamic refresh from Codebuff source)
- Multi-token rotation with round-robin selection
- OpenAI-compatible streaming and non-streaming responses
- Automatic retry on session/run invalidation

Environment Variables:
- FREEBUFF_API_BASE: Override base URL (default: https://codebuff.com)
- FREEBUFF_MODELS: Custom model list (JSON array or dict)
"""

import asyncio
import copy
import json
import logging
import os
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import litellm
from litellm.exceptions import RateLimitError

from ..model_definitions import ModelDefinitions
from ..timeout_config import TimeoutConfig
from ..transaction_logger import ProviderLogger
from .freebuff_auth_base import FreebuffAuthBase, _generate_client_session_id
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")
lib_logger.propagate = False
if not lib_logger.handlers:
    lib_logger.addHandler(logging.NullHandler())

SUPPORTED_PARAMS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "stream",
    "tools",
    "tool_choice",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "stop",
    "seed",
    "response_format",
}


class FreebuffProvider(FreebuffAuthBase, ProviderInterface):
    """
    Freebuff provider with custom session/run management.

    Uses Freebuff's free-tier API which requires:
    1. An active free session (may involve queuing)
    2. An active agent run for the target model
    3. codebuff_metadata injection into every request

    Supports multi-token rotation for higher throughput.
    """

    skip_cost_calculation = True
    provider_env_name = "freebuff"

    tier_priorities = {
        "active": 1,
        "queued": 2,
        "no-session": 3,
    }
    default_tier_priority = 3

    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, credential: str, client: httpx.AsyncClient) -> List[str]:
        models = []
        seen_ids: set = set()

        static_models = self.model_definitions.get_all_provider_models("freebuff")
        if static_models:
            for model in static_models:
                model_id = model.split("/")[-1] if "/" in model else model
                models.append(model)
                seen_ids.add(model_id)
            lib_logger.debug(f"Freebuff: loaded {len(static_models)} static models")

        await self.refresh_model_mapping(client)

        for model_id in self.get_available_models():
            if model_id not in seen_ids:
                models.append(f"freebuff/{model_id}")
                seen_ids.add(model_id)

        if not models:
            for model_id in self.get_available_models():
                if model_id not in seen_ids:
                    models.append(f"freebuff/{model_id}")
                    seen_ids.add(model_id)

        return models

    def _clean_tool_schemas(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned = []
        for tool in tools:
            cleaned_tool = copy.deepcopy(tool)
            if "function" in cleaned_tool:
                func = cleaned_tool["function"]
                func.pop("strict", None)
                if "parameters" in func and isinstance(func["parameters"], dict):
                    params = func["parameters"]
                    params.pop("additionalProperties", None)
                    if "properties" in params:
                        self._clean_schema_properties(params["properties"])
                    self._resolve_refs(params)
            cleaned_tools.append(cleaned_tool)
        return cleaned

    def _clean_schema_properties(self, properties: Dict[str, Any]) -> None:
        for prop_name, prop_schema in properties.items():
            if isinstance(prop_schema, dict):
                prop_schema.pop("strict", None)
                prop_schema.pop("additionalProperties", None)
                if "properties" in prop_schema:
                    self._clean_schema_properties(prop_schema["properties"])
                if "items" in prop_schema and isinstance(prop_schema["items"], dict):
                    self._clean_schema_properties({"item": prop_schema["items"]})

    def _resolve_refs(self, schema: Dict[str, Any]) -> None:
        defs = {}
        for key in ("definitions", "$defs"):
            if key in schema and isinstance(schema[key], dict):
                defs.update(schema[key])
        if not defs:
            return

        def _resolve(node: Any, depth: int = 0) -> Any:
            if depth > 12 or not isinstance(node, dict):
                return node
            ref = node.get("$ref", "")
            if ref and len(node) == 1:
                if ref.startswith("#/definitions/"):
                    name = ref[len("#/definitions/"):]
                elif ref.startswith("#/$defs/"):
                    name = ref[len("#/$defs/"):]
                else:
                    return node
                if name in defs:
                    return _resolve(copy.deepcopy(defs[name]), depth + 1)
            return node

        for prop_schema in schema.get("properties", {}).values():
            if isinstance(prop_schema, dict):
                resolved = _resolve(prop_schema)
                if isinstance(resolved, dict):
                    prop_schema.update(resolved)
                    prop_schema.pop("$ref", None)

        schema.pop("definitions", None)
        schema.pop("$defs", None)

    def _build_request_payload(
        self,
        model: str,
        run_id: str,
        session_instance_id: Optional[str],
        **kwargs,
    ) -> Dict[str, Any]:
        payload = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}
        payload["model"] = model
        payload["stream"] = True

        if payload.get("tools") and isinstance(payload["tools"], list) and payload["tools"]:
            payload["tools"] = self._clean_tool_schemas(payload["tools"])

        metadata = kwargs.get("codebuff_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["run_id"] = run_id
        metadata["cost_mode"] = "free"
        metadata["client_id"] = _generate_client_session_id()
        if session_instance_id:
            metadata["freebuff_instance_id"] = session_instance_id
        payload["codebuff_metadata"] = metadata

        return payload

    def _convert_chunk_to_openai(self, chunk: Dict[str, Any], model_id: str):
        if not isinstance(chunk, dict):
            return
        choices = chunk.get("choices", [])
        usage_data = chunk.get("usage")

        if choices and usage_data:
            yield {
                "choices": choices,
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk.get("id", f"chatcmpl-freebuff-{time.time()}"),
                "created": chunk.get("created", int(time.time())),
            }
            yield {
                "choices": [],
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk.get("id", f"chatcmpl-freebuff-{time.time()}"),
                "created": chunk.get("created", int(time.time())),
                "usage": {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                },
            }
            return

        if usage_data:
            yield {
                "choices": [],
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk.get("id", f"chatcmpl-freebuff-{time.time()}"),
                "created": chunk.get("created", int(time.time())),
                "usage": {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                },
            }
            return

        if choices:
            yield {
                "choices": choices,
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk.get("id", f"chatcmpl-freebuff-{time.time()}"),
                "created": chunk.get("created", int(time.time())),
            }

    def _stream_to_completion_response(
        self, chunks: List[litellm.ModelResponse]
    ) -> litellm.ModelResponse:
        if not chunks:
            raise ValueError("No chunks provided for reassembly")

        final_message: Dict[str, Any] = {"role": "assistant"}
        aggregated_tool_calls: Dict[int, Dict[str, Any]] = {}
        usage_data = None
        chunk_finish_reason = None
        first_chunk = chunks[0]

        for chunk in chunks:
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.get("delta", {}) if hasattr(choice, "get") else {}

            if "content" in delta and delta["content"] is not None:
                final_message.setdefault("content", "")
                final_message["content"] += delta["content"]

            if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                final_message.setdefault("reasoning_content", "")
                final_message["reasoning_content"] += delta["reasoning_content"]

            if "tool_calls" in delta and delta["tool_calls"]:
                for tc_chunk in delta["tool_calls"]:
                    index = tc_chunk.get("index", 0)
                    if index not in aggregated_tool_calls:
                        aggregated_tool_calls[index] = {
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if "id" in tc_chunk:
                        aggregated_tool_calls[index]["id"] = tc_chunk["id"]
                    if "type" in tc_chunk:
                        aggregated_tool_calls[index]["type"] = tc_chunk["type"]
                    if "function" in tc_chunk:
                        fn = tc_chunk["function"]
                        if fn.get("name"):
                            aggregated_tool_calls[index]["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            aggregated_tool_calls[index]["function"]["arguments"] += fn["arguments"]

            if "function_call" in delta and delta["function_call"] is not None:
                if "function_call" not in final_message:
                    final_message["function_call"] = {"name": "", "arguments": ""}
                if delta["function_call"].get("name"):
                    final_message["function_call"]["name"] += delta["function_call"]["name"]
                if delta["function_call"].get("arguments"):
                    final_message["function_call"]["arguments"] += delta["function_call"]["arguments"]

            if hasattr(choice, "get") and choice.get("finish_reason"):
                chunk_finish_reason = choice["finish_reason"]

        for chunk in reversed(chunks):
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage
                break

        if aggregated_tool_calls:
            final_message["tool_calls"] = list(aggregated_tool_calls.values())

        for field in ("content", "tool_calls", "function_call"):
            final_message.setdefault(field, None)

        if aggregated_tool_calls:
            finish_reason = "tool_calls"
        elif chunk_finish_reason:
            finish_reason = chunk_finish_reason
        else:
            finish_reason = "stop"

        final_response_data = {
            "id": first_chunk.id,
            "object": "chat.completion",
            "created": first_chunk.created,
            "model": first_chunk.model,
            "choices": [
                {
                    "index": 0,
                    "message": final_message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage_data,
        }
        return litellm.ModelResponse(**final_response_data)

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential_path = kwargs.pop("credential_identifier")
        transaction_context = kwargs.pop("transaction_context", None)
        model = kwargs["model"]

        file_logger = ProviderLogger(transaction_context)

        model_name = model.split("/")[-1] if "/" in model else model
        agent_id = self.get_agent_for_model(model_name)
        if not agent_id:
            raise ValueError(f"Freebuff: unsupported model '{model_name}'")

        await self.refresh_model_mapping(client)

        async def make_request():
            pool = self._get_pool(credential_path)
            if not pool:
                raise ValueError(f"Freebuff: no token pool for credential {credential_path}")

            session_instance = await self.ensure_session(client, pool)
            run = await self.ensure_run(client, pool, agent_id)
            self.acquire_run(run)

            payload = self._build_request_payload(
                model_name,
                run.run_id,
                session_instance,
                **kwargs,
            )

            headers = {
                "Authorization": f"Bearer {pool.token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "User-Agent": "ai-sdk/openai-compatible/1.0.25/codebuff",
            }

            url = f"{self.base_url}/api/v1/chat/completions"
            file_logger.log_request(payload)
            lib_logger.debug(f"Freebuff request: model={model_name}, agent={agent_id}, run={run.run_id}")

            return client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=TimeoutConfig.streaming(),
            ), pool, run

        async def stream_handler(response_stream, pool, run, attempt=1):
            try:
                async with response_stream as response:
                    if response.status_code >= 400:
                        error_bytes = await response.aread()
                        error_text = error_bytes.decode("utf-8") if isinstance(error_bytes, bytes) else error_bytes

                        if self.is_session_invalid_error(response.status_code, error_text):
                            lib_logger.info(f"Freebuff [{pool.name}]: session invalid, refreshing and retrying")
                            self.invalidate_session(pool, error_text)
                            if attempt < 2:
                                retry_stream, retry_pool, retry_run = await make_request()
                                async for chunk in stream_handler(
                                    retry_stream, retry_pool, retry_run, attempt + 1
                                ):
                                    yield chunk
                                return

                        if self.is_run_invalid_error(response.status_code, error_text):
                            lib_logger.info(f"Freebuff [{pool.name}]: run {run.run_id} invalid, rotating")
                            self.invalidate_run(pool, run, error_text)
                            if attempt < 2:
                                retry_stream, retry_pool, retry_run = await make_request()
                                async for chunk in stream_handler(
                                    retry_stream, retry_pool, retry_run, attempt + 1
                                ):
                                    yield chunk
                                return

                        if response.status_code == 401:
                            from datetime import timedelta
                            pool.cooldown_until = time.monotonic() + 1800
                            self.invalidate_session(pool, "auth rejected")
                            self.release_run(pool, run)
                            raise RateLimitError(
                                f"Freebuff auth rejected: {error_text}",
                                llm_provider="freebuff",
                                model=model,
                                response=response,
                            )

                        if response.status_code == 429:
                            self.release_run(pool, run)
                            raise RateLimitError(
                                f"Freebuff rate limit: {error_text}",
                                llm_provider="freebuff",
                                model=model,
                                response=response,
                            )

                        self.release_run(pool, run)
                        raise httpx.HTTPStatusError(
                            f"HTTP {response.status_code}: {error_text}",
                            request=response.request,
                            response=response,
                        )

                    async for line in response.aiter_lines():
                        file_logger.log_response_chunk(line)
                        if line.startswith("data:"):
                            data_str = line[6:] if line.startswith("data: ") else line[5:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                for openai_chunk in self._convert_chunk_to_openai(chunk, model):
                                    yield litellm.ModelResponse(**openai_chunk)
                            except json.JSONDecodeError:
                                lib_logger.warning(f"Freebuff: could not decode JSON: {line}")

                    self.release_run(pool, run)

            except httpx.HTTPStatusError:
                raise
            except RateLimitError:
                raise
            except Exception as e:
                file_logger.log_error(f"Freebuff stream error: {e}")
                lib_logger.error(f"Freebuff stream error: {e}", exc_info=True)
                raise

        async def logging_stream_wrapper():
            openai_chunks = []
            try:
                stream, pool, run = await make_request()
                async for chunk in stream_handler(stream, pool, run):
                    openai_chunks.append(chunk)
                    yield chunk
            finally:
                if openai_chunks:
                    final_response = self._stream_to_completion_response(openai_chunks)
                    file_logger.log_final_response(final_response.dict())

        if kwargs.get("stream"):
            return logging_stream_wrapper()
        else:
            async def non_stream_wrapper():
                chunks = [chunk async for chunk in logging_stream_wrapper()]
                return self._stream_to_completion_response(chunks)

            return await non_stream_wrapper()
