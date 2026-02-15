# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/anthropic_provider.py

import copy
import hashlib
import json
import os
import time
import logging
import uuid
from typing import Union, AsyncGenerator, List, Dict, Any, Optional
from pathlib import Path

import httpx
import litellm
from litellm.exceptions import RateLimitError

from .provider_interface import ProviderInterface
from .anthropic_auth_base import AnthropicAuthBase
from .provider_cache import create_provider_cache
from ..timeout_config import TimeoutConfig
from ..transaction_logger import ProviderLogger

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# CLAUDE CODE IMPERSONATION CONSTANTS
# =============================================================================

CLAUDE_CODE_VERSION = "2.1.42"
TOOL_PREFIX = "mcp_"

ANTHROPIC_API_BASE = "https://api.anthropic.com"

ANTHROPIC_BETA_FEATURES = ",".join(
    [
        "claude-code-20250219",
        "oauth-2025-04-20",
        "interleaved-thinking-2025-05-14",
        "fine-grained-tool-streaming-2025-05-14",
    ]
)

CLAUDE_CODE_SYSTEM_PREFIX = "You are Claude Code, Anthropic's official CLI for Claude."

# Fallback model list — only used if live fetch fails
FALLBACK_MODELS = [
    "claude-opus-4-6",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
]

# Stop reason mapping: Anthropic -> OpenAI
STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
    "pause_turn": "stop",
}

# Lazy-initialised server-side cache for thinking block signatures.
# Allows us to re-attach signatures when OpenAI-format clients send back
# reasoning_content without the signature (which they can't preserve).
_thinking_sig_cache = None


def _get_thinking_cache():
    global _thinking_sig_cache
    if _thinking_sig_cache is None:
        _thinking_sig_cache = create_provider_cache(
            "anthropic_thinking_signatures",
            memory_ttl_seconds=7200,    # 2 hours in memory
            disk_ttl_seconds=172800,    # 48 hours on disk
        )
    return _thinking_sig_cache


class AnthropicProvider(AnthropicAuthBase, ProviderInterface):
    """
    Anthropic provider using OAuth authentication (Claude Pro/Max).
    Calls Anthropic's Messages API directly, impersonating Claude Code.
    """

    skip_cost_calculation = True

    def __init__(self):
        super().__init__()

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch models live from Anthropic API, falling back to hardcoded list."""
        try:
            access_token = await self.get_access_token(api_key)
            headers = self._build_anthropic_headers(access_token)
            resp = await client.get(
                f"{ANTHROPIC_API_BASE}/v1/models",
                headers=headers,
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = [
                    f"anthropic/{m['id']}"
                    for m in data.get("data", [])
                    if m.get("id")
                ]
                if models:
                    return models
        except Exception as e:
            lib_logger.debug(f"Failed to fetch Anthropic models live: {e}")

        return [f"anthropic/{m}" for m in FALLBACK_MODELS]

    # =========================================================================
    # OPENAI -> ANTHROPIC MESSAGE CONVERSION
    # =========================================================================

    def _openai_messages_to_anthropic(self, messages: List[Dict[str, Any]]) -> tuple:
        """
        Convert OpenAI-format messages to Anthropic Messages API format.
        Returns (system_blocks, anthropic_messages).
        """
        system_blocks = []
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_blocks.append(
                                {"type": "text", "text": block.get("text", "")}
                            )
                continue

            if role == "tool":
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": content if isinstance(content, str) else json.dumps(content),
                }
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    if isinstance(anthropic_messages[-1]["content"], list):
                        anthropic_messages[-1]["content"].append(tool_result)
                    else:
                        anthropic_messages[-1]["content"] = [
                            {
                                "type": "text",
                                "text": anthropic_messages[-1]["content"],
                            },
                            tool_result,
                        ]
                else:
                    anthropic_messages.append(
                        {"role": "user", "content": [tool_result]}
                    )
                continue

            if role == "assistant":
                blocks = []

                reasoning = msg.get("reasoning_content")
                if reasoning:
                    # Try server-side cache first (signature preserved from
                    # the original Anthropic response)
                    cached = self._retrieve_thinking_blocks(reasoning)
                    if cached:
                        blocks.extend(cached)
                    else:
                        # Fallback: inline signature from client (custom clients)
                        thinking_sig = msg.get("thinking_signature")
                        if thinking_sig and len(thinking_sig) >= 100:
                            blocks.append({
                                "type": "thinking",
                                "thinking": reasoning,
                                "signature": thinking_sig,
                            })
                        # else: no signature → drop thinking block,
                        # model generates fresh thinking (cache miss on prefix)

                if isinstance(content, str) and content.strip():
                    blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if (
                                block.get("type") == "text"
                                and block.get("text", "").strip()
                            ):
                                blocks.append({"type": "text", "text": block["text"]})
                            elif block.get("type") == "image_url":
                                url = block.get("image_url", {}).get("url", "")
                                if url.startswith("data:"):
                                    parts = url.split(",", 1)
                                    media_type = (
                                        parts[0]
                                        .replace("data:", "")
                                        .replace(";base64", "")
                                    )
                                    blocks.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": parts[1]
                                                if len(parts) > 1
                                                else "",
                                            },
                                        }
                                    )

                tool_calls = msg.get("tool_calls") or []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    try:
                        input_data = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        input_data = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                            "name": func.get("name", ""),
                            "input": input_data,
                        }
                    )

                if blocks:
                    anthropic_messages.append({"role": "assistant", "content": blocks})
                continue

            # User messages
            if isinstance(content, str):
                if content.strip():
                    anthropic_messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                blocks = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            blocks.append(
                                {"type": "text", "text": block.get("text", "")}
                            )
                        elif block.get("type") == "image_url":
                            url = block.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                parts = url.split(",", 1)
                                media_type = (
                                    parts[0].replace("data:", "").replace(";base64", "")
                                )
                                blocks.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": parts[1] if len(parts) > 1 else "",
                                        },
                                    }
                                )
                if blocks:
                    anthropic_messages.append({"role": "user", "content": blocks})

        return system_blocks, anthropic_messages

    def _retrieve_thinking_blocks(
        self, reasoning_content: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Look up cached thinking blocks with signatures for given thinking content."""
        cache_key = hashlib.sha256(reasoning_content.encode()).hexdigest()
        cached = _get_thinking_cache().retrieve(cache_key)
        if not cached:
            return None
        try:
            blocks_data = json.loads(cached)
            result = [
                {
                    "type": "thinking",
                    "thinking": b["thinking"],
                    "signature": b["signature"],
                }
                for b in blocks_data
                if b.get("signature")
            ]
            return result if result else None
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _openai_tools_to_anthropic(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        result = []
        for tool in tools:
            func = tool.get("function", {})
            result.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object"}),
                }
            )
        return result

    # =========================================================================
    # MCP_ TOOL NAME PREFIXING / STRIPPING
    # =========================================================================

    def _prefix_tool_names(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Add mcp_ prefix to tool names in definitions and messages."""
        payload = copy.deepcopy(payload)

        if payload.get("tools"):
            for tool in payload["tools"]:
                if tool.get("name"):
                    tool["name"] = f"{TOOL_PREFIX}{tool['name']}"

        if payload.get("messages"):
            for msg in payload["messages"]:
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            if block.get("name"):
                                block["name"] = f"{TOOL_PREFIX}{block['name']}"

        return payload

    def _strip_tool_prefix(self, name: str) -> str:
        """Remove mcp_ prefix from a tool name."""
        if name and name.startswith(TOOL_PREFIX):
            return name[len(TOOL_PREFIX) :]
        return name

    # =========================================================================
    # SYSTEM PROMPT HANDLING
    # =========================================================================

    def _inject_system_prompt(
        self, system_blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepend Claude Code identity to system prompt."""
        result = [{"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX}]
        if system_blocks:
            for block in system_blocks:
                result.append(block)
            if len(result) >= 2:
                result[1]["text"] = (
                    CLAUDE_CODE_SYSTEM_PREFIX + "\n\n" + result[1]["text"]
                )
                result.pop(0)
        return result

    # =========================================================================
    # PROMPT CACHING
    # =========================================================================

    def _inject_cache_control(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject cache_control breakpoints into the Anthropic payload to enable
        prompt caching. Marks the last system block, last tool, and the last
        content block of the final message for caching.

        Anthropic caches the full prefix up to each breakpoint. This saves
        ~90% on cached input token costs and reduces latency.
        """
        cache_marker = {"type": "ephemeral"}

        # 1. Cache the last system block (system prompt rarely changes)
        system = payload.get("system")
        if system and isinstance(system, list) and len(system) > 0:
            system[-1]["cache_control"] = cache_marker

        # 2. Cache the last tool definition (tools rarely change)
        tools = payload.get("tools")
        if tools and isinstance(tools, list) and len(tools) > 0:
            tools[-1]["cache_control"] = cache_marker

        # 3. Cache the end of conversation history for multi-turn caching.
        #    Mark the last content block of the second-to-last message
        #    (the turn before the new user message) so the growing
        #    conversation prefix gets cached across turns.
        messages = payload.get("messages")
        if messages and len(messages) >= 2:
            # The last message is the new user turn; cache up to the one before it
            prev_msg = messages[-2]
            content = prev_msg.get("content")
            if isinstance(content, list) and len(content) > 0:
                content[-1]["cache_control"] = cache_marker
            elif isinstance(content, str) and content:
                # Convert string content to block format so we can attach cache_control
                prev_msg["content"] = [
                    {"type": "text", "text": content, "cache_control": cache_marker}
                ]

        return payload

    # =========================================================================
    # ANTHROPIC SSE -> OPENAI CHUNK CONVERSION
    # =========================================================================

    def _anthropic_event_to_openai_chunks(
        self,
        event_type: str,
        data: Dict[str, Any],
        model_id: str,
        stream_state: Dict[str, Any],
    ):
        """
        Convert a single Anthropic SSE event to OpenAI-format chunk(s).
        Yields litellm.ModelResponse-compatible dicts.
        """
        if event_type == "message_start":
            message = data.get("message", {})
            usage = message.get("usage", {})
            stream_state["input_tokens"] = usage.get("input_tokens", 0)
            stream_state["message_id"] = message.get(
                "id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
            )
            return

        if event_type == "content_block_start":
            block = data.get("content_block", {})
            block_type = block.get("type")
            index = data.get("index", 0)
            stream_state["current_block_type"] = block_type
            stream_state["current_block_index"] = index

            if block_type == "thinking":
                stream_state["_block_thinking"] = ""
                stream_state["_block_signature"] = ""

            if block_type == "tool_use":
                tool_id = block.get("id", f"toolu_{uuid.uuid4().hex[:12]}")
                raw_name = block.get("name", "")
                name = self._strip_tool_prefix(raw_name)
                stream_state.setdefault("tool_calls", {})
                stream_state["tool_calls"][index] = {
                    "id": tool_id,
                    "name": name,
                    "arguments": "",
                    "tc_index": len(stream_state["tool_calls"]),
                }
                stream_state["has_tool_calls"] = True
                yield {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": stream_state["tool_calls"][index][
                                            "tc_index"
                                        ],
                                        "id": tool_id,
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": "",
                                        },
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                    "model": model_id,
                    "object": "chat.completion.chunk",
                    "id": stream_state.get(
                        "message_id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
                    ),
                    "created": int(time.time()),
                }
            return

        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                text = delta.get("text", "")
                if text:
                    stream_state["accumulated_text"] = (
                        stream_state.get("accumulated_text", "") + text
                    )
                    yield {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                        "model": model_id,
                        "object": "chat.completion.chunk",
                        "id": stream_state.get(
                            "message_id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
                        ),
                        "created": int(time.time()),
                    }

            elif delta_type == "thinking_delta":
                thinking = delta.get("thinking", "")
                if thinking:
                    stream_state["accumulated_thinking"] = (
                        stream_state.get("accumulated_thinking", "") + thinking
                    )
                    stream_state["_block_thinking"] = (
                        stream_state.get("_block_thinking", "") + thinking
                    )
                    yield {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": thinking},
                                "finish_reason": None,
                            }
                        ],
                        "model": model_id,
                        "object": "chat.completion.chunk",
                        "id": stream_state.get(
                            "message_id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
                        ),
                        "created": int(time.time()),
                    }

            elif delta_type == "input_json_delta":
                partial = delta.get("partial_json", "")
                block_index = data.get("index", 0)
                tc_info = stream_state.get("tool_calls", {}).get(block_index)
                if tc_info and partial:
                    tc_info["arguments"] += partial
                    yield {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": tc_info["tc_index"],
                                            "function": {"arguments": partial},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                        "model": model_id,
                        "object": "chat.completion.chunk",
                        "id": stream_state.get(
                            "message_id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
                        ),
                        "created": int(time.time()),
                    }

            elif delta_type == "signature_delta":
                sig = delta.get("signature", "")
                stream_state["thinking_signature"] = (
                    stream_state.get("thinking_signature", "") + sig
                )
                stream_state["_block_signature"] = (
                    stream_state.get("_block_signature", "") + sig
                )

            return

        if event_type == "content_block_stop":
            if stream_state.get("current_block_type") == "thinking":
                block_thinking = stream_state.pop("_block_thinking", "")
                block_sig = stream_state.pop("_block_signature", "")
                if block_thinking and block_sig:
                    stream_state.setdefault("_thinking_blocks", []).append({
                        "thinking": block_thinking,
                        "signature": block_sig,
                    })
            return

        if event_type == "message_delta":
            delta = data.get("delta", {})
            usage = data.get("usage", {})
            stop_reason = delta.get("stop_reason", "end_turn")
            finish_reason = STOP_REASON_MAP.get(stop_reason, "stop")
            stream_state["finish_reason"] = finish_reason

            output_tokens = usage.get("output_tokens", 0)
            input_tokens = stream_state.get("input_tokens", 0)

            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason,
                    }
                ],
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": stream_state.get(
                    "message_id", f"chatcmpl-{uuid.uuid4().hex[:8]}"
                ),
                "created": int(time.time()),
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }

            # Cache thinking blocks with signatures for multi-turn preservation
            thinking_blocks = stream_state.get("_thinking_blocks")
            if thinking_blocks:
                full_thinking = "".join(b["thinking"] for b in thinking_blocks)
                cache_key = hashlib.sha256(full_thinking.encode()).hexdigest()
                _get_thinking_cache().store(cache_key, json.dumps(thinking_blocks))

            return

    # =========================================================================
    # MAIN API CALL
    # =========================================================================

    def _build_anthropic_headers(self, access_token: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ANTHROPIC_BETA_FEATURES,
            "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION} (external, cli)",
            "x-app": "cli",
            "anthropic-dangerous-direct-browser-access": "true",
        }

    def _build_anthropic_payload(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Build the Anthropic Messages API payload from OpenAI-format kwargs."""
        messages = kwargs.get("messages", [])
        model = kwargs.get("model", "")

        if "/" in model:
            model = model.split("/", 1)[1]

        system_blocks, anthropic_messages = self._openai_messages_to_anthropic(messages)
        system_blocks = self._inject_system_prompt(system_blocks)

        tools = self._openai_tools_to_anthropic(kwargs.get("tools"))

        payload = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 16384),
            "stream": True,
        }

        if system_blocks:
            payload["system"] = system_blocks

        if tools:
            payload["tools"] = tools

        tool_choice = kwargs.get("tool_choice")
        if tool_choice:
            if tool_choice == "auto":
                payload["tool_choice"] = {"type": "auto"}
            elif tool_choice == "required":
                payload["tool_choice"] = {"type": "any"}
            elif tool_choice == "none":
                payload["tool_choice"] = {"type": "none"}
            elif isinstance(tool_choice, dict):
                func_name = tool_choice.get("function", {}).get("name", "")
                if func_name:
                    payload["tool_choice"] = {"type": "tool", "name": func_name}

        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]

        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort and str(reasoning_effort).lower() not in (
            "none",
            "disabled",
            "off",
            "false",
            "disable",
        ):
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._reasoning_effort_to_budget(
                    reasoning_effort, kwargs.get("max_tokens", 16384)
                ),
            }

        payload = self._prefix_tool_names(payload)
        payload = self._inject_cache_control(payload)
        return payload

    def _reasoning_effort_to_budget(self, effort: Any, max_tokens: int) -> int:
        effort_str = str(effort).lower().strip()
        budget_map = {
            "low": 4096,
            "medium": 8192,
            "high": 16384,
        }
        budget = budget_map.get(effort_str)
        if budget is not None:
            return min(budget, max(max_tokens - 1000, 4096))
        try:
            return int(effort_str)
        except (ValueError, TypeError):
            return min(8192, max(max_tokens - 1000, 4096))

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        credential_path = kwargs.pop("credential_identifier")
        transaction_context = kwargs.pop("transaction_context", None)
        model = kwargs.get("model", "")
        file_logger = ProviderLogger(transaction_context)

        async def make_request():
            access_token = await self.get_access_token(credential_path)
            headers = self._build_anthropic_headers(access_token)
            payload = self._build_anthropic_payload(kwargs)

            file_logger.log_request(payload)

            url = f"{ANTHROPIC_API_BASE}/v1/messages?beta=true"
            return client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=TimeoutConfig.streaming(),
            )

        async def stream_handler(response_stream, attempt=1):
            stream_state: Dict[str, Any] = {}
            try:
                async with response_stream as response:
                    if response.status_code >= 400:
                        error_text = await response.aread()
                        error_text = (
                            error_text.decode("utf-8")
                            if isinstance(error_text, bytes)
                            else error_text
                        )

                        if response.status_code == 401 and attempt == 1:
                            lib_logger.warning(
                                "Anthropic returned 401. Forcing token refresh and retrying."
                            )
                            await self._refresh_token(credential_path, force=True)
                            retry_stream = await make_request()
                            async for chunk in stream_handler(retry_stream, attempt=2):
                                yield chunk
                            return

                        if response.status_code == 429:
                            raise RateLimitError(
                                f"Anthropic rate limit: {error_text}",
                                llm_provider="anthropic",
                                model=model,
                                response=response,
                            )

                        error_msg = (
                            f"Anthropic HTTP {response.status_code}: {error_text}"
                        )
                        file_logger.log_error(error_msg)
                        raise httpx.HTTPStatusError(
                            error_msg,
                            request=response.request,
                            response=response,
                        )

                    current_event = None
                    async for line in response.aiter_lines():
                        file_logger.log_response_chunk(line)

                        if line.startswith("event:"):
                            current_event = line[6:].strip()
                            continue

                        if line.startswith("data:"):
                            data_str = (
                                line[5:].strip()
                                if line.startswith("data: ")
                                else line[5:]
                            )
                            if not data_str or data_str == "[DONE]":
                                continue
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            if current_event:
                                for chunk in self._anthropic_event_to_openai_chunks(
                                    current_event, data, model, stream_state
                                ):
                                    yield litellm.ModelResponse(**chunk)

            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                file_logger.log_error(f"Error during Anthropic stream: {e}")
                lib_logger.error(f"Anthropic stream error: {e}", exc_info=True)
                raise

        async def logging_stream_wrapper():
            chunks = []
            try:
                async for chunk in stream_handler(await make_request()):
                    chunks.append(chunk)
                    yield chunk
            finally:
                if chunks:
                    final = self._stream_to_completion_response(chunks)
                    file_logger.log_final_response(final.dict())

        if kwargs.get("stream"):
            return logging_stream_wrapper()
        else:

            async def non_stream():
                all_chunks = [c async for c in logging_stream_wrapper()]
                return self._stream_to_completion_response(all_chunks)

            return await non_stream()

    def _stream_to_completion_response(
        self, chunks: List[litellm.ModelResponse]
    ) -> litellm.ModelResponse:
        if not chunks:
            raise ValueError("No chunks to reassemble")

        final_message = {"role": "assistant"}
        aggregated_tool_calls = {}
        usage_data = None
        finish_reason = "stop"
        first_chunk = chunks[0]

        for chunk in chunks:
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue

            choice = chunk.choices[0]
            if hasattr(choice, "get"):
                delta = choice.get("delta", {})
                choice_finish = choice.get("finish_reason")
            elif hasattr(choice, "delta"):
                delta = choice.delta if choice.delta else {}
                if hasattr(delta, "model_dump"):
                    delta = delta.model_dump(exclude_none=True)
                elif hasattr(delta, "__dict__") and not isinstance(delta, dict):
                    delta = {
                        k: v
                        for k, v in delta.__dict__.items()
                        if not k.startswith("_") and v is not None
                    }
                choice_finish = getattr(choice, "finish_reason", None)
            else:
                delta = {}
                choice_finish = None

            if delta.get("content"):
                final_message.setdefault("content", "")
                final_message["content"] += delta["content"]

            if delta.get("reasoning_content"):
                final_message.setdefault("reasoning_content", "")
                final_message["reasoning_content"] += delta["reasoning_content"]

            tool_calls = delta.get("tool_calls") or []
            for tc in tool_calls:
                idx = tc.get("index", 0)
                if idx not in aggregated_tool_calls:
                    aggregated_tool_calls[idx] = {
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.get("id"):
                    aggregated_tool_calls[idx]["id"] = tc["id"]
                func = tc.get("function", {})
                if func.get("name"):
                    aggregated_tool_calls[idx]["function"]["name"] += func["name"]
                if func.get("arguments"):
                    aggregated_tool_calls[idx]["function"]["arguments"] += func[
                        "arguments"
                    ]

            if choice_finish:
                finish_reason = choice_finish

        for chunk in reversed(chunks):
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage
                break

        if aggregated_tool_calls:
            final_message["tool_calls"] = list(aggregated_tool_calls.values())
            finish_reason = "tool_calls"

        for f in ["content", "tool_calls", "function_call"]:
            if f not in final_message:
                final_message[f] = None

        return litellm.ModelResponse(
            **{
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
        )
