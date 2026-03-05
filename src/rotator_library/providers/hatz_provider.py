# SPDX-License-Identifier: LGPL-3.0-only

"""
Hatz AI Provider - Responses API Translation Layer

Provider implementation for the Hatz AI API (https://ai.hatz.ai).
Translates between OpenAI Chat Completions format (client-facing) and
Hatz's Responses API format (backend).

Request flow:
    Client (OpenAI Chat Completions) -> translate -> Hatz Responses API
    Hatz Responses API SSE -> translate -> OpenAI Chat Completions SSE

Authentication: X-API-Key header
Responses endpoint: POST /v1/openai/responses
Models endpoint: GET /v1/chat/models

Environment variables:
    HATZ_API_BASE: API base URL (default: https://ai.hatz.ai/v1)
    HATZ_API_KEY_1, HATZ_API_KEY_2, ...: API keys for rotation
"""

import json
import time
import os
import httpx
import logging
from typing import Union, AsyncGenerator, List, Dict, Any, Optional
from .provider_interface import ProviderInterface
from ..model_definitions import ModelDefinitions
from ..timeout_config import TimeoutConfig
from ..transaction_logger import ProviderLogger
import litellm
from litellm.exceptions import RateLimitError, AuthenticationError

lib_logger = logging.getLogger("rotator_library")


class HatzProvider(ProviderInterface):
    """
    Provider for the Hatz AI Responses API.

    Accepts OpenAI Chat Completions requests from clients, translates them
    to Hatz Responses API format, and translates the streaming/non-streaming
    responses back to OpenAI Chat Completions format.
    """

    skip_cost_calculation = True

    def __init__(self):
        self.api_base = os.environ.get("HATZ_API_BASE", "https://ai.hatz.ai/v1")
        self.model_definitions = ModelDefinitions()

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(
        self, api_key: str, client: httpx.AsyncClient
    ) -> List[str]:
        """
        Fetch available models from Hatz's /chat/models endpoint.

        Hatz returns {"data": [{"name": "model-id", ...}]}.
        Combines with static model definitions from HATZ_MODELS env var.
        """
        models = []
        env_var_ids = set()

        # Source 1: Static model definitions from HATZ_MODELS env var
        static_models = self.model_definitions.get_all_provider_models("hatz")
        if static_models:
            for model in static_models:
                model_name = model.split("/")[-1] if "/" in model else model
                models.append(model if "/" in model else f"hatz/{model}")
                env_var_ids.add(model_name)
            lib_logger.info(
                f"Loaded {len(static_models)} static models for hatz"
            )

        # Source 2: Dynamic discovery from Hatz API
        try:
            models_url = f"{self.api_base.rstrip('/')}/chat/models"
            response = await client.get(
                models_url,
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()
            data = response.json().get("data", [])

            for model_info in data:
                model_id = model_info.get("name", "")
                if model_id and model_id not in env_var_ids:
                    models.append(f"hatz/{model_id}")

            lib_logger.info(
                f"Discovered {len(data)} models from Hatz API "
                f"({len(models)} total after dedup)"
            )
        except httpx.RequestError as e:
            lib_logger.warning(f"Failed to fetch Hatz models: {e}")
        except Exception as e:
            lib_logger.warning(f"Failed to parse Hatz models response: {e}")

        return models

    async def get_auth_header(
        self, credential_identifier: str
    ) -> Dict[str, str]:
        """Return X-API-Key header for Hatz authentication."""
        return {"X-API-Key": credential_identifier}

    # =========================================================================
    # Request Translation: OpenAI Chat Completions -> Hatz Responses API
    # =========================================================================

    def _convert_messages_to_input(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert OpenAI Chat Completions messages to Hatz Responses API input items.

        The Responses API accepts simple role-based messages (user, assistant, system)
        but does NOT accept:
          - role: "tool" (tool results) -> must become function_call_output items
          - assistant messages with tool_calls -> must be split into text message +
            separate function_call items

        Conversions:
            {"role": "user/system", "content": "..."} -> pass through as-is
            {"role": "assistant", "content": "..."} -> pass through (text only)
            {"role": "assistant", "tool_calls": [...]} -> split into:
                - {"role": "assistant", "content": "..."} (if content exists)
                - {"type": "function_call", "id": "...", "call_id": "...",
                   "name": "...", "arguments": "..."} for each tool_call
            {"role": "tool", "tool_call_id": "...", "content": "..."} ->
                {"type": "function_call_output", "call_id": "...", "output": "..."}
        """
        input_items: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")

            if role == "tool":
                # Tool result -> function_call_output
                call_id = msg.get("tool_call_id", "")
                content = msg.get("content", "")
                # Handle content that may be a list of content parts
                if isinstance(content, list):
                    # Join text parts for the output string
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            text_parts.append(part.get("text", str(part)))
                        else:
                            text_parts.append(str(part))
                    content = "\n".join(text_parts)

                input_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": content if isinstance(content, str) else str(content),
                })

            elif role == "assistant":
                tool_calls = msg.get("tool_calls")

                if tool_calls:
                    # Assistant message with tool calls: emit text (if any), then
                    # each tool call as a separate function_call item
                    content = msg.get("content")
                    if content:
                        input_items.append({
                            "type": "message",
                            "role": "assistant",
                            "content": content,
                        })

                    for tc in tool_calls:
                        func = tc.get("function", {})
                        tc_id = tc.get("id", "")
                        input_items.append({
                            "type": "function_call",
                            "id": f"fc_{tc_id}",
                            "call_id": tc_id,
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", "{}"),
                        })
                else:
                    # Simple assistant text message - pass through
                    item: Dict[str, Any] = {"role": "assistant"}
                    content = msg.get("content")
                    if content is not None:
                        item["content"] = content
                    # Preserve reasoning_content if present (for thinking models)
                    if msg.get("reasoning_content"):
                        item["reasoning_content"] = msg["reasoning_content"]
                    input_items.append(item)

            else:
                # user, system, developer, etc. - pass through as-is
                # Only include known safe fields to avoid sending unsupported
                # OpenAI-specific fields (like thinking_signature)
                item = {"role": role}
                content = msg.get("content")
                if content is not None:
                    item["content"] = content
                # Pass through 'name' if present (for multi-user scenarios)
                if msg.get("name"):
                    item["name"] = msg["name"]
                input_items.append(item)

        return input_items

    def _translate_tools(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Convert OpenAI Chat Completions tool format to Responses API format.

        OpenAI: {"type": "function", "function": {"name": "fn", "description": "...", "parameters": {...}}}
        Responses: {"type": "function", "name": "fn", "description": "...", "parameters": {...}}
        """
        if not tools:
            return None

        converted = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                resp_tool: Dict[str, Any] = {"type": "function"}
                if "name" in func:
                    resp_tool["name"] = func["name"]
                if "description" in func:
                    resp_tool["description"] = func["description"]
                if "parameters" in func:
                    resp_tool["parameters"] = func["parameters"]
                converted.append(resp_tool)
            else:
                # Pass through unknown tool types as-is
                converted.append(tool)

        return converted if converted else None

    def _build_responses_payload(self, **kwargs) -> Dict[str, Any]:
        """
        Translate an OpenAI Chat Completions request into a Hatz Responses API payload.

        Mappings:
            messages -> input
            model -> model (strip hatz/ prefix)
            tools -> tools (flatten function wrapper)
            max_tokens -> max_output_tokens
            reasoning_effort -> reasoning.effort
            temperature, top_p, stop, seed, stream -> pass through
        """
        payload: Dict[str, Any] = {}

        # Model (already stripped of hatz/ prefix by caller)
        if "model" in kwargs:
            payload["model"] = kwargs["model"]

        # Messages -> input (convert from OpenAI Chat format to Responses API format)
        if "messages" in kwargs:
            payload["input"] = self._convert_messages_to_input(kwargs["messages"])

        # Stream
        payload["stream"] = True  # Always stream internally

        # Tools: convert from Chat Completions format to Responses API format
        tools = self._translate_tools(kwargs.get("tools"))
        if tools is not None:
            payload["tools"] = tools

        # max_tokens -> max_output_tokens
        # Hatz requires max_output_tokens >= 4096; smaller values cause errors.
        # Only send if the value is >= 4096 (otherwise let Hatz use its default).
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            max_tokens = kwargs["max_tokens"]
            if max_tokens >= 4096:
                payload["max_output_tokens"] = max_tokens

        # reasoning_effort -> reasoning.effort
        if "reasoning_effort" in kwargs and kwargs["reasoning_effort"] is not None:
            payload["reasoning"] = {"effort": kwargs["reasoning_effort"]}

        # Pass-through parameters
        for param in ("temperature", "top_p", "stop", "seed", "tool_choice"):
            if param in kwargs and kwargs[param] is not None:
                payload[param] = kwargs[param]

        return payload

    # =========================================================================
    # Response Translation: Hatz Responses API -> OpenAI Chat Completions
    # =========================================================================

    def _make_openai_chunk(
        self,
        response_id: str,
        model: str,
        created: int,
        delta: Dict[str, Any],
        finish_reason: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build an OpenAI Chat Completions streaming chunk dict."""
        chunk: Dict[str, Any] = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
        }

        if usage is not None:
            # Usage-only chunk has empty choices
            chunk["choices"] = []
            chunk["usage"] = usage
        else:
            chunk["choices"] = [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ]

        return chunk

    def _convert_responses_event(
        self,
        event: Dict[str, Any],
        model: str,
        state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert a single Hatz Responses API SSE event to OpenAI Chat Completions chunks.

        Args:
            event: Parsed JSON from a Responses API SSE line
            model: The model name to use in output (with hatz/ prefix)
            state: Mutable state dict tracking response_id, created, tool indices, etc.

        Returns:
            List of OpenAI-format chunk dicts (may be empty, one, or multiple)
        """
        event_type = event.get("type", "")
        chunks: List[Dict[str, Any]] = []

        resp_id = state.get("response_id", "chatcmpl-hatz")
        created = state.get("created", int(time.time()))

        # --- response.created: extract IDs and emit initial role chunk ---
        if event_type == "response.created":
            resp_data = event.get("response", {})
            state["response_id"] = resp_data.get("id", resp_id)
            state["created"] = resp_data.get("created_at", created)
            resp_id = state["response_id"]
            created = state["created"]

            chunks.append(self._make_openai_chunk(
                resp_id, model, created,
                delta={"role": "assistant"},
            ))

        # --- response.output_text.delta: text content ---
        elif event_type == "response.output_text.delta":
            text_delta = event.get("delta", "")
            if text_delta:
                chunks.append(self._make_openai_chunk(
                    resp_id, model, created,
                    delta={"content": text_delta},
                ))

        # --- response.output_item.added: start of a new output item ---
        elif event_type == "response.output_item.added":
            item = event.get("item", {})
            output_index = event.get("output_index", 0)

            if item.get("type") == "function_call":
                # Assign a tool_call index (0-based, only counting function_calls)
                tc_index = state.get("next_tool_call_index", 0)
                state["next_tool_call_index"] = tc_index + 1
                # Map output_index -> tool_call_index for argument deltas
                state.setdefault("output_to_tc_index", {})[output_index] = tc_index

                # Extract the call_id for OpenAI's tool_call id field
                call_id = item.get("call_id", item.get("id", f"call_{tc_index}"))
                func_name = item.get("name", "")

                # Store the call_id mapping for later reference
                state.setdefault("tc_call_ids", {})[output_index] = call_id

                chunks.append(self._make_openai_chunk(
                    resp_id, model, created,
                    delta={
                        "tool_calls": [{
                            "index": tc_index,
                            "id": call_id,
                            "type": "function",
                            "function": {"name": func_name, "arguments": ""},
                        }]
                    },
                ))

        # --- response.function_call_arguments.delta: tool call argument streaming ---
        elif event_type == "response.function_call_arguments.delta":
            output_index = event.get("output_index", 0)
            arg_delta = event.get("delta", "")
            tc_index = state.get("output_to_tc_index", {}).get(output_index)

            if tc_index is not None and arg_delta:
                chunks.append(self._make_openai_chunk(
                    resp_id, model, created,
                    delta={
                        "tool_calls": [{
                            "index": tc_index,
                            "function": {"arguments": arg_delta},
                        }]
                    },
                ))

        # --- response.completed: emit finish + usage chunk ---
        elif event_type == "response.completed":
            resp_data = event.get("response", {})

            # Determine finish reason
            has_tool_calls = state.get("next_tool_call_index", 0) > 0
            finish_reason = "tool_calls" if has_tool_calls else "stop"

            # Map Responses API usage to Chat Completions usage
            usage_data = resp_data.get("usage", {})
            prompt_tokens = usage_data.get("input_tokens", 0)
            completion_tokens = usage_data.get("output_tokens", 0)
            total_tokens = usage_data.get("total_tokens", 0)

            # Ensure completion_tokens >= 1 so the proxy's streaming wrapper
            # recognizes this as the final chunk and emits finish_reason.
            # Hatz Responses API often returns 0 tokens in streaming mode.
            if completion_tokens == 0:
                completion_tokens = 1
                total_tokens = max(total_tokens, prompt_tokens + 1)

            # Emit a single combined finish + usage chunk.
            # The proxy's _safe_streaming_wrapper requires completion_tokens > 0
            # on the same chunk as finish_reason to emit it correctly.
            chunk: Dict[str, Any] = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            chunks.append(chunk)

        # --- response.incomplete: handle truncated responses ---
        elif event_type == "response.incomplete":
            resp_data = event.get("response", {})

            # Emit finish chunk with "length" reason (truncated)
            usage_data = resp_data.get("usage", {})
            prompt_tokens = usage_data.get("input_tokens", 0)
            completion_tokens = usage_data.get("output_tokens", 0)
            total_tokens = usage_data.get("total_tokens", 0)
            if completion_tokens == 0:
                completion_tokens = 1
                total_tokens = max(total_tokens, prompt_tokens + 1)

            chunk: Dict[str, Any] = {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "length",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            chunks.append(chunk)

        # Other event types (output_item.done, etc.) are ignored
        return chunks

    def _convert_non_streaming_response(
        self, response_data: Dict[str, Any], model: str
    ) -> Dict[str, Any]:
        """
        Convert a non-streaming Hatz Responses API response to OpenAI Chat Completions format.

        Maps output items (message content + function_calls) to choices[0].message.
        """
        content_parts = []
        tool_calls = []
        tc_index = 0

        for item in response_data.get("output", []):
            item_type = item.get("type", "")

            if item_type == "message":
                # Extract text from content blocks
                for block in item.get("content", []):
                    if block.get("type") == "output_text":
                        content_parts.append(block.get("text", ""))

            elif item_type == "function_call":
                call_id = item.get("call_id", item.get("id", f"call_{tc_index}"))
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                })
                tc_index += 1

        # Build message
        message: Dict[str, Any] = {"role": "assistant"}
        content = "".join(content_parts)
        message["content"] = content if content else None

        if tool_calls:
            message["tool_calls"] = tool_calls

        # Determine finish reason
        finish_reason = "tool_calls" if tool_calls else "stop"

        # Map usage
        raw_usage = response_data.get("usage", {})
        usage = {
            "prompt_tokens": raw_usage.get("input_tokens", 0),
            "completion_tokens": raw_usage.get("output_tokens", 0),
            "total_tokens": raw_usage.get("total_tokens", 0),
        }

        return {
            "id": response_data.get("id", "chatcmpl-hatz"),
            "object": "chat.completion",
            "created": response_data.get("created_at", int(time.time())),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }

    # =========================================================================
    # Stream-to-Completion Reassembly
    # =========================================================================

    def _stream_to_completion_response(
        self, chunks: List[litellm.ModelResponse]
    ) -> litellm.ModelResponse:
        """
        Reassemble streaming OpenAI chunks into a single completion response.
        Used for non-streaming mode (which internally uses streaming).
        """
        if not chunks:
            return litellm.ModelResponse(
                id="chatcmpl-hatz-empty",
                object="chat.completion",
                model="unknown",
                choices=[{
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }],
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            )

        first_chunk = chunks[0]
        content_parts = []
        tool_calls_by_index: Dict[int, Dict[str, Any]] = {}
        usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        chunk_finish_reason = None
        reasoning_content_parts = []

        def _get(obj, key, default=None):
            """Get a value from either a dict or an object attribute."""
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        for chunk in chunks:
            for choice in chunk.choices:
                delta = _get(choice, "delta")
                if not delta:
                    continue

                # Accumulate text content
                content = _get(delta, "content")
                if content:
                    content_parts.append(content)

                # Accumulate reasoning content
                reasoning = _get(delta, "reasoning_content")
                if reasoning:
                    reasoning_content_parts.append(reasoning)

                # Accumulate tool calls
                tc_list = _get(delta, "tool_calls")
                if tc_list:
                    for tc in tc_list:
                        idx = _get(tc, "index", 0)
                        if idx not in tool_calls_by_index:
                            tool_calls_by_index[idx] = {
                                "id": _get(tc, "id", f"call_{idx}"),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        tc_id = _get(tc, "id")
                        if tc_id:
                            tool_calls_by_index[idx]["id"] = tc_id
                        tc_func = _get(tc, "function")
                        if tc_func:
                            tc_name = _get(tc_func, "name")
                            if tc_name:
                                tool_calls_by_index[idx]["function"]["name"] = tc_name
                            tc_args = _get(tc_func, "arguments")
                            if tc_args:
                                tool_calls_by_index[idx]["function"][
                                    "arguments"
                                ] += tc_args

                # Track finish reason
                fr = _get(choice, "finish_reason")
                if fr:
                    chunk_finish_reason = fr

            # Track usage if present
            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
                pt = _get(usage, "prompt_tokens", 0)
                ct = _get(usage, "completion_tokens", 0)
                tt = _get(usage, "total_tokens", 0)
                if pt:
                    usage_data["prompt_tokens"] = pt
                if ct:
                    usage_data["completion_tokens"] = ct
                if tt:
                    usage_data["total_tokens"] = tt

        # Build final message
        final_message: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts) if content_parts else None,
        }

        if reasoning_content_parts:
            final_message["reasoning_content"] = "".join(reasoning_content_parts)

        aggregated_tool_calls = [
            tool_calls_by_index[idx]
            for idx in sorted(tool_calls_by_index.keys())
        ]
        if aggregated_tool_calls:
            final_message["tool_calls"] = aggregated_tool_calls

        # Determine finish_reason
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
            "choices": [{
                "index": 0,
                "message": final_message,
                "finish_reason": finish_reason,
            }],
            "usage": usage_data,
        }

        return litellm.ModelResponse(**final_response_data)

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Execute a chat completion request against Hatz's Responses API.

        Translates OpenAI Chat Completions requests to Hatz Responses API format,
        streams the response, and translates SSE events back to OpenAI format.

        For non-streaming requests, internally uses streaming and reassembles.
        """
        credential = kwargs.pop("credential_identifier")
        transaction_context = kwargs.pop("transaction_context", None)
        model = kwargs["model"]

        # Create provider logger from transaction context
        file_logger = ProviderLogger(transaction_context)

        async def make_request():
            """Prepare and send the API request to Hatz Responses API."""
            # Strip provider prefix: "hatz/anthropic.claude-opus-4-6" -> "anthropic.claude-opus-4-6"
            model_name = model.split("/")[-1] if "/" in model else model
            kwargs_with_stripped_model = {**kwargs, "model": model_name}

            # Build Responses API payload from Chat Completions params
            payload = self._build_responses_payload(**kwargs_with_stripped_model)

            headers = {
                "X-API-Key": credential,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }

            # Responses API endpoint
            url = f"{self.api_base.rstrip('/')}/openai/responses"

            # Log request
            file_logger.log_request(payload)
            lib_logger.debug(f"Hatz Responses API Request URL: {url}")

            return client.stream(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=TimeoutConfig.streaming(),
            )

        async def stream_handler(response_stream):
            """Handle the Responses API SSE stream, translate to OpenAI chunks."""
            # Mutable state for tracking across SSE events
            state: Dict[str, Any] = {
                "response_id": "chatcmpl-hatz",
                "created": int(time.time()),
                "next_tool_call_index": 0,
                "output_to_tc_index": {},
                "tc_call_ids": {},
            }

            try:
                async with response_stream as response:
                    # Check for HTTP errors
                    if response.status_code >= 400:
                        error_text = await response.aread()
                        error_text = (
                            error_text.decode("utf-8")
                            if isinstance(error_text, bytes)
                            else error_text
                        )

                        if response.status_code == 401:
                            raise AuthenticationError(
                                f"Hatz authentication failed: {error_text}",
                                llm_provider="hatz",
                                model=model,
                                response=response,
                            )
                        elif response.status_code == 429:
                            raise RateLimitError(
                                f"Hatz rate limit exceeded: {error_text}",
                                llm_provider="hatz",
                                model=model,
                                response=response,
                            )
                        else:
                            error_msg = (
                                f"Hatz HTTP {response.status_code} error: {error_text}"
                            )
                            file_logger.log_error(error_msg)
                            raise httpx.HTTPStatusError(
                                f"HTTP {response.status_code}: {error_text}",
                                request=response.request,
                                response=response,
                            )

                    # Process Responses API SSE events
                    async for line in response.aiter_lines():
                        file_logger.log_response_chunk(line)

                        if not line.startswith("data:"):
                            continue

                        # Handle both "data:" and "data: " formats
                        if line.startswith("data: "):
                            data_str = line[6:]
                        else:
                            data_str = line[5:]

                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            lib_logger.warning(
                                f"Could not decode JSON from Hatz: {line}"
                            )
                            continue

                        # Translate Responses API event -> OpenAI chunks
                        openai_chunks = self._convert_responses_event(
                            event, model, state
                        )
                        for chunk_dict in openai_chunks:
                            yield litellm.ModelResponse(**chunk_dict)

            except httpx.HTTPStatusError:
                raise
            except Exception as e:
                file_logger.log_error(f"Error during Hatz stream processing: {e}")
                lib_logger.error(
                    f"Error during Hatz stream processing: {e}", exc_info=True
                )
                raise

        async def logging_stream_wrapper():
            """Wrap the stream to log the final reassembled response."""
            openai_chunks = []
            try:
                async for chunk in stream_handler(await make_request()):
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
