# src/rotator_library/providers/codex_cli_provider.py

from __future__ import annotations

import copy
import json
import httpx
import logging
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator, Union, Optional, Tuple

import litellm
from litellm.exceptions import RateLimitError

from .provider_interface import ProviderInterface
from .codex_auth_base import CodexAuthBase
from ..model_definitions import ModelDefinitions
from ..transaction_logger import ProviderLogger
from ..timeout_config import TimeoutConfig
from ..error_handler import extract_retry_after_from_body
from ..utils.paths import get_cache_dir

lib_logger = logging.getLogger("rotator_library")

# Codex API endpoint
CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_API_PATH = "/codex/responses"

# Supported Codex models
HARDCODED_MODELS = [
    "gpt-5.2-codex-low",
    "gpt-5.2-codex-medium",
    "gpt-5.2-codex-high",
    "gpt-5.2-codex-xhigh",
    "gpt-5.1-codex-max-low",
    "gpt-5.1-codex-max-medium",
    "gpt-5.1-codex-max-high",
    "gpt-5.1-codex-max-xhigh",
    "gpt-5.1-codex-low",
    "gpt-5.1-codex-medium",
    "gpt-5.1-codex-high",
    "gpt-5.1-codex-mini-medium",
    "gpt-5.1-codex-mini-high",
]

# Model variant to reasoning effort mapping
VARIANT_TO_EFFORT = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}


def _get_codex_cache_dir() -> Path:
    """Get the Codex cache directory."""
    return get_cache_dir(subdir="codex_instructions")


class CodexCliProvider(CodexAuthBase, ProviderInterface):
    """Provider for OpenAI Codex CLI using ChatGPT Plus/Pro OAuth."""

    skip_cost_calculation = True
    default_rotation_mode = "sequential"
    provider_env_name = "codex"

    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()
        self._instructions_cache: Dict[str, Tuple[str, float]] = {}  # model -> (instructions, expiry)
        self._cache_ttl = 900  # 15 minutes

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, credential: str, client: httpx.AsyncClient) -> List[str]:
        """
        Return list of available Codex models.

        Models are loaded from three sources:
        1. Environment variable models (via CODEX_MODELS) - ALWAYS included
        2. Hardcoded models (fallback list) - added only if ID not in env vars
        """
        models = []
        env_var_ids = set()  # Track IDs from env vars to prevent hardcoded duplicates

        # Source 1: Load environment variable models (ALWAYS include ALL of them)
        static_models = self.model_definitions.get_all_provider_models("codex")
        if static_models:
            for model in static_models:
                models.append(model)
                # Extract model name from "codex/ModelName" format
                model_name = model.split("/")[-1] if "/" in model else model
                # Track the ID to prevent hardcoded duplicates
                env_var_ids.add(model_name)
            lib_logger.info(
                f"Loaded {len(static_models)} static models for codex from environment variables"
            )

        # Source 2: Add hardcoded models that weren't in env vars
        for model_name in HARDCODED_MODELS:
            if model_name not in env_var_ids:
                models.append(f"codex/{model_name}")

        return models

    def _parse_model_variant(self, model: str) -> Tuple[str, str]:
        """
        Parse model name and variant from full model string.

        Examples:
            "gpt-5.2-codex-medium" -> ("gpt-5.2-codex", "medium")
            "codex/gpt-5.1-codex-max-high" -> ("gpt-5.1-codex-max", "high")
        """
        # Strip provider prefix
        clean_model = model.split("/")[-1]

        # Check for -max suffix
        if "-max-" in clean_model:
            parts = clean_model.split("-max-")
            base = parts[0] + "-max"
            variant = parts[1] if len(parts) > 1 else "medium"
        else:
            # Find last hyphen (variant separator)
            last_hyphen = clean_model.rfind("-")
            if last_hyphen > 0:
                base = clean_model[:last_hyphen]
                variant = clean_model[last_hyphen + 1 :]
            else:
                base = clean_model
                variant = "medium"

        return base, variant

    async def _fetch_codex_instructions(self, model: str) -> str:
        """
        Fetch Codex system instructions from GitHub.

        Model mapping:
        - gpt-5.2-codex -> gpt-5.2-codex_prompt.md
        - gpt-5.1-codex-max -> gpt-5.1-codex-max_prompt.md
        - gpt-5.1-codex -> gpt_5_codex_prompt.md
        - gpt-5.1-codex-mini -> gpt_5_codex_mini_prompt.md

        From: https://github.com/openai/codex/tree/main/codex-rs/core
        """
        # Check cache
        if model in self._instructions_cache:
            instructions, expiry = self._instructions_cache[model]
            if time.time() < expiry:
                return instructions

        # Map model to prompt file
        model_to_prompt = {
            "gpt-5.2-codex": "gpt-5.2-codex_prompt.md",
            "gpt-5.1-codex-max": "gpt-5.1-codex-max_prompt.md",
            "gpt-5.1-codex": "gpt_5_codex_prompt.md",
            "gpt-5.1-codex-mini": "gpt_5_codex_mini_prompt.md",
        }

        prompt_file = model_to_prompt.get(model)
        if not prompt_file:
            lib_logger.warning(f"No prompt file mapping for model: {model}")
            return ""

        # GitHub raw URL
        url = f"https://raw.githubusercontent.com/openai/codex/refs/heads/main/codex-rs/core/{prompt_file}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                response.raise_for_status()
                instructions = response.text

                # Cache the result
                self._instructions_cache[model] = (instructions, time.time() + self._cache_ttl)

                return instructions

        except Exception as e:
            lib_logger.warning(f"Failed to fetch Codex instructions for {model}: {e}")
            return ""

    def _transform_request(
        self,
        kwargs: Dict[str, Any],
        model_name: str,
        variant: str,
        instructions: str,
    ) -> Dict[str, Any]:
        """
        Transform OpenAI request to Codex API format.

        Codex API format (from openai/codex):
        - input: Array of input items (messages converted to input format)
        - instructions: Top-level string field (REQUIRED)
        - model: Normalized model name
        - store: Must be false
        - stream: Should be true
        - reasoning: Object with 'effort' property
        - include: Array of fields to include
        - text: Object with 'verbosity' property

        Note: Codex API does NOT allow system messages in input array.
        System prompts should be provided via the 'instructions' field.
        """
        messages = copy.deepcopy(kwargs.get("messages", []))

        # Filter message IDs and convert OpenAI messages to Codex input format
        input_items = []
        for msg in messages:
            # Remove 'id' field
            msg_data = {k: v for k, v in msg.items() if k != "id"}

            # Skip system messages - Codex API doesn't allow them
            # System prompts should be provided via 'instructions' field
            if msg_data.get("role") == "system":
                continue

            # Convert OpenAI message format to Codex input item format
            # OpenAI: {"role": "user", "content": "..."}
            # Codex: {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "..."}]}
            input_item = {
                "type": "message",
                "role": msg_data.get("role"),
                "content": [],
            }

            # Handle content (string or array)
            content = msg_data.get("content", "")
            if isinstance(content, str):
                input_item["content"] = [{"type": "input_text", "text": content}]
            elif isinstance(content, list):
                # Handle array content (e.g., with tool calls)
                for item in content:
                    if isinstance(item, str):
                        input_item["content"].append({"type": "input_text", "text": item})
                    elif isinstance(item, dict):
                        input_item["content"].append(item)

            # Handle tool_calls
            if "tool_calls" in msg_data and msg_data["tool_calls"]:
                input_item["tool_calls"] = msg_data["tool_calls"]

            # Handle tool_call_id (for tool response messages)
            if "tool_call_id" in msg_data:
                input_item["tool_call_id"] = msg_data["tool_call_id"]

            # Handle name (for tool/function messages)
            if "name" in msg_data:
                input_item["name"] = msg_data["name"]

            input_items.append(input_item)

        # Build Codex request payload
        payload = {
            "model": model_name,
            "instructions": instructions,  # REQUIRED top-level field
            "input": input_items,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {
                "effort": VARIANT_TO_EFFORT.get(variant, "medium"),
                "summary": "auto",
            },
            "text": {
                "verbosity": "medium",
            },
        }

        # Note: Codex API does NOT support temperature, max_tokens, top_p, etc.
        # Only the fields above are supported.

        return payload

    def _convert_codex_to_openai(
        self,
        codex_chunk: Dict[str, Any],
        model: str,
        accumulator: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert Codex SSE chunk to OpenAI streaming format.

        Codex API event types:
        - response.created: Initial response creation
        - response.in_progress: Initial response metadata
        - response.output_item.added: New output item added
        - response.content_part.added: New content part added
        - response.output_text.delta: {"delta": "H", ...}
        - response.output_text.done: {"text": "Hi!", ...}
        - response.content_part.done: Content part completed
        - response.output_item.done: {"item": {...}}
        - response.completed: {"response": {...}}
        - response.reasoning_summary_part.added: Reasoning metadata
        - response.reasoning_summary_text.delta: Reasoning delta
        - response.reasoning_summary_text.done: Reasoning done
        - response.reasoning_summary_part.done: Reasoning part done
        """
        if accumulator is None:
            accumulator = {
                "has_content": False,
                "has_reasoning": False,
                "tool_idx": 0,
                "is_complete": False,
                "accumulated_content": "",
                "response_id": None,
            }

        chunk_type = codex_chunk.get("type", "")
        delta = {}
        finish_reason = None
        usage_data = None

        # Handle response.created - capture response metadata at start
        if chunk_type == "response.created":
            response_obj = codex_chunk.get("response", {})
            accumulator["response_id"] = response_obj.get("id")
            return None

        # Handle response.in_progress - capture response metadata
        elif chunk_type == "response.in_progress":
            response_obj = codex_chunk.get("response", {})
            accumulator["response_id"] = response_obj.get("id")
            return None

        # Handle reasoning summary events - these are metadata, skip them
        elif chunk_type in (
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
        ):
            return None

        # Handle response.output_item.added - new output item (metadata only)
        elif chunk_type == "response.output_item.added":
            return None

        # Handle response.content_part.added - new content part (metadata only)
        elif chunk_type == "response.content_part.added":
            return None

        # Handle response.content_part.done - content part completed (skip to avoid duplication)
        elif chunk_type == "response.content_part.done":
            return None

        # Handle text delta chunks - this is the ONLY place we yield incremental content
        elif chunk_type == "response.output_text.delta":
            text_delta = codex_chunk.get("delta", "")
            if text_delta:
                delta["content"] = text_delta
                accumulator["accumulated_content"] += text_delta
                accumulator["has_content"] = True

        # Skip response.output_text.done - it repeats the full text, don't yield it
        elif chunk_type == "response.output_text.done":
            return None

        # Skip response.output_item.done for non-completed items or when type is 'message'
        # We only handle it for the final response.completed event
        elif chunk_type == "response.output_item.done":
            return None

        # Handle response.completed chunks (final response with usage)
        # Only yield this for the finish_reason and usage, NOT for content
        # (content was already sent via response.output_text.delta events)
        elif chunk_type == "response.completed":
            response_obj = codex_chunk.get("response", {})
            usage_data = response_obj.get("usage", {})
            accumulator["is_complete"] = True
            finish_reason = "stop"
            # Don't extract content here - it was already sent via delta events

        # Skip if no delta and no usage
        if not delta and not usage_data:
            return None

        # Build OpenAI chunk
        created_at = codex_chunk.get("response", {}).get("created_at", int(time.time()))
        if isinstance(created_at, str):
            created_at = int(created_at)

        response_id = accumulator.get("response_id") or codex_chunk.get("response", {}).get("id")

        openai_chunk = {
            "id": response_id or f"chatcmpl-codex-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": created_at,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }

        # Add usage if available
        if usage_data:
            openai_chunk["usage"] = {
                "prompt_tokens": usage_data.get("input_tokens", 0),
                "completion_tokens": usage_data.get("output_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
            }

        return openai_chunk

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """Handle Codex API completion with request/response transformation."""
        model = kwargs["model"]
        credential_path = kwargs.pop("credential_identifier")
        transaction_context = kwargs.pop("transaction_context", None)

        # Get auth header
        auth_header = await self.get_auth_header(credential_path)

        # Load credentials to get account_id
        creds = await self._load_credentials(credential_path)
        account_id = creds.get("account_id")

        if not account_id:
            raise ValueError(f"account_id not found in credentials for {credential_path}")

        # Parse model name and variant
        model_name, variant = self._parse_model_variant(model)

        # Fetch Codex instructions
        instructions = await self._fetch_codex_instructions(model_name)

        # Transform request payload
        codex_payload = self._transform_request(
            kwargs, model_name, variant, instructions
        )

        # Build Codex-specific headers
        access_token = auth_header["Authorization"].split()[1]
        headers = {
            "Authorization": f"Bearer {access_token}",
            "chatgpt-account-id": account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_rs",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }

        # Create provider logger
        file_logger = ProviderLogger(transaction_context)

        url = f"{CODEX_BASE_URL}{CODEX_API_PATH}"

        # Track accumulator for streaming
        accumulator = {
            "has_content": False,
            "has_reasoning": False,
            "tool_idx": 0,
            "is_complete": False,
            "accumulated_content": "",
            "response_id": None,
        }

        async def stream_handler():
            """Stream response from Codex API and convert to OpenAI format."""
            try:
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=codex_payload,
                    timeout=TimeoutConfig.streaming(),
                ) as response:
                    # Log error before raising
                    if response.status_code >= 400:
                        try:
                            error_body = await response.aread()
                            lib_logger.error(
                                f"Codex API error {response.status_code}: {error_body.decode()}"
                            )
                            file_logger.log_error(
                                f"API error {response.status_code}: {error_body.decode()}"
                            )
                        except Exception:
                            pass

                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        file_logger.log_response_chunk(line)

                        if not line.strip():
                            continue

                        # SSE format: skip event lines, only parse data lines
                        # Codex API returns:
                        #   event: response.created
                        #   data: {...}
                        if line.startswith("event:"):
                            continue

                        # Lines may be prefixed with "data: "
                        if not line.startswith("data:"):
                            lib_logger.debug(f"Skipping non-data SSE line: {line[:100]}")
                            continue

                        # Remove "data: " prefix
                        line = line[5:].strip()

                        if line == "[DONE]":
                            break

                        # Parse JSON data
                        try:
                            codex_chunks = json.loads(line)

                            # Handle single object (not wrapped in array)
                            if isinstance(codex_chunks, dict):
                                codex_chunks = [codex_chunks]

                            for codex_chunk in codex_chunks:
                                lib_logger.info(f"Codex chunk received: {codex_chunk}")
                                try:
                                    openai_chunk = self._convert_codex_to_openai(
                                        codex_chunk, model, accumulator
                                    )
                                    if openai_chunk:
                                        lib_logger.info(f"OpenAI chunk converted: {openai_chunk}")
                                        yield litellm.ModelResponse(**openai_chunk)
                                    else:
                                        lib_logger.warning(f"Failed to convert chunk: {codex_chunk}")
                                except KeyError as e:
                                    lib_logger.error(f"KeyError converting chunk {codex_chunk.get('type')}: {e}")
                                    lib_logger.error(f"Chunk data: {codex_chunk}")
                                    raise

                        except json.JSONDecodeError:
                            lib_logger.warning(
                                f"Could not decode JSON from Codex: {line[:200]}"
                            )

                    # Emit final chunk if stream ended without complete marker
                    if not accumulator.get("is_complete"):
                        final_chunk = {
                            "id": f"chatcmpl-codex-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ],
                        }
                        yield litellm.ModelResponse(**final_chunk)

            except httpx.HTTPStatusError as e:
                error_body = None
                if e.response is not None:
                    try:
                        error_body = e.response.text
                    except Exception:
                        pass

                if error_body:
                    file_logger.log_error(
                        f"HTTPStatusError {e.response.status_code}: {error_body}"
                    )

                if e.response.status_code == 429:
                    retry_after = extract_retry_after_from_body(error_body)
                    retry_info = f" (retry after {retry_after}s)" if retry_after else ""
                    error_msg = f"Codex CLI rate limit exceeded{retry_info}"
                    if error_body:
                        error_msg = f"{error_msg} | {error_body}"

                    lib_logger.debug(f"Codex CLI 429 rate limit: retry_after={retry_after}s")

                    raise RateLimitError(
                        message=error_msg,
                        llm_provider="codex",
                        model=model,
                        response=e.response,
                    )

                raise e

            except Exception as e:
                file_logger.log_error(f"Stream handler exception: {str(e)}")
                raise

        async def logging_stream_wrapper():
            """Wrap stream to log final response."""
            openai_chunks = []
            try:
                async for chunk in stream_handler():
                    openai_chunks.append(chunk)
                    yield chunk
            finally:
                if openai_chunks:
                    # Log final aggregated response
                    final_response = self._stream_to_completion_response(
                        openai_chunks, model
                    )
                    file_logger.log_final_response(final_response)

        # Check if the original request was for streaming
        stream = kwargs.get("stream", False)

        if not stream:
            # For non-streaming requests, consume the stream and return a single response
            openai_chunks = []
            async for chunk in stream_handler():
                openai_chunks.append(chunk)

            if openai_chunks:
                # Aggregate all chunks into a single response
                final_response_dict = self._stream_to_completion_response(
                    openai_chunks, model
                )
                file_logger.log_final_response(final_response_dict)
                return litellm.ModelResponse(**final_response_dict)
            else:
                # Return empty response if no chunks
                return litellm.ModelResponse(
                    id=f"chatcmpl-codex-{int(time.time())}",
                    object="chat.completion",
                    created=int(time.time()),
                    model=model,
                    choices=[{
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop"
                    }],
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                )

        # For streaming requests, return the async generator
        return logging_stream_wrapper()

    def _stream_to_completion_response(
        self, chunks: List, model: str
    ) -> Dict[str, Any]:
        """
        Reassemble streaming chunks into a complete response.
        """
        if not chunks:
            return {}

        final_message = {"role": "assistant"}
        usage_data = None
        chunk_finish_reason = None

        # Process chunks
        for chunk in chunks:
            if not hasattr(chunk, "choices") or not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.get("delta", {})

            # Aggregate content
            if "content" in delta and delta["content"] is not None:
                if "content" not in final_message:
                    final_message["content"] = ""
                final_message["content"] += delta["content"]

            # Aggregate reasoning content
            if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                if "reasoning_content" not in final_message:
                    final_message["reasoning_content"] = ""
                final_message["reasoning_content"] += delta["reasoning_content"]

            # Track finish_reason
            if choice.get("finish_reason"):
                chunk_finish_reason = choice["finish_reason"]

            # Capture usage data
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage

        # Ensure standard fields exist
        for field in ["content", "reasoning_content"]:
            if field not in final_message:
                final_message[field] = None

        # Determine finish_reason
        finish_reason = chunk_finish_reason if chunk_finish_reason else "stop"

        # Build final response
        first_chunk = chunks[0]
        final_response = {
            "id": first_chunk.id,
            "object": "chat.completion",
            "created": first_chunk.created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": final_message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage_data,
        }

        return final_response
