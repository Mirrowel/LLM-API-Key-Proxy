# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Streaming wrapper for converting OpenAI streaming format to Anthropic streaming format.

This module provides a framework-agnostic streaming wrapper that converts
OpenAI SSE (Server-Sent Events) format to Anthropic's streaming format.
"""

import json
import logging
import uuid
from typing import AsyncGenerator, Callable, Optional, Awaitable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..transaction_logger import TransactionLogger

logger = logging.getLogger("rotator_library.anthropic_compat")


async def anthropic_streaming_wrapper(
    openai_stream: AsyncGenerator[str, None],
    original_model: str,
    request_id: Optional[str] = None,
    is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    transaction_logger: Optional["TransactionLogger"] = None,
) -> AsyncGenerator[str, None]:
    """
    Convert OpenAI streaming format to Anthropic streaming format.

    This is a framework-agnostic wrapper that can be used with any async web framework.
    Instead of taking a FastAPI Request object, it accepts an optional callback function
    to check for client disconnection.

    Anthropic SSE events:
    - message_start: Initial message metadata
    - content_block_start: Start of a content block
    - content_block_delta: Content chunk
    - content_block_stop: End of a content block
    - message_delta: Final message metadata (stop_reason, usage)
    - message_stop: End of message

    Args:
        openai_stream: AsyncGenerator yielding OpenAI SSE format strings
        original_model: The model name to include in responses
        request_id: Optional request ID (auto-generated if not provided)
        is_disconnected: Optional async callback that returns True if client disconnected
        transaction_logger: Optional TransactionLogger for logging the final Anthropic response

    Yields:
        SSE format strings in Anthropic's streaming format
    """
    if request_id is None:
        request_id = f"msg_{uuid.uuid4().hex[:24]}"

    message_started = False
    content_block_started = False
    thinking_block_started = False
    current_block_index = 0
    tool_calls_by_index = {}  # Track tool calls by their index
    tool_block_indices = {}  # Track which block index each tool call uses
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0  # Track cached tokens for proper Anthropic format
    accumulated_text = ""  # Track accumulated text for logging
    accumulated_thinking = ""  # Track accumulated thinking for logging
    stop_reason_final = "end_turn"  # Track final stop reason for logging

    try:
        async for chunk_str in openai_stream:
            # Check for client disconnection if callback provided
            if is_disconnected is not None and await is_disconnected():
                break

            if not chunk_str.strip() or not chunk_str.startswith("data:"):
                continue

            data_content = chunk_str[len("data:") :].strip()
            if data_content == "[DONE]":
                # CRITICAL: Send message_start if we haven't yet (e.g., empty response)
                # Claude Code and other clients require message_start before message_stop
                if not message_started:
                    # Build usage with cached tokens properly handled
                    usage_dict = {
                        "input_tokens": input_tokens - cached_tokens,
                        "output_tokens": 0,
                    }
                    if cached_tokens > 0:
                        usage_dict["cache_read_input_tokens"] = cached_tokens
                        usage_dict["cache_creation_input_tokens"] = 0

                    message_start = {
                        "type": "message_start",
                        "message": {
                            "id": request_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": original_model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": usage_dict,
                        },
                    }
                    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
                    message_started = True

                # Close any open thinking block
                if thinking_block_started:
                    yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {current_block_index}}}\n\n'
                    current_block_index += 1
                    thinking_block_started = False

                # Close any open text block
                if content_block_started:
                    yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {current_block_index}}}\n\n'
                    current_block_index += 1
                    content_block_started = False

                # Close all open tool_use blocks
                for tc_index in sorted(tool_block_indices.keys()):
                    block_idx = tool_block_indices[tc_index]
                    yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {block_idx}}}\n\n'

                # Determine stop_reason based on whether we had tool calls
                stop_reason = "tool_use" if tool_calls_by_index else "end_turn"
                stop_reason_final = stop_reason

                # Build final usage dict with cached tokens
                final_usage = {"output_tokens": output_tokens}
                if cached_tokens > 0:
                    final_usage["cache_read_input_tokens"] = cached_tokens
                    final_usage["cache_creation_input_tokens"] = 0

                # Send message_delta with final info
                yield f'event: message_delta\ndata: {{"type": "message_delta", "delta": {{"stop_reason": "{stop_reason}", "stop_sequence": null}}, "usage": {json.dumps(final_usage)}}}\n\n'

                # Send message_stop
                yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

                # Log final Anthropic response if logger provided
                if transaction_logger:
                    # Build content blocks for logging
                    content_blocks = []
                    if accumulated_thinking:
                        content_blocks.append(
                            {
                                "type": "thinking",
                                "thinking": accumulated_thinking,
                            }
                        )
                    if accumulated_text:
                        content_blocks.append(
                            {
                                "type": "text",
                                "text": accumulated_text,
                            }
                        )
                    # Add tool use blocks
                    for tc_index in sorted(tool_calls_by_index.keys()):
                        tc = tool_calls_by_index[tc_index]
                        # Parse arguments JSON string to dict
                        try:
                            input_data = json.loads(tc.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            input_data = {}
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": tc.get("name", ""),
                                "input": input_data,
                            }
                        )

                    # Build usage for logging
                    log_usage = {
                        "input_tokens": input_tokens - cached_tokens,
                        "output_tokens": output_tokens,
                    }
                    if cached_tokens > 0:
                        log_usage["cache_read_input_tokens"] = cached_tokens
                        log_usage["cache_creation_input_tokens"] = 0

                    anthropic_response = {
                        "id": request_id,
                        "type": "message",
                        "role": "assistant",
                        "content": content_blocks,
                        "model": original_model,
                        "stop_reason": stop_reason_final,
                        "stop_sequence": None,
                        "usage": log_usage,
                    }
                    transaction_logger.log_response(
                        anthropic_response,
                        filename="anthropic_response.json",
                    )

                break

            try:
                chunk = json.loads(data_content)
            except json.JSONDecodeError:
                continue

            # Extract usage if present
            # Note: Google's promptTokenCount INCLUDES cached tokens, but Anthropic's
            # input_tokens EXCLUDES cached tokens. We extract cached tokens and subtract.
            if "usage" in chunk and chunk["usage"]:
                usage = chunk["usage"]
                input_tokens = usage.get("prompt_tokens", input_tokens)
                output_tokens = usage.get("completion_tokens", output_tokens)
                # Extract cached tokens from prompt_tokens_details
                if usage.get("prompt_tokens_details"):
                    cached_tokens = usage["prompt_tokens_details"].get(
                        "cached_tokens", cached_tokens
                    )

            # Send message_start on first chunk
            if not message_started:
                # Build usage with cached tokens properly handled for Anthropic format
                usage_dict = {
                    "input_tokens": input_tokens - cached_tokens,
                    "output_tokens": 0,
                }
                if cached_tokens > 0:
                    usage_dict["cache_read_input_tokens"] = cached_tokens
                    usage_dict["cache_creation_input_tokens"] = 0

                message_start = {
                    "type": "message_start",
                    "message": {
                        "id": request_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": original_model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": usage_dict,
                    },
                }
                yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
                message_started = True

            choices = chunk.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta", {})

            # Handle reasoning/thinking content (from OpenAI-style reasoning_content)
            reasoning_content = delta.get("reasoning_content")
            if reasoning_content:
                if not thinking_block_started:
                    # Start a thinking content block
                    block_start = {
                        "type": "content_block_start",
                        "index": current_block_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                    thinking_block_started = True

                # Send thinking delta
                block_delta = {
                    "type": "content_block_delta",
                    "index": current_block_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning_content},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"
                # Accumulate thinking for logging
                accumulated_thinking += reasoning_content

            # Handle text content
            content = delta.get("content")
            if content:
                # If we were in a thinking block, close it first
                if thinking_block_started and not content_block_started:
                    yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {current_block_index}}}\n\n'
                    current_block_index += 1
                    thinking_block_started = False

                if not content_block_started:
                    # Start a text content block
                    block_start = {
                        "type": "content_block_start",
                        "index": current_block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                    content_block_started = True

                # Send content delta
                block_delta = {
                    "type": "content_block_delta",
                    "index": current_block_index,
                    "delta": {"type": "text_delta", "text": content},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"
                # Accumulate text for logging
                accumulated_text += content

            # Handle tool calls
            # Use `or []` to handle providers that send "tool_calls": null
            tool_calls = delta.get("tool_calls") or []
            for tc in tool_calls:
                tc_index = tc.get("index", 0)

                if tc_index not in tool_calls_by_index:
                    # Close previous thinking block if open
                    if thinking_block_started:
                        yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {current_block_index}}}\n\n'
                        current_block_index += 1
                        thinking_block_started = False

                    # Close previous text block if open
                    if content_block_started:
                        yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {current_block_index}}}\n\n'
                        current_block_index += 1
                        content_block_started = False

                    # Start new tool use block
                    tool_calls_by_index[tc_index] = {
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": "",
                    }
                    # Track which block index this tool call uses
                    tool_block_indices[tc_index] = current_block_index

                    block_start = {
                        "type": "content_block_start",
                        "index": current_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_calls_by_index[tc_index]["id"],
                            "name": tool_calls_by_index[tc_index]["name"],
                            "input": {},
                        },
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                    # Increment for the next block
                    current_block_index += 1

                # Accumulate arguments
                func = tc.get("function", {})
                if func.get("name"):
                    tool_calls_by_index[tc_index]["name"] = func["name"]
                if func.get("arguments"):
                    tool_calls_by_index[tc_index]["arguments"] += func["arguments"]

                    # Send partial JSON delta using the correct block index for this tool
                    block_delta = {
                        "type": "content_block_delta",
                        "index": tool_block_indices[tc_index],
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": func["arguments"],
                        },
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"

            # Note: We intentionally ignore finish_reason here.
            # Block closing is handled when we receive [DONE] to avoid
            # premature closes with providers that send finish_reason on each chunk.

    except Exception as e:
        logger.error(f"Error in Anthropic streaming wrapper: {e}")

        # If we haven't sent message_start yet, send it now so the client can display the error
        # Claude Code and other clients may ignore events that come before message_start
        if not message_started:
            # Build usage with cached tokens properly handled
            usage_dict = {
                "input_tokens": input_tokens - cached_tokens,
                "output_tokens": 0,
            }
            if cached_tokens > 0:
                usage_dict["cache_read_input_tokens"] = cached_tokens
                usage_dict["cache_creation_input_tokens"] = 0

            message_start = {
                "type": "message_start",
                "message": {
                    "id": request_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": original_model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": usage_dict,
                },
            }
            yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

        # Send the error as a text content block so it's visible to the user
        error_message = f"Error: {str(e)}"
        error_block_start = {
            "type": "content_block_start",
            "index": current_block_index,
            "content_block": {"type": "text", "text": ""},
        }
        yield f"event: content_block_start\ndata: {json.dumps(error_block_start)}\n\n"

        error_block_delta = {
            "type": "content_block_delta",
            "index": current_block_index,
            "delta": {"type": "text_delta", "text": error_message},
        }
        yield f"event: content_block_delta\ndata: {json.dumps(error_block_delta)}\n\n"

        yield f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {current_block_index}}}\n\n'

        # Build final usage with cached tokens
        final_usage = {"output_tokens": 0}
        if cached_tokens > 0:
            final_usage["cache_read_input_tokens"] = cached_tokens
            final_usage["cache_creation_input_tokens"] = 0

        # Send message_delta and message_stop to properly close the stream
        yield f'event: message_delta\ndata: {{"type": "message_delta", "delta": {{"stop_reason": "end_turn", "stop_sequence": null}}, "usage": {json.dumps(final_usage)}}}\n\n'
        yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

        # Also send the formal error event for clients that handle it
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
