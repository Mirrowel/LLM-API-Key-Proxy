# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Route-support logic for the proxy shell.

The main module stays a thin route surface; the pieces here are the
proxy-level behaviors that surround client calls: stream framing with
in-band error handling, request overrides, and embedding fan-out.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

from rotator_library.client.protocol_selection import format_client_protocol_error

if TYPE_CHECKING:
    from fastapi import Request

    from .detailed_logger import RawIOLogger


def _stream_error_frames(error: BaseException, *, input_protocol: str) -> list[str]:
    """Terminal in-band frames for a failed stream, in the client protocol."""

    if input_protocol == "gemini":
        _, payload = format_client_protocol_error(
            input_protocol="gemini",
            error=error,
            error_type="internal_error",
            status_code=500,
        )
        return [f"data: {json.dumps(payload)}\n\n"]
    _, payload = format_client_protocol_error(
        input_protocol="openai_chat",
        error=error,
        error_type="proxy_internal_error",
        status_code=500,
    )
    return [f"data: {json.dumps(payload)}\n\n", "data: [DONE]\n\n"]


async def streaming_response_wrapper(
    request: "Request",
    request_data: dict[str, Any],
    response_stream: AsyncGenerator,
    logger: Optional["RawIOLogger"] = None,
    *,
    input_protocol: str = "openai_chat",
) -> AsyncGenerator[str, None]:
    """
    Wraps a streaming response to log the full response after completion
    and ensures any errors during the stream are sent to the client as a
    terminal frame in the client's own protocol.
    """

    response_chunks = []
    full_response: dict[str, Any] = {}

    try:
        async for chunk_str in response_stream:
            if await request.is_disconnected():
                logging.warning("Client disconnected, stopping stream.")
                break
            yield chunk_str
            if chunk_str.strip() and chunk_str.startswith("data:"):
                content = chunk_str[len("data:") :].strip()
                if content != "[DONE]":
                    try:
                        chunk_data = json.loads(content)
                        response_chunks.append(chunk_data)
                        if logger:
                            logger.log_stream_chunk(chunk_data)
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        logging.error(f"An error occurred during the response stream: {e}")
        # Yield a terminal error frame in the client's protocol so the
        # stream never ends silently.
        for frame in _stream_error_frames(e, input_protocol=input_protocol):
            yield frame
        if logger:
            logger.log_final_response(
                status_code=500, headers=None, body={"error": str(e)}
            )
        return  # Stop further processing
    finally:
        if response_chunks and input_protocol == "openai_chat":
            full_response = _aggregate_chat_chunks(response_chunks)
        if logger:
            logger.log_final_response(
                status_code=200,
                headers=None,  # Headers are not available at this stage
                body=full_response,
            )


def _aggregate_chat_chunks(response_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble streamed chat chunks into one final response shape."""

    final_message: dict[str, Any] = {"role": "assistant"}
    aggregated_tool_calls: dict[int, dict[str, Any]] = {}
    usage_data = None
    finish_reason = None

    for chunk in response_chunks:
        if "choices" in chunk and chunk["choices"]:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})

            for key, value in delta.items():
                if value is None:
                    continue
                if key == "content":
                    final_message.setdefault("content", "")
                    if value:
                        final_message["content"] += value
                elif key == "tool_calls":
                    for tc_chunk in value:
                        index = tc_chunk["index"]
                        entry = aggregated_tool_calls.setdefault(
                            index,
                            {"type": "function", "function": {"name": "", "arguments": ""}},
                        )
                        if tc_chunk.get("id"):
                            entry["id"] = tc_chunk["id"]
                        if "function" in tc_chunk:
                            if tc_chunk["function"].get("name"):
                                entry["function"]["name"] += tc_chunk["function"]["name"]
                            if tc_chunk["function"].get("arguments"):
                                entry["function"]["arguments"] += tc_chunk["function"]["arguments"]
                elif key == "function_call":
                    call = final_message.setdefault("function_call", {"name": "", "arguments": ""})
                    if value.get("name"):
                        call["name"] += value["name"]
                    if value.get("arguments"):
                        call["arguments"] += value["arguments"]
                else:
                    # Role always replaces; other keys concatenate strings,
                    # extend lists, or replace on shape changes (provider
                    # extension fields can change shape across chunks).
                    if key == "role":
                        final_message[key] = value
                    elif key not in final_message:
                        final_message[key] = value
                    elif isinstance(final_message.get(key), str) and isinstance(value, str):
                        final_message[key] += value
                    elif isinstance(final_message.get(key), list) and isinstance(value, list):
                        final_message[key].extend(value)
                    else:
                        final_message[key] = value

            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

        if chunk.get("usage"):
            usage_data = chunk["usage"]

    if aggregated_tool_calls:
        final_message["tool_calls"] = list(aggregated_tool_calls.values())
        # Agentic systems continue the conversation loop on this reason.
        finish_reason = "tool_calls"

    for field in ("content", "tool_calls", "function_call"):
        final_message.setdefault(field, None)

    first_chunk = response_chunks[0]
    return {
        "id": first_chunk.get("id"),
        "object": "chat.completion",
        "created": first_chunk.get("created"),
        "model": first_chunk.get("model"),
        "choices": [
            {"index": 0, "message": final_message, "finish_reason": finish_reason}
        ],
        "usage": usage_data,
    }


def apply_temperature_override(request_data: dict[str, Any]) -> None:
    """Apply the OVERRIDE_TEMPERATURE_ZERO env knob in place.

    Low temperature makes models deterministic and prone to following
    training data instead of actual schemas, which can cause tool
    hallucination. Modes: "remove" deletes the key, "set" (or truthy
    spellings) rewrites to 1.0, anything else is disabled.
    """

    mode = os.getenv("OVERRIDE_TEMPERATURE_ZERO", "false").lower()
    if (
        mode in ("remove", "set", "true", "1", "yes")
        and request_data.get("temperature") == 0
    ):
        if mode == "remove":
            del request_data["temperature"]
            logging.debug(
                "OVERRIDE_TEMPERATURE_ZERO=remove: Removed temperature=0 from request"
            )
        else:
            request_data["temperature"] = 1.0
            logging.debug(
                "OVERRIDE_TEMPERATURE_ZERO=set: Converting temperature=0 to temperature=1.0"
            )


async def execute_embeddings(
    batcher: Any,
    client: Any,
    payload: dict[str, Any],
    *,
    raw_request: Any = None,
) -> Any:
    """Run an embeddings request, batching when the batcher is available."""

    if batcher is not None:
        import asyncio

        import litellm

        inputs = payload.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]

        tasks = []
        for single_input in inputs:
            individual_request = payload.copy()
            individual_request["input"] = single_input
            tasks.append(batcher.add_request(individual_request))

        results = await asyncio.gather(*tasks)

        all_data = []
        total_prompt_tokens = 0
        total_tokens = 0
        for i, result in enumerate(results):
            result["data"][0]["index"] = i
            all_data.extend(result["data"])
            total_prompt_tokens += result["usage"]["prompt_tokens"]
            total_tokens += result["usage"]["total_tokens"]

        return litellm.EmbeddingResponse(
            **{
                "object": "list",
                "model": results[0]["model"],
                "data": all_data,
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "total_tokens": total_tokens,
                },
            }
        )

    if isinstance(payload.get("input"), str):
        payload["input"] = [payload["input"]]
    return await client.aembedding(request=raw_request, **payload)
