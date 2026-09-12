# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Route-support logic for the proxy shell.

The main module stays a thin route surface; the pieces here are the
proxy-level behaviors that surround client calls: stream framing with
in-band error handling, request overrides, and embedding fan-out.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

import httpx

from rotator_library.client.protocol_selection import (
    canonical_protocol_name,
    format_client_protocol_error,
)
from rotator_library.core.errors import (
    StructuredAPIResponseError,
    protocol_error_payload,
)
from rotator_library.error_handler import classify_error
from rotator_library.transaction_logger import TransactionLogger

if TYPE_CHECKING:
    from fastapi import Request

    from .detailed_logger import RawIOLogger


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# Grounded route status ladder (docs/experimental/error-reference.md 5.9): the
# internal classifier type decides the client-visible status. Server-family
# statuses that carry a stronger wire signal (503 unavailable / 504 deadline /
# 529 overloaded) are preserved so the anthropic envelope can render
# timeout_error/overloaded_error.
_GROUNDED_ROUTE_STATUS = {
    "invalid_request": 400,
    "context_window_exceeded": 400,
    "authentication": 401,
    "forbidden": 403,
    "not_found": 404,
    "conflict": 409,
    "request_too_large": 413,
    "rate_limit": 429,
    "quota_exceeded": 429,
    "server_error": 502,
    "api_connection": 502,
    "proxy_timeout": 504,
    "proxy_all_credentials_exhausted": 503,
}
_SERVER_FAMILY_TYPES = {
    "server_error",
    "api_connection",
    "proxy_all_credentials_exhausted",
}
_PRESERVED_SERVER_STATUSES = {503, 504, 529}


def _is_timeout_exception(error: BaseException) -> bool:
    """True for litellm/httpx/asyncio timeout classes (any MRO name match)."""

    if isinstance(error, (asyncio.TimeoutError, httpx.TimeoutException)):
        return True
    return any("timeout" in cls.__name__.lower() for cls in type(error).__mro__)


def classify_route_error(error: BaseException) -> tuple[str, int]:
    """Classify a route-side failure into the grounded (type, status) pair."""

    if _is_timeout_exception(error):
        return "proxy_timeout", 504

    classified = classify_error(error)
    error_type = classified.error_type
    raw_status = classified.status_code

    if error_type == "unknown":
        # The bare-ValueError family is the request-path validation contract
        # (bad addressing, bad model reference, malformed bodies) → 400.
        if isinstance(error, ValueError):
            error_type = "invalid_request"
        elif isinstance(error, asyncio.CancelledError):
            error_type = "cancelled"
        else:
            error_type = "server_error"

    if error_type in _SERVER_FAMILY_TYPES and raw_status in _PRESERVED_SERVER_STATUSES:
        return error_type, raw_status

    return error_type, _GROUNDED_ROUTE_STATUS.get(error_type, 502)


def route_error_response(
    error: BaseException | str, *, protocol: str
) -> tuple[int, dict[str, Any]]:
    """Render one shell-side failure in the route's own client protocol.

    The shell stays thin: classification lives in the library
    (``classify_error``) and rendering lives in the library
    (``protocol_error_payload``). This helper only bridges the two and picks
    the grounded status ladder.
    """

    canonical = canonical_protocol_name(protocol)
    if isinstance(error, BaseException):
        error_type, status_code = classify_route_error(error)
    else:
        error_type, status_code = "invalid_request", 400
    if canonical == "ollama":
        # Ollama's native error convention is a plain string body under the
        # top-level ``error`` key — never a nested protocol envelope.
        message = str(error) if str(error) else "Request failed"
        return status_code, {"error": message}
    return protocol_error_payload(
        error, canonical, error_type=error_type, status_code=status_code
    )


def protocol_for_route_path(path: str | None) -> str:
    """Map an HTTP route path onto its client protocol dialect."""

    path = path or ""
    if path.endswith((":generateContent", ":streamGenerateContent", ":countTokens")):
        return "gemini"
    if path.startswith("/api/"):
        return "ollama"
    if path.startswith("/v1/messages"):
        return "anthropic_messages"
    if path.startswith("/v1/responses"):
        return "responses"
    return "openai_chat"


def stable_error_response(
    message: str,
    *,
    protocol: str,
    error_type: str = "server_error",
    status_code: int = 502,
) -> tuple[int, dict[str, Any]]:
    """Render a fixed, non-leaking message for management-route failures."""

    return route_error_response(
        StructuredAPIResponseError(
            message, error_type=error_type, status_code=status_code
        ),
        protocol=protocol,
    )


def _stream_error_frames(error: BaseException, *, input_protocol: str) -> list[str]:
    """Terminal in-band frames for a failed stream, in the client protocol."""

    if input_protocol == "gemini":
        _, payload = format_client_protocol_error(
            input_protocol="gemini",
            error=error,
            error_type="server_error",
            status_code=500,
        )
        return [f"data: {json.dumps(payload)}\n\n"]
    if input_protocol == "anthropic_messages":
        _, payload = format_client_protocol_error(
            input_protocol="anthropic_messages",
            error=error,
            error_type="api_error",
            status_code=500,
        )
        return [f"event: error\ndata: {json.dumps(payload)}\n\n"]
    if input_protocol == "ollama":
        # NDJSON terminal error frame: one object line, no SSE framing.
        return [json.dumps({"error": str(error) or "Provider stream failed"}) + "\n"]
    _, payload = format_client_protocol_error(
        input_protocol="openai_chat",
        error=error,
        error_type="server_error",
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
    """Assemble streamed chat chunks into one final response shape.

    Delegates to the canonical assembler so every aggregation surface shares
    one ruling: per-choice sibling retention, provider finish wins, and the
    infer-only-when-absent fallback.
    """

    return TransactionLogger.assemble_streaming_response(response_chunks)


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
