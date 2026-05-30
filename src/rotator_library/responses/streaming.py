# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""HTTP SSE formatting for the Responses API compatibility layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..protocols import serialize_value


@dataclass(frozen=True)
class ResponsesStreamState:
    """Mutable-by-replacement state accumulated from chat stream chunks."""

    response_id: str
    model: str
    output_text: str = ""
    output_item_id: str = "msg_0"


class ResponsesSSEFormatter:
    """Format Responses API events as HTTP Server-Sent Events."""

    transport = "sse"

    def format_event(self, event_name: str, payload: dict[str, Any]) -> str:
        return f"event: {event_name}\ndata: {json.dumps(serialize_value(payload), ensure_ascii=False)}\n\n"

    def done(self) -> str:
        """Return the final compatibility sentinel used by many SSE clients."""

        return "data: [DONE]\n\n"


class ResponsesWebSocketFormatter:
    """Placeholder transport seam for future WebSocket Responses support."""

    transport = "websocket"
    future_supported = True

    def format_event(self, event_name: str, payload: dict[str, Any]) -> str:
        raise NotImplementedError("Responses WebSocket transport is planned but not implemented yet")


def parse_chat_sse_chunk(chunk: Any) -> dict[str, Any] | None:
    """Decode a chat-completions stream chunk into a dict if possible."""

    if isinstance(chunk, dict):
        return chunk
    if not isinstance(chunk, str):
        return None
    text = chunk.strip()
    if not text:
        return None
    if text.startswith("data:"):
        text = text[len("data:") :].strip()
    if text == "[DONE]":
        return {"type": "done"}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def response_created_payload(response_id: str, model: str) -> dict[str, Any]:
    return {"id": response_id, "object": "response", "status": "in_progress", "model": model, "output": []}


def output_item_added_payload(state: ResponsesStreamState) -> dict[str, Any]:
    return {
        "response_id": state.response_id,
        "output_index": 0,
        "item": {"id": state.output_item_id, "type": "message", "role": "assistant", "content": []},
    }


def output_text_delta_payload(state: ResponsesStreamState, delta: str) -> dict[str, Any]:
    return {
        "response_id": state.response_id,
        "item_id": state.output_item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    }


def output_item_done_payload(state: ResponsesStreamState) -> dict[str, Any]:
    return {
        "response_id": state.response_id,
        "output_index": 0,
        "item": {
            "id": state.output_item_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": state.output_text}],
        },
    }


def response_completed_payload(state: ResponsesStreamState, usage: Any = None) -> dict[str, Any]:
    payload = {
        "id": state.response_id,
        "object": "response",
        "status": "completed",
        "model": state.model,
        "output": [output_item_done_payload(state)["item"]],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def response_failed_payload(response_id: str, model: str, error: Any) -> dict[str, Any]:
    return {"id": response_id, "object": "response", "status": "failed", "model": model, "error": serialize_value(error)}
