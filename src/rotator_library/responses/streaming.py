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


@dataclass(frozen=True)
class ResponsesStreamEvent:
    """Transport-neutral Responses stream event.

    Service code yields these events. Formatters decide how to serialize them for
    SSE, WebSocket, or future transports without duplicating protocol logic.
    """

    event_name: str
    payload: dict[str, Any]
    terminal: bool = False

    @property
    def heartbeat(self) -> bool:
        """Return whether this event is transport metadata, not model output."""

        return self.event_name == "heartbeat"


class ResponsesSSEFormatter:
    """Format Responses API events as HTTP Server-Sent Events."""

    transport = "sse"

    def format_event(self, event_name: str, payload: dict[str, Any]) -> str:
        return f"event: {event_name}\ndata: {json.dumps(serialize_value(payload), ensure_ascii=False)}\n\n"

    def format_stream_event(self, event: ResponsesStreamEvent) -> str:
        """Format one transport-neutral event for HTTP SSE."""

        if event.heartbeat:
            return self.format_heartbeat(str(event.payload.get("comment") or "heartbeat"))
        if event.terminal:
            return self.done()
        return self.format_event(event.event_name, event.payload)

    def format_heartbeat(self, comment: str = "heartbeat") -> str:
        """Return a non-visible SSE comment heartbeat frame."""

        safe_comment = comment.replace("\r", " ").replace("\n", " ")
        return f": {safe_comment}\n\n"

    def done(self) -> str:
        """Return the final compatibility sentinel used by many SSE clients."""

        return "data: [DONE]\n\n"


def parse_chat_sse_chunk(chunk: Any) -> dict[str, Any] | None:
    """Decode a chat-completions stream chunk into a dict if possible."""

    if isinstance(chunk, dict):
        return chunk
    if not isinstance(chunk, str):
        return None
    text = chunk.strip()
    if not text:
        return None
    event_name = None
    data_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if data_lines:
        text = "\n".join(data_lines).strip()
    if text == "[DONE]":
        return {"type": "done"}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and event_name and payload.get("type") is None:
        payload["type"] = event_name
    return payload if isinstance(payload, dict) else None


# Monotonic event ordering counter (spec: every stream event carries
# sequence_number). Module-level for the service/bridge path; the protocol
# converter keeps its own per-stream counter in StreamFormatState.sequence.
_sequence_counter = {"value": 0}


def _next_sequence() -> int:
    value = _sequence_counter["value"]
    _sequence_counter["value"] += 1
    return value


def reset_stream_sequence() -> None:
    _sequence_counter["value"] = 0


def next_sequence_value() -> int:
    """Public accessor for service-level event builders."""

    return _next_sequence()


def response_created_payload(response_id: str, model: str) -> dict[str, Any]:
    return {"type": "response.created", "sequence_number": _next_sequence(), "response": {"id": response_id, "object": "response", "status": "in_progress", "model": model, "output": []}}


def response_in_progress_payload(response_id: str, model: str) -> dict[str, Any]:
    return {"type": "response.in_progress", "sequence_number": _next_sequence(), "response": {"id": response_id, "object": "response", "status": "in_progress", "model": model, "output": []}}


def output_item_added_payload(state: ResponsesStreamState) -> dict[str, Any]:
    return {
        "type": "response.output_item.added",
        "sequence_number": _next_sequence(),
        "output_index": 0,
        "item": {"id": state.output_item_id, "type": "message", "role": "assistant", "content": [], "status": "in_progress"},
    }


def content_part_added_payload(state: ResponsesStreamState) -> dict[str, Any]:
    return {
        "type": "response.content_part.added",
        "sequence_number": _next_sequence(),
        "item_id": state.output_item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    }


def output_text_delta_payload(state: ResponsesStreamState, delta: str) -> dict[str, Any]:
    return {
        "type": "response.output_text.delta",
        "sequence_number": _next_sequence(),
        "item_id": state.output_item_id,
        "output_index": 0,
        "content_index": 0,
        "delta": delta,
    }


def output_text_done_payload(state: ResponsesStreamState) -> dict[str, Any]:
    return {
        "type": "response.output_text.done",
        "sequence_number": _next_sequence(),
        "item_id": state.output_item_id,
        "output_index": 0,
        "content_index": 0,
        "text": state.output_text,
    }


def content_part_done_payload(state: ResponsesStreamState) -> dict[str, Any]:
    return {
        "type": "response.content_part.done",
        "sequence_number": _next_sequence(),
        "item_id": state.output_item_id,
        "output_index": 0,
        "content_index": 0,
        "part": {"type": "output_text", "text": state.output_text, "annotations": []},
    }


def output_item_done_payload(state: ResponsesStreamState) -> dict[str, Any]:
    return {
        "type": "response.output_item.done",
        "sequence_number": _next_sequence(),
        "output_index": 0,
        "item": {
            "id": state.output_item_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": state.output_text, "annotations": []}],
            "status": "completed",
        },
    }


def response_completed_payload(state: ResponsesStreamState, usage: Any = None) -> dict[str, Any]:
    payload = {
        "type": "response.completed",
        "sequence_number": _next_sequence(),
        "response": {
            "id": state.response_id,
            "object": "response",
            "status": "completed",
            "model": state.model,
            "output": [output_item_done_payload(state)["item"]],
        },
    }
    if usage is not None:
        payload["response"]["usage"] = usage
    return payload


def response_failed_payload(response_id: str, model: str, error: Any) -> dict[str, Any]:
    return {
        "type": "response.failed",
        "sequence_number": _next_sequence(),
        "response": {"id": response_id, "object": "response", "status": "failed", "model": model, "output": [], "error": serialize_value(error)},
    }
