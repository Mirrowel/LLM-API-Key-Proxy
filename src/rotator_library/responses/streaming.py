# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""HTTP SSE formatting for the Responses API compatibility layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..protocols import serialize_value


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
