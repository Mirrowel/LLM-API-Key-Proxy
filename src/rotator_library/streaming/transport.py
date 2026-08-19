# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Transport formatters for normalized stream events."""

from __future__ import annotations

import json

from .events import StreamEvent


class SSEStreamFormatter:
    """Format stream events as HTTP Server-Sent Events."""

    transport = "sse"

    def format_event(self, event: StreamEvent) -> str:
        return f"event: {event.event_type}\ndata: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"

    def format_error(self, event: StreamEvent) -> str:
        return self.format_event(event)

    def format_done(self) -> str:
        return "data: [DONE]\n\n"

    def format_heartbeat(self, comment: str = "heartbeat") -> str:
        """Return an SSE comment heartbeat frame.

        Comment frames keep HTTP connections active without becoming model
        output, so routing/session code must continue treating them as
        non-visible stream data.
        """

        safe_comment = comment.replace("\r", " ").replace("\n", " ")
        return f": {safe_comment}\n\n"

    def is_terminal_event(self, event: StreamEvent) -> bool:
        return event.event_type in {"completed", "cancelled", "error"}


class WebSocketStreamFormatter:
    """Format stream events as WebSocket JSON messages without exposing a route."""

    transport = "websocket"
    future_supported = True

    def format_event(self, event: StreamEvent) -> dict:
        return {"type": event.event_type, "payload": event.to_dict()}

    def format_error(self, event: StreamEvent) -> dict:
        return self.format_event(event)

    def format_done(self) -> dict:
        return {"type": "completed", "payload": {"done": True}}

    def format_heartbeat(self) -> dict:
        """Return the future WebSocket heartbeat message shape."""

        return {"type": "heartbeat", "payload": {"visible_output": False}}

    def is_terminal_event(self, event: StreamEvent) -> bool:
        return event.event_type in {"completed", "cancelled", "error"}


class JSONLineStreamFormatter:
    """Format stream events as newline-delimited JSON for provider tests."""

    transport = "jsonl"

    def format_event(self, event: StreamEvent) -> str:
        return json.dumps(event.to_dict(), ensure_ascii=False) + "\n"

    def format_error(self, event: StreamEvent) -> str:
        return self.format_event(event)

    def format_done(self) -> str:
        return json.dumps({"event_type": "completed", "done": True}) + "\n"

    def format_heartbeat(self) -> str:
        """Return a JSONL heartbeat record for test transports."""

        return json.dumps({"event_type": "heartbeat", "visible_output": False}) + "\n"

    def is_terminal_event(self, event: StreamEvent) -> bool:
        return event.event_type in {"completed", "cancelled", "error"}
