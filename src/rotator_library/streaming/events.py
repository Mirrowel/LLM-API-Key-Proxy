# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Transport-neutral stream event model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from typing import Any, Optional

from ..protocols import serialize_value


@dataclass(frozen=True)
class StreamEvent:
    """A normalized stream event before transport-specific formatting.

    The event model is intentionally broad so providers can attach native data
    without inventing a new stream lifecycle object. Raw data is for trace/debug
    use only and should still pass through transaction-log redaction when logged.
    """

    event_type: str
    data: Any = field(default_factory=dict)
    protocol: Optional[str] = None
    transport: str = "sse"
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    visible_output: bool = False
    timestamp_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for traces and transports."""

        return {
            "event_type": self.event_type,
            "protocol": self.protocol,
            "transport": self.transport,
            "data": serialize_value(self.data),
            "raw": serialize_value(self.raw),
            "metadata": serialize_value(self.metadata),
            "visible_output": self.visible_output,
            "timestamp_utc": self.timestamp_utc,
        }


def stream_event_from_sse_chunk(chunk: Any, *, protocol: str = "openai_chat") -> StreamEvent:
    """Parse an SSE `data:` chunk into a normalized stream event.

    Malformed or non-JSON chunks are treated as metadata and fail closed for
    visibility. This avoids accidental fallback after ambiguous client output.
    """

    if isinstance(chunk, dict):
        data = chunk
        raw = chunk
    elif isinstance(chunk, str):
        raw = chunk
        text = chunk.strip()
        if text.startswith("data:"):
            text = text[len("data:") :].strip()
        if text == "[DONE]":
            return StreamEvent("completed", protocol=protocol, raw=chunk, data={"done": True})
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return StreamEvent("metadata", protocol=protocol, raw=chunk, data={"malformed": True})
    else:
        return StreamEvent("metadata", protocol=protocol, raw=chunk, data={"unsupported_chunk": True})

    if isinstance(data, dict) and data.get("error"):
        return StreamEvent("error", protocol=protocol, raw=raw, data=data, visible_output=False)
    visible = _openai_chat_visible(data) if protocol == "openai_chat" else False
    event_type = "delta" if visible else "parsed_chunk"
    return StreamEvent(event_type, protocol=protocol, raw=raw, data=data, visible_output=visible)


def _openai_chat_visible(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    choices = data.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        for source in (delta, message):
            if not isinstance(source, dict):
                continue
            if _has_visible_text(source.get("content")) or _has_visible_text(source.get("text")):
                return True
            if source.get("tool_calls") or source.get("function_call"):
                return True
    return False


def _has_visible_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return True
            if isinstance(item, dict) and _has_visible_text(item.get("text")):
                return True
    return False
