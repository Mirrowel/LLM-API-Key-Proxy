# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Stream retry and visible-output policy."""

from __future__ import annotations

import json
from typing import Any, Optional

_REASONING_FIELDS = (
    "reasoning",
    "reasoning_content",
    "thinking",
    "thinking_content",
)


def can_retry_stream_after_error(last_streamed_chunk: Optional[str], allow_reasoning_only_retry: bool) -> bool:
    """Return whether an upstream stream can be retried after an error."""

    if last_streamed_chunk is None:
        return True
    if not allow_reasoning_only_retry:
        return False
    data = _sse_json(last_streamed_chunk, malformed_is_visible=False)
    if data is None:
        return False

    has_reasoning = False
    choices = data.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            return False
        for source in (choice, choice.get("delta"), choice.get("message")):
            if not isinstance(source, dict):
                continue
            if _has_visible_text(source.get("content")) or _has_visible_text(source.get("text")):
                return False
            if source.get("tool_calls") or source.get("function_call"):
                return False
            if any(_has_visible_text(source.get(key)) for key in _REASONING_FIELDS):
                has_reasoning = True
    return has_reasoning


def is_visible_stream_output(chunk: Optional[str], *, protocol: str = "openai_chat") -> bool:
    """Return whether a formatted stream chunk should block fallback.

    Malformed or ambiguous chunks fail closed by counting as visible output. This
    preserves the existing safety rule that route fallback must not happen after
    a client may have received model output.
    """

    if chunk is None:
        return False
    data = _sse_json(chunk, malformed_is_visible=True)
    if data is _MALFORMED_VISIBLE:
        return True
    if data is None:
        return False
    if data.get("error"):
        return False
    event_type = data.get("event_type") or data.get("type")
    if isinstance(event_type, str) and event_type.startswith("response."):
        return _responses_visible(data)
    if protocol == "responses":
        return _responses_visible(data)
    return _openai_chat_visible(data)


_MALFORMED_VISIBLE = object()


def _sse_json(chunk: str, *, malformed_is_visible: bool) -> dict[str, Any] | object | None:
    payload = chunk.strip()
    if not payload:
        return None
    if all(line.startswith(":") for line in payload.splitlines() if line.strip()):
        return None
    event_type = None
    data_lines: list[str] = []
    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            continue
        if stripped.startswith("event:"):
            event_type = stripped[6:].strip()
            continue
        if stripped.startswith("data:"):
            data_lines.append(stripped[5:].strip())
            continue
        return _MALFORMED_VISIBLE if malformed_is_visible else None
    if not data_lines:
        if event_type in {"error", "response.failed"}:
            return {"event_type": event_type}
        return None
    payload = "\n".join(data_lines).strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return _MALFORMED_VISIBLE if malformed_is_visible else None
    if not isinstance(parsed, dict):
        return _MALFORMED_VISIBLE if malformed_is_visible else None
    if event_type and "event_type" not in parsed:
        parsed["event_type"] = event_type
    return parsed


def _openai_chat_visible(data: dict[str, Any]) -> bool:
    choices = data.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for source in (choice.get("delta"), choice.get("message")):
            if not isinstance(source, dict):
                continue
            if _has_visible_text(source.get("content")) or _has_visible_text(source.get("text")):
                return True
            if source.get("tool_calls") or source.get("function_call"):
                return True
    return False


def _responses_visible(data: dict[str, Any]) -> bool:
    event_type = data.get("event_type") or data.get("type")
    if event_type == "response.output_text.delta":
        return bool(str(data.get("delta", "")).strip())
    if event_type == "response.failed":
        return False
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
