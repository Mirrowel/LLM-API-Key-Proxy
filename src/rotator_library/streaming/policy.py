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
    if is_stream_heartbeat_or_comment(last_streamed_chunk):
        return True
    metadata = _sse_json(last_streamed_chunk, malformed_is_visible=False)
    if isinstance(metadata, dict) and (metadata.get("event_type") or metadata.get("type")) == "cost":
        return True
    if metadata is None:
        return False
    if is_visible_stream_output(last_streamed_chunk):
        return False

    has_reasoning = False
    choices = metadata.get("choices")
    if not isinstance(choices, list):
        return True
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
    return allow_reasoning_only_retry if has_reasoning else True


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
    if isinstance(event_type, str) and event_type.startswith("content_block_"):
        return _anthropic_visible(data)
    if isinstance(data.get("candidates"), list):
        return _gemini_visible(data)
    if protocol == "responses":
        return _responses_visible(data)
    return _openai_chat_visible(data)


def is_stream_heartbeat_or_comment(chunk: Optional[str]) -> bool:
    """Return true for SSE comment-only frames that must not affect retry state."""

    if chunk is None:
        return False
    payload = chunk.strip()
    return bool(payload) and all(line.startswith(":") for line in payload.splitlines() if line.strip())


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
        if event_type == "cost":
            return {"event_type": "cost", "value": parsed}
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
    if event_type == "response.output_item.added":
        item = data.get("item")
        if isinstance(item, dict) and item.get("type") in {"function_call", "custom_tool_call"}:
            return bool(item.get("call_id") or item.get("id") or item.get("name"))
    if isinstance(event_type, str) and ("function_call" in event_type or "tool_call" in event_type):
        return _has_visible_text(data.get("delta")) or _has_visible_text(data.get("arguments")) or _has_visible_text(data.get("item"))
    if isinstance(event_type, str) and "reasoning" in event_type and event_type.endswith(".delta"):
        return _has_visible_text(data.get("delta"))
    if event_type == "response.failed":
        return False
    return False


def _anthropic_visible(data: dict[str, Any]) -> bool:
    """Return whether an Anthropic content event exposes model output."""

    event_type = data.get("event_type") or data.get("type")
    if event_type == "content_block_start":
        block = data.get("content_block")
        return isinstance(block, dict) and block.get("type") == "tool_use"
    if event_type != "content_block_delta":
        return False
    delta = data.get("delta")
    if not isinstance(delta, dict):
        return False
    return any(
        _has_visible_text(delta.get(key))
        for key in ("text", "thinking", "partial_json")
    )


def _gemini_visible(data: dict[str, Any]) -> bool:
    """Return whether a Gemini candidate frame exposes content or a tool call."""

    for candidate in data.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            if _has_visible_text(part.get("text")) or part.get("functionCall") or part.get("function_call"):
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
