# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Retry-safety policy for stream errors after output has started."""

import json
from typing import Any, Optional


_REASONING_FIELDS = (
    "reasoning",
    "reasoning_content",
    "thinking",
    "thinking_content",
)


def can_retry_stream_after_error(
    last_streamed_chunk: Optional[str],
    allow_reasoning_only_retry: bool,
) -> bool:
    """
    Return whether a stream can be retried after an upstream error.

    Retrying is always safe before any chunk has been emitted. After that, it is
    only allowed when explicitly enabled and the latest chunk is clearly
    reasoning/thinking-only. Ambiguous chunks fail closed.
    """
    if last_streamed_chunk is None:
        return True
    if not allow_reasoning_only_retry:
        return False

    payload = last_streamed_chunk.strip()
    if not payload.startswith("data:"):
        return False
    payload = payload[5:].strip()
    if not payload or payload == "[DONE]":
        return False
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
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
            if (
                _has_text(source.get("content"))
                or _has_text(source.get("text"))
                or source.get("tool_calls")
                or source.get("function_call")
            ):
                return False
            if any(_has_text(source.get(key)) for key in _REASONING_FIELDS):
                has_reasoning = True

    return has_reasoning


def _has_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return True
            if isinstance(item, dict) and _has_text(item.get("text")):
                return True
    return False
