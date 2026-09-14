# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""NanoGPT wire adapter (chat-completions family).

Response/stream normalization for the aggregator's vendor spellings. The
canonical length parameter maps to NanoGPT's ``max_tokens`` spelling through
a declared ``model_rules`` row on the provider (not adapter code).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .base import AdapterContext, PayloadAdapter


class NanoGPTAdapter(PayloadAdapter):
    """NanoGPT's OpenAI-compatible aggregator, made honest.

    The ``reasoning`` field spelling (with the legacy ``reasoning_content``
    already handled by the protocol layer) is normalized on responses and
    stream deltas, and a top-level ``reasoning_tokens`` count is folded into
    ``completion_tokens_details`` so usage accounting reads the standard
    OpenAI shape. Request-stage behavior is declared (``model_rules``).
    """

    name = "nanogpt"
    supported_stages = ("response", "stream_event")

    async def transform_response(self, payload: Any, context: AdapterContext) -> Any:
        return _normalize(payload)

    async def transform_stream_event(self, payload: Any, context: AdapterContext) -> Any:
        return _normalize(payload)


def _normalize(payload: Any) -> Any:
    payload = _rename_reasoning(payload)
    if not isinstance(payload, dict):
        return payload
    usage = payload.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("reasoning_tokens"), (int, float)):
        details = usage.get("completion_tokens_details")
        if not isinstance(details, dict) or "reasoning_tokens" not in details:
            payload = deepcopy(payload)
            usage = payload["usage"]
            details = dict(details or {})
            details["reasoning_tokens"] = usage.pop("reasoning_tokens")
            usage["completion_tokens_details"] = details
    return payload


def _rename_reasoning(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    updated = deepcopy(payload)
    changed = False
    for choice in updated.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and "reasoning" in message and "reasoning_content" not in message:
            message["reasoning_content"] = message.pop("reasoning")
            changed = True
        delta = choice.get("delta")
        if isinstance(delta, dict) and "reasoning" in delta and "reasoning_content" not in delta:
            delta["reasoning_content"] = delta.pop("reasoning")
            changed = True
    if not changed:
        return payload
    return updated
