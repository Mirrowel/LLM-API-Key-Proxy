# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Chutes and NanoGPT wire adapters (chat-completions family)."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from .base import AdapterContext, PayloadAdapter

logger = logging.getLogger("rotator_library.adapters")


class ChutesAdapter(PayloadAdapter):
    """Chutes' OpenAI-compatible gateway, made honest.

    Requests outside the documented sampling whitelist hard-fail with a
    422, so unsupported OpenAI knobs are stripped and
    ``max_completion_tokens`` maps to the ``max_tokens`` spelling Chutes
    expects. Responses and streams may carry vLLM's ``reasoning``
    spelling; it is renamed to the chat-family ``reasoning_content``.
    """

    name = "chutes"
    supported_stages = ("request", "response", "stream_event")

    _UNSUPPORTED = (
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
    )

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        if not isinstance(payload, dict):
            return payload
        updated = deepcopy(payload)
        for key in self._UNSUPPORTED:
            updated.pop(key, None)
        if "max_completion_tokens" in updated and "max_tokens" not in updated:
            updated["max_tokens"] = updated.pop("max_completion_tokens")
        return updated

    async def transform_response(self, payload: Any, context: AdapterContext) -> Any:
        return _rename_reasoning(payload)

    async def transform_stream_event(self, payload: Any, context: AdapterContext) -> Any:
        return _rename_reasoning(payload)


class NanoGPTAdapter(PayloadAdapter):
    """NanoGPT's OpenAI-compatible aggregator, made honest.

    The canonical length parameter maps to NanoGPT's ``max_tokens``
    spelling, and the ``reasoning`` field spelling (with the legacy
    ``reasoning_content`` already handled by the protocol layer) is
    normalized on responses and stream deltas.
    """

    name = "nanogpt"
    supported_stages = ("request", "response", "stream_event")

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        if not isinstance(payload, dict):
            return payload
        updated = deepcopy(payload)
        if "max_completion_tokens" in updated and "max_tokens" not in updated:
            updated["max_tokens"] = updated.pop("max_completion_tokens")
        return updated

    async def transform_response(self, payload: Any, context: AdapterContext) -> Any:
        return _rename_reasoning(payload)

    async def transform_stream_event(self, payload: Any, context: AdapterContext) -> Any:
        return _rename_reasoning(payload)


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
