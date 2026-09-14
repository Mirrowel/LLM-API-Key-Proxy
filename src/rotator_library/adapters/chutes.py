# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Chutes wire adapter (chat-completions family).

The request/response vocabulary of Chutes' data centers (vLLM/SGLang behind
the gateway). The shared ``max_completion_tokens`` → ``max_tokens`` rename is
NOT here: it is a declared ``model_rules`` row on the provider (the same
declaration NanoGPT carries).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .base import AdapterContext, PayloadAdapter


class ChutesAdapter(PayloadAdapter):
    """Chutes' OpenAI-compatible gateway, made honest about its data centers.

    Requests outside the documented sampling whitelist hard-fail with a
    422, so unsupported OpenAI knobs are stripped and ``n`` is pinned to a
    single choice. Responses and streams may carry vLLM's ``reasoning``
    spelling instead of SGLang's ``reasoning_content``; both are normalized
    to the chat-family field here (the request-stage hygiene that a flat
    ``model_rules`` row can express — the shared length rename — lives on
    the provider's capability cascade instead).
    """

    name = "chutes"
    supported_stages = ("request", "response", "stream_event")

    _UNSUPPORTED = (
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "best_of",
    )

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        if not isinstance(payload, dict):
            return payload
        updated = deepcopy(payload)
        for key in self._UNSUPPORTED:
            updated.pop(key, None)
        if updated.get("n") not in (None, 1):
            updated["n"] = 1
        return updated

    async def transform_response(self, payload: Any, context: AdapterContext) -> Any:
        return _rename_reasoning(payload)

    async def transform_stream_event(self, payload: Any, context: AdapterContext) -> Any:
        return _rename_reasoning(payload)


def _rename_reasoning(payload: Any) -> Any:
    """Rename the vLLM ``reasoning`` spelling to ``reasoning_content``.

    Both response messages and stream deltas are covered; a payload that
    already carries ``reasoning_content`` is left untouched (never
    overwrite the canonical field), and a payload without ``reasoning`` is
    returned unchanged (no copy).
    """

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
