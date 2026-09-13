# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Groq wire adapters (chat-completions family)."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from .base import AdapterContext, PayloadAdapter

logger = logging.getLogger("rotator_library.adapters")


class GroqAdapter(PayloadAdapter):
    """Groq's OpenAI-compatible surface, made honest.

    Request: strips knobs Groq hard-rejects (frequency/presence penalty,
    logit_bias, logprobs families), clamps ``temperature`` 0 to the
    float32-safe epsilon Groq requires, forces ``reasoning_format:
    parsed`` when tools or JSON output are present (raw thinking plus
    tools is a documented 400), and keeps ``n`` at 1.

    Response/stream: renames Groq's ``reasoning`` fields to the
    chat-family ``reasoning_content`` spelling the protocol layer speaks.
    """

    name = "groq"
    aliases = ("groq_params", "groq_reasoning")
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
            if key in updated:
                updated.pop(key)
        temperature = updated.get("temperature")
        if isinstance(temperature, (int, float)) and float(temperature) == 0.0:
            updated["temperature"] = 1e-8
        if updated.get("n") not in (None, 1):
            updated["n"] = 1
        needs_parsed = updated.get("tools") or (
            isinstance(updated.get("response_format"), dict)
        )
        if needs_parsed and "reasoning_format" not in updated:
            updated["reasoning_format"] = "parsed"
        return updated

    async def transform_response(self, payload: Any, context: AdapterContext) -> Any:
        return self._rename(payload)

    async def transform_stream_event(self, payload: Any, context: AdapterContext) -> Any:
        return self._rename(payload)

    @staticmethod
    def _rename(payload: Any) -> Any:
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
