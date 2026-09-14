# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Groq wire adapter (chat-completions family).

Parameter hygiene lives in the provider's ``model_rules`` cascade and runs
through the always-on generic ``param_rules`` stage; this adapter owns only
the pieces a flat declaration cannot express:

- Request: ``reasoning_format: parsed`` is forced whenever tools or JSON
  output are present (raw thinking alongside tools is a documented 400),
  and ``include_reasoning`` — mutually exclusive with ``reasoning_format``
  — is dropped in the same rewrite.
- Response/stream: Groq's ``reasoning`` field is renamed to the
  chat-family ``reasoning_content`` spelling, and the terminal
  ``x_groq.usage`` payload is lifted into the standard ``usage`` slot.

Everything else — the unsupported-knob strip list, the ``temperature``
clamp, the ``n`` pin — is declared on the provider and enforced before
this adapter ever sees the payload.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .base import AdapterContext, PayloadAdapter


class GroqAdapter(PayloadAdapter):
    """Groq's OpenAI-compatible surface, made honest.

    Request: conditional ``reasoning_format`` selection plus the
    ``include_reasoning`` drop. Response/stream: reasoning-field rename
    and ``x_groq.usage`` lifting.
    """

    name = "groq"
    aliases = ("groq_params", "groq_reasoning")
    supported_stages = ("request", "response", "stream_event")

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        if not isinstance(payload, dict):
            return payload
        needs_parsed = payload.get("tools") or (
            isinstance(payload.get("response_format"), dict)
        )
        if not needs_parsed:
            return payload
        # Explicit raw alongside tools is a documented 400 — parsed wins
        # regardless of what the client asked for; and include_reasoning
        # is mutually exclusive with reasoning_format, so it cannot ride
        # along.
        updated = deepcopy(payload)
        updated["reasoning_format"] = "parsed"
        updated.pop("include_reasoning", None)
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
        if updated.get("usage") is None:
            x_groq = updated.get("x_groq")
            if isinstance(x_groq, dict) and isinstance(x_groq.get("usage"), dict):
                updated["usage"] = x_groq["usage"]
                changed = True
        if not changed:
            return payload
        return updated
