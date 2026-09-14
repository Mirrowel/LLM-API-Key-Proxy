# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Mistral wire adapter (G8): declared param rules plus the mistral-specific bits.

The adapter EXTENDS the generic ``param_rules`` engine (subclass), so one
chain entry — ``adapter_names = ("mistral",)`` — applies the provider's
declared strip/clamp/map/rename tables AND the pieces no flat declaration
can express:

- Request: replayed reasoning fields (``reasoning_content`` /
  ``thinking_blocks``) are stripped from history messages — Mistral 422s
  unknown message fields. The field-cache rule re-injects correlated
  values after the adapter chain; its ``when_missing_only`` guard keeps
  any client-carried value authoritative wherever one survives.
- Request: ``seed`` moves into ``extra_body.random_seed`` — a rename to a
  NESTED target, which the flat param vocabulary cannot declare.
- Response/stream: Mistral's structured thinking content (message content
  as a LIST of ``{"type": "thinking", "thinking": [{"type": "text",
  "text": ...}]}`` plus ``{"type": "text", "text": ...}`` chunks) folds
  into the chat-family spelling: ``reasoning_content`` carries the
  concatenated thinking text, ``content`` the concatenated plain text.
  Stream deltas accumulate the same way (dict chunks and the neutral
  parsed event both handled). Plain-string content passes through
  untouched. Unknown chunk types ride as text-joined content with an
  info log line — never dropped.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

from ..protocols.types import ContentBlock, ReasoningBlock
from .base import AdapterContext
from .param_rules import ParamRulesAdapter

logger = logging.getLogger("rotator_library.adapters")

# Replay spellings Mistral hard-rejects (422) on history messages.
_REPLAY_REASONING_FIELDS = ("reasoning_content", "thinking_blocks")


def _thinking_texts(inner: Any) -> list[str]:
    """Text fragments from one think-chunk's nested ``thinking`` payload."""
    if isinstance(inner, str):
        return [inner]
    if not isinstance(inner, list):
        return []
    texts: list[str] = []
    for part in inner:
        if isinstance(part, str):
            texts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            texts.append(part["text"])
    return texts


def _unknown_chunk_text(chunk: Any) -> str:
    """Textual projection of an unrecognized content chunk (never dropped)."""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict) and isinstance(chunk.get("text"), str):
        return chunk["text"]
    try:
        return json.dumps(chunk, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(chunk)


def _log_unknown_chunk(chunk: Any) -> None:
    kind = chunk.get("type") if isinstance(chunk, dict) else type(chunk).__name__
    logger.info(
        "mistral adapter: unknown content chunk type %r rides as text-joined content",
        kind,
    )


def _split_chunks(chunks: list[Any]) -> tuple[str, str]:
    """Split a Mistral content-chunk list into (thinking text, plain text)."""
    reasoning: list[str] = []
    text: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, str):
            text.append(chunk)
            continue
        if not isinstance(chunk, dict):
            _log_unknown_chunk(chunk)
            text.append(_unknown_chunk_text(chunk))
            continue
        chunk_type = chunk.get("type")
        if chunk_type == "thinking":
            reasoning.extend(_thinking_texts(chunk.get("thinking")))
        elif chunk_type == "text":
            if isinstance(chunk.get("text"), str):
                text.append(chunk["text"])
        else:
            _log_unknown_chunk(chunk)
            text.append(_unknown_chunk_text(chunk))
    return "".join(reasoning), "".join(text)


class MistralAdapter(ParamRulesAdapter):
    """Mistral's OpenAI-compatible surface, made honest.

    Request: the declared param_rules tables (inherited engine) plus
    history-reasoning stripping and the nested ``seed`` rename. Response
    and stream: think-chunk lists fold into ``reasoning_content`` plus
    plain string content.
    """

    name = "mistral"
    supported_stages = ("request", "response", "stream_event")

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        """Rewrite the outgoing request payload.

        Runs in two steps:

        1. The inherited param engine applies the declared tables
           (strip/clamp/map/rename from provider code, model rows, and
           runtime config — resolved for this provider+model).
        2. The Mistral-specific surgery, only when something needs it
           (the early-return keeps healthy payloads untouched and
           copy-free): replayed reasoning fields strip from every
           history message (Mistral 422s unknown message fields — any
           client that echoes our cached reasoning back would hard-fail
           without this), and ``seed`` moves under
           ``extra_body.random_seed`` because the target is nested and
           the flat rename vocabulary cannot express it.
        """

        updated = await super().transform_request(payload, context)
        if not isinstance(updated, dict):
            return updated
        messages = updated.get("messages")
        history_dirty = isinstance(messages, list) and any(
            isinstance(message, dict) and any(field in message for field in _REPLAY_REASONING_FIELDS)
            for message in messages
        )
        if not history_dirty and "seed" not in updated:
            # Nothing to fix — hand the payload through untouched.
            return updated
        updated = deepcopy(updated)
        messages = updated.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                for field in _REPLAY_REASONING_FIELDS:
                    if field in message:
                        del message[field]
        if "seed" in updated:
            # Nested rename (top-level seed -> extra_body.random_seed):
            # not expressible in the flat param_rules vocabulary.
            extra_body = updated.get("extra_body")
            if not isinstance(extra_body, dict):
                extra_body = {}
            extra_body["random_seed"] = updated.pop("seed")
            updated["extra_body"] = extra_body
        return updated

    async def transform_response(self, payload: Any, context: AdapterContext) -> Any:
        """Fold think-chunk content on an assembled provider response.

        Mistral's reasoning models return ``choices[N].message.content``
        as a LIST of typed chunks (thinking + text) instead of a plain
        string. Each choice's message is rewritten so
        ``reasoning_content`` carries the concatenated thinking text and
        ``content`` becomes the plain string — the chat-family spelling
        every downstream consumer understands. Responses whose content
        is already a string pass through untouched.
        """

        if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
            return payload
        if not self._response_needs_fold(payload):
            return payload
        updated = deepcopy(payload)
        for choice in updated.get("choices") or []:
            if isinstance(choice, dict) and isinstance(choice.get("message"), dict):
                self._fold_message(choice["message"])
        return updated

    async def transform_stream_event(self, payload: Any, context: AdapterContext) -> Any:
        """Fold think-chunk content on ONE streamed event.

        Two shapes arrive here depending on the seam: raw provider
        chunks (plain dicts — the stream relay's parsed copy) and
        neutral parsed events (objects with ``delta``/``message`` — the
        executor's neutral-event path). Both get the same fold; the
        neutral variant additionally moves thinking text into the
        delta's reasoning BLOCKS so formatters and the field-cache
        stream sibling see the canonical spelling.
        """

        if isinstance(payload, dict):
            return self._convert_stream_chunk(payload)
        if hasattr(payload, "delta") or hasattr(payload, "message"):
            return self._convert_neutral_event(payload)
        return payload

    # -- response ----------------------------------------------------------

    @staticmethod
    def _response_needs_fold(payload: dict) -> bool:
        for choice in payload.get("choices") or []:
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), list):
                    return True
        return False

    @staticmethod
    def _fold_message(message: dict) -> None:
        reasoning, text = _split_chunks(message["content"])
        if reasoning:
            message["reasoning_content"] = reasoning
        message["content"] = text

    # -- stream ------------------------------------------------------------

    @staticmethod
    def _convert_stream_chunk(payload: dict) -> Any:
        """Fold think-chunk lists in a raw provider stream chunk (dict shape)."""
        if not isinstance(payload.get("choices"), list):
            return payload
        needs_fold = False
        for choice in payload.get("choices") or []:
            if isinstance(choice, dict):
                for target_key in ("delta", "message"):
                    target = choice.get(target_key)
                    if isinstance(target, dict) and isinstance(target.get("content"), list):
                        needs_fold = True
        if not needs_fold:
            return payload
        updated = deepcopy(payload)
        for choice in updated.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            for target_key in ("delta", "message"):
                target = choice.get(target_key)
                if not isinstance(target, dict) or not isinstance(target.get("content"), list):
                    continue
                reasoning, text = _split_chunks(target["content"])
                if reasoning:
                    target["reasoning_content"] = reasoning
                target["content"] = text
        return updated

    @staticmethod
    def _convert_neutral_event(event: Any) -> Any:
        """Fold think-chunk blocks in a neutral parsed stream event.

        The executor's stream seam runs adapters on the NEUTRAL event
        (W7 contract): the openai_chat parse has already turned the raw
        chunk's list content into typed content blocks, with thinking
        chunks preserved as unknown blocks carrying their raw payload.
        Moving their text into the delta's reasoning blocks makes every
        downstream formatter (and the field-cache stream sibling) see the
        chat-family ``reasoning_content`` spelling.
        """
        delta = getattr(event, "delta", None) or getattr(event, "message", None)
        blocks = list(getattr(delta, "content", None) or [])
        if not blocks or all(getattr(block, "type", "") == "text" for block in blocks):
            return event
        converted = deepcopy(event)
        target = getattr(converted, "delta", None) or getattr(converted, "message", None)
        if target is None:
            return converted
        reasoning_texts: list[str] = []
        new_blocks: list[Any] = []
        for block in list(target.content or []):
            block_type = getattr(block, "type", "")
            if block_type == "thinking":
                raw = block.raw if isinstance(getattr(block, "raw", None), dict) else {}
                inner = raw.get("thinking")
                if inner is None and isinstance(getattr(block, "extra", None), dict):
                    inner = block.extra.get("thinking")
                reasoning_texts.extend(_thinking_texts(inner))
            elif block_type == "text" or block_type == "refusal":
                new_blocks.append(block)
            else:
                _log_unknown_chunk(getattr(block, "raw", None))
                new_blocks.append(
                    ContentBlock(type="text", text=_unknown_chunk_text(getattr(block, "raw", None)))
                )
        target.content = new_blocks
        if reasoning_texts:
            target.reasoning = [
                *(target.reasoning or []),
                ReasoningBlock(
                    type="reasoning_content",
                    text="".join(reasoning_texts),
                    extra={"source_field": "reasoning_content"},
                ),
            ]
        return converted
