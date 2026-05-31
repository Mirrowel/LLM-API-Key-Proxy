# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Ollama-native chat, generate, and embeddings protocol adapter."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, ClassVar

from .base import ProtocolAdapter
from .operation import OPERATION_EMBEDDINGS, OPERATION_OLLAMA_CHAT, OPERATION_OLLAMA_GENERATE, normalize_operation
from .types import ContentBlock, ProtocolContext, UnifiedMessage, UnifiedRequest, UnifiedResponse, UnifiedStreamEvent, Usage

_OPTION_FIELDS = {"options", "format", "keep_alive", "template", "context", "raw", "suffix"}
_CORE_FIELDS = {"operation", "model", "messages", "prompt", "input", "stream", "system", *_OPTION_FIELDS}


class OllamaProtocol(ProtocolAdapter):
    """Adapter for Ollama `/api/chat`, `/api/generate`, and embeddings shapes."""

    name: ClassVar[str] = "ollama"
    aliases: ClassVar[tuple[str, ...]] = ("ollama_native",)
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_OLLAMA_CHAT, OPERATION_OLLAMA_GENERATE, OPERATION_EMBEDDINGS)
    supported_transports: ClassVar[tuple[str, ...]] = ("http", "jsonl")

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        operation = _ollama_operation(request, context)
        input_field = "input"
        if operation == OPERATION_EMBEDDINGS and "input" not in request and "prompt" in request:
            # Older Ollama embeddings endpoints use `prompt`; newer endpoints use
            # `input`. Preserve the original spelling so round-trips are lossless.
            input_field = "prompt"
        return UnifiedRequest(
            operation=operation,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            messages=[_message_from_ollama(message) for message in request.get("messages") or []],
            system=[ContentBlock(type="text", text=str(request["system"]))] if "system" in request else [],
            stream=bool(request.get("stream", False)),
            input=deepcopy(request.get(input_field) if operation == OPERATION_EMBEDDINGS else request.get("prompt")),
            generation_params={k: deepcopy(request[k]) for k in _OPTION_FIELDS if k in request},
            metadata={**({"embedding_input_field": input_field} if operation == OPERATION_EMBEDDINGS else {}), "has_stream": "stream" in request},
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": unified_request.model}
        if unified_request.stream or unified_request.metadata.get("has_stream"):
            payload["stream"] = unified_request.stream
        if unified_request.operation == OPERATION_OLLAMA_CHAT:
            payload["messages"] = [_message_to_ollama(message) for message in unified_request.messages]
        elif unified_request.operation == OPERATION_EMBEDDINGS:
            input_field = str(unified_request.metadata.get("embedding_input_field") or "input")
            payload[input_field] = deepcopy(unified_request.input)
        else:
            payload["prompt"] = deepcopy(unified_request.input)
        if unified_request.system:
            payload["system"] = "".join(block.text or "" for block in unified_request.system if block.type == "text")
        payload.update(deepcopy(unified_request.generation_params))
        payload.update(deepcopy(unified_request.extra))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = raw_response if isinstance(raw_response, dict) else {}
        output = []
        messages: list[UnifiedMessage] = []
        if isinstance(response.get("message"), dict):
            message = _message_from_ollama(response["message"])
            messages.append(message)
            output.append(message.to_dict())
        elif "response" in response:
            output.append(response.get("response"))
        return UnifiedResponse(
            operation=_ollama_operation(response, context),
            model=response.get("model") or getattr(context, "model", None),
            messages=messages,
            output=output,
            data=deepcopy(response.get("embeddings") or response.get("embedding") or []),
            usage=_ollama_usage(response),
            raw=deepcopy(raw_response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"model", "message", "response", "embeddings", "embedding", "prompt_eval_count", "eval_count", "total_duration", "load_duration", "prompt_eval_duration", "eval_duration"}},
        )

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        data = _json_event(raw_event)
        if not isinstance(data, dict):
            return UnifiedStreamEvent(type="metadata", operation=OPERATION_OLLAMA_GENERATE, raw=deepcopy(raw_event), extra={"unparsed": True})
        event_type = "done" if data.get("done") else "message_delta"
        delta = None
        if isinstance(data.get("message"), dict):
            delta = _message_from_ollama(data["message"])
        elif data.get("response"):
            delta = UnifiedMessage(role="assistant", content=[ContentBlock(type="text", text=str(data["response"]))])
        return UnifiedStreamEvent(type=event_type, operation=_ollama_operation(data, context), delta=delta, usage=_ollama_usage(data), raw=deepcopy(raw_event), extra=deepcopy(data))


def _ollama_operation(request: dict[str, Any], context: ProtocolContext | None = None) -> str:
    explicit = normalize_operation(request.get("operation"))
    if explicit in {OPERATION_OLLAMA_CHAT, OPERATION_OLLAMA_GENERATE, OPERATION_EMBEDDINGS}:
        return explicit
    if context and isinstance(context.provider_options, dict):
        operation = normalize_operation(context.provider_options.get("operation"))
        if operation in {OPERATION_OLLAMA_CHAT, OPERATION_OLLAMA_GENERATE, OPERATION_EMBEDDINGS}:
            return operation
    if context and isinstance(context.metadata, dict):
        operation = normalize_operation(context.metadata.get("operation"))
        if operation in {OPERATION_OLLAMA_CHAT, OPERATION_OLLAMA_GENERATE, OPERATION_EMBEDDINGS}:
            return operation
    if "messages" in request or "message" in request:
        return OPERATION_OLLAMA_CHAT
    if "prompt" in request and request.get("endpoint") != "embeddings":
        return OPERATION_OLLAMA_GENERATE
    if "embeddings" in request or "embedding" in request or "input" in request or request.get("endpoint") == "embeddings":
        return OPERATION_EMBEDDINGS
    return OPERATION_OLLAMA_GENERATE


def _message_from_ollama(message: dict[str, Any]) -> UnifiedMessage:
    return UnifiedMessage(role=str(message.get("role") or "assistant"), content=[ContentBlock(type="text", text=str(message.get("content") or ""))], raw=deepcopy(message), extra={k: deepcopy(v) for k, v in message.items() if k not in {"role", "content"}})


def _message_to_ollama(message: UnifiedMessage) -> dict[str, Any]:
    payload = {"role": message.role, "content": "".join(block.text or "" for block in message.content if block.type == "text")}
    payload.update(deepcopy(message.extra))
    return payload


def _json_event(raw_event: Any) -> Any:
    if isinstance(raw_event, dict):
        return raw_event
    if not isinstance(raw_event, str):
        return None
    text = raw_event.strip()
    if text.startswith("data:"):
        text = text[5:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _ollama_usage(response: dict[str, Any]) -> Usage | None:
    prompt_tokens = int(response.get("prompt_eval_count") or 0)
    output_tokens = int(response.get("eval_count") or 0)
    if not prompt_tokens and not output_tokens:
        return None
    return Usage(input_tokens=prompt_tokens, output_tokens=output_tokens, raw={k: deepcopy(v) for k, v in response.items() if k.endswith("count") or k.endswith("duration")})
