# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenAI-compatible audio transcription/translation and speech protocol adapter."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar

from .base import ProtocolAdapter
from .operation import OPERATION_AUDIO_TRANSCRIPTION, OPERATION_AUDIO_TRANSLATION, OPERATION_SPEECH, normalize_operation
from .types import ProtocolContext, UnifiedRequest, UnifiedResponse

_AUDIO_OPTION_FIELDS = {"language", "prompt", "response_format", "temperature", "timestamp_granularities"}
_SPEECH_OPTION_FIELDS = {"voice", "response_format", "speed"}
_CORE_FIELDS = {"operation", "model", "file", "input", *_AUDIO_OPTION_FIELDS, *_SPEECH_OPTION_FIELDS}


class OpenAIAudioProtocol(ProtocolAdapter):
    """Adapter for OpenAI audio transcription/translation and speech requests.

    Audio file and generated-audio bytes are intentionally preserved instead of
    interpreted. Transports/providers decide multipart and binary handling; this
    protocol only records enough structure for routing, tracing, and tests.
    """

    name: ClassVar[str] = "openai_audio"
    aliases: ClassVar[tuple[str, ...]] = ("audio", "audio_transcription", "speech", "tts")
    supported_operations: ClassVar[tuple[str, ...]] = (
        OPERATION_AUDIO_TRANSCRIPTION,
        OPERATION_AUDIO_TRANSLATION,
        OPERATION_SPEECH,
    )
    supported_transports: ClassVar[tuple[str, ...]] = ("http",)

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        operation = _audio_operation(request)
        files = []
        if "file" in request:
            files.append({"field": "file", "value": deepcopy(request["file"])})
        options = _SPEECH_OPTION_FIELDS if operation == OPERATION_SPEECH else _AUDIO_OPTION_FIELDS
        return UnifiedRequest(
            operation=operation,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            input=deepcopy(request.get("input") if operation == OPERATION_SPEECH else request.get("prompt")),
            files=files,
            generation_params={k: deepcopy(request[k]) for k in options if k in request},
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": unified_request.model}
        if unified_request.operation == OPERATION_SPEECH:
            payload["input"] = deepcopy(unified_request.input)
        elif unified_request.input is not None:
            payload["prompt"] = deepcopy(unified_request.input)
        for file_entry in unified_request.files:
            if isinstance(file_entry, dict) and file_entry.get("field"):
                payload[str(file_entry["field"])] = deepcopy(file_entry.get("value"))
        payload.update(deepcopy(unified_request.generation_params))
        payload.update(deepcopy(unified_request.extra))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        if isinstance(raw_response, dict):
            response = raw_response
            return UnifiedResponse(
                operation=normalize_operation(response.get("operation")),
                model=response.get("model") or getattr(context, "model", None),
                output=[deepcopy(response["text"])] if "text" in response else [],
                data=deepcopy(response.get("data") or []),
                content_type=response.get("content_type") or "application/json",
                raw=deepcopy(raw_response),
                extra={k: deepcopy(v) for k, v in response.items() if k not in {"operation", "model", "text", "data", "content_type"}},
            )
        content_type = "text/plain" if isinstance(raw_response, str) else "application/octet-stream"
        return UnifiedResponse(content_type=content_type, raw=deepcopy(raw_response), output=[deepcopy(raw_response)] if isinstance(raw_response, str) else [])


def _audio_operation(request: dict[str, Any]) -> str:
    explicit = normalize_operation(request.get("operation"))
    if explicit in {OPERATION_AUDIO_TRANSCRIPTION, OPERATION_AUDIO_TRANSLATION, OPERATION_SPEECH}:
        return explicit
    if "voice" in request or ("input" in request and "file" not in request):
        return OPERATION_SPEECH
    return OPERATION_AUDIO_TRANSCRIPTION
