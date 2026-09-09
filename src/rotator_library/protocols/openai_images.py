# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenAI-compatible image generation/edit/variation protocol adapter."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar

from .base import ProtocolAdapter
from .operation import OPERATION_IMAGE_EDIT, OPERATION_IMAGE_GENERATION, OPERATION_IMAGE_VARIATION, normalize_operation
from .types import ProtocolContext, UnifiedRequest, UnifiedResponse

_OPTION_FIELDS = {"n", "size", "quality", "style", "response_format", "user", "background", "moderation"}
_CORE_FIELDS = {"operation", "model", "prompt", "image", "mask", *_OPTION_FIELDS}


class OpenAIImagesProtocol(ProtocolAdapter):
    """Adapter for image generation, edit, and variation request shapes.

    File references are preserved as metadata in ``UnifiedRequest.files``. The
    adapter never reads file contents; multipart assembly belongs to transport or
    provider execution code so protocol parsing remains side-effect free.
    """

    name: ClassVar[str] = "openai_images"
    aliases: ClassVar[tuple[str, ...]] = ("images", "image_generation", "openai_image")
    supported_operations: ClassVar[tuple[str, ...]] = (
        OPERATION_IMAGE_GENERATION,
        OPERATION_IMAGE_EDIT,
        OPERATION_IMAGE_VARIATION,
    )
    supported_transports: ClassVar[tuple[str, ...]] = ("http",)

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        operation = _image_operation(request)
        files = []
        if "image" in request:
            files.append({"field": "image", "value": deepcopy(request["image"])})
        if "mask" in request:
            files.append({"field": "mask", "value": deepcopy(request["mask"])})
        return UnifiedRequest(
            operation=operation,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            input=deepcopy(request.get("prompt")),
            files=files,
            generation_params={k: deepcopy(request[k]) for k in _OPTION_FIELDS if k in request},
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if unified_request.model:
            payload["model"] = unified_request.model
        if unified_request.input is not None:
            payload["prompt"] = deepcopy(unified_request.input)
        for file_entry in unified_request.files:
            if isinstance(file_entry, dict) and file_entry.get("field"):
                payload[str(file_entry["field"])] = deepcopy(file_entry.get("value"))
        payload.update(deepcopy(unified_request.generation_params))
        payload.update(deepcopy(unified_request.extra))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = raw_response if isinstance(raw_response, dict) else {}
        return UnifiedResponse(
            operation=_context_operation(context, OPERATION_IMAGE_GENERATION),
            model=response.get("model") or getattr(context, "model", None),
            data=deepcopy(response.get("data") or []),
            raw=deepcopy(raw_response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"model", "data"}},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:
        payload = {"data": deepcopy(unified_response.data)}
        if unified_response.model:
            payload["model"] = unified_response.model
        payload.update(deepcopy(unified_response.extra))
        return payload


def _image_operation(request: dict[str, Any]) -> str:
    explicit = normalize_operation(request.get("operation"))
    if explicit in {OPERATION_IMAGE_GENERATION, OPERATION_IMAGE_EDIT, OPERATION_IMAGE_VARIATION}:
        return explicit
    if "image" in request and "prompt" in request:
        return OPERATION_IMAGE_EDIT
    if "image" in request:
        return OPERATION_IMAGE_VARIATION
    return OPERATION_IMAGE_GENERATION


def _context_operation(context: ProtocolContext | None, default: str) -> str:
    if context and isinstance(context.provider_options, dict):
        operation = normalize_operation(context.provider_options.get("operation"))
        if operation in {OPERATION_IMAGE_GENERATION, OPERATION_IMAGE_EDIT, OPERATION_IMAGE_VARIATION}:
            return operation
    if context and isinstance(context.metadata, dict):
        operation = normalize_operation(context.metadata.get("operation"))
        if operation in {OPERATION_IMAGE_GENERATION, OPERATION_IMAGE_EDIT, OPERATION_IMAGE_VARIATION}:
            return operation
    return default
