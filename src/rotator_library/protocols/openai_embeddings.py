# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenAI-compatible embeddings protocol adapter."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar

from .base import ProtocolAdapter
from .operation import OPERATION_EMBEDDINGS
from .types import ProtocolContext, UnifiedRequest, UnifiedResponse, Usage

_REQUEST_CORE_FIELDS = {"model", "input", "encoding_format", "dimensions", "user", "operation"}
_REQUEST_OPTION_FIELDS = {"encoding_format", "dimensions", "user"}


class OpenAIEmbeddingsProtocol(ProtocolAdapter):
    """Adapter for `/v1/embeddings` style request and response payloads.

    The adapter intentionally treats embedding vectors as opaque data entries.
    That keeps it usable for OpenAI-compatible providers with additional index,
    metadata, or sparse-vector fields without narrowing the schema too early.
    """

    name: ClassVar[str] = "openai_embeddings"
    aliases: ClassVar[tuple[str, ...]] = ("embeddings", "openai_embedding")
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_EMBEDDINGS,)
    supported_transports: ClassVar[tuple[str, ...]] = ("http",)

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        return UnifiedRequest(
            operation=OPERATION_EMBEDDINGS,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            input=deepcopy(request.get("input")),
            generation_params={k: deepcopy(request[k]) for k in _REQUEST_OPTION_FIELDS if k in request},
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        payload = {"model": unified_request.model, "input": deepcopy(unified_request.input)}
        payload.update(deepcopy(unified_request.generation_params))
        payload.update(deepcopy(unified_request.extra))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = raw_response if isinstance(raw_response, dict) else {}
        return UnifiedResponse(
            operation=OPERATION_EMBEDDINGS,
            model=response.get("model") or getattr(context, "model", None),
            data=deepcopy(response.get("data") or []),
            usage=self.extract_usage(response, context),
            raw=deepcopy(raw_response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"model", "data", "usage"}},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:
        payload = {"object": "list", "data": deepcopy(unified_response.data)}
        if unified_response.model:
            payload["model"] = unified_response.model
        if unified_response.usage:
            payload["usage"] = unified_response.usage.raw or unified_response.usage.to_dict()
        payload.update(deepcopy(unified_response.extra))
        return payload

    def extract_usage(self, raw_or_unified: Any, context: ProtocolContext | None = None) -> Usage | None:
        if isinstance(raw_or_unified, UnifiedResponse):
            return raw_or_unified.usage
        usage = raw_or_unified.get("usage") if isinstance(raw_or_unified, dict) else None
        if not isinstance(usage, dict):
            return None
        return Usage(
            input_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("total_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            raw=deepcopy(usage),
        )
