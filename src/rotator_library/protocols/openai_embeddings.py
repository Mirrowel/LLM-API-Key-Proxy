# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenAI-compatible embeddings protocol adapter."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, Mapping, Optional

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

    G10 Phase B disclosure: unknown option keys are passed through verbatim
    (round-trip fidelity) but recorded as ``unsupported_optional_control``
    request warnings; :meth:`format_response` carries them under the private
    ``_proxy_warnings`` transport key for the finalizer to drain and strip.
    NOTE: the live embeddings route currently bypasses this adapter (G9
    first-class embeddings is pending), so the disclosure fires whenever the
    adapter is engaged (native/embedding transports) rather than on the
    legacy passthrough.
    """

    name: ClassVar[str] = "openai_embeddings"
    aliases: ClassVar[tuple[str, ...]] = ("embeddings", "openai_embedding")
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_EMBEDDINGS,)
    supported_transports: ClassVar[tuple[str, ...]] = ("http",)

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        warnings: list = []
        unknown_options = sorted(key for key in request if key not in _REQUEST_CORE_FIELDS)
        if unknown_options:
            # Unknown option keys are passed through verbatim (same-protocol
            # round-trip fidelity), but provider support is not guaranteed —
            # disclosed rather than silent.
            from .canonical import record_conversion_warning

            record_conversion_warning(
                warnings,
                code="unsupported_optional_control",
                message=(
                    "embeddings option key(s) outside the canonical surface pass through "
                    f"verbatim (provider support not guaranteed): {', '.join(unknown_options)}"
                ),
                field="options",
                source_protocol=self.name,
                target_protocol=self.name,
            )
        return UnifiedRequest(
            operation=OPERATION_EMBEDDINGS,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            input=deepcopy(request.get("input")),
            generation_params={k: deepcopy(request[k]) for k in _REQUEST_OPTION_FIELDS if k in request},
            raw=deepcopy(raw_request),
            warnings=warnings,
            extra={k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None, capabilities: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
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

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None, capabilities: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        payload = {"object": "list", "data": deepcopy(unified_response.data)}
        if unified_response.model:
            payload["model"] = unified_response.model
        if unified_response.usage:
            payload["usage"] = unified_response.usage.raw or unified_response.usage.to_dict()
        payload.update(deepcopy(unified_response.extra))
        # Private transport key (G10 Phase B): request option-drop warnings
        # ride the response to the finalizer, which sinks them into the
        # change log and strips the key before the client sees it.
        if unified_response.warnings:
            payload["_proxy_warnings"] = list(unified_response.warnings)
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
