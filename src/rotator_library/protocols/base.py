# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Base classes for native protocol adapters.

Protocol adapters are intentionally override-friendly. They provide reusable
defaults for custom providers, but providers can override any method when a
service uses an almost-standard protocol with provider-specific quirks.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar

from .operation import OPERATION_UNKNOWN, normalize_operation
from .types import (
    ProtocolContext,
    ProtocolError,
    UnifiedRequest,
    UnifiedResponse,
    UnifiedStreamEvent,
    Usage,
)


class ProtocolAdapter:
    """Base adapter for converting between raw protocol payloads and unified types.

    Subclasses should override only the methods they need. The default behavior
    is deliberately conservative: preserve raw payloads and avoid lossy
    assumptions. Later phases will layer transform logging, field-cache rules,
    and provider override hooks around this interface.
    """

    name: ClassVar[str] = "base"
    aliases: ClassVar[tuple[str, ...]] = ()
    supported_transports: ClassVar[tuple[str, ...]] = ("http", "sse")
    future_transports: ClassVar[tuple[str, ...]] = ()
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_UNKNOWN,)

    def supports_transport(self, transport_name: str) -> bool:
        """Return whether this protocol can format the requested transport."""

        return transport_name in self.supported_transports

    def is_future_transport(self, transport_name: str) -> bool:
        """Return whether this protocol has an intentional future transport seam."""

        return transport_name in self.future_transports

    def supports_operation(self, operation_name: str) -> bool:
        """Return whether this adapter natively models an operation.

        Operation names are string based so custom protocols can add their own
        values. The default base adapter only claims ``unknown`` and keeps raw
        payloads intact; concrete adapters should list every operation they can
        parse/build without relying on LiteLLM.
        """

        return normalize_operation(operation_name) in self.supported_operations

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        """Parse a raw client/provider request into a unified request."""

        request = dict(raw_request or {})
        return UnifiedRequest(
            operation=normalize_operation(request.get("operation")),
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            stream=bool(request.get("stream", False)),
            input=deepcopy(request.get("input")),
            modalities=list(request.get("modalities") or []),
            files=list(request.get("files") or []),
            raw=deepcopy(raw_request),
            extra={
                k: deepcopy(v)
                for k, v in request.items()
                if k not in {"operation", "model", "stream", "input", "modalities", "files"}
            },
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        """Build a provider request from a unified request.

        The base implementation returns the original raw dict when present. This
        keeps fallback providers safe and gives custom protocol subclasses a
        predictable starting point.
        """

        if isinstance(unified_request.raw, dict):
            return deepcopy(unified_request.raw)
        if not isinstance(unified_request.raw, type(None)):
            raise ProtocolError(
                "cannot build dict request from non-dict raw payload",
                protocol=self.name,
                pass_name="build_request",
                payload=unified_request.raw,
            )
        payload = {"model": unified_request.model, "stream": unified_request.stream}
        if unified_request.operation != OPERATION_UNKNOWN:
            payload["operation"] = unified_request.operation
        if unified_request.input is not None:
            payload["input"] = deepcopy(unified_request.input)
        if unified_request.modalities:
            payload["modalities"] = deepcopy(unified_request.modalities)
        if unified_request.files:
            payload["files"] = deepcopy(unified_request.files)
        payload.update(deepcopy(unified_request.extra))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        """Parse a raw response into a unified response."""

        response = raw_response if isinstance(raw_response, dict) else {}
        return UnifiedResponse(
            operation=normalize_operation(response.get("operation") if isinstance(response, dict) else None),
            id=response.get("id") if isinstance(response, dict) else None,
            model=response.get("model") if isinstance(response, dict) else getattr(context, "model", None),
            data=deepcopy(response.get("data") or []) if isinstance(response, dict) else [],
            content_type=response.get("content_type") if isinstance(response, dict) else None,
            raw=deepcopy(raw_response),
            extra=deepcopy(response) if isinstance(response, dict) else {},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:
        """Format a unified response for a client protocol."""

        if isinstance(unified_response.raw, dict):
            return deepcopy(unified_response.raw)
        payload = deepcopy(unified_response.extra)
        if unified_response.operation != OPERATION_UNKNOWN:
            payload.setdefault("operation", unified_response.operation)
        if unified_response.id is not None:
            payload.setdefault("id", unified_response.id)
        if unified_response.model is not None:
            payload.setdefault("model", unified_response.model)
        if unified_response.data:
            payload.setdefault("data", deepcopy(unified_response.data))
        if unified_response.content_type is not None:
            payload.setdefault("content_type", unified_response.content_type)
        return payload

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        """Parse one raw stream event.

        Subclasses should preserve the original event in ``raw`` because Phase 2
        transform logging needs both provider-native and unified states.
        """

        event_type = "done" if raw_event == "[DONE]" else "chunk"
        return UnifiedStreamEvent(type=event_type, raw=deepcopy(raw_event))

    def format_stream_event(self, unified_event: UnifiedStreamEvent, context: ProtocolContext | None = None) -> Any:
        """Format one unified stream event for the target transport."""

        return deepcopy(unified_event.raw) if unified_event.raw is not None else unified_event.to_dict()

    def extract_usage(self, raw_or_unified: Any, context: ProtocolContext | None = None) -> Usage | None:
        """Extract normalized usage when the protocol can identify it."""

        if isinstance(raw_or_unified, UnifiedResponse):
            return raw_or_unified.usage
        if isinstance(raw_or_unified, UnifiedStreamEvent):
            return raw_or_unified.usage
        if isinstance(raw_or_unified, dict) and isinstance(raw_or_unified.get("usage"), dict):
            usage = raw_or_unified["usage"]
            return Usage(
                input_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
                total_tokens=int(usage.get("total_tokens") or 0),
                raw=deepcopy(usage),
            )
        return None
