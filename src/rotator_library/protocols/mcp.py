# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""MCP JSON-RPC carrier protocol adapter.

This is not a full MCP proxy implementation. It gives the native protocol layer a
lossless request/response carrier for future MCP gateway work, keeping method,
params, ids, results, and errors intact for transform logging and routing.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar

from .base import ProtocolAdapter
from .operation import OPERATION_MCP
from .types import ProtocolContext, UnifiedRequest, UnifiedResponse, UnifiedStreamEvent

_REQUEST_CORE_FIELDS = {"jsonrpc", "id", "method", "params"}
_RESPONSE_CORE_FIELDS = {"jsonrpc", "id", "result", "error"}


class MCPProtocol(ProtocolAdapter):
    """Adapter for MCP-style JSON-RPC request and response envelopes."""

    name: ClassVar[str] = "mcp"
    aliases: ClassVar[tuple[str, ...]] = ("model_context_protocol", "jsonrpc_mcp")
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_MCP,)
    supported_transports: ClassVar[tuple[str, ...]] = ("http", "sse")
    future_transports: ClassVar[tuple[str, ...]] = ("websocket",)

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        metadata = {
            "jsonrpc": request.get("jsonrpc", "2.0"),
            "id": deepcopy(request.get("id")),
            "method": request.get("method"),
        }
        return UnifiedRequest(
            operation=OPERATION_MCP,
            input=deepcopy(request.get("params") or {}),
            metadata=metadata,
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        payload = {
            "jsonrpc": unified_request.metadata.get("jsonrpc", "2.0"),
            "method": unified_request.metadata.get("method"),
            "params": deepcopy(unified_request.input or {}),
        }
        if "id" in unified_request.metadata:
            payload["id"] = deepcopy(unified_request.metadata.get("id"))
        payload.update(deepcopy(unified_request.extra))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = raw_response if isinstance(raw_response, dict) else {}
        metadata = {"jsonrpc": response.get("jsonrpc", "2.0"), "id": deepcopy(response.get("id"))}
        extra = {k: deepcopy(v) for k, v in response.items() if k not in _RESPONSE_CORE_FIELDS}
        if "error" in response:
            # JSON-RPC errors are not provider exceptions here; they are protocol
            # payloads that must survive transform logging and response rebuilds.
            extra["error"] = deepcopy(response["error"])
        return UnifiedResponse(
            operation=OPERATION_MCP,
            data=[deepcopy(response["result"])] if "result" in response else [],
            metadata=metadata,
            raw=deepcopy(raw_response),
            extra=extra,
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:
        payload = {"jsonrpc": unified_response.metadata.get("jsonrpc", "2.0")}
        if "id" in unified_response.metadata:
            payload["id"] = deepcopy(unified_response.metadata.get("id"))
        if unified_response.extra.get("error") is not None:
            payload["error"] = deepcopy(unified_response.extra["error"])
        else:
            payload["result"] = deepcopy(unified_response.data[0] if unified_response.data else None)
        payload.update({k: deepcopy(v) for k, v in unified_response.extra.items() if k != "error"})
        return payload

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        data = raw_event if isinstance(raw_event, dict) else {"event": raw_event}
        return UnifiedStreamEvent(type=str(data.get("method") or data.get("type") or "message"), operation=OPERATION_MCP, raw=deepcopy(raw_event), extra=deepcopy(data))
