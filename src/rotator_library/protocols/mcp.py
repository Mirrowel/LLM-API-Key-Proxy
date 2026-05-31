# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""MCP JSON-RPC carrier protocol adapter.

This is not a full MCP proxy implementation. It gives the native protocol layer a
lossless request/response carrier for future MCP gateway work, keeping method,
params, ids, results, errors, and JSON-RPC batch arrays intact for transform
logging and routing.
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
    supported_transports: ClassVar[tuple[str, ...]] = ("http",)
    future_transports: ClassVar[tuple[str, ...]] = ("sse", "websocket")

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        if isinstance(raw_request, list):
            return UnifiedRequest(operation=OPERATION_MCP, input=deepcopy(raw_request), metadata={"batch": True}, raw=deepcopy(raw_request))
        request = dict(raw_request or {})
        metadata = {
            "jsonrpc": request.get("jsonrpc", "2.0"),
            "method": request.get("method"),
            "has_id": "id" in request,
            "has_params": "params" in request,
        }
        if "id" in request:
            metadata["id"] = deepcopy(request.get("id"))
        return UnifiedRequest(
            operation=OPERATION_MCP,
            input=deepcopy(request.get("params")) if "params" in request else None,
            metadata=metadata,
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> Any:
        if unified_request.metadata.get("batch"):
            return deepcopy(unified_request.input or [])
        payload = {
            "jsonrpc": unified_request.metadata.get("jsonrpc", "2.0"),
            "method": unified_request.metadata.get("method"),
        }
        if unified_request.metadata.get("has_params", True):
            payload["params"] = deepcopy(unified_request.input)
        if unified_request.metadata.get("has_id"):
            payload["id"] = deepcopy(unified_request.metadata.get("id"))
        payload.update(deepcopy(unified_request.extra))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        if isinstance(raw_response, list):
            return UnifiedResponse(operation=OPERATION_MCP, data=deepcopy(raw_response), metadata={"batch": True}, raw=deepcopy(raw_response))
        response = raw_response if isinstance(raw_response, dict) else {}
        metadata = {"jsonrpc": response.get("jsonrpc", "2.0"), "has_id": "id" in response}
        if "id" in response:
            metadata["id"] = deepcopy(response.get("id"))
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

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> Any:
        if unified_response.metadata.get("batch"):
            return deepcopy(unified_response.data)
        payload = {"jsonrpc": unified_response.metadata.get("jsonrpc", "2.0")}
        if unified_response.metadata.get("has_id"):
            payload["id"] = deepcopy(unified_response.metadata.get("id"))
        if unified_response.extra.get("error") is not None:
            payload["error"] = deepcopy(unified_response.extra["error"])
        else:
            payload["result"] = deepcopy(unified_response.data[0] if unified_response.data else None)
        payload.update({k: deepcopy(v) for k, v in unified_response.extra.items() if k != "error"})
        return payload

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        data = raw_event if isinstance(raw_event, dict) else {"event": raw_event}
        return UnifiedStreamEvent(type=str(data.get("method") or data.get("type") or "message"), operation=OPERATION_MCP, error=deepcopy(data.get("error")), raw=deepcopy(raw_event), extra=deepcopy(data))
