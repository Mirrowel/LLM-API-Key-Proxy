# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Service layer for the OpenAI-compatible Responses API."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Optional

from ..protocols import ProtocolContext
from ..protocols.responses import ResponsesProtocol
from .bridge import ResponsesBridge
from .store import InMemoryResponsesStore, ResponsesStore
from .types import StoredResponse


class ResponsesServiceError(ValueError):
    """Error with an HTTP-compatible status code for proxy routes."""

    def __init__(self, message: str, *, status_code: int = 400, error_type: str = "invalid_request_error") -> None:
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(message)


class ResponsesService:
    """Create, store, retrieve, and delete Responses API objects.

    Phase 4 deliberately bridges through the existing chat-completions execution
    path. Native Responses-capable provider execution will replace the bridge for
    covered providers in later phases without changing the route/storage surface.
    """

    def __init__(
        self,
        *,
        protocol: Optional[ResponsesProtocol] = None,
        bridge: Optional[ResponsesBridge] = None,
        store: Optional[ResponsesStore] = None,
    ) -> None:
        self.protocol = protocol or ResponsesProtocol()
        self.bridge = bridge or ResponsesBridge(self.protocol)
        self.store = store or InMemoryResponsesStore()

    async def create_response(
        self,
        raw_request: dict[str, Any],
        client: Any,
        *,
        request: Optional[Any] = None,
        transaction_logger: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Create a non-streaming Responses object through the chat bridge."""

        if not raw_request.get("model"):
            raise ResponsesServiceError("'model' is required", status_code=400)
        if raw_request.get("stream"):
            raise ResponsesServiceError("Use stream_response for streaming requests", status_code=400)

        self._trace(transaction_logger, "raw_responses_request", raw_request, direction="request", stage="client")
        unified = self.protocol.parse_request(raw_request, ProtocolContext(source_protocol="responses"))
        self._trace(transaction_logger, "parsed_unified_request", unified.to_dict(), direction="request", stage="protocol")

        parent = await self._load_previous_response(unified.previous_response_id, transaction_logger)
        chat_kwargs = self.bridge.to_chat_kwargs(unified, parent_response=parent.response if parent else None)
        bridge_metadata = chat_kwargs.pop("_responses_bridge", {})
        self._trace(
            transaction_logger,
            "responses_bridge_chat_request",
            chat_kwargs,
            direction="request",
            stage="adapter",
            metadata={"bridge_metadata": bridge_metadata},
        )

        chat_response = await client.acompletion(request=request, **chat_kwargs)
        self._trace(transaction_logger, "raw_chat_bridge_response", self._response_to_dict(chat_response), direction="response", stage="provider")

        response_payload = self.bridge.from_chat_response(chat_response, unified)
        self._trace(transaction_logger, "parsed_unified_response", response_payload, direction="response", stage="protocol")

        if raw_request.get("store", True):
            stored = self._stored_response(raw_request, response_payload, parent)
            await self.store.save(stored)
            self._trace(transaction_logger, "stored_responses_response", stored.to_dict(), direction="metadata", stage="final")

        self._trace(transaction_logger, "final_responses_response", response_payload, direction="response", stage="final")
        return response_payload

    async def get_response(self, response_id: str) -> dict[str, Any]:
        """Return a stored response payload or raise a 404-compatible error."""

        stored = await self.store.get(response_id)
        if stored is None:
            raise ResponsesServiceError(f"Response not found: {response_id}", status_code=404, error_type="not_found_error")
        return deepcopy(stored.response)

    async def delete_response(self, response_id: str) -> dict[str, Any]:
        """Delete a stored response and return a compatible deletion object."""

        deleted = await self.store.delete(response_id)
        if not deleted:
            raise ResponsesServiceError(f"Response not found: {response_id}", status_code=404, error_type="not_found_error")
        return {"id": response_id, "object": "response.deleted", "deleted": True}

    async def list_input_items(self, response_id: str) -> dict[str, Any]:
        """Return stored input items for a response continuation."""

        items = await self.store.list_input_items(response_id)
        if items is None:
            raise ResponsesServiceError(f"Response not found: {response_id}", status_code=404, error_type="not_found_error")
        return {"object": "list", "data": items}

    async def _load_previous_response(self, response_id: Optional[str], transaction_logger: Optional[Any]) -> Optional[StoredResponse]:
        if not response_id:
            return None
        parent = await self.store.get(response_id)
        if parent is None:
            raise ResponsesServiceError(f"Previous response not found: {response_id}", status_code=404, error_type="not_found_error")
        self._trace(
            transaction_logger,
            "responses_previous_response_loaded",
            parent.to_dict(),
            direction="metadata",
            stage="adapter",
            metadata={
                "previous_response_id": response_id,
                "output_count": len(parent.output_items),
                "input_item_count": len(parent.input_items),
                "bridge_context_expanded": True,
            },
        )
        return parent

    def _stored_response(
        self,
        raw_request: dict[str, Any],
        response_payload: dict[str, Any],
        parent: Optional[StoredResponse],
    ) -> StoredResponse:
        return StoredResponse(
            id=str(response_payload["id"]),
            created_at=float(response_payload.get("created_at") or time.time()),
            model=str(response_payload.get("model") or raw_request.get("model") or ""),
            status=str(response_payload.get("status") or "completed"),
            request=deepcopy(raw_request),
            response=deepcopy(response_payload),
            input_items=_input_items(raw_request),
            output_items=deepcopy(response_payload.get("output") or []),
            usage=deepcopy(response_payload.get("usage")) if isinstance(response_payload.get("usage"), dict) else None,
            metadata={"previous_response_id": parent.id if parent else raw_request.get("previous_response_id")},
        )

    @staticmethod
    def _response_to_dict(response: Any) -> Any:
        if isinstance(response, dict):
            return deepcopy(response)
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "dict"):
            return response.dict()
        return repr(response)

    @staticmethod
    def _trace(
        transaction_logger: Optional[Any],
        pass_name: str,
        data: Any,
        *,
        direction: str,
        stage: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if not transaction_logger:
            return
        transaction_logger.log_transform_pass(
            pass_name,
            data,
            direction=direction,
            stage=stage,
            protocol="responses",
            metadata=metadata or {},
        )


def _input_items(raw_request: dict[str, Any]) -> list[Any]:
    value = raw_request.get("input")
    if value is None:
        return []
    return deepcopy(value if isinstance(value, list) else [value])
