# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Service layer for the OpenAI-compatible Responses API."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, AsyncGenerator, Optional

from ..protocols import ProtocolContext
from ..streaming import StreamEvent, StreamMonitor
from ..usage.accounting import extract_usage_record
from ..protocols.responses import ResponsesProtocol
from .bridge import ResponsesBridge
from .store import InMemoryResponsesStore, ResponsesStore
from .streaming import (
    ResponsesSSEFormatter,
    ResponsesStreamState,
    output_item_added_payload,
    output_item_done_payload,
    output_text_delta_payload,
    parse_chat_sse_chunk,
    response_completed_payload,
    response_created_payload,
    response_failed_payload,
)
from .types import StoredResponse
from .types import generate_response_id


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
        self._trace_responses_usage(transaction_logger, response_payload, unified.model, source="responses_response")

        if raw_request.get("store", True):
            stored = self._stored_response(raw_request, response_payload, parent)
            await self.store.save(stored)
            self._trace(transaction_logger, "stored_responses_response", stored.to_dict(), direction="metadata", stage="final")

        self._trace(transaction_logger, "final_responses_response", response_payload, direction="response", stage="final")
        return response_payload

    async def stream_response(
        self,
        raw_request: dict[str, Any],
        client: Any,
        *,
        request: Optional[Any] = None,
        transaction_logger: Optional[Any] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a Responses API request as HTTP SSE events."""

        if not raw_request.get("model"):
            raise ResponsesServiceError("'model' is required", status_code=400)
        formatter = ResponsesSSEFormatter()
        stream_request = dict(raw_request)
        stream_request["stream"] = True
        self._trace(transaction_logger, "raw_responses_request", stream_request, direction="request", stage="client")
        unified = self.protocol.parse_request(stream_request, ProtocolContext(source_protocol="responses", transport="sse"))
        self._trace(transaction_logger, "parsed_unified_request", unified.to_dict(), direction="request", stage="protocol")
        parent = await self._load_previous_response(unified.previous_response_id, transaction_logger)
        chat_kwargs = self.bridge.to_chat_kwargs(unified, parent_response=parent.response if parent else None)
        bridge_metadata = chat_kwargs.pop("_responses_bridge", {})
        chat_kwargs["stream"] = True
        self._trace(
            transaction_logger,
            "responses_bridge_chat_request",
            chat_kwargs,
            direction="request",
            stage="adapter",
            metadata={"bridge_metadata": bridge_metadata, "transport": "sse"},
        )

        response_id = generate_response_id()
        state = ResponsesStreamState(response_id=response_id, model=unified.model)
        usage = None
        item_started = False
        monitor = StreamMonitor(clock=time.monotonic)
        self._trace(
            transaction_logger,
            "stream_started",
            {"event": StreamEvent("started", protocol="responses").to_dict(), "metrics": monitor.metrics.to_dict()},
            direction="stream",
            stage="client",
            metadata={"transport": "sse"},
        )
        yield formatter.format_event("response.created", response_created_payload(response_id, unified.model))
        try:
            chat_stream = await client.acompletion(request=request, **chat_kwargs)
            async for raw_chunk in chat_stream:
                if monitor.metrics.first_byte_at is None:
                    monitor.record_event(StreamEvent("raw_chunk", protocol="responses", raw=raw_chunk))
                    self._trace(
                        transaction_logger,
                        "stream_first_byte",
                        {"metrics": monitor.metrics.to_dict()},
                        direction="stream",
                        stage="provider",
                        metadata={"transport": "sse"},
                    )
                self._trace(transaction_logger, "raw_chat_bridge_stream_chunk", raw_chunk, direction="stream", stage="provider")
                chunk = parse_chat_sse_chunk(raw_chunk)
                if not chunk or chunk.get("type") == "done":
                    continue
                self._trace(transaction_logger, "parsed_unified_stream_event", chunk, direction="stream", stage="protocol")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                delta = _chunk_text_delta(chunk)
                if not delta:
                    continue
                if not item_started:
                    item_started = True
                    added = output_item_added_payload(state)
                    self._trace(transaction_logger, "formatted_responses_stream_event", added, direction="stream", stage="final")
                    yield formatter.format_event("response.output_item.added", added)
                state = ResponsesStreamState(
                    response_id=state.response_id,
                    model=state.model,
                    output_text=state.output_text + delta,
                    output_item_id=state.output_item_id,
                )
                event = output_text_delta_payload(state, delta)
                first_visible = monitor.metrics.first_visible_output_at is None
                monitor.record_event(
                    StreamEvent(
                        "delta",
                        protocol="responses",
                        data=event,
                        visible_output=True,
                    )
                )
                if first_visible:
                    self._trace(
                        transaction_logger,
                        "stream_first_visible_output",
                        {"event": event, "metrics": monitor.metrics.to_dict()},
                        direction="stream",
                        stage="final",
                        metadata={"transport": "sse"},
                    )
                self._trace(transaction_logger, "formatted_responses_stream_event", event, direction="stream", stage="final")
                yield formatter.format_event("response.output_text.delta", event)

            if not item_started:
                yield formatter.format_event("response.output_item.added", output_item_added_payload(state))
            done_item = output_item_done_payload(state)
            yield formatter.format_event("response.output_item.done", done_item)
            completed = response_completed_payload(state, _usage_to_responses_stream(usage))
            self._trace_responses_usage(transaction_logger, completed, unified.model, source="responses_stream")
            await self._store_stream_response(stream_request, completed, parent)
            self._trace(transaction_logger, "stored_responses_stream_response", completed, direction="metadata", stage="final")
            monitor.complete()
            self._trace(
                transaction_logger,
                "stream_completed",
                {"metrics": monitor.metrics.to_dict()},
                direction="stream",
                stage="final",
                metadata={"transport": "sse"},
            )
            self._trace(
                transaction_logger,
                "stream_metrics_final",
                {"metrics": monitor.metrics.to_dict()},
                direction="stream",
                stage="final",
                metadata={"transport": "sse"},
            )
            yield formatter.format_event("response.completed", completed)
            yield formatter.done()
        except Exception as exc:
            monitor.record_event(StreamEvent("error", protocol="responses", data={"error_type": exc.__class__.__name__}))
            failed = response_failed_payload(response_id, unified.model, {"message": str(exc), "type": exc.__class__.__name__})
            self._log_transform_error(transaction_logger, "responses_stream", exc, stream_request)
            self._trace(
                transaction_logger,
                "stream_metrics_final",
                {"metrics": monitor.metrics.to_dict()},
                direction="stream",
                stage="final",
                metadata={"transport": "sse", "failed": True},
            )
            yield formatter.format_event("response.failed", failed)
            yield formatter.done()

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

    @staticmethod
    def _log_transform_error(transaction_logger: Optional[Any], pass_name: str, error: BaseException, payload: Any) -> None:
        if transaction_logger:
            transaction_logger.log_transform_error(pass_name, error, payload=payload, stage="adapter", protocol="responses")

    def _trace_responses_usage(
        self,
        transaction_logger: Optional[Any],
        response_payload: dict[str, Any],
        model: str,
        *,
        source: str,
    ) -> None:
        """Trace normalized Responses usage without changing stored payloads."""

        usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
        if not usage:
            return
        record = extract_usage_record(usage, provider="responses", model=model, source=source)
        self._trace(
            transaction_logger,
            "usage_accounting_summary",
            {"usage": record.to_dict()},
            direction="metadata",
            stage="final",
            metadata={"source": source},
        )

    async def _store_stream_response(
        self,
        raw_request: dict[str, Any],
        response_payload: dict[str, Any],
        parent: Optional[StoredResponse],
    ) -> None:
        if not raw_request.get("store", True):
            return
        await self.store.save(self._stored_response(raw_request, response_payload, parent))


def _input_items(raw_request: dict[str, Any]) -> list[Any]:
    value = raw_request.get("input")
    if value is None:
        return []
    return deepcopy(value if isinstance(value, list) else [value])


def _chunk_text_delta(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
    if not choices:
        return ""
    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _usage_to_responses_stream(usage: Any) -> Any:
    if not isinstance(usage, dict):
        return usage
    return {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
        "total_tokens": usage.get("total_tokens", 0),
    }
