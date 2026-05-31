# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Service layer for the OpenAI-compatible Responses API."""

from __future__ import annotations

import asyncio
import json
import time
from copy import deepcopy
from typing import Any, AsyncGenerator, Optional

from ..protocols import ProtocolContext
from ..streaming import StreamEvent, StreamMonitor
from ..config.experimental import get_stream_runtime_settings
from ..usage.accounting import extract_usage_record
from ..usage.costs import CostCalculator
from ..protocols.responses import ResponsesProtocol
from .bridge import ResponsesBridge, responses_session_hints
from .store import InMemoryResponsesStore, ResponsesStore
from .streaming import (
    ResponsesSSEFormatter,
    ResponsesStreamEvent,
    ResponsesStreamState,
    output_item_added_payload,
    output_item_done_payload,
    output_text_delta_payload,
    parse_chat_sse_chunk,
    response_completed_payload,
    response_created_payload,
    response_failed_payload,
)
from .types import ResponsesStoreSettings, StoredResponse
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
        store_settings: Optional[ResponsesStoreSettings] = None,
    ) -> None:
        self.store_settings = store_settings or ResponsesStoreSettings()
        self.protocol = protocol or ResponsesProtocol()
        self.bridge = bridge or ResponsesBridge(self.protocol)
        self.store = store or InMemoryResponsesStore(max_items=self.store_settings.max_items)

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

        self._trace(transaction_logger, "responses_raw_request", raw_request, direction="request", stage="client")
        try:
            unified = self.protocol.parse_request(raw_request, ProtocolContext(source_protocol="responses"))
        except Exception as exc:
            self._log_transform_error(transaction_logger, "responses_parse_request", exc, raw_request)
            raise
        if transaction_logger:
            self._trace(transaction_logger, "responses_parsed_request", unified.to_dict(), direction="request", stage="protocol")

        parent = await self._load_previous_response(unified.previous_response_id, transaction_logger)
        try:
            parent_lineage = await self._load_response_lineage(parent)
            chat_kwargs = self.bridge.to_chat_kwargs(unified, parent_responses=[stored.to_dict() for stored in parent_lineage] if parent_lineage else None)
        except Exception as exc:
            self._log_transform_error(transaction_logger, "responses_bridge_chat_request", exc, unified.to_dict())
            raise
        bridge_metadata = chat_kwargs.pop("_responses_bridge", {})
        session_hints = chat_kwargs.pop("_session_tracking_hints", None)
        session_hints = _responses_session_hints(unified.previous_response_id, parent, session_hints)
        session_info: dict[str, Any] = {}
        chat_kwargs.update(_internal_client_kwargs(client, session_hints, session_info))
        trace_chat_kwargs = _without_internal_kwargs(chat_kwargs)
        self._trace(
            transaction_logger,
            "responses_bridge_chat_request",
            trace_chat_kwargs,
            direction="request",
            stage="adapter",
            metadata={"bridge_metadata": {**bridge_metadata, "has_session_hints": bool(session_hints)}},
        )

        chat_response = await client.acompletion(request=request, **chat_kwargs)
        if transaction_logger:
            self._trace(transaction_logger, "responses_bridge_chat_response", self._response_to_dict(chat_response), direction="response", stage="provider")

        try:
            response_payload = self.bridge.from_chat_response(chat_response, unified)
        except Exception as exc:
            self._log_transform_error(transaction_logger, "responses_bridge_chat_response", exc, self._response_to_dict(chat_response))
            raise
        _record_responses_session_anchor(session_info, response_payload)
        self._trace(transaction_logger, "responses_parsed_response", response_payload, direction="response", stage="protocol")
        self._trace_responses_usage(transaction_logger, response_payload, unified.model, source="responses_response")

        if raw_request.get("store", True):
            stored = self._stored_response(raw_request, response_payload, parent, session_info=session_info)
            try:
                await self.store.save(stored)
            except Exception as exc:
                self._log_transform_error(transaction_logger, "responses_store_response", exc, stored.to_dict())
                raise
            self._trace(transaction_logger, "responses_stored_response", stored.to_dict(), direction="metadata", stage="final")

        self._trace(transaction_logger, "responses_final_response", response_payload, direction="response", stage="final")
        return response_payload

    async def stream_response(
        self,
        raw_request: dict[str, Any],
        client: Any,
        *,
        request: Optional[Any] = None,
        transaction_logger: Optional[Any] = None,
        transport: str = "sse",
    ) -> AsyncGenerator[str, None]:
        """Stream a Responses API request as HTTP SSE events."""

        formatter = ResponsesSSEFormatter()
        async for event in self.stream_events(raw_request, client, request=request, transaction_logger=transaction_logger, transport=transport):
            formatted = formatter.format_stream_event(event)
            self._trace(
                transaction_logger,
                "responses_sse_formatted_event",
                formatted,
                direction="stream",
                stage="final",
                metadata={"event_name": event.event_name, "terminal": event.terminal, "transport": transport},
                scrub_strings=True,
            )
            if event.heartbeat:
                self._trace(
                    transaction_logger,
                    "responses_sse_formatted_heartbeat",
                    formatted,
                    direction="stream",
                    stage="final",
                    metadata={"transport": transport},
                    scrub_strings=True,
                )
            yield formatted

    async def validate_stream_request(self, raw_request: dict[str, Any]) -> None:
        """Validate stream-only preconditions before an HTTP response starts."""

        if not raw_request.get("model"):
            raise ResponsesServiceError("'model' is required", status_code=400)
        previous_response_id = raw_request.get("previous_response_id")
        if previous_response_id:
            await self._load_previous_response(str(previous_response_id), None)

    async def stream_events(
        self,
        raw_request: dict[str, Any],
        client: Any,
        *,
        request: Optional[Any] = None,
        transaction_logger: Optional[Any] = None,
        transport: str = "sse",
    ) -> AsyncGenerator[ResponsesStreamEvent, None]:
        """Yield transport-neutral Responses events for streaming transports."""

        if not raw_request.get("model"):
            raise ResponsesServiceError("'model' is required", status_code=400)
        stream_request = dict(raw_request)
        stream_request["stream"] = True
        self._trace(transaction_logger, "responses_raw_request", stream_request, direction="request", stage="client")
        try:
            unified = self.protocol.parse_request(stream_request, ProtocolContext(source_protocol="responses", transport=transport))
        except Exception as exc:
            self._log_transform_error(transaction_logger, "responses_parse_request", exc, stream_request)
            raise
        if transaction_logger:
            self._trace(transaction_logger, "responses_parsed_request", unified.to_dict(), direction="request", stage="protocol")
        parent = await self._load_previous_response(unified.previous_response_id, transaction_logger)
        try:
            parent_lineage = await self._load_response_lineage(parent)
            chat_kwargs = self.bridge.to_chat_kwargs(unified, parent_responses=[stored.to_dict() for stored in parent_lineage] if parent_lineage else None)
        except Exception as exc:
            self._log_transform_error(transaction_logger, "responses_bridge_chat_request", exc, unified.to_dict())
            raise
        bridge_metadata = chat_kwargs.pop("_responses_bridge", {})
        session_hints = chat_kwargs.pop("_session_tracking_hints", None)
        session_hints = _responses_session_hints(unified.previous_response_id, parent, session_hints)
        session_info: dict[str, Any] = {}
        chat_kwargs.update(_internal_client_kwargs(client, session_hints, session_info))
        chat_kwargs["stream"] = True
        trace_chat_kwargs = _without_internal_kwargs(chat_kwargs)
        self._trace(
            transaction_logger,
            "responses_bridge_chat_request",
            trace_chat_kwargs,
            direction="request",
            stage="adapter",
            metadata={"bridge_metadata": {**bridge_metadata, "has_session_hints": bool(session_hints)}, "transport": transport},
        )

        response_id = generate_response_id()
        state = ResponsesStreamState(response_id=response_id, model=unified.model)
        usage = None
        item_started = False
        monitor = StreamMonitor(clock=time.monotonic)
        stream_settings = get_stream_runtime_settings()
        chat_stream = None
        stream_iterator = None
        upstream_closed = False
        pending_next_task = None
        pending_next_started_at = None
        pending_next_last_heartbeat_at = None
        acquire_task = None
        acquire_started_at = None
        acquire_last_heartbeat_at = None
        ttfb_started_at = time.monotonic()

        async def cancel_task(task: Any) -> None:
            """Cancel and await an in-flight stream task before closing its source."""

            if task is None or task.done():
                return
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, StopAsyncIteration):
                return
            except Exception:
                return

        async def close_upstream(reason: str) -> None:
            """Best-effort close for upstream Responses bridge streams."""

            nonlocal upstream_closed
            if upstream_closed:
                return
            attempted = False
            for candidate in (stream_iterator, chat_stream):
                if candidate is None:
                    continue
                attempted = True
                try:
                    closer = getattr(candidate, "aclose", None)
                    if callable(closer):
                        await closer()
                        upstream_closed = True
                        self._trace(transaction_logger, "responses_stream_upstream_closed", {"reason": reason}, direction="stream", stage="provider", metadata={"transport": transport})
                        return
                    closer = getattr(candidate, "close", None)
                    if callable(closer):
                        closer()
                        upstream_closed = True
                        self._trace(transaction_logger, "responses_stream_upstream_closed", {"reason": reason}, direction="stream", stage="provider", metadata={"transport": transport})
                        return
                except Exception as exc:
                    self._trace(transaction_logger, "responses_stream_upstream_close_failed", {"reason": reason, "error_type": type(exc).__name__}, direction="stream", stage="provider", metadata={"transport": transport})
                    continue
            if attempted:
                self._trace(transaction_logger, "responses_stream_upstream_close_failed", {"reason": reason, "error_type": "no_close_method"}, direction="stream", stage="provider", metadata={"transport": transport})

        async def next_upstream_chunk(*, first: bool) -> tuple[str, Any]:
            """Return the next upstream chunk or a control marker."""

            timeout = stream_settings.ttfb_timeout_seconds if first else stream_settings.stall_timeout_seconds
            heartbeat = stream_settings.heartbeat_seconds
            nonlocal pending_next_task, pending_next_started_at, pending_next_last_heartbeat_at
            if pending_next_task is None:
                pending_next_task = asyncio.create_task(stream_iterator.__anext__())
                pending_next_started_at = time.monotonic()
                pending_next_last_heartbeat_at = pending_next_started_at
            next_task = pending_next_task
            started_at = ttfb_started_at if first else (pending_next_started_at or time.monotonic())
            while True:
                if next_task.done():
                    pending_next_task = None
                    pending_next_started_at = None
                    pending_next_last_heartbeat_at = None
                    return "chunk", next_task.result()
                if request is not None and await request.is_disconnected():
                    self._trace(transaction_logger, "responses_stream_disconnected", {"reason": "client_disconnected"}, direction="stream", stage="client", metadata={"transport": transport})
                    if stream_settings.cancel_upstream_on_disconnect:
                        await cancel_task(next_task)
                        pending_next_task = None
                        await close_upstream("client_disconnected")
                    return "disconnect", None
                elapsed = time.monotonic() - started_at
                waits = []
                if timeout is not None:
                    remaining_timeout = timeout - elapsed
                    if remaining_timeout <= 0:
                        await cancel_task(next_task)
                        pending_next_task = None
                        await close_upstream("ttfb_timeout" if first else "stall_timeout")
                        raise ResponsesServiceError(
                            f"Responses stream {'TTFB' if first else 'stall'} timeout",
                            status_code=504,
                            error_type="api_connection",
                        )
                    waits.append(remaining_timeout)
                if heartbeat is not None:
                    last_heartbeat_at = pending_next_last_heartbeat_at or started_at
                    remaining_heartbeat = heartbeat - (time.monotonic() - last_heartbeat_at)
                    if remaining_heartbeat <= 0:
                        pending_next_last_heartbeat_at = time.monotonic()
                        return "heartbeat", None
                    waits.append(remaining_heartbeat)
                wait_timeout = min(waits) if waits else None
                if wait_timeout is None:
                    chunk = await next_task
                    pending_next_task = None
                    pending_next_started_at = None
                    pending_next_last_heartbeat_at = None
                    return "chunk", chunk
                done, _ = await asyncio.wait({next_task}, timeout=wait_timeout)
                if done:
                    pending_next_task = None
                    pending_next_started_at = None
                    pending_next_last_heartbeat_at = None
                    return "chunk", next_task.result()
                last_heartbeat_at = pending_next_last_heartbeat_at or started_at
                if heartbeat is not None and time.monotonic() - last_heartbeat_at >= heartbeat:
                    pending_next_last_heartbeat_at = time.monotonic()
                    return "heartbeat", None

        async def acquire_upstream_stream() -> tuple[str, Any]:
            """Acquire the upstream stream under the same TTFB/disconnect policy."""

            nonlocal acquire_task, acquire_started_at, acquire_last_heartbeat_at
            if acquire_task is None:
                acquire_task = asyncio.create_task(client.acompletion(request=request, **chat_kwargs))
                acquire_started_at = ttfb_started_at
                acquire_last_heartbeat_at = acquire_started_at
            task = acquire_task
            started_at = acquire_started_at or time.monotonic()
            timeout = stream_settings.ttfb_timeout_seconds
            heartbeat = stream_settings.heartbeat_seconds
            while True:
                if task.done():
                    acquire_task = None
                    acquire_started_at = None
                    acquire_last_heartbeat_at = None
                    return "stream", task.result()
                if request is not None and await request.is_disconnected():
                    self._trace(transaction_logger, "responses_stream_disconnected", {"reason": "client_disconnected", "phase": "acquire"}, direction="stream", stage="client", metadata={"transport": transport})
                    await cancel_task(task)
                    acquire_task = None
                    acquire_started_at = None
                    acquire_last_heartbeat_at = None
                    return "disconnect", None
                waits = []
                elapsed = time.monotonic() - started_at
                if timeout is not None:
                    remaining_timeout = timeout - elapsed
                    if remaining_timeout <= 0:
                        await cancel_task(task)
                        acquire_task = None
                        acquire_started_at = None
                        acquire_last_heartbeat_at = None
                        raise ResponsesServiceError("Responses stream TTFB timeout", status_code=504, error_type="api_connection")
                    waits.append(remaining_timeout)
                if heartbeat is not None:
                    last_heartbeat_at = acquire_last_heartbeat_at or started_at
                    remaining_heartbeat = heartbeat - (time.monotonic() - last_heartbeat_at)
                    if remaining_heartbeat <= 0:
                        acquire_last_heartbeat_at = time.monotonic()
                        return "heartbeat", None
                    waits.append(remaining_heartbeat)
                wait_timeout = min(waits) if waits else None
                if wait_timeout is None:
                    stream = await task
                    acquire_task = None
                    acquire_started_at = None
                    acquire_last_heartbeat_at = None
                    return "stream", stream
                done, _ = await asyncio.wait({task}, timeout=wait_timeout)
                if done:
                    acquire_task = None
                    acquire_started_at = None
                    acquire_last_heartbeat_at = None
                    return "stream", task.result()
                last_heartbeat_at = acquire_last_heartbeat_at or started_at
                if heartbeat is not None and time.monotonic() - last_heartbeat_at >= heartbeat:
                    acquire_last_heartbeat_at = time.monotonic()
                    return "heartbeat", None

        if transaction_logger:
            self._trace(
                transaction_logger,
                "stream_started",
                {"event": StreamEvent("started", protocol="responses").to_dict(), "metrics": monitor.metrics.to_dict()},
                direction="stream",
                stage="client",
                metadata={"transport": transport},
            )
        created = response_created_payload(response_id, unified.model)
        self._trace(transaction_logger, "responses_stream_event_created", created, direction="stream", stage="final", metadata={"transport": transport})
        await self._store_stream_current_state(stream_request, created, parent, transaction_logger=transaction_logger)
        yield ResponsesStreamEvent("response.created", created)
        try:
            while chat_stream is None:
                marker, acquired = await acquire_upstream_stream()
                if marker == "disconnect":
                    return
                if marker == "heartbeat":
                    self._trace(transaction_logger, "responses_stream_heartbeat", {"comment": "heartbeat", "phase": "acquire"}, direction="stream", stage="transport", metadata={"transport": transport})
                    yield ResponsesStreamEvent("heartbeat", {"comment": "heartbeat"})
                    continue
                chat_stream = acquired
            stream_iterator = chat_stream.__aiter__()
            first_chunk = True
            while True:
                try:
                    marker, raw_chunk = await next_upstream_chunk(first=first_chunk)
                except StopAsyncIteration:
                    break
                if marker == "disconnect":
                    return
                if marker == "heartbeat":
                    self._trace(transaction_logger, "responses_stream_heartbeat", {"comment": "heartbeat"}, direction="stream", stage="transport", metadata={"transport": transport})
                    yield ResponsesStreamEvent("heartbeat", {"comment": "heartbeat"})
                    continue
                first_chunk = False
                if monitor.metrics.first_byte_at is None:
                    monitor.record_event(StreamEvent("raw_chunk", protocol="responses", raw=raw_chunk))
                    if transaction_logger:
                        self._trace(
                            transaction_logger,
                            "stream_first_byte",
                            {"metrics": monitor.metrics.to_dict()},
                            direction="stream",
                            stage="provider",
                            metadata={"transport": transport},
                        )
                self._trace(transaction_logger, "raw_chat_bridge_stream_chunk", raw_chunk, direction="stream", stage="provider")
                cost_usage = _responses_sse_cost_usage(raw_chunk)
                if cost_usage is not None:
                    usage = _merge_responses_stream_usage(usage, cost_usage)
                    self._trace(transaction_logger, "responses_stream_cost_event", cost_usage, direction="stream", stage="metadata", metadata={"transport": transport})
                    continue
                chunk = parse_chat_sse_chunk(raw_chunk)
                if not chunk or chunk.get("type") == "done":
                    continue
                self._trace(transaction_logger, "parsed_unified_stream_event", chunk, direction="stream", stage="protocol")
                if chunk.get("error") is not None or chunk.get("type") == "error":
                    raise ResponsesServiceError(_stream_error_message(chunk), status_code=502, error_type="upstream_error")
                if chunk.get("usage"):
                    usage = _merge_responses_stream_usage(chunk["usage"], usage)
                delta = _chunk_text_delta(chunk)
                if not delta:
                    continue
                if not item_started:
                    item_started = True
                    added = output_item_added_payload(state)
                    self._trace(transaction_logger, "responses_stream_event_output_item_added", added, direction="stream", stage="final", metadata={"transport": transport})
                    yield ResponsesStreamEvent("response.output_item.added", added)
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
                    if transaction_logger:
                        self._trace(
                            transaction_logger,
                            "stream_first_visible_output",
                            {"event": event, "metrics": monitor.metrics.to_dict()},
                            direction="stream",
                            stage="final",
                            metadata={"transport": transport},
                        )
                self._trace(transaction_logger, "responses_stream_event_output_text_delta", event, direction="stream", stage="final", metadata={"transport": transport})
                await self._store_stream_current_state(stream_request, _current_stream_payload(state), parent, transaction_logger=transaction_logger, session_info=session_info)
                yield ResponsesStreamEvent("response.output_text.delta", event)

            if not item_started:
                added = output_item_added_payload(state)
                self._trace(transaction_logger, "responses_stream_event_output_item_added", added, direction="stream", stage="final", metadata={"transport": transport})
                yield ResponsesStreamEvent("response.output_item.added", added)
            done_item = output_item_done_payload(state)
            self._trace(transaction_logger, "responses_stream_event_output_item_done", done_item, direction="stream", stage="final", metadata={"transport": transport})
            yield ResponsesStreamEvent("response.output_item.done", done_item)
            completed = response_completed_payload(state, _usage_to_responses_stream(usage))
            _record_responses_session_anchor(session_info, completed)
            self._trace_responses_usage(transaction_logger, completed, unified.model, source="responses_stream")
            stored = await self._store_stream_response(stream_request, completed, parent, transaction_logger=transaction_logger, session_info=session_info)
            if stored:
                self._trace(transaction_logger, "responses_stored_stream_response", completed, direction="metadata", stage="final")
            else:
                self._trace(transaction_logger, "responses_store_skipped", {"response_id": completed.get("id")}, direction="metadata", stage="final")
            monitor.complete()
            if transaction_logger:
                self._trace(
                    transaction_logger,
                    "stream_completed",
                    {"metrics": monitor.metrics.to_dict()},
                    direction="stream",
                    stage="final",
                    metadata={"transport": transport},
                )
                self._trace(
                    transaction_logger,
                    "stream_metrics_final",
                    {"metrics": monitor.metrics.to_dict()},
                    direction="stream",
                    stage="final",
                    metadata={"transport": transport},
                )
            self._trace(transaction_logger, "responses_stream_event_completed", completed, direction="stream", stage="final", metadata={"transport": transport})
            yield ResponsesStreamEvent("response.completed", completed)
            self._trace(transaction_logger, "stream_done_event", {"raw": "done"}, direction="stream", stage="final", metadata={"transport": transport})
            yield ResponsesStreamEvent("done", {}, terminal=True)
        except Exception as exc:
            monitor.record_event(StreamEvent("error", protocol="responses", data={"error_type": exc.__class__.__name__}))
            failed = response_failed_payload(response_id, unified.model, _stream_failure_error(exc))
            if state.output_text:
                failed["output"] = [output_item_done_payload(state)["item"]]
            self._log_transform_error(transaction_logger, "responses_stream", exc, stream_request)
            stored = await self._store_stream_response(stream_request, failed, parent, failed=True, transaction_logger=transaction_logger, session_info=session_info)
            if stored:
                self._trace(transaction_logger, "responses_stored_failed_stream_response", {"response_id": failed.get("id"), "status": "failed"}, direction="metadata", stage="final")
            self._trace(transaction_logger, "responses_stream_event_failed", failed, direction="stream", stage="final", metadata={"transport": transport}, scrub_strings=True)
            if transaction_logger:
                self._trace(
                    transaction_logger,
                    "stream_metrics_final",
                    {"metrics": monitor.metrics.to_dict()},
                    direction="stream",
                    stage="final",
                    metadata={"transport": transport, "failed": True},
                )
            yield ResponsesStreamEvent("response.failed", failed)
            self._trace(transaction_logger, "stream_done_event", {"raw": "done"}, direction="stream", stage="final", metadata={"transport": transport, "failed": True})
            yield ResponsesStreamEvent("done", {}, terminal=True)
        finally:
            if chat_stream is None and acquire_task is not None and acquire_task.done() and not acquire_task.cancelled():
                try:
                    chat_stream = acquire_task.result()
                    stream_iterator = chat_stream.__aiter__()
                except Exception:
                    chat_stream = None
            await cancel_task(pending_next_task)
            await cancel_task(acquire_task)
            if chat_stream is not None and not upstream_closed:
                await close_upstream("wrapper_exit")

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

    async def _load_response_lineage(self, parent: Optional[StoredResponse], *, max_depth: int = 20) -> list[StoredResponse]:
        """Return parent continuation lineage from oldest to newest."""

        if parent is None:
            return []
        lineage: list[StoredResponse] = []
        seen: set[str] = set()
        current: Optional[StoredResponse] = parent
        while current is not None and current.id not in seen and len(lineage) < max_depth:
            seen.add(current.id)
            lineage.append(current)
            previous_id = current.request.get("previous_response_id") if isinstance(current.request, dict) else None
            if not previous_id:
                break
            current = await self.store.get(str(previous_id))
        return list(reversed(lineage))

    async def _load_previous_response(self, response_id: Optional[str], transaction_logger: Optional[Any]) -> Optional[StoredResponse]:
        if not response_id:
            return None
        parent = await self.store.get(response_id)
        if parent is None:
            raise ResponsesServiceError(f"Previous response not found: {response_id}", status_code=404, error_type="not_found_error")
        if transaction_logger:
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
        *,
        session_info: Optional[dict[str, Any]] = None,
    ) -> StoredResponse:
        session_info = session_info or {}
        session_affinity_key = session_info.get("session_affinity_key") or (parent.metadata.get("session_affinity_key") if parent else None)
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
            metadata={
                "previous_response_id": parent.id if parent else raw_request.get("previous_response_id"),
                "response_id": response_payload.get("id"),
                "session_affinity_key": session_affinity_key,
            },
            session_id=session_info.get("session_id") or (parent.session_id if parent else None),
            scope_key=session_info.get("scope_key") or (parent.scope_key if parent else None),
            classifier=session_info.get("classifier") or (parent.classifier if parent else None),
            expires_at=_expires_at(self.store_settings),
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
        scrub_strings: bool = False,
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
            scrub_strings=scrub_strings,
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

        if not transaction_logger:
            return
        usage = response_payload.get("usage") if isinstance(response_payload, dict) else None
        if not usage:
            return
        record = extract_usage_record(usage, provider="responses", model=model, source=source)
        cost_breakdown = CostCalculator().calculate(record, model=model, provider="responses")
        self._trace(
            transaction_logger,
            "usage_accounting_summary",
            {"usage": record.to_dict(), "cost": cost_breakdown.to_dict()},
            direction="metadata",
            stage="final",
            metadata={"source": source, "pricing_source": cost_breakdown.pricing_source},
        )

    async def _store_stream_response(
        self,
        raw_request: dict[str, Any],
        response_payload: dict[str, Any],
        parent: Optional[StoredResponse],
        *,
        failed: bool = False,
        transaction_logger: Optional[Any] = None,
        session_info: Optional[dict[str, Any]] = None,
    ) -> bool:
        if not raw_request.get("store", True):
            return False
        if failed and not self.store_settings.store_failed:
            return False
        stored = self._stored_response(raw_request, response_payload, parent, session_info=session_info)
        try:
            await self.store.save(stored)
        except Exception as exc:
            self._log_transform_error(transaction_logger, "responses_store_stream_response", exc, stored.to_dict())
            raise
        return True

    async def _store_stream_current_state(
        self,
        raw_request: dict[str, Any],
        response_payload: dict[str, Any],
        parent: Optional[StoredResponse],
        *,
        transaction_logger: Optional[Any],
        session_info: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Optionally persist in-progress stream state for retrieval surfaces."""

        if not self.store_settings.store_in_progress or not raw_request.get("store", True):
            return False
        stored = self._stored_response(raw_request, response_payload, parent, session_info=session_info)
        try:
            await self.store.save(stored)
        except Exception as exc:
            self._log_transform_error(transaction_logger, "responses_store_stream_current_state", exc, stored.to_dict())
            raise
        self._trace(
            transaction_logger,
            "responses_stored_stream_current_state",
            {"response_id": response_payload.get("id"), "status": response_payload.get("status")},
            direction="metadata",
            stage="final",
        )
        return True


def _input_items(raw_request: dict[str, Any]) -> list[Any]:
    value = raw_request.get("input")
    if value is None:
        return []
    return deepcopy(value if isinstance(value, list) else [value])


def _current_stream_payload(state: ResponsesStreamState) -> dict[str, Any]:
    """Return a retrievable in-progress Responses object for stream state."""

    payload = response_completed_payload(state)
    payload["status"] = "in_progress"
    return payload


def _expires_at(settings: ResponsesStoreSettings) -> Optional[float]:
    """Return the expiration timestamp for a new stored response, if enabled."""

    ttl = settings.ttl_seconds
    if ttl is None or ttl <= 0:
        return None
    return time.time() + ttl


def _internal_client_kwargs(client: Any, session_hints: Any, session_info: dict[str, Any]) -> dict[str, Any]:
    """Return hidden kwargs only for the internal RotatingClient path."""

    if not _supports_internal_context_kwargs(client):
        return {}
    kwargs: dict[str, Any] = {"_request_context_callback": _capture_request_context(session_info)}
    if session_hints:
        kwargs["_session_tracking_hints"] = session_hints
    return kwargs


def _supports_internal_context_kwargs(client: Any) -> bool:
    """Return whether a client is the proxy's internal rotating client."""

    return hasattr(client, "_request_builder") and hasattr(client, "_executor")


def _capture_request_context(session_info: dict[str, Any]):
    """Build a callback that records non-secret request context metadata."""

    def capture(context: Any) -> None:
        session_info["session_id"] = getattr(context, "session_id", None)
        session_info["session_affinity_key"] = getattr(context, "session_affinity_key", None)
        session_info["scope_key"] = getattr(context, "usage_manager_key", None)
        session_info["classifier"] = getattr(context, "classifier", None)
        session_info["session_tracker"] = getattr(context, "session_tracker", None)
        session_info["provider"] = getattr(context, "provider", None)
        session_info["model"] = getattr(context, "model", None)
        session_info["tracking_namespace"] = getattr(context, "session_tracking_namespace", None)

    return capture


def _without_internal_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return trace/provider-visible kwargs without proxy-internal controls."""

    return {key: deepcopy(value) for key, value in kwargs.items() if not key.startswith("_")}


def _responses_session_hints(previous_response_id: Optional[str], parent: Optional[StoredResponse], fallback: Any) -> Any:
    """Prefer parent stored affinity when building Responses continuation hints."""

    if not previous_response_id:
        return None
    parent_affinity = parent.metadata.get("session_affinity_key") if parent else None
    return responses_session_hints(previous_response_id, affinity_key=parent_affinity) or fallback


def _record_responses_session_anchor(session_info: dict[str, Any], response_payload: dict[str, Any]) -> None:
    """Record emitted Responses IDs as response-derived session evidence."""

    tracker = session_info.get("session_tracker")
    session_id = session_info.get("session_id")
    if not tracker or not session_id or not response_payload.get("id"):
        return
    tracker.record_response(
        session_id,
        provider=session_info.get("provider"),
        model=session_info.get("model"),
        scope_key=session_info.get("scope_key"),
        tracking_namespace=session_info.get("tracking_namespace"),
        response={"id": response_payload.get("id"), "object": "response"},
    )


def _stream_error_message(chunk: dict[str, Any]) -> str:
    """Return a compact, client-safe message for upstream stream error chunks."""

    error = chunk.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("type")
        if message:
            return str(message)
    message = chunk.get("message")
    return str(message) if message else "Upstream stream error"


def _stream_failure_error(exc: Exception) -> dict[str, Any]:
    """Return a client-safe Responses stream failure object."""

    error_type = getattr(exc, "error_type", None) or exc.__class__.__name__
    result = {"message": str(exc), "type": str(error_type)}
    if isinstance(exc, ResponsesServiceError) and error_type == "api_connection":
        text = str(exc).lower()
        if "ttfb" in text:
            result["timeout_type"] = "ttfb"
        elif "stall" in text:
            result["timeout_type"] = "stall"
    return result


def _responses_sse_cost_usage(chunk: Any) -> Optional[dict[str, Any]]:
    """Extract Responses stream cost metadata from SSE comments/events."""

    if not isinstance(chunk, str):
        return None
    payload = _responses_sse_cost_payload(chunk)
    if payload is None:
        return None
    if isinstance(payload, (int, float, str)):
        payload = {"provider_reported_cost": payload, "source": "responses_sse_cost"}
    if not isinstance(payload, dict):
        return None
    cost = payload.get("provider_reported_cost", payload.get("total_cost", payload.get("cost")))
    if cost is None:
        return None
    return {
        "provider_reported_cost": cost,
        "currency": payload.get("currency", "USD"),
        "cost_details": payload,
    }


def _responses_sse_cost_payload(chunk: str) -> Any:
    event_type = None
    data_lines: list[str] = []
    for line in chunk.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith(":"):
            comment = stripped[1:].strip()
            if comment.startswith("cost"):
                return _parse_cost_text(comment[4:].strip())
            continue
        if stripped.startswith("event:"):
            event_type = stripped[6:].strip()
            continue
        if stripped.startswith("data:"):
            data_lines.append(stripped[5:].strip())
    if event_type == "cost" and data_lines:
        return _parse_cost_text("\n".join(data_lines).strip())
    return None


def _parse_cost_text(text: str) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return float(text)
        except ValueError:
            return None


def _merge_responses_stream_usage(primary: Any, fallback_cost: Any) -> Any:
    """Merge earlier stream cost metadata into later token usage when needed."""

    if not isinstance(primary, dict):
        return fallback_cost if primary is None else primary
    if not isinstance(fallback_cost, dict):
        return primary
    merged = deepcopy(primary)
    has_cost = any(key in merged for key in ("cost_details", "cost", "total_cost", "provider_reported_cost"))
    if has_cost:
        return merged
    for key in ("cost_details", "cost", "total_cost", "provider_reported_cost", "currency"):
        if key in fallback_cost:
            merged[key] = deepcopy(fallback_cost[key])
    return merged


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
    result = {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
        "total_tokens": usage.get("total_tokens", 0),
    }
    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    if isinstance(prompt_details, dict):
        result["input_tokens_details"] = {"cached_tokens": prompt_details.get("cached_tokens", 0)}
    completion_details = usage.get("completion_tokens_details") or usage.get("output_tokens_details")
    if isinstance(completion_details, dict):
        result["output_tokens_details"] = {"reasoning_tokens": completion_details.get("reasoning_tokens", 0)}
    for key in ("cost_details", "cost", "total_cost", "provider_reported_cost", "currency"):
        if key in usage:
            result[key] = deepcopy(usage[key])
    return result
