"""Responses HTTP/neutral streaming tests.

The Responses execution surface is native (``client.agenerate`` with
``input_protocol="responses"``); the chat-completions bridge and its
``stream_events`` legacy surface were deleted in G11 Phase D, so these tests
drive provider Responses frames directly. Timing/heartbeat/TTFB machinery for
the neutral pipeline lives in the client streaming suites.
"""

from __future__ import annotations

from pathlib import Path

import json

import pytest

from rotator_library.protocols.canonical import complete_responses_object
from rotator_library.responses import (
    InMemoryResponsesStore,
    ResponsesSSEFormatter,
    ResponsesService,
    ResponsesStoreSettings,
    ResponsesStreamEvent,
    ResponsesWebSocketFormatter,
)
from rotator_library.transaction_logger import TransactionLogger
from tests.txn_helpers import error_records


_REQUIRED_RESPONSE_FIELDS = (
    "id",
    "object",
    "created_at",
    "status",
    "model",
    "output",
    "parallel_tool_calls",
    "tool_choice",
    "tools",
    "reasoning",
    "usage",
    "error",
    "incomplete_details",
    "metadata",
)


def _assert_sdk_response_object(obj: dict) -> None:
    assert isinstance(obj, dict)
    for field in _REQUIRED_RESPONSE_FIELDS:
        assert field in obj, f"missing SDK-required field {field!r}: {sorted(obj)}"
    assert obj["object"] == "response"
    assert isinstance(obj["created_at"], int)
    assert isinstance(obj["output"], list)
    assert isinstance(obj["tool_choice"], (str, dict))
    assert isinstance(obj["tools"], list)
    assert isinstance(obj["metadata"], dict)


def _event_names(events: list[str]) -> list[str]:
    return [line.removeprefix("event: ") for event in events for line in event.splitlines() if line.startswith("event: ")]


def _payloads_from_sse(events: list[str]) -> list[dict]:
    payloads = []
    for event in events:
        for line in event.splitlines():
            if not line.startswith("data: "):
                continue
            text = line[len("data: "):].strip()
            if text == "[DONE]":
                continue
            try:
                payloads.append(json.loads(text))
            except json.JSONDecodeError:
                pass
    return payloads


class NativeResponsesStreamingClient:
    def __init__(self) -> None:
        self.calls = []

    async def agenerate(self, payload, **kwargs):
        self.calls.append((payload, kwargs))

        async def chunks():
            yield 'event: response.created\ndata: {"type":"response.created","sequence_number":0,"response":{"id":"resp_native","object":"response","status":"in_progress","model":"gpt-test","output":[]}}\n\n'
            yield 'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_0","output_index":0,"content_index":0,"delta":"native"}\n\n'
            yield 'event: response.completed\ndata: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_native","object":"response","status":"completed","model":"gpt-test","output":[{"id":"msg_0","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"native"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'

        return chunks()


class NativeFailingAtCallClient:
    """Native client whose agenerate raises before the first frame."""

    async def agenerate(self, payload, *, input_protocol, **kwargs):
        raise RuntimeError("native agenerate exploded")


class NativeMidStreamRaisingClient:
    """Native client that yields one frame, then raises mid-stream."""

    async def agenerate(self, payload, *, input_protocol, **kwargs):
        async def stream():
            yield 'event: response.created\ndata: {"type":"response.created","sequence_number":0,"response":{"id":"resp_mid","object":"response","status":"in_progress","model":"gpt-test","output":[]}}\n\n'
            raise RuntimeError("mid-stream explosion")

        return stream()


class NativeTerminalLessClient:
    """Native client that ends without any terminal event."""

    async def agenerate(self, payload, *, input_protocol, **kwargs):
        async def stream():
            yield 'event: response.created\ndata: {"type":"response.created","sequence_number":0,"response":{"id":"resp_none","object":"response","status":"in_progress","model":"gpt-test","output":[]}}\n\n'

        return stream()


class NativeCompletedWithoutObjectClient:
    """Native client that emits a terminal completed event with no object."""

    async def agenerate(self, payload, *, input_protocol, **kwargs):
        async def stream():
            yield 'event: response.completed\ndata: {"type":"response.completed","sequence_number":0}\n\n'

        return stream()


class NativeErrorEventClient:
    """Native client that emits an out-of-band provider ``error`` event once."""

    async def agenerate(self, payload, *, input_protocol, **kwargs):
        async def stream():
            yield 'event: response.created\ndata: {"type":"response.created","sequence_number":0,"response":{"id":"resp_err","object":"response","status":"in_progress","model":"gpt-test","output":[]}}\n\n'
            yield 'event: error\ndata: {"type":"error","code":"server_error","message":"provider blew up","sequence_number":1}\n\n'
            yield 'event: response.completed\ndata: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_err","object":"response","status":"completed","model":"gpt-test","output":[]}}\n\n'

        return stream()


@pytest.mark.asyncio
async def test_native_responses_stream_uses_agenerate_and_stores_terminal_object() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)
    client = NativeResponsesStreamingClient()

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, client)]

    assert client.calls[0][1]["input_protocol"] == "responses"
    assert "event: response.output_text.delta" in "".join(events)
    stored = await store.get("resp_native")
    assert stored is not None
    assert stored.output_items[0]["content"][0]["text"] == "native"


@pytest.mark.asyncio
async def test_native_responses_stream_passes_frames_through() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)
    client = NativeResponsesStreamingClient()

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, client)]

    output = "".join(events)
    assert _event_names(events) == ["response.created", "response.output_text.delta", "response.completed"]
    assert "native" in output
    assert await store.get("resp_native") is not None


@pytest.mark.asyncio
async def test_native_stream_sequence_numbers_are_monotonic() -> None:
    """The stream path's provider sequence_number domain is strictly increasing."""
    service = ResponsesService(store=InMemoryResponsesStore())

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, NativeResponsesStreamingClient())]

    sequences = [payload["sequence_number"] for payload in _payloads_from_sse(events)]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)


@pytest.mark.asyncio
async def test_native_stream_neutral_events_are_transport_neutral() -> None:
    service = ResponsesService(store=InMemoryResponsesStore())

    events = [event async for event in service.stream_turn_events({"model": "gpt-test", "input": "Hello", "stream": True}, NativeResponsesStreamingClient())]

    assert isinstance(events[0], ResponsesStreamEvent)
    assert [event.event_name for event in events] == ["response.created", "response.output_text.delta", "response.completed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_class,expected_fragment",
    [
        (NativeFailingAtCallClient, "native agenerate exploded"),
        (NativeMidStreamRaisingClient, "mid-stream explosion"),
        (NativeTerminalLessClient, "ended without a terminal response event"),
    ],
)
async def test_native_stream_failures_end_in_terminal_frames(client_class, expected_fragment) -> None:
    """Every post-start native failure ends in protocol-valid terminal
    frames and a failed stored response — never a raised exception."""
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, client_class())]

    event_text = "".join(events)
    assert "event: response.failed" in event_text
    assert expected_fragment in event_text, f"failure mode {client_class.__name__} did not produce its own error"
    assert event_text.rstrip().endswith("data: [DONE]")
    failed_ids = [
        line.split('"id": "')[1].split('"')[0]
        for line in events
        if '"type": "response.failed"' in line or '"status":"failed"' in line or '"status": "failed"' in line
    ]
    assert failed_ids, "failed payload carries a response id"
    stored = await store.get(failed_ids[0])
    assert stored is not None
    assert stored.status == "failed"


@pytest.mark.asyncio
async def test_native_stream_failure_survives_store_errors() -> None:
    """A failing store must never cost the client its terminal frames."""

    class ExplodingStore(InMemoryResponsesStore):
        async def save(self, stored):
            raise RuntimeError("store exploded")

    service = ResponsesService(store=ExplodingStore())

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, NativeMidStreamRaisingClient())]

    event_text = "".join(events)
    assert "mid-stream explosion" in event_text  # the real mid-stream mode ran
    assert "event: response.failed" in event_text
    assert event_text.rstrip().endswith("data: [DONE]")


@pytest.mark.asyncio
async def test_synthesized_failure_continues_stream_sequence() -> None:
    """A synthesized response.failed after provider frames continues the SAME
    stream counter (never a reset, never the module global)."""
    service = ResponsesService(store=InMemoryResponsesStore())

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, NativeMidStreamRaisingClient())]

    failed = next(payload for payload in _payloads_from_sse(events) if payload.get("type") == "response.failed")
    assert failed["sequence_number"] == 1  # provider emitted sequence 0


@pytest.mark.asyncio
async def test_synthesized_failure_object_is_sdk_shaped() -> None:
    service = ResponsesService(store=InMemoryResponsesStore())

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, NativeMidStreamRaisingClient())]

    failed = next(payload for payload in _payloads_from_sse(events) if payload.get("type") == "response.failed")
    _assert_sdk_response_object(failed["response"])
    assert failed["response"]["status"] == "failed"
    assert isinstance(failed["response"]["error"], dict)
    assert isinstance(failed["response"]["error"]["code"], str)
    assert failed["response"]["incomplete_details"] is None


@pytest.mark.asyncio
async def test_synthesized_failure_correlates_provider_response_id() -> None:
    """The terminal failure reuses the provider's response id (no fresh mint)."""
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, NativeMidStreamRaisingClient())]

    failed = next(payload for payload in _payloads_from_sse(events) if payload.get("type") == "response.failed")
    assert failed["response"]["id"] == "resp_mid"
    assert await store.get("resp_mid") is not None


@pytest.mark.asyncio
async def test_native_provider_error_event_is_single_terminal() -> None:
    """A provider out-of-band error event ENDS the stream: no synthesized
    response.failed follows it (one terminal per stream)."""
    service = ResponsesService(store=InMemoryResponsesStore())

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, NativeErrorEventClient())]

    names = _event_names(events)
    assert "error" in names
    assert "response.failed" not in names
    assert "provider blew up" in "".join(events)


@pytest.mark.asyncio
async def test_native_completed_without_object_is_single_terminal() -> None:
    """A provider ``response.completed`` with no nested object is still a
    terminal: no synthesized response.failed follows it."""
    service = ResponsesService(store=InMemoryResponsesStore())

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, NativeCompletedWithoutObjectClient())]

    names = _event_names(events)
    assert names.count("response.completed") == 1
    assert "response.failed" not in names


@pytest.mark.asyncio
async def test_stream_response_store_failures_emit_store_specific_trace(tmp_path) -> None:
    class FailingStore:
        async def save(self, response):
            raise RuntimeError("store failed Authorization: Bearer secret-token")

        async def get(self, response_id, scope_key: str = "public"):
            return None

        async def delete(self, response_id, scope_key: str = "public"):
            return False

        async def list_input_items(self, response_id, scope_key: str = "public"):
            return None

    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    service = ResponsesService(store=FailingStore())

    events = [
        chunk
        async for chunk in service.stream_response(
            {"model": "gpt-test", "input": "Hello", "stream": True},
            NativeResponsesStreamingClient(),
            transaction_logger=logger,
        )
    ]

    # A failing store must never cost the client its terminal frames.
    assert any("response.completed" in event for event in events)

    records = error_records(logger)
    assert any(entry["failed_pass_name"] == "responses_store_stream_response" for entry in records)
    assert "secret-token" not in json.dumps(records)


@pytest.mark.asyncio
async def test_stream_current_state_store_failures_emit_store_specific_trace(tmp_path) -> None:
    class FailingStore:
        async def save(self, response):
            raise RuntimeError("store failed Authorization: Bearer secret-token")

        async def get(self, response_id, scope_key: str = "public"):
            return None

        async def delete(self, response_id, scope_key: str = "public"):
            return False

        async def list_input_items(self, response_id, scope_key: str = "public"):
            return None

    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    service = ResponsesService(store=FailingStore(), store_settings=ResponsesStoreSettings(store_in_progress=True))

    events = [
        event
        async for event in service.stream_turn_events(
            {"model": "gpt-test", "input": "Hello", "stream": True},
            NativeResponsesStreamingClient(),
            transaction_logger=logger,
        )
    ]

    # In-progress snapshots are best-effort: the stream still completes.
    assert any(event.event_name == "response.completed" for event in events)

    records = error_records(logger)
    assert any(entry["failed_pass_name"] == "responses_store_stream_current_state" for entry in records)
    assert "secret-token" not in json.dumps(records)


@pytest.mark.asyncio
async def test_native_stream_can_store_in_progress_state() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store, store_settings=ResponsesStoreSettings(store_in_progress=True))
    stream = service.stream_turn_events({"model": "gpt-test", "input": "Hello", "stream": True}, NativeResponsesStreamingClient())

    created = await anext(stream)
    stored = await store.get(created.payload["response"]["id"])
    await stream.aclose()

    assert stored is not None
    assert stored.status == "in_progress"


def test_transport_formatters_expose_sse_and_websocket_seam() -> None:
    assert ResponsesSSEFormatter().transport == "sse"
    assert ResponsesSSEFormatter().format_stream_event(ResponsesStreamEvent("heartbeat", {"comment": "heartbeat"})) == ": heartbeat\n\n"
    websocket = ResponsesWebSocketFormatter()
    assert websocket.transport == "websocket"
    # WS frames ARE the event objects (type + payload); SSE artifacts drop.
    assert (
        websocket.format_stream_event(ResponsesStreamEvent("response.created", {"type": "response.created", "id": "resp"}))
        == '{"type": "response.created", "id": "resp"}'
    )
    assert websocket.format_stream_event(ResponsesStreamEvent("heartbeat", {})) is None
    assert websocket.format_stream_event(ResponsesStreamEvent("done", {}, terminal=True)) is None


def test_complete_responses_object_fills_absence_and_keeps_provider_values() -> None:
    provider = {"id": "resp_x", "status": "completed", "model": "m", "usage": {"input_tokens": 1}, "tool_choice": "required"}
    filled = complete_responses_object(provider, response_id="ignored", model="m", status="completed")

    assert filled["id"] == "resp_x"
    assert filled["tool_choice"] == "required"  # provider value wins
    assert filled["usage"] == {"input_tokens": 1}
    _assert_sdk_response_object(filled)
