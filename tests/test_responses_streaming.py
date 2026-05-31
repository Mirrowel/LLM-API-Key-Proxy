from __future__ import annotations

import asyncio
import json

import pytest

from rotator_library.responses import InMemoryResponsesStore, ResponsesSSEFormatter, ResponsesService, ResponsesStoreSettings, ResponsesStreamEvent, ResponsesWebSocketFormatter
from rotator_library.transaction_logger import TransactionLogger


def _event_names(events: list[str]) -> list[str]:
    return [line.removeprefix("event: ") for event in events for line in event.splitlines() if line.startswith("event: ")]


class FakeStreamingClient:
    async def acompletion(self, **kwargs):
        async def chunks():
            yield 'data: {"id":"chat_stream","model":"gpt-test","choices":[{"delta":{"role":"assistant"}}]}\n\n'
            yield 'data: {"id":"chat_stream","model":"gpt-test","choices":[{"delta":{"content":"Hel"}}]}\n\n'
            yield 'data: {"id":"chat_stream","model":"gpt-test","choices":[{"delta":{"content":"lo"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3,"cost_details":{"total_cost":0.044,"source":"stream_provider"}}}\n\n'
            yield "data: [DONE]\n\n"

        return chunks()


class DelayedCloseableStream:
    def __init__(self, chunks: list[str], delays: list[float]) -> None:
        self.chunks = chunks
        self.delays = delays
        self.index = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        delay = self.delays[self.index]
        chunk = self.chunks[self.index]
        self.index += 1
        await asyncio.sleep(delay)
        return chunk

    async def aclose(self) -> None:
        self.closed = True


class DelayedStreamingClient:
    def __init__(self, stream: DelayedCloseableStream) -> None:
        self.stream = stream

    async def acompletion(self, **kwargs):
        return self.stream


class SlowAcquireStreamingClient:
    def __init__(self, stream: DelayedCloseableStream, delay: float) -> None:
        self.stream = stream
        self.delay = delay
        self.calls = 0

    async def acompletion(self, **kwargs):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return self.stream


class CancelAwareAcquireClient:
    def __init__(self, delay: float = 1.0) -> None:
        self.cancelled = False
        self.delay = delay

    async def acompletion(self, **kwargs):
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return DelayedCloseableStream([], [])


class DisconnectRequest:
    async def is_disconnected(self) -> bool:
        return True


class DisconnectAfterAcquireRequest:
    def __init__(self) -> None:
        self.calls = 0

    async def is_disconnected(self) -> bool:
        self.calls += 1
        return self.calls > 1


class FailingStreamingClient:
    def __init__(self, message: str = "stream exploded") -> None:
        self.message = message

    async def acompletion(self, **kwargs):
        message = self.message

        async def chunks():
            yield 'data: {"choices":[{"delta":{"content":"before"}}]}\n\n'
            raise RuntimeError(message)

        return chunks()


class ErrorChunkStreamingClient:
    async def acompletion(self, **kwargs):
        async def chunks():
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            yield 'data: {"error":{"message":"provider failed"}}\n\n'

        return chunks()


class EventErrorStreamingClient:
    async def acompletion(self, **kwargs):
        async def chunks():
            yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            yield 'event: error\ndata: {"message":"event failed"}\n\n'

        return chunks()


class FailingStore:
    async def save(self, response):
        raise RuntimeError("store failed Authorization: Bearer secret-token")

    async def get(self, response_id):
        return None

    async def delete(self, response_id):
        return False

    async def list_input_items(self, response_id):
        return None


@pytest.mark.asyncio
async def test_stream_response_emits_responses_sse_events_and_stores_final_response() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, FakeStreamingClient())]

    event_text = "".join(events)
    assert _event_names(events) == [
        "response.created",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_item.done",
        "response.completed",
    ]
    assert event_text.endswith("data: [DONE]\n\n")

    response_id = events[0].split('"id": "')[1].split('"')[0]
    stored = await store.get(response_id)
    assert stored is not None
    assert stored.output_items[0]["content"][0]["text"] == "Hello"
    assert stored.usage == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3, "cost_details": {"total_cost": 0.044, "source": "stream_provider"}}


@pytest.mark.asyncio
async def test_stream_response_store_false_does_not_persist(tmp_path) -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)
    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True, "store": False}, FakeStreamingClient(), transaction_logger=logger)]

    response_id = events[0].split('"id": "')[1].split('"')[0]
    assert await store.get(response_id) is None
    trace_text = (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8")
    assert "responses_store_skipped" in trace_text
    assert "responses_stored_stream_response" not in trace_text


@pytest.mark.asyncio
async def test_stream_response_errors_emit_failed_event() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, FailingStreamingClient())]

    event_text = "".join(events)
    assert "event: response.failed" in event_text
    assert "stream exploded" in event_text
    assert event_text.endswith("data: [DONE]\n\n")
    response_id = events[0].split('"id": "')[1].split('"')[0]
    stored = await store.get(response_id)
    assert stored is not None
    assert stored.status == "failed"


@pytest.mark.asyncio
async def test_stream_response_error_chunks_store_failed_with_partial_output() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, ErrorChunkStreamingClient())]

    event_text = "".join(events)
    response_id = events[0].split('"id": "')[1].split('"')[0]
    stored = await store.get(response_id)
    assert "event: response.failed" in event_text
    assert stored is not None
    assert stored.status == "failed"
    assert stored.output_items[0]["content"][0]["text"] == "partial"


@pytest.mark.asyncio
async def test_stream_response_event_error_frames_are_failed() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, EventErrorStreamingClient())]

    event_text = "".join(events)
    response_id = events[0].split('"id": "')[1].split('"')[0]
    stored = await store.get(response_id)
    assert "event: response.failed" in event_text
    assert stored is not None
    assert stored.response["error"]["message"] == "event failed"


@pytest.mark.asyncio
async def test_stream_response_can_skip_failed_storage() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store, store_settings=ResponsesStoreSettings(store_failed=False))

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, FailingStreamingClient())]

    response_id = events[0].split('"id": "')[1].split('"')[0]
    assert await store.get(response_id) is None


@pytest.mark.asyncio
async def test_stream_events_can_store_in_progress_state() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store, store_settings=ResponsesStoreSettings(store_in_progress=True))
    stream = service.stream_events({"model": "gpt-test", "input": "Hello", "stream": True}, FakeStreamingClient())

    created = await anext(stream)
    stored = await store.get(created.payload["id"])
    await stream.aclose()

    assert stored is not None
    assert stored.status == "in_progress"


@pytest.mark.asyncio
async def test_stream_response_store_failures_emit_store_specific_trace(tmp_path) -> None:
    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    service = ResponsesService(store=FailingStore())

    with pytest.raises(RuntimeError):
        _ = [
            chunk
            async for chunk in service.stream_response(
                {"model": "gpt-test", "input": "Hello", "stream": True},
                FakeStreamingClient(),
                transaction_logger=logger,
            )
        ]

    entries = [json.loads(line) for line in (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = [entry for entry in entries if entry["pass_name"] == "transform_log_error"]
    assert any(entry["data"]["failed_pass_name"] == "responses_store_stream_response" for entry in errors)
    assert "secret-token" not in json.dumps(errors)


@pytest.mark.asyncio
async def test_stream_current_state_store_failures_emit_store_specific_trace(tmp_path) -> None:
    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    service = ResponsesService(store=FailingStore(), store_settings=ResponsesStoreSettings(store_in_progress=True))

    with pytest.raises(RuntimeError):
        _ = [
            event
            async for event in service.stream_events(
                {"model": "gpt-test", "input": "Hello", "stream": True},
                FakeStreamingClient(),
                transaction_logger=logger,
            )
        ]

    entries = [json.loads(line) for line in (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    errors = [entry for entry in entries if entry["pass_name"] == "transform_log_error"]
    assert any(entry["data"]["failed_pass_name"] == "responses_store_stream_current_state" for entry in errors)
    assert "secret-token" not in json.dumps(errors)


@pytest.mark.asyncio
async def test_stream_response_failure_trace_scrubs_header_like_secret_text(tmp_path) -> None:
    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    service = ResponsesService(store=InMemoryResponsesStore())

    _ = [
        chunk
        async for chunk in service.stream_response(
            {"model": "gpt-test", "input": "Hello", "stream": True},
            FailingStreamingClient("{'Authorization': 'Bearer secret-token'}"),
            transaction_logger=logger,
        )
    ]

    trace_text = (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8")
    assert "secret-token" not in trace_text
    assert "[REDACTED]" in trace_text


def test_transport_formatters_expose_sse_and_websocket_seam() -> None:
    assert ResponsesSSEFormatter().transport == "sse"
    assert ResponsesSSEFormatter().format_stream_event(ResponsesStreamEvent("heartbeat", {"comment": "heartbeat"})) == ": heartbeat\n\n"
    websocket = ResponsesWebSocketFormatter()
    assert websocket.transport == "websocket"
    assert websocket.future_supported is True
    assert websocket.format_stream_event(ResponsesStreamEvent("response.created", {"id": "resp"})) == '{"event": "response.created", "data": {"id": "resp"}}'


@pytest.mark.asyncio
async def test_stream_response_emits_non_visible_heartbeat(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", "0.01")
    stream = DelayedCloseableStream(
        ['data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', "data: [DONE]\n\n"],
        [0.03, 0.0],
    )
    service = ResponsesService(store=InMemoryResponsesStore())

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, DelayedStreamingClient(stream))]

    assert ": heartbeat\n\n" in events
    assert "event: response.output_text.delta" in "".join(events)


@pytest.mark.asyncio
async def test_stream_response_ttfb_timeout_closes_upstream(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_TTFB_TIMEOUT_SECONDS", "0.01")
    stream = DelayedCloseableStream(['data: {"choices":[{"delta":{"content":"late"}}]}\n\n'], [0.05])
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, DelayedStreamingClient(stream))]

    event_text = "".join(events)
    assert stream.closed is True
    assert "event: response.failed" in event_text
    assert "ttfb" in event_text.lower()
    assert event_text.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_stream_response_heartbeat_does_not_reset_ttfb_timeout(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", "0.01")
    monkeypatch.setenv("STREAM_TTFB_TIMEOUT_SECONDS", "0.03")
    stream = DelayedCloseableStream(['data: {"choices":[{"delta":{"content":"late"}}]}\n\n'], [0.1])
    service = ResponsesService(store=InMemoryResponsesStore())

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, DelayedStreamingClient(stream))]

    event_text = "".join(events)
    assert ": heartbeat\n\n" in events
    assert stream.closed is True
    assert "event: response.failed" in event_text
    assert "ttfb" in event_text.lower()


@pytest.mark.asyncio
async def test_stream_response_acquire_wait_honors_ttfb_timeout(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_TTFB_TIMEOUT_SECONDS", "0.01")
    stream = DelayedCloseableStream(['data: {"choices":[{"delta":{"content":"late"}}]}\n\n'], [0.0])
    service = ResponsesService(store=InMemoryResponsesStore())

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, SlowAcquireStreamingClient(stream, 0.05))]

    event_text = "".join(events)
    assert "event: response.failed" in event_text
    assert "ttfb" in event_text.lower()


@pytest.mark.asyncio
async def test_stream_response_acquire_wait_can_emit_heartbeat(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", "0.01")
    stream = DelayedCloseableStream(['data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', "data: [DONE]\n\n"], [0.0, 0.0])
    service = ResponsesService(store=InMemoryResponsesStore())

    client = SlowAcquireStreamingClient(stream, 0.03)

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, client)]

    assert ": heartbeat\n\n" in events
    assert client.calls == 1
    assert "event: response.completed" in "".join(events)


@pytest.mark.asyncio
async def test_stream_response_heartbeat_does_not_drop_completed_first_chunk(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", "0.01")
    stream = DelayedCloseableStream(
        ['data: {"choices":[{"delta":{"content":"kept"}}]}\n\n', "data: [DONE]\n\n"],
        [0.05, 0.0],
    )
    service = ResponsesService(store=InMemoryResponsesStore())
    events = service.stream_events({"model": "gpt-test", "input": "Hello", "stream": True}, DelayedStreamingClient(stream))

    created = await anext(events)
    heartbeat = await anext(events)
    await asyncio.sleep(0.06)
    added = await anext(events)
    delta = await anext(events)
    await events.aclose()

    assert created.event_name == "response.created"
    assert heartbeat.heartbeat is True
    assert added.event_name == "response.output_item.added"
    assert delta.event_name == "response.output_text.delta"
    assert delta.payload["delta"] == "kept"


@pytest.mark.asyncio
async def test_stream_events_aclose_cancels_pending_read_after_heartbeat(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", "0.01")
    stream = DelayedCloseableStream(['data: {"choices":[{"delta":{"content":"late"}}]}\n\n'], [1.0])
    service = ResponsesService(store=InMemoryResponsesStore())
    events = service.stream_events({"model": "gpt-test", "input": "Hello", "stream": True}, DelayedStreamingClient(stream))

    assert (await anext(events)).event_name == "response.created"
    assert (await anext(events)).heartbeat is True
    await events.aclose()

    assert stream.closed is True


@pytest.mark.asyncio
async def test_stream_events_aclose_cancels_pending_acquire_after_heartbeat(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", "0.01")
    client = CancelAwareAcquireClient()
    service = ResponsesService(store=InMemoryResponsesStore())
    events = service.stream_events({"model": "gpt-test", "input": "Hello", "stream": True}, client)

    assert (await anext(events)).event_name == "response.created"
    assert (await anext(events)).heartbeat is True
    await events.aclose()

    assert client.cancelled is True


@pytest.mark.asyncio
async def test_stream_response_stall_timeout_preserves_partial_output(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_STALL_TIMEOUT_SECONDS", "0.01")
    stream = DelayedCloseableStream(
        ['data: {"choices":[{"delta":{"content":"partial"}}]}\n\n', 'data: {"choices":[{"delta":{"content":"late"}}]}\n\n'],
        [0.0, 0.05],
    )
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, DelayedStreamingClient(stream))]

    event_text = "".join(events)
    response_id = events[0].split('"id": "')[1].split('"')[0]
    stored = await store.get(response_id)
    assert stream.closed is True
    assert "event: response.failed" in event_text
    assert "stall" in event_text.lower()
    assert stored is not None
    assert stored.output_items[0]["content"][0]["text"] == "partial"


@pytest.mark.asyncio
async def test_stream_events_disconnect_closes_upstream(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_CANCEL_UPSTREAM_ON_DISCONNECT", "true")
    stream = DelayedCloseableStream(['data: {"choices":[{"delta":{"content":"late"}}]}\n\n'], [0.05])
    service = ResponsesService(store=InMemoryResponsesStore())

    events = [event async for event in service.stream_events({"model": "gpt-test", "input": "Hello", "stream": True}, DelayedStreamingClient(stream), request=DisconnectAfterAcquireRequest())]

    assert [event.event_name for event in events] == ["response.created"]
    assert stream.closed is True


@pytest.mark.asyncio
async def test_responses_stream_records_common_stream_metrics(tmp_path) -> None:
    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    service = ResponsesService(store=InMemoryResponsesStore())

    _ = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, FakeStreamingClient(), transaction_logger=logger)]

    trace_text = (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8")
    assert "stream_started" in trace_text
    assert "stream_first_byte" in trace_text
    assert "stream_first_visible_output" in trace_text
    assert "stream_completed" in trace_text
    assert "stream_metrics_final" in trace_text
    assert "stream_done_event" in trace_text
    assert "responses_stream_event_created" in trace_text
    assert "responses_stream_event_output_text_delta" in trace_text
    assert "responses_stream_event_completed" in trace_text
    assert "responses_sse_formatted_event" in trace_text
    usage_entry = [json.loads(line) for line in (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines() if '"usage_accounting_summary"' in line][-1]
    assert usage_entry["data"]["cost"]["provider_reported_cost"] == 0.044


@pytest.mark.asyncio
async def test_stream_events_are_transport_neutral_and_sse_wraps_them() -> None:
    service = ResponsesService(store=InMemoryResponsesStore())

    events = [event async for event in service.stream_events({"model": "gpt-test", "input": "Hello", "stream": True}, FakeStreamingClient())]
    sse = [ResponsesSSEFormatter().format_stream_event(event) for event in events]

    assert [event.event_name for event in events if not event.terminal] == [
        "response.created",
        "response.output_item.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[-1].terminal is True
    assert "".join(sse).endswith("data: [DONE]\n\n")
