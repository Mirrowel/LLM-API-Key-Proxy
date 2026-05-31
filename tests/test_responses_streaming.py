from __future__ import annotations

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
            yield 'data: {"id":"chat_stream","model":"gpt-test","choices":[{"delta":{"content":"lo"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n'
            yield "data: [DONE]\n\n"

        return chunks()


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
    assert stored.usage == {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}


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
    websocket = ResponsesWebSocketFormatter()
    assert websocket.transport == "websocket"
    assert websocket.future_supported is True
    assert websocket.format_stream_event(ResponsesStreamEvent("response.created", {"id": "resp"})) == '{"event": "response.created", "data": {"id": "resp"}}'


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
