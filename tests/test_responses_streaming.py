from __future__ import annotations

import pytest

from rotator_library.responses import InMemoryResponsesStore, ResponsesSSEFormatter, ResponsesService, ResponsesWebSocketFormatter
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
    async def acompletion(self, **kwargs):
        async def chunks():
            yield 'data: {"choices":[{"delta":{"content":"before"}}]}\n\n'
            raise RuntimeError("stream exploded")

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
async def test_stream_response_store_false_does_not_persist() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True, "store": False}, FakeStreamingClient())]

    response_id = events[0].split('"id": "')[1].split('"')[0]
    assert await store.get(response_id) is None


@pytest.mark.asyncio
async def test_stream_response_errors_emit_failed_event() -> None:
    service = ResponsesService(store=InMemoryResponsesStore())

    events = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "Hello", "stream": True}, FailingStreamingClient())]

    event_text = "".join(events)
    assert "event: response.failed" in event_text
    assert "stream exploded" in event_text
    assert event_text.endswith("data: [DONE]\n\n")


def test_transport_formatters_expose_sse_and_websocket_seam() -> None:
    assert ResponsesSSEFormatter().transport == "sse"
    websocket = ResponsesWebSocketFormatter()
    assert websocket.transport == "websocket"
    assert websocket.future_supported is True
    with pytest.raises(NotImplementedError):
        websocket.format_event("response.created", {})


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
