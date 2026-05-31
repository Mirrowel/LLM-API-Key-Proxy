from __future__ import annotations

import json

import pytest

from rotator_library.adapters import PayloadAdapter, register_adapter
from rotator_library.field_cache import FieldCacheInjection, FieldCacheRule
from rotator_library.native_provider import NativeHTTPTransport, NativeProviderContext, NativeProviderExecutor
from rotator_library.transaction_logger import TransactionLogger


class FakeStreamingClient:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    async def stream_json_lines(self, endpoint, *, headers, json):
        self.calls.append({"endpoint": endpoint, "headers": headers, "json": json})
        for chunk in self.chunks:
            yield chunk


def _trace_entries(log_dir):
    return [json.loads(line) for line in (log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_native_provider_stream_traces_and_yields_formatted_events(tmp_path) -> None:
    logger = TransactionLogger("native", "gpt-test", parent_dir=tmp_path)
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
        field_cache_rules=(
            FieldCacheRule(name="stream_reasoning", source="stream_event", path="raw.choices.0.delta.reasoning_content", allow_missing_session=True),
            FieldCacheRule(name="stream_vendor_state", source="stream_event", path="raw.choices.0.delta.vendor_state", allow_missing_session=True),
        ),
        transaction_logger=logger,
    )
    chunks = [
        {"choices": [{"delta": {"content": "hi", "reasoning_content": "hidden", "vendor_state": "opaque-vendor-state"}}]},
        "[DONE]",
    ]
    client = FakeStreamingClient(chunks)

    events = [event async for event in NativeProviderExecutor().stream({"model": "gpt-test", "messages": []}, context, NativeHTTPTransport(client))]

    assert events == [chunks[0]]
    assert client.calls[0]["json"]["stream"] is True
    pass_names = [entry["pass_name"] for entry in _trace_entries(logger.log_dir)]
    assert "native_provider_stream_request" in pass_names
    assert pass_names.count("raw_native_provider_stream_chunk") == 2
    assert pass_names.count("parsed_native_stream_event") == 2
    assert "after_field_cache_extraction" in pass_names
    assert "after_field_cache_stream_extraction" in pass_names
    assert pass_names.count("formatted_client_stream_event") == 1
    trace_text = (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8")
    assert "opaque-vendor-state" not in trace_text


@pytest.mark.asyncio
async def test_native_provider_stream_traces_usage_accounting_summary(tmp_path) -> None:
    logger = TransactionLogger("native", "gpt-test", parent_dir=tmp_path)
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
        transaction_logger=logger,
    )
    chunks = [
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "cost_details": {"total_cost": 0.04, "currency": "USD", "source": "stream_usage"},
            },
        },
        "[DONE]",
    ]

    _ = [event async for event in NativeProviderExecutor().stream({"model": "gpt-test", "messages": []}, context, NativeHTTPTransport(FakeStreamingClient(chunks)))]

    entries = _trace_entries(logger.log_dir)
    summaries = [entry for entry in entries if entry["pass_name"] == "usage_accounting_summary"]
    assert summaries
    assert summaries[-1]["data"]["usage"]["input_tokens"] == 2
    assert summaries[-1]["data"]["usage"]["completion_tokens"] == 3
    assert summaries[-1]["data"]["usage"]["provider_reported_cost"] == 0.04
    assert summaries[-1]["data"]["cost"]["provider_reported_cost"] == 0.04


@pytest.mark.asyncio
async def test_native_provider_stream_logs_errors(tmp_path) -> None:
    class BrokenClient:
        async def stream_json_lines(self, endpoint, *, headers, json):
            raise RuntimeError("broken stream")
            yield None

    logger = TransactionLogger("native", "gpt-test", parent_dir=tmp_path)
    context = NativeProviderContext(provider="native", model="gpt-test", protocol_name="openai_chat", endpoint="https://example.test/chat", transaction_logger=logger)

    with pytest.raises(RuntimeError):
        [event async for event in NativeProviderExecutor().stream({"model": "gpt-test", "messages": []}, context, NativeHTTPTransport(BrokenClient()))]

    pass_names = [entry["pass_name"] for entry in _trace_entries(logger.log_dir)]
    assert "transform_log_error" in pass_names


@pytest.mark.asyncio
async def test_native_provider_stream_runs_stream_event_adapter_chain(tmp_path) -> None:
    class StreamTextAdapter(PayloadAdapter):
        name = "test_stream_text_adapter"
        supported_stages = ("stream_event",)

        async def transform_stream_event(self, payload, context):
            payload.delta.content[0].text = "adapted"
            return payload

    register_adapter(StreamTextAdapter, replace=True)
    logger = TransactionLogger("native", "gpt-test", parent_dir=tmp_path)
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
        adapter_names=("test_stream_text_adapter",),
        transaction_logger=logger,
    )

    events = [
        event
        async for event in NativeProviderExecutor().stream(
            {"model": "gpt-test", "messages": []},
            context,
            NativeHTTPTransport(FakeStreamingClient([{"choices": [{"delta": {"content": "before"}}]}, "[DONE]"])),
        )
    ]

    assert events[0]["choices"][0]["delta"]["content"] == "adapted"
    pass_names = [entry["pass_name"] for entry in _trace_entries(logger.log_dir)]
    assert "after_stream_event_adapter_chain" in pass_names


@pytest.mark.asyncio
async def test_native_cross_protocol_stream_formats_openai_chat_sse() -> None:
    context = NativeProviderContext(
        provider="claude_code",
        model="claude-sonnet-4-5",
        protocol_name="anthropic_messages",
        client_protocol_name="openai_chat",
        endpoint="https://example.test/messages",
        operation="messages",
    )
    chunks = [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        "[DONE]",
    ]

    events = [event async for event in NativeProviderExecutor().stream({"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 1}, context, NativeHTTPTransport(FakeStreamingClient(chunks)))]

    assert events[0].startswith("data: ")
    payload = json.loads(events[0][len("data: ") :].strip())
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "hi"
    assert "content_block_delta" not in events[0]


@pytest.mark.asyncio
async def test_native_provider_stream_extracts_unified_stream_events_for_later_requests() -> None:
    rule = FieldCacheRule(
        name="unified_stream_text",
        source="unified_stream_event",
        path="delta.content.0.text",
        inject=FieldCacheInjection(target="request", path="metadata.cached_stream_text"),
        allow_missing_session=True,
    )
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
        field_cache_rules=(rule,),
    )
    executor = NativeProviderExecutor()

    _ = [
        event
        async for event in executor.stream(
            {"model": "gpt-test", "messages": []},
            context,
            NativeHTTPTransport(FakeStreamingClient([{"choices": [{"delta": {"content": "stream-state"}}]}, "[DONE]"])),
        )
    ]
    second_client = FakeStreamingClient(["[DONE]"])
    _ = [
        event
        async for event in executor.stream(
            {"model": "gpt-test", "messages": []},
            context,
            NativeHTTPTransport(second_client),
        )
    ]

    assert second_client.calls[0]["json"]["metadata"]["cached_stream_text"] == "stream-state"
