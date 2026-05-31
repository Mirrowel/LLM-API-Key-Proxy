from __future__ import annotations

import json

import pytest

from rotator_library.field_cache import FieldCacheRule
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

    assert events == chunks
    assert client.calls[0]["json"]["stream"] is True
    pass_names = [entry["pass_name"] for entry in _trace_entries(logger.log_dir)]
    assert "native_provider_stream_request" in pass_names
    assert pass_names.count("raw_native_provider_stream_chunk") == 2
    assert pass_names.count("parsed_native_stream_event") == 2
    assert "after_field_cache_extraction" in pass_names
    assert "after_field_cache_stream_extraction" in pass_names
    assert pass_names.count("formatted_client_stream_event") == 2
    trace_text = (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8")
    assert "opaque-vendor-state" not in trace_text


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
