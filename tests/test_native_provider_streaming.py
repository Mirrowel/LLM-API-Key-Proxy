from __future__ import annotations

from pathlib import Path

import json

import pytest

from rotator_library.adapters import PayloadAdapter, register_adapter
from rotator_library.core.errors import StreamedAPIError
from rotator_library.field_cache import FieldCacheInjection, FieldCacheRule
from rotator_library.native_provider import NativeHTTPTransport, NativeProviderContext, NativeProviderExecutor
from rotator_library.transaction_logger import TransactionLogger




@pytest.fixture(autouse=True)
def _trace_level_2(monkeypatch):
    """Trace mechanics live at L2 (D15 tiers)."""
    monkeypatch.setenv("TRANSACTION_LOG_LEVEL", "2")
class FakeStreamingClient:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    async def stream_json_lines(self, endpoint, *, headers, json):
        self.calls.append({"endpoint": endpoint, "headers": headers, "json": json})
        for chunk in self.chunks:
            yield chunk


def _trace_entries(log_dir):
    from rotator_library.utils import zstd_io

    return zstd_io.read_jsonl_any(Path(log_dir) / "transform_trace.jsonl")

def _trace_text(log_dir):
    from rotator_library.utils import zstd_io

    entries = zstd_io.read_jsonl_any(Path(log_dir) / "transform_trace.jsonl")
    return "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries)

    return zstd_io.read_jsonl_any(Path(log_dir) / "transform_trace.jsonl")



@pytest.mark.asyncio
async def test_native_provider_stream_traces_and_yields_formatted_events(tmp_path) -> None:
    logger = TransactionLogger("native", "gpt-test", parent_dir=tmp_path)
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
        field_cache_rules=(
            FieldCacheRule(name="stream_reasoning", source="stream_event", path="raw.choices.0.delta.reasoning_content", allow_missing_session=True, scope=("provider", "model")),
            FieldCacheRule(name="stream_vendor_state", source="stream_event", path="raw.choices.0.delta.vendor_state", allow_missing_session=True, scope=("provider", "model")),
        ),
        transaction_logger=logger,
    )
    chunks = [
        {"choices": [{"delta": {"content": "hi", "reasoning_content": "hidden", "vendor_state": "opaque-vendor-state"}}]},
        "[DONE]",
    ]
    client = FakeStreamingClient(chunks)

    events = [event async for event in NativeProviderExecutor().stream({"model": "gpt-test", "messages": []}, context, NativeHTTPTransport(client))]

    assert len(events) == 2
    first_event = events[0]
    assert first_event.type == "message_delta"
    assert first_event.delta.content[0].text == "hi"
    assert first_event.delta.reasoning is not None
    assert first_event.delta.reasoning[0].text == "hidden"
    assert events[-1].type == "done"
    assert client.calls[0]["json"]["stream"] is True
    pass_names = [entry["pass_name"] for entry in _trace_entries(logger.log_dir)]
    assert "native_provider_stream_request" in pass_names
    assert pass_names.count("raw_native_provider_stream_chunk") == 2
    assert pass_names.count("parsed_native_stream_event") == 2
    assert "after_field_cache_extraction" in pass_names
    assert "after_field_cache_stream_extraction" in pass_names
    trace_text = _trace_text(logger.log_dir)
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
async def test_native_provider_stream_preserves_earlier_cost_when_later_usage_arrives(tmp_path) -> None:
    logger = TransactionLogger("native", "gpt-test", parent_dir=tmp_path)
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
        transaction_logger=logger,
    )
    chunks = [
        {"choices": [], "usage": {"cost_details": {"total_cost": 0.07, "source": "early_cost"}}},
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        "[DONE]",
    ]

    _ = [event async for event in NativeProviderExecutor().stream({"model": "gpt-test", "messages": []}, context, NativeHTTPTransport(FakeStreamingClient(chunks)))]

    summaries = [entry for entry in _trace_entries(logger.log_dir) if entry["pass_name"] == "usage_accounting_summary"]
    assert summaries[-1]["data"]["usage"]["input_tokens"] == 2
    assert summaries[-1]["data"]["usage"]["completion_tokens"] == 3
    assert summaries[-1]["data"]["usage"]["provider_reported_cost"] == 0.07


@pytest.mark.asyncio
async def test_native_provider_stream_preserves_cost_when_later_raw_usage_arrives(tmp_path) -> None:
    logger = TransactionLogger("native", "gpt-test", parent_dir=tmp_path)
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
        transaction_logger=logger,
    )
    chunks = [
        {"choices": [], "usage": {"cost_details": {"total_cost": 0.08, "source": "early_cost"}}},
        {"choices": [], "prompt_tokens": 2, "completion_tokens": 3},
        "[DONE]",
    ]

    _ = [event async for event in NativeProviderExecutor().stream({"model": "gpt-test", "messages": []}, context, NativeHTTPTransport(FakeStreamingClient(chunks)))]

    summaries = [entry for entry in _trace_entries(logger.log_dir) if entry["pass_name"] == "usage_accounting_summary"]
    assert summaries[-1]["data"]["usage"]["input_tokens"] == 2
    assert summaries[-1]["data"]["usage"]["completion_tokens"] == 3
    assert summaries[-1]["data"]["usage"]["provider_reported_cost"] == 0.08


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
async def test_native_provider_error_event_raises_for_rotation_before_client_formatting() -> None:
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
    )

    with pytest.raises(StreamedAPIError) as raised:
        _ = [
            event
            async for event in NativeProviderExecutor().stream(
                {"model": "gpt-test", "messages": []},
                context,
                NativeHTTPTransport(FakeStreamingClient([{"error": {"type": "rate_limit", "message": "rotate me"}}])),
            )
        ]

    assert raised.value.data["error"]["type"] == "rate_limit"


@pytest.mark.asyncio
async def test_native_provider_stream_runs_stream_event_adapter_chain(tmp_path) -> None:
    class StreamTextAdapter(PayloadAdapter):
        name = "test_stream_text_adapter"
        supported_stages = ("stream_event",)

        async def transform_stream_event(self, payload, context):
            # W7 contract (plan §2.5): stream adapters run on the NEUTRAL
            # parsed event — protocol-free, client-agnostic, after parsing.
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

    # The neutral-event edit landed after parse — the output carries it.
    assert events[0].delta.content[0].text == "adapted"
    pass_names = [entry["pass_name"] for entry in _trace_entries(logger.log_dir)]
    assert "after_stream_event_adapter_chain" in pass_names


@pytest.mark.asyncio
async def test_native_cross_protocol_stream_formats_openai_chat_sse() -> None:
    from rotator_library.client.stream_ops import NeutralStreamPipeline
    from rotator_library.protocols.types import ProtocolContext

    context = NativeProviderContext(
        provider="synthetic",
        model="claude-sonnet-4-5",
        protocol_name="anthropic_messages",
        input_protocol_name="anthropic_messages",
        client_protocol_name="openai_chat",
        endpoint="https://example.test/messages",
        operation="messages",
    )
    chunks = [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        "[DONE]",
    ]

    events = [event async for event in NativeProviderExecutor().stream({"model": "claude-sonnet-4-5", "messages": [], "max_tokens": 1}, context, NativeHTTPTransport(FakeStreamingClient(chunks)))]

    async def event_source():
        for event in events:
            yield event

    pipeline = NeutralStreamPipeline(
        client_protocol_name="openai_chat",
        protocol_context=ProtocolContext(
            provider="synthetic",
            model="claude-sonnet-4-5",
            source_protocol="anthropic_messages",
            target_protocol="openai_chat",
            input_protocol="anthropic_messages",
            client_protocol="openai_chat",
        ),
        model="claude-sonnet-4-5",
    )
    frames = [frame async for frame in pipeline.run(event_source())]

    payload = json.loads(frames[0][len("data: ") :].strip())
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "hi"
    assert "content_block_delta" not in frames[0]
    assert frames[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_native_provider_stream_extracts_unified_stream_events_for_later_requests() -> None:
    rule = FieldCacheRule(
        name="unified_stream_text",
        source="unified_stream_event",
        path="delta.content.0.text",
        inject=FieldCacheInjection(target="request", path="metadata.cached_stream_text"),
        allow_missing_session=True,
        scope=("provider", "model"),
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol", "operation", "request_payload", "chunks", "injected_path", "expected", "secret"),
    [
        (
            "gemini",
            "stream_generate",
            {"model": "gpt-test", "contents": []},
            [{"candidates": [{"content": {"role": "model", "parts": [{"text": "private", "thought": True, "thoughtSignature": "gem-signature"}]}}]}, "[DONE]"],
            ("metadata", "thoughtSignatures"),
            ["gem-signature"],
            "gem-signature",
        ),
        (
            "anthropic_messages",
            "messages",
            {"model": "gpt-test", "messages": [], "max_tokens": 1},
            [{"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "claude-signature"}}, {"type": "message_stop"}],
            ("metadata", "thinking_signatures"),
            ["claude-signature"],
            "claude-signature",
        ),
        (
            "responses",
            "responses",
            {"model": "gpt-test", "input": []},
            [{"type": "response.completed", "response": {"id": "resp-continuation", "status": "completed", "model": "gpt-test", "output": []}}, "[DONE]"],
            ("previous_response_id",),
            "resp-continuation",
            "resp-continuation",
        ),
    ],
)
async def test_provider_stream_state_is_cached_for_followups_but_not_exposed(
    protocol,
    operation,
    request_payload,
    chunks,
    injected_path,
    expected,
    secret,
) -> None:
    rules_by_protocol = {
        "gemini": (
            FieldCacheRule(
                name="gemini_thought_signature",
                source="stream_event",
                path="raw.candidates.0.content.parts.0.thoughtSignature",
                mode="all",
                inject=FieldCacheInjection(target="request", path="metadata.thoughtSignatures", as_list=True),
                allow_missing_session=True,
                scope=("provider", "model", "credential", "session"),
                metadata={"provider_continuation": True},
            ),
        ),
        "anthropic_messages": (
            FieldCacheRule(
                name="anthropic_thinking_signature",
                source="stream_event",
                path="raw.delta.signature",
                mode="all",
                inject=FieldCacheInjection(target="request", path="metadata.thinking_signatures", as_list=True),
                allow_missing_session=True,
                scope=("provider", "model", "credential", "session"),
                metadata={"provider_continuation": True},
            ),
        ),
        "responses": (
            FieldCacheRule(
                name="responses_continuation",
                source="stream_event",
                path="raw.response.id",
                mode="last",
                inject=FieldCacheInjection(target="request", path="previous_response_id"),
                allow_missing_session=True,
                scope=("provider", "model", "credential", "session"),
                metadata={"provider_continuation": True},
            ),
        ),
    }
    context = NativeProviderContext(
        provider="synthetic",
        model="gpt-test",
        protocol_name=protocol,
        input_protocol_name=protocol,
        client_protocol_name=protocol,
        endpoint="https://example.test/stream",
        operation=operation,
        credential_id="credential-1",
        session_id="session-1",
        scope_key="scope-1",
        field_cache_rules=rules_by_protocol[protocol],
    )
    executor = NativeProviderExecutor()

    first_events = [
        event
        async for event in executor.stream(
            request_payload,
            context,
            NativeHTTPTransport(FakeStreamingClient(chunks)),
        )
    ]
    second_client = FakeStreamingClient(["[DONE]"])
    _ = [
        event
        async for event in executor.stream(
            request_payload,
            context,
            NativeHTTPTransport(second_client),
        )
    ]

    current = second_client.calls[0]["json"]
    for key in injected_path:
        current = current[key]
    assert current == expected

    # The client never sees cached provider state: neutral events are internal,
    # and the pipeline's client formatting emits content blocks only.
    from rotator_library.client.stream_ops import NeutralStreamPipeline
    from rotator_library.protocols.types import ProtocolContext

    async def event_source():
        for event in first_events:
            yield event

    pipeline = NeutralStreamPipeline(
        client_protocol_name=protocol,
        protocol_context=ProtocolContext(
            provider="synthetic",
            model="gpt-test",
            source_protocol=protocol,
            target_protocol=protocol,
            input_protocol=protocol,
            client_protocol=protocol,
        ),
        model="gpt-test",
    )
    output_text = "".join([frame async for frame in pipeline.run(event_source())])
    assert secret not in output_text
