from __future__ import annotations

import json

import pytest

from rotator_library.field_cache import FieldCacheInjection, FieldCacheRule
from rotator_library.native_provider import NativeHTTPTransport, NativeProviderContext, NativeProviderExecutor
from rotator_library.protocols import ProtocolError
from rotator_library.transaction_logger import TransactionLogger


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHTTPClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post(self, endpoint, *, headers, json):
        self.calls.append({"endpoint": endpoint, "headers": headers, "json": json})
        return FakeHTTPResponse(self.response)


def _trace_entries(log_dir):
    return [json.loads(line) for line in (log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_native_provider_executor_runs_protocol_adapter_cache_and_trace(tmp_path) -> None:
    logger = TransactionLogger("native", "gpt-test", parent_dir=tmp_path)
    rule = FieldCacheRule(
        name="reasoning",
        source="response",
        path="choices.0.message.reasoning_content",
        inject=FieldCacheInjection(target="request", path="metadata.reasoning_content"),
        allow_missing_session=True,
    )
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
        headers={"authorization": "Bearer test"},
        adapter_names=("model_override",),
        adapter_config={"model_override": {"model": "provider/gpt-test"}},
        field_cache_rules=(rule,),
        transaction_logger=logger,
    )
    response = {
        "id": "chat_1",
        "model": "provider/gpt-test",
        "choices": [{"message": {"role": "assistant", "content": "ok", "reasoning_content": "hidden"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_cost": 0.01},
    }
    client = FakeHTTPClient(response)

    result = await NativeProviderExecutor().execute(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        context,
        NativeHTTPTransport(client),
    )

    assert result["id"] == "chat_1"
    assert client.calls[0]["json"]["model"] == "provider/gpt-test"
    trace_text = (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8")
    assert "hidden" not in trace_text
    pass_names = [entry["pass_name"] for entry in _trace_entries(logger.log_dir)]
    assert "native_protocol_selected" in pass_names
    assert "raw_native_client_request" in pass_names
    assert "parsed_native_unified_request" in pass_names
    assert "built_native_provider_request" in pass_names
    assert "after_request_adapter_chain" in pass_names
    assert "field_cache_injection_start" in pass_names
    assert "after_field_cache_injection" in pass_names
    assert "field_cache_injection_complete" in pass_names
    assert "native_provider_request" in pass_names
    assert "raw_native_provider_response" in pass_names
    assert "parsed_native_unified_response" in pass_names
    assert "formatted_native_response" in pass_names
    assert "after_response_adapter_chain" in pass_names
    assert "field_cache_extraction_start" in pass_names
    assert "after_field_cache_extraction" in pass_names
    assert "field_cache_extraction_complete" in pass_names
    usage_entries = [entry for entry in _trace_entries(logger.log_dir) if entry["pass_name"] == "usage_accounting_summary"]
    assert usage_entries[-1]["data"]["cost"]["provider_reported_cost"] == 0.01
    assert "final_client_response" in pass_names


@pytest.mark.asyncio
async def test_native_provider_default_field_cache_persists_across_requests() -> None:
    rule = FieldCacheRule(
        name="reasoning",
        source="response",
        path="choices.0.message.reasoning_content",
        inject=FieldCacheInjection(target="request", path="metadata.reasoning_content"),
        allow_missing_session=True,
    )
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
        field_cache_rules=(rule,),
    )
    first_client = FakeHTTPClient(
        {
            "id": "chat_1",
            "model": "gpt-test",
            "choices": [{"message": {"role": "assistant", "content": "ok", "reasoning_content": "cached-state"}}],
        }
    )
    second_client = FakeHTTPClient(
        {
            "id": "chat_2",
            "model": "gpt-test",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        }
    )
    executor = NativeProviderExecutor()

    await executor.execute({"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}, context, NativeHTTPTransport(first_client))
    await executor.execute({"model": "gpt-test", "messages": [{"role": "user", "content": "again"}]}, context, NativeHTTPTransport(second_client))

    assert second_client.calls[0]["json"]["metadata"]["reasoning_content"] == "cached-state"


@pytest.mark.asyncio
async def test_native_provider_trace_redacts_configured_injection_paths(tmp_path) -> None:
    logger = TransactionLogger("native", "gpt-test", parent_dir=tmp_path)
    rule = FieldCacheRule(
        name="state",
        source="response",
        path="choices.0.message.reasoning_content",
        inject=FieldCacheInjection(target="request", path="metadata.state"),
        allow_missing_session=True,
    )
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
        field_cache_rules=(rule,),
        transaction_logger=logger,
    )
    executor = NativeProviderExecutor()

    await executor.execute(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        context,
        NativeHTTPTransport(FakeHTTPClient({"id": "chat_1", "choices": [{"message": {"role": "assistant", "content": "ok", "reasoning_content": "opaque-state"}}]})),
    )
    await executor.execute(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "again"}]},
        context,
        NativeHTTPTransport(FakeHTTPClient({"id": "chat_2", "choices": [{"message": {"role": "assistant", "content": "ok"}}]})),
    )

    trace_text = (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8")
    assert "opaque-state" not in trace_text
    assert '"state": "[REDACTED]"' in trace_text


@pytest.mark.asyncio
async def test_native_provider_executor_logs_transform_errors(tmp_path) -> None:
    logger = TransactionLogger("native", "gpt-test", parent_dir=tmp_path)
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
        adapter_names=("missing_adapter",),
        transaction_logger=logger,
    )

    with pytest.raises(KeyError):
        await NativeProviderExecutor().execute({"model": "gpt-test", "messages": []}, context, NativeHTTPTransport(FakeHTTPClient({})))

    pass_names = [entry["pass_name"] for entry in _trace_entries(logger.log_dir)]
    assert "transform_log_error" in pass_names


@pytest.mark.asyncio
async def test_native_provider_executor_rejects_unsupported_operation_before_transport() -> None:
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        operation="embeddings",
        endpoint="https://example.test/chat",
    )
    client = FakeHTTPClient({"id": "should_not_call"})

    with pytest.raises(ProtocolError, match="unsupported operation"):
        await NativeProviderExecutor().execute(
            {"model": "gpt-test", "messages": []},
            context,
            NativeHTTPTransport(client),
        )

    assert client.calls == []


@pytest.mark.asyncio
async def test_native_provider_stream_rejects_unsupported_operation_before_transport() -> None:
    context = NativeProviderContext(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        operation="embeddings",
        endpoint="https://example.test/chat",
    )
    client = FakeHTTPClient({"id": "should_not_call"})

    with pytest.raises(ProtocolError, match="unsupported operation"):
        async for _ in NativeProviderExecutor().stream(
            {"model": "gpt-test", "messages": []},
            context,
            NativeHTTPTransport(client),
        ):
            pass

    assert client.calls == []
