from __future__ import annotations

import pytest

from rotator_library.client.stream_ops import (
    ChatWireStreamAdapter,
    NeutralStreamPipeline,
)
from rotator_library.protocols.types import ProtocolContext
from rotator_library.transaction_logger import TransactionLogger
from rotator_library.utils import zstd_io


@pytest.fixture(autouse=True)
def _trace_level_2(monkeypatch):
    """Trace mechanics live at L2 (D15 tiers)."""
    monkeypatch.setenv("TRANSACTION_LOG_LEVEL", "2")


def _trace_entries(logger):
    from tests.txn_helpers import changes

    return changes(logger)


class FakeCredentialContext:
    def __init__(self) -> None:
        self.success_kwargs = None

    def mark_success(self, **kwargs) -> None:
        self.success_kwargs = kwargs


def _protocol_context(model: str) -> ProtocolContext:
    provider = model.split("/", 1)[0] if "/" in model else "openai"
    return ProtocolContext(
        provider=provider,
        model=model,
        source_protocol="openai_chat",
        target_protocol="openai_chat",
        input_protocol="openai_chat",
        provider_protocol="openai_chat",
        client_protocol="openai_chat",
        transport="sse",
    )


def _pipeline(model: str, **kwargs) -> NeutralStreamPipeline:
    return NeutralStreamPipeline(
        client_protocol_name="openai_chat",
        protocol_context=_protocol_context(model),
        model=model,
        **kwargs,
    )


async def _run(chunks_fn, model: str, **kwargs):
    pipeline = _pipeline(model, **kwargs)
    adapter = ChatWireStreamAdapter(model, repair_state=pipeline.repair_state)
    return [frame async for frame in pipeline.run(adapter.events(chunks_fn(), pipeline.usage))]


async def _usage_chunks():
    yield {"id": "chunk_1", "choices": [{"delta": {"content": "hi"}}]}
    yield {
        "id": "chunk_2",
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 40, "cache_creation_tokens": 5},
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
    }


async def _zero_usage_chunks():
    yield {"id": "chunk_1", "choices": [{"delta": {"content": "hi"}}]}
    yield {"id": "chunk_2", "choices": [{"delta": {}, "finish_reason": "stop"}]}


async def _cost_comment_chunks():
    yield ': cost {"total_cost":0.042,"currency":"USD","source":"provider_sse"}\n\n'
    yield {"id": "chunk_1", "choices": [{"delta": {"content": "hi"}}]}
    yield {"id": "chunk_2", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


async def _cost_event_chunks():
    yield 'event: cost\ndata: {"total_cost":0.021,"currency":"EUR","source":"event_cost"}\n\n'
    yield {"id": "chunk_1", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


async def _scalar_cost_event_chunks():
    yield "event: cost\ndata: 0.033\n\n"
    yield {"id": "chunk_1", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


async def _request_cost_comment_chunks():
    yield ': cost {"request_cost_usd":0.044,"source":"reference_sse"}\n\n'
    yield {"id": "chunk_1", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


async def _estimated_cost_comment_chunks():
    yield ': cost {"estimated_cost":0.045,"source":"reference_estimate"}\n\n'
    yield {"id": "chunk_1", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


async def _top_level_cost_usage_sse_chunks():
    yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1},"total_cost":0.055}\n\n'


async def _top_level_cost_usage_dict_chunks():
    yield {"id": "chunk_1", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "total_cost": 0.066}


async def _cost_comment_overridden_by_final_usage_chunks():
    yield ': cost 0.042\n\n'
    yield {
        "id": "chunk_2",
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost_details": {"total_cost": 0.084, "source": "final_usage"}},
    }


@pytest.mark.asyncio
async def test_streaming_usage_uses_normalized_accounting_and_trace(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "rotator_library.usage.costs.litellm.get_model_info",
        lambda model: {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
    )
    cred_context = FakeCredentialContext()
    logger = TransactionLogger("openai", "gpt-test", parent_dir=tmp_path)

    chunks = await _run(_usage_chunks, "gpt-test", cred_context=cred_context, transaction_logger=logger)

    assert chunks[-1] == "data: [DONE]\n\n"
    assert cred_context.success_kwargs["prompt_tokens"] == 55
    assert cred_context.success_kwargs["prompt_tokens_cache_read"] == 40
    assert cred_context.success_kwargs["prompt_tokens_cache_write"] == 5
    assert cred_context.success_kwargs["completion_tokens"] == 20
    assert cred_context.success_kwargs["thinking_tokens"] == 10
    assert cred_context.success_kwargs["approx_cost"] > 0
    entries = _trace_entries(logger)
    assert any(entry["pass_name"] == "usage_accounting_summary" for entry in entries)


@pytest.mark.asyncio
async def test_streaming_usage_skip_cost_returns_zero() -> None:
    cred_context = FakeCredentialContext()

    _ = await _run(_usage_chunks, "gpt-test", cred_context=cred_context, skip_cost_calculation=True)

    assert cred_context.success_kwargs["approx_cost"] == 0.0


@pytest.mark.asyncio
async def test_streaming_without_usage_still_marks_success_with_zero_usage() -> None:
    cred_context = FakeCredentialContext()

    _ = await _run(_zero_usage_chunks, "gpt-test", cred_context=cred_context)

    assert cred_context.success_kwargs["prompt_tokens"] == 0
    assert cred_context.success_kwargs["completion_tokens"] == 0
    assert cred_context.success_kwargs["thinking_tokens"] == 0
    assert cred_context.success_kwargs["prompt_tokens_cache_read"] == 0
    assert cred_context.success_kwargs["prompt_tokens_cache_write"] == 0


@pytest.mark.asyncio
async def test_streaming_completed_calls_success_callback() -> None:
    called = []

    _ = await _run(_zero_usage_chunks, "gpt-test", success_callback=lambda: called.append(True))

    assert called == [True]


@pytest.mark.asyncio
async def test_streaming_usage_uses_configured_env_pricing(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PRICE_OPENAI_GPT_TEST_INPUT", "2.0")
    cred_context = FakeCredentialContext()

    _ = await _run(_usage_chunks, "openai/gpt-test", cred_context=cred_context)

    assert cred_context.success_kwargs["approx_cost"] == 110.0


@pytest.mark.asyncio
async def test_streaming_cost_comment_updates_approx_cost() -> None:
    cred_context = FakeCredentialContext()

    _ = await _run(_cost_comment_chunks, "gpt-test", cred_context=cred_context)

    assert cred_context.success_kwargs["approx_cost"] == 0.042


@pytest.mark.asyncio
async def test_streaming_cost_event_updates_approx_cost() -> None:
    cred_context = FakeCredentialContext()

    _ = await _run(_cost_event_chunks, "gpt-test", cred_context=cred_context)

    assert cred_context.success_kwargs["approx_cost"] == 0.021


@pytest.mark.asyncio
async def test_streaming_scalar_cost_event_updates_approx_cost() -> None:
    cred_context = FakeCredentialContext()

    _ = await _run(_scalar_cost_event_chunks, "gpt-test", cred_context=cred_context)

    assert cred_context.success_kwargs["approx_cost"] == 0.033


@pytest.mark.asyncio
async def test_streaming_reference_request_cost_comment_updates_approx_cost() -> None:
    cred_context = FakeCredentialContext()

    _ = await _run(_request_cost_comment_chunks, "gpt-test", cred_context=cred_context)

    assert cred_context.success_kwargs["approx_cost"] == 0.044


@pytest.mark.asyncio
async def test_streaming_estimated_cost_comment_updates_approx_cost() -> None:
    cred_context = FakeCredentialContext()

    _ = await _run(_estimated_cost_comment_chunks, "gpt-test", cred_context=cred_context)

    assert cred_context.success_kwargs["approx_cost"] == 0.045


@pytest.mark.asyncio
async def test_streaming_sse_usage_preserves_top_level_cost() -> None:
    cred_context = FakeCredentialContext()

    _ = await _run(_top_level_cost_usage_sse_chunks, "gpt-test", cred_context=cred_context)

    assert cred_context.success_kwargs["approx_cost"] == 0.055


@pytest.mark.asyncio
async def test_streaming_dict_usage_preserves_top_level_cost() -> None:
    cred_context = FakeCredentialContext()

    _ = await _run(_top_level_cost_usage_dict_chunks, "gpt-test", cred_context=cred_context)

    assert cred_context.success_kwargs["approx_cost"] == 0.066


@pytest.mark.asyncio
async def test_streaming_final_usage_cost_overrides_comment_cost() -> None:
    cred_context = FakeCredentialContext()

    _ = await _run(_cost_comment_overridden_by_final_usage_chunks, "gpt-test", cred_context=cred_context)

    assert cred_context.success_kwargs["approx_cost"] == 0.084
