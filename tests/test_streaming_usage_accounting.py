from __future__ import annotations

import json

import pytest

from rotator_library.client.streaming import StreamingHandler
from rotator_library.transaction_logger import TransactionLogger


class FakeCredentialContext:
    def __init__(self) -> None:
        self.success_kwargs = None

    def mark_success(self, **kwargs) -> None:
        self.success_kwargs = kwargs


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

    chunks = [chunk async for chunk in StreamingHandler().wrap_stream(_usage_chunks(), "cred", "gpt-test", cred_context=cred_context, transaction_logger=logger)]

    assert chunks[-1] == "data: [DONE]\n\n"
    assert cred_context.success_kwargs["prompt_tokens"] == 55
    assert cred_context.success_kwargs["prompt_tokens_cache_read"] == 40
    assert cred_context.success_kwargs["prompt_tokens_cache_write"] == 5
    assert cred_context.success_kwargs["completion_tokens"] == 20
    assert cred_context.success_kwargs["thinking_tokens"] == 10
    assert cred_context.success_kwargs["approx_cost"] > 0
    entries = [json.loads(line) for line in (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(entry["pass_name"] == "usage_accounting_summary" for entry in entries)


@pytest.mark.asyncio
async def test_streaming_usage_skip_cost_returns_zero() -> None:
    cred_context = FakeCredentialContext()

    _ = [chunk async for chunk in StreamingHandler().wrap_stream(_usage_chunks(), "cred", "gpt-test", cred_context=cred_context, skip_cost_calculation=True)]

    assert cred_context.success_kwargs["approx_cost"] == 0.0


@pytest.mark.asyncio
async def test_streaming_without_usage_still_marks_success_with_zero_usage() -> None:
    cred_context = FakeCredentialContext()

    _ = [chunk async for chunk in StreamingHandler().wrap_stream(_zero_usage_chunks(), "cred", "gpt-test", cred_context=cred_context)]

    assert cred_context.success_kwargs["prompt_tokens"] == 0
    assert cred_context.success_kwargs["completion_tokens"] == 0
    assert cred_context.success_kwargs["thinking_tokens"] == 0
    assert cred_context.success_kwargs["prompt_tokens_cache_read"] == 0
    assert cred_context.success_kwargs["prompt_tokens_cache_write"] == 0


@pytest.mark.asyncio
async def test_streaming_completed_calls_success_callback() -> None:
    called = []

    _ = [chunk async for chunk in StreamingHandler().wrap_stream(_zero_usage_chunks(), "cred", "gpt-test", success_callback=lambda: called.append(True))]

    assert called == [True]


@pytest.mark.asyncio
async def test_streaming_usage_uses_configured_env_pricing(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PRICE_OPENAI_GPT_TEST_INPUT", "2.0")
    cred_context = FakeCredentialContext()

    _ = [chunk async for chunk in StreamingHandler().wrap_stream(_usage_chunks(), "cred", "openai/gpt-test", cred_context=cred_context)]

    assert cred_context.success_kwargs["approx_cost"] == 110.0


@pytest.mark.asyncio
async def test_streaming_cost_comment_updates_approx_cost() -> None:
    cred_context = FakeCredentialContext()

    chunks = [chunk async for chunk in StreamingHandler().wrap_stream(_cost_comment_chunks(), "cred", "gpt-test", cred_context=cred_context)]

    assert chunks[0].startswith(": cost")
    assert cred_context.success_kwargs["approx_cost"] == 0.042


@pytest.mark.asyncio
async def test_streaming_cost_event_updates_approx_cost() -> None:
    cred_context = FakeCredentialContext()

    _ = [chunk async for chunk in StreamingHandler().wrap_stream(_cost_event_chunks(), "cred", "gpt-test", cred_context=cred_context)]

    assert cred_context.success_kwargs["approx_cost"] == 0.021


@pytest.mark.asyncio
async def test_streaming_final_usage_cost_overrides_comment_cost() -> None:
    cred_context = FakeCredentialContext()

    _ = [chunk async for chunk in StreamingHandler().wrap_stream(_cost_comment_overridden_by_final_usage_chunks(), "cred", "gpt-test", cred_context=cred_context)]

    assert cred_context.success_kwargs["approx_cost"] == 0.084
