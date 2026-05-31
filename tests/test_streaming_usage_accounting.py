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
    assert cred_context.success_kwargs["prompt_tokens"] == 60
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
