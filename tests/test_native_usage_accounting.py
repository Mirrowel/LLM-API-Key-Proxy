from __future__ import annotations

import json

import pytest

from rotator_library.native_provider import NativeHTTPTransport, NativeProviderContext, NativeProviderExecutor
from rotator_library.transaction_logger import TransactionLogger


class FakeNativeTransport(NativeHTTPTransport):
    def __init__(self):
        pass

    async def post_json(self, endpoint, *, headers, payload):
        return {
            "id": "chatcmpl_1",
            "model": "gpt-test",
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "completion_tokens_details": {"reasoning_tokens": 2}},
        }


@pytest.mark.asyncio
async def test_native_executor_traces_normalized_usage(tmp_path) -> None:
    logger = TransactionLogger("openai", "gpt-test", parent_dir=tmp_path)
    context = NativeProviderContext(
        provider="openai",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
        headers={},
        transaction_logger=logger,
    )

    await NativeProviderExecutor().execute({"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}, context, FakeNativeTransport())

    entries = [json.loads(line) for line in (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    usage_entries = [entry for entry in entries if entry["pass_name"] == "usage_accounting_summary"]
    assert usage_entries[-1]["data"]["usage"]["completion_tokens"] == 4
    assert usage_entries[-1]["data"]["usage"]["reasoning_tokens"] == 2
