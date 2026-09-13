from __future__ import annotations

from pathlib import Path

import json

import pytest

from rotator_library.native_provider import NativeHTTPTransport, NativeProviderContext, NativeProviderExecutor
from rotator_library.transaction_logger import TransactionLogger
from tests.txn_helpers import by_pass


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

    summary = by_pass(logger, "usage_accounting_summary")[-1]
    assert summary["stage"] == "final"
    assert summary["detail"] == "usage_accounting_summary/metadata/final"
    # Provider usage (including reasoning details) is captured on the final
    # client response; the accounting pass derives the normalized completion.
    final_usage = by_pass(logger, "final_client_response")[-1]["data"]["usage"]
    assert final_usage["completion_tokens"] == 6
    assert final_usage["completion_tokens_details"]["reasoning_tokens"] == 2
