from __future__ import annotations

import json

import pytest

from rotator_library.responses import InMemoryResponsesStore, ResponsesService
from rotator_library.transaction_logger import TransactionLogger


class FakeUsageClient:
    async def acompletion(self, **kwargs):
        return {
            "id": "chatcmpl_1",
            "model": "gpt-test",
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        }


@pytest.mark.asyncio
async def test_responses_create_traces_normalized_usage(tmp_path) -> None:
    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    service = ResponsesService(store=InMemoryResponsesStore())

    await service.create_response({"model": "gpt-test", "input": "hello"}, FakeUsageClient(), transaction_logger=logger)

    entries = [json.loads(line) for line in (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    usage_entries = [entry for entry in entries if entry["pass_name"] == "usage_accounting_summary"]
    assert usage_entries
    assert usage_entries[-1]["data"]["usage"]["completion_tokens"] == 5
    assert usage_entries[-1]["data"]["usage"]["reasoning_tokens"] == 3
