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


class FakeStreamingUsageClient:
    async def acompletion(self, **kwargs):
        async def gen():
            yield 'data: {"id":"chunk_1","choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield 'data: {"id":"chunk_2","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":8,"total_tokens":28,"completion_tokens_details":{"reasoning_tokens":3}}}\n\n'

        return gen()


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


@pytest.mark.asyncio
async def test_responses_stream_preserves_usage_details_in_completed_event() -> None:
    service = ResponsesService(store=InMemoryResponsesStore())

    chunks = [chunk async for chunk in service.stream_response({"model": "gpt-test", "input": "hello", "stream": True}, FakeStreamingUsageClient())]
    completed = [chunk for chunk in chunks if "response.completed" in chunk][0]
    payload = json.loads(completed.split("data: ", 1)[1])

    assert payload["usage"]["output_tokens_details"] == {"reasoning_tokens": 3}
