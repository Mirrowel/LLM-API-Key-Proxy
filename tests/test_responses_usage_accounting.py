from __future__ import annotations

from pathlib import Path

import json

import pytest

from rotator_library.responses import InMemoryResponsesStore, ResponsesService
from rotator_library.transaction_logger import TransactionLogger




class FakeUsageClient:
    async def agenerate(self, payload, *, input_protocol, request=None, **kwargs):
        return {
            "id": "resp_usage_1",
            "object": "response",
            "model": payload.get("model", "gpt-test"),
            "status": "completed",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 8,
                "output_tokens_details": {"reasoning_tokens": 3},
            },
        }


def _trace_entries(logger):
    from tests.txn_helpers import changes

    return changes(logger)

class FakeStreamingUsageClient:
    async def agenerate(self, payload, *, input_protocol, request=None, **kwargs):
        async def gen():
            yield 'event: response.completed\ndata: {"type":"response.completed","sequence_number":0,"response":{"id":"resp_usage_stream","object":"response","status":"completed","model":"gpt-test","output":[],"usage":{"input_tokens":20,"output_tokens":8,"total_tokens":28,"output_tokens_details":{"reasoning_tokens":3}}}}\n\n'

        return gen()


@pytest.mark.asyncio
async def test_responses_create_traces_normalized_usage(tmp_path) -> None:
    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    service = ResponsesService(store=InMemoryResponsesStore())

    await service.create_response({"model": "gpt-test", "input": "hello"}, FakeUsageClient(), transaction_logger=logger)

    entries = _trace_entries(logger)
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
    # Completed events nest the response object under "response".
    response_obj = payload.get("response", payload)

    assert response_obj["usage"]["output_tokens_details"] == {"reasoning_tokens": 3}
