from __future__ import annotations

import json

import pytest

from rotator_library.responses import InMemoryResponsesStore, ResponsesService, ResponsesServiceError, StoredResponse
from rotator_library.transaction_logger import TransactionLogger


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    async def acompletion(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "id": "chat_response_1",
            "model": kwargs["model"],
            "choices": [{"message": {"role": "assistant", "content": "Hello back"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }


def _trace_entries(log_dir):
    return [json.loads(line) for line in (log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_create_response_stores_non_streaming_response() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)
    client = FakeClient()

    response = await service.create_response({"model": "gpt-test", "input": "Hello"}, client)

    assert response["id"] == "chat_response_1"
    assert response["output"][0]["content"][0]["text"] == "Hello back"
    assert (await store.get("chat_response_1")) is not None
    assert client.calls[0]["messages"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.asyncio
async def test_store_false_does_not_persist_response() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)

    response = await service.create_response({"model": "gpt-test", "input": "Hello", "store": False}, FakeClient())

    assert await store.get(response["id"]) is None


@pytest.mark.asyncio
async def test_previous_response_id_loads_parent_context() -> None:
    store = InMemoryResponsesStore()
    await store.save(
        StoredResponse(
            id="resp_parent",
            model="gpt-test",
            status="completed",
            request={"input": "Earlier"},
            response={
                "id": "resp_parent",
                "object": "response",
                "model": "gpt-test",
                "status": "completed",
                "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Earlier"}]}],
            },
            input_items=["Earlier"],
            output_items=[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Earlier"}]}],
        )
    )
    client = FakeClient()
    service = ResponsesService(store=store)

    await service.create_response({"model": "gpt-test", "input": "Continue", "previous_response_id": "resp_parent"}, client)

    assert client.calls[0]["messages"] == [
        {"role": "assistant", "content": "Earlier"},
        {"role": "user", "content": "Continue"},
    ]


@pytest.mark.asyncio
async def test_missing_previous_response_id_raises_not_found() -> None:
    service = ResponsesService(store=InMemoryResponsesStore())

    with pytest.raises(ResponsesServiceError) as exc_info:
        await service.create_response({"model": "gpt-test", "input": "Continue", "previous_response_id": "missing"}, FakeClient())

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_delete_and_list_input_items() -> None:
    service = ResponsesService(store=InMemoryResponsesStore())
    response = await service.create_response({"model": "gpt-test", "input": ["Hello"]}, FakeClient())

    assert (await service.get_response(response["id"]))["id"] == response["id"]
    assert await service.list_input_items(response["id"]) == {"object": "list", "data": ["Hello"]}
    assert await service.delete_response(response["id"]) == {"id": response["id"], "object": "response.deleted", "deleted": True}
    with pytest.raises(ResponsesServiceError):
        await service.get_response(response["id"])


@pytest.mark.asyncio
async def test_service_emits_transform_trace_passes(tmp_path) -> None:
    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    service = ResponsesService(store=InMemoryResponsesStore())

    await service.create_response({"model": "gpt-test", "input": "Hello"}, FakeClient(), transaction_logger=logger)

    pass_names = [entry["pass_name"] for entry in _trace_entries(logger.log_dir)]
    assert pass_names == [
        "raw_responses_request",
        "parsed_unified_request",
        "responses_bridge_chat_request",
        "raw_chat_bridge_response",
        "parsed_unified_response",
        "usage_accounting_summary",
        "stored_responses_response",
        "final_responses_response",
    ]
