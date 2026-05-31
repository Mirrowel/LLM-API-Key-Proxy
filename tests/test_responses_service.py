from __future__ import annotations

import json

import pytest

import rotator_library.responses.service as responses_service_module
from rotator_library.responses import InMemoryResponsesStore, ResponsesService, ResponsesServiceError, ResponsesStoreSettings, StoredResponse
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


class FakeInternalClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self._request_builder = object()
        self._executor = object()

    async def acompletion(self, **kwargs):
        callback = kwargs.pop("_request_context_callback", None)
        hints = kwargs.pop("_session_tracking_hints", None)
        if callback:
            callback(
                type(
                    "Context",
                    (),
                    {
                        "session_id": "session-parent",
                        "session_affinity_key": "affinity-parent",
                        "usage_manager_key": "scope-parent",
                        "classifier": "global",
                    },
                )()
            )
        self.internal_hints = hints
        return await super().acompletion(**kwargs)


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
async def test_create_response_applies_storage_ttl() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store, store_settings=ResponsesStoreSettings(ttl_seconds=60))

    response = await service.create_response({"model": "gpt-test", "input": "Hello"}, FakeClient())
    stored = await store.get(response["id"])

    assert stored is not None
    assert stored.expires_at is not None
    assert stored.expires_at > stored.created_at
    assert stored.metadata["response_id"] == response["id"]


@pytest.mark.asyncio
async def test_service_default_store_honors_max_items() -> None:
    class SequencedClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.index = 0

        async def acompletion(self, **kwargs):
            self.index += 1
            response = await super().acompletion(**kwargs)
            response["id"] = f"chat_response_{self.index}"
            return response

    service = ResponsesService(store_settings=ResponsesStoreSettings(max_items=1))
    client = SequencedClient()

    first = await service.create_response({"model": "gpt-test", "input": "one"}, client)
    second = await service.create_response({"model": "gpt-test", "input": "two"}, client)

    assert await service.store.get(first["id"]) is None
    assert await service.store.get(second["id"]) is not None


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
async def test_internal_session_hints_do_not_leak_to_direct_clients_or_traces(tmp_path) -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)
    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    await store.save(
        StoredResponse(
            id="resp_parent",
            model="gpt-test",
            status="completed",
            response={"id": "resp_parent", "object": "response", "output": []},
            metadata={"session_affinity_key": "affinity-parent"},
        )
    )
    client = FakeClient()

    await service.create_response({"model": "gpt-test", "input": "Continue", "previous_response_id": "resp_parent"}, client, transaction_logger=logger)

    assert "_session_tracking_hints" not in client.calls[0]
    trace_text = (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8")
    assert "_session_tracking_hints" not in trace_text
    assert "has_session_hints" in trace_text


@pytest.mark.asyncio
async def test_internal_client_context_metadata_is_stored_with_response() -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)
    await store.save(
        StoredResponse(
            id="resp_parent",
            model="gpt-test",
            status="completed",
            response={"id": "resp_parent", "object": "response", "output": []},
            metadata={"session_affinity_key": "affinity-parent"},
        )
    )
    client = FakeInternalClient()

    response = await service.create_response({"model": "gpt-test", "input": "Continue", "previous_response_id": "resp_parent"}, client)
    stored = await store.get(response["id"])

    assert client.internal_hints["affinity_key"] == "affinity-parent"
    assert stored is not None
    assert stored.session_id == "session-parent"
    assert stored.metadata["session_affinity_key"] == "affinity-parent"


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
        "responses_raw_request",
        "responses_parsed_request",
        "responses_bridge_chat_request",
        "responses_bridge_chat_response",
        "responses_parsed_response",
        "usage_accounting_summary",
        "responses_stored_response",
        "responses_final_response",
    ]


def test_trace_responses_usage_returns_before_conversion_without_logger(monkeypatch) -> None:
    service = ResponsesService(store=InMemoryResponsesStore())

    def fail_extract(*args, **kwargs):
        raise AssertionError("usage conversion should be skipped when tracing is disabled")

    monkeypatch.setattr(responses_service_module, "extract_usage_record", fail_extract)

    service._trace_responses_usage(None, {"usage": {"input_tokens": 1}}, "gpt-test", source="test")


@pytest.mark.asyncio
async def test_previous_response_trace_payload_skipped_without_logger() -> None:
    class Parent:
        id = "resp_parent"
        response = {"output": []}
        output_items = []
        input_items = []

        def to_dict(self):
            raise AssertionError("previous response trace payload should not be built without a logger")

    class Store(InMemoryResponsesStore):
        async def get(self, response_id):
            return Parent()

    service = ResponsesService(store=Store())

    parent = await service._load_previous_response("resp_parent", None)

    assert parent.id == "resp_parent"
