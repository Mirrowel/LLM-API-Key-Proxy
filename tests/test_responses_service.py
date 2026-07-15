from __future__ import annotations

import json

import pytest

import rotator_library.responses.service as responses_service_module
from rotator_library.responses import InMemoryResponsesStore, ResponsesService, ResponsesServiceError, ResponsesStoreSettings, StoredResponse, create_configured_responses_store
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


class FakeCostClient(FakeClient):
    async def acompletion(self, **kwargs):
        response = await super().acompletion(**kwargs)
        response["usage"]["cost_details"] = {"total_cost": 0.033, "source": "responses_provider"}
        return response


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
                        "session_tracker": None,
                        "provider": "openai",
                        "model": "gpt-test",
                        "session_tracking_namespace": "namespace",
                    },
                )()
            )
        self.internal_hints = hints
        return await super().acompletion(**kwargs)


def _trace_entries(log_dir):
    return [json.loads(line) for line in (log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]


def test_responses_service_owns_request_scope_derivation() -> None:
    service = ResponsesService()

    assert service.request_scope_key({"model": "gpt-test"}) == "public"
    assert service.request_scope_key(
        {
            "model": "gpt-test",
            "api_keys": {"openai": ["private-key"]},
            "private": True,
        }
    ).startswith("bundle:")


def test_responses_service_recursively_redacts_transport_logging_payload() -> None:
    service = ResponsesService()
    request = {
        "model": "gpt-test",
        "api_keys": {"openai": ["top-level-secret"]},
        "metadata": {
            "providers": {"private": {"api_key": "nested-secret"}},
            "keep": "visible",
        },
        "input": [{"authorization": "Bearer secret", "value": "visible"}],
    }

    redacted = service.redact_request_for_logging(request)

    assert redacted == {
        "model": "gpt-test",
        "metadata": {"keep": "visible"},
        "input": [{"value": "visible"}],
    }
    assert request["api_keys"]["openai"] == ["top-level-secret"]
    assert request["metadata"]["providers"]["private"]["api_key"] == "nested-secret"


@pytest.mark.asyncio
async def test_scoped_response_requires_unforgeable_access_capability() -> None:
    service = ResponsesService(store=InMemoryResponsesStore())
    request = {
        "model": "gpt-test",
        "input": "private",
        "api_keys": {"openai": ["private-key"]},
        "private": True,
    }
    request_scope = service.prepare_request_scope(request)
    response = await service.create_response(
        request,
        FakeClient(),
        request_scope=request_scope,
    )
    raw_domain = request_scope.access_token.rsplit(".", 1)[0]

    loaded = await service.get_response_with_access_token(
        response["id"],
        request_scope.access_token,
    )

    assert loaded["id"] == response["id"]
    with pytest.raises(ResponsesServiceError) as raw_error:
        await service.get_response_with_access_token(response["id"], raw_domain)
    assert raw_error.value.status_code == 404
    with pytest.raises(ResponsesServiceError) as forged_error:
        await service.get_response_with_access_token(
            response["id"],
            request_scope.access_token[:-1] + "x",
        )
    assert forged_error.value.status_code == 404


@pytest.mark.asyncio
async def test_scoped_access_capability_survives_durable_store_restart(tmp_path) -> None:
    env = {
        "RESPONSES_STORE_BACKEND": "provider_cache",
        "RESPONSES_STORE_CACHE_NAME": "responses_capability_test",
        "RESPONSES_STORE_CACHE_PREFIX": "responses",
        "RESPONSES_STORE_CACHE_DIR": str(tmp_path),
        "RESPONSES_STORE_CACHE_MEMORY_TTL_SECONDS": "60",
        "RESPONSES_STORE_CACHE_DISK_TTL_SECONDS": "60",
    }
    request = {
        "model": "gpt-test",
        "input": "private",
        "api_keys": {"openai": ["private-key"]},
        "private": True,
    }
    first = ResponsesService(store=create_configured_responses_store(env=env))
    request_scope = first.prepare_request_scope(request)
    response = await first.create_response(
        request,
        FakeClient(),
        request_scope=request_scope,
    )
    restarted = ResponsesService(store=create_configured_responses_store(env=env))

    loaded = await restarted.get_response_with_access_token(
        response["id"],
        request_scope.access_token,
    )

    assert loaded["id"] == response["id"]


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
            scope_key="public",
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
        {"role": "user", "content": "Earlier"},
        {"role": "assistant", "content": "Earlier"},
        {"role": "user", "content": "Continue"},
    ]


@pytest.mark.asyncio
async def test_previous_response_id_loads_full_lineage_oldest_first() -> None:
    store = InMemoryResponsesStore()
    await store.save(
        StoredResponse(
            id="resp_grandparent",
            model="gpt-test",
            status="completed",
            scope_key="public",
            request={"model": "gpt-test", "input": "First"},
            response={"id": "resp_grandparent", "object": "response", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "First answer"}]}]},
        )
    )
    await store.save(
        StoredResponse(
            id="resp_parent",
            model="gpt-test",
            status="completed",
            scope_key="public",
            request={"model": "gpt-test", "input": "Second", "previous_response_id": "resp_grandparent"},
            response={"id": "resp_parent", "object": "response", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Second answer"}]}]},
        )
    )
    client = FakeClient()

    await ResponsesService(store=store).create_response({"model": "gpt-test", "input": "Third", "previous_response_id": "resp_parent"}, client)

    assert client.calls[0]["messages"] == [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second"},
        {"role": "assistant", "content": "Second answer"},
        {"role": "user", "content": "Third"},
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
            scope_key="public",
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
            scope_key="public",
            response={"id": "resp_parent", "object": "response", "output": []},
            metadata={"session_affinity_key": "affinity-parent"},
        )
    )
    client = FakeInternalClient()

    response = await service.create_response({"model": "gpt-test", "input": "Continue", "previous_response_id": "resp_parent"}, client)
    stored = await store.get(response["id"])

    assert client.internal_hints.affinity_key == "responses_previous_response_id:resp_parent"
    assert stored is not None
    assert stored.session_id == "session-parent"
    assert "session_affinity_key" not in stored.metadata


@pytest.mark.asyncio
async def test_scoped_responses_preserve_routing_but_never_store_or_trace_secrets(tmp_path) -> None:
    store = InMemoryResponsesStore()
    service = ResponsesService(store=store)
    client = FakeClient()
    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    raw_request = {
        "model": "gpt-test",
        "input": "Scoped request",
        "api_keys": {"openai": ["super-secret-routing-key"]},
        "providers": {
            "openai": {
                "api_base": "https://private.example",
                "authorization": "provider-secret-header",
            }
        },
        "private": True,
    }

    request_scope = service.prepare_request_scope(raw_request)
    response = await service.create_response(
        raw_request,
        client,
        transaction_logger=logger,
        request_scope=request_scope,
    )
    scope_key = responses_service_module._request_isolation_key(raw_request)
    stored = await store.get(response["id"], scope_key)

    assert stored is not None
    assert stored.scope_key is not None and stored.scope_key.startswith("bundle:")
    assert client.calls[0]["api_keys"] == raw_request["api_keys"]
    assert client.calls[0]["providers"] == raw_request["providers"]
    persisted_text = json.dumps(stored.to_dict())
    trace_text = (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8")
    assert "super-secret-routing-key" not in persisted_text
    assert "provider-secret-header" not in persisted_text
    assert "super-secret-routing-key" not in trace_text
    assert "provider-secret-header" not in trace_text

    with pytest.raises(ResponsesServiceError) as public_access:
        await service.get_response(response["id"])
    assert public_access.value.status_code == 404
    assert (await service.get_response(response["id"], scope_key=stored.scope_key))["id"] == response["id"]

    mismatched = dict(raw_request)
    mismatched["api_keys"] = {"openai": ["different-private-key"]}
    mismatched["previous_response_id"] = response["id"]
    with pytest.raises(ResponsesServiceError) as cross_scope:
        await service.create_response(mismatched, FakeClient())
    assert cross_scope.value.status_code == 404

    continued = dict(raw_request)
    continued["previous_response_id"] = response["id"]
    await service.create_response(
        continued,
        FakeClient(),
        request_scope=service.prepare_request_scope(continued),
        previous_response_access_token=request_scope.access_token,
    )


@pytest.mark.asyncio
async def test_responses_service_records_response_id_session_anchor() -> None:
    class Tracker:
        def __init__(self) -> None:
            self.calls = []

        def record_response(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    tracker = Tracker()

    class Client(FakeInternalClient):
        async def acompletion(self, **kwargs):
            callback = kwargs.pop("_request_context_callback", None)
            kwargs.pop("_session_tracking_hints", None)
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
                            "session_tracker": tracker,
                            "provider": "openai",
                            "model": "gpt-test",
                            "session_tracking_namespace": "namespace",
                        },
                    )()
                )
            self.calls.append(kwargs)
            return {
                "id": "resp_parent",
                "model": kwargs["model"],
                "choices": [{"message": {"role": "assistant", "content": "Hello back"}, "finish_reason": "stop"}],
            }

    await ResponsesService(store=InMemoryResponsesStore()).create_response({"model": "gpt-test", "input": "Hello"}, Client())

    assert tracker.calls[0][0][0] == "session-parent"
    assert tracker.calls[0][1]["response"] == {"id": "resp_parent", "object": "response"}


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


@pytest.mark.asyncio
async def test_service_usage_trace_includes_provider_reported_cost(tmp_path) -> None:
    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    service = ResponsesService(store=InMemoryResponsesStore())

    await service.create_response({"model": "gpt-test", "input": "Hello"}, FakeCostClient(), transaction_logger=logger)

    usage_entry = [entry for entry in _trace_entries(logger.log_dir) if entry["pass_name"] == "usage_accounting_summary"][-1]
    assert usage_entry["data"]["cost"]["provider_reported_cost"] == 0.033
    assert usage_entry["metadata"]["pricing_source"] == "usage.cost_details"


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
        scope_key = "public"
        response = {"output": []}
        output_items = []
        input_items = []

        def to_dict(self):
            raise AssertionError("previous response trace payload should not be built without a logger")

    class Store(InMemoryResponsesStore):
        async def get(self, response_id, scope_key="public"):
            return Parent()

    service = ResponsesService(store=Store())

    parent = await service._load_previous_response(
        "resp_parent",
        None,
        expected_scope_key="public",
    )

    assert parent.id == "resp_parent"
