"""G11 Phase B STORE pins: poison containment, never-fail store, durable default."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from rotator_library.responses import (
    InMemoryResponsesStore,
    ProviderCacheResponsesStore,
    ResponsesService,
    ResponsesStoreSettings,
    StoredResponse,
    create_configured_responses_store,
)
from rotator_library.transaction_logger import TransactionLogger


class _FakeCache:
    """Minimal key-value cache mirroring the ProviderCache async surface."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def store_async(self, key: str, value: str) -> None:
        self.values[key] = value

    async def retrieve_async(self, key: str):
        return self.values.get(key)

    async def delete_async(self, key: str) -> bool:
        return self.values.pop(key, None) is not None

    async def shutdown(self) -> None:
        return None


class _FailingStore:
    async def save(self, response) -> None:
        raise RuntimeError("store down Authorization: Bearer secret-token")

    async def get(self, response_id, scope_key: str = "public"):
        return None

    async def delete(self, response_id, scope_key: str = "public") -> bool:
        return False

    async def list_input_items(self, response_id, scope_key: str = "public"):
        return None


class _FakeClient:
    async def agenerate(self, payload, *, input_protocol, request=None, **kwargs):
        return {
            "id": "resp_create",
            "object": "response",
            "model": payload.get("model", "gpt-test"),
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }


def _stored(response_id: str = "resp_test", *, expires_at=None) -> StoredResponse:
    return StoredResponse(
        id=response_id,
        model="gpt-test",
        status="completed",
        request={"model": "gpt-test", "input": "hello"},
        response={"id": response_id, "object": "response", "output": []},
        input_items=[{"type": "message", "role": "user", "content": "hello"}],
        output_items=[],
        metadata={},
        scope_key="public",
        expires_at=expires_at,
    )


def _raw_row(expires_at_value) -> str:
    row = _stored().to_dict()
    row["expires_at"] = expires_at_value
    return json.dumps(row, allow_nan=True)


# --------------------------------------------------- poison-row containment


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "poison",
    ["not-a-number", float("nan"), float("inf"), float("-inf")],
)
async def test_durable_poison_expiry_degrades_to_unexpired(poison) -> None:
    """Non-numeric and non-finite expiries are unusable, not fatal: they
    degrade to ``None`` so the row lives instead of raising on every read."""

    cache = _FakeCache()
    store = ProviderCacheResponsesStore(cache)
    cache.values[store._key("resp_test", "public")] = _raw_row(poison)

    loaded = await store.get("resp_test")

    assert loaded is not None
    assert loaded.expires_at is None


@pytest.mark.asyncio
async def test_durable_negative_expiry_is_a_miss() -> None:
    cache = _FakeCache()
    store = ProviderCacheResponsesStore(cache)
    cache.values[store._key("resp_test", "public")] = _raw_row(time.time() - 1)

    assert await store.get("resp_test") is None


@pytest.mark.asyncio
async def test_in_memory_poison_expiry_is_a_miss_not_an_error() -> None:
    store = InMemoryResponsesStore()
    poison = _stored("resp_poison")
    poison.expires_at = "not-a-number"  # type: ignore[assignment]
    store._responses[("public", "resp_poison")] = poison

    assert await store.get("resp_poison") is None


def test_from_dict_coerces_expiry_edges() -> None:
    base = _stored().to_dict()

    for poison, expected in (
        ("not-a-number", None),
        (float("nan"), None),
        (float("inf"), None),
        ("12.5", 12.5),
        (-3.0, -3.0),
    ):
        row = dict(base)
        row["expires_at"] = poison
        assert StoredResponse.from_dict(row).expires_at == expected


# ------------------------------------------------------- store never fails


@pytest.mark.asyncio
async def test_create_response_survives_failing_store(tmp_path) -> None:
    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    service = ResponsesService(store=_FailingStore())

    response = await service.create_response(
        {"model": "gpt-test", "input": "Hello"},
        _FakeClient(),
        transaction_logger=logger,
    )

    assert response["id"] == "resp_create"
    from tests.txn_helpers import error_records, record_errors

    errors = error_records(logger)
    assert any(entry["failed_pass_name"] == "responses_store_response" for entry in errors)
    assert "secret-token" not in json.dumps(errors)
    assert "secret-token" not in json.dumps(record_errors(logger), default=str)


@pytest.mark.asyncio
async def test_native_stream_store_failure_does_not_double_terminal(tmp_path) -> None:
    logger = TransactionLogger("responses", "gpt-test", parent_dir=tmp_path)
    service = ResponsesService(store=_FailingStore())

    from rotator_library.responses import ResponsesStreamEvent

    terminal = [
        event
        async for event in service._stream_native_response(
            {"model": "gpt-test", "input": "Hello", "stream": True},
            _NativeClient(),
            request=None,
            transaction_logger=logger,
            transport="sse",
            request_scope=None,
            previous_response_access_token=None,
            as_events=True,
        )
    ]

    names = [event.event_name for event in terminal]
    assert names.count("response.completed") == 1
    assert "response.failed" not in names
    assert isinstance(terminal[0], ResponsesStreamEvent)


class _NativeClient:
    async def agenerate(self, payload, **kwargs):
        async def chunks():
            yield 'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_native","object":"response","status":"completed","model":"gpt-test","output":[]}}\n\n'

        return chunks()


# ------------------------------------------------------ durable store policy


@pytest.mark.asyncio
async def test_configured_store_defaults_to_provider_cache(tmp_path) -> None:
    store = create_configured_responses_store(env={"RESPONSES_STORE_CACHE_DIR": str(tmp_path)})
    try:
        assert isinstance(store, ProviderCacheResponsesStore)
        assert store.max_items == 10000
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_durable_store_enforces_max_items() -> None:
    store = ProviderCacheResponsesStore(_FakeCache(), max_items=2)
    for index in range(3):
        row = _stored(f"resp_{index}")
        row.created_at = float(index + 1)
        await store.save(row)

    assert await store.get("resp_0") is None
    assert await store.get("resp_1") is not None
    assert await store.get("resp_2") is not None


# -------------------------------------------------------- created_at stamp


def test_stored_response_always_proxy_stamps_created_at() -> None:
    service = ResponsesService.__new__(ResponsesService)
    service.store_settings = ResponsesStoreSettings()
    service.request_scope_key = lambda raw: "public"  # type: ignore[method-assign]

    before = time.time()
    stored = service._stored_response(
        {"model": "gpt-test", "input": "hi"},
        {"id": "resp_1", "model": "gpt-test", "created_at": 1.0, "output": []},
        None,
        session_info={"provider": "gpt-test", "scope_key": "public"},
    )
    after = time.time()

    assert before <= stored.created_at <= after
    assert stored.metadata["provider_created_at"] == 1.0
