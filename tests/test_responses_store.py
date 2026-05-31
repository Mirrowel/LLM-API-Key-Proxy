from __future__ import annotations

import json
import time

import pytest

from rotator_library.responses import InMemoryResponsesStore, ProviderCacheResponsesStore, StoredResponse, create_configured_responses_store, generate_response_id


def _stored(response_id: str = "resp_test") -> StoredResponse:
    return StoredResponse(
        id=response_id,
        model="gpt-test",
        status="completed",
        request={"model": "gpt-test", "input": "hello"},
        response={"id": response_id, "object": "response", "output": []},
        input_items=[{"type": "message", "role": "user", "content": "hello"}],
        output_items=[{"type": "message", "role": "assistant", "content": []}],
        usage={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        metadata={"previous_response_id": None},
    )


def test_generate_response_id_uses_responses_prefix() -> None:
    assert generate_response_id().startswith("resp_")
    assert generate_response_id() != generate_response_id()


@pytest.mark.asyncio
async def test_in_memory_store_save_get_delete_and_input_items() -> None:
    store = InMemoryResponsesStore()
    stored = _stored()

    await store.save(stored)
    loaded = await store.get(stored.id)
    input_items = await store.list_input_items(stored.id)
    deleted = await store.delete(stored.id)

    assert loaded is not None
    assert loaded.to_dict() == stored.to_dict()
    assert input_items == stored.input_items
    assert deleted is True
    assert await store.get(stored.id) is None


@pytest.mark.asyncio
async def test_in_memory_store_returns_copies_and_expires() -> None:
    store = InMemoryResponsesStore()
    stored = _stored("resp_expiring")
    stored.expires_at = time.time() + 100
    await store.save(stored)

    loaded = await store.get(stored.id)
    assert loaded is not None
    loaded.response["mutated"] = True
    assert (await store.get(stored.id)).response.get("mutated") is None

    stored.expires_at = time.time() - 1
    await store.save(stored)
    assert await store.get(stored.id) is None


@pytest.mark.asyncio
async def test_in_memory_store_prunes_oldest_when_max_items_exceeded() -> None:
    store = InMemoryResponsesStore(max_items=2)
    first = _stored("resp_first")
    first.created_at = 1
    second = _stored("resp_second")
    second.created_at = 2
    third = _stored("resp_third")
    third.created_at = 3

    await store.save(first)
    await store.save(second)
    await store.save(third)

    assert await store.get("resp_first") is None
    assert await store.get("resp_second") is not None
    assert await store.get("resp_third") is not None


@pytest.mark.asyncio
async def test_provider_cache_store_serializes_json_and_does_not_clear_without_key_delete() -> None:
    class FakeProviderCache:
        def __init__(self) -> None:
            self.values = {}

        async def store_async(self, key: str, value: str) -> None:
            json.loads(value)
            self.values[key] = value

        async def retrieve_async(self, key: str):
            return self.values.get(key)

    cache = FakeProviderCache()
    store = ProviderCacheResponsesStore(cache)
    stored = _stored("resp_provider_cache")

    await store.save(stored)
    loaded = await store.get(stored.id)

    assert loaded is not None
    assert loaded.id == stored.id
    assert await store.list_input_items(stored.id) == stored.input_items
    assert await store.delete(stored.id) is False
    assert cache.values


@pytest.mark.asyncio
async def test_provider_cache_store_uses_key_delete_when_available() -> None:
    class FakeProviderCache:
        def __init__(self) -> None:
            self.values = {}

        async def store_async(self, key: str, value: str) -> None:
            self.values[key] = value

        async def retrieve_async(self, key: str):
            return self.values.get(key)

        async def delete_async(self, key: str) -> bool:
            return self.values.pop(key, None) is not None

    store = ProviderCacheResponsesStore(FakeProviderCache())
    stored = _stored("resp_delete")
    await store.save(stored)

    assert await store.delete(stored.id) is True
    assert await store.get(stored.id) is None


@pytest.mark.asyncio
async def test_configured_provider_cache_store_persists_between_instances(tmp_path) -> None:
    env = {
        "RESPONSES_STORE_BACKEND": "provider_cache",
        "RESPONSES_STORE_CACHE_NAME": "responses_test",
        "RESPONSES_STORE_CACHE_PREFIX": "responses",
        "RESPONSES_STORE_CACHE_DIR": str(tmp_path),
        "RESPONSES_STORE_CACHE_MEMORY_TTL_SECONDS": "60",
        "RESPONSES_STORE_CACHE_DISK_TTL_SECONDS": "60",
    }
    first_store = create_configured_responses_store(env=env)
    stored = _stored("resp_durable")

    await first_store.save(stored)
    second_store = create_configured_responses_store(env=env)
    loaded = await second_store.get(stored.id)

    assert loaded is not None
    assert loaded.to_dict() == stored.to_dict()
