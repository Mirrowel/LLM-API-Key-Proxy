# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G17 port integration pins: the durable stores write engine rows, never a
whole-store JSON blob."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from rotator_library.responses import ProviderCacheResponsesStore, StoredResponse
from rotator_library.storage.engine import get_engine
from rotator_library.usage.persistence.storage import UsageStorage
from rotator_library.usage.types import CredentialState


def _stored(response_id: str) -> StoredResponse:
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
    )


@pytest.mark.asyncio
async def test_responses_save_touches_exactly_one_engine_row() -> None:
    store = ProviderCacheResponsesStore(prefix="responses_rows")
    engine = store._backend()
    response = _stored("resp_rows")

    before_sets = engine.stats_counters["sets"]
    await store.save(response)
    assert engine.stats_counters["sets"] - before_sets == 1
    assert engine.size() == 1

    # A byte-identical re-save is a dedupe skip: no second blob write.
    before_sets = engine.stats_counters["sets"]
    before_skips = engine.stats_counters["dedupe_skips"]
    await store.save(response)
    assert engine.stats_counters["sets"] - before_sets == 1
    assert engine.stats_counters["dedupe_skips"] - before_skips == 1
    assert engine.size() == 1


@pytest.mark.asyncio
async def test_usage_save_writes_one_row_per_credential(tmp_path: Path) -> None:
    storage = UsageStorage(tmp_path / "usage.json")
    states = {
        f"cred-{index}": CredentialState(
            stable_id=f"cred-{index}",
            provider="test",
            accessor=f"key-{index}",
        )
        for index in range(3)
    }
    engine = storage._engine

    before_sets = engine.stats_counters["sets"]
    assert await storage.save(states, force=True)

    # One row per credential plus the single metadata row.
    assert engine.stats_counters["sets"] - before_sets == 4
    assert len(engine.keys(prefix=storage._prefix())) == 4

    loaded, _fair, is_loaded = await storage.load()
    assert is_loaded is True
    assert set(loaded) == set(states)


@pytest.mark.asyncio
async def test_provider_cache_disk_ttl_env_flows_to_engine_row(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_REASONING_DISK_TTL", "123")
    monkeypatch.setenv("DEEPSEEK_REASONING_CACHE_TTL", "1")

    from rotator_library.providers.deepseek_provider import DeepseekProvider

    cache = DeepseekProvider()._get_reasoning_cache()
    before = time.time()
    await cache.store_async("reasoning-key", "reasoning-value")

    engine = get_engine("cache")
    row = next(iter(engine.iterate(prefix=f"{cache._namespace}:")))
    assert row[1] == b"reasoning-value"
    expires_at = row[2]["expires_at"]
    assert expires_at is not None
    assert before + 100 < expires_at < before + 140
