from __future__ import annotations

import json

import pytest

from rotator_library.field_cache import (
    FieldCacheContext,
    FieldCacheEngine,
    FieldCacheInjection,
    FieldCacheRule,
    InMemoryFieldCacheStore,
    ProviderCacheFieldStore,
    build_cache_key,
)


def _reasoning_rule(mode: str = "last", scope=("provider", "model", "session")) -> FieldCacheRule:
    return FieldCacheRule(
        name="reasoning_content",
        source="response",
        path="choices.*.message.reasoning_content",
        mode=mode,
        scope=scope,
        inject=FieldCacheInjection(target="request", path="messages[-1].reasoning_content"),
    )


def _context(**overrides) -> FieldCacheContext:
    values = {"provider": "openai", "model": "gpt-test", "session_id": "session-a", "classifier": "global"}
    values.update(overrides)
    return FieldCacheContext(**values)


@pytest.mark.asyncio
async def test_extract_response_value_and_inject_into_next_request() -> None:
    engine = FieldCacheEngine([_reasoning_rule()])
    response = {"choices": [{"message": {"reasoning_content": "hidden"}}]}
    request = {"messages": [{"role": "user", "content": "hi"}]}

    operations = await engine.extract("response", response, _context())
    updated, injection_operations = await engine.inject("request", request, _context())

    assert operations[0].matched == 1
    assert injection_operations[0].hit is True
    assert updated["messages"][-1]["reasoning_content"] == "hidden"
    assert "reasoning_content" not in request["messages"][-1]


@pytest.mark.asyncio
async def test_last_mode_overwrites_prior_value() -> None:
    engine = FieldCacheEngine([_reasoning_rule()])

    await engine.extract("response", {"choices": [{"message": {"reasoning_content": "first"}}]}, _context())
    await engine.extract("response", {"choices": [{"message": {"reasoning_content": "second"}}]}, _context())
    updated, _ = await engine.inject("request", {"messages": [{}]}, _context())

    assert updated["messages"][-1]["reasoning_content"] == "second"


@pytest.mark.asyncio
async def test_all_mode_appends_values() -> None:
    engine = FieldCacheEngine([_reasoning_rule(mode="all")])

    await engine.extract("response", {"choices": [{"message": {"reasoning_content": "first"}}]}, _context())
    await engine.extract("response", {"choices": [{"message": {"reasoning_content": "second"}}]}, _context())
    updated, _ = await engine.inject("request", {"messages": [{}]}, _context())

    assert updated["messages"][-1]["reasoning_content"] == ["first", "second"]


@pytest.mark.asyncio
async def test_scope_isolation_by_session_and_classifier() -> None:
    rule = _reasoning_rule(scope=("provider", "model", "session", "classifier"))
    engine = FieldCacheEngine([rule])

    await engine.extract("response", {"choices": [{"message": {"reasoning_content": "a"}}]}, _context(session_id="session-a"))
    await engine.extract("response", {"choices": [{"message": {"reasoning_content": "b"}}]}, _context(session_id="session-b"))

    updated, _ = await engine.inject("request", {"messages": [{}]}, _context(session_id="session-a"))

    assert updated["messages"][-1]["reasoning_content"] == "a"
    assert build_cache_key(rule, _context(session_id="session-a")) != build_cache_key(rule, _context(session_id="session-b"))


@pytest.mark.asyncio
async def test_scope_isolation_by_credential_and_provider() -> None:
    rule = _reasoning_rule(scope=("provider", "model", "credential"))
    engine = FieldCacheEngine([rule])

    await engine.extract(
        "response",
        {"choices": [{"message": {"reasoning_content": "cred-a"}}]},
        _context(provider="openai", credential_id="credential-a"),
    )
    await engine.extract(
        "response",
        {"choices": [{"message": {"reasoning_content": "cred-b"}}]},
        _context(provider="openai", credential_id="credential-b"),
    )

    updated, _ = await engine.inject("request", {"messages": [{}]}, _context(provider="openai", credential_id="credential-a"))

    assert updated["messages"][-1]["reasoning_content"] == "cred-a"
    assert build_cache_key(rule, _context(provider="openai", credential_id="credential-a")) != build_cache_key(
        rule, _context(provider="other", credential_id="credential-a")
    )


@pytest.mark.asyncio
async def test_missing_session_scope_skips_by_default() -> None:
    engine = FieldCacheEngine([_reasoning_rule()])

    operations = await engine.extract("response", {"choices": [{"message": {"reasoning_content": "x"}}]}, _context(session_id=None))

    assert operations[0].skipped is True
    assert operations[0].reason == "missing_required_scope"


@pytest.mark.asyncio
async def test_missing_path_is_noop() -> None:
    engine = FieldCacheEngine([_reasoning_rule()])

    operations = await engine.extract("response", {"choices": []}, _context())

    assert operations[0].matched == 0
    assert operations[0].changed is False


@pytest.mark.asyncio
async def test_stream_event_extraction() -> None:
    rule = FieldCacheRule(
        name="provider_session_id",
        source="stream_event",
        path="metadata.provider_session_id",
        scope=("provider", "model", "session"),
        inject=FieldCacheInjection(target="request", path="metadata.provider_session_id"),
    )
    engine = FieldCacheEngine([rule])

    await engine.extract("stream_event", {"metadata": {"provider_session_id": "sid_1"}}, _context())
    updated, _ = await engine.inject("request", {"metadata": {}}, _context())

    assert updated["metadata"]["provider_session_id"] == "sid_1"


@pytest.mark.asyncio
async def test_trace_sample_values_are_truncated() -> None:
    rule = _reasoning_rule()
    engine = FieldCacheEngine([rule])
    long_value = "x" * 700

    operations = await engine.extract(
        "response",
        {"choices": [{"message": {"reasoning_content": long_value}}]},
        _context(),
    )

    assert operations[0].sample_values[0].endswith("...<truncated 200 chars>")


def test_per_tool_call_requires_tool_call_id_path() -> None:
    with pytest.raises(ValueError):
        FieldCacheEngine([
            FieldCacheRule(name="tool_state", source="response", path="tool_calls.*", mode="per_tool_call")
        ])


@pytest.mark.asyncio
async def test_provider_cache_field_store_wraps_json_string_cache() -> None:
    class FakeProviderCache:
        def __init__(self) -> None:
            self.values = {}

        async def retrieve_async(self, key: str):
            return self.values.get(key)

        async def store_async(self, key: str, value: str) -> None:
            json.loads(value)
            self.values[key] = value

        async def clear(self) -> None:
            self.values.clear()

    store = ProviderCacheFieldStore(FakeProviderCache())

    await store.set("key", {"value": 1})
    assert await store.get("key") == {"value": 1}
    await store.append("key", [{"value": 2}])
    assert await store.get("key") == [{"value": 2}]


@pytest.mark.asyncio
async def test_in_memory_store_returns_deep_copies() -> None:
    store = InMemoryFieldCacheStore()
    await store.set("key", {"nested": []})
    value = await store.get("key")
    value["nested"].append("mutated")

    assert await store.get("key") == {"nested": []}
