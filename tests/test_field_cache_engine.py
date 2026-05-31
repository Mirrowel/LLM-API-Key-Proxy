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
from rotator_library.transaction_logger import TransactionLogger


def _trace_entries(log_dir):
    return [json.loads(line) for line in (log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]


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
async def test_missing_credential_scope_skips_instead_of_sharing_none_bucket() -> None:
    rule = _reasoning_rule(scope=("provider", "model", "credential"))
    engine = FieldCacheEngine([rule])

    operations = await engine.extract(
        "response",
        {"choices": [{"message": {"reasoning_content": "x"}}]},
        _context(credential_id=None),
    )
    updated, injection_operations = await engine.inject("request", {"messages": [{}]}, _context(credential_id=None))

    assert operations[0].skipped is True
    assert operations[0].reason == "missing_required_scope"
    assert injection_operations[0].skipped is True
    assert injection_operations[0].reason == "missing_required_scope"
    assert updated == {"messages": [{}]}
    assert build_cache_key(rule, _context(credential_id=None)) is None


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


@pytest.mark.asyncio
async def test_field_cache_trace_omits_raw_sample_values(tmp_path) -> None:
    logger = TransactionLogger("openai", "gpt-test", parent_dir=tmp_path)
    rule = _reasoning_rule()
    engine = FieldCacheEngine([rule])

    await engine.extract(
        "response",
        {"choices": [{"message": {"reasoning_content": "provider-signature-secret"}}]},
        _context(),
        transaction_logger=logger,
    )

    trace_text = (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8")
    assert "provider-signature-secret" not in trace_text
    entries = _trace_entries(logger.log_dir)
    after_entry = next(entry for entry in entries if entry["pass_name"] == "after_field_cache_extraction")
    assert after_entry["metadata"]["sample_value_count"] == 1
    assert after_entry["metadata"]["sample_value_types"] == ["str"]


@pytest.mark.asyncio
async def test_field_cache_error_trace_omits_raw_payload_values(tmp_path) -> None:
    class FailingStore(InMemoryFieldCacheStore):
        async def set(self, key, value, *, ttl_seconds=None):
            raise RuntimeError("store failed")

    logger = TransactionLogger("openai", "gpt-test", parent_dir=tmp_path)
    engine = FieldCacheEngine([_reasoning_rule()], store=FailingStore())

    with pytest.raises(RuntimeError):
        await engine.extract(
            "response",
            {"choices": [{"message": {"reasoning_content": "provider-signature-secret"}}]},
            _context(),
            transaction_logger=logger,
        )

    trace_text = (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8")
    assert "provider-signature-secret" not in trace_text
    assert "payload_type" in trace_text


@pytest.mark.asyncio
async def test_field_cache_traces_start_and_complete_even_without_matching_rules(tmp_path) -> None:
    logger = TransactionLogger("openai", "gpt-test", parent_dir=tmp_path)
    engine = FieldCacheEngine([])

    operations = await engine.extract("response", {"choices": []}, _context(), transaction_logger=logger)
    updated, injection_operations = await engine.inject("request", {"messages": []}, _context(), transaction_logger=logger)

    assert operations == []
    assert injection_operations == []
    assert updated == {"messages": []}
    entries = _trace_entries(logger.log_dir)
    pass_names = [entry["pass_name"] for entry in entries]
    assert pass_names == [
        "field_cache_extraction_start",
        "field_cache_extraction_complete",
        "field_cache_injection_start",
        "field_cache_injection_complete",
    ]
    assert entries[1]["metadata"]["rule_count"] == 0
    assert entries[-1]["metadata"]["operation_count"] == 0


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
async def test_in_memory_store_expires_ttl_values() -> None:
    now = 100.0
    store = InMemoryFieldCacheStore(clock=lambda: now)

    await store.set("key", "value", ttl_seconds=5)
    assert await store.get("key") == "value"
    now = 106.0
    assert await store.get("key") is None


@pytest.mark.asyncio
async def test_provider_cache_field_store_expires_ttl_values(monkeypatch) -> None:
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

    now = 100.0
    monkeypatch.setattr("rotator_library.field_cache.store.time.time", lambda: now)
    store = ProviderCacheFieldStore(FakeProviderCache())

    await store.set("key", "value", ttl_seconds=5)
    assert await store.get("key") == "value"
    now = 106.0
    assert await store.get("key") is None


@pytest.mark.asyncio
async def test_in_memory_store_returns_deep_copies() -> None:
    store = InMemoryFieldCacheStore()
    await store.set("key", {"nested": []})
    value = await store.get("key")
    value["nested"].append("mutated")

    assert await store.get("key") == {"nested": []}


@pytest.mark.asyncio
async def test_last_user_turn_uses_latest_user_message() -> None:
    rule = FieldCacheRule(
        name="user_signature",
        source="request",
        path="messages.*.metadata.signature",
        mode="last_user_turn",
        inject=FieldCacheInjection(target="request", path="metadata.signature"),
        allow_missing_session=True,
    )
    engine = FieldCacheEngine([rule])

    operations = await engine.extract(
        "request",
        {
            "messages": [
                {"role": "user", "metadata": {"signature": "first-user"}},
                {"role": "assistant", "metadata": {"signature": "assistant"}},
                {"role": "user", "metadata": {"signature": "last-user"}},
            ]
        },
        _context(session_id=None),
    )
    updated, _ = await engine.inject("request", {"metadata": {}}, _context(session_id=None))

    assert operations[0].changed is True
    assert updated["metadata"]["signature"] == "last-user"


@pytest.mark.asyncio
async def test_last_mode_preserves_list_valued_field() -> None:
    rule = FieldCacheRule(
        name="list_value",
        source="response",
        path="metadata.signatures",
        inject=FieldCacheInjection(target="request", path="metadata.signatures"),
        allow_missing_session=True,
    )
    engine = FieldCacheEngine([rule])

    await engine.extract("response", {"metadata": {"signatures": ["a", "b"]}}, _context(session_id=None))
    updated, _ = await engine.inject("request", {"metadata": {}}, _context(session_id=None))

    assert updated["metadata"]["signatures"] == ["a", "b"]


@pytest.mark.asyncio
async def test_as_list_unwraps_last_mode_value_envelope() -> None:
    rule = FieldCacheRule(
        name="value",
        source="response",
        path="value",
        inject=FieldCacheInjection(target="request", path="metadata.values", as_list=True),
        allow_missing_session=True,
    )
    engine = FieldCacheEngine([rule])

    await engine.extract("response", {"value": "sig"}, _context(session_id=None))
    updated, _ = await engine.inject("request", {"metadata": {}}, _context(session_id=None))

    assert updated["metadata"]["values"] == ["sig"]


@pytest.mark.asyncio
async def test_last_assistant_turn_skips_without_turn_context() -> None:
    rule = FieldCacheRule(
        name="assistant_signature",
        source="response",
        path="choices.*.message.signature",
        mode="last_assistant_turn",
        inject=FieldCacheInjection(target="request", path="metadata.signature"),
        allow_missing_session=True,
    )
    engine = FieldCacheEngine([rule])

    operations = await engine.extract("response", {"choices": [{"message": {"signature": "sig"}}]}, _context(session_id=None))

    assert operations[0].skipped is True
    assert operations[0].reason == "turn_context_not_found"


@pytest.mark.asyncio
async def test_turn_mode_uses_metadata_configured_relative_paths() -> None:
    rule = FieldCacheRule(
        name="assistant_signature",
        source="response",
        path="unused.global.path",
        mode="last_assistant_turn",
        inject=FieldCacheInjection(target="request", path="metadata.signature"),
        allow_missing_session=True,
        metadata={"turn_container_path": "messages", "turn_role_path": "kind", "turn_value_path": "parts.*.signature"},
    )
    engine = FieldCacheEngine([rule])

    await engine.extract(
        "response",
        {"messages": [{"kind": "assistant", "parts": [{"signature": "first"}]}, {"kind": "assistant", "parts": [{"signature": "second"}]}]},
        _context(session_id=None),
    )
    updated, _ = await engine.inject("request", {"metadata": {}}, _context(session_id=None))

    assert updated["metadata"]["signature"] == "second"


@pytest.mark.asyncio
async def test_per_tool_call_correlates_sibling_id_and_value_for_injection() -> None:
    rule = FieldCacheRule(
        name="tool_signature",
        source="response",
        path="tool_calls.*.signature",
        mode="per_tool_call",
        inject=FieldCacheInjection(target="request", path="metadata.signature"),
        allow_missing_session=True,
        metadata={
            "tool_container_path": "tool_calls",
            "tool_call_id_path": "id",
            "tool_value_path": "signature",
        },
    )
    engine = FieldCacheEngine([rule])

    await engine.extract("response", {"tool_calls": [{"id": "call_a", "signature": "sig-a"}, {"id": "call_b", "signature": "sig-b"}]}, _context(session_id=None))
    updated, operations = await engine.inject("request", {"metadata": {}}, _context(session_id=None, metadata={"tool_call_id": "call_b"}))

    assert operations[0].hit is True
    assert updated["metadata"]["signature"] == "sig-b"


@pytest.mark.asyncio
async def test_per_tool_call_as_list_injects_matching_values() -> None:
    rule = FieldCacheRule(
        name="tool_signature",
        source="response",
        path="tool_calls.*",
        mode="per_tool_call",
        inject=FieldCacheInjection(target="request", path="metadata.signatures", as_list=True),
        allow_missing_session=True,
        metadata={"tool_call_id_path": "id", "inject_tool_call_id_path": "tool_ids.*"},
    )
    engine = FieldCacheEngine([rule])

    await engine.extract("response", {"tool_calls": [{"id": "a", "signature": "sig-a"}, {"id": "b", "signature": "sig-b"}]}, _context(session_id=None))
    updated, _ = await engine.inject("request", {"metadata": {}, "tool_ids": ["a", "b"]}, _context(session_id=None))

    assert updated["metadata"]["signatures"] == [{"id": "a", "signature": "sig-a"}, {"id": "b", "signature": "sig-b"}]


@pytest.mark.asyncio
async def test_per_tool_call_preserves_list_valued_match() -> None:
    rule = FieldCacheRule(
        name="tool_signatures",
        source="response",
        path="tool_calls.*.signatures",
        mode="per_tool_call",
        inject=FieldCacheInjection(target="request", path="metadata.signatures"),
        allow_missing_session=True,
        metadata={"tool_container_path": "tool_calls", "tool_call_id_path": "id", "tool_value_path": "signatures"},
    )
    engine = FieldCacheEngine([rule])

    await engine.extract("response", {"tool_calls": [{"id": "call", "signatures": ["a", "b"]}]}, _context(session_id=None))
    updated, _ = await engine.inject("request", {"metadata": {}}, _context(session_id=None, metadata={"tool_call_id": "call"}))

    assert updated["metadata"]["signatures"] == ["a", "b"]


@pytest.mark.asyncio
async def test_engine_supports_legacy_store_without_ttl_keyword() -> None:
    class LegacyStore:
        def __init__(self) -> None:
            self.values = {}

        async def get(self, key):
            return self.values.get(key)

        async def set(self, key, value):
            self.values[key] = value

        async def append(self, key, values):
            self.values.setdefault(key, []).extend(values)
            return self.values[key]

        async def clear(self):
            self.values.clear()

    engine = FieldCacheEngine([_reasoning_rule()], store=LegacyStore())

    await engine.extract("response", {"choices": [{"message": {"reasoning_content": "legacy"}}]}, _context())
    updated, _ = await engine.inject("request", {"messages": [{}]}, _context())

    assert updated["messages"][-1]["reasoning_content"] == "legacy"


@pytest.mark.asyncio
async def test_per_tool_call_skips_when_current_tool_id_is_ambiguous() -> None:
    rule = FieldCacheRule(
        name="tool_signature",
        source="response",
        path="tool_calls.*",
        mode="per_tool_call",
        inject=FieldCacheInjection(target="request", path="metadata.signature"),
        allow_missing_session=True,
        metadata={"tool_call_id_path": "id"},
    )
    engine = FieldCacheEngine([rule])

    await engine.extract("response", {"tool_calls": [{"id": "call_a", "signature": "sig-a"}]}, _context(session_id=None))
    updated, operations = await engine.inject("request", {"metadata": {}}, _context(session_id=None))

    assert operations[0].skipped is True
    assert operations[0].reason == "tool_call_id_not_found"
    assert updated == {"metadata": {}}


@pytest.mark.asyncio
async def test_insert_injection_adds_list_entry() -> None:
    rule = FieldCacheRule(
        name="prefix_message",
        source="response",
        path="message",
        inject=FieldCacheInjection(target="request", path="messages.0", insert=True),
        allow_missing_session=True,
    )
    engine = FieldCacheEngine([rule])

    await engine.extract("response", {"message": {"role": "system", "content": "cached"}}, _context(session_id=None))
    updated, _ = await engine.inject("request", {"messages": [{"role": "user", "content": "hi"}]}, _context(session_id=None))

    assert updated["messages"] == [{"role": "system", "content": "cached"}, {"role": "user", "content": "hi"}]
