"""G2 field-cache slice pins.

Covers the response-side injection targets, provider force-cache
declarations, hydration honesty, insert+auto insertion, rule-error
containment, and the weakening guard with response targets.
"""

from __future__ import annotations

import logging

import pytest

from rotator_library.client.executor import (
    RoutingExecutionError,
    _env_cache_replay_cached,
    _merged_field_cache_rules,
)
from rotator_library.field_cache import (
    FieldCacheContext,
    FieldCacheEngine,
    FieldCacheInjection,
    FieldCacheRule,
    InMemoryFieldCacheStore,
    parse_cache_replay_config,
)
from rotator_library.native_provider.context import NativeProviderContext
from rotator_library.native_provider.executor import NativeProviderExecutor
from rotator_library.native_provider.http import NativeHTTPTransport


class _FakeResponse:
    def __init__(self, body: dict, status: int = 200):
        self._body = body
        self.status_code = status

    def json(self):
        return self._body


class _FakeClient:
    def __init__(self, response_body: dict):
        self._response_body = response_body
        self.calls: list[dict] = []

    async def post(self, endpoint, headers=None, json=None, **kwargs):
        self.calls.append({"endpoint": endpoint, "headers": dict(headers or {}), "json": json})
        return _FakeResponse(self._response_body)


def _chat_response(reasoning: str | None = None) -> dict:
    message: dict = {"role": "assistant", "content": "hi"}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-test",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _context(**overrides) -> NativeProviderContext:
    base = dict(
        provider="native",
        model="gpt-test",
        protocol_name="openai_chat",
        input_protocol_name="openai_chat",
        client_protocol_name="openai_chat",
        operation="chat",
        endpoint="https://example.test/chat",
    )
    base.update(overrides)
    return NativeProviderContext(**base)


_REQUEST = {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]}


def _response_rule(inject: FieldCacheInjection, name: str = "resp_state") -> FieldCacheRule:
    return FieldCacheRule(
        name=name,
        source="response",
        path="choices.0.message.reasoning_content",
        inject=inject,
        allow_missing_session=True,
        scope=("provider", "model"),
    )


# --- response-side injection targets ---------------------------------------


@pytest.mark.asyncio
async def test_response_target_injection_round_trip() -> None:
    rule = _response_rule(
        FieldCacheInjection(target="response", path="choices.0.message.cached_state")
    )
    context = _context(field_cache_rules=(rule,))
    executor = NativeProviderExecutor()

    # Prime the cache from a response carrying the field.
    await executor.execute(
        _REQUEST,
        context,
        NativeHTTPTransport(_FakeClient(_chat_response(reasoning="hidden"))),
    )
    # The NEXT provider response gets the cached state injected into its wire.
    result = await executor.execute(
        _REQUEST,
        context,
        NativeHTTPTransport(_FakeClient(_chat_response())),
    )
    assert result["choices"][0]["message"]["cached_state"] == "hidden"


@pytest.mark.asyncio
async def test_response_target_injection_miss_is_clean_noop() -> None:
    rule = _response_rule(
        FieldCacheInjection(target="response", path="choices.0.message.cached_state")
    )
    context = _context(field_cache_rules=(rule,))
    result = await NativeProviderExecutor().execute(
        _REQUEST,
        context,
        NativeHTTPTransport(_FakeClient(_chat_response())),
    )
    # Nothing cached -> the client response is untouched.
    assert "cached_state" not in result["choices"][0]["message"]


@pytest.mark.asyncio
async def test_unified_response_target_injection_round_trip() -> None:
    rule = _response_rule(
        FieldCacheInjection(
            target="unified_response",
            path="messages.0.extra.injected_state",
        )
    )
    context = _context(field_cache_rules=(rule,))
    executor = NativeProviderExecutor()

    await executor.execute(
        _REQUEST,
        context,
        NativeHTTPTransport(_FakeClient(_chat_response(reasoning="hidden"))),
    )
    result = await executor.execute(
        _REQUEST,
        context,
        NativeHTTPTransport(_FakeClient(_chat_response())),
    )
    assert result["choices"][0]["message"]["injected_state"] == "hidden"


# --- provider force-cache declarations -------------------------------------


class _ForceCachePlugin:
    protocol_name = "openai_chat"
    cache_replay = [
        {
            "name": "always_reasoning",
            "source": "response",
            "path": "choices.*.message.reasoning_content",
            "keep": "all",
            "inject": {"target": "request", "path": "messages[-1].reasoning_content"},
        }
    ]

    def get_field_cache_rules(self, model: str = ""):
        return ()


@pytest.mark.asyncio
async def test_force_cache_declaration_compiles_and_runs_from_provider_class() -> None:
    rules = _merged_field_cache_rules("prov", "prov/m1", _ForceCachePlugin(), config=None)
    matches = [rule for rule in rules if rule.name == "always_reasoning"]
    assert len(matches) == 1
    rule = matches[0]
    assert isinstance(rule, FieldCacheRule)
    assert rule.source == "response"
    assert rule.mode == "all"
    assert rule.inject.target == "request"
    assert rule.metadata["cache_replay"] is True

    engine = FieldCacheEngine([rule])
    context = FieldCacheContext(
        provider="prov",
        model="m1",
        credential_id="c",
        session_id="s",
        classifier="global",
    )
    await engine.extract(
        "response", {"choices": [{"message": {"reasoning_content": "think"}}]}, context
    )
    updated, operations = await engine.inject(
        "request", {"messages": [{"role": "user"}]}, context
    )
    assert operations[0].hit is True
    assert updated["messages"][-1]["reasoning_content"] == ["think"]


@pytest.mark.asyncio
async def test_replay_source_dimension_accepts_response_and_rejects_unknown() -> None:
    rules = parse_cache_replay_config(
        '[{"name": "r", "source": "unified_response", "path": "p"}]', provider="prov"
    )
    assert rules[0].source == "unified_response"
    with pytest.raises(ValueError, match="source must be one of"):
        parse_cache_replay_config('[{"name": "r", "source": "nonsense", "path": "p"}]', provider="prov")


# --- hydration honesty -----------------------------------------------------


@pytest.mark.asyncio
async def test_unified_request_injection_hydrates_message_fields() -> None:
    rule = _response_rule(
        FieldCacheInjection(
            target="unified_request",
            path="messages[-1].reasoning_content",
        ),
        name="msg_state",
    )
    context = _context(field_cache_rules=(rule,))
    executor = NativeProviderExecutor()

    await executor.execute(
        _REQUEST,
        context,
        NativeHTTPTransport(_FakeClient(_chat_response(reasoning="hidden"))),
    )
    second_client = _FakeClient(_chat_response())
    await executor.execute(_REQUEST, context, NativeHTTPTransport(second_client))

    assert second_client.calls[0]["json"]["messages"][-1]["reasoning_content"] == "hidden"


# --- insert + auto ---------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_auto_inserts_when_absent_and_is_idempotent() -> None:
    rule = FieldCacheRule(
        name="prefix_message",
        source="response",
        path="system_message",
        inject=FieldCacheInjection(
            target="request",
            path="messages.0",
            insert=True,
            when_missing_only=True,
        ),
        allow_missing_session=True,
        scope=("provider", "model"),
    )
    engine = FieldCacheEngine([rule])
    context = FieldCacheContext(provider="p", model="m", session_id="s")
    cached = {"role": "system", "content": "cached"}
    await engine.extract("response", {"system_message": cached}, context)

    updated, operations = await engine.inject(
        "request", {"messages": [{"role": "user"}]}, context
    )
    assert operations[0].changed is True
    assert updated["messages"] == [cached, {"role": "user"}]

    # A second injection must not duplicate the equal entry.
    again, _ = await engine.inject("request", updated, context)
    assert again["messages"] == updated["messages"]

    # Empty list path also inserts.
    empty, _ = await engine.inject("request", {"messages": []}, context)
    assert empty["messages"] == [cached]


# --- rule-error containment ------------------------------------------------


class _BrokenStore(InMemoryFieldCacheStore):
    async def set(self, key, value, *, ttl_seconds=None):
        raise RuntimeError("store boom")


@pytest.mark.asyncio
async def test_rule_error_is_contained_and_warning_logged() -> None:
    rule = FieldCacheRule(name="boom", source="response", path="v")
    engine = FieldCacheEngine([rule], store=_BrokenStore())
    context = FieldCacheContext(provider="p", model="m", session_id="s")

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("rotator_library.field_cache")
    handler = _Capture(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        operations = await engine.extract("response", {"v": "x"}, context)
    finally:
        logger.removeHandler(handler)

    assert operations[0].skipped is True
    assert operations[0].reason == "rule_error:RuntimeError"
    assert any("skipping rule" in record.getMessage() for record in records)


@pytest.mark.asyncio
async def test_containment_skips_only_the_failing_rule() -> None:
    bad = FieldCacheRule(name="bad", source="response", path="payload", max_bytes=4)
    good = FieldCacheRule(name="good", source="response", path="other")
    engine = FieldCacheEngine([bad, good])
    context = FieldCacheContext(provider="p", model="m", session_id="s")

    operations = await engine.extract("response", {"payload": "x" * 32, "other": "ok"}, context)

    assert operations[0].skipped is True
    assert operations[1].changed is True


@pytest.mark.asyncio
async def test_critical_rule_error_still_raises() -> None:
    rule = FieldCacheRule(name="boom", source="response", path="v", critical=True)
    engine = FieldCacheEngine([rule], store=_BrokenStore())
    context = FieldCacheContext(provider="p", model="m", session_id="s")

    with pytest.raises(RuntimeError, match="store boom"):
        await engine.extract("response", {"v": "x"}, context)


# --- weakening guard with response targets ---------------------------------


def test_weakening_guard_denies_response_target_change(monkeypatch) -> None:
    class _Plugin:
        protocol_name = "openai_chat"

        def get_field_cache_rules(self, model: str = ""):
            return (
                FieldCacheRule(
                    name="state",
                    source="response",
                    path="signature",
                    inject=FieldCacheInjection(target="request", path="sig"),
                    scope=("provider", "model"),
                ),
            )

    monkeypatch.setenv(
        "SHADOWPROV_CACHE_REPLAY",
        '[{"name": "state", "source": "response", "path": "signature", '
        '"inject": {"target": "response", "path": "sig"}}]',
    )
    _env_cache_replay_cached.cache_clear()
    try:
        # Same scope/shape, only the injection target widens request->response:
        # the guard must still deny it.
        with pytest.raises(RoutingExecutionError, match="cannot weaken"):
            _merged_field_cache_rules("shadowprov", "shadowprov/m", _Plugin(), config=None)
    finally:
        _env_cache_replay_cached.cache_clear()
