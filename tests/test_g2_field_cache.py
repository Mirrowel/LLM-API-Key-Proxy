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


_REQUEST = {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]}


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
        "request", {"messages": [{"role": "user"}, {"role": "assistant"}]}, context
    )
    assert operations[0].hit is True
    assert updated["messages"][-1]["reasoning_content"] == "think"


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


# --- G8 step 3: turn-region vocabulary --------------------------------------


def _regions_for(payload: dict) -> list[int]:
    from rotator_library.field_cache.engine import _resolve_turn_shape, _turn_region_indexes

    rule = FieldCacheRule(name="region_probe", source="response", path="unused.path")
    shape = _resolve_turn_shape(rule, payload)
    assert shape is not None
    container_path, role_path, content_path = shape
    from rotator_library.field_cache.engine import _container_items

    return _turn_region_indexes(_container_items(payload, container_path), role_path, content_path)


def test_turn_boundaries_openai_chat_shape() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "tool_call_id": "t1", "content": "tool output"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]},
            {"role": "user", "content": "q2"},
        ]
    }
    # system prefix is region 0; q1 starts region 1; the tool message AND the
    # tool-result-only user message stay in region 1; q2 starts region 2.
    assert _regions_for(payload) == [0, 1, 1, 1, 1, 2]


def test_turn_boundaries_anthropic_shape() -> None:
    payload = {
        "messages": [
            {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x"}]},
            {"role": "user", "content": [{"type": "text", "text": "q2"}]},
        ]
    }
    # tool_result-only user messages never start a region.
    assert _regions_for(payload) == [0, 0, 1]


def test_turn_boundaries_gemini_shape() -> None:
    payload = {
        "contents": [
            {"role": "model", "parts": [{"text": "a1"}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "f", "response": {}}}]},
            {"role": "user", "parts": [{"text": "q2"}]},
        ]
    }
    # functionResponse parts stay in the current region.
    assert _regions_for(payload) == [0, 0, 1]


def test_turn_boundaries_responses_shape() -> None:
    payload = {
        "input": [
            {"type": "function_call_output", "call_id": "t1", "output": "x"},
            {"role": "user", "content": "q2"},
        ]
    }
    # function_call_output items never start a region.
    assert _regions_for(payload) == [0, 1]


def test_consecutive_user_messages_start_separate_regions() -> None:
    payload = {"messages": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]}
    assert _regions_for(payload) == [0, 1]


def _region_rule(mode: str, turn_count: int = 1) -> FieldCacheRule:
    return FieldCacheRule(
        name=f"reasoning_{mode}",
        source="response",
        path="choices.*.message.reasoning_content",
        mode=mode,
        turn_count=turn_count,
        inject=FieldCacheInjection(target="request", path="messages.*.reasoning_content"),
        allow_missing_session=True,
        scope=("provider", "model"),
    )


def _three_region_request() -> dict:
    return {
        "messages": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
    }


async def _extract_three_regions(engine: FieldCacheEngine, context: FieldCacheContext) -> None:
    for content, reasoning in (("a1", "r1"), ("a2", "r2"), ("a3", "r3")):
        await engine.extract(
            "response",
            {"choices": [{"message": {"role": "assistant", "content": content, "reasoning_content": reasoning}}]},
            context,
        )


@pytest.mark.asyncio
async def test_turn_mode_injects_only_the_latest_region() -> None:
    engine = FieldCacheEngine([_region_rule("turn")])
    context = FieldCacheContext(provider="p", model="m")
    await _extract_three_regions(engine, context)

    updated, operations = await engine.inject("request", _three_region_request(), context)

    assert operations[0].hit is True
    assert [m.get("reasoning_content") for m in updated["messages"]] == [None, None, None, None, None, "r3"]


@pytest.mark.asyncio
async def test_turns_mode_injects_last_turn_count_regions() -> None:
    engine = FieldCacheEngine([_region_rule("turns", turn_count=2)])
    context = FieldCacheContext(provider="p", model="m")
    await _extract_three_regions(engine, context)

    updated, operations = await engine.inject("request", _three_region_request(), context)

    assert operations[0].hit is True
    assert [m.get("reasoning_content") for m in updated["messages"]] == [None, None, None, "r2", None, "r3"]


@pytest.mark.asyncio
async def test_all_mode_injects_every_region() -> None:
    engine = FieldCacheEngine([_region_rule("all")])
    context = FieldCacheContext(provider="p", model="m")
    await _extract_three_regions(engine, context)

    updated, operations = await engine.inject("request", _three_region_request(), context)

    assert operations[0].hit is True
    assert [m.get("reasoning_content") for m in updated["messages"]] == [None, "r1", None, "r2", None, "r3"]


@pytest.mark.asyncio
async def test_correlation_prefers_tool_call_ids_over_content_sha() -> None:
    rule = FieldCacheRule(
        name="tool_reasoning",
        source="response",
        path="choices.*.message.reasoning_content",
        mode="all",
        inject=FieldCacheInjection(target="request", path="messages.*.reasoning_content"),
        allow_missing_session=True,
        scope=("provider", "model"),
        metadata={"tool_call_id_path": "tool_calls.*.id"},
    )
    engine = FieldCacheEngine([rule])
    context = FieldCacheContext(provider="p", model="m")

    # Identical completion content: the content sha can only ever hold the
    # newest value, so correct per-occurrence values prove id-first lookup.
    for tool_id, reasoning in (("t1", "r1"), ("t2", "r2")):
        await engine.extract(
            "response",
            {"choices": [{"message": {"role": "assistant", "content": "same", "reasoning_content": reasoning, "tool_calls": [{"id": tool_id}]}}]},
            context,
        )
    request = {
        "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "same", "tool_calls": [{"id": "t1"}]},
            {"role": "assistant", "content": "same", "tool_calls": [{"id": "t2"}]},
        ]
    }
    updated, _ = await engine.inject("request", request, context)

    assert updated["messages"][1]["reasoning_content"] == "r1"
    assert updated["messages"][2]["reasoning_content"] == "r2"


@pytest.mark.asyncio
async def test_correlation_falls_back_to_content_sha_without_tool_ids() -> None:
    rule = _region_rule("all")
    engine = FieldCacheEngine([rule])
    context = FieldCacheContext(provider="p", model="m")
    await _extract_three_regions(engine, context)

    updated, _ = await engine.inject("request", _three_region_request(), context)

    assert updated["messages"][1]["reasoning_content"] == "r1"
    assert updated["messages"][3]["reasoning_content"] == "r2"
    assert updated["messages"][5]["reasoning_content"] == "r3"


@pytest.mark.asyncio
async def test_placeholder_injects_and_warns_when_no_key_correlates() -> None:
    rule = FieldCacheRule(
        name="placeholder_rule",
        source="response",
        path="choices.*.message.reasoning_content",
        mode="all",
        placeholder="<no-reasoning>",
        inject=FieldCacheInjection(target="request", path="messages.*.reasoning_content"),
        allow_missing_session=True,
        scope=("provider", "model"),
    )
    engine = FieldCacheEngine([rule])
    context = FieldCacheContext(provider="prov-x", model="model-y")

    await engine.extract(
        "response",
        {"choices": [{"message": {"role": "assistant", "content": "known", "reasoning_content": "r1"}}]},
        context,
    )
    request = {
        "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "known"},
            {"role": "assistant", "content": "never-seen"},
        ]
    }

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("rotator_library.field_cache")
    handler = _Capture(level=logging.WARNING)
    logger.addHandler(handler)
    try:
        updated, operations = await engine.inject("request", request, context)
    finally:
        logger.removeHandler(handler)

    assert updated["messages"][1]["reasoning_content"] == "r1"
    assert updated["messages"][2]["reasoning_content"] == "<no-reasoning>"
    warnings = [record for record in records if record.levelno == logging.WARNING and "placeholder" in record.getMessage()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "placeholder_rule" in message and "prov-x" in message and "model-y" in message
    assert operations[0].skipped is False


@pytest.mark.asyncio
async def test_auto_vs_always_per_occurrence() -> None:
    def _rule(name: str, when_missing_only: bool) -> FieldCacheRule:
        return FieldCacheRule(
            name=name,
            source="response",
            path="choices.*.message.reasoning_content",
            mode="turn",
            inject=FieldCacheInjection(
                target="request",
                path="messages.*.reasoning_content",
                when_missing_only=when_missing_only,
            ),
            allow_missing_session=True,
            scope=("provider", "model"),
        )

    context = FieldCacheContext(provider="p", model="m")
    request = {"messages": [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1", "reasoning_content": "client"}]}
    response = {"choices": [{"message": {"role": "assistant", "content": "a1", "reasoning_content": "cached"}}]}

    # Auto: the occurrence already carries a client value — preserved.
    auto_engine = FieldCacheEngine([_rule("auto_state", True)])
    await auto_engine.extract("response", response, context)
    updated, _ = await auto_engine.inject("request", request, context)
    assert updated["messages"][1]["reasoning_content"] == "client"

    # Always: the cached value overrides the client value.
    always_engine = FieldCacheEngine([_rule("always_state", False)])
    await always_engine.extract("response", response, context)
    updated, _ = await always_engine.inject("request", request, context)
    assert updated["messages"][1]["reasoning_content"] == "cached"


# --- request-source extraction toggle ---------------------------------------


def _request_source_rule() -> FieldCacheRule:
    return FieldCacheRule(
        name="request_marker",
        source="request",
        path="messages.*.marker",
        inject=FieldCacheInjection(target="request", path="metadata.marker"),
        allow_missing_session=True,
        scope=("provider", "model"),
    )


def _response_source_rule() -> FieldCacheRule:
    return FieldCacheRule(
        name="response_marker",
        source="response",
        path="choices.0.message.reasoning_content",
        inject=FieldCacheInjection(target="request", path="metadata.cached_reasoning"),
        allow_missing_session=True,
        scope=("provider", "model"),
    )


_REQUEST_WITH_MARKER = {
    "model": "gpt-test",
    "messages": [{"role": "user", "content": "hello", "marker": "from-request"}],
}


@pytest.mark.asyncio
async def test_request_source_extraction_off_by_default() -> None:
    context = _context(field_cache_rules=(_request_source_rule(),))
    executor = NativeProviderExecutor()

    await executor.execute(
        _REQUEST_WITH_MARKER,
        context,
        NativeHTTPTransport(_FakeClient(_chat_response())),
    )
    second_client = _FakeClient(_chat_response())
    await executor.execute(_REQUEST_WITH_MARKER, context, NativeHTTPTransport(second_client))

    # Toggle off: the request pass never writes, so nothing is replayed.
    assert "marker" not in second_client.calls[0]["json"].get("metadata", {})


@pytest.mark.asyncio
async def test_request_source_extraction_enabled_via_toggle(monkeypatch) -> None:
    monkeypatch.setenv("FIELD_CACHE_REQUEST_EXTRACTION", "1")
    context = _context(field_cache_rules=(_request_source_rule(),))
    executor = NativeProviderExecutor()

    await executor.execute(
        _REQUEST_WITH_MARKER,
        context,
        NativeHTTPTransport(_FakeClient(_chat_response())),
    )
    second_client = _FakeClient(_chat_response())
    await executor.execute(_REQUEST_WITH_MARKER, context, NativeHTTPTransport(second_client))

    assert second_client.calls[0]["json"]["metadata"]["marker"] == "from-request"


@pytest.mark.asyncio
async def test_extraction_flows_from_response_sources_not_request(monkeypatch) -> None:
    monkeypatch.delenv("FIELD_CACHE_REQUEST_EXTRACTION", raising=False)
    context = _context(field_cache_rules=(_request_source_rule(), _response_source_rule()))
    executor = NativeProviderExecutor()

    await executor.execute(
        _REQUEST_WITH_MARKER,
        context,
        NativeHTTPTransport(_FakeClient(_chat_response(reasoning="hidden"))),
    )
    second_client = _FakeClient(_chat_response())
    await executor.execute(_REQUEST_WITH_MARKER, context, NativeHTTPTransport(second_client))

    sent = second_client.calls[0]["json"]
    # Response-side rule fires; request-side never wrote (toggle off).
    assert sent["metadata"]["cached_reasoning"] == "hidden"
    assert "marker" not in sent["metadata"]
