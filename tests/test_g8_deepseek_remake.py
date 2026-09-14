# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G8 deepseek remake pins (final envelope).

The provider is a pure declaration: three ``speaks`` faces, the
``model_rules`` capability cascade for parameter hygiene, and
reasoning-content preservation through ONE field-addressed cache rule
(response + stream twins expanded by the engine, paths resolved from the
protocol registry). No custom execution path exists anymore.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from rotator_library.adapters import AdapterContext, get_adapter, run_adapter_chain
from rotator_library.field_cache import (
    FieldCacheContext,
    FieldCacheEngine,
    InMemoryFieldCacheStore,
    build_cache_key,
)
from rotator_library.providers import PROVIDER_PLUGINS


def _plugin():
    return PROVIDER_PLUGINS["deepseek"]()


def _rule():
    """The one field-addressed rule (engine expands it to the twins)."""

    (rule,) = _plugin().field_cache_rules
    assert rule.field == "reasoning"
    # No declared key/TTL: the engine derives provider:field and the
    # store's 3-day inactivity default owns retention.
    assert rule.cache_key is None
    assert rule.ttl_seconds is None
    return rule


def _response_twin(rule):
    """The response sibling the engine itself expands from ``sources``."""

    return replace(rule, source="response", sources=None)


# --- three-face speaks resolution ----------------------------------------------


def test_default_face_is_chat_completions() -> None:
    plugin = _plugin()
    assert plugin.get_protocol_name("deepseek-v4-pro") == "openai_chat"
    assert (
        plugin.get_native_endpoint(model="deepseek-v4-pro", operation="chat")
        == "https://api.deepseek.com/chat/completions"
    )


def test_responses_profile_resolves_responses_face() -> None:
    plugin = _plugin()
    assert plugin.get_protocol_name("m", profile="responses") == "responses"
    assert (
        plugin.get_native_endpoint(model="m", operation="responses", profile="responses")
        == "https://api.deepseek.com/responses"
    )


def test_anthropic_profile_resolves_anthropic_face() -> None:
    plugin = _plugin()
    assert plugin.get_protocol_name("m", profile="anthropic_messages") == "anthropic_messages"
    assert (
        plugin.get_native_operation("m", None, stream=False, profile="anthropic_messages")
        == "messages"
    )
    assert (
        plugin.get_native_endpoint(model="m", operation="messages", profile="anthropic_messages")
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        plugin.get_native_endpoint(
            model="m", operation="count_tokens", profile="anthropic_messages"
        )
        == "https://api.deepseek.com/anthropic/v1/messages/count_tokens"
    )
    assert plugin.supports_native_operation("m", "messages", profile="anthropic_messages") is True
    assert plugin.supports_native_operation("m", "chat", profile="anthropic_messages") is False


def test_speaks_is_the_transport_declaration() -> None:
    """The envelope replaces the legacy wiring: speaks resolves three faces,
    the hand-rolled tables are gone, and the anthropic-compatibility face
    inherits the protocol's conventional x-api-key auth."""
    plugin = _plugin()
    assert plugin.transport_profiles is None
    assert plugin.protocol_name is None
    assert plugin.default_profile is None
    profiles = plugin._speaks_profiles()
    assert set(profiles) - {"__default__"} == {"openai_chat", "responses", "anthropic_messages"}
    assert profiles["__default__"] is profiles["openai_chat"]
    assert plugin.get_native_headers("sk-test", profile="anthropic_messages") == {
        "x-api-key": "sk-test"
    }
    assert plugin.get_native_headers("sk-test") == {"Authorization": "Bearer sk-test"}


def test_transport_base_has_no_v1_suffix_and_env_overrides(monkeypatch) -> None:
    plugin = _plugin()
    assert plugin.get_provider_api_base() == "https://api.deepseek.com"
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://mirror.example/prefix")
    assert plugin.get_provider_api_base() == "https://mirror.example/prefix"
    assert (
        plugin.get_native_endpoint(model="m", operation="chat")
        == "https://mirror.example/prefix/chat/completions"
    )


def test_retired_custom_path_is_gone() -> None:
    from rotator_library.providers.deepseek_provider import DeepseekProvider

    plugin = _plugin()
    assert plugin.has_custom_logic() is False
    assert plugin.adapter_names == ()  # param engine is always-on, not declared
    assert plugin.get_adapter_names("deepseek/deepseek-v4-flash") == ("param_rules",)
    assert not hasattr(plugin, "_get_reasoning_cache")
    # Listing is the inherited shared implementation, not provider code.
    assert "get_models" not in vars(DeepseekProvider)
    assert plugin.native_streaming_supported is True


# --- declared parameter hygiene (model_rules cascade) ---------------------------


async def _adapt(payload: dict, model: str) -> dict:
    plugin = _plugin()
    config = plugin.get_adapter_config(model)
    context = AdapterContext(
        provider="deepseek",
        model=model,
        protocol="openai_chat",
        adapter_config=config,
    )
    return await run_adapter_chain([get_adapter("param_rules")], payload, context, stage="request")


@pytest.mark.asyncio
async def test_max_completion_tokens_renamed_via_declaration() -> None:
    result = await _adapt(
        {"model": "deepseek-v4-pro", "messages": [], "max_completion_tokens": 128},
        "deepseek-v4-pro",
    )
    assert result["max_tokens"] == 128
    assert "max_completion_tokens" not in result


def test_capability_declarations_replace_the_effort_map() -> None:
    """The model_rules rows carry effort_accept/toggle; no vocabulary map
    survives anywhere in the declaration."""

    plugin = _plugin()
    assert plugin.reasoning_effort_accept == ("off", "low", "medium", "high", "max")
    assert plugin.reasoning_effort_toggle is True
    assert tuple(row["match"] for row in plugin.model_rules) == (
        "*",
        "deepseek-v4-pro-08*",
        "deepseek-v4-flash-08*",
    )
    assert not any("effort_map" in row for row in plugin.model_rules)
    for row in plugin.model_rules[1:]:
        assert row["effort_accept"] == ["off", "low", "high", "max"]
        assert row["toggle"] is True


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        # Current models: provider-level native vocabulary.
        ("deepseek-v4-pro", "low", "low"),
        ("deepseek-v4-pro", "medium", "medium"),
        ("deepseek-v4-pro", "xhigh", "high"),
        ("deepseek-v4-pro", "max", "max"),
        # Dated -0813-era snapshots: {off, low, high, max}; the ladder
        # folds medium UP to high (the official tie-up).
        ("deepseek-v4-pro-0813", "medium", "high"),
        ("deepseek-v4-flash-0813", "medium", "high"),
        ("deepseek-v4-pro-0813", "xhigh", "high"),
        ("deepseek-v4-pro-0813", "max", "max"),
        # The flash alias is provider-level, not a dated snapshot.
        ("deepseek-flash", "medium", "medium"),
        ("deepseek-chat", "medium", "medium"),
    ],
)
def test_effort_normalization_rides_the_ladder(model: str, effort: str, expected: str) -> None:
    from rotator_library.protocols.effort import normalize_effort, resolve_accepted_effort

    accepted, source = resolve_accepted_effort(
        _plugin(), model, protocol_family="openai_chat"
    )
    assert source != "protocol_base"  # a real declaration decided
    assert normalize_effort(effort, accepted)[0] == expected


def test_dated_snapshot_ids_match_the_wildcard() -> None:
    """The ``deepseek-v4-*-08*`` rows shrink the dated -0813-era ids the
    provider-level vocabulary; the ladder still yields the official fold."""

    from rotator_library.protocols.effort import normalize_effort, resolve_accepted_effort

    accepted, source = resolve_accepted_effort(
        _plugin(), "deepseek-v4-pro-0813", protocol_family="openai_chat"
    )
    assert accepted == ("off", "low", "high", "max")
    assert source == "model_rules:deepseek-v4-pro-08*"
    assert normalize_effort("medium", accepted)[0] == "high"


@pytest.mark.asyncio
async def test_nothing_sent_nothing_injected() -> None:
    payload = {"model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "hi"}]}
    result = await _adapt(dict(payload), "deepseek-v4-pro")
    assert result == payload
    assert "reasoning_effort" not in result
    assert "thinking" not in result
    assert "max_tokens" not in result


# --- reasoning cache through the real field-cache engine ----------------------


def _context(**overrides) -> FieldCacheContext:
    base = dict(
        provider="deepseek",
        model="deepseek-v4-pro",
        credential_id="cred-1",
        session_id="session-1",
        protocol_family="openai_chat",
    )
    base.update(overrides)
    return FieldCacheContext(**base)


def _tool_call(call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{}"},
    }


def _assistant_turn(content: str, call_id: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": [_tool_call(call_id)]}


def _response(content: str, call_id: str, reasoning: str) -> dict:
    message = {
        "role": "assistant",
        "content": content,
        "reasoning_content": reasoning,
        "tool_calls": [_tool_call(call_id)],
    }
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "deepseek-v4-pro",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
    }


def test_field_addressed_rule_shape_and_twin_expansion() -> None:
    rule = _rule()
    context = _context()
    assert rule.sources == ("response", "stream_event")
    assert rule.mode == "all"
    assert rule.placeholder == "Reasoning content unavailable."
    assert rule.ttl_seconds is None
    assert rule.inject.when_missing_only is True
    assert rule.inject.target == "request"
    assert rule.inject.path == ""  # registry derivation owns the path
    assert build_cache_key(rule, context) is not None
    # The shared-signature contract must hold for the expanded twins and
    # both twins auto-derive the SAME key (field + provider).
    engine = FieldCacheEngine([rule])
    assert all(twin.cache_key is None for twin in engine._expanded_rules)
    assert build_cache_key(engine._expanded_rules[0], context) == build_cache_key(
        engine._expanded_rules[1], context
    )
    # Correlation (tool_call_id_path) and paths derive from the registry.
    from rotator_library.protocols.defaults import field_locations

    slots = field_locations("reasoning", "openai_chat")
    assert slots["tool_call_id_path"] == "tool_calls.*.id"
    assert slots["inject_path"] == "messages.*.reasoning_content"


@pytest.mark.asyncio
async def test_reasoning_round_trip_all_history_default() -> None:
    rule = _rule()
    store = InMemoryFieldCacheStore()
    engine = FieldCacheEngine([rule], store=store)
    context = _context()

    await engine.extract("response", _response("turn one", "call_1", "reasoning one"), context)
    await engine.extract("response", _response("turn two", "call_2", "reasoning two"), context)

    request = {
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "user", "content": "q1"},
            _assistant_turn("turn one", "call_1"),
            {"role": "tool", "tool_call_id": "call_1", "content": "tool output"},
            {"role": "user", "content": "q2"},
            _assistant_turn("turn two", "call_2"),
        ],
    }
    updated, operations = await engine.inject("request", request, context)

    reasoning_operation = next(op for op in operations if op.rule_name.startswith("reasoning"))
    assert reasoning_operation.hit is True
    assert reasoning_operation.changed is True
    # The derivation is visible on the operation: the store key is the
    # provider:field-derived identity shared by both twins.
    assert reasoning_operation.cache_key == build_cache_key(rule, context)
    assert updated["messages"][1]["reasoning_content"] == "reasoning one"
    assert updated["messages"][4]["reasoning_content"] == "reasoning two"
    # when_missing_only: user/tool messages are never touched.
    assert "reasoning_content" not in updated["messages"][0]
    assert "reasoning_content" not in updated["messages"][2]


@pytest.mark.asyncio
async def test_reasoning_correlates_by_tool_call_id() -> None:
    rule = _rule()
    engine = FieldCacheEngine([rule])
    context = _context()

    # Identical content: only the tool-call id can distinguish the values.
    await engine.extract("response", _response("same", "call_a", "reasoning a"), context)
    await engine.extract("response", _response("same", "call_b", "reasoning b"), context)

    request = {
        "messages": [
            {"role": "user", "content": "q"},
            _assistant_turn("same", "call_a"),
            _assistant_turn("same", "call_b"),
        ]
    }
    updated, _ = await engine.inject("request", request, context)
    assert updated["messages"][1]["reasoning_content"] == "reasoning a"
    assert updated["messages"][2]["reasoning_content"] == "reasoning b"


@pytest.mark.asyncio
async def test_stream_twin_writes_the_same_store() -> None:
    rule = _rule()
    store = InMemoryFieldCacheStore()
    context = _context()
    engine = FieldCacheEngine([rule], store=store)

    # One streamed reasoning fragment: the serialized neutral event carries
    # the provider chunk under ``raw`` (the registry stream slot).
    event = {
        "type": "message_delta",
        "raw": {"choices": [{"index": 0, "delta": {"reasoning_content": "streamed thought"}}]},
    }
    operations = await engine.extract("stream_event", event, context)
    assert operations[0].changed is True

    # Both twins share ONE store entry map under ONE cache key.
    stored = await store.get(build_cache_key(rule, context))
    assert isinstance(stored, dict) and "streamed thought" in str(stored.values())

    # The response twin reads the same store: an occurrence whose content
    # matches the fragment sha restores it.
    request = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "streamed thought"},
        ]
    }
    response_only_engine = FieldCacheEngine([_response_twin(rule)], store=store)
    updated, operations = await response_only_engine.inject("request", request, context)
    assert operations[0].hit is True
    assert updated["messages"][1]["reasoning_content"] == "streamed thought"


@pytest.mark.asyncio
async def test_turn_scope_override_injects_only_latest_region() -> None:
    rule = _rule()
    store = InMemoryFieldCacheStore()
    context = _context()
    engine = FieldCacheEngine([rule], store=store)

    await engine.extract("response", _response("turn one", "call_1", "reasoning one"), context)
    await engine.extract("response", _response("turn two", "call_2", "reasoning two"), context)

    request = {
        "messages": [
            {"role": "user", "content": "q1"},
            _assistant_turn("turn one", "call_1"),
            {"role": "user", "content": "q2"},
            _assistant_turn("turn two", "call_2"),
        ]
    }

    # A config-style override rebuilding the rule with mode="turn" narrows
    # injection to the latest region only; the store is untouched.
    turned = replace(rule, mode="turn")
    assert turned.mode == "turn"
    updated, operations = await FieldCacheEngine([turned], store=store).inject(
        "request", request, context
    )
    assert operations[0].hit is True
    assert updated["messages"][1].get("reasoning_content") is None
    assert updated["messages"][3]["reasoning_content"] == "reasoning two"


@pytest.mark.asyncio
async def test_placeholder_when_occurrence_does_not_correlate() -> None:
    rule = _response_twin(_rule())
    engine = FieldCacheEngine([rule])
    context = _context()
    # The store holds reasoning for one conversation turn; the request also
    # carries an assistant message the cache has never seen.
    await engine.extract("response", _response("known", "call_known", "cached"), context)
    request = {
        "messages": [
            {"role": "user", "content": "q"},
            _assistant_turn("known", "call_known"),
            _assistant_turn("never seen", "call_unknown"),
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

    assert updated["messages"][1]["reasoning_content"] == "cached"
    assert updated["messages"][2]["reasoning_content"] == "Reasoning content unavailable."
    warnings = [
        record
        for record in records
        if record.levelno == logging.WARNING and "placeholder" in record.getMessage()
    ]
    # Exactly one warning: the stream twin sees the placeholder already
    # present (when_missing_only) and never re-warns.
    assert len(warnings) == 1
    assert operations[0].skipped is False


@pytest.mark.asyncio
async def test_existing_client_reasoning_is_preserved() -> None:
    rule = _rule()
    engine = FieldCacheEngine([rule])
    context = _context()
    await engine.extract("response", _response("same", "call_1", "cached"), context)

    request = {
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "same",
                "tool_calls": [_tool_call("call_1")],
                "reasoning_content": "client-provided",
            },
        ]
    }
    updated, _ = await engine.inject("request", request, context)
    assert updated["messages"][1]["reasoning_content"] == "client-provided"


# --- model listing (shared interface implementation) ----------------------------


class _FakeResponse:
    def __init__(self, body: dict, status: int = 200):
        self._body = body
        self.status_code = status

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class _FakeClient:
    def __init__(self, body=None, error: Exception | None = None):
        self._body = body
        self._error = error
        self.calls: list[str] = []

    async def get(self, url, headers=None, **kwargs):
        self.calls.append(url)
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._body)


@pytest.mark.asyncio
async def test_model_listing_via_shared_implementation() -> None:
    plugin = _plugin()
    client = _FakeClient(
        body={"data": [{"id": "deepseek-v4-pro"}, {"id": "deepseek-v4-flash"}, {"object": "model"}]}
    )
    models = await plugin.get_models("sk-test", client)
    assert models == ["deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash"]
    # Listing face resolves from speaks (openai_chat, first in priority).
    assert client.calls[0] == "https://api.deepseek.com/models"


@pytest.mark.asyncio
async def test_failed_listing_is_an_honest_empty() -> None:
    plugin = _plugin()
    client = _FakeClient(error=RuntimeError("network down"))
    models = await plugin.get_models("sk-test", client)
    # No hardcoded fallback anymore: a failed listing is an honest empty.
    assert models == []
