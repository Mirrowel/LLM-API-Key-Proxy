# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G8 deepseek remake pins.

The provider is a pure declaration: three transport faces, param_rules +
model_param_rules parameter hygiene, and reasoning-content preservation
through the real field-cache engine (response + stream siblings sharing
one store). No custom execution path exists anymore.
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
from rotator_library.providers.deepseek_provider import HARDCODED_MODELS


def _plugin():
    return PROVIDER_PLUGINS["deepseek"]()


# --- three-face profile resolution ------------------------------------------


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
    assert plugin.get_protocol_name("m", profile="anthropic") == "anthropic_messages"
    assert plugin.get_native_operation("m", None, stream=False, profile="anthropic") == "messages"
    assert (
        plugin.get_native_endpoint(model="m", operation="messages", profile="anthropic")
        == "https://api.deepseek.com/anthropic/v1/messages"
    )
    assert (
        plugin.get_native_endpoint(model="m", operation="count_tokens", profile="anthropic")
        == "https://api.deepseek.com/anthropic/v1/messages/count_tokens"
    )
    assert plugin.supports_native_operation("m", "messages", profile="anthropic") is True
    assert plugin.supports_native_operation("m", "chat", profile="anthropic") is False


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
    plugin = _plugin()
    assert plugin.has_custom_logic() is False
    assert plugin.adapter_names == ("param_rules",)
    assert not hasattr(plugin, "_get_reasoning_cache")
    assert plugin.native_streaming_supported is True


# --- declared parameter hygiene (param_rules + model_param_rules) -------------


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        ("low", "low"),
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "high"),
        ("max", "max"),
    ],
)
async def test_per_model_effort_mapping(effort: str, expected: str) -> None:
    for model in ("deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"):
        result = await _adapt(
            {"model": model, "messages": [], "reasoning_effort": effort},
            model,
        )
        assert result["reasoning_effort"] == expected, model


@pytest.mark.asyncio
async def test_unknown_model_effort_is_unmapped() -> None:
    result = await _adapt(
        {"model": "deepseek-flash", "messages": [], "reasoning_effort": "medium"},
        "deepseek-flash",
    )
    assert result["reasoning_effort"] == "medium"


@pytest.mark.asyncio
async def test_nothing_sent_nothing_injected() -> None:
    payload = {"model": "deepseek-v4-pro", "messages": [{"role": "user", "content": "hi"}]}
    result = await _adapt(dict(payload), "deepseek-v4-pro")
    assert result == payload
    assert "reasoning_effort" not in result
    assert "thinking" not in result
    assert "max_tokens" not in result


# --- reasoning cache through the real field-cache engine ----------------------


def _rules():
    response_rule, stream_rule = _plugin().field_cache_rules
    assert response_rule.cache_key == stream_rule.cache_key == "deepseek_reasoning"
    return response_rule, stream_rule


def _context(**overrides) -> FieldCacheContext:
    base = dict(
        provider="deepseek",
        model="deepseek-v4-pro",
        credential_id="cred-1",
        session_id="session-1",
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


def test_sibling_rules_share_one_cache_key_and_construct() -> None:
    response_rule, stream_rule = _rules()
    context = _context()
    assert build_cache_key(response_rule, context) == build_cache_key(stream_rule, context)
    assert response_rule.placeholder == stream_rule.placeholder == "Reasoning content unavailable."
    assert response_rule.ttl_seconds == stream_rule.ttl_seconds == 604800
    assert response_rule.metadata["tool_call_id_path"] == "tool_calls.*.id"
    # The shared-signature contract must hold for the real engine.
    FieldCacheEngine([response_rule, stream_rule])


@pytest.mark.asyncio
async def test_reasoning_round_trip_all_history_default() -> None:
    response_rule, stream_rule = _rules()
    store = InMemoryFieldCacheStore()
    engine = FieldCacheEngine([response_rule, stream_rule], store=store)
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

    reasoning_operation = next(op for op in operations if op.rule_name == "reasoning")
    assert reasoning_operation.hit is True
    assert reasoning_operation.changed is True
    assert updated["messages"][1]["reasoning_content"] == "reasoning one"
    assert updated["messages"][4]["reasoning_content"] == "reasoning two"
    # when_missing_only: user/tool messages are never touched.
    assert "reasoning_content" not in updated["messages"][0]
    assert "reasoning_content" not in updated["messages"][2]


@pytest.mark.asyncio
async def test_reasoning_correlates_by_tool_call_id() -> None:
    response_rule, stream_rule = _rules()
    engine = FieldCacheEngine([response_rule, stream_rule])
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
async def test_stream_sibling_writes_the_same_store() -> None:
    response_rule, stream_rule = _rules()
    store = InMemoryFieldCacheStore()
    context = _context()
    engine = FieldCacheEngine([response_rule, stream_rule], store=store)

    # One streamed reasoning fragment: the serialized neutral event carries
    # the provider chunk under ``raw``.
    event = {
        "type": "message_delta",
        "raw": {"choices": [{"index": 0, "delta": {"reasoning_content": "streamed thought"}}]},
    }
    operations = await engine.extract("stream_event", event, context)
    assert operations[0].changed is True

    # The response sibling reads the same store: an occurrence whose content
    # matches the fragment sha restores it.
    request = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "streamed thought"},
        ]
    }
    response_only_engine = FieldCacheEngine([response_rule], store=store)
    updated, operations = await response_only_engine.inject("request", request, context)
    assert operations[0].hit is True
    assert updated["messages"][1]["reasoning_content"] == "streamed thought"


@pytest.mark.asyncio
async def test_turn_scope_override_injects_only_latest_region() -> None:
    response_rule, stream_rule = _rules()
    store = InMemoryFieldCacheStore()
    context = _context()
    engine = FieldCacheEngine([response_rule, stream_rule], store=store)

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
    turned = replace(response_rule, mode="turn")
    assert turned.mode == "turn"
    updated, operations = await FieldCacheEngine([turned], store=store).inject(
        "request", request, context
    )
    assert operations[0].hit is True
    assert updated["messages"][1].get("reasoning_content") is None
    assert updated["messages"][3]["reasoning_content"] == "reasoning two"


@pytest.mark.asyncio
async def test_placeholder_when_occurrence_does_not_correlate() -> None:
    response_rule, _ = _rules()
    engine = FieldCacheEngine([response_rule])
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
    assert len(warnings) == 1
    assert operations[0].skipped is False


@pytest.mark.asyncio
async def test_existing_client_reasoning_is_preserved() -> None:
    response_rule, stream_rule = _rules()
    engine = FieldCacheEngine([response_rule, stream_rule])
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


# --- model listing -------------------------------------------------------------


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
async def test_model_listing_parses_provider_ids() -> None:
    plugin = _plugin()
    client = _FakeClient(
        body={"data": [{"id": "deepseek-v4-pro"}, {"id": "deepseek-v4-flash"}, {"object": "model"}]}
    )
    models = await plugin.get_models("sk-test", client)
    assert models == ["deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash"]
    assert client.calls[0] == "https://api.deepseek.com/models"


@pytest.mark.asyncio
async def test_model_listing_falls_back_to_hardcoded_list() -> None:
    plugin = _plugin()
    client = _FakeClient(error=RuntimeError("network down"))
    models = await plugin.get_models("sk-test", client)
    assert models == [f"deepseek/{model}" for model in HARDCODED_MODELS]
    assert HARDCODED_MODELS == [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v4-flash-vision-exp",
        "deepseek-flash",
    ]
