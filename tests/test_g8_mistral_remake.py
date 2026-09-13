# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G8 mistral remake pins.

The provider is a declaration plus one adapter that EXTENDS the generic
param_rules engine: provider/model parameter tables (strip_override
re-admitting reasoning_effort on the four reasoning models), think-chunk
folding for response and stream, history-reasoning stripping, and the
nested seed rename. Reasoning-content preservation runs through the real
field-cache engine (response + stream siblings sharing one store, mode
left at the global ``turn`` default, no placeholder).
"""

from __future__ import annotations

import json
import logging

import pytest

from rotator_library.adapters import AdapterContext, get_adapter, run_adapter_chain
from rotator_library.adapters.mistral import MistralAdapter
from rotator_library.field_cache import (
    FieldCacheContext,
    FieldCacheEngine,
    InMemoryFieldCacheStore,
    build_cache_key,
)
from rotator_library.providers import PROVIDER_PLUGINS
from rotator_library.providers.mistral_provider import MISTRAL_REASONING_MODELS


def _plugin():
    return PROVIDER_PLUGINS["mistral"]()


NON_REASONING_MODEL = "mistral-large-latest"
REASONING_MODEL = "mistral-medium-3-5"


# --- declaration identity --------------------------------------------------------


def test_mistral_declaration_identity() -> None:
    plugin = _plugin()
    assert plugin.provider_env_name == "mistral"
    assert plugin.protocol_name == "openai_chat"
    assert plugin.native_streaming_supported is True
    assert plugin.default_api_base == "https://api.mistral.ai/v1"
    assert plugin.get_native_endpoint(model="m", operation="chat") == "https://api.mistral.ai/v1/chat/completions"
    assert plugin.adapter_names == ("mistral",)
    assert get_adapter("mistral").name == "mistral"


def test_reasoning_models_constant_and_rule_keys_agree() -> None:
    assert MISTRAL_REASONING_MODELS == (
        "mistral-small-latest",
        "mistral-small-2603",
        "mistral-medium-3-5",
        "mistral-medium-2604",
    )
    plugin = _plugin()
    assert plugin.MISTRAL_REASONING_MODELS == MISTRAL_REASONING_MODELS
    assert set(plugin.model_param_rules) == set(MISTRAL_REASONING_MODELS)


def test_retired_handler_and_patterns_are_gone() -> None:
    plugin = _plugin()
    assert not hasattr(plugin, "handle_thinking_parameter")
    assert not hasattr(plugin, "_is_mistral_reasoning")
    assert not hasattr(plugin, "MISTRAL_MODEL_PATTERNS")
    assert not hasattr(plugin, "DISABLE_VALUES")


def test_retired_client_transform_entry_is_gone() -> None:
    from rotator_library.client.transforms import ProviderTransforms

    transforms = ProviderTransforms(provider_plugins={})
    assert "mistral" not in transforms._transforms


def test_adapter_config_exposes_resolved_tables_under_own_key() -> None:
    plugin = _plugin()
    config = plugin.get_adapter_config(REASONING_MODEL)
    rules = config["mistral"]
    # strip_override already resolved into the ordinary strip table
    assert "reasoning_effort" not in rules["strip"]
    assert rules["strip"] == ["logit_bias", "logprobs", "top_logprobs"]
    assert "strip_override" not in rules


# --- declared parameter hygiene (param_rules via the mistral adapter) ------------


async def _adapt(payload: dict, model: str) -> dict:
    plugin = _plugin()
    context = AdapterContext(
        provider="mistral",
        model=model,
        protocol="openai_chat",
        adapter_config=plugin.get_adapter_config(model),
    )
    return await run_adapter_chain([get_adapter("mistral")], payload, context, stage="request")


async def test_provider_level_strip_clamp_rename_on_plain_model() -> None:
    result = await _adapt(
        {
            "model": NON_REASONING_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "medium",
            "logit_bias": {"1": 2},
            "logprobs": True,
            "top_logprobs": 5,
            "temperature": 1.7,
            "n": 3,
            "max_completion_tokens": 512,
        },
        NON_REASONING_MODEL,
    )
    assert "reasoning_effort" not in result
    assert "logit_bias" not in result
    assert "logprobs" not in result
    assert "top_logprobs" not in result
    assert result["temperature"] == 1.0
    assert result["n"] == 1
    assert result["max_tokens"] == 512
    assert "max_completion_tokens" not in result


async def test_non_reasoning_model_strips_reasoning_effort() -> None:
    result = await _adapt(
        {"model": "mistral-small-2410", "messages": [], "reasoning_effort": "high"},
        "mistral-small-2410",
    )
    assert "reasoning_effort" not in result


@pytest.mark.parametrize(
    ("effort", "expected"),
    [
        ("minimal", "high"),
        ("low", "high"),
        ("medium", "high"),
        ("xhigh", "high"),
        ("high", "high"),
        ("none", "none"),
    ],
)
async def test_reasoning_model_effort_table(effort: str, expected: str) -> None:
    for model in MISTRAL_REASONING_MODELS:
        result = await _adapt(
            {"model": model, "messages": [], "reasoning_effort": effort},
            model,
        )
        assert result["reasoning_effort"] == expected, model


async def test_reasoning_model_still_strips_logit_bias_family() -> None:
    result = await _adapt(
        {
            "model": REASONING_MODEL,
            "messages": [],
            "reasoning_effort": "medium",
            "logit_bias": {},
            "logprobs": True,
            "top_logprobs": 2,
        },
        REASONING_MODEL,
    )
    assert result["reasoning_effort"] == "high"
    assert "logit_bias" not in result
    assert "logprobs" not in result
    assert "top_logprobs" not in result


async def test_tool_choice_required_maps_to_any_via_declaration() -> None:
    result = await _adapt(
        {"model": REASONING_MODEL, "messages": [], "tool_choice": "required"},
        REASONING_MODEL,
    )
    assert result["tool_choice"] == "any"
    # object spellings pass through untouched
    result = await _adapt(
        {"model": REASONING_MODEL, "messages": [], "tool_choice": {"type": "function", "function": {"name": "f"}}},
        REASONING_MODEL,
    )
    assert result["tool_choice"] == {"type": "function", "function": {"name": "f"}}


async def test_nothing_sent_nothing_injected() -> None:
    payload = {"model": REASONING_MODEL, "messages": [{"role": "user", "content": "hi"}]}
    result = await _adapt(dict(payload), REASONING_MODEL)
    assert result == payload
    assert "reasoning_effort" not in result
    assert "thinking" not in result
    assert "extra_body" not in result


# --- adapter: think-chunk folding -------------------------------------------------


def _ctx(model: str = REASONING_MODEL) -> AdapterContext:
    return AdapterContext(provider="mistral", model=model, protocol="openai_chat")


def _think_chunk(*texts: str) -> dict:
    return {"type": "thinking", "thinking": [{"type": "text", "text": text} for text in texts]}


async def test_response_think_chunks_fold_into_reasoning_content() -> None:
    adapter = MistralAdapter()
    payload = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": REASONING_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        _think_chunk("step one. ", "step two."),
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": " there"},
                    ],
                },
                "finish_reason": "stop",
            }
        ],
    }
    result = await adapter.transform_response(payload, _ctx())
    message = result["choices"][0]["message"]
    assert message["reasoning_content"] == "step one. step two."
    assert message["content"] == "Hello there"
    # input payload untouched (adapter copies before editing)
    assert isinstance(payload["choices"][0]["message"]["content"], list)


async def test_response_plain_string_content_passthrough() -> None:
    adapter = MistralAdapter()
    payload = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "plain"}, "finish_reason": "stop"}
        ],
    }
    result = await adapter.transform_response(payload, _ctx())
    assert result is payload


async def test_unknown_response_chunk_rides_as_text() -> None:
    adapter = MistralAdapter()
    payload = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "mystery", "text": "weird"},
                        {"type": "opaque", "payload": {"kept": True}},
                    ],
                },
            }
        ]
    }
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("rotator_library.adapters")
    handler = _Capture(level=logging.INFO)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        result = await adapter.transform_response(payload, _ctx())
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    message = result["choices"][0]["message"]
    # never dropped: known-text chunks ride as text, shapeless ones as their
    # JSON projection, both with an info line
    assert message["content"].startswith("weird")
    assert '"kept": true' in message["content"]
    assert sum("unknown content chunk" in record.getMessage() for record in records) == 2
    assert "reasoning_content" not in message


async def test_stream_chunk_think_folding() -> None:
    adapter = MistralAdapter()
    payload = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": [_think_chunk("pondering "), {"type": "text", "text": "answer"}]
                },
            }
        ],
    }
    result = await adapter.transform_stream_event(payload, _ctx())
    delta = result["choices"][0]["delta"]
    assert delta["reasoning_content"] == "pondering "
    assert delta["content"] == "answer"


async def test_stream_chunk_plain_string_passthrough() -> None:
    adapter = MistralAdapter()
    payload = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "plain"}}],
    }
    result = await adapter.transform_stream_event(payload, _ctx())
    assert result is payload


async def test_stream_neutral_event_think_folding_reaches_chat_clients() -> None:
    from rotator_library.protocols.openai_chat import OpenAIChatProtocol
    from rotator_library.protocols.streaming import format_canonical_stream_event, stream_format_state
    from rotator_library.protocols.types import ProtocolContext

    adapter = MistralAdapter()
    protocol = OpenAIChatProtocol()
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": [_think_chunk("hidden thought"), {"type": "text", "text": "visible"}]
                },
            }
        ],
    }
    event = protocol.parse_stream_events(chunk)[0]
    adapted = await adapter.transform_stream_event(event, _ctx())
    # neutral delta carries the folded reasoning blocks
    assert [(block.type, block.text) for block in adapted.delta.reasoning] == [
        ("reasoning_content", "hidden thought")
    ]
    assert [(block.type, block.text) for block in adapted.delta.content] == [("text", "visible")]
    # ... and a chat client's wire frame shows the chat-family spelling
    context = ProtocolContext(provider="mistral", model=REASONING_MODEL)
    state = stream_format_state(context, "openai_chat")
    frames = format_canonical_stream_event(adapted, "openai_chat", context, state=state)
    delta = json.loads(frames[0][len("data: ") :])["choices"][0]["delta"]
    assert delta["reasoning_content"] == "hidden thought"
    assert delta["content"] == "visible"


async def test_stream_neutral_event_text_only_identity() -> None:
    from rotator_library.protocols.openai_chat import OpenAIChatProtocol

    adapter = MistralAdapter()
    protocol = OpenAIChatProtocol()
    chunk = {
        "choices": [{"index": 0, "delta": {"content": [{"type": "text", "text": "just text"}]}}]
    }
    event = protocol.parse_stream_events(chunk)[0]
    result = await adapter.transform_stream_event(event, _ctx())
    assert result is event


# --- adapter: request specifics ---------------------------------------------------


async def test_history_reasoning_fields_stripped_from_request() -> None:
    payload = {
        "model": REASONING_MODEL,
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "a",
                "reasoning_content": "replayed thought",
                "thinking_blocks": [{"type": "thinking", "thinking": []}],
            },
            {"role": "user", "content": "next"},
        ],
    }
    result = await _adapt(dict(payload), REASONING_MODEL)
    assert result["messages"][1] == {"role": "assistant", "content": "a"}
    assert result["messages"][0] == {"role": "user", "content": "q"}
    assert result["messages"][2] == {"role": "user", "content": "next"}


async def test_seed_moves_into_extra_body_random_seed() -> None:
    result = await _adapt(
        {"model": REASONING_MODEL, "messages": [], "seed": 42},
        REASONING_MODEL,
    )
    assert "seed" not in result
    assert result["extra_body"]["random_seed"] == 42
    # nothing injected when no seed was sent
    result = await _adapt({"model": REASONING_MODEL, "messages": []}, REASONING_MODEL)
    assert "extra_body" not in result


# --- reasoning cache through the real field-cache engine --------------------------


def _rules():
    response_rule, stream_rule = _plugin().field_cache_rules
    assert response_rule.cache_key == stream_rule.cache_key == "mistral_reasoning"
    return response_rule, stream_rule


def _cache_context(**overrides) -> FieldCacheContext:
    base = dict(
        provider="mistral",
        model=REASONING_MODEL,
        credential_id="cred-1",
        session_id="session-1",
    )
    base.update(overrides)
    return FieldCacheContext(**base)


def _response(content: str, reasoning: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": REASONING_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
                "finish_reason": "stop",
            }
        ],
    }


def test_sibling_rules_share_store_and_default_turn_mode() -> None:
    response_rule, stream_rule = _rules()
    context = _cache_context()
    assert build_cache_key(response_rule, context) == build_cache_key(stream_rule, context)
    # mode UNDECLATED: the global default ("turn") is the declaration
    assert response_rule.mode == stream_rule.mode == "turn"
    # no placeholder: no 400-on-missing contract to satisfy
    assert response_rule.placeholder is None
    assert stream_rule.placeholder is None
    assert response_rule.inject.when_missing_only is True
    assert stream_rule.inject.path == "messages.*.reasoning_content"
    # the shared-signature contract must hold for the real engine
    FieldCacheEngine([response_rule, stream_rule])


async def test_reasoning_round_trip_current_turn_only() -> None:
    response_rule, stream_rule = _rules()
    engine = FieldCacheEngine([response_rule, stream_rule])
    context = _cache_context()

    await engine.extract("response", _response("turn one", "reasoning one"), context)
    await engine.extract("response", _response("turn two", "reasoning two"), context)

    request = {
        "model": REASONING_MODEL,
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "turn one"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "turn two"},
        ],
    }
    updated, operations = await engine.inject("request", request, context)

    reasoning_operation = next(op for op in operations if op.rule_name == "reasoning")
    assert reasoning_operation.hit is True
    assert reasoning_operation.changed is True
    # mode=turn: only the latest region's assistant message is touched
    assert updated["messages"][3]["reasoning_content"] == "reasoning two"
    assert "reasoning_content" not in updated["messages"][1]
    # user messages are never injected
    assert "reasoning_content" not in updated["messages"][0]
    assert "reasoning_content" not in updated["messages"][2]


async def test_client_carried_reasoning_survives_when_missing_only() -> None:
    response_rule, stream_rule = _rules()
    engine = FieldCacheEngine([response_rule, stream_rule])
    context = _cache_context()
    await engine.extract("response", _response("same", "cached"), context)

    request = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "same", "reasoning_content": "client-provided"},
        ]
    }
    updated, _ = await engine.inject("request", request, context)
    assert updated["messages"][1]["reasoning_content"] == "client-provided"


async def test_miss_leaves_message_clean_no_placeholder() -> None:
    response_rule, _ = _rules()
    engine = FieldCacheEngine([response_rule])
    context = _cache_context()
    await engine.extract("response", _response("known", "cached"), context)

    request = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "never seen"},
        ]
    }
    updated, operations = await engine.inject("request", request, context)
    assert "reasoning_content" not in updated["messages"][1]
    assert updated["messages"][1]["content"] == "never seen"
    assert operations[0].skipped is True


async def test_stream_sibling_writes_the_same_store() -> None:
    from rotator_library.adapters.mistral import MistralAdapter as Adapter
    from rotator_library.protocols.openai_chat import OpenAIChatProtocol

    response_rule, stream_rule = _rules()
    store = InMemoryFieldCacheStore()
    context = _cache_context()
    engine = FieldCacheEngine([response_rule, stream_rule], store=store)

    # One streamed reasoning fragment: the neutral event after the mistral
    # adapter folded the think-chunk (extraction sees the serialized form).
    protocol = OpenAIChatProtocol()
    chunk = {
        "choices": [
            {"index": 0, "delta": {"content": [_think_chunk("streamed thought")]}}
        ]
    }
    event = protocol.parse_stream_events(chunk)[0]
    adapted = await Adapter().transform_stream_event(event, _ctx())
    operations = await engine.extract("stream_event", adapted.to_dict(), context)
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


async def test_model_listing_parses_provider_ids() -> None:
    plugin = _plugin()
    client = _FakeClient(
        body={"data": [{"id": "mistral-medium-3-5"}, {"id": "mistral-small-latest"}, {"object": "model"}]}
    )
    models = await plugin.get_models("sk-test", client)
    assert models == ["mistral/mistral-medium-3-5", "mistral/mistral-small-latest"]
    assert client.calls[0] == "https://api.mistral.ai/v1/models"
