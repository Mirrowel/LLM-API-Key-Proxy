# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Field-addressed cache rules (G8): one rule, every path-addressable face.

A rule declaring ``field="reasoning"`` resolves its effective path/inject/
metadata from ``protocols.defaults.FIELD_LOCATIONS`` for the payload's
protocol family — no hand-wired paths, and explicit declarations still
override any single registry slot.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _ctx(family, provider="prov", model="model-x", session="s1"):
    from rotator_library.field_cache import FieldCacheContext

    return FieldCacheContext(
        provider=provider,
        model=model,
        session_id=session,
        protocol_family=family,
    )


def _rule(**overrides):
    from rotator_library.field_cache import FieldCacheRule

    sources = overrides.pop("sources", None)
    source = overrides.pop("source", "response" if sources is None else None)
    return FieldCacheRule(
        name=overrides.pop("name", "reasoning"),
        source=source,
        sources=sources,
        field=overrides.pop("field", "reasoning"),
        mode="all",
        **overrides,
    )


OPENAI_RESPONSE = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "think-a",
                "tool_calls": [{"id": "call_a", "type": "function"}],
            }
        }
    ]
}

OPENAI_STREAM_EVENT = {
    "raw": {
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "think-b",
                }
            }
        ]
    }
}


def test_field_rule_resolves_openai_chat_family_paths():
    """One field-addressed rule resolves response + stream + inject +
    correlation slots for the openai_chat family and round-trips."""

    from rotator_library.field_cache import FieldCacheEngine

    engine = FieldCacheEngine([_rule(sources=("response", "stream_event"))])
    context = _ctx("openai_chat")

    response_ops = asyncio.run(engine.extract("response", OPENAI_RESPONSE, context))
    response_op = next(op for op in response_ops if op.matched)
    assert response_op.changed is True
    assert response_op.sample_values and "think-a" in str(response_op.sample_values)

    stream_ops = asyncio.run(engine.extract("stream_event", OPENAI_STREAM_EVENT, context))
    stream_op = next(op for op in stream_ops if op.matched)
    assert stream_op.changed is True
    assert "think-b" in str(stream_op.sample_values)

    request = {
        "messages": [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "answer",
                "tool_calls": [{"id": "call_a", "type": "function"}],
            },
        ]
    }
    injected, inject_ops = asyncio.run(engine.inject("request", request, context))
    op = next(o for o in inject_ops if o.rule_name.endswith("response"))
    assert op.hit is True and op.changed is True
    # inject_path resolved to messages.*.reasoning_content; the occurrence
    # correlated by its tool-call id (tool_call_id_path slot).
    assert injected["messages"][1]["reasoning_content"] == "think-a"


def test_sources_twins_share_one_store():
    """sources=("response", "stream_event") expands to sibling rules that
    write ONE cache entry map under one shared cache key."""

    from rotator_library.field_cache import FieldCacheEngine, build_cache_key

    rule = _rule(sources=("response", "stream_event"))
    engine = FieldCacheEngine([rule])
    context = _ctx("openai_chat")

    asyncio.run(engine.extract("response", OPENAI_RESPONSE, context))
    asyncio.run(engine.extract("stream_event", OPENAI_STREAM_EVENT, context))

    shared_key = build_cache_key(rule, context)
    assert shared_key is not None
    stored = asyncio.run(engine.store.get(shared_key))
    assert isinstance(stored, dict) and len(stored) >= 2
    values = [entry.get("value") for entry in stored.values()]
    assert "think-a" in values and "think-b" in values


def test_explicit_declarations_override_registry_slots():
    """Explicit path/inject/metadata declarations beat the registry slot
    for that one position (override-everything contract)."""

    from rotator_library.field_cache import FieldCacheEngine, FieldCacheInjection, build_cache_key

    engine = FieldCacheEngine(
        [
            _rule(
                path="choices.0.message.custom_reasoning",
                inject=FieldCacheInjection(target="request", path="messages.*.custom_reasoning"),
                metadata={"tool_call_id_path": "alt_calls.*.id"},
            )
        ]
    )
    context = _ctx("openai_chat")
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "registry-value",
                    "custom_reasoning": "explicit-value",
                    "alt_calls": [{"id": "alt_1"}],
                }
            }
        ]
    }
    ops = asyncio.run(engine.extract("response", payload, context))
    op = ops[0]
    assert op.matched >= 1  # correlation entries (alt id + content sha)
    assert "explicit-value" in str(op.sample_values)
    assert "registry-value" not in str(op.sample_values)

    stored = asyncio.run(engine.store.get(build_cache_key(engine.rules[0], context)))
    assert "alt_1" in stored  # metadata tool_call_id_path overrode the slot

    request = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "answer", "alt_calls": [{"id": "alt_1"}]},
        ]
    }
    injected, _ = asyncio.run(engine.inject("request", request, context))
    # Explicit inject path + explicit correlation metadata governed; the
    # registry slots (reasoning_content / tool_calls) never fired.
    assert injected["messages"][1]["custom_reasoning"] == "explicit-value"
    assert "reasoning_content" not in injected["messages"][1]


def test_unknown_field_raises_at_engine_entry():
    from rotator_library.field_cache import FieldCacheEngine

    try:
        FieldCacheEngine([_rule(field="watson")])
    except ValueError as exc:
        assert "watson" in str(exc)
        assert "reasoning" in str(exc)  # names the legal fields
    else:
        raise AssertionError("unknown field must raise at engine entry")


def test_family_without_locations_raises_naming_field():
    from rotator_library.field_cache import FieldCacheEngine

    engine = FieldCacheEngine([_rule(field="signature")])  # openai_chat only
    try:
        asyncio.run(engine.extract("response", {"choices": []}, _ctx("ollama")))
    except ValueError as exc:
        assert "signature" in str(exc)
        assert "ollama" in str(exc)
    else:
        raise AssertionError("missing family locations must raise")


def test_missing_family_on_context_raises():
    from rotator_library.field_cache import FieldCacheEngine

    engine = FieldCacheEngine([_rule()])
    try:
        asyncio.run(engine.extract("response", OPENAI_RESPONSE, _ctx(None)))
    except ValueError as exc:
        assert "protocol family" in str(exc)
    else:
        raise AssertionError("field rule without a family must raise")


def test_same_rule_serves_ollama_family():
    """The same field rule serves a second protocol family through the
    ollama locations (message.thinking shapes)."""

    from rotator_library.field_cache import FieldCacheEngine

    engine = FieldCacheEngine([_rule(sources=("response", "stream_event"))])
    context = _ctx("ollama")

    ollama_response = {
        "model": "llama-x",
        "message": {"role": "assistant", "content": "answer", "thinking": "olla-think"},
        "done": True,
    }
    ollama_stream = {"raw": {"message": {"thinking": "olla-stream"}}}

    response_ops = asyncio.run(engine.extract("response", ollama_response, context))
    assert any(op.matched and "olla-think" in str(op.sample_values) for op in response_ops)
    stream_ops = asyncio.run(engine.extract("stream_event", ollama_stream, context))
    assert any(op.matched and "olla-stream" in str(op.sample_values) for op in stream_ops)

    # Derived inject path: messages.*.thinking. The assistant occurrence
    # correlates by content sha (plain-path extraction keys values by the
    # sha of the extracted content itself).
    request = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "olla-think"}]}
    injected, ops = asyncio.run(engine.inject("request", request, context))
    assert injected["messages"][1]["thinking"] == "olla-think"
    assert any(op.changed for op in ops)


def test_native_context_fills_protocol_family():
    from rotator_library.native_provider.context import NativeProviderContext

    context = NativeProviderContext(
        provider="prov",
        model="model-x",
        protocol_name="openai_chat",
        endpoint="/chat/completions",
    )
    assert context.field_cache_context().protocol_family == "openai_chat"

    variant = NativeProviderContext(
        provider="prov",
        model="model-x",
        protocol_name="responses_stateful",
        endpoint="/responses",
    )
    # Variant resolves to its wire family.
    assert variant.field_cache_context().protocol_family == "responses"


def test_field_rule_without_family_slot_source_errors_at_entry():
    """A field rule watching requests (no registry slot) must declare an
    explicit path — startup-style error, not a silent no-op."""

    from rotator_library.field_cache import FieldCacheEngine

    try:
        FieldCacheEngine([_rule(source="request")])
    except ValueError as exc:
        assert "request" in str(exc)
    else:
        raise AssertionError("request-source field rule without a path must raise")


def test_single_source_spelling_stays_compatible():
    """source= (single) keeps working unchanged alongside the new surface."""

    from rotator_library.field_cache import FieldCacheEngine

    engine = FieldCacheEngine([_rule(source="response")])
    context = _ctx("openai_chat")
    ops = asyncio.run(engine.extract("response", OPENAI_RESPONSE, context))
    assert any(op.changed for op in ops)
    # No twin rename for the single-source spelling.
    assert [rule.name for rule in engine.rules] == ["reasoning"]
