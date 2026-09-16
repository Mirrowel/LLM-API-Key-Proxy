# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G8 gemini protocol-vs-provider split pins.

The split moves every per-model Gemini fact out of the protocol builders
and into declared ``model_rules`` capability rows:

- ``resolve_model_capabilities`` cascade-resolves the record (dialect,
  budget bounds, tool-call ids, signature strictness, modalities, hosted
  tools, candidate ceiling);
- the native execution seam resolves it ONCE per request (extended with
  the acceptance chain's ``effort_accept``) and threads it to the
  consumers;
- every consumer keeps its EXACT pre-declaration behavior when a key is
  absent — these pins lock that byte-for-byte for None/undeclared.

Layers pinned here: cascade resolution, canonical reasoning emission
(dialect/budget clamp/OFF), Gemini tool-call ids + sentinel strictness
(build, stream, raw strip), validation modalities + hosted tools,
candidateCount clamping, the provider backfill rows, and the deleted
request-sanitizer model list.
"""

from __future__ import annotations

import asyncio
import json

from rotator_library.native_provider import (
    NativeHTTPTransport,
    NativeProviderContext,
    NativeProviderExecutor,
)
from rotator_library.native_provider.effort_emission import resolve_request_capabilities
from rotator_library.protocols import ProtocolContext, get_protocol
from rotator_library.protocols.canonical import format_reasoning_controls
from rotator_library.protocols.effort import resolve_model_capabilities
from rotator_library.protocols.opaque_strip import strip_foreign_opaque_state
from rotator_library.protocols.streaming import format_canonical_stream_event, stream_format_state
from rotator_library.protocols.types import ToolDefinition, UnifiedRequest
from rotator_library.providers.gemini_provider import GeminiProvider
from rotator_library.request_sanitizer import sanitize_request_payload


def _ctx(source: str = "gemini", target: str = "gemini", *, source_provider: str = "provider-a", target_provider: str = "provider-b") -> ProtocolContext:
    return ProtocolContext(
        source_protocol=source,
        target_protocol=target,
        input_protocol=source,
        provider_protocol=target,
        client_protocol=source,
        source_provider=source_provider,
        target_provider=target_provider,
    )


def _warn_codes(container) -> set[str]:
    return {warning.code for warning in container.warnings}


# ---------------------------------------------------------------------------
# Capability resolution cascade + seam record


class _CascadePlugin:
    """Two-row declaration: the general row first, the narrow row last."""

    provider_env_name = "cascade"
    model_rules = (
        {"match": "*", "output_modalities": ["text"], "hosted_tools": ["googleSearch"]},
        {"match": "*-image", "output_modalities": ["image", "text"]},
        {"match": "pro-*", "thinking_dialect": "budget", "thinking_budget_range": [128, 4096],
         "effort_accept": ["low", "high"]},
    )


def test_capability_cascade_later_rows_win_and_inherit_non_conflicting_keys() -> None:
    record = resolve_model_capabilities(_CascadePlugin(), "pro-v1")
    assert record["thinking_dialect"] == "budget"
    assert record["thinking_budget_range"] == [128, 4096]
    assert record["output_modalities"] == ["text"]
    assert record["hosted_tools"] == ["googleSearch"]

    image = resolve_model_capabilities(_CascadePlugin(), "x-image")
    assert image["output_modalities"] == ["image", "text"]
    assert "thinking_dialect" not in image
    assert image["hosted_tools"] == ["googleSearch"]


def test_resolve_request_capabilities_adds_resolved_effort_accept_only_when_declared() -> None:
    class NoEffortPlugin:
        provider_env_name = "noeffort"
        model_rules = ({"match": "*", "thinking_dialect": "level"},)

    caps = resolve_request_capabilities(_CascadePlugin(), "pro-v1", protocol_name="gemini")
    # The acceptance chain's resolved vocabulary rides the record under the
    # row-key name so consumers can gate the OFF caveat.
    assert caps["effort_accept"] == ("low", "high")
    assert caps["thinking_dialect"] == "budget"

    plain = resolve_request_capabilities(NoEffortPlugin(), "m", protocol_name="gemini")
    # Protocol base is a floor, not a declaration: no effort_accept key.
    assert "effort_accept" not in plain

    undeclared = resolve_request_capabilities(None, "m", protocol_name="gemini")
    assert undeclared == {}


# ---------------------------------------------------------------------------
# canonical: dialect, budget clamp, OFF acceptance


def test_canonical_undeclared_reasoning_is_byte_identical() -> None:
    legacy = format_reasoning_controls({"effort": "medium"}, "gemini", UnifiedRequest())
    assert legacy == {"generation_config": {"thinkingConfig": {"thinkingLevel": "medium"}}}

    with_empty = format_reasoning_controls({"effort": "medium"}, "gemini", UnifiedRequest(), capabilities={})
    assert with_empty == legacy


def test_declared_budget_dialect_never_emits_thinking_level() -> None:
    emissions = format_reasoning_controls(
        {"effort": "medium"},
        "gemini",
        UnifiedRequest(),
        capabilities={"thinking_dialect": "budget"},
    )
    config = emissions["generation_config"]["thinkingConfig"]
    assert "thinkingLevel" not in config
    assert config["thinkingBudget"] == 8192

    # Declared level dialect keeps the exact native level lever.
    level = format_reasoning_controls(
        {"effort": "high"},
        "gemini",
        UnifiedRequest(),
        capabilities={"thinking_dialect": "level"},
    )
    assert level["generation_config"]["thinkingConfig"] == {"thinkingLevel": "high"}


def test_declared_budget_range_clamps_the_table_fallback_with_disclosure() -> None:
    request = UnifiedRequest()
    emissions = format_reasoning_controls(
        {"effort": "high"},
        "gemini",
        request,
        capabilities={"thinking_dialect": "budget", "thinking_budget_range": [128, 4096]},
    )
    # The table says 16384; the declared model ceiling wins.
    assert emissions["generation_config"]["thinkingConfig"]["thinkingBudget"] == 4096
    assert "reasoning_budget_coerced" in _warn_codes(request)

    # Undeclared range: the table value rides untouched (today's behavior).
    plain = format_reasoning_controls({"effort": "high"}, "gemini", UnifiedRequest(), capabilities={"thinking_dialect": "budget"})
    assert plain["generation_config"]["thinkingConfig"]["thinkingBudget"] == 16384


def test_declared_off_acceptance_suppresses_the_model_dependence_caveat() -> None:
    accepted = UnifiedRequest()
    format_reasoning_controls(
        {"effort": "none"},
        "gemini",
        accepted,
        capabilities={"effort_accept": ["off", "low"]},
    )
    # OFF is a declared capability: thinkingBudget=0 is exact, not
    # "model-dependent".
    assert "reasoning_disabled_model_dependent" not in _warn_codes(accepted)

    unaccepted = UnifiedRequest()
    format_reasoning_controls(
        {"effort": "none"},
        "gemini",
        unaccepted,
        capabilities={"effort_accept": ["low", "high"]},
    )
    assert "reasoning_disabled_model_dependent" in _warn_codes(unaccepted)

    undeclared = UnifiedRequest()
    format_reasoning_controls({"effort": "none"}, "gemini", undeclared)
    assert "reasoning_disabled_model_dependent" in _warn_codes(undeclared)


# ---------------------------------------------------------------------------
# Gemini tool-call ids


def _id_request() -> dict:
    """A same-protocol Gemini conversation carrying 3.x-style correlation
    ids on both the call and its response."""

    return {
        "contents": [
            {"role": "user", "parts": [{"text": "hi"}]},
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "lookup", "args": {"q": "x"}, "id": "call_1"}, "thoughtSignature": "sig-a"}],
            },
            {
                "role": "user",
                "parts": [{"functionResponse": {"name": "lookup", "response": {"value": 1}, "id": "call_1"}}],
            },
        ]
    }


def test_tool_call_ids_declared_false_never_emits_or_echoes_ids() -> None:
    gemini = get_protocol("gemini")
    context = _ctx()

    built = gemini.build_request(
        gemini.parse_request(_id_request()),
        context,
        capabilities={"tool_call_ids": False},
    )
    wire = json.dumps(built)
    assert "call_1" not in wire
    assert '"id"' not in wire


def test_tool_call_ids_undeclared_keeps_the_current_3x_behavior() -> None:
    gemini = get_protocol("gemini")
    context = _ctx()

    undeclared = json.dumps(gemini.build_request(gemini.parse_request(_id_request()), context))
    declared_true = json.dumps(
        gemini.build_request(gemini.parse_request(_id_request()), context, capabilities={"tool_call_ids": True})
    )
    # Undeclared and declared-true are byte-identical: both keep the genuine
    # wire ids (the synthetic-id filter is unchanged).
    assert "call_1" in undeclared
    assert undeclared == declared_true


# ---------------------------------------------------------------------------
# Thought-signature strictness


def _two_calls() -> dict:
    return {
        "contents": [
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "a", "args": {"x": 1}}, "thoughtSignature": "sig-a"},
                    {"functionCall": {"name": "b", "args": {"y": 2}}},
                ],
            }
        ]
    }


def test_signature_strictness_gates_the_build_sentinel() -> None:
    gemini = get_protocol("gemini")
    context = _ctx()  # foreign provider pair: bound signatures are suppressed

    undeclared = json.dumps(gemini.build_request(gemini.parse_request(_two_calls()), context))
    assert "skip_thought_signature_validator" in undeclared  # sibling heuristic

    never = json.dumps(
        gemini.build_request(gemini.parse_request(_two_calls()), context, capabilities={"requires_thought_signatures": False})
    )
    assert "skip_thought_signature_validator" not in never

    strict = json.dumps(
        gemini.build_request(gemini.parse_request(_two_calls()), context, capabilities={"requires_thought_signatures": True})
    )
    assert strict.count("skip_thought_signature_validator") == 2


def test_signature_strictness_gates_the_raw_strip_sentinel() -> None:
    def signed_pair() -> dict:
        return {
            "contents": [
                {
                    "parts": [
                        {"functionCall": {"name": "a"}, "thoughtSignature": "sig-a"},
                        {"functionCall": {"name": "b"}},
                    ]
                }
            ]
        }

    never = signed_pair()
    stripped = strip_foreign_opaque_state(never, "gemini", capabilities={"requires_thought_signatures": False})
    assert stripped
    assert "skip_thought_signature_validator" not in json.dumps(never)

    strict = signed_pair()
    strip_foreign_opaque_state(strict, "gemini", capabilities={"requires_thought_signatures": True})
    assert strict["contents"][0]["parts"][0]["thoughtSignature"] == "skip_thought_signature_validator"
    assert strict["contents"][0]["parts"][1]["thoughtSignature"] == "skip_thought_signature_validator"

    undeclared = signed_pair()
    strip_foreign_opaque_state(undeclared, "gemini")
    assert "skip_thought_signature_validator" in json.dumps(undeclared)


def test_signature_strictness_gates_the_stream_sentinel_and_tool_ids() -> None:
    gemini = get_protocol("gemini")
    context = _ctx()
    wire = {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": "a", "args": {"x": 1}, "id": "call_1"}, "thoughtSignature": "sig-a"}
    ]}}]}

    def _frames(state) -> str:
        frames: list = []
        for event in gemini.parse_stream_events(dict(wire), context):
            frames.extend(format_canonical_stream_event(event, "gemini", context, state=state))
        return "".join(frames)

    state = stream_format_state(context, "gemini")
    assert "sig-a" in _frames(state)

    # A second, sibling-less call with declarations threaded on the state:
    # the strict flag forbids ids and the heuristic default would not fire
    # (no sibling ever returned a signature).
    single = {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": "b", "args": {"y": 2}}, "id": "call_2"}
    ]}}]}

    def _single(state) -> str:
        frames: list = []
        for event in gemini.parse_stream_events(dict(single), context):
            frames.extend(format_canonical_stream_event(event, "gemini", context, state=state))
        return "".join(frames)

    never_state = stream_format_state(ProtocolContext(source_protocol="gemini", target_protocol="gemini"), "gemini")
    never_state.capabilities = {"requires_thought_signatures": False, "tool_call_ids": False}
    never = _single(never_state)
    assert "skip_thought_signature_validator" not in never
    assert "call_2" not in never

    strict_state = stream_format_state(ProtocolContext(source_protocol="gemini", target_protocol="gemini"), "gemini")
    strict_state.capabilities = {"requires_thought_signatures": True}
    assert "skip_thought_signature_validator" in _single(strict_state)


# ---------------------------------------------------------------------------
# Validation: modalities + hosted tools


def test_declared_output_modalities_narrow_and_widen() -> None:
    def _request() -> UnifiedRequest:
        request = UnifiedRequest(modalities=["text", "audio"])
        request.source_protocol = "openai_chat"
        return request

    text_only = _request()
    from rotator_library.protocols.validation import validate_generative_request

    validate_generative_request(text_only, "gemini", _ctx("openai_chat", "gemini"), capabilities={"output_modalities": ["text"]})
    assert text_only.modalities == ["text"]
    assert "unsupported_output_modality" in _warn_codes(text_only)

    image = UnifiedRequest(modalities=["text", "audio", "image"])
    image.source_protocol = "openai_chat"
    validate_generative_request(image, "gemini", _ctx("openai_chat", "gemini"), capabilities={"output_modalities": ["image", "text"]})
    assert image.modalities == ["text", "image"]

    # Undeclared: the protocol table stands (gemini keeps audio).
    undeclared = _request()
    validate_generative_request(undeclared, "gemini", _ctx("openai_chat", "gemini"))
    assert undeclared.modalities == ["text", "audio"]
    assert "unsupported_output_modality" not in _warn_codes(undeclared)


def test_declared_hosted_tools_limit_the_allowlist() -> None:
    from rotator_library.protocols.validation import validate_generative_request

    def _request() -> UnifiedRequest:
        request = UnifiedRequest(
            tools=[
                ToolDefinition(name="codeExecution", type="server", extra={"server_tool_type": "codeExecution"}),
                ToolDefinition(name="web_search", type="web_search"),
            ]
        )
        request.source_protocol = "openai_chat"
        return request

    limited = _request()
    validate_generative_request(limited, "gemini", _ctx("openai_chat", "gemini"), capabilities={"hosted_tools": ["googleSearch"]})
    # googleSearch-only model: the Responses web_search maps onto it and
    # survives; codeExecution drops with an informational warning.
    assert [tool.type for tool in limited.tools] == ["web_search"]
    assert "unsupported_optional_control" in _warn_codes(limited)

    undeclared = _request()
    validate_generative_request(undeclared, "gemini", _ctx("openai_chat", "gemini"))
    assert len(undeclared.tools) == 2

    permissive = _request()
    validate_generative_request(
        permissive,
        "gemini",
        _ctx("openai_chat", "gemini"),
        capabilities={"hosted_tools": ["googleSearch", "codeExecution"]},
    )
    assert len(permissive.tools) == 2


# ---------------------------------------------------------------------------
# candidateCount ceiling


def test_declared_max_candidates_clamps_with_disclosure() -> None:
    gemini = get_protocol("gemini")
    raw = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"candidateCount": 4},
    }

    clamped = gemini.build_request(gemini.parse_request(raw), _ctx(), capabilities={"max_candidates": 2})
    assert clamped["generationConfig"]["candidateCount"] == 2

    undeclared = gemini.build_request(gemini.parse_request(raw), _ctx())
    assert undeclared["generationConfig"]["candidateCount"] == 4


# ---------------------------------------------------------------------------
# Provider backfill rows


def test_provider_rows_resolve_spot_checks() -> None:
    plugin = GeminiProvider()

    pro = resolve_request_capabilities(plugin, "gemini-3.1-pro-preview", protocol_name="gemini")
    assert pro["thinking_dialect"] == "level"
    # Cannot disable: the accepted vocabulary has no OFF rung.
    assert "off" not in pro["effort_accept"]
    assert set(pro["effort_accept"]) == {"low", "medium", "high"}
    assert pro["tool_call_ids"] is True
    assert pro["requires_thought_signatures"] is True

    flash = resolve_request_capabilities(plugin, "gemini-2.5-flash", protocol_name="gemini")
    assert flash["thinking_dialect"] == "budget"
    assert flash["thinking_budget_range"] == [0, 24576]
    assert "off" in flash["effort_accept"]
    assert flash["tool_call_ids"] is False
    assert flash["requires_thought_signatures"] is False

    image = resolve_request_capabilities(plugin, "gemini-2.5-flash-image", protocol_name="gemini")
    assert image["output_modalities"] == ["image", "text"]

    lite = resolve_request_capabilities(plugin, "gemini-3.1-flash-image-lite", protocol_name="gemini")
    assert lite["output_modalities"] == ["text"]
    assert lite["hosted_tools"] == []

    tts = resolve_request_capabilities(plugin, "gemini-2.5-pro-preview-tts", protocol_name="gemini")
    assert tts["output_modalities"] == ["audio"]


# ---------------------------------------------------------------------------
# The executor seam: one resolution, threaded into the built wire


class _FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _FakeHTTPClient:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    async def post(self, endpoint, *, headers, json):
        self.calls.append(json)
        return _FakeHTTPResponse(self.response)


_GEMINI_RESPONSE = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
}


class _BudgetPlugin:
    """A gemini-like provider whose model declares the budget dialect."""

    provider_env_name = "budgetgemini"
    model_rules = (
        {
            "match": "*",
            "thinking_dialect": "budget",
            "thinking_budget_range": [128, 4096],
        },
    )


def test_executor_threads_the_capability_record_into_the_builder() -> None:
    client = _FakeHTTPClient(_GEMINI_RESPONSE)
    context = NativeProviderContext(
        provider="budgetgemini",
        model="declared-model",
        protocol_name="gemini",
        endpoint="https://example.test/v1beta/models/declared-model:generateContent",
        input_protocol_name="openai_chat",
        client_protocol_name="openai_chat",
        provider_plugin=_BudgetPlugin(),
    )

    async def _drive():
        await NativeProviderExecutor().execute(
            {
                "model": "declared-model",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            },
            context,
            NativeHTTPTransport(client),
        )

    asyncio.run(_drive())
    generation = client.calls[0]["generationConfig"]
    # The declared dialect replaced thinkingLevel with the budget shape and
    # the declared model ceiling clamped the table's 16384 down to 4096.
    assert "thinkingLevel" not in generation.get("thinkingConfig", {})
    assert generation["thinkingConfig"]["thinkingBudget"] == 4096


# ---------------------------------------------------------------------------
# Deleted surfaces


def test_request_sanitizer_no_longer_owns_gemini_thinking() -> None:
    payload = {"thinking": {"type": "enabled", "budget_tokens": -1}}
    assert sanitize_request_payload(dict(payload), "gemini/gemini-2.5-pro") == payload
    assert sanitize_request_payload(dict(payload), "someone/else") == payload

    # The dimensions prefix hack died too (G9): per-model legality is
    # capability data (validation checks the shape; the model database
    # seam will own legality) — the control never silently vanishes here.
    assert sanitize_request_payload({"dimensions": 3}, "someone/else") == {"dimensions": 3}
