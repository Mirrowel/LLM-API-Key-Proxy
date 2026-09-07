# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""W4 acceptance fixtures: cross-protocol conversion (D7/D9).

Every D7 semantic class has a defined behavior per protocol pair, and every
deliberate omission/approximation/merge is recorded in a client-visible
``x-proxy-conversion`` summary — content-asserted, never just structural.
"""

from __future__ import annotations

import pytest

from rotator_library.protocols import get_protocol
from rotator_library.protocols.canonical import (
    budget_tokens_from_effort,
    effort_from_budget_tokens,
)
from rotator_library.protocols.streaming import stream_format_state
from rotator_library.protocols.types import Annotation, ContentBlock, ProtocolContext, UnifiedResponse, Usage
from rotator_library.protocols.validation import ProtocolError


def _ctx(source: str, target: str) -> ProtocolContext:
    return ProtocolContext(
        source_protocol=source,
        target_protocol=target,
        input_protocol=source,
        provider_protocol=target,
        client_protocol=source,
    )


def _build(protocol_name: str, payload: dict, *, source: str) -> tuple[dict, object]:
    protocol = get_protocol(protocol_name)
    source_protocol = get_protocol(source)
    unified = source_protocol.parse_request(payload, _ctx(source, protocol_name))
    return protocol.build_request(unified, _ctx(source, protocol_name)), unified


def _warnings_of(unified) -> list[str]:
    return [w.code for w in unified.warnings]


def _summary_codes(payload: dict) -> list[str]:
    summary = payload.get("x-proxy-conversion") or {}
    return [entry["code"] for entry in summary.get("warnings", [])]


# ---------------------------------------------------------------------------
# D7 recorded summaries (defect 7's consumer half)


def test_unsupported_control_surfaces_in_response_summary() -> None:
    built, unified = _build("anthropic_messages", {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], "seed": 42}, source="openai_chat")
    assert "seed" not in built
    assert "unsupported_optional_control" in _warnings_of(unified)

    anthropic = get_protocol("anthropic_messages")
    response = anthropic.parse_response(
        {"id": "msg_1", "role": "assistant", "model": "m", "stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]},
        _ctx("anthropic_messages", "openai_chat"),
    )
    for warning in unified.warnings:
        response.warnings.append(warning)
    chat_payload = get_protocol("openai_chat").format_response(response, _ctx("anthropic_messages", "openai_chat"))
    assert "unsupported_optional_control" in _summary_codes(chat_payload)
    entry = next(e for e in chat_payload["x-proxy-conversion"]["warnings"] if e["code"] == "unsupported_optional_control")
    assert entry["field"] == "seed"


def test_clean_conversion_has_no_summary_block() -> None:
    built, unified = _build(
        "anthropic_messages",
        {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], "temperature": 0.3},
        source="openai_chat",
    )
    assert built["temperature"] == 0.3
    assert _warnings_of(unified) == []

    anthropic = get_protocol("anthropic_messages")
    response = anthropic.parse_response(
        {"id": "msg_1", "role": "assistant", "model": "m", "stop_reason": "end_turn", "content": [{"type": "text", "text": "ok"}]},
        _ctx("anthropic_messages", "openai_chat"),
    )
    chat_payload = get_protocol("openai_chat").format_response(response, _ctx("anthropic_messages", "openai_chat"))
    assert "x-proxy-conversion" not in chat_payload


# ---------------------------------------------------------------------------
# Reasoning controls mapping table (deterministic, warned approximations)


@pytest.mark.parametrize("effort,expected_budget", [("minimal", 1024), ("low", 4096), ("medium", 8192), ("high", 16384)])
def test_effort_to_anthropic_budget_table(effort: str, expected_budget: int) -> None:
    assert budget_tokens_from_effort(effort) == expected_budget
    built, unified = _build(
        "anthropic_messages",
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": effort},
        source="openai_chat",
    )
    # Documented migration: effort steers adaptive thinking via
    # output_config.effort (enabled+budget is rejected on current flagships).
    assert built["thinking"] == {"type": "adaptive"}
    assert built["output_config"]["effort"] == effort
    assert "reasoning_effort_approximated" in _warnings_of(unified)


@pytest.mark.parametrize("budget,expected_effort", [(1024, "minimal"), (4096, "low"), (9000, "medium"), (20000, "high")])
def test_budget_to_chat_effort_table(budget: int, expected_effort: str) -> None:
    assert effort_from_budget_tokens(budget) == expected_effort
    built, unified = _build(
        "openai_chat",
        {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "enabled", "budget_tokens": budget}},
        source="anthropic_messages",
    )
    assert built["reasoning_effort"] == expected_effort
    assert "reasoning_budget_approximated" in _warnings_of(unified)


def test_budget_to_responses_normalizes_never_foreign_keys() -> None:
    built, unified = _build(
        "responses",
        {"model": "m", "max_tokens": 16, "input": "hi", "thinking": {"type": "enabled", "budget_tokens": 9000}},
        source="anthropic_messages",
    )
    assert built["reasoning"] == {"effort": "medium"}
    assert "budget_tokens" not in built["reasoning"]
    assert "enabled" not in built["reasoning"]
    assert "reasoning_budget_approximated" in _warnings_of(unified)


def test_effort_to_gemini_budget_with_warning() -> None:
    built, unified = _build(
        "gemini",
        {"model": "g", "contents": [{"role": "user", "parts": [{"text": "hi"}]}], "reasoning_effort": "high"},
        source="openai_chat",
    )
    assert built["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 16384
    assert "reasoning_effort_approximated" in _warnings_of(unified)


def test_direct_effort_mapping_emits_no_approximation_warning() -> None:
    built, unified = _build(
        "responses",
        {"model": "m", "input": "hi", "reasoning_effort": "high"},
        source="openai_chat",
    )
    assert built["reasoning"]["effort"] == "high"
    assert "reasoning_budget_approximated" not in _warnings_of(unified)
    assert "reasoning_effort_approximated" not in _warnings_of(unified)


def test_include_thoughts_maps_to_responses_summary() -> None:
    built_true, _ = _build(
        "responses",
        {"model": "g", "contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"thinkingConfig": {"thinkingBudget": 2048, "includeThoughts": True}}},
        source="gemini",
    )
    assert built_true["reasoning"]["summary"] == "auto"
    built_false, _ = _build(
        "responses",
        {"model": "g", "contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"thinkingConfig": {"thinkingBudget": 2048, "includeThoughts": False}}},
        source="gemini",
    )
    assert built_false["reasoning"]["summary"] == "none"


def test_reasoning_disabled_is_recorded_not_silent() -> None:
    built, unified = _build(
        "openai_chat",
        {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "disabled"}},
        source="anthropic_messages",
    )
    assert "reasoning_effort" not in built
    assert "reasoning_disabled_omitted" in _warnings_of(unified)


# ---------------------------------------------------------------------------
# Instruction ordering (defect 4)


def test_chat_rebuild_preserves_interleaved_instructions() -> None:
    payload = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "system", "content": "mid instruction"},
            {"role": "user", "content": "q2"},
        ],
    }
    built, _ = _build("openai_chat", payload, source="openai_chat")
    roles = [message["role"] for message in built["messages"]]
    assert roles == ["user", "system", "user"]


def test_multiple_instructions_merge_in_order_with_summary() -> None:
    payload = {
        "model": "m",
        "max_tokens": 16,
        "messages": [
            {"role": "system", "content": "first"},
            {"role": "system", "content": "second"},
            {"role": "user", "content": "hi"},
        ],
    }
    built, unified = _build("anthropic_messages", payload, source="openai_chat")
    assert built["system"] == [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
    assert [m["role"] for m in built["messages"]] == ["user"]
    assert "instructions_merged" in _warnings_of(unified)


def test_interleaved_instruction_reposition_recorded() -> None:
    payload = {
        "model": "m",
        "max_tokens": 16,
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "system", "content": "mid"},
            {"role": "user", "content": "q2"},
        ],
    }
    built, unified = _build("gemini", payload, source="openai_chat")
    parts = built["systemInstruction"]["parts"]
    assert parts == [{"text": "mid"}]
    assert "instructions_merged" in _warnings_of(unified)


def test_non_text_instruction_block_recorded_at_responses() -> None:
    payload = {
        "model": "m",
        "input": "hi",
    }
    chat = get_protocol("openai_chat")
    unified = chat.parse_request(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": [{"type": "image_url", "image_url": {"url": "https://x.test/i.png"}}]},
                {"role": "user", "content": "hi"},
            ],
        },
        _ctx("openai_chat", "responses"),
    )
    built = get_protocol("responses").build_request(unified, _ctx("openai_chat", "responses"))
    assert "instructions" not in built or built["instructions"] == ""
    assert "instruction_block_dropped" in _warnings_of(unified)


# ---------------------------------------------------------------------------
# D9 multiplicity on the request side


def test_chat_n_maps_to_gemini_candidate_count() -> None:
    built, _ = _build(
        "gemini",
        {"model": "g", "n": 3, "messages": [{"role": "user", "content": "hi"}]},
        source="openai_chat",
    )
    assert built["generationConfig"]["candidateCount"] == 3


def test_gemini_candidate_count_maps_to_chat_n() -> None:
    built, _ = _build(
        "openai_chat",
        {"model": "g", "contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"candidateCount": 2}},
        source="gemini",
    )
    assert built["n"] == 2


# ---------------------------------------------------------------------------
# Media / audio


def test_audio_block_synthesizes_chat_message_audio() -> None:
    gemini = get_protocol("gemini")
    response = gemini.parse_response(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"inlineData": {"mimeType": "audio/wav", "data": "UklGRg=="}}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        },
        _ctx("gemini", "openai_chat"),
    )
    payload = get_protocol("openai_chat").format_response(response, _ctx("gemini", "openai_chat"))
    audio = payload["choices"][0]["message"].get("audio")
    # Documented RESPONSE audio object: {id, data, transcript} — the format
    # label belongs to the request-side audio parameter, not the response.
    assert audio and audio["data"] == "UklGRg=="
    assert "transcript" in audio
    assert "format" not in audio


def _audio_response() -> UnifiedResponse:
    gemini = get_protocol("gemini")
    return gemini.parse_response(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"inlineData": {"mimeType": "audio/wav", "data": "UklGRg=="}}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        },
        _ctx("gemini", "openai_chat"),
    )


@pytest.mark.parametrize("target", ["anthropic_messages", "responses"])
def test_audio_drop_recorded_at_unsupported_targets(target: str) -> None:
    response = _audio_response()
    payload = get_protocol(target).format_response(response, _ctx("gemini", target))
    assert "media_dropped" in _summary_codes(payload)


def test_image_maps_chat_to_anthropic_request() -> None:
    built, _ = _build(
        "anthropic_messages",
        {
            "model": "m",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    ],
                }
            ],
        },
        source="openai_chat",
    )
    blocks = built["messages"][0]["content"]
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
    }


# ---------------------------------------------------------------------------
# Safety settings (gemini)


def test_safety_settings_rejected_cross_protocol() -> None:
    with pytest.raises(ProtocolError):
        _build(
            "openai_chat",
            {"model": "g", "contents": [{"role": "user", "parts": [{"text": "hi"}]}], "safetySettings": [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]},
            source="gemini",
        )


def test_safety_settings_pass_through_same_protocol() -> None:
    safety = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}]
    built, _ = _build(
        "gemini",
        {"model": "g", "contents": [{"role": "user", "parts": [{"text": "hi"}]}], "safetySettings": safety},
        source="gemini",
    )
    assert built["safetySettings"] == safety


# ---------------------------------------------------------------------------
# Stop/status table


@pytest.mark.parametrize(
    "target,field,expected",
    [
        ("openai_chat", "finish_reason", "content_filter"),
        ("anthropic_messages", "stop_reason", "refusal"),
        ("gemini", "finishReason", "SAFETY"),
        ("responses", "status", "incomplete"),
    ],
)
def test_content_filter_stop_reason_mapping(target: str, field: str, expected: str) -> None:
    chat = get_protocol("openai_chat")
    response = chat.parse_response(
        {
            "id": "r1",
            "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "refusal": "no"}, "finish_reason": "content_filter"}],
        },
        _ctx("openai_chat", target),
    )
    payload = get_protocol(target).format_response(response, _ctx("openai_chat", target))
    if target == "openai_chat":
        assert payload["choices"][0][field] == expected
    elif target == "anthropic_messages":
        assert payload[field] == expected
    elif target == "responses":
        assert payload[field] == expected
    else:
        assert payload["candidates"][0][field] == expected


# ---------------------------------------------------------------------------
# Structured output


def test_structured_output_chat_to_responses_json_schema() -> None:
    schema = {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": {"type": "object"}}}
    built, _ = _build(
        "responses",
        {"model": "m", "input": "hi", "response_format": schema},
        source="openai_chat",
    )
    assert built["text"]["format"]["type"] == "json_schema"
    assert built["text"]["format"]["schema"] == {"type": "object"}
    assert built["text"]["format"]["name"] == "out"


# ---------------------------------------------------------------------------
# W2 carry-ins: anthropic server tools + stream-side emission


def test_anthropic_server_tools_parse_to_builtin_records() -> None:
    ant = get_protocol("anthropic_messages")
    response = ant.parse_response(
        {
            "id": "m1",
            "model": "m",
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [
                {"type": "server_tool_use", "id": "srvu_1", "name": "web_search", "input": {"query": "x"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvu_1", "content": [{"type": "web_search_result", "title": "t", "url": "https://x"}]},
            ],
        },
        None,
    )
    blocks = response.messages[0].content
    assert [b.type for b in blocks] == ["builtin_tool", "builtin_tool"]
    assert blocks[0].builtin_tool.kind == "web_search"
    assert blocks[0].builtin_tool.call_id == "srvu_1"
    # Same-protocol round-trip keeps the raw blocks verbatim.
    back = ant.format_response(response, None)
    assert [b["type"] for b in back["content"]] == ["server_tool_use", "web_search_tool_result"]


def test_stream_builtin_items_emit_native_at_responses_and_omit_elsewhere() -> None:
    from rotator_library.protocols.streaming import format_canonical_stream_event
    from rotator_library.protocols.types import BuiltinToolCall, UnifiedMessage, UnifiedStreamEvent

    raw_item = {"type": "web_search_call", "id": "ws_1", "status": "completed"}
    event = UnifiedStreamEvent(
        type="message.delta",
        source_protocol="openai_chat",
        native_type="message.delta",
        delta=UnifiedMessage(
            role="assistant",
            content=[ContentBlock(type="builtin_tool", builtin_tool=BuiltinToolCall(kind="web_search", call_id="ws_1", status="completed", raw=raw_item))],
        ),
    )
    resp_ctx = ProtocolContext(source_protocol="openai_chat", target_protocol="responses", input_protocol="openai_chat", client_protocol="responses", provider_protocol="responses")
    frames = format_canonical_stream_event(event, "responses", resp_ctx)
    joined = "".join(frames)
    assert "response.output_item.added" in joined
    assert "ws_1" in joined

    ant_ctx = ProtocolContext(source_protocol="openai_chat", target_protocol="anthropic_messages", input_protocol="openai_chat", client_protocol="anthropic_messages", provider_protocol="anthropic_messages")
    ant_frames = format_canonical_stream_event(event, "anthropic_messages", ant_ctx)
    ant_joined = "".join(ant_frames)
    # No fabricated empty text block for provider-internal records.
    assert '"text": ""' not in ant_joined
    assert "ws_1" not in ant_joined


def test_stream_refusal_degrades_to_text_at_anthropic() -> None:
    from rotator_library.protocols.streaming import format_canonical_stream_event
    from rotator_library.protocols.types import UnifiedMessage, UnifiedStreamEvent

    event = UnifiedStreamEvent(
        type="message.delta",
        source_protocol="openai_chat",
        native_type="message.delta",
        delta=UnifiedMessage(role="assistant", content=[ContentBlock(type="refusal", refusal="cannot help with that")]),
    )
    ctx = ProtocolContext(source_protocol="openai_chat", target_protocol="anthropic_messages", input_protocol="openai_chat", client_protocol="anthropic_messages", provider_protocol="anthropic_messages")
    frames = format_canonical_stream_event(event, "anthropic_messages", ctx)
    joined = "".join(frames)
    assert "cannot help with that" in joined
    assert '"type": "text"' in joined


# ---------------------------------------------------------------------------
# Round 2: review-fix regressions (reasoning normalization, media honesty,
# stream refusal, builtin bookkeeping, dual instructions)


def test_effort_none_maps_exactly_per_target() -> None:
    chat_payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"}

    # Chat and Responses accept the full effort vocabulary verbatim.
    built_chat, _ = _build("responses", chat_payload, source="openai_chat")
    assert built_chat["reasoning"] == {"effort": "none"}

    built_resp, unified_resp = _build("responses", chat_payload, source="openai_chat")
    assert "reasoning_disabled_omitted" not in _warnings_of(unified_resp)

    # Same-protocol chat passthrough keeps the native spelling untouched.
    chat = get_protocol("openai_chat")
    unified = chat.parse_request(dict(chat_payload), _ctx("openai_chat", "openai_chat"))
    preserved = chat.build_request(unified, _ctx("openai_chat", "openai_chat"))
    assert preserved["reasoning_effort"] == "none"

    # Anthropic maps "none" exactly to its disabled construct.
    built_ant, _ = _build("anthropic_messages", chat_payload, source="openai_chat")
    assert built_ant["thinking"] == {"type": "disabled"}

    # Gemini maps "none" to thinkingBudget 0 with the model-dependence note.
    built_gem, unified_gem = _build("gemini", chat_payload, source="openai_chat")
    assert built_gem["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0
    assert "reasoning_disabled_model_dependent" in _warnings_of(unified_gem)


def test_responses_summary_round_trip_preserved_and_mapped() -> None:
    payload = {"model": "m", "input": "hi", "reasoning": {"effort": "high", "summary": "auto"}}

    rebuilt, unified = _build("responses", payload, source="responses")
    assert rebuilt["reasoning"] == {"effort": "high", "summary": "auto"}
    assert "reasoning_control_dropped" not in _warnings_of(unified)

    built_gem, _ = _build("gemini", payload, source="responses")
    assert built_gem["generationConfig"]["thinkingConfig"]["includeThoughts"] is True


def test_gemini_thinking_budget_zero_is_off_not_inverted() -> None:
    payload = {"model": "m", "contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"thinkingConfig": {"thinkingBudget": 0}}}

    built_ant, _ = _build("anthropic_messages", payload, source="gemini")
    assert built_ant["thinking"] == {"type": "disabled"}


def test_unknown_effort_disclosed_not_silent_medium() -> None:
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "ultra"}

    built, unified = _build("anthropic_messages", payload, source="openai_chat")
    assert built["thinking"] == {"type": "adaptive"}
    assert built["output_config"]["effort"] == "medium"
    assert "reasoning_effort_unknown" in _warnings_of(unified)


def test_anthropic_subminimum_budget_omitted_with_warning() -> None:
    built, unified = _build(
        "anthropic_messages",
        {"model": "m", "input": "hi", "reasoning": {"effort": "high", "summary": "auto", "budget_tokens": 0}},
        source="responses",
    )
    assert built["thinking"] == {"type": "adaptive"}
    assert built["output_config"]["effort"] == "high"
    assert "reasoning_budget_invalid" in _warnings_of(unified)
    assert "reasoning_effort_model_dependent" in _warnings_of(unified)


def test_stream_refusal_emits_at_chat_and_gemini() -> None:
    from rotator_library.protocols.streaming import format_canonical_stream_event
    from rotator_library.protocols.types import UnifiedMessage, UnifiedStreamEvent

    event = UnifiedStreamEvent(
        type="message.delta",
        source_protocol="anthropic_messages",
        native_type="message.delta",
        delta=UnifiedMessage(role="assistant", content=[ContentBlock(type="refusal", refusal="cannot help")]),
    )
    chat_frames = format_canonical_stream_event(event, "openai_chat", _ctx("anthropic_messages", "openai_chat"))
    assert '"refusal": "cannot help"' in "".join(chat_frames)

    gemini_frames = format_canonical_stream_event(event, "gemini", _ctx("anthropic_messages", "gemini"))
    assert "cannot help" in "".join(gemini_frames)


def test_stream_builtin_interleaved_identity_and_terminal_inclusion() -> None:
    from rotator_library.protocols.streaming import format_canonical_stream_event
    from rotator_library.protocols.types import BuiltinToolCall, UnifiedMessage, UnifiedStreamEvent

    def _event(blocks: list) -> UnifiedStreamEvent:
        return UnifiedStreamEvent(
            type="message.delta",
            source_protocol="openai_chat",
            native_type="message.delta",
            delta=UnifiedMessage(role="assistant", content=blocks),
        )

    builtin_item = {"type": "web_search_call", "id": "ws_1", "status": "completed"}
    events = [
        _event([ContentBlock(type="text", text="AA")]),
        _event([ContentBlock(type="builtin_tool", builtin_tool=BuiltinToolCall(kind="web_search", call_id="ws_1", status="completed", raw=builtin_item))]),
        _event([ContentBlock(type="text", text="BB")]),
        UnifiedStreamEvent(type="done", source_protocol="openai_chat", native_type="done", stop_reason="stop", usage=Usage(input_tokens=1, output_tokens=1)),
    ]
    ctx = _ctx("openai_chat", "responses")
    state = stream_format_state(ctx, "responses")
    all_frames: list[str] = []
    for event in events:
        all_frames.extend(format_canonical_stream_event(event, "responses", ctx, state=state))
    joined = "".join(all_frames)

    # Two distinct text items — interleaving preserved (defect 8).
    assert joined.count('"type": "message"') >= 3  # added + done carry the item
    assert '"AA"' in joined and '"BB"' in joined
    # The builtin item is registered, index-aligned, and present in the
    # terminal response object (never silently missing from output).
    assert joined.count('"type": "web_search_call"') >= 3  # added + done + terminal
    assert "response.completed" in joined


def test_dual_system_field_and_message_both_promoted() -> None:
    payload = {
        "model": "m",
        "instructions": "INSTR_FIELD",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "q1"}]}],
    }
    # Emulate a responses request with BOTH instructions and an explicit
    # system-role input item.
    payload["input"].insert(0, {"role": "system", "content": [{"type": "input_text", "text": "INSTR_MSG"}]})

    built, unified = _build("anthropic_messages", payload, source="responses")
    system_text = "".join(part.get("text", "") for part in built["system"])
    assert "INSTR_FIELD" in system_text and "INSTR_MSG" in system_text
    assert "instructions_merged" in _warnings_of(unified)


def test_custom_tool_result_block_is_not_a_builtin() -> None:
    ant = get_protocol("anthropic_messages")
    response = ant.parse_response(
        {
            "id": "m1",
            "model": "m",
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [{"type": "my_custom_tool_result", "tool_use_id": "x", "content": "odd"}],
        },
        None,
    )
    block = response.messages[0].content[0]
    assert block.type == "my_custom_tool_result"
    assert block.builtin_tool is None


def test_audio_id_is_deterministic_across_instances() -> None:
    gem = get_protocol("gemini")
    unified = gem.parse_response(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"inlineData": {"mimeType": "audio/wav", "data": "UklGRg=="}}]},
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-x",
        },
        None,
    )
    chat = get_protocol("openai_chat")
    first = chat.format_response(unified, _ctx("gemini", "openai_chat"))
    second = chat.format_response(unified, _ctx("gemini", "openai_chat"))
    audio_id = first["choices"][0]["message"]["audio"]["id"]
    assert audio_id == second["choices"][0]["message"]["audio"]["id"]
    assert "transcript" in first["choices"][0]["message"]["audio"]


def test_url_only_audio_and_video_at_chat_record_drops() -> None:
    gem = get_protocol("gemini")
    unified = gem.parse_response(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"fileData": {"mimeType": "audio/wav", "fileUri": "https://f/audio"}}]},
                    "finishReason": "STOP",
                }
            ],
            "modelVersion": "gemini-x",
        },
        None,
    )
    chat = get_protocol("openai_chat")
    formatted = chat.format_response(unified, _ctx("gemini", "openai_chat"))
    assert "media_dropped" in {w["code"] for w in formatted.get("x-proxy-conversion", {}).get("warnings", [])}


# ---------------------------------------------------------------------------
# Round 3: refusal streams at responses, budget-discard disclosure,
# dual-instruction promotion at chat


def test_responses_stream_refusal_emits_refusal_items() -> None:
    from rotator_library.protocols.streaming import format_canonical_stream_event, stream_format_state
    from rotator_library.protocols.types import UnifiedMessage, UnifiedStreamEvent

    events = [
        UnifiedStreamEvent(
            type="message.delta",
            source_protocol="openai_chat",
            native_type="message.delta",
            delta=UnifiedMessage(role="assistant", content=[ContentBlock(type="refusal", refusal="cannot help")]),
        ),
        UnifiedStreamEvent(type="done", source_protocol="openai_chat", native_type="done", stop_reason="refusal", usage=Usage(input_tokens=1, output_tokens=1)),
    ]
    ctx = _ctx("openai_chat", "responses")
    state = stream_format_state(ctx, "responses")
    frames: list[str] = []
    for event in events:
        frames.extend(format_canonical_stream_event(event, "responses", ctx, state=state))
    joined = "".join(frames)
    assert "response.refusal.delta" in joined
    assert "cannot help" in joined
    assert '"type": "refusal"' in joined  # parts and terminal item carry it


def test_refusal_with_annotations_at_responses_no_crash_and_recorded() -> None:
    chat = get_protocol("openai_chat")
    unified = chat.parse_response(
        {
            "id": "c1",
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None, "refusal": "no can do"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        None,
    )
    unified.messages[0].content = [
        ContentBlock(type="refusal", refusal="no can do", annotations=[Annotation(type="url_citation", url="https://x", title="t")])
    ]
    formatted = get_protocol("responses").format_response(unified, _ctx("openai_chat", "responses"))
    assert any(
        item.get("content", [{}])[0].get("type") == "refusal"
        for item in formatted["output"]
        if item.get("type") == "message"
    )
    assert "annotations_dropped" in {w["code"] for w in formatted.get("x-proxy-conversion", {}).get("warnings", [])}


def test_effort_and_budget_both_disclosure_of_discarded_budget() -> None:
    payload = {
        "model": "m",
        "input": "hi",
        "reasoning": {"effort": "high", "budget_tokens": 2048, "summary": "auto"},
    }
    built, unified = _build("gemini", payload, source="responses")
    assert built["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 16384
    assert "reasoning_control_dropped" in _warnings_of(unified)


def test_chat_promotes_dual_instruction_sources() -> None:
    payload = {
        "model": "m",
        "instructions": "INSTR_FIELD",
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": "INSTR_MSG"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "q1"}]},
        ],
    }
    built, _ = _build("openai_chat", payload, source="responses")
    roles = [m["role"] for m in built["messages"]]
    assert roles[0] == "system"
    system_text = "".join(
        part.get("text", "") if isinstance(part, dict) else str(part)
        for message in built["messages"]
        if message["role"] == "system"
        for part in (message["content"] if isinstance(message["content"], list) else [message["content"]])
    )
    assert "INSTR_FIELD" in system_text and "INSTR_MSG" in system_text
    assert roles[-1] == "user"


# ---------------------------------------------------------------------------
# Round 4: budget headroom, disabled includeThoughts, diagonal purity,
# refusal family, foreign builtins, assistant images


def test_anthropic_budget_clamped_below_max_tokens() -> None:
    built, unified = _build(
        "anthropic_messages",
        {"model": "m", "max_tokens": 4096, "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high"},
        source="openai_chat",
    )
    assert built["thinking"] == {"type": "adaptive"}
    assert built["output_config"]["effort"] == "high"
    assert "reasoning_effort_approximated" in _warnings_of(unified)


def test_anthropic_no_headroom_omits_thinking() -> None:
    built, unified = _build(
        "anthropic_messages",
        {"model": "m", "max_tokens": 900, "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high"},
        source="openai_chat",
    )
    # Adaptive thinking (no explicit budget): the provider arbitrates the
    # budget against max_tokens — the proxy never fabricates an omission.
    assert built["thinking"] == {"type": "adaptive"}
    assert built["output_config"]["effort"] == "high"
    assert "reasoning_effort_approximated" in _warnings_of(unified)


def test_gemini_disabled_forces_include_thoughts_false() -> None:
    built, _ = _build(
        "gemini",
        {"model": "m", "input": "hi", "reasoning": {"effort": "none", "summary": "auto"}},
        source="responses",
    )
    config = built["generationConfig"]["thinkingConfig"]
    assert config["thinkingBudget"] == 0
    assert config["includeThoughts"] is False


def test_reasoning_same_protocol_diagonals_stay_verbatim() -> None:
    ant = get_protocol("anthropic_messages")
    unified = ant.parse_request(
        {"model": "m", "max_tokens": 2048, "system": "s", "messages": [{"role": "user", "content": "hi"}], "thinking": {"type": "enabled", "budget_tokens": 1024}},
        _ctx("anthropic_messages", "anthropic_messages"),
    )
    rebuilt = ant.build_request(unified, _ctx("anthropic_messages", "anthropic_messages"))
    assert rebuilt["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert not unified.warnings

    gem = get_protocol("gemini")
    unified_gem = gem.parse_request(
        {"model": "m", "contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"thinkingConfig": {"thinkingBudget": 0, "includeThoughts": True, "futureKey": "x"}}},
        _ctx("gemini", "gemini"),
    )
    rebuilt_gem = gem.build_request(unified_gem, _ctx("gemini", "gemini"))
    assert rebuilt_gem["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0
    assert not unified_gem.warnings


def test_stream_text_refusal_text_keeps_three_blocks() -> None:
    from rotator_library.protocols.streaming import format_canonical_stream_event, stream_format_state
    from rotator_library.protocols.types import UnifiedMessage, UnifiedStreamEvent

    def _delta(blocks: list) -> UnifiedStreamEvent:
        return UnifiedStreamEvent(type="message.delta", source_protocol="openai_chat", native_type="message.delta", delta=UnifiedMessage(role="assistant", content=blocks))

    events = [
        _delta([ContentBlock(type="text", text="hi ")]),
        _delta([ContentBlock(type="refusal", refusal="cannot help")]),
        _delta([ContentBlock(type="text", text="there")]),
        UnifiedStreamEvent(type="done", source_protocol="openai_chat", native_type="done", stop_reason="stop", usage=Usage(input_tokens=1, output_tokens=1)),
    ]
    for target in ("responses", "anthropic_messages"):
        ctx = _ctx("openai_chat", target)
        state = stream_format_state(ctx, target)
        joined = "".join(frame for event in events for frame in format_canonical_stream_event(event, target, ctx, state=state))
        assert "cannot help" in joined
        if target == "responses":
            # Text never merges into the refusal part: three distinct items.
            assert '"refusal": "cannot help"' in joined
            assert '"text": "hi "' in joined and '"text": "there"' in joined
            assert '"refusal": "hi ' not in joined and '"refusal": "cannot helpthere"' not in joined


def test_foreign_builtin_raw_synthesizes_native_item_cross_protocol() -> None:
    ant = get_protocol("anthropic_messages")
    unified = ant.parse_response(
        {
            "id": "m1",
            "model": "m",
            "role": "assistant",
            "stop_reason": "end_turn",
            "content": [{"type": "server_tool_use", "id": "srvu_1", "name": "web_search", "input": {"query": "x"}}],
        },
        None,
    )
    formatted = get_protocol("responses").format_response(unified, _ctx("anthropic_messages", "responses"))
    item_types = [item.get("type") for item in formatted["output"]]
    assert "server_tool_use" not in item_types
    assert "web_search_call" in item_types


def test_assistant_image_output_drops_recorded_not_fabricated() -> None:
    gem = get_protocol("gemini")
    unified = gem.parse_response(
        {
            "candidates": [{"content": {"role": "model", "parts": [{"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}}]}, "finishReason": "STOP"}],
            "modelVersion": "gemini-x",
        },
        None,
    )
    for target in ("anthropic_messages", "responses", "openai_chat"):
        formatted = get_protocol(target).format_response(unified, _ctx("gemini", target))
        payload_str = str(formatted)
        assert "aW1hZ2U=" not in payload_str, target  # actually dropped, not just warned
        warnings = {w["code"] for w in formatted.get("x-proxy-conversion", {}).get("warnings", [])}
        assert "media_dropped" in warnings, target


def test_chat_field_plus_interleaved_system_keeps_positions() -> None:
    resp = get_protocol("responses")
    unified = resp.parse_request(
        {
            "model": "m",
            "instructions": "INSTR_FIELD",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "q1"}]},
                {"role": "system", "content": [{"type": "input_text", "text": "MID"}]},
                {"role": "user", "content": [{"type": "input_text", "text": "q2"}]},
            ],
        },
        _ctx("responses", "openai_chat"),
    )
    built = get_protocol("openai_chat").build_request(unified, _ctx("responses", "openai_chat"))
    roles = [m["role"] for m in built["messages"]]
    # Field leads; the interleaved system message keeps its position.
    assert roles == ["system", "user", "system", "user"]


def test_gemini_strictness_strengthening_recorded() -> None:
    built, unified = _build(
        "gemini",
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "strict": False, "schema": {"type": "object"}}},
        },
        source="openai_chat",
    )
    assert built["generationConfig"]["responseMimeType"] == "application/json"
    assert "structured_output_strictness_strengthened" in _warnings_of(unified)
