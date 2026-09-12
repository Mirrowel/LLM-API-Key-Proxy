# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G3 slice A pins: opaque provider-identity gates on the CANONICAL lane.

Phase 1 stripped foreign opaque state from the RAW wire (``opaque_strip``);
this slice closes the canonical rebuild lane. Per-provider bound state
(Gemini thought signatures, Responses ``encrypted_content``, chat
``extra_content.google.thought_signature``) may leave canonical state only
when the provider pair proves compatibility (same provider id, or an
explicitly compatible domain). Foreign pairs strip the bound field, keep the
portable payload (text, summary, reasoning_content), and RECORD the drop.

Every assertion drives a real ``build_request`` / ``format_response`` flow
with a :class:`ProtocolContext` — no private formatter poking.
"""

from __future__ import annotations

import json

from rotator_library.protocols import ProtocolContext, get_protocol

_OPAQUE_CODE = "opaque_state_suppressed"


def _ctx(
    protocol: str,
    *,
    source_provider: str = "provider-a",
    target_provider: str = "provider-b",
    compatible: bool = False,
) -> ProtocolContext:
    return ProtocolContext(
        source_protocol=protocol,
        target_protocol=protocol,
        source_provider=source_provider,
        target_provider=target_provider,
        provider_state_compatible=compatible,
    )


def _codes(warnings) -> set[str]:
    return {warning.code for warning in warnings}


# ----------------------------------------------------------------------------
# Gemini: thought signatures on text / media / tool_result / tool_call
# ----------------------------------------------------------------------------


def _gemini_request() -> dict:
    return {
        "contents": [
            {"role": "user", "parts": [{"text": "hi"}]},
            {
                "role": "model",
                "parts": [
                    {"text": "thinking", "thought": True, "thoughtSignature": "sig-thought"},
                    {"text": "answer", "thoughtSignature": "sig-text"},
                    {"inlineData": {"mimeType": "image/png", "data": "AAAA"}, "thoughtSignature": "sig-media"},
                    {"functionCall": {"name": "lookup", "args": {"q": "x"}}, "thoughtSignature": "sig-call"},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "lookup", "response": {"value": 1}}, "thoughtSignature": "sig-result"}
                ],
            },
        ]
    }


def test_gemini_same_provider_emits_all_bound_signatures() -> None:
    adapter = get_protocol("gemini")
    parsed = adapter.parse_request(_gemini_request())

    built = adapter.build_request(parsed, _ctx("gemini", target_provider="provider-a"))

    wire = json.dumps(built)
    for signature in ("sig-thought", "sig-text", "sig-media", "sig-call", "sig-result"):
        assert signature in wire, signature
    assert _OPAQUE_CODE not in _codes(parsed.warnings)


def test_gemini_cross_provider_strips_signatures_and_warns() -> None:
    adapter = get_protocol("gemini")
    parsed = adapter.parse_request(_gemini_request())

    built = adapter.build_request(parsed, _ctx("gemini"))

    wire = json.dumps(built)
    for signature in ("sig-thought", "sig-text", "sig-media", "sig-call", "sig-result"):
        assert signature not in wire, signature
    # Portable payload survives: thought text, answer text, media data.
    assert "thinking" in wire and "answer" in wire and "AAAA" in wire
    assert _OPAQUE_CODE in _codes(parsed.warnings)


def test_gemini_thought_part_degrades_to_text_thought_without_signature() -> None:
    adapter = get_protocol("gemini")
    parsed = adapter.parse_request(_gemini_request())

    built = adapter.build_request(parsed, _ctx("gemini"))

    thought_parts = [
        part
        for content in built["contents"]
        for part in content["parts"]
        if part.get("thought") is True
    ]
    assert thought_parts and thought_parts[0]["text"] == "thinking"
    assert "thoughtSignature" not in thought_parts[0]


def _gemini_two_calls(*, signed_first: bool, signed_second: bool) -> dict:
    first = {"functionCall": {"name": "a", "args": {"x": 1}}}
    second = {"functionCall": {"name": "b", "args": {"y": 2}}}
    if signed_first:
        first["thoughtSignature"] = "sig-a"
    if signed_second:
        second["thoughtSignature"] = "sig-b"
    return {"contents": [{"role": "model", "parts": [first, second]}]}


def test_gemini_sentinel_fires_for_unsigned_sibling_on_foreign_pair() -> None:
    adapter = get_protocol("gemini")
    parsed = adapter.parse_request(_gemini_two_calls(signed_first=True, signed_second=False))

    built = adapter.build_request(parsed, _ctx("gemini"))

    wire = json.dumps(built)
    assert "sig-a" not in wire
    assert '"thoughtSignature": "skip_thought_signature_validator"' in wire


def test_gemini_sentinel_not_fabricated_when_no_call_was_signed() -> None:
    adapter = get_protocol("gemini")
    parsed = adapter.parse_request(_gemini_two_calls(signed_first=False, signed_second=False))

    built = adapter.build_request(parsed, _ctx("gemini"))

    assert "skip_thought_signature_validator" not in json.dumps(built)


def test_gemini_same_provider_emits_real_signature_and_sibling_sentinel() -> None:
    """Stream parity: a signed call keeps its signature; an unsigned sibling
    in the same turn gets the skip sentinel (Gemini-3 validates every call)."""

    adapter = get_protocol("gemini")
    parsed = adapter.parse_request(_gemini_two_calls(signed_first=True, signed_second=False))

    built = adapter.build_request(parsed, _ctx("gemini", target_provider="provider-a"))

    wire = json.dumps(built)
    assert "sig-a" in wire
    assert "skip_thought_signature_validator" in wire


# ----------------------------------------------------------------------------
# Responses: encrypted_content on input items and output items
# ----------------------------------------------------------------------------


def _responses_input_request() -> dict:
    return {
        "model": "model-a",
        "input": [
            {"role": "user", "content": "think"},
            {
                "type": "reasoning",
                "id": "rs_0",
                "summary": [{"type": "summary_text", "text": "because"}],
                "encrypted_content": "ENC-SECRET",
            },
        ],
    }


def test_responses_same_provider_keeps_encrypted_content() -> None:
    adapter = get_protocol("responses")
    parsed = adapter.parse_request(_responses_input_request())

    built = adapter.build_request(parsed, _ctx("responses", target_provider="provider-a"))

    reasoning = [item for item in built["input"] if item.get("type") == "reasoning"]
    assert reasoning and reasoning[0]["encrypted_content"] == "ENC-SECRET"
    assert _OPAQUE_CODE not in _codes(parsed.warnings)


def test_responses_cross_provider_reasoning_item_survives_without_encrypted_content() -> None:
    adapter = get_protocol("responses")
    parsed = adapter.parse_request(_responses_input_request())

    built = adapter.build_request(parsed, _ctx("responses"))

    reasoning = [item for item in built["input"] if item.get("type") == "reasoning"]
    assert reasoning, "the reasoning item itself must survive"
    assert "encrypted_content" not in reasoning[0]
    assert reasoning[0]["summary"] == [{"type": "summary_text", "text": "because"}]
    assert _OPAQUE_CODE in _codes(parsed.warnings)


def _responses_response() -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "model-a",
        "output": [
            {
                "id": "rs_0",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "because"}],
                "encrypted_content": "ENC-SECRET",
            },
            {
                "id": "msg_0",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok"}],
            },
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


def test_responses_response_same_provider_keeps_encrypted_content() -> None:
    adapter = get_protocol("responses")
    parsed = adapter.parse_response(_responses_response())

    built = adapter.format_response(parsed, _ctx("responses", target_provider="provider-a"))

    reasoning = [item for item in built["output"] if item.get("type") == "reasoning"]
    assert reasoning and reasoning[0]["encrypted_content"] == "ENC-SECRET"


def test_responses_response_cross_provider_drops_encrypted_content_only() -> None:
    adapter = get_protocol("responses")
    parsed = adapter.parse_response(_responses_response())

    built = adapter.format_response(parsed, _ctx("responses"))

    reasoning = [item for item in built["output"] if item.get("type") == "reasoning"]
    assert reasoning, "reasoning output item must survive"
    assert "encrypted_content" not in reasoning[0]
    assert reasoning[0]["summary"] == [{"type": "summary_text", "text": "because"}]
    assert _OPAQUE_CODE in _codes(parsed.warnings)


# ----------------------------------------------------------------------------
# OpenAI Chat: vendor-keyed extra_content signatures; reasoning is portable
# ----------------------------------------------------------------------------


def _chat_request() -> dict:
    return {
        "model": "model-a",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "portable-reasoning",
                "extra_content": {"google": {"thought_signature": "sig-msg"}},
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                        "extra_content": {"google": {"thought_signature": "sig-call"}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ],
    }


def test_chat_same_provider_emits_vendor_signatures() -> None:
    adapter = get_protocol("openai_chat")
    parsed = adapter.parse_request(_chat_request())

    built = adapter.build_request(parsed, _ctx("openai_chat", target_provider="provider-a"))

    assistant = built["messages"][1]
    assert assistant["extra_content"]["google"]["thought_signature"] == "sig-msg"
    assert assistant["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "sig-call"
    assert _OPAQUE_CODE not in _codes(parsed.warnings)


def test_chat_cross_provider_strips_vendor_signatures_and_warns() -> None:
    adapter = get_protocol("openai_chat")
    parsed = adapter.parse_request(_chat_request())

    built = adapter.build_request(parsed, _ctx("openai_chat"))

    assistant = built["messages"][1]
    assert "extra_content" not in assistant
    assert "extra_content" not in assistant["tool_calls"][0]
    assert "sig-msg" not in json.dumps(built)
    assert "sig-call" not in json.dumps(built)
    assert _OPAQUE_CODE in _codes(parsed.warnings)


def test_chat_reasoning_content_is_portable_and_never_stripped() -> None:
    adapter = get_protocol("openai_chat")
    parsed = adapter.parse_request(_chat_request())

    built = adapter.build_request(parsed, _ctx("openai_chat"))

    assistant = built["messages"][1]
    assert assistant["reasoning_content"] == "portable-reasoning"
    # reasoning_content is NOT an opaque carrier: it must not be reported.
    assert "reasoning_content" not in json.dumps(
        [{"field": warning.field} for warning in parsed.warnings if warning.code == _OPAQUE_CODE]
    )


def test_chat_reasoning_content_survives_with_no_signatures_present() -> None:
    adapter = get_protocol("openai_chat")
    parsed = adapter.parse_request(
        {
            "model": "model-a",
            "messages": [
                {"role": "assistant", "content": "a", "reasoning_content": "portable"},
            ],
        }
    )

    built = adapter.build_request(parsed, _ctx("openai_chat"))

    assert built["messages"][0]["reasoning_content"] == "portable"


# ----------------------------------------------------------------------------
# Cross-protocol sanity: a foreign source never leaks signatures
# ----------------------------------------------------------------------------


def test_cross_protocol_source_signature_never_reaches_target_wire() -> None:
    gemini = get_protocol("gemini")
    responses = get_protocol("responses")
    parsed = gemini.parse_request(_gemini_request())

    built = responses.build_request(
        parsed,
        ProtocolContext(source_protocol="gemini", target_protocol="responses"),
    )

    assert "sig-thought" not in json.dumps(built)
    assert "sig-text" not in json.dumps(built)
    assert "sig-call" not in json.dumps(built)


def test_unscoped_same_protocol_suppresses_signatures() -> None:
    """A missing provider domain is NOT proof of ownership."""

    adapter = get_protocol("gemini")
    parsed = adapter.parse_request(_gemini_request())

    built = adapter.build_request(parsed)

    assert "sig-thought" not in json.dumps(built)
