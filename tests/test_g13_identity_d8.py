"""G13 slice B pins: D8 opaque-state gating unification, provider identity
preservation, logprobs/obfuscation riders, and dead-code removal guards.

Opaque provider state (anthropic thinking signatures, Gemini thought
signatures, Responses ``encrypted_content``) may only leave the stream cache
when the destination protocol matches the stream's source protocol. The
non-stream path keys on provider identity; the stream formatter has none, so
protocol equality on BOTH the latched stream source and the current event's
source is the gate. Cross-protocol state is extract-to-cache only.
"""

from __future__ import annotations

import json

import pytest

from rotator_library.protocols.registry import get_protocol
from rotator_library.protocols.streaming import (
    format_canonical_stream_event,
    stream_format_state,
)
from rotator_library.protocols.types import (
    ContentBlock,
    ProtocolContext,
    ReasoningBlock,
    UnifiedMessage,
    UnifiedStreamEvent,
)

_D8_CODE = "opaque_state_suppressed"


def _ctx(target: str, source: str) -> ProtocolContext:
    return ProtocolContext(
        model="model-a",
        source_protocol=source,
        target_protocol=target,
        transport="sse",
        request_id="req_d8",
    )


def _payloads(frames: list[str]) -> list[dict]:
    payloads = []
    for frame in frames:
        for line in frame.splitlines():
            if not line.startswith("data: "):
                continue
            text = line[len("data: ") :].strip()
            if text == "[DONE]":
                continue
            try:
                payloads.append(json.loads(text))
            except json.JSONDecodeError:
                pass
    return payloads


def _warning_codes(state) -> set[str]:
    return {warning.code for warning in state.warnings}


def _reasoning_event(
    source: str,
    *,
    text: str = "",
    signature: str | None = None,
    encrypted: str | None = None,
) -> UnifiedStreamEvent:
    return UnifiedStreamEvent(
        type="message_delta",
        source_protocol=source,
        message=UnifiedMessage(
            role="assistant",
            content=[
                ContentBlock(
                    type="reasoning",
                    reasoning=ReasoningBlock(
                        text=text,
                        signature=signature,
                        encrypted_content=encrypted,
                    ),
                )
            ],
        ),
    )


# ----------------------------------------------------------------- D8 matrix


def test_d8_anthropic_signature_same_protocol_is_emitted() -> None:
    ctx = _ctx("anthropic_messages", "anthropic_messages")
    state = stream_format_state(ctx, "anthropic_messages")
    frames = list(format_canonical_stream_event(_reasoning_event("anthropic_messages", text="t", signature="sig-ok"), "anthropic_messages", ctx, state=state))
    frames += list(format_canonical_stream_event(UnifiedStreamEvent(type="message_stop", source_protocol="anthropic_messages"), "anthropic_messages", ctx, state=state))
    assert "sig-ok" in "".join(frames)
    assert _D8_CODE not in _warning_codes(state)


def test_d8_anthropic_signature_cross_protocol_is_suppressed_and_warned() -> None:
    # Same target protocol, foreign source: the signature is provider-owned
    # and must not be translated into the Anthropic stream.
    ctx = _ctx("anthropic_messages", "gemini")
    state = stream_format_state(ctx, "anthropic_messages")
    frames = list(format_canonical_stream_event(_reasoning_event("gemini", text="visible", signature="sig-secret"), "anthropic_messages", ctx, state=state))
    frames += list(format_canonical_stream_event(UnifiedStreamEvent(type="message_stop", source_protocol="gemini"), "anthropic_messages", ctx, state=state))
    joined = "".join(frames)
    assert "visible" in joined
    assert "sig-secret" not in joined
    assert _D8_CODE in _warning_codes(state)


def test_d8_gemini_signature_cross_protocol_is_suppressed_and_warned() -> None:
    ctx = _ctx("gemini", "anthropic_messages")
    state = stream_format_state(ctx, "gemini")
    frames = list(format_canonical_stream_event(_reasoning_event("anthropic_messages", text="visible", signature="sig-secret"), "gemini", ctx, state=state))
    joined = "".join(frames)
    assert "visible" in joined
    assert "sig-secret" not in joined
    assert _D8_CODE in _warning_codes(state)


def test_d8_gemini_signature_same_protocol_is_emitted() -> None:
    ctx = _ctx("gemini", "gemini")
    state = stream_format_state(ctx, "gemini")
    frames = list(format_canonical_stream_event(_reasoning_event("gemini", text="t", signature="sig-ok"), "gemini", ctx, state=state))
    assert "sig-ok" in "".join(frames)
    assert _D8_CODE not in _warning_codes(state)


def test_d8_responses_encrypted_content_cross_protocol_suppressed_but_cached() -> None:
    ctx = _ctx("responses", "gemini")
    state = stream_format_state(ctx, "responses")
    frames = list(format_canonical_stream_event(_reasoning_event("gemini", text="r", encrypted="E-SECRET"), "responses", ctx, state=state))
    frames += list(format_canonical_stream_event(UnifiedStreamEvent(type="response.completed", source_protocol="gemini"), "responses", ctx, state=state))
    joined = "".join(frames)
    assert "E-SECRET" not in joined
    # Extract-to-cache: the value is still harvested for the field cache.
    assert "E-SECRET" in state.reasoning_encrypted.values()
    assert _D8_CODE in _warning_codes(state)


def test_d8_responses_encrypted_content_same_protocol_reemitted() -> None:
    ctx = _ctx("responses", "responses")
    state = stream_format_state(ctx, "responses")
    frames = list(format_canonical_stream_event(_reasoning_event("responses", text="r", encrypted="E-KEEP"), "responses", ctx, state=state))
    frames += list(format_canonical_stream_event(UnifiedStreamEvent(type="response.completed", source_protocol="responses"), "responses", ctx, state=state))
    assert "E-KEEP" in "".join(frames)
    assert _D8_CODE not in _warning_codes(state)


def test_d8_latched_source_alone_suppresses_when_events_lie() -> None:
    """The predicate consults the latch too: a same-target event on a stream
    whose latched source is foreign still suppresses (no per-event laundering)."""

    ctx = _ctx("gemini", "anthropic_messages")
    state = stream_format_state(ctx, "gemini")
    # First event latches the foreign source.
    list(format_canonical_stream_event(_reasoning_event("anthropic_messages", text="a"), "gemini", ctx, state=state))
    # A later event falsely claims the target protocol.
    frames = list(format_canonical_stream_event(_reasoning_event("gemini", text="b", signature="laundered"), "gemini", ctx, state=state))
    assert "laundered" not in "".join(frames)
    assert _D8_CODE in _warning_codes(state)


# ----------------------------------------------------------- identity lift


def _identity_state():
    from rotator_library.protocols.streaming import StreamFormatState

    return StreamFormatState(protocol="openai_chat", response_id="chatcmpl_minted", model="fallback-model")


def test_chat_identity_from_provider_is_preserved() -> None:
    state = _identity_state()
    ctx = _ctx("openai_chat", "openai_chat")
    wire = {"id": "chat_prov_9", "model": "prov-model", "choices": [{"index": 0, "delta": {"content": "hi"}}]}
    frames = [f for event in get_protocol("openai_chat").parse_stream_events(wire, ctx) for f in format_canonical_stream_event(event, "openai_chat", ctx, state=state)]
    payload = _payloads(frames)[0]
    assert payload["id"] == "chat_prov_9"
    assert payload["model"] == "prov-model"


def test_chat_identity_absent_falls_back_to_minted() -> None:
    state = _identity_state()
    ctx = _ctx("openai_chat", "openai_chat")
    wire = {"choices": [{"index": 0, "delta": {"content": "hi"}}]}
    frames = [f for event in get_protocol("openai_chat").parse_stream_events(wire, ctx) for f in format_canonical_stream_event(event, "openai_chat", ctx, state=state)]
    payload = _payloads(frames)[0]
    assert payload["id"] == "chatcmpl_minted"
    assert payload["model"] == "fallback-model"


def test_chat_fingerprint_and_service_tier_are_lifted() -> None:
    state = _identity_state()
    ctx = _ctx("openai_chat", "openai_chat")
    wire = {
        "system_fingerprint": "fp_abc",
        "service_tier": "flex",
        "choices": [{"index": 0, "delta": {"content": "hi"}}],
    }
    frames = [f for event in get_protocol("openai_chat").parse_stream_events(wire, ctx) for f in format_canonical_stream_event(event, "openai_chat", ctx, state=state)]
    payload = _payloads(frames)[0]
    assert payload["system_fingerprint"] == "fp_abc"
    assert payload["service_tier"] == "flex"


def test_gemini_identity_response_id_and_model_version_are_preserved() -> None:
    ctx = _ctx("gemini", "gemini")
    state = stream_format_state(ctx, "gemini")
    wire = {
        "responseId": "gem_resp_9",
        "modelVersion": "gem-model-2",
        "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}}],
    }
    frames = [f for event in get_protocol("gemini").parse_stream_events(wire, ctx) for f in format_canonical_stream_event(event, "gemini", ctx, state=state)]
    joined = "".join(frames)
    assert '"responseId": "gem_resp_9"' in joined
    assert '"modelVersion": "gem-model-2"' in joined


# --------------------------------------------------------------- riders


def test_logprobs_on_foreign_target_warns_once() -> None:
    ctx = _ctx("gemini", "openai_chat")
    state = stream_format_state(ctx, "gemini")
    wire = {"choices": [{"index": 0, "delta": {"content": "x"}, "logprobs": {"content": [{"token": "x"}]}}]}
    for event in get_protocol("openai_chat").parse_stream_events(wire, ctx):
        format_canonical_stream_event(event, "gemini", ctx, state=state)
    assert sum(1 for w in state.warnings if w.code == "logprobs_dropped") == 1


def test_logprobs_on_chat_target_is_not_warned() -> None:
    ctx = _ctx("openai_chat", "openai_chat")
    state = stream_format_state(ctx, "openai_chat")
    wire = {"choices": [{"index": 0, "delta": {"content": "x"}, "logprobs": {"content": [{"token": "x"}]}}]}
    frames = [f for event in get_protocol("openai_chat").parse_stream_events(wire, ctx) for f in format_canonical_stream_event(event, "openai_chat", ctx, state=state)]
    assert "logprobs" in "".join(frames)
    assert not any(w.code == "logprobs_dropped" for w in state.warnings)


def test_obfuscation_passthrough_on_chat_target() -> None:
    ctx = _ctx("openai_chat", "openai_chat")
    state = stream_format_state(ctx, "openai_chat")
    event = UnifiedStreamEvent(
        type="message_delta",
        source_protocol="openai_chat",
        message=UnifiedMessage(role="assistant", content=[ContentBlock(type="text", text="hi")]),
        extra={"obfuscation": {"seed": 42}},
    )
    frames = list(format_canonical_stream_event(event, "openai_chat", ctx, state=state))
    payload = _payloads(frames)[0]
    assert payload["choices"][0]["delta"]["obfuscation"] == {"seed": 42}


def test_obfuscation_does_not_leak_to_foreign_target() -> None:
    ctx = _ctx("gemini", "openai_chat")
    state = stream_format_state(ctx, "gemini")
    event = UnifiedStreamEvent(
        type="message_delta",
        source_protocol="openai_chat",
        message=UnifiedMessage(role="assistant", content=[ContentBlock(type="text", text="hi")]),
        extra={"obfuscation": {"seed": 42}},
    )
    frames = list(format_canonical_stream_event(event, "gemini", ctx, state=state))
    assert "obfuscation" not in "".join(frames)
    assert "seed" not in "".join(frames)


def test_gemini_skip_signature_sentinel_only_when_signatures_expected() -> None:
    ctx = _ctx("gemini", "gemini")
    state = stream_format_state(ctx, "gemini")
    source = get_protocol("gemini")

    signed = {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": "a", "args": {"x": 1}}, "thoughtSignature": "sig-a"}
    ]}}]}
    unsigned = {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": "b", "args": {"y": 2}}}
    ]}}]}
    frames = [f for event in source.parse_stream_events(signed, ctx) for f in format_canonical_stream_event(event, "gemini", ctx, state=state)]
    frames += [f for event in source.parse_stream_events(unsigned, ctx) for f in format_canonical_stream_event(event, "gemini", ctx, state=state)]
    joined = "".join(frames)
    assert "sig-a" in joined
    assert '"skip_thought_signature_validator": true' in joined


def test_gemini_no_sentinel_when_no_signatures_seen() -> None:
    ctx = _ctx("gemini", "gemini")
    state = stream_format_state(ctx, "gemini")
    source = get_protocol("gemini")
    wire = {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": "a", "args": {"x": 1}}}
    ]}}]}
    frames = [f for event in source.parse_stream_events(wire, ctx) for f in format_canonical_stream_event(event, "gemini", ctx, state=state)]
    assert "skip_thought_signature_validator" not in "".join(frames)


# ----------------------------------------------------------- dead code


def test_removed_stream_seams_are_gone() -> None:
    import rotator_library.protocols.streaming as streaming

    with pytest.raises(ImportError):
        from rotator_library.protocols.streaming import convert_protocol_stream  # noqa: F401

    assert not hasattr(streaming, "_ANTHROPIC_SERVER_TOOL_BLOCK_TYPES")
    assert not hasattr(streaming, "finish_emitted")
