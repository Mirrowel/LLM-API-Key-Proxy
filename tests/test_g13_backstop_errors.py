"""G13 backstop pins: stream-side disclosure for unrepresentable content and
the per-dialect terminal error vocabulary.

Real parse->format flows drive the media/citation cases (the fabricated
fixture era hid both bugs); unknown-family blocks are hand-built because no
provider wire emits them.
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
    Annotation,
    ContentBlock,
    ProtocolContext,
    UnifiedMessage,
    UnifiedStreamEvent,
)

TARGETS = ("openai_chat", "anthropic_messages", "responses", "gemini")
_DROP_CODE = "unrepresentable_content_dropped"


def _context(target: str, source: str = "openai_chat") -> ProtocolContext:
    return ProtocolContext(
        model="model-a",
        source_protocol=source,
        target_protocol=target,
        transport="sse",
        request_id="req_g13",
    )


def _run(source: str, target: str, wire_frames: list) -> tuple[list[str], object]:
    """Feed real source wire frames through parse->format for one target."""

    ctx = _context(target, source)
    state = stream_format_state(ctx, target)
    source_protocol = get_protocol(source)
    frames: list[str] = []
    for raw in wire_frames:
        for event in source_protocol.parse_stream_events(raw, ctx):
            frames.extend(format_canonical_stream_event(event, target, ctx, state=state))
    return frames, state


def _run_event(target: str, event: UnifiedStreamEvent, source: str = "openai_chat") -> tuple[list[str], object]:
    ctx = _context(target, source)
    state = stream_format_state(ctx, target)
    event.source_protocol = event.source_protocol or source
    return list(format_canonical_stream_event(event, target, ctx, state=state)), state


def _payloads(frames: list[str]) -> list[dict]:
    payloads = []
    for frame in frames:
        for line in frame.splitlines():
            if not line.startswith("data: "):
                continue
            text = line[len("data: "):].strip()
            if text == "[DONE]":
                continue
            try:
                payloads.append(json.loads(text))
            except json.JSONDecodeError:
                pass
    return payloads


def _dropped(state) -> bool:
    return any(warning.code == _DROP_CODE for warning in state.warnings)


# ------------------------------------------------------------------ media


_IMAGE_WIRE = {
    "id": "chat_1",
    "model": "model-a",
    "choices": [{"index": 0, "delta": {"content": [{"type": "image_url", "image_url": {"url": "https://img.test/a.png"}}]}}],
}


@pytest.mark.parametrize("target", ("openai_chat", "anthropic_messages", "responses"))
def test_media_block_never_mints_a_phantom_block(target: str) -> None:
    frames, state = _run("openai_chat", target, [_IMAGE_WIRE])
    payloads = _payloads(frames)
    joined = "".join(frames)

    if target == "openai_chat":
        assert payloads == [], "chat must not emit a content-less chunk for media"
    elif target == "anthropic_messages":
        assert "content_block_start" not in joined, "no phantom empty text block"
    else:
        assert "response.output_item.added" not in joined, "no empty output item minted"
    assert _dropped(state), state.warnings


def test_gemini_media_emits_the_honest_part() -> None:
    frames, state = _run("openai_chat", "gemini", [_IMAGE_WIRE])
    joined = "".join(frames)
    assert "fileData" in joined or "inlineData" in joined
    assert not _dropped(state)
    # never a phantom text part for the media block
    assert '"text": ""' not in joined


# --------------------------------------------------------------- citations


_RESPONSES_ANNOTATION = {
    "type": "response.output_text.annotation.added",
    "output_index": 1,
    "annotation": {"type": "url_citation", "url": "https://cite.test/1", "title": "Cite"},
}


def test_citations_block_responses_skips_and_warns() -> None:
    frames, state = _run("responses", "responses", [_RESPONSES_ANNOTATION])
    joined = "".join(frames)
    assert "response.output_item.added" not in joined
    assert _dropped(state), state.warnings


def test_citations_block_gemini_skips_and_warns() -> None:
    frames, state = _run("responses", "gemini", [_RESPONSES_ANNOTATION])
    joined = "".join(frames)
    assert "candidates" not in joined
    assert _dropped(state), state.warnings


def test_citations_block_chat_reemits_annotations() -> None:
    frames, state = _run("responses", "openai_chat", [_RESPONSES_ANNOTATION])
    payloads = _payloads(frames)
    annotations = [a for payload in payloads for a in (payload.get("choices", [{}])[0].get("delta", {}).get("annotations") or [])]
    assert annotations and annotations[0]["url"] == "https://cite.test/1"
    assert not _dropped(state)


_ANTHROPIC_TEXT_OPEN = [
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "answer"}},
]
_ANTHROPIC_CITATION = {
    "type": "content_block_delta",
    "index": 0,
    "delta": {"type": "citations_delta", "citation": {"type": "web_search_result_location", "url": "https://a.test", "title": "A", "cited_text": "x"}},
}


def test_anthropic_citations_all_render_and_missing_text_warns() -> None:
    frames, state = _run("anthropic_messages", "anthropic_messages", [*_ANTHROPIC_TEXT_OPEN, _ANTHROPIC_CITATION, _ANTHROPIC_CITATION])
    rendered = [
        payload
        for payload in _payloads(frames)
        if payload.get("type") == "content_block_delta" and payload.get("delta", {}).get("type") == "citations_delta"
    ]
    assert len(rendered) == 2, "every citation on the open text block renders its own frame"
    assert not _dropped(state)

    # No open text block: the citation has no attachment point -> disclosed.
    frames, state = _run("responses", "anthropic_messages", [_RESPONSES_ANNOTATION])
    assert "citations_delta" not in "".join(frames)
    assert _dropped(state), state.warnings


def test_anthropic_batched_citations_all_render() -> None:
    """A foreign block may batch several annotations; each renders its own frame.

    No single real Anthropic wire frame batches citations, so the batched
    block is constructed directly (the only way to pin annotations[0]-only).
    """

    event = UnifiedStreamEvent(
        type="message_delta",
        source_protocol="openai_chat",
        message=UnifiedMessage(
            role="assistant",
            content=[ContentBlock(type="text", text="t")],
        ),
    )
    ctx = _context("anthropic_messages", "openai_chat")
    state = stream_format_state(ctx, "anthropic_messages")
    frames = list(format_canonical_stream_event(event, "anthropic_messages", ctx, state=state))
    citation_event = UnifiedStreamEvent(
        type="message_delta",
        source_protocol="openai_chat",
        message=UnifiedMessage(
            role="assistant",
            content=[
                ContentBlock(
                    type="citations_delta",
                    annotations=[
                        Annotation(type="url_citation", url="https://one.test", raw={"type": "url_citation", "url": "https://one.test"}),
                        Annotation(type="url_citation", url="https://two.test", raw={"type": "url_citation", "url": "https://two.test"}),
                    ],
                )
            ],
        ),
    )
    frames.extend(format_canonical_stream_event(citation_event, "anthropic_messages", ctx, state=state))
    rendered = [
        payload
        for payload in _payloads(frames)
        if payload.get("type") == "content_block_delta" and payload.get("delta", {}).get("type") == "citations_delta"
    ]
    assert len(rendered) == 2


# ----------------------------------------------------------------- unknown


def _unknown_event() -> UnifiedStreamEvent:
    return UnifiedStreamEvent(
        type="message_delta",
        source_protocol="openai_chat",
        message=UnifiedMessage(role="assistant", content=[ContentBlock(type="unknown", raw={"mystery": True})]),
    )


@pytest.mark.parametrize("target", TARGETS)
def test_unknown_block_skips_and_warns_on_every_target(target: str) -> None:
    frames, state = _run_event(target, _unknown_event())
    joined = "".join(frames)
    assert "mystery" not in joined
    assert _dropped(state), state.warnings


def test_chat_builtin_record_skips_and_warns() -> None:
    from rotator_library.protocols.types import BuiltinToolCall

    event = UnifiedStreamEvent(
        type="message_delta",
        source_protocol="openai_chat",
        message=UnifiedMessage(
            role="assistant",
            content=[ContentBlock(type="builtin_tool", builtin_tool=BuiltinToolCall(kind="web_search", raw={"type": "web_search_call"}))],
        ),
    )
    frames, state = _run_event("openai_chat", event)
    assert _payloads(frames) == []
    assert _dropped(state), state.warnings


# ------------------------------------------------------------------- audio


def test_chat_stream_audio_emits_delta_audio() -> None:
    wire = {
        "id": "chat_1",
        "model": "model-a",
        "choices": [{"index": 0, "delta": {"audio": {"id": "aud_1", "data": "QUJD", "transcript": "hello", "format": "mp3"}}}],
    }
    frames, state = _run("openai_chat", "openai_chat", [wire])
    payloads = _payloads(frames)
    audio = next(
        payload["choices"][0]["delta"]["audio"]
        for payload in payloads
        if payload.get("choices") and payload["choices"][0].get("delta", {}).get("audio")
    )
    assert audio["id"] == "aud_1"
    assert audio["data"] == "QUJD"
    assert audio["transcript"] == "hello"
    assert audio["format"] == "mp3"
    assert not _dropped(state)


def test_chat_audio_without_inline_data_is_disclosed() -> None:
    wire = {
        "id": "chat_1",
        "model": "model-a",
        "choices": [{"index": 0, "delta": {"audio": {"url": "https://audio.test/a.mp3"}}}],
    }
    frames, state = _run("openai_chat", "openai_chat", [wire])
    assert _payloads(frames) == []
    assert _dropped(state), state.warnings


# --------------------------------------------------------------- errors


def _error_event(error) -> UnifiedStreamEvent:
    return UnifiedStreamEvent(type="error", error=error)


def _render_error(target: str, error) -> dict:
    frames, _ = _run_event(target, _error_event(error))
    payloads = _payloads(frames)
    if target == "anthropic_messages":
        error_payloads = [payload["error"] for payload in payloads if payload.get("type") == "error"]
    elif target == "responses":
        error_payloads = [payload["response"]["error"] for payload in payloads if payload.get("type") == "response.failed"]
    else:
        error_payloads = [payload["error"] for payload in payloads if "error" in payload]
    assert error_payloads, frames
    return error_payloads[0]


@pytest.mark.parametrize(
    ("error", "family_expectations"),
    [
        (
            {"type": "rate_limit_error", "message": "slow"},
            {"chat_type": "rate_limit_error", "anthropic_type": "rate_limit_error", "responses_code": "rate_limit_exceeded", "gemini": (429, "RESOURCE_EXHAUSTED")},
        ),
        (
            {"type": "overloaded_error", "message": "busy"},
            {"chat_type": "server_error", "anthropic_type": "overloaded_error", "responses_code": "server_error", "gemini": (503, "UNAVAILABLE")},
        ),
        (
            {"type": "billing_error", "message": "pay up"},
            {"chat_type": "insufficient_quota", "anthropic_type": "billing_error", "responses_code": "rate_limit_exceeded", "gemini": (429, "RESOURCE_EXHAUSTED")},
        ),
        (
            {"status": "DEADLINE_EXCEEDED", "message": "slow"},
            {"chat_type": "server_error", "anthropic_type": "timeout_error", "responses_code": "server_error", "gemini": (504, "DEADLINE_EXCEEDED")},
        ),
        (
            {"type": "invalid_request_error", "message": "bad", "param": "messages"},
            {"chat_type": "invalid_request_error", "anthropic_type": "invalid_request_error", "responses_code": "invalid_prompt", "gemini": (400, "INVALID_ARGUMENT")},
        ),
        (
            {"type": "server_error", "message": "boom"},
            {"chat_type": "server_error", "anthropic_type": "api_error", "responses_code": "server_error", "gemini": (500, "INTERNAL")},
        ),
        (
            "provider exploded",
            {"chat_type": "server_error", "anthropic_type": "api_error", "responses_code": "server_error", "gemini": (500, "INTERNAL")},
        ),
        (
            "rate limit exceeded",
            {"chat_type": "rate_limit_error", "anthropic_type": "rate_limit_error", "responses_code": "rate_limit_exceeded", "gemini": (429, "RESOURCE_EXHAUSTED")},
        ),
    ],
)
def test_stream_error_vocabulary_per_dialect(error, family_expectations) -> None:
    chat = _render_error("openai_chat", error)
    assert chat["type"] == family_expectations["chat_type"]
    assert chat["message"]
    assert chat["param"] is None or isinstance(chat["param"], str)
    assert "code" in chat

    anthropic = _render_error("anthropic_messages", error)
    assert anthropic["type"] == family_expectations["anthropic_type"], anthropic
    assert "request_id" not in anthropic  # header-only fact, never fabricated

    responses = _render_error("responses", error)
    assert responses["code"] == family_expectations["responses_code"], responses

    gemini = _render_error("gemini", error)
    expected_code, expected_status = family_expectations["gemini"]
    assert gemini["code"] == expected_code
    assert gemini["status"] == expected_status


def test_chat_error_frame_keeps_done_trailer() -> None:
    frames, _ = _run_event("openai_chat", _error_event({"type": "server_error", "message": "boom"}))
    assert frames[-1] == "data: [DONE]\n\n"


def test_chat_dict_error_keeps_provider_body_keys() -> None:
    chat = _render_error("openai_chat", {"type": "rate_limit", "message": "slow", "custom_field": "kept"})
    assert chat["custom_field"] == "kept"
    assert chat["type"] == "rate_limit_error"
