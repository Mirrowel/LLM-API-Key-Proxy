"""G13 formatter core pins: rotation survival, relay shadow, close-all
terminals, no-raise degrade, and real-source block identity."""

from __future__ import annotations

import pytest

from rotator_library.protocols.streaming import (
    ProtocolStreamConverter,
    StreamFormatState,
    format_canonical_stream_event,
    stream_format_state,
)
from rotator_library.protocols.registry import get_protocol
from rotator_library.protocols.types import ProtocolContext, UnifiedStreamEvent


def _context(client: str, source: str = "openai_chat") -> ProtocolContext:
    return ProtocolContext(
        model="model-a",
        source_protocol=source,
        target_protocol=client,
        transport="sse",
        request_id="req_pin",
    )


def _converter(client: str, source: str = "openai_chat") -> ProtocolStreamConverter:
    return ProtocolStreamConverter(get_protocol(source), get_protocol(client), _context(client, source))


# ---------------------------------------------------------------- rotation


def test_reset_for_attempt_survives_lifecycle_and_spent_reality() -> None:
    state = StreamFormatState(protocol="openai_chat", response_id="chatcmpl_x", model="m")
    state.started = True
    state.usage = None
    from rotator_library.protocols.types import Usage

    state.usage = Usage(input_tokens=11, output_tokens=7, total_tokens=18)
    state.warnings.append("kept")
    state.role_emitted = True
    state.terminal = True
    state.finished_choices.update({0, 1})
    state.stop_reason = "tool_calls"
    state.tool_arguments["tool:a"] = '{"x":'

    state.reset_for_attempt()

    assert state.terminal is False
    assert state.finished_choices == set()
    assert state.stop_reason is None
    assert state.tool_arguments == {}
    # survivors
    assert state.started is True
    assert state.usage is not None and state.usage.input_tokens == 11
    assert state.warnings == ["kept"]
    assert state.role_emitted is True


def test_error_terminal_no_longer_zeroes_the_retry() -> None:
    """Attempt 1's error frame latched terminal; attempt 2 must still emit."""

    converter = _converter("openai_chat")
    # attempt 1: error frame
    converter.convert({"error": {"message": "boom", "type": "server_error"}})
    assert converter.state.terminal is True
    # rotation happens
    converter.state.reset_for_attempt()
    out = converter.convert({"choices": [{"index": 0, "delta": {"role": "assistant", "content": "still "}}]})
    assert any("still " in frame for frame in out)
    out2 = converter.convert({"choices": [{"index": 0, "delta": {"content": "alive"}}]})
    assert any("alive" in frame for frame in out2)
    done = converter.convert("[DONE]")
    assert any("[DONE]" in frame for frame in done)


# ---------------------------------------------------------------- relay shadow


def test_relay_shadow_keeps_formatter_warm_for_disengage() -> None:
    state = StreamFormatState(protocol="openai_chat", response_id="chatcmpl_x", model="m")
    context = _context("openai_chat")
    chat = get_protocol("openai_chat")

    def wire(text: str) -> list:
        return chat.parse_stream_events(
            {"choices": [{"index": 0, "delta": {"content": text}}]}, context
        )

    # relay frames 1-2: observe-only, frames discarded but state accumulates
    state.observe_only = True
    for ev in wire("hello "):
        assert format_canonical_stream_event(ev, "openai_chat", context, state=state) == []
    for ev in wire("world"):
        assert format_canonical_stream_event(ev, "openai_chat", context, state=state) == []
    state.observe_only = False
    # role was consumed during shadow — post-disengage deltas must not need it
    frames = []
    for ev in wire("!!"):
        frames.extend(format_canonical_stream_event(ev, "openai_chat", context, state=state))
    joined = "".join(frames)
    assert "!!" in joined
    # disengage terminal: finish + [DONE] emit normally
    for ev in chat.parse_stream_events({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}, context):
        joined += "".join(format_canonical_stream_event(ev, "openai_chat", context, state=state))
    assert "finish_reason" in joined
    for ev in chat.parse_stream_events("[DONE]", context):
        joined += "".join(format_canonical_stream_event(ev, "openai_chat", context, state=state))
    assert "data: [DONE]" in joined


# ---------------------------------------------------------------- close-all


def test_chat_tail_closes_every_seen_choice() -> None:
    converter = _converter("openai_chat")
    # two candidates stream content, provider finishes only choice 0
    out = "".join(converter.convert({"choices": [
        {"index": 0, "delta": {"content": "first"}},
        {"index": 1, "delta": {"content": "second"}},
    ]}))
    out += "".join(converter.convert({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
    out += "".join(converter.convert("[DONE]"))
    # both choices must be finished on the wire
    import json as _json

    finishes = {}
    for line in out.splitlines():
        if line.startswith("data: {"):
            payload = _json.loads(line[6:])
            for choice in payload.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finishes[choice["index"]] = choice["finish_reason"]
    assert finishes.get(0) == "stop"
    assert finishes.get(1) == "stop"  # repaired close-all, never hangs an SDK


def test_gemini_tail_closes_every_seen_candidate() -> None:
    converter = _converter("gemini")
    source_gemini = get_protocol("gemini")
    ctx = converter.context
    out = ""
    for ev in source_gemini.parse_stream_events(
        {"candidates": [
            {"index": 0, "content": {"role": "model", "parts": [{"text": "a"}]}},
            {"index": 1, "content": {"role": "model", "parts": [{"text": "b"}]}},
        ], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5}},
        ctx,
    ):
        out += "".join(format_canonical_stream_event(ev, "gemini", ctx, state=converter.state))
    # only candidate 0 finished on the wire from the provider side
    for ev in source_gemini.parse_stream_events(
        {"candidates": [{"index": 0, "finishReason": "STOP"}]},
        ctx,
    ):
        out += "".join(format_canonical_stream_event(ev, "gemini", ctx, state=converter.state))
    # EOF repair tail: synthesized done for the stream
    done = UnifiedStreamEvent(type="done", stop_reason="STOP", source_protocol="gemini")
    out += "".join(format_canonical_stream_event(done, "gemini", ctx, state=converter.state))
    import json as _json

    finished_indexes = set()
    for line in out.splitlines():
        if line.startswith("data: {"):
            payload = _json.loads(line[6:])
            for cand in payload.get("candidates", []):
                if cand.get("finishReason"):
                    finished_indexes.add(cand["index"])
    assert 0 in finished_indexes  # provider finish survived
    assert 1 in finished_indexes  # closed by the repair


# ---------------------------------------------------------------- no-raise


def test_gemini_incomplete_arguments_degrade_not_raise() -> None:
    converter = _converter("gemini")
    converter.convert({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "call_1", "type": "function", "function": {"name": "w", "arguments": '{"a":'}}
    ]}}]})
    out = "".join(converter.convert("[DONE]"))
    assert "functionCall" in out
    assert any(w.code == "tool_arguments_incomplete" for w in converter.state.warnings)


# ---------------------------------------------------------------- identity


def test_real_chat_source_text_tool_text_three_blocks() -> None:
    """Chat-shaped sources (output_index always set, content_index never):
    text -> tool -> text must be three distinct blocks on the responses
    target (the fabricated-fixture era collapsed them)."""

    converter = _converter("responses")
    out = "".join(converter.convert({"choices": [{"index": 0, "delta": {"content": "before "}}]}))
    out += "".join(converter.convert({"choices": [{"index": 0, "delta": {"tool_calls": [
        {"index": 0, "id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ]}}]}))
    out += "".join(converter.convert({"choices": [{"index": 0, "delta": {"content": " after"}}]}))
    out += "".join(converter.convert({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}))
    out += "".join(converter.convert("[DONE]"))

    import json as _json

    added = []
    for line in out.splitlines():
        if line.startswith("data: {") and "response.output_item.added" in line:
            added.append(_json.loads(line[6:])["item"]["type"])
    assert added.count("message") == 2
    assert added.count("function_call") == 1


def test_gemini_idless_fragments_of_one_call_never_split() -> None:
    converter = _converter("gemini")
    source_gemini = get_protocol("gemini")
    ctx = converter.context
    # genuinely partial fragments of ONE id-less call (Gemini-legal shape)
    out = ""
    for wire_payload in (
        {"candidates": [{"index": 0, "content": {"role": "model", "parts": [
            {"functionCall": {"name": "search", "args": {"q": "pa"}}}
        ]}}]},
        {"candidates": [{"index": 0, "content": {"role": "model", "parts": [
            {"functionCall": {"name": "search", "args": {"q": "paris"}}}
        ]}}]},
    ):
        for ev in source_gemini.parse_stream_events(wire_payload, ctx):
            out += "".join(format_canonical_stream_event(ev, "gemini", ctx, state=converter.state))
    done = UnifiedStreamEvent(type="done", stop_reason="STOP", source_protocol="gemini")
    out += "".join(format_canonical_stream_event(done, "gemini", ctx, state=converter.state))
    # exactly one call on the wire (first complete object); the snapshot
    # correction after emit degrades disclosed, never a second call
    assert out.count("functionCall") == 1
    assert any(w.code == "tool_arguments_late_fragment" for w in converter.state.warnings)


# ---------------------------------------------------------------- responses [DONE]


_SDK_RESPONSE_FIELDS = (
    "id",
    "object",
    "created_at",
    "status",
    "model",
    "output",
    "parallel_tool_calls",
    "tool_choice",
    "tools",
    "reasoning",
    "usage",
    "error",
    "incomplete_details",
    "metadata",
)


def _responses_objects(out: str) -> list[dict]:
    import json as _json

    objects = []
    for line in out.splitlines():
        if line.startswith("data: {") and '"response"' in line:
            payload = _json.loads(line[6:])
            if isinstance(payload.get("response"), dict):
                objects.append(payload["response"])
    return objects


def test_responses_formatter_sequence_is_single_sourced_and_monotonic() -> None:
    """The formatter lane's own counter (created -> delta -> terminal) is the
    single sequence authority: strictly increasing with no repeats."""
    import json as _json

    converter = _converter("responses")
    out = "".join(converter.convert({"choices": [{"index": 0, "delta": {"content": "a"}}]}))
    out += "".join(converter.convert({"choices": [{"index": 0, "delta": {"content": "b"}}]}))
    out += "".join(converter.convert({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))

    sequences = []
    for line in out.splitlines():
        if line.startswith("data: {"):
            payload = _json.loads(line[6:])
            if "sequence_number" in payload:
                sequences.append(payload["sequence_number"])
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == len(sequences)
    assert sequences[0] == 0


def test_responses_formatter_objects_are_sdk_shaped() -> None:
    """Every synthesized created/in_progress/terminal object carries the SDK
    member set (one shared builder across the formatter lane)."""
    converter = _converter("responses")
    out = "".join(converter.convert({"choices": [{"index": 0, "delta": {"content": "hi"}}]}))
    out += "".join(converter.convert({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))

    objects = _responses_objects(out)
    assert objects
    for obj in objects:
        for field in _SDK_RESPONSE_FIELDS:
            assert field in obj, f"missing {field!r}: {sorted(obj)}"
        assert obj["object"] == "response"


def test_responses_terminals_have_no_chat_done_sentinel() -> None:
    converter = _converter("responses")
    out = "".join(converter.convert({"choices": [{"index": 0, "delta": {"content": "hi"}}]}))
    out += "".join(converter.convert("[DONE]"))
    assert "response.incomplete" in out or "response.completed" in out
    assert 'incomplete_details": {' not in out  # unknown reason never fabricates one
    assert "data: [DONE]" not in out
