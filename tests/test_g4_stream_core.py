"""G4 stream core pins: relay, repair, n>1, usage honesty, visibility."""

from __future__ import annotations

import json

import pytest

from rotator_library.client.stream_ops import ChatWireStreamAdapter, NeutralStreamPipeline, StreamUsageTracker
from rotator_library.native_provider.context import NativeProviderContext
from rotator_library.native_provider.executor import NativeProviderExecutor
from rotator_library.native_provider.http import NativeHTTPTransport
from rotator_library.protocols.types import ProtocolContext
from rotator_library.streaming.relay import StreamRepairState
from rotator_library.usage.costs import CostCalculator  # noqa: F401


def _chat_context(**overrides) -> NativeProviderContext:
    base = dict(
        provider="openai_test",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://provider.example/v1/chat/completions",
        input_protocol_name="openai_chat",
        client_protocol_name="openai_chat",
        operation="chat",
        headers={"Authorization": "Bearer test"},
    )
    base.update(overrides)
    return NativeProviderContext(**base)


def _protocol_context() -> ProtocolContext:
    return ProtocolContext(
        provider="openai_test",
        model="gpt-test",
        source_protocol="openai_chat",
        target_protocol="openai_chat",
        input_protocol="openai_chat",
        provider_protocol="openai_chat",
        client_protocol="openai_chat",
        transport="sse",
    )


class _FakeStreamClient:
    """httpx-style client streaming raw SSE text lines."""

    def __init__(self, frames: list[str]):
        self._frames = frames
        self.calls: list[dict] = []

    def stream(self, method, endpoint, headers=None, json=None, **kwargs):
        # httpx contract: sync call returning an async context manager.
        self.calls.append({"endpoint": endpoint, "headers": dict(headers or {}), "json": json})
        lines: list[str] = []
        for frame in self._frames:
            lines.extend(frame.splitlines())
        text_lines = lines
        outer = self

        class _Resp:
            status_code = 200

            def aiter_lines(self):
                async def _gen():
                    for line in text_lines:
                        yield line
                return _gen()

        class _Ctx:
            async def __aenter__(self):
                return _Resp()

            async def __aexit__(self, *args):
                return False

        return _Ctx()


def _sse(payload: dict) -> str:
    # Trailing blank line: SSE events are delimited by empty lines — without
    # it the decoder merges consecutive data frames.
    return "data: " + json.dumps(payload) + "\n\n"


def _delta(text: str, index: int = 0) -> dict:
    return {
        "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test",
        "choices": [{"index": index, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
    }


def _finish(reason: str | None, index: int = 0) -> dict:
    return {
        "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test",
        "choices": [{"index": index, "delta": {}, "finish_reason": reason}],
    }


def _usage_chunk(tokens: int = 5) -> dict:
    return {
        "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test",
        "choices": [],
        "usage": {"prompt_tokens": tokens, "completion_tokens": 2, "total_tokens": tokens + 2},
    }


async def _run_native(frames: list[str], context: NativeProviderContext):
    """Drive the native executor + pipeline exactly as the client layer does."""
    from rotator_library.streaming.relay import StreamRepairState

    repair = StreamRepairState()
    context.stream_repair_state = repair
    stream = NativeProviderExecutor().stream(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        context,
        NativeHTTPTransport(_FakeStreamClient(frames)),
    )
    protocol_context = _protocol_context()
    pipeline = NeutralStreamPipeline(
        client_protocol_name="openai_chat",
        protocol_context=protocol_context,
        model="gpt-test",
        repair_state=repair,
        relay_eligible=(
            context.client_protocol_name == context.protocol_name
            and context.protocol_name in ("openai_chat", "gemini")
            and not context.adapter_names
        ),
    )
    output = ""
    async for frame in pipeline.run(stream, usage_provider=lambda: getattr(context, "stream_usage_record", None)):
        output += frame
    return output, pipeline


# ---------------------------------------------------------------------------
# Relay: same protocol, no edits → provider bytes forwarded untouched


@pytest.mark.asyncio
async def test_relay_forwards_provider_bytes_identically():
    frames = [_sse(_delta("he")), _sse(_finish("stop")), _sse(_usage_chunk()), "data: [DONE]"]
    output, _ = await _run_native(frames, _chat_context())
    # Byte-identical relay: the provider's exact frames appear verbatim.
    assert _sse(_delta("he")) in output
    assert _sse(_finish("stop")) in output
    assert "data: [DONE]" in output
    # No formatter duplicates (exactly one [DONE])
    assert output.count("[DONE]") == 1


@pytest.mark.asyncio
async def test_relay_disengages_on_error():
    error_frame = "data: " + json.dumps({"error": {"message": "boom", "type": "server_error"}}) + "\n"
    frames = [_sse(_delta("he")), error_frame]
    with pytest.raises(Exception):
        await _run_native(frames, _chat_context())


# ---------------------------------------------------------------------------
# Repair: bare EOF, missing finish/usage


@pytest.mark.asyncio
async def test_bare_eof_repairs_finish_and_zero_usage():
    # Stream ends WITHOUT [DONE], without finish, without usage.
    frames = [_sse(_delta("hello"))]
    output, pipeline = await _run_native(frames, _chat_context())
    # Repaired terminal: finish_reason stop + zeros usage frame present.
    assert '"finish_reason":"stop"' in output.replace(" ", "")
    assert '"usage"' in output
    usage = pipeline.usage.usage_record
    assert usage.input_tokens == 0  # present-but-zeros, never fabricated


@pytest.mark.asyncio
async def test_bare_eof_flushes_held_reason_and_provider_reason_wins():
    # Intermediate finish says length, no usage, then EOF.
    frames = [_sse(_delta("par")), _sse(_finish("length"))]
    output, _ = await _run_native(frames, _chat_context())
    assert '"finish_reason":"length"' in output.replace(" ", "")


@pytest.mark.asyncio
async def test_tools_seen_without_provider_reason_infers_tool_calls():
    tool_delta = {
        "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test",
        "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "f", "arguments": "{}"}}]}, "finish_reason": None}],
    }
    frames = ["data: " + json.dumps(tool_delta) + "\n"]
    output, _ = await _run_native(frames, _chat_context())
    assert '"finish_reason":"tool_calls"' in output.replace(" ", "")


# ---------------------------------------------------------------------------
# n>1: siblings survive on every source


@pytest.mark.asyncio
async def test_native_chat_n1_siblings_survive():
    chunk = {
        "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test",
        "choices": [
            {"index": 0, "delta": {"content": "alpha"}, "finish_reason": None},
            {"index": 1, "delta": {"content": "beta"}, "finish_reason": None},
        ],
    }
    frames = ["data: " + json.dumps(chunk) + "\n", "data: [DONE]"]
    output, _ = await _run_native(frames, _chat_context())
    assert "alpha" in output
    assert "beta" in output


@pytest.mark.asyncio
async def test_chatwire_usage_final_frame_keeps_siblings():
    chunk = {
        "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test",
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": "stop"},
            {"index": 1, "delta": {}, "finish_reason": "stop"},
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
    }

    async def source():
        yield chunk

    adapter = ChatWireStreamAdapter("gpt-test")
    events = [event async for event in adapter.events(source(), StreamUsageTracker("gpt-test"))]
    indexes = sorted({getattr(e, "output_index", None) for e in events})
    assert indexes == [0, 1]


# ---------------------------------------------------------------------------
# Usage honesty


@pytest.mark.asyncio
async def test_late_empty_usage_object_does_not_zero():
    tracker = StreamUsageTracker("m")
    tracker.merge_usage_payload({"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    tracker.merge_usage_payload({"usage": {}})
    assert tracker.usage_record.input_tokens == 10
    assert tracker.usage_record.completion_tokens == 5


# ---------------------------------------------------------------------------
# include_usage gating (client's own request)


@pytest.mark.asyncio
async def test_include_usage_false_suppresses_client_usage_frame():
    frames = [_sse(_delta("hi")), _sse(_finish("stop")), _sse(_usage_chunk()), "data: [DONE]"]
    context = _chat_context()
    repair = StreamRepairState()
    context.stream_repair_state = repair
    stream = NativeProviderExecutor().stream(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        context,
        NativeHTTPTransport(_FakeStreamClient(frames)),
    )
    pipeline = NeutralStreamPipeline(
        client_protocol_name="openai_chat",
        protocol_context=_protocol_context(),
        model="gpt-test",
        repair_state=repair,
        client_include_usage=False,
        relay_eligible=False,  # force the formatter path so gating is observable
    )
    output = ""
    async for frame in pipeline.run(stream):
        output += frame
    # No usage-only terminal chunk; intermediate chunks carry no usage key.
    assert '"choices": []' not in output
    assert '"usage": null' not in output


@pytest.mark.asyncio
async def test_include_usage_true_emits_usage_frame():
    frames = [_sse(_delta("hi")), _sse(_finish("stop")), _sse(_usage_chunk()), "data: [DONE]"]
    context = _chat_context()
    repair = StreamRepairState()
    context.stream_repair_state = repair
    stream = NativeProviderExecutor().stream(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        context,
        NativeHTTPTransport(_FakeStreamClient(frames)),
    )
    pipeline = NeutralStreamPipeline(
        client_protocol_name="openai_chat",
        protocol_context=_protocol_context(),
        model="gpt-test",
        repair_state=repair,
        client_include_usage=True,
        relay_eligible=False,
    )
    output = ""
    async for frame in pipeline.run(stream):
        output += frame
    assert '"choices": []' in output.replace(" ", "") or '"choices":[]' in output.replace(" ", "")
    assert '"prompt_tokens": 5' in output


# ---------------------------------------------------------------------------
# Visibility: real attributes (text + reasoning)


def _visible(event) -> bool:
    return NeutralStreamPipeline._event_visible(event)


def _make_event(text: str = "", reasoning: str = "", tools: bool = False):
    from rotator_library.protocols.types import ContentBlock, ReasoningBlock, UnifiedMessage, UnifiedStreamEvent

    content = [ContentBlock(type="text", text=text)] if text else []
    reasoning_blocks = [ReasoningBlock(text=reasoning)] if reasoning else []
    message = UnifiedMessage(role="assistant", content=content, reasoning=reasoning_blocks)
    if tools:
        from rotator_library.protocols.types import ToolCall

        message = UnifiedMessage(role="assistant", content=[], tool_calls=[ToolCall(id="c", name="f", arguments="{}")])
    return UnifiedStreamEvent(type="message_delta", delta=message)


def test_text_delta_is_visible():
    assert _visible(_make_event(text="hello"))


def test_reasoning_only_delta_is_visible():
    assert _visible(_make_event(reasoning="thinking..."))


def test_tool_calls_visible():
    assert _visible(_make_event(tools=True))


def test_empty_metadata_not_visible():
    assert not _visible(_make_event())
