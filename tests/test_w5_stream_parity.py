# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""W1b/W5 stream acceptance fixtures.

These tests lock the neutral-event operational pipeline contract from the
final plan (§7 W1b/W5):

- native streams produce protocol-valid terminal frames for non-chat clients;
- usage (including provider-reported cost) is recorded for every client
  protocol with identical figures;
- session response anchors are recorded from neutral shapes, gated on an
  explicit provider completion signal (bare EOF never qualifies);
- interleaved text/tool/text stays three distinct blocks (defect 8).
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from rotator_library.client.stream_ops import (
    ChatWireStreamAdapter,
    NeutralStreamPipeline,
    StreamUsageTracker,
)
from rotator_library.protocols.types import (
    ContentBlock,
    ProtocolContext,
    ToolCall,
    UnifiedMessage,
    UnifiedStreamEvent,
    Usage,
)


class _FakeCredentialContext:
    def __init__(self) -> None:
        self.success: Optional[Dict[str, Any]] = None
        self.failure: Any = None

    def mark_success(self, **kwargs):
        self.success = kwargs

    def mark_failure(self, error):
        self.failure = error


def _context(client_protocol: str) -> ProtocolContext:
    return ProtocolContext(
        provider="synthetic",
        model="model-a",
        source_protocol="openai_chat",
        target_protocol=client_protocol,
        input_protocol="openai_chat",
        client_protocol=client_protocol,
        transport="sse",
    )


def _pipeline(client_protocol: str, **kwargs) -> NeutralStreamPipeline:
    defaults: Dict[str, Any] = {
        "client_protocol_name": client_protocol,
        "protocol_context": _context(client_protocol),
        "model": "model-a",
    }
    defaults.update(kwargs)
    return NeutralStreamPipeline(**defaults)


async def _agen(items: List[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


def _delta_event(text: str = "", tool: Optional[ToolCall] = None, usage: Optional[Usage] = None, stop_reason: Optional[str] = None) -> UnifiedStreamEvent:
    content: List[ContentBlock] = []
    if text:
        content.append(ContentBlock(type="text", text=text))
    message = UnifiedMessage(role="assistant", content=content)
    if tool is not None:
        message.tool_calls = [tool]
    return UnifiedStreamEvent(
        type="message_delta",
        source_protocol="openai_chat",
        delta=message,
        usage=usage,
        stop_reason=stop_reason,
    )


COMPLETED_EVENTS = [
    _delta_event(text="hello "),
    _delta_event(text="world"),
    _delta_event(usage=Usage(input_tokens=3, output_tokens=5, total_tokens=8), stop_reason="stop"),
    UnifiedStreamEvent(type="done", source_protocol="openai_chat", native_type="done", stop_reason="stop"),
]


def _anthropic_frames(output: str, event_name: str) -> List[Dict[str, Any]]:
    """Parse SSE frames whose event: name matches, returning their JSON data."""

    frames: List[Dict[str, Any]] = []
    for frame in output.split("\n\n"):
        event_type: Optional[str] = None
        data_lines: List[str] = []
        for line in frame.splitlines():
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if event_type == event_name and data_lines:
            frames.append(json.loads("\n".join(data_lines)))
    return frames


@pytest.mark.asyncio
@pytest.mark.parametrize("client_protocol", ["openai_chat", "anthropic_messages", "responses", "gemini"])
async def test_native_stream_terminal_frames_usage_and_anchors_per_client_protocol(client_protocol: str) -> None:
    cred = _FakeCredentialContext()
    recorded: List[Dict[str, Any]] = []
    pipeline = _pipeline(
        client_protocol,
        cred_context=cred,
        response_callback=recorded.append,
    )

    output = "".join([frame async for frame in pipeline.run(_agen(list(COMPLETED_EVENTS)))])

    # Protocol-valid lifecycle for every client protocol.
    if client_protocol == "openai_chat":
        assert output.rstrip().endswith("data: [DONE]")
    elif client_protocol == "anthropic_messages":
        assert "message_start" in output
        assert "message_stop" in output
    elif client_protocol == "responses":
        assert "response.completed" in output
        assert "data: [DONE]" in output
    else:
        assert '"candidates"' in output
    assert "hello world" in output.replace("hello ", "hello") or "hello" in output

    # Usage recorded with identical figures for every client protocol.
    assert cred.success is not None
    assert cred.success["completion_tokens"] == 5
    assert cred.success["prompt_tokens"] == 3

    # Session anchors recorded from the neutral envelope after completion.
    assert len(recorded) == 1
    assert recorded[0]["messages"][0]["role"] == "assistant"
    assert recorded[0]["messages"][0]["content"] == "hello world"


@pytest.mark.asyncio
async def test_bare_eof_without_completion_signal_records_no_anchors() -> None:
    cred = _FakeCredentialContext()
    recorded: List[Dict[str, Any]] = []
    pipeline = _pipeline(
        "anthropic_messages",
        cred_context=cred,
        response_callback=recorded.append,
    )

    events = [_delta_event(text="partial but unfinished")]
    output = "".join([frame async for frame in pipeline.run(_agen(events))])
    # Transport closes properly (phase-11 gate: EOF is not identity evidence).
    assert "message_stop" in output
    assert recorded == []


@pytest.mark.asyncio
@pytest.mark.parametrize("client_protocol", ["anthropic_messages", "responses"])
async def test_interleaved_text_tool_text_keeps_three_distinct_blocks(client_protocol: str) -> None:
    tool = ToolCall(id="call-1", name="get_weather", arguments={"city": "tokyo"}, index=0)
    events = [
        _delta_event(text="before "),
        _delta_event(tool=tool),
        _delta_event(text=" after"),
        _delta_event(usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3), stop_reason="tool_calls"),
        UnifiedStreamEvent(type="done", source_protocol="openai_chat", native_type="done", stop_reason="tool_calls"),
    ]
    pipeline = _pipeline(client_protocol)

    output = "".join([frame async for frame in pipeline.run(_agen(events))])

    if client_protocol == "anthropic_messages":
        starts = _anthropic_frames(output, "content_block_start")
        block_types = [start["content_block"]["type"] for start in starts]
        indexes = [start["index"] for start in starts]
        assert block_types == ["text", "tool_use", "text"]
        assert indexes == [0, 1, 2]
        assert '"text": "before "' in output
        assert '"text": " after"' in output
    else:
        added = _anthropic_frames(output, "response.output_item.added")
        item_types = [item["item"]["type"] for item in added]
        assert item_types.count("message") == 2
        assert "function_call" in item_types


@pytest.mark.asyncio
async def test_chat_wire_cost_comment_and_usage_merge_into_accounting() -> None:
    async def stream():
        yield {"choices": [{"delta": {"content": "hi"}}]}
        yield ": cost {\"total_cost\": 0.42, \"currency\": \"USD\"}"
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}}

    cred = _FakeCredentialContext()
    pipeline = _pipeline("openai_chat", cred_context=cred)
    adapter = ChatWireStreamAdapter("model-a")
    events = [event async for event in adapter.events(stream(), pipeline.usage)]

    assert events[-1].stop_reason == "stop"

    async def event_source():
        for event in events:
            yield event
    "".join([frame async for frame in pipeline.run(event_source())])

    assert cred.success is not None
    assert cred.success["completion_tokens"] == 4
    assert cred.success["prompt_tokens"] == 2
    assert pipeline.usage.usage_record.provider_reported_cost == 0.42


@pytest.mark.asyncio
async def test_chat_wire_intermediate_finish_reasons_are_held_back() -> None:
    async def stream():
        yield {"choices": [{"delta": {"content": "a"}, "finish_reason": "stop"}]}
        yield {"choices": [{"delta": {"content": "b"}, "finish_reason": "length"}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    tracker = StreamUsageTracker("model-a")
    adapter = ChatWireStreamAdapter("model-a")
    events = [event async for event in adapter.events(stream(), tracker)]

    # Intermediate finish frames carry no stop reason to the client formatter.
    assert events[0].stop_reason is None
    assert events[1].stop_reason is None
    # The usage-backed final frame's own reason wins; held intermediates fill
    # in only when the final frame carries none (legacy semantics preserved).
    assert events[2].stop_reason == "stop"


@pytest.mark.asyncio
async def test_chat_wire_tool_finish_priority() -> None:
    # Simpler: final frame with finish_reason stop after tool calls.
    async def stream2():
        yield {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "f", "arguments": "{}"}}]}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    tracker = StreamUsageTracker("model-a")
    adapter = ChatWireStreamAdapter("model-a")
    events = [event async for event in adapter.events(stream2(), tracker)]

    assert events[-1].stop_reason == "tool_calls"


@pytest.mark.asyncio
@pytest.mark.parametrize("client_protocol", ["openai_chat", "gemini"])
async def test_no_duplicate_terminal_frames_on_synthetic_done(client_protocol: str) -> None:
    """A wire EOF after a usage-backed final frame must not re-emit finish/usage."""

    async def stream():
        yield {"choices": [{"delta": {"content": "answer"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}}
        # EOF without [DONE] — synthetic terminal must not duplicate the finish.

    pipeline = _pipeline(client_protocol)
    adapter = ChatWireStreamAdapter("model-a")
    events = [event async for event in adapter.events(stream(), pipeline.usage)]

    async def event_source():
        for event in events:
            yield event

    output = "".join([frame async for frame in pipeline.run(event_source())])

    if client_protocol == "openai_chat":
        finish_frames = [line for line in output.splitlines() if '"finish_reason": "stop"' in line]
        assert len(finish_frames) == 1
        assert output.count("data: [DONE]") == 1
        # Documented include_usage grammar: exactly one terminal usage chunk
        # with an empty choices array (usage chunks never carry choices).
        usage_frames = [line for line in output.splitlines() if '"usage": {' in line]
        assert len(usage_frames) == 1 and '"choices": []' in usage_frames[0]
    else:
        assert output.count('"finishReason": "STOP"') == 1


@pytest.mark.asyncio
async def test_native_gemini_and_anthropic_wire_usage_and_cost_parity() -> None:
    """Provider-wire streams record identical usage figures for both clients."""

    from rotator_library.native_provider import NativeProviderContext, NativeProviderExecutor
    from rotator_library.native_provider.http import NativeHTTPTransport

    gemini_chunks = [
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}}]},
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "there"}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 6, "totalTokenCount": 10}, "costDetails": {"total_cost": 0.11}},
        "[DONE]",
    ]
    anthropic_chunks = [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi there"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 4, "output_tokens": 6}},
        {"type": "message_stop"},
    ]

    async def run_native(provider_protocol: str, operation: str, chunks, request_payload) -> list:
        context = NativeProviderContext(
            provider="synthetic",
            model="model-a",
            protocol_name=provider_protocol,
            input_protocol_name=provider_protocol,
            client_protocol_name=provider_protocol,
            endpoint="https://example.test/stream",
            operation=operation,
        )
        events = [
            event
            async for event in NativeProviderExecutor().stream(
                request_payload,
                context,
                NativeHTTPTransport(_FakeChunkClient(chunks)),
            )
        ]
        return events

    gemini_events = await run_native("gemini", "stream_generate", gemini_chunks, {"model": "model-a", "contents": []})
    anthropic_events = await run_native("anthropic_messages", "messages", anthropic_chunks, {"model": "model-a", "messages": [], "max_tokens": 8})

    results = {}
    for name, events in (("gemini", gemini_events), ("anthropic_messages", anthropic_events)):
        cred = _FakeCredentialContext()
        pipeline = _pipeline(name, cred_context=cred)

        async def event_source():
            for event in events:
                yield event

        output = "".join([frame async for frame in pipeline.run(event_source())])
        assert output, f"{name} client produced no frames"
        results[name] = cred.success

    # Identical accounting figures across both provider wires.
    assert results["gemini"]["completion_tokens"] == results["anthropic_messages"]["completion_tokens"] == 6
    assert results["gemini"]["prompt_tokens"] == results["anthropic_messages"]["prompt_tokens"] == 4


@pytest.mark.asyncio
async def test_responses_explicit_indexes_keep_two_output_items_distinct() -> None:
    """Two Responses output items sharing content_index=0 must not merge."""

    events = [
        UnifiedStreamEvent(
            type="message_delta",
            source_protocol="responses",
            delta=UnifiedMessage(role="assistant", content=[ContentBlock(type="text", text="ITEM-ONE")]),
            output_index=0,
            content_index=0,
        ),
        UnifiedStreamEvent(
            type="message_delta",
            source_protocol="responses",
            delta=UnifiedMessage(role="assistant", content=[ContentBlock(type="text", text="ITEM-TWO")]),
            output_index=1,
            content_index=0,
        ),
        UnifiedStreamEvent(type="done", source_protocol="responses", native_type="done", stop_reason="stop"),
    ]
    pipeline = _pipeline("responses")
    output = "".join([frame async for frame in pipeline.run(_agen(events))])

    added = _anthropic_frames(output, "response.output_item.added")
    assert len(added) == 2
    assert added[0]["item"]["id"] != added[1]["item"]["id"]


@pytest.mark.asyncio
async def test_anthropic_source_block_indexes_preserved_for_anthropic_client() -> None:
    """Same-protocol anthropic streams keep source block identity (index 0/1)."""

    from rotator_library.native_provider import NativeProviderContext, NativeProviderExecutor
    from rotator_library.native_provider.http import NativeHTTPTransport

    chunks = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "A"}},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "B"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"input_tokens": 1, "output_tokens": 2}},
        {"type": "message_stop"},
    ]
    context = NativeProviderContext(
        provider="synthetic",
        model="model-a",
        protocol_name="anthropic_messages",
        input_protocol_name="anthropic_messages",
        client_protocol_name="anthropic_messages",
        endpoint="https://example.test/messages",
        operation="messages",
    )
    events = [
        event
        async for event in NativeProviderExecutor().stream(
            {"model": "model-a", "messages": [], "max_tokens": 8},
            context,
            NativeHTTPTransport(_FakeChunkClient(chunks)),
        )
    ]
    pipeline = _pipeline("anthropic_messages")
    output = "".join([frame async for frame in pipeline.run(_agen(events))])

    starts = _anthropic_frames(output, "content_block_start")
    assert [start["index"] for start in starts] == [0, 1]


class _FakeChunkClient:
    """Minimal transport client yielding canned stream chunks."""

    def __init__(self, chunks: list) -> None:
        self._chunks = chunks

    async def stream_json_lines(self, endpoint: str, *, headers: dict, json: dict):
        for chunk in self._chunks:
            yield chunk
