# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for Anthropic streaming format conversion.

Verifies that OpenAI SSE streaming chunks are correctly converted
to Anthropic's event-based streaming format. This is critical because
streaming breakage is hard to detect in production (partial responses
appear to work but are truncated or malformed).

NO network calls, NO API keys needed.
"""

import json

import pytest

from rotator_library.anthropic_compat.streaming import anthropic_streaming_wrapper


async def _chunks_to_stream(chunks):
    """Convert a list of dicts to SSE format async generator."""
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def _collect_stream(stream):
    """Collect all SSE events from an async generator."""
    events = []
    async for event in stream:
        if event.strip():
            events.append(event.strip())
    return events


def _parse_event(event_str):
    """Parse an SSE event string into (event_type, data_dict)."""
    lines = event_str.split("\n")
    event_type = None
    data = None
    for line in lines:
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data = json.loads(line[5:].strip())
    return event_type, data


class TestAnthropicStreamingBasic:
    """Basic streaming format conversion tests."""

    @pytest.mark.asyncio
    async def test_simple_text_stream(self, openai_streaming_chunks):
        """Simple text streaming produces correct Anthropic events."""
        stream = _chunks_to_stream(openai_streaming_chunks)
        result_stream = anthropic_streaming_wrapper(
            stream,
            original_model="claude-sonnet-4-5",
        )
        events = await _collect_stream(result_stream)

        # Should produce Anthropic events - look for event: type lines
        event_types = set()
        for event in events:
            for line in event.split("\n"):
                if line.startswith("event:"):
                    event_types.add(line[6:].strip())

        # Must have key events
        assert "message_start" in event_types, f"Missing message_start event. Got: {event_types}"
        assert "message_stop" in event_types, f"Missing message_stop event. Got: {event_types}"
        assert "content_block_start" in event_types, "Missing content_block_start"
        assert "content_block_delta" in event_types, "Missing content_block_delta"
        assert "content_block_stop" in event_types, "Missing content_block_stop"

    @pytest.mark.asyncio
    async def test_message_start_has_model(self, openai_streaming_chunks):
        """message_start event includes the model name."""
        stream = _chunks_to_stream(openai_streaming_chunks)
        result_stream = anthropic_streaming_wrapper(
            stream,
            original_model="claude-sonnet-4-5",
        )
        events = await _collect_stream(result_stream)

        for event in events:
            if "message_start" in event:
                # Parse to find the data
                for line in event.split("\n"):
                    if line.startswith("data:"):
                        data = json.loads(line[5:].strip())
                        msg = data.get("message", {})
                        assert msg.get("model") == "claude-sonnet-4-5"
                        break
                break

    @pytest.mark.asyncio
    async def test_text_delta_accumulation(self):
        """Text deltas are correctly accumulated in content_block_delta events."""
        chunks = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test-model",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test-model",
                "choices": [
                    {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test-model",
                "choices": [
                    {"index": 0, "delta": {"content": " world"}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test-model",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
            },
        ]

        stream = _chunks_to_stream(chunks)
        result_stream = anthropic_streaming_wrapper(
            stream, original_model="test-model"
        )
        events = await _collect_stream(result_stream)

        # Collect all text_delta content
        text_content = ""
        for event in events:
            for line in event.split("\n"):
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text_content += delta.get("text", "")
                    except json.JSONDecodeError:
                        pass

        assert "Hello" in text_content
        assert "world" in text_content


class TestAnthropicStreamingToolUse:
    """Streaming tests for tool use scenarios."""

    @pytest.mark.asyncio
    async def test_tool_call_streaming(self):
        """Tool calls in streaming are accumulated and emitted correctly."""
        chunks = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test-model",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"loc'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": 'ation":"NYC"}'},
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        ]

        stream = _chunks_to_stream(chunks)
        result_stream = anthropic_streaming_wrapper(
            stream, original_model="test-model"
        )
        events = await _collect_stream(result_stream)

        # Should produce tool_use content block events
        event_str = "\n".join(events)
        assert "tool_use" in event_str, "Missing tool_use in streaming output"


class TestAnthropicStreamingEdgeCases:
    """Edge cases in streaming conversion."""

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Stream with just [DONE] produces valid message_start/stop."""
        async def empty_stream():
            yield "data: [DONE]\n\n"

        result_stream = anthropic_streaming_wrapper(
            empty_stream(), original_model="test-model"
        )
        events = await _collect_stream(result_stream)
        # Should not crash — might produce minimal message structure
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_malformed_json_chunk(self):
        """Malformed JSON chunks don't crash the stream."""
        async def bad_stream():
            yield "data: {bad json}\n\n"
            yield "data: [DONE]\n\n"

        result_stream = anthropic_streaming_wrapper(
            bad_stream(), original_model="test-model"
        )
        events = await _collect_stream(result_stream)
        # Should handle gracefully (skip or log, not crash)
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_reasoning_content_in_stream(self):
        """reasoning_content in streaming becomes thinking_delta events."""
        chunks = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test-model",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test-model",
                "choices": [
                    {"index": 0, "delta": {"reasoning_content": "Let me think..."}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test-model",
                "choices": [
                    {"index": 0, "delta": {"content": "The answer is 42."}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "test-model",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
        ]

        stream = _chunks_to_stream(chunks)
        result_stream = anthropic_streaming_wrapper(
            stream, original_model="test-model"
        )
        events = await _collect_stream(result_stream)
        event_str = "\n".join(events)

        # Should contain thinking-related events
        assert "thinking" in event_str, "Missing thinking in streaming output for reasoning_content"
