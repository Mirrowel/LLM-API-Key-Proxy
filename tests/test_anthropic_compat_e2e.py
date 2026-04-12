# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
End-to-end tests for the Anthropic Messages API compatibility layer.

These tests simulate the full round-trip that Claude Code or other
Anthropic API clients would make:
1. Client sends Anthropic-format request to /v1/messages
2. Proxy translates to OpenAI format
3. (Mocked) RotatingClient returns OpenAI-format response
4. Proxy translates back to Anthropic format
5. Client receives Anthropic-format response

This is the integration point most likely to break during branch merges
because each branch may modify the translator, streaming, or request handling
differently.

NO network calls, NO API keys needed.
"""

import json

import pytest

from rotator_library.anthropic_compat.translator import (
    translate_anthropic_request,
    openai_to_anthropic_response,
)
from rotator_library.anthropic_compat.models import AnthropicMessagesRequest
from rotator_library.anthropic_compat.streaming import anthropic_streaming_wrapper


class TestAnthropicE2ESimpleText:
    """E2E test: simple text request → response."""

    def test_full_round_trip(self, anthropic_simple_request, openai_simple_response):
        """Simple text: Anthropic request → OpenAI format → OpenAI response → Anthropic response."""
        # Step 1: Translate request
        req = AnthropicMessagesRequest(**anthropic_simple_request)
        openai_req = translate_anthropic_request(req)

        # Verify request translation
        assert openai_req["model"] == "claude-sonnet-4-5"
        assert len(openai_req["messages"]) >= 1

        # Step 2: (In production, RotatingClient would send this to a provider)
        # Simulate receiving an OpenAI response

        # Step 3: Translate response back
        anthropic_resp = openai_to_anthropic_response(
            openai_simple_response,
            original_model="claude-sonnet-4-5",
        )

        # Verify response format
        assert anthropic_resp["type"] == "message"
        assert anthropic_resp["role"] == "assistant"
        assert len(anthropic_resp["content"]) >= 1
        assert anthropic_resp["content"][0]["type"] == "text"
        assert "Hello" in anthropic_resp["content"][0]["text"]
        assert anthropic_resp["stop_reason"] == "end_turn"
        assert "usage" in anthropic_resp
        assert anthropic_resp["usage"]["input_tokens"] > 0
        assert anthropic_resp["usage"]["output_tokens"] > 0


class TestAnthropicE2EToolUse:
    """E2E test: tool use request → tool call response."""

    def test_full_round_trip(self, anthropic_tool_request, openai_tool_response):
        """Tool use: request with tools → response with tool calls."""
        # Step 1: Translate request
        req = AnthropicMessagesRequest(**anthropic_tool_request)
        openai_req = translate_anthropic_request(req)

        # Verify tools translated
        assert "tools" in openai_req
        assert openai_req["tools"][0]["function"]["name"] == "get_weather"

        # Step 2: Translate response back
        anthropic_resp = openai_to_anthropic_response(
            openai_tool_response,
            original_model="claude-sonnet-4-5",
        )

        # Verify tool_use block
        tool_blocks = [b for b in anthropic_resp["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "get_weather"
        assert tool_blocks[0]["input"]["location"] == "New York, NY"
        assert anthropic_resp["stop_reason"] == "tool_use"


class TestAnthropicE2EThinking:
    """E2E test: thinking/extended reasoning request → response."""

    def test_full_round_trip(self, anthropic_thinking_request, openai_thinking_response):
        """Thinking: request with thinking → response with reasoning_content."""
        # Step 1: Translate request
        req = AnthropicMessagesRequest(**anthropic_thinking_request)
        openai_req = translate_anthropic_request(req)

        # Verify thinking → reasoning_effort
        assert "reasoning_effort" in openai_req

        # Step 2: Translate response back
        anthropic_resp = openai_to_anthropic_response(
            openai_thinking_response,
            original_model="claude-sonnet-4-5",
        )

        # Verify thinking block present
        thinking_blocks = [b for b in anthropic_resp["content"] if b["type"] == "thinking"]
        assert len(thinking_blocks) >= 1
        assert "Let me think" in thinking_blocks[0]["thinking"]

        # Verify text block also present
        text_blocks = [b for b in anthropic_resp["content"] if b["type"] == "text"]
        assert len(text_blocks) >= 1
        assert "42" in text_blocks[0]["text"]


class TestAnthropicE2EMultiTurn:
    """E2E test: multi-turn conversation with tool results."""

    def test_multiturn_preserves_context(self, anthropic_multiturn_request):
        """Multi-turn: conversation history is preserved in translation."""
        req = AnthropicMessagesRequest(**anthropic_multiturn_request)
        openai_req = translate_anthropic_request(req)

        # Should have multiple messages (user + assistant + tool_result)
        assert len(openai_req["messages"]) >= 2

        # Tool result should be present
        tool_msgs = [m for m in openai_req["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        assert "72°F" in tool_msgs[0]["content"]


class TestAnthropicE2EStreaming:
    """E2E test: streaming request produces valid Anthropic events."""

    @pytest.mark.asyncio
    async def test_streaming_round_trip(self, openai_streaming_chunks):
        """Streaming: OpenAI SSE chunks → Anthropic SSE events."""
        async def mock_stream():
            for chunk in openai_streaming_chunks:
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        result_stream = anthropic_streaming_wrapper(
            mock_stream(),
            original_model="claude-sonnet-4-5",
        )

        events = []
        async for event in result_stream:
            if event.strip():
                events.append(event)

        # Must produce a complete Anthropic message stream
        event_str = "\n".join(events)
        assert "message_start" in event_str
        assert "message_stop" in event_str
        assert "content_block_start" in event_str
        assert "content_block_delta" in event_str
