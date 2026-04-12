# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for Anthropic↔OpenAI format translation.

These tests verify the bidirectional translation between Anthropic's Messages API
format and OpenAI's Chat Completions format. This is one of the most critical
integration points — breakage here silently corrupts all Claude Code / Anthropic
client requests.

NO network calls, NO API keys needed.
"""

import pytest

from rotator_library.anthropic_compat.translator import (
    translate_anthropic_request,
    openai_to_anthropic_response,
    _budget_to_reasoning_effort,
    _reorder_assistant_content,
)
from rotator_library.anthropic_compat.models import AnthropicMessagesRequest


# =============================================================================
# Request Translation: Anthropic → OpenAI
# =============================================================================


class TestTranslateAnthropicRequest:
    """Test Anthropic Messages API → OpenAI Chat Completions format."""

    def test_simple_text_request(self, anthropic_simple_request):
        """Basic single-turn text message translates correctly."""
        req = AnthropicMessagesRequest(**anthropic_simple_request)
        result = translate_anthropic_request(req)

        assert result["model"] == "claude-sonnet-4-5"
        assert result["max_tokens"] == 1024
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello, world!"

    def test_system_message_extraction(self):
        """Anthropic 'system' field becomes an OpenAI system message."""
        req = AnthropicMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hi"}],
        )
        result = translate_anthropic_request(req)

        messages = result["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"

    def test_tool_translation(self, anthropic_tool_request):
        """Anthropic tools with input_schema become OpenAI tools with parameters."""
        req = AnthropicMessagesRequest(**anthropic_tool_request)
        result = translate_anthropic_request(req)

        assert "tools" in result
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert "parameters" in tool["function"]
        assert tool["function"]["parameters"]["properties"]["location"]["type"] == "string"

    def test_tool_choice_translation(self):
        """Anthropic tool_choice types map to OpenAI equivalents."""
        # type: "auto" stays "auto"
        req = AnthropicMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{
                "name": "test_tool",
                "input_schema": {"type": "object", "properties": {}},
            }],
            tool_choice={"type": "auto"},
        )
        result = translate_anthropic_request(req)
        assert result["tool_choice"] == "auto"

        # type: "any" → "required"
        req.tool_choice = {"type": "any"}
        result = translate_anthropic_request(req)
        assert result["tool_choice"] == "required"

        # type: "tool" → specific function choice
        req.tool_choice = {"type": "tool", "name": "test_tool"}
        result = translate_anthropic_request(req)
        assert result["tool_choice"]["type"] == "function"
        assert result["tool_choice"]["function"]["name"] == "test_tool"

    def test_thinking_enabled(self, anthropic_thinking_request):
        """Thinking enabled → reasoning_effort is set."""
        req = AnthropicMessagesRequest(**anthropic_thinking_request)
        result = translate_anthropic_request(req)

        assert "reasoning_effort" in result
        # budget_tokens=10000 → medium (10000 <= 16384)
        assert result["reasoning_effort"] in ("medium", "low_medium")

    def test_thinking_disabled(self):
        """Thinking disabled → reasoning_effort = 'disable'."""
        req = AnthropicMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hi"}],
            thinking={"type": "disabled"},
        )
        result = translate_anthropic_request(req)
        assert result.get("reasoning_effort") == "disable"

    def test_image_block_translation(self):
        """Anthropic image blocks become OpenAI image_url format."""
        req = AnthropicMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo=",
                            },
                        },
                        {"type": "text", "text": "Describe this image"},
                    ],
                }
            ],
        )
        result = translate_anthropic_request(req)
        msg = result["messages"][0]
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "image_url"
        assert "data:image/png;base64," in msg["content"][0]["image_url"]["url"]

    def test_tool_result_translation(self):
        """Anthropic tool_result blocks become OpenAI tool messages."""
        req = AnthropicMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "Check weather"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_abc",
                            "name": "get_weather",
                            "input": {"city": "NYC"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc",
                            "content": "72°F, sunny",
                        }
                    ],
                },
            ],
        )
        result = translate_anthropic_request(req)

        # Should have: user, assistant (tool_calls), tool (response)
        roles = [m["role"] for m in result["messages"]]
        assert "tool" in roles

        tool_msg = [m for m in result["messages"] if m["role"] == "tool"][0]
        assert tool_msg["tool_call_id"] == "toolu_abc"

    def test_multiturn_with_thinking(self, anthropic_multiturn_request):
        """Multi-turn conversations with thinking blocks translate correctly."""
        req = AnthropicMessagesRequest(**anthropic_multiturn_request)
        result = translate_anthropic_request(req)

        # Should not crash, should have multiple messages
        assert len(result["messages"]) >= 2
        # Thinking blocks should be handled (not dropped silently)
        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        assert len(assistant_msgs) >= 1

    def test_empty_messages_handled(self):
        """Edge case: empty content list doesn't crash."""
        req = AnthropicMessagesRequest(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": ""}],
        )
        result = translate_anthropic_request(req)
        assert "messages" in result


# =============================================================================
# Response Translation: OpenAI → Anthropic
# =============================================================================


class TestOpenAIToAnthropicResponse:
    """Test OpenAI Chat Completions → Anthropic Messages format."""

    def test_simple_text_response(self, openai_simple_response):
        """Basic text response translates to Anthropic format."""
        result = openai_to_anthropic_response(
            openai_simple_response,
            original_model="claude-sonnet-4-5",
        )

        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-sonnet-4-5"
        assert len(result["content"]) >= 1

        text_block = result["content"][0]
        assert text_block["type"] == "text"
        assert text_block["text"] == "Hello! How can I help you?"

        # Check stop_reason
        assert result["stop_reason"] == "end_turn"

        # Check usage
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 8

    def test_tool_use_response(self, openai_tool_response):
        """Tool calls in OpenAI format become tool_use blocks."""
        result = openai_to_anthropic_response(
            openai_tool_response,
            original_model="claude-sonnet-4-5",
        )

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "get_weather"
        assert tool_blocks[0]["input"] == {"location": "New York, NY"}
        assert result["stop_reason"] == "tool_use"

    def test_thinking_response(self, openai_thinking_response):
        """reasoning_content becomes thinking blocks."""
        result = openai_to_anthropic_response(
            openai_thinking_response,
            original_model="claude-sonnet-4-5",
        )

        thinking_blocks = [b for b in result["content"] if b["type"] == "thinking"]
        assert len(thinking_blocks) >= 1
        assert "Let me think" in thinking_blocks[0]["thinking"]

    def test_finish_reason_mapping(self):
        """OpenAI finish_reasons map to Anthropic stop_reasons."""
        for openai_reason, anthropic_reason in [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
        ]:
            response = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "test"},
                        "finish_reason": openai_reason,
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
            result = openai_to_anthropic_response(response, original_model="test-model")
            assert result["stop_reason"] == anthropic_reason, (
                f"Expected {anthropic_reason} for {openai_reason}, got {result['stop_reason']}"
            )

    def test_cached_tokens_in_usage(self):
        """Cached tokens are mapped to cache_read_input_tokens."""
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_tokens_details": {"cached_tokens": 30},
            },
        }
        result = openai_to_anthropic_response(response, original_model="test-model")
        assert result["usage"]["cache_read_input_tokens"] == 30
        # input_tokens should be prompt_tokens minus cached
        assert result["usage"]["input_tokens"] == 70


# =============================================================================
# Budget to Reasoning Effort Mapping
# =============================================================================


class TestBudgetToReasoningEffort:
    """Test thinking budget_tokens → reasoning_effort mapping."""

    def test_zero_budget(self):
        # 0 <= 4096 (minimal threshold) → simplified to "low" for non-granular
        result = _budget_to_reasoning_effort(0, "test-model")
        assert result in ("minimal", "low")

    def test_low_budget(self):
        assert _budget_to_reasoning_effort(5000, "test-model") == "low"

    def test_high_budget(self):
        assert _budget_to_reasoning_effort(50000, "test-model") == "high"

    def test_granular_provider(self):
        """Antigravity provider gets granular levels."""
        result = _budget_to_reasoning_effort(10000, "antigravity/test-model")
        assert result in ("low_medium", "medium")  # Granular level

    def test_non_granular_provider_simplifies(self):
        """Non-antigravity providers get simplified levels."""
        result = _budget_to_reasoning_effort(10000, "openai/test-model")
        assert result in ("low", "medium", "high")  # Simplified


# =============================================================================
# Content Reordering
# =============================================================================


class TestReorderAssistantContent:
    """Test that assistant content blocks are correctly reordered."""

    def test_thinking_before_text(self):
        """Thinking blocks must come before text blocks."""
        content = [
            {"type": "text", "text": "result"},
            {"type": "thinking", "thinking": "reasoning"},
        ]
        result = _reorder_assistant_content(content)
        types = [b["type"] for b in result]
        assert types.index("thinking") < types.index("text")

    def test_tool_use_after_text(self):
        """Tool use blocks must come after text blocks."""
        content = [
            {"type": "tool_use", "id": "t1", "name": "test", "input": {}},
            {"type": "text", "text": "let me check"},
        ]
        result = _reorder_assistant_content(content)
        types = [b["type"] for b in result]
        assert types.index("text") < types.index("tool_use")

    def test_single_block_unchanged(self):
        """Single-block content is returned as-is."""
        content = [{"type": "text", "text": "hello"}]
        result = _reorder_assistant_content(content)
        assert result == content

    def test_correct_order(self):
        """Full correct order: thinking → text → tool_use."""
        content = [
            {"type": "tool_use", "id": "t1", "name": "test", "input": {}},
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "let me"},
        ]
        result = _reorder_assistant_content(content)
        types = [b["type"] for b in result]
        assert types == ["thinking", "text", "tool_use"]
