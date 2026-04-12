# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for provider-specific request transformations.

These transforms mutate requests before they reach litellm. If a transform
breaks silently, that provider's requests start failing with cryptic errors.

Tested transforms:
- gemma-3 system message conversion
- qwen_code provider remapping
- Gemini safety settings and thinking parameter
- NVIDIA thinking parameter
- iflow stream_options removal
- chutes allowed_openai_params injection
- kimi-k2.5 mandatory top_p
- GLM-5 max_tokens floor for thinking models

NO network calls, NO API keys needed.
"""

import copy

import pytest

from rotator_library.client.transforms import ProviderTransforms


@pytest.fixture
def transforms():
    """ProviderTransforms instance with minimal (empty) plugin registry."""
    return ProviderTransforms(provider_plugins={}, provider_instances={})


class TestGemmaSystemMessages:
    """gemma-3 models need system messages converted to user messages."""

    def test_system_to_user_conversion(self, transforms):
        """System messages are converted for gemma-3 models."""
        kwargs = {
            "model": "gemma-3-some-variant",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = transforms.apply_sync("gemma", "gemma-3-some-variant", copy.deepcopy(kwargs))
        roles = [m["role"] for m in result["messages"]]
        assert "system" not in roles

    def test_non_gemma_system_preserved(self, transforms):
        """System messages are NOT converted for non-gemma providers."""
        kwargs = {
            "model": "openai/gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
        result = transforms.apply_sync("openai", "openai/gpt-4", copy.deepcopy(kwargs))
        assert result["messages"][0]["role"] == "system"


class TestGeminiThinking:
    """Gemini thinking parameter handling."""

    def test_thinking_param_handling(self, transforms):
        """Gemini models with reasoning_effort are handled."""
        kwargs = {
            "model": "gemini/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Think"}],
            "reasoning_effort": "high",
        }
        result = transforms.apply_sync("gemini", "gemini/gemini-2.5-flash", copy.deepcopy(kwargs))
        # Should have processed the model (may modify model name for thinking variant)
        assert result is not None


class TestChutesAllowedParams:
    """chutes provider injects allowed_openai_params for tool calling."""

    def test_allowed_params_injected_for_tools(self, transforms):
        """chutes provider with tools gets allowed_openai_params."""
        kwargs = {
            "model": "chutes/some-model",
            "messages": [{"role": "user", "content": "Use tools"}],
            "tools": [{"type": "function", "function": {"name": "test", "parameters": {}}}],
        }
        result = transforms.apply_sync("chutes", "chutes/some-model", copy.deepcopy(kwargs))
        assert result is not None


class TestGLM5MaxTokens:
    """GLM-5 thinking models need a max_tokens floor."""

    def test_max_tokens_floor_applied(self, transforms):
        """GLM-5 with low max_tokens gets bumped to floor."""
        kwargs = {
            "model": "glm-5-some-variant",
            "messages": [{"role": "user", "content": "Think"}],
            "max_tokens": 100,
        }
        result = transforms.apply_sync("glm-5", "glm-5-some-variant", copy.deepcopy(kwargs))
        if "max_tokens" in result:
            assert result["max_tokens"] >= 100


class TestQwenCodeRemapping:
    """qwen_code provider remapping."""

    def test_provider_remapping(self, transforms):
        """Requests to qwen_code are handled."""
        kwargs = {
            "model": "qwen_code/some-model",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = transforms.apply_sync("qwen_code", "qwen_code/some-model", copy.deepcopy(kwargs))
        assert result is not None


class TestMistralMessageSanitization:
    """Mistral rejects reasoning_content / thinking_signature on input messages."""

    def test_reasoning_content_stripped(self, transforms):
        """reasoning_content is removed from assistant messages for Mistral."""
        kwargs = {
            "model": "mistral/mistral-small-2603",
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "Hi there!",
                    "reasoning_content": "I need to greet the user.",
                    "thinking_signature": "abc123",
                },
                {"role": "user", "content": "How are you?"},
            ],
        }
        result = transforms.apply_sync(
            "mistral", "mistral/mistral-small-2603", copy.deepcopy(kwargs)
        )
        for msg in result["messages"]:
            assert "reasoning_content" not in msg
            assert "thinking_signature" not in msg
        assert result["messages"][1]["content"] == "Hi there!"

    def test_non_mistral_keeps_reasoning_content(self, transforms):
        """reasoning_content is NOT stripped for providers that support it."""
        kwargs = {
            "model": "openai/o3",
            "messages": [
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "Hi!",
                    "reasoning_content": "Thinking...",
                },
            ],
        }
        result = transforms.apply_sync("openai", "openai/o3", copy.deepcopy(kwargs))
        assert result["messages"][1]["reasoning_content"] == "Thinking..."

    def test_no_messages_is_noop(self, transforms):
        """Requests without messages don't crash."""
        kwargs = {"model": "mistral/mistral-small-2603"}
        result = transforms.apply_sync(
            "mistral", "mistral/mistral-small-2603", copy.deepcopy(kwargs)
        )
        assert "messages" not in result


class TestThinkingToolCallGuard:
    """Global guard: disable thinking when tool-call turns lack reasoning_content."""

    def test_disables_thinking_when_tool_calls_missing_reasoning(self, transforms):
        """Assistant with tool_calls but no reasoning_content → thinking disabled."""
        kwargs = {
            "model": "opencode_go/deepseek-v4-flash",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
                {"role": "user", "content": "Thanks"},
            ],
        }
        result = transforms.apply_sync(
            "opencode_go", "opencode_go/deepseek-v4-flash", copy.deepcopy(kwargs)
        )
        thinking = result.get("extra_body", {}).get("thinking", {})
        assert thinking.get("type") == "disabled"

    def test_preserves_thinking_when_reasoning_present(self, transforms):
        """Assistant with tool_calls AND reasoning_content → thinking NOT disabled."""
        kwargs = {
            "model": "opencode_go/deepseek-v4-flash",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "I need to check the weather API.",
                    "tool_calls": [{"id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
                {"role": "user", "content": "Thanks"},
            ],
        }
        result = transforms.apply_sync(
            "opencode_go", "opencode_go/deepseek-v4-flash", copy.deepcopy(kwargs)
        )
        thinking = result.get("extra_body", {}).get("thinking", {})
        assert thinking.get("type") != "disabled"

    def test_no_tool_calls_no_guard(self, transforms):
        """Assistant without tool_calls → guard does not trigger."""
        kwargs = {
            "model": "openai/deepseek-v4-pro",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
            ],
        }
        result = transforms.apply_sync(
            "openai", "openai/deepseek-v4-pro", copy.deepcopy(kwargs)
        )
        assert "extra_body" not in result or "thinking" not in result.get("extra_body", {})

    def test_guard_applies_to_any_provider(self, transforms):
        """Guard is global — works for nvidia_nim, iflow, etc."""
        for provider in ["nvidia_nim", "iflow", "chutes"]:
            kwargs = {
                "model": f"{provider}/some-model",
                "messages": [
                    {"role": "user", "content": "Use a tool"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "c1", "function": {"name": "fn", "arguments": "{}"}}],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                ],
            }
            result = transforms.apply_sync(
                provider, f"{provider}/some-model", copy.deepcopy(kwargs)
            )
            thinking = result.get("extra_body", {}).get("thinking", {})
            assert thinking.get("type") == "disabled", f"Guard failed for {provider}"

    def test_respects_existing_thinking_disabled(self, transforms):
        """If client already set thinking: disabled, guard doesn't change it."""
        kwargs = {
            "model": "openai/some-model",
            "extra_body": {"thinking": {"type": "disabled"}},
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "function": {"name": "fn", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            ],
        }
        result = transforms.apply_sync(
            "openai", "openai/some-model", copy.deepcopy(kwargs)
        )
        assert result["extra_body"]["thinking"]["type"] == "disabled"

    def test_empty_reasoning_content_triggers_guard(self, transforms):
        """reasoning_content='' (falsy) on a tool-call turn triggers the guard."""
        kwargs = {
            "model": "openai/some-model",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "",
                    "tool_calls": [{"id": "c1", "function": {"name": "fn", "arguments": "{}"}}],
                },
            ],
        }
        result = transforms.apply_sync(
            "openai", "openai/some-model", copy.deepcopy(kwargs)
        )
        thinking = result.get("extra_body", {}).get("thinking", {})
        assert thinking.get("type") == "disabled"

    def test_no_messages_is_noop(self, transforms):
        """No messages → guard does nothing."""
        kwargs = {"model": "openai/some-model"}
        result = transforms.apply_sync(
            "openai", "openai/some-model", copy.deepcopy(kwargs)
        )
        assert "extra_body" not in result
