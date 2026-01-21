# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Anthropic API compatibility module for rotator_library.

This module provides format translation between Anthropic's Messages API
and OpenAI's Chat Completions API, enabling any OpenAI-compatible provider
to work with Anthropic clients like Claude Code.

Usage:
    from rotator_library.anthropic_compat import (
        AnthropicMessagesRequest,
        AnthropicMessagesResponse,
        translate_anthropic_request,
        openai_to_anthropic_response,
        anthropic_streaming_wrapper,
    )
"""

from .models import (
    AnthropicTextBlock,
    AnthropicImageSource,
    AnthropicImageBlock,
    AnthropicToolUseBlock,
    AnthropicToolResultBlock,
    AnthropicMessage,
    AnthropicTool,
    AnthropicThinkingConfig,
    AnthropicMessagesRequest,
    AnthropicUsage,
    AnthropicMessagesResponse,
    AnthropicCountTokensRequest,
    AnthropicCountTokensResponse,
)

from .translator import (
    anthropic_to_openai_messages,
    anthropic_to_openai_tools,
    anthropic_to_openai_tool_choice,
    openai_to_anthropic_response,
    translate_anthropic_request,
)

from .streaming import anthropic_streaming_wrapper

__all__ = [
    # Models
    "AnthropicTextBlock",
    "AnthropicImageSource",
    "AnthropicImageBlock",
    "AnthropicToolUseBlock",
    "AnthropicToolResultBlock",
    "AnthropicMessage",
    "AnthropicTool",
    "AnthropicThinkingConfig",
    "AnthropicMessagesRequest",
    "AnthropicUsage",
    "AnthropicMessagesResponse",
    "AnthropicCountTokensRequest",
    "AnthropicCountTokensResponse",
    # Translator functions
    "anthropic_to_openai_messages",
    "anthropic_to_openai_tools",
    "anthropic_to_openai_tool_choice",
    "openai_to_anthropic_response",
    "translate_anthropic_request",
    # Streaming
    "anthropic_streaming_wrapper",
]
