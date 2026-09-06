# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Pydantic models for the Anthropic Messages API.

These models define the request and response formats for Anthropic's Messages API,
enabling compatibility with Claude Code and other Anthropic API clients.
"""

from typing import Any, List, Optional, Union
from pydantic import BaseModel, ConfigDict


# --- Content Blocks ---
class AnthropicTextBlock(BaseModel):

    # D4 fidelity: unknown/extension fields survive the round-trip.
    model_config = ConfigDict(extra="allow")
    """Anthropic text content block."""

    type: str = "text"
    text: str


class AnthropicImageSource(BaseModel):

    # D4 fidelity: unknown/extension fields survive the round-trip.
    model_config = ConfigDict(extra="allow")
    """Anthropic image source for base64 images."""

    type: str = "base64"
    media_type: str
    data: str


class AnthropicImageBlock(BaseModel):

    # D4 fidelity: unknown/extension fields survive the round-trip.
    model_config = ConfigDict(extra="allow")
    """Anthropic image content block."""

    type: str = "image"
    source: AnthropicImageSource


class AnthropicToolUseBlock(BaseModel):

    # D4 fidelity: unknown/extension fields survive the round-trip.
    model_config = ConfigDict(extra="allow")
    """Anthropic tool use content block."""

    type: str = "tool_use"
    id: str
    name: str
    input: dict


class AnthropicToolResultBlock(BaseModel):

    # D4 fidelity: unknown/extension fields survive the round-trip.
    model_config = ConfigDict(extra="allow")
    """Anthropic tool result content block."""

    type: str = "tool_result"
    tool_use_id: str
    content: Union[str, List[Any]]
    is_error: Optional[bool] = None


# --- Message and Tool Definitions ---
class AnthropicMessage(BaseModel):

    # D4 fidelity: unknown/extension fields survive the round-trip.
    model_config = ConfigDict(extra="allow")
    """Anthropic message format."""

    role: str
    content: Union[
        str,
        List[
            Union[
                AnthropicTextBlock,
                AnthropicImageBlock,
                AnthropicToolUseBlock,
                AnthropicToolResultBlock,
                dict,
            ]
        ],
    ]


class AnthropicTool(BaseModel):

    # D4 fidelity: unknown/extension fields survive the round-trip.
    model_config = ConfigDict(extra="allow")
    """Anthropic tool definition."""

    name: str
    description: Optional[str] = None
    input_schema: dict


class AnthropicThinkingConfig(BaseModel):

    # D4 fidelity: unknown/extension fields survive the round-trip.
    model_config = ConfigDict(extra="allow")
    """Anthropic thinking configuration."""

    type: str  # "enabled" or "disabled"
    budget_tokens: Optional[int] = None


# --- Messages Request ---
class AnthropicMessagesRequest(BaseModel):
    """Anthropic Messages API request format.

    ``extra="allow"`` keeps unknown/extension fields alive through the
    pydantic round-trip so same-protocol transport preserves them (D4);
    the facade retires in W6+W9 when the route unifies on the runtime.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[AnthropicMessage]
    max_tokens: int
    system: Optional[Union[str, List[dict]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[dict] = None
    metadata: Optional[dict] = None
    thinking: Optional[AnthropicThinkingConfig] = None


# --- Messages Response ---
class AnthropicUsage(BaseModel):
    """Anthropic usage statistics."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AnthropicMessagesResponse(BaseModel):
    """Anthropic Messages API response format."""

    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[Union[AnthropicTextBlock, AnthropicToolUseBlock, dict]]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


# --- Count Tokens ---
class AnthropicCountTokensRequest(BaseModel):
    """Anthropic count_tokens API request format."""

    model: str
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[dict]]] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[dict] = None
    thinking: Optional[AnthropicThinkingConfig] = None


class AnthropicCountTokensResponse(BaseModel):
    """Anthropic count_tokens API response format."""

    input_tokens: int
