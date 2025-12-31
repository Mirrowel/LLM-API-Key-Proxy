"""
Format translation functions between Anthropic and OpenAI API formats.

This module provides functions to convert requests and responses between
Anthropic's Messages API format and OpenAI's Chat Completions API format.
This enables any OpenAI-compatible provider to work with Anthropic clients.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Union

from .models import AnthropicMessagesRequest

MIN_THINKING_SIGNATURE_LENGTH = 100


def anthropic_to_openai_messages(
    anthropic_messages: List[dict], system: Optional[Union[str, List[dict]]] = None
) -> List[dict]:
    """
    Convert Anthropic message format to OpenAI format.

    Key differences:
    - Anthropic: system is a separate field, content can be string or list of blocks
    - OpenAI: system is a message with role="system", content is usually string

    Args:
        anthropic_messages: List of messages in Anthropic format
        system: Optional system message (string or list of text blocks)

    Returns:
        List of messages in OpenAI format
    """
    openai_messages = []

    # Handle system message
    if system:
        if isinstance(system, str):
            openai_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # System can be list of text blocks in Anthropic format
            system_text = " ".join(
                block.get("text", "")
                for block in system
                if isinstance(block, dict) and block.get("type") == "text"
            )
            if system_text:
                openai_messages.append({"role": "system", "content": system_text})

    for msg in anthropic_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Handle content blocks
            openai_content = []
            tool_calls = []
            reasoning_content = ""
            thinking_signature = ""

            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "text")

                    if block_type == "text":
                        openai_content.append(
                            {"type": "text", "text": block.get("text", "")}
                        )
                    elif block_type == "image":
                        # Convert Anthropic image format to OpenAI
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            openai_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                                    },
                                }
                            )
                        elif source.get("type") == "url":
                            openai_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": source.get("url", "")},
                                }
                            )
                    elif block_type == "thinking":
                        signature = block.get("signature", "")
                        if signature and len(signature) >= MIN_THINKING_SIGNATURE_LENGTH:
                            thinking_text = block.get("thinking", "")
                            if thinking_text:
                                reasoning_content += thinking_text
                            thinking_signature = signature
                    elif block_type == "redacted_thinking":
                        signature = block.get("signature", "")
                        if signature and len(signature) >= MIN_THINKING_SIGNATURE_LENGTH:
                            thinking_signature = signature
                    elif block_type == "tool_use":
                        # Anthropic tool_use -> OpenAI tool_calls
                        tool_calls.append(
                            {
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            }
                        )
                    elif block_type == "tool_result":
                        # Tool results become separate messages in OpenAI format
                        # Content can be string, or list of text/image blocks
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, str):
                            # Simple string content
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": tool_content,
                                }
                            )
                        elif isinstance(tool_content, list):
                            # List of content blocks - may include text and images
                            tool_content_parts = []
                            for b in tool_content:
                                if not isinstance(b, dict):
                                    continue
                                b_type = b.get("type", "")
                                if b_type == "text":
                                    tool_content_parts.append(
                                        {"type": "text", "text": b.get("text", "")}
                                    )
                                elif b_type == "image":
                                    # Convert Anthropic image format to OpenAI format
                                    source = b.get("source", {})
                                    if source.get("type") == "base64":
                                        tool_content_parts.append(
                                            {
                                                "type": "image_url",
                                                "image_url": {
                                                    "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                                                },
                                            }
                                        )
                                    elif source.get("type") == "url":
                                        tool_content_parts.append(
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": source.get("url", "")},
                                            }
                                        )

                            # If we only have text parts, join them as a string for compatibility
                            # Otherwise use the array format for multimodal content
                            if all(p.get("type") == "text" for p in tool_content_parts):
                                combined_text = " ".join(
                                    p.get("text", "") for p in tool_content_parts
                                )
                                openai_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": block.get("tool_use_id", ""),
                                        "content": combined_text,
                                    }
                                )
                            elif tool_content_parts:
                                # Multimodal content (includes images)
                                openai_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": block.get("tool_use_id", ""),
                                        "content": tool_content_parts,
                                    }
                                )
                            else:
                                # Empty content
                                openai_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": block.get("tool_use_id", ""),
                                        "content": "",
                                    }
                                )
                        else:
                            # Fallback for unexpected content type
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": str(tool_content) if tool_content else "",
                                }
                            )
                        continue  # Don't add to current message

            # Build the message
            if tool_calls:
                # Assistant message with tool calls
                msg_dict = {"role": role}
                if openai_content:
                    # If there's text content alongside tool calls
                    text_parts = [
                        c.get("text", "")
                        for c in openai_content
                        if c.get("type") == "text"
                    ]
                    msg_dict["content"] = " ".join(text_parts) if text_parts else None
                else:
                    msg_dict["content"] = None
                if reasoning_content:
                    msg_dict["reasoning_content"] = reasoning_content
                if thinking_signature:
                    msg_dict["thinking_signature"] = thinking_signature
                msg_dict["tool_calls"] = tool_calls
                openai_messages.append(msg_dict)
            elif openai_content:
                # Check if it's just text or mixed content
                if len(openai_content) == 1 and openai_content[0].get("type") == "text":
                    msg_dict = {
                        "role": role,
                        "content": openai_content[0].get("text", ""),
                    }
                    if reasoning_content:
                        msg_dict["reasoning_content"] = reasoning_content
                    if thinking_signature:
                        msg_dict["thinking_signature"] = thinking_signature
                    openai_messages.append(msg_dict)
                else:
                    msg_dict = {"role": role, "content": openai_content}
                    if reasoning_content:
                        msg_dict["reasoning_content"] = reasoning_content
                    if thinking_signature:
                        msg_dict["thinking_signature"] = thinking_signature
                    openai_messages.append(msg_dict)
            elif reasoning_content:
                msg_dict = {"role": role, "content": ""}
                msg_dict["reasoning_content"] = reasoning_content
                if thinking_signature:
                    msg_dict["thinking_signature"] = thinking_signature
                openai_messages.append(msg_dict)

    return openai_messages


def anthropic_to_openai_tools(
    anthropic_tools: Optional[List[dict]],
) -> Optional[List[dict]]:
    """
    Convert Anthropic tool definitions to OpenAI format.

    Args:
        anthropic_tools: List of tools in Anthropic format

    Returns:
        List of tools in OpenAI format, or None if no tools provided
    """
    if not anthropic_tools:
        return None

    openai_tools = []
    for tool in anthropic_tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return openai_tools


def anthropic_to_openai_tool_choice(
    anthropic_tool_choice: Optional[dict],
) -> Optional[Union[str, dict]]:
    """
    Convert Anthropic tool_choice to OpenAI format.

    Args:
        anthropic_tool_choice: Tool choice in Anthropic format

    Returns:
        Tool choice in OpenAI format
    """
    if not anthropic_tool_choice:
        return None

    choice_type = anthropic_tool_choice.get("type", "auto")

    if choice_type == "auto":
        return "auto"
    elif choice_type == "any":
        return "required"
    elif choice_type == "tool":
        return {
            "type": "function",
            "function": {"name": anthropic_tool_choice.get("name", "")},
        }
    elif choice_type == "none":
        return "none"

    return "auto"


def openai_to_anthropic_response(openai_response: dict, original_model: str) -> dict:
    """
    Convert OpenAI chat completion response to Anthropic Messages format.

    Args:
        openai_response: Response from OpenAI-compatible API
        original_model: The model name requested by the client

    Returns:
        Response in Anthropic Messages format
    """
    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = openai_response.get("usage", {})

    # Build content blocks
    content_blocks = []

    # Add thinking content block if reasoning_content is present
    reasoning_content = message.get("reasoning_content")
    if reasoning_content:
        thinking_signature = message.get("thinking_signature", "")
        signature = (
            thinking_signature
            if thinking_signature
            and len(thinking_signature) >= MIN_THINKING_SIGNATURE_LENGTH
            else ""
        )
        content_blocks.append(
            {
                "type": "thinking",
                "thinking": reasoning_content,
                "signature": signature,
            }
        )

    # Add text content if present
    text_content = message.get("content")
    if text_content:
        content_blocks.append({"type": "text", "text": text_content})

    # Add tool use blocks if present
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            input_data = {}

        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": func.get("name", ""),
                "input": input_data,
            }
        )

    # Map finish_reason to stop_reason
    finish_reason = choice.get("finish_reason", "end_turn")
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
        "function_call": "tool_use",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

    # Build usage
    anthropic_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }

    # Add cache tokens if present
    if usage.get("prompt_tokens_details"):
        details = usage["prompt_tokens_details"]
        if details.get("cached_tokens"):
            anthropic_usage["cache_read_input_tokens"] = details["cached_tokens"]

    return {
        "id": openai_response.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": original_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": anthropic_usage,
    }


def translate_anthropic_request(request: AnthropicMessagesRequest) -> Dict[str, Any]:
    """
    Translate a complete Anthropic Messages API request to OpenAI format.

    This is a high-level function that handles all aspects of request translation,
    including messages, tools, tool_choice, and thinking configuration.

    Args:
        request: An AnthropicMessagesRequest object

    Returns:
        Dictionary containing the OpenAI-compatible request parameters
    """
    anthropic_request = request.model_dump(exclude_none=True)

    openai_messages = anthropic_to_openai_messages(
        anthropic_request.get("messages", []), anthropic_request.get("system")
    )

    openai_tools = anthropic_to_openai_tools(anthropic_request.get("tools"))
    openai_tool_choice = anthropic_to_openai_tool_choice(
        anthropic_request.get("tool_choice")
    )

    # Build OpenAI-compatible request
    openai_request = {
        "model": request.model,
        "messages": openai_messages,
        "max_tokens": request.max_tokens,
        "stream": request.stream or False,
    }

    if request.temperature is not None:
        openai_request["temperature"] = request.temperature
    if request.top_p is not None:
        openai_request["top_p"] = request.top_p
    if request.top_k is not None:
        openai_request["top_k"] = request.top_k
    if request.stop_sequences:
        openai_request["stop"] = request.stop_sequences
    if openai_tools:
        openai_request["tools"] = openai_tools
    if openai_tool_choice:
        openai_request["tool_choice"] = openai_tool_choice

    # Note: request.metadata is intentionally not mapped.
    # OpenAI's API doesn't have an equivalent field for client-side metadata.
    # The metadata is typically used by Anthropic clients for tracking purposes
    # and doesn't affect the model's behavior.

    # Handle Anthropic thinking config -> reasoning_effort translation
    # The provider (antigravity_provider.py) applies a // 4 reduction to thinking budget
    # unless custom_reasoning_budget is True. This conserves thinking tokens.
    #
    # Reasoning budget thresholds map to provider budgets:
    # - Claude "high" = 32768 tokens (but // 4 = 8192 unless custom_reasoning_budget)
    # - Claude "medium" = 16384 tokens (// 4 = 4096)
    # - Claude "low" = 8192 tokens (// 4 = 2048)
    #
    # We only set custom_reasoning_budget=True when user explicitly requests
    # a large budget (32000+), indicating they want full thinking capacity.
    if request.thinking:
        if request.thinking.type == "enabled":
            budget = request.thinking.budget_tokens or 10000
            if budget >= 32000:
                # User explicitly wants full thinking capacity
                openai_request["reasoning_effort"] = "high"
                openai_request["custom_reasoning_budget"] = True
            elif budget >= 10000:
                openai_request["reasoning_effort"] = "high"
                # custom_reasoning_budget defaults to False, so // 4 applies
            elif budget >= 5000:
                openai_request["reasoning_effort"] = "medium"
            else:
                openai_request["reasoning_effort"] = "low"
        elif request.thinking.type == "disabled":
            openai_request["reasoning_effort"] = "disable"
    elif _is_opus_model(request.model):
        # Enable thinking for Opus models when no thinking config is provided
        # Use "high" effort but NOT custom_reasoning_budget, so // 4 applies
        # This gives 8192 thinking tokens (32768 // 4) which is reasonable for most tasks
        # Users who want full capacity can explicitly set thinking.budget_tokens >= 32000
        openai_request["reasoning_effort"] = "high"
        # Note: NOT setting custom_reasoning_budget here to conserve tokens

    return openai_request


def _is_opus_model(model_name: str) -> bool:
    """
    Check if a model name refers to a Claude Opus model.

    Uses specific pattern matching to avoid false positives with model names
    that might contain "opus" as part of another word.

    Args:
        model_name: The model name to check

    Returns:
        True if the model is a Claude Opus model, False otherwise
    """
    import re

    model_lower = model_name.lower()
    # Match Claude Opus models specifically:
    # - "claude-opus-4-5", "claude-4-opus", "claude_opus"
    # - "opus-4", "opus-4.5", "opus4" (standalone with version)
    # - "antigravity/claude-opus-4-5"
    # Avoid matching things like "magnum-opus" or other non-Claude models
    opus_patterns = [
        r'claude[-_]?opus',        # "claude-opus", "claude_opus", "claudeopus"
        r'opus[-_]?\d',            # "opus-4", "opus_4", "opus4" (with version number)
        r'\d[-_]?opus(?:[-_]|$)',  # "4-opus", "4_opus" at word boundary
    ]
    return any(re.search(pattern, model_lower) for pattern in opus_patterns)
