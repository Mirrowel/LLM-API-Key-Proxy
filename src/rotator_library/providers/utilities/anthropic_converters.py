# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Shared converters between OpenAI and Anthropic message formats for providers.

Used by any provider that needs to convert OpenAI-format messages/tools
to Anthropic Messages API format (e.g. AnthropicProvider, NanoGptProvider).
"""

import json
import uuid
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

lib_logger = logging.getLogger("rotator_library")

TOOL_PREFIX = "mcp_"


def convert_openai_to_anthropic_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Convert OpenAI chat format messages to Anthropic Messages format.

    Returns:
        Tuple of (system_prompt, anthropic_messages)
    """
    system_prompt = None
    anthropic_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "system":
            # Extract system message
            if isinstance(content, str):
                system_prompt = content
            elif isinstance(content, list):
                # Handle multipart system content
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                system_prompt = "\n".join(texts)
            continue

        if role == "user":
            if isinstance(content, str):
                anthropic_messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # Convert multipart content
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            parts.append({"type": "text", "text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url", {})
                            url = image_url.get("url", "") if isinstance(image_url, dict) else image_url
                            if url.startswith("data:"):
                                try:
                                    header, data = url.split(",", 1)
                                    media_type = header.split(":")[1].split(";")[0]
                                    parts.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": data,
                                        },
                                    })
                                except (ValueError, IndexError):
                                    lib_logger.debug(
                                        "Failed to parse data URI image in user message, skipping."
                                    )
                if parts:
                    anthropic_messages.append({"role": "user", "content": parts})
            continue

        if role == "assistant":
            content_blocks = []

            # Handle text content
            if isinstance(content, str) and content:
                content_blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            content_blocks.append({"type": "text", "text": part.get("text", "")})

            # Handle tool calls
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                if isinstance(tc, dict) and tc.get("type") == "function":
                    func = tc.get("function", {})
                    arguments = func.get("arguments", "{}")
                    if isinstance(arguments, dict):
                        input_data = arguments
                    else:
                        try:
                            input_data = json.loads(arguments)
                        except (json.JSONDecodeError, TypeError):
                            input_data = {}

                    tool_name = func.get("name", "")
                    # Add mcp_ prefix if not already present
                    if not tool_name.startswith(TOOL_PREFIX):
                        tool_name = f"{TOOL_PREFIX}{tool_name}"

                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", str(uuid.uuid4())),
                        "name": tool_name,
                        "input": input_data,
                    })

            if content_blocks:
                anthropic_messages.append({"role": "assistant", "content": content_blocks})
            continue

        if role == "tool":
            # Tool result message
            tool_call_id = msg.get("tool_call_id", "")
            tool_content = content
            if isinstance(tool_content, str):
                try:
                    tool_content = json.loads(tool_content)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Anthropic expects tool results as user messages with tool_result blocks
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": str(tool_content) if not isinstance(tool_content, str) else tool_content,
                }],
            })
            continue

    return system_prompt, anthropic_messages


def convert_tools_to_anthropic_format(
    tools: Optional[List[Dict[str, Any]]]
) -> Optional[List[Dict[str, Any]]]:
    """Convert OpenAI tools format to Anthropic tool definitions."""
    if not tools:
        return None

    anthropic_tools = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name", "")
        if not name:
            continue

        # Add mcp_ prefix if not already present
        if not name.startswith(TOOL_PREFIX):
            name = f"{TOOL_PREFIX}{name}"

        anthropic_tools.append({
            "name": name,
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })

    return anthropic_tools if anthropic_tools else None
