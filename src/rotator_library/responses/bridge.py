# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Bridge between Responses requests and the current chat-completions executor."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

from ..protocols import ContentBlock, ToolDefinition, UnifiedMessage, UnifiedRequest, serialize_value
from ..protocols.responses import ResponsesProtocol
from .types import generate_response_id

_CHAT_GENERATION_KEYS = {
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "max_output_tokens",
    "metadata",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "reasoning",
    "reasoning_effort",
    "response_format",
    "seed",
    "stop",
    "stream_options",
    "temperature",
    "tool_choice",
    "top_logprobs",
    "top_p",
    "user",
}


class ResponsesBridge:
    """Temporary compatibility bridge from Responses to chat completions.

    Later provider phases should call native Responses-capable providers directly.
    Until then, this bridge makes `/v1/responses` useful while preserving enough
    metadata and trace detail to debug fields that cannot be represented in chat.
    """

    def __init__(self, protocol: Optional[ResponsesProtocol] = None) -> None:
        self.protocol = protocol or ResponsesProtocol()

    def to_chat_kwargs(
        self,
        unified: UnifiedRequest,
        *,
        parent_response: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Convert a parsed Responses request to chat-completions kwargs."""

        messages: list[dict[str, Any]] = []
        system_text = _blocks_to_text(unified.system)
        if system_text:
            messages.append({"role": "system", "content": system_text})
        if parent_response:
            messages.extend(_parent_output_to_messages(parent_response.get("output") or []))
        messages.extend(_message_to_chat(message) for message in unified.messages)
        kwargs: dict[str, Any] = {
            "model": unified.model,
            "messages": messages,
            "stream": unified.stream,
        }
        if unified.tools:
            kwargs["tools"] = [_tool_to_chat(tool) for tool in unified.tools]
        for key, value in unified.generation_params.items():
            if key in _CHAT_GENERATION_KEYS:
                kwargs[_chat_generation_key(key)] = deepcopy(value)
        if unified.metadata:
            kwargs.setdefault("metadata", deepcopy(unified.metadata))
        unsupported = {
            "previous_response_id": unified.previous_response_id,
            "extra": deepcopy(unified.extra),
        }
        kwargs["_responses_bridge"] = {k: v for k, v in unsupported.items() if v}
        return kwargs

    def from_chat_response(
        self,
        chat_response: Any,
        unified_request: UnifiedRequest,
        *,
        response_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Convert a chat-completions response into a Responses object."""

        response = _as_dict(chat_response)
        output = []
        for index, choice in enumerate(response.get("choices") or []):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            output.extend(_chat_message_to_output_items(message, index))
        responses_payload = {
            "id": response_id or response.get("id") or generate_response_id(),
            "object": "response",
            "created_at": response.get("created") or response.get("created_at"),
            "model": response.get("model") or unified_request.model,
            "status": _status_from_chat(response),
            "output": output,
            "usage": _usage_to_responses(response.get("usage")),
        }
        if unified_request.metadata:
            responses_payload["metadata"] = deepcopy(unified_request.metadata)
        return self.protocol.format_response(self.protocol.parse_response(responses_payload))


def _chat_generation_key(key: str) -> str:
    if key == "max_output_tokens":
        return "max_tokens"
    return key


def _message_to_chat(message: UnifiedMessage) -> dict[str, Any]:
    payload = {"role": message.role, "content": _blocks_to_chat_content(message.content)}
    if message.name:
        payload["name"] = message.name
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        payload["tool_calls"] = [call.to_dict() for call in message.tool_calls]
    return payload


def _blocks_to_chat_content(blocks: list[ContentBlock]) -> Any:
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0].text is not None:
        return blocks[0].text
    content = []
    for block in blocks:
        if block.text is not None:
            content.append({"type": "text", "text": block.text})
        elif block.source is not None:
            content.append({"type": block.type, "source": deepcopy(block.source)})
        elif block.raw is not None:
            content.append(deepcopy(block.raw))
    return content or ""


def _blocks_to_text(blocks: list[ContentBlock]) -> str:
    return "".join(block.text or "" for block in blocks if block.text)


def _tool_to_chat(tool: ToolDefinition) -> dict[str, Any]:
    if tool.type == "function":
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": deepcopy(tool.input_schema),
            },
        }
    payload = {"type": tool.type, "name": tool.name}
    payload.update(deepcopy(tool.extra))
    return payload


def _parent_output_to_messages(output: list[Any]) -> list[dict[str, Any]]:
    messages = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        text = _responses_content_to_text(item.get("content") or [])
        if text:
            messages.append({"role": item.get("role") or "assistant", "content": text})
    return messages


def _chat_message_to_output_items(message: dict[str, Any], index: int) -> list[dict[str, Any]]:
    items = []
    content = message.get("content")
    if content is not None:
        items.append(
            {
                "id": f"msg_{index}",
                "type": "message",
                "role": message.get("role") or "assistant",
                "content": [{"type": "output_text", "text": content if isinstance(content, str) else serialize_value(content)}],
            }
        )
    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        if isinstance(function, dict):
            items.append(
                {
                    "id": tool_call.get("id"),
                    "type": "function_call",
                    "call_id": tool_call.get("id"),
                    "name": function.get("name"),
                    "arguments": function.get("arguments"),
                }
            )
    return items


def _responses_content_to_text(content: list[Any]) -> str:
    parts = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text")
            if text:
                parts.append(str(text))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


def _status_from_chat(response: dict[str, Any]) -> str:
    for choice in response.get("choices") or []:
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        if finish_reason in {"length", "content_filter"}:
            return "incomplete"
    return "completed"


def _usage_to_responses(usage: Any) -> Any:
    if not isinstance(usage, dict):
        return usage
    return {
        "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
        "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
        "total_tokens": usage.get("total_tokens", 0),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return serialize_value(value) if isinstance(serialize_value(value), dict) else {}
