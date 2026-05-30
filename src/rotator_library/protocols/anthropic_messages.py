# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Anthropic Messages protocol adapter.

This adapter captures the native Messages shape as a reusable base. The existing
compatibility routes remain active; this module gives future provider-native
execution a loss-conscious parser/builder with thinking and tool block support.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, ClassVar, Iterable

from .base import ProtocolAdapter
from .types import (
    ContentBlock,
    ProtocolContext,
    ReasoningBlock,
    ToolCall,
    ToolDefinition,
    ToolResult,
    UnifiedMessage,
    UnifiedRequest,
    UnifiedResponse,
    UnifiedStreamEvent,
    Usage,
    first_text,
    text_blocks,
)

_GENERATION_PARAMS = {
    "max_tokens",
    "metadata",
    "stop_sequences",
    "temperature",
    "thinking",
    "tool_choice",
    "top_k",
    "top_p",
}

_REQUEST_CORE_FIELDS = {"model", "messages", "system", "tools", "stream", *_GENERATION_PARAMS}


class AnthropicMessagesProtocol(ProtocolAdapter):
    """Adapter for Anthropic Messages requests, responses, and stream events.

    Thinking and redacted-thinking blocks are represented as reasoning blocks so
    later field-cache rules can extract signatures without relying on a bespoke
    provider implementation.
    """

    name: ClassVar[str] = "anthropic_messages"
    aliases: ClassVar[tuple[str, ...]] = ("anthropic", "messages", "claude_messages")
    supported_transports: ClassVar[tuple[str, ...]] = ("http", "sse")

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        return UnifiedRequest(
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            messages=[self._parse_message(message) for message in request.get("messages") or []],
            system=self._parse_system(request.get("system")),
            tools=[self._parse_tool_definition(tool) for tool in request.get("tools") or []],
            stream=bool(request.get("stream", False)),
            generation_params={k: deepcopy(request[k]) for k in _GENERATION_PARAMS if k in request and k != "metadata"},
            metadata=deepcopy(request.get("metadata") or {}),
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": unified_request.model,
            "messages": [self._format_message(message) for message in unified_request.messages],
        }
        system = self._format_system(unified_request.system)
        if system is not None:
            payload["system"] = system
        if unified_request.tools:
            payload["tools"] = [self._format_tool_definition(tool) for tool in unified_request.tools]
        if unified_request.stream:
            payload["stream"] = True
        if unified_request.metadata:
            payload["metadata"] = deepcopy(unified_request.metadata)
        payload.update(deepcopy(unified_request.generation_params))
        payload.update(deepcopy(unified_request.extra))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = _as_dict(raw_response)
        message = UnifiedMessage(
            role=str(response.get("role") or "assistant"),
            content=self._parse_content(response.get("content")),
            raw=deepcopy(response),
            extra={"type": response.get("type")},
        )
        self._promote_message_blocks(message)
        return UnifiedResponse(
            id=response.get("id"),
            model=response.get("model") or getattr(context, "model", None),
            messages=[message] if response else [],
            stop_reason=response.get("stop_reason"),
            usage=self.extract_usage(response, context),
            metadata={"stop_sequence": response.get("stop_sequence"), "type": response.get("type")},
            raw=deepcopy(response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"id", "type", "role", "content", "model", "stop_reason", "stop_sequence", "usage"}},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:
        if isinstance(unified_response.raw, dict):
            return deepcopy(unified_response.raw)
        message = unified_response.messages[0] if unified_response.messages else UnifiedMessage(role="assistant")
        payload = {
            "id": unified_response.id,
            "type": unified_response.metadata.get("type", "message"),
            "role": message.role,
            "content": self._format_content(message.content),
            "model": unified_response.model,
            "stop_reason": unified_response.stop_reason,
            "stop_sequence": unified_response.metadata.get("stop_sequence"),
            "usage": self._format_usage(unified_response.usage),
        }
        payload.update(deepcopy(unified_response.extra))
        return payload

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        event = _decode_sse_data(raw_event)
        if event == "[DONE]":
            return UnifiedStreamEvent(type="done", raw=deepcopy(raw_event))
        data = _as_dict(event)
        event_type = str(data.get("type") or "chunk")

        if event_type == "error" or data.get("error") is not None:
            return UnifiedStreamEvent(type="error", error=deepcopy(data.get("error", data)), raw=deepcopy(raw_event), extra={"payload": data})
        if event_type == "message_start":
            response = self.parse_response(data.get("message") or {}, context)
            return UnifiedStreamEvent(type="message_start", message=response.messages[0] if response.messages else None, usage=response.usage, raw=deepcopy(raw_event), extra={"payload": data})
        if event_type == "message_delta":
            return UnifiedStreamEvent(type="message_delta", usage=self.extract_usage(data.get("usage") or {}, context), raw=deepcopy(raw_event), extra={"payload": data, "stop_reason": (data.get("delta") or {}).get("stop_reason")})
        if event_type in {"content_block_start", "content_block_delta", "content_block_stop"}:
            return self._parse_content_stream_event(data, raw_event)
        return UnifiedStreamEvent(type=event_type, raw=deepcopy(raw_event), extra={"payload": data})

    def extract_usage(self, raw_or_unified: Any, context: ProtocolContext | None = None) -> Usage | None:
        if isinstance(raw_or_unified, (UnifiedResponse, UnifiedStreamEvent)):
            return raw_or_unified.usage
        payload = _as_dict(raw_or_unified)
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else payload
        if not isinstance(usage, dict) or not any(k.endswith("tokens") for k in usage):
            return None
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cache_write = int(usage.get("cache_creation_input_tokens") or 0)
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=int(usage.get("total_tokens") or input_tokens + output_tokens),
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            raw=deepcopy(usage),
        )

    def _parse_system(self, system: Any) -> list[ContentBlock]:
        if system is None:
            return []
        if isinstance(system, str):
            return text_blocks(system)
        return self._parse_content(system)

    def _format_system(self, blocks: Iterable[ContentBlock]) -> Any:
        block_list = list(blocks)
        if not block_list:
            return None
        if all(block.type == "text" for block in block_list):
            return first_text(block_list) or ""
        return self._format_content(block_list)

    def _parse_message(self, message: dict[str, Any]) -> UnifiedMessage:
        payload = dict(message or {})
        unified = UnifiedMessage(
            role=str(payload.get("role") or "user"),
            content=self._parse_content(payload.get("content")),
            raw=deepcopy(message),
            extra={k: deepcopy(v) for k, v in payload.items() if k not in {"role", "content"}},
        )
        self._promote_message_blocks(unified)
        return unified

    def _format_message(self, message: UnifiedMessage) -> dict[str, Any]:
        payload = {"role": message.role, "content": self._format_content(message.content)}
        payload.update(deepcopy(message.extra))
        return payload

    def _parse_content(self, content: Any) -> list[ContentBlock]:
        if content is None:
            return []
        if isinstance(content, str):
            return text_blocks(content)
        if not isinstance(content, list):
            return [ContentBlock(type="unknown", raw=deepcopy(content))]
        return [self._parse_content_block(block) for block in content]

    def _parse_content_block(self, block: Any) -> ContentBlock:
        if isinstance(block, str):
            return ContentBlock(type="text", text=block, raw=block)
        if not isinstance(block, dict):
            return ContentBlock(type="unknown", raw=deepcopy(block))
        block_type = str(block.get("type") or "text")
        if block_type == "text":
            return ContentBlock(type="text", text=block.get("text", ""), raw=deepcopy(block))
        if block_type in {"image", "document"}:
            return ContentBlock(type=block_type, source=deepcopy(block.get("source")), raw=deepcopy(block), extra=_without(block, {"type", "source"}))
        if block_type in {"thinking", "redacted_thinking"}:
            reasoning = ReasoningBlock(
                type=block_type,
                text=block.get("thinking"),
                signature=block.get("signature"),
                redacted=block_type == "redacted_thinking",
                extra=_without(block, {"type", "thinking", "signature"}),
            )
            return ContentBlock(type=block_type, reasoning=reasoning, raw=deepcopy(block))
        if block_type == "tool_use":
            return ContentBlock(
                type="tool_use",
                tool_call=ToolCall(id=block.get("id"), name=block.get("name"), arguments=deepcopy(block.get("input")), type="tool_use"),
                raw=deepcopy(block),
                extra=_without(block, {"type", "id", "name", "input"}),
            )
        if block_type == "tool_result":
            return ContentBlock(
                type="tool_result",
                tool_result=ToolResult(tool_call_id=block.get("tool_use_id"), content=deepcopy(block.get("content")), is_error=block.get("is_error")),
                raw=deepcopy(block),
                extra=_without(block, {"type", "tool_use_id", "content", "is_error"}),
            )
        return ContentBlock(type=block_type, raw=deepcopy(block), extra=_without(block, {"type"}))

    def _format_content(self, blocks: Iterable[ContentBlock]) -> list[dict[str, Any]]:
        formatted = []
        for block in blocks:
            if isinstance(block.raw, dict):
                formatted.append(deepcopy(block.raw))
            elif block.type == "text":
                formatted.append({"type": "text", "text": block.text or ""})
            elif block.reasoning:
                payload = {"type": block.reasoning.type}
                if block.reasoning.text is not None:
                    payload["thinking"] = block.reasoning.text
                if block.reasoning.signature is not None:
                    payload["signature"] = block.reasoning.signature
                payload.update(deepcopy(block.reasoning.extra))
                formatted.append(payload)
            elif block.tool_call:
                formatted.append({"type": "tool_use", "id": block.tool_call.id, "name": block.tool_call.name, "input": deepcopy(block.tool_call.arguments)})
            elif block.tool_result:
                payload = {"type": "tool_result", "tool_use_id": block.tool_result.tool_call_id, "content": deepcopy(block.tool_result.content)}
                if block.tool_result.is_error is not None:
                    payload["is_error"] = block.tool_result.is_error
                formatted.append(payload)
            else:
                payload = {"type": block.type}
                payload.update(deepcopy(block.extra))
                formatted.append(payload)
        return formatted

    def _parse_tool_definition(self, tool: dict[str, Any]) -> ToolDefinition:
        payload = dict(tool or {})
        return ToolDefinition(
            name=str(payload.get("name") or ""),
            description=payload.get("description"),
            input_schema=deepcopy(payload.get("input_schema") or {}),
            type="tool",
            extra=_without(payload, {"name", "description", "input_schema"}),
        )

    def _format_tool_definition(self, tool: ToolDefinition) -> dict[str, Any]:
        payload = {"name": tool.name, "input_schema": deepcopy(tool.input_schema)}
        if tool.description is not None:
            payload["description"] = tool.description
        payload.update(deepcopy(tool.extra))
        return payload

    def _format_usage(self, usage: Usage | None) -> dict[str, int] | None:
        if usage is None:
            return None
        payload = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}
        if usage.cache_write_tokens:
            payload["cache_creation_input_tokens"] = usage.cache_write_tokens
        if usage.cache_read_tokens:
            payload["cache_read_input_tokens"] = usage.cache_read_tokens
        return payload

    def _promote_message_blocks(self, message: UnifiedMessage) -> None:
        for block in message.content:
            if block.tool_call:
                message.tool_calls.append(block.tool_call)
            if block.reasoning:
                message.reasoning.append(block.reasoning)

    def _parse_content_stream_event(self, data: dict[str, Any], raw_event: Any) -> UnifiedStreamEvent:
        block = data.get("content_block") if isinstance(data.get("content_block"), dict) else None
        delta = data.get("delta") if isinstance(data.get("delta"), dict) else None
        content_block = None
        if block:
            content_block = self._parse_content_block(block)
        elif delta:
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                content_block = ContentBlock(type="text", text=delta.get("text"), raw=deepcopy(delta))
            elif delta_type in {"thinking_delta", "signature_delta"}:
                reasoning = ReasoningBlock(type=str(delta_type), text=delta.get("thinking"), signature=delta.get("signature"), extra=_without(delta, {"type", "thinking", "signature"}))
                content_block = ContentBlock(type=str(delta_type), reasoning=reasoning, raw=deepcopy(delta))
        message = UnifiedMessage(role="assistant", content=[content_block] if content_block else [])
        self._promote_message_blocks(message)
        return UnifiedStreamEvent(type=str(data.get("type") or "content_block_delta"), delta=message, raw=deepcopy(raw_event), extra={"payload": data, "index": data.get("index")})


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _decode_sse_data(raw_event: Any) -> Any:
    if not isinstance(raw_event, str):
        return raw_event
    text = raw_event.strip()
    if text.startswith("data:"):
        text = text[5:].strip()
    if text == "[DONE]":
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw_event


def _without(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: deepcopy(v) for k, v in payload.items() if k not in keys}
