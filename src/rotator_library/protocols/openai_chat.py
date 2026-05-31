# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenAI Chat Completions protocol adapter.

The adapter models the common OpenAI-compatible chat shape used by many current
providers. It is a reusable base, not a final authority: providers can subclass
or override pieces when they need non-standard fields, stricter ordering, or
different stream semantics.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, ClassVar, Iterable

from .base import ProtocolAdapter
from .operation import OPERATION_CHAT
from .types import (
    ContentBlock,
    CostDetails,
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
    serialize_value,
    text_blocks,
)

_GENERATION_PARAMS = {
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "reasoning_effort",
    "seed",
    "service_tier",
    "stop",
    "stream_options",
    "temperature",
    "tool_choice",
    "top_logprobs",
    "top_p",
    "user",
}

_REQUEST_CORE_FIELDS = {
    "model",
    "messages",
    "tools",
    "stream",
    "response_format",
    "metadata",
    *_GENERATION_PARAMS,
}


class OpenAIChatProtocol(ProtocolAdapter):
    """Adapter for OpenAI Chat Completions request, response, and stream chunks.

    Unknown OpenAI-compatible extension fields are preserved in ``extra`` so a
    custom provider can still use them through later adapter or field-cache
    phases. Lossy conversions are avoided unless the source shape itself uses a
    compact representation, such as string message content.
    """

    name: ClassVar[str] = "openai_chat"
    aliases: ClassVar[tuple[str, ...]] = (
        "openai",
        "chat_completions",
        "openai_chat_completions",
    )
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_CHAT,)
    supported_transports: ClassVar[tuple[str, ...]] = ("http", "sse")

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        messages = [self._parse_message(message) for message in request.get("messages") or []]
        tools = [self._parse_tool_definition(tool) for tool in request.get("tools") or []]
        generation_params = {k: deepcopy(request[k]) for k in _GENERATION_PARAMS if k in request}
        extra = {k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS}

        return UnifiedRequest(
            operation=OPERATION_CHAT,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            messages=messages,
            tools=tools,
            stream=bool(request.get("stream", False)),
            generation_params=generation_params,
            response_format=deepcopy(request.get("response_format")),
            metadata=deepcopy(request.get("metadata") or {}),
            raw=deepcopy(raw_request),
            extra=extra,
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": unified_request.model,
            "messages": [self._format_message(message) for message in unified_request.messages],
        }
        if unified_request.tools:
            payload["tools"] = [self._format_tool_definition(tool) for tool in unified_request.tools]
        if unified_request.stream:
            payload["stream"] = True
        if unified_request.response_format is not None:
            payload["response_format"] = deepcopy(unified_request.response_format)
        if unified_request.metadata:
            payload["metadata"] = deepcopy(unified_request.metadata)
        payload.update(deepcopy(unified_request.generation_params))
        payload.update(deepcopy(unified_request.extra))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = _as_dict(raw_response)
        messages: list[UnifiedMessage] = []
        stop_reason = None
        for choice in response.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message_payload = choice.get("message") or {}
            if message_payload:
                messages.append(self._parse_message(message_payload))
            if choice.get("finish_reason") is not None:
                stop_reason = choice.get("finish_reason")

        return UnifiedResponse(
            operation=OPERATION_CHAT,
            id=response.get("id"),
            model=response.get("model") or getattr(context, "model", None),
            messages=messages,
            stop_reason=stop_reason,
            usage=self.extract_usage(response, context),
            metadata={
                "object": response.get("object"),
                "created": response.get("created"),
                "system_fingerprint": response.get("system_fingerprint"),
            },
            raw=deepcopy(response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"id", "object", "created", "model", "choices", "usage", "system_fingerprint"}},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:
        choices = []
        for index, message in enumerate(unified_response.messages):
            choices.append(
                {
                    "index": index,
                    "message": self._format_message(message),
                    "finish_reason": unified_response.stop_reason,
                }
            )
        payload = {
            "id": unified_response.id,
            "object": unified_response.metadata.get("object", "chat.completion"),
            "created": unified_response.metadata.get("created"),
            "model": unified_response.model,
            "choices": choices,
            "usage": _format_openai_usage(unified_response.usage),
        }
        payload.update(deepcopy(unified_response.extra))
        return {k: v for k, v in payload.items() if v is not None}

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        event = _decode_sse_data(raw_event)
        if event == "[DONE]":
            return UnifiedStreamEvent(type="done", operation=OPERATION_CHAT, raw=deepcopy(raw_event))
        data = _as_dict(event)
        if data.get("error") is not None:
            return UnifiedStreamEvent(type="error", operation=OPERATION_CHAT, error=deepcopy(data["error"]), raw=deepcopy(raw_event), extra={"payload": data})

        delta_message = None
        finish_reason = None
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            if delta:
                delta_message = self._parse_message({"role": delta.get("role", "assistant"), **delta})
            finish_reason = choice.get("finish_reason") if choice.get("finish_reason") is not None else finish_reason
            break

        usage = self.extract_usage(data, context)
        return UnifiedStreamEvent(
            type="message_delta" if delta_message else "chunk",
            operation=OPERATION_CHAT,
            delta=delta_message,
            usage=usage,
            raw=deepcopy(raw_event),
            extra={
                "id": data.get("id"),
                "model": data.get("model"),
                "finish_reason": finish_reason,
                "payload": data,
            },
        )

    def format_stream_event(self, unified_event: UnifiedStreamEvent, context: ProtocolContext | None = None) -> Any:
        if unified_event.type == "done":
            return "data: [DONE]\n\n"
        if unified_event.raw is not None:
            return deepcopy(unified_event.raw)
        return f"data: {json.dumps(unified_event.to_dict())}\n\n"

    def extract_usage(self, raw_or_unified: Any, context: ProtocolContext | None = None) -> Usage | None:
        if isinstance(raw_or_unified, (UnifiedResponse, UnifiedStreamEvent)):
            return raw_or_unified.usage
        payload = _as_dict(raw_or_unified)
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        if not isinstance(prompt_details, dict):
            prompt_details = {}
        if not isinstance(completion_details, dict):
            completion_details = {}
        cost = None
        cost_details = usage.get("cost_details")
        if isinstance(cost_details, dict):
            provider_cost = cost_details.get("total_cost") or cost_details.get("cost")
            cost = CostDetails(
                provider_reported_cost=float(provider_cost) if provider_cost is not None else None,
                currency=str(cost_details.get("currency") or "USD"),
                source="usage.cost_details",
                metadata={k: deepcopy(v) for k, v in cost_details.items() if k not in {"total_cost", "cost", "currency"}},
            )
        return Usage(
            input_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            cache_read_tokens=int(prompt_details.get("cached_tokens") or usage.get("cache_read_tokens") or 0),
            cache_write_tokens=int(prompt_details.get("cache_creation_tokens") or usage.get("cache_creation_tokens") or 0),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0),
            cost=cost,
            raw=deepcopy(usage),
        )

    def _parse_message(self, message: dict[str, Any]) -> UnifiedMessage:
        payload = dict(message or {})
        reasoning = _extract_reasoning(payload)
        return UnifiedMessage(
            role=str(payload.get("role") or "assistant"),
            content=self._parse_content(payload.get("content")),
            name=payload.get("name"),
            tool_call_id=payload.get("tool_call_id"),
            tool_calls=self._parse_message_tool_calls(payload),
            reasoning=reasoning,
            raw=deepcopy(message),
            extra={k: deepcopy(v) for k, v in payload.items() if k not in {"role", "content", "name", "tool_call_id", "tool_calls", "reasoning", "reasoning_content"}},
        )

    def _format_message(self, message: UnifiedMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role}
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        content = self._format_content(message.content)
        if content is not None:
            payload["content"] = content
        legacy_function_call = message.extra.get("function_call")
        if message.tool_calls and not legacy_function_call:
            payload["tool_calls"] = [self._format_tool_call(call) for call in message.tool_calls]
        if message.reasoning:
            # OpenAI-compatible providers use multiple names for reasoning text.
            # Prefer the common extension field while keeping all blocks in extra.
            text = "".join(block.text or "" for block in message.reasoning if block.text)
            if text:
                payload["reasoning_content"] = text
        payload.update(deepcopy(message.extra))
        return payload

    def _parse_message_tool_calls(self, payload: dict[str, Any]) -> list[ToolCall]:
        """Return modern and legacy OpenAI function calls as unified tools."""

        modern_calls = payload.get("tool_calls") or []
        if modern_calls:
            return [self._parse_tool_call(call) for call in modern_calls]
        legacy_call = payload.get("function_call")
        if isinstance(legacy_call, dict):
            return [
                ToolCall(
                    id=None,
                    name=legacy_call.get("name"),
                    arguments=legacy_call.get("arguments"),
                    type="function",
                    raw=deepcopy(legacy_call),
                    extra={"legacy_function_call": True},
                )
            ]
        return []

    def _parse_content(self, content: Any) -> list[ContentBlock]:
        if content is None:
            return []
        if isinstance(content, str):
            return text_blocks(content)
        if not isinstance(content, list):
            return [ContentBlock(type="unknown", raw=deepcopy(content))]
        blocks = []
        for block in content:
            if isinstance(block, str):
                blocks.append(ContentBlock(type="text", text=block, raw=block))
                continue
            if not isinstance(block, dict):
                blocks.append(ContentBlock(type="unknown", raw=deepcopy(block)))
                continue
            block_type = block.get("type", "text")
            if block_type == "text":
                blocks.append(ContentBlock(type="text", text=block.get("text", ""), raw=deepcopy(block), extra=_without(block, {"type", "text"})))
            elif block_type in {"image_url", "input_image"}:
                blocks.append(ContentBlock(type=block_type, source=deepcopy(block.get("image_url") or block.get("source")), raw=deepcopy(block), extra=_without(block, {"type", "image_url", "source"})))
            else:
                blocks.append(ContentBlock(type=str(block_type), raw=deepcopy(block), extra=_without(block, {"type"})))
        return blocks

    def _format_content(self, blocks: Iterable[ContentBlock]) -> Any:
        block_list = list(blocks)
        if not block_list:
            return None
        if all(block.type == "text" and not isinstance(block.raw, dict) and not block.extra for block in block_list):
            return first_text(block_list) or ""
        formatted = []
        for block in block_list:
            if block.type == "text":
                payload = deepcopy(block.raw) if isinstance(block.raw, dict) else {"type": "text"}
                payload["type"] = payload.get("type", "text")
                payload["text"] = block.text or ""
                payload.update(deepcopy(block.extra))
                formatted.append(payload)
            elif block.type in {"image_url", "input_image"}:
                payload = deepcopy(block.raw) if isinstance(block.raw, dict) else {"type": block.type}
                payload["type"] = block.type
                payload["image_url"] = deepcopy(block.source)
                payload.update(deepcopy(block.extra))
                formatted.append(payload)
            elif isinstance(block.raw, dict):
                formatted.append(deepcopy(block.raw))
            else:
                payload = {"type": block.type}
                payload.update(deepcopy(block.extra))
                formatted.append(payload)
        return formatted

    def _parse_tool_definition(self, tool: dict[str, Any]) -> ToolDefinition:
        payload = dict(tool or {})
        function = payload.get("function") if isinstance(payload.get("function"), dict) else payload
        return ToolDefinition(
            name=str(function.get("name") or ""),
            description=function.get("description"),
            input_schema=deepcopy(function.get("parameters") or function.get("input_schema") or {}),
            type=str(payload.get("type") or "function"),
            extra={"raw": deepcopy(tool), **_without(payload, {"type", "function"})},
        )

    def _format_tool_definition(self, tool: ToolDefinition) -> dict[str, Any]:
        raw = tool.extra.get("raw")
        if isinstance(raw, dict):
            return deepcopy(raw)
        return {
            "type": tool.type,
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": deepcopy(tool.input_schema),
            },
        }

    def _parse_tool_call(self, call: dict[str, Any]) -> ToolCall:
        payload = dict(call or {})
        function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
        arguments: Any = function.get("arguments")
        return ToolCall(
            id=payload.get("id"),
            name=function.get("name") or payload.get("name"),
            arguments=arguments,
            type=str(payload.get("type") or "function"),
            index=payload.get("index"),
            raw=deepcopy(call),
            extra={**_without(function, {"name", "arguments"}), **_without(payload, {"id", "function", "type", "index", "name"})},
        )

    def _format_tool_call(self, call: ToolCall) -> dict[str, Any]:
        payload = deepcopy(call.raw) if isinstance(call.raw, dict) else {}
        payload["type"] = call.type
        if call.id:
            payload["id"] = call.id
        if call.index is not None:
            payload["index"] = call.index
        function = deepcopy(payload.get("function")) if isinstance(payload.get("function"), dict) else {}
        function["name"] = call.name or ""
        function["arguments"] = _format_arguments(call.arguments)
        payload["function"] = function
        return payload


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


def _extract_reasoning(payload: dict[str, Any]) -> list[ReasoningBlock]:
    blocks = []
    for field_name in ("reasoning_content", "reasoning"):
        value = payload.get(field_name)
        if value:
            blocks.append(ReasoningBlock(type=field_name, text=str(value), extra={"source_field": field_name}))
    return blocks


def _format_arguments(arguments: Any) -> str:
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    return json.dumps(serialize_value(arguments), separators=(",", ":"))


def _format_openai_usage(usage: Usage | None) -> dict[str, Any] | None:
    """Format normalized usage using OpenAI Chat's public field names."""

    if usage is None:
        return None
    payload: dict[str, Any] = {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens or (usage.input_tokens + usage.output_tokens),
    }
    prompt_details: dict[str, Any] = {}
    if usage.cache_read_tokens:
        prompt_details["cached_tokens"] = usage.cache_read_tokens
    if usage.cache_write_tokens:
        prompt_details["cache_creation_tokens"] = usage.cache_write_tokens
    if prompt_details:
        payload["prompt_tokens_details"] = prompt_details
    completion_details: dict[str, Any] = {}
    if usage.reasoning_tokens:
        completion_details["reasoning_tokens"] = usage.reasoning_tokens
    if completion_details:
        payload["completion_tokens_details"] = completion_details
    if usage.cost:
        cost_details: dict[str, Any] = dict(usage.cost.metadata)
        if usage.cost.provider_reported_cost is not None:
            cost_details["total_cost"] = usage.cost.provider_reported_cost
        elif usage.cost.estimated_cost is not None:
            cost_details["estimated_cost"] = usage.cost.estimated_cost
        cost_details["currency"] = usage.cost.currency
        if usage.cost.source:
            cost_details["source"] = usage.cost.source
        payload["cost_details"] = cost_details
    return payload


def _without(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: deepcopy(v) for k, v in payload.items() if k not in keys}
