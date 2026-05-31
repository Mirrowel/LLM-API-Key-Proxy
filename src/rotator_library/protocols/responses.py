# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""OpenAI Responses protocol adapter.

Responses is important enough to model natively rather than forcing it through a
chat-completions shape. This adapter focuses on loss-conscious parsing and
formatting; storage, routes, and WebSocket transport are later phases.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, ClassVar, Iterable

from .base import ProtocolAdapter
from .operation import OPERATION_RESPONSES
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
    text_blocks,
)

_GENERATION_PARAMS = {
    "include",
    "instructions",
    "max_output_tokens",
    "parallel_tool_calls",
    "reasoning",
    "store",
    "temperature",
    "text",
    "tool_choice",
    "top_p",
    "truncation",
    "user",
}

_REQUEST_CORE_FIELDS = {
    "model",
    "input",
    "metadata",
    "previous_response_id",
    "stream",
    "tools",
    *_GENERATION_PARAMS,
}


class ResponsesProtocol(ProtocolAdapter):
    """Adapter for OpenAI Responses request, response, and event stream shapes.

    The protocol keeps output items in addition to parsed messages because later
    response storage and continuation features need item-level fidelity.
    """

    name: ClassVar[str] = "responses"
    aliases: ClassVar[tuple[str, ...]] = ("openai_responses", "response_api")
    supported_transports: ClassVar[tuple[str, ...]] = ("http", "sse")
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_RESPONSES,)
    future_transports: ClassVar[tuple[str, ...]] = ("websocket",)

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        generation_params = {k: deepcopy(request[k]) for k in _GENERATION_PARAMS if k in request and k != "instructions"}
        return UnifiedRequest(
            operation=OPERATION_RESPONSES,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            messages=self._parse_input(request.get("input")),
            system=text_blocks(request.get("instructions")) if request.get("instructions") is not None else [],
            tools=[self._parse_tool(tool) for tool in request.get("tools") or []],
            stream=bool(request.get("stream", False)),
            generation_params=generation_params,
            previous_response_id=request.get("previous_response_id"),
            metadata=deepcopy(request.get("metadata") or {}),
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": unified_request.model,
            "input": [self._format_input_message(message) for message in unified_request.messages],
        }
        instructions = first_text(unified_request.system)
        if instructions is not None:
            payload["instructions"] = instructions
        if unified_request.previous_response_id:
            payload["previous_response_id"] = unified_request.previous_response_id
        if unified_request.tools:
            payload["tools"] = [self._format_tool(tool) for tool in unified_request.tools]
        if unified_request.stream:
            payload["stream"] = True
        if unified_request.metadata:
            payload["metadata"] = deepcopy(unified_request.metadata)
        payload.update(deepcopy(unified_request.generation_params))
        payload.update(deepcopy(unified_request.extra))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = _as_dict(raw_response)
        output = deepcopy(response.get("output") or [])
        messages: list[UnifiedMessage] = []
        for index, item in enumerate(output):
            if isinstance(item, dict):
                parsed = self._parse_output_item(item)
                if parsed:
                    parsed.extra["_output_index"] = index
                    messages.append(parsed)
        return UnifiedResponse(
            operation=OPERATION_RESPONSES,
            id=response.get("id"),
            model=response.get("model") or getattr(context, "model", None),
            messages=messages,
            output=output,
            stop_reason=response.get("status"),
            usage=self.extract_usage(response, context),
            metadata={"object": response.get("object"), "created_at": response.get("created_at")},
            raw=deepcopy(response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"id", "object", "created_at", "model", "output", "usage", "status"}},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:
        if unified_response.output:
            output = deepcopy(unified_response.output)
            for fallback_index, message in enumerate(unified_response.messages):
                output_index = message.extra.get("_output_index", fallback_index)
                if isinstance(output_index, int) and 0 <= output_index < len(output):
                    output[output_index] = self._format_output_message(message, output_index)
                else:
                    output.append(self._format_output_message(message, fallback_index))
        else:
            output = [self._format_output_message(message, index) for index, message in enumerate(unified_response.messages)]
        payload = {
            "id": unified_response.id,
            "object": unified_response.metadata.get("object", "response"),
            "created_at": unified_response.metadata.get("created_at"),
            "model": unified_response.model,
            "status": unified_response.stop_reason,
            "output": output,
            "usage": _format_responses_usage(unified_response.usage),
        }
        payload.update(deepcopy(unified_response.extra))
        return {k: v for k, v in payload.items() if v is not None}

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        event = _decode_sse_data(raw_event)
        if event == "[DONE]":
            return UnifiedStreamEvent(type="done", operation=OPERATION_RESPONSES, raw=deepcopy(raw_event))
        data = _as_dict(event)
        event_type = str(data.get("type") or data.get("event") or "chunk")
        if event_type in {"error", "response.error"} or data.get("error") is not None:
            return UnifiedStreamEvent(type="error", operation=OPERATION_RESPONSES, error=deepcopy(data.get("error", data)), raw=deepcopy(raw_event), extra={"payload": data})
        if event_type in {"response.completed", "response.failed", "response.incomplete"}:
            response = self.parse_response(data.get("response") or {}, context)
            return UnifiedStreamEvent(type=event_type, operation=OPERATION_RESPONSES, message=response.messages[0] if response.messages else None, usage=response.usage, raw=deepcopy(raw_event), extra={"payload": data})
        if event_type == "response.output_text.delta":
            message = UnifiedMessage(role="assistant", content=text_blocks(data.get("delta") or ""))
            return UnifiedStreamEvent(type="message_delta", operation=OPERATION_RESPONSES, delta=message, raw=deepcopy(raw_event), extra={"payload": data, "output_index": data.get("output_index"), "content_index": data.get("content_index")})
        if event_type in {"response.output_item.added", "response.output_item.done"} and isinstance(data.get("item"), dict):
            message = self._parse_output_item(data["item"])
            return UnifiedStreamEvent(type=event_type, operation=OPERATION_RESPONSES, message=message, raw=deepcopy(raw_event), extra={"payload": data})
        return UnifiedStreamEvent(type=event_type, operation=OPERATION_RESPONSES, raw=deepcopy(raw_event), extra={"payload": data})

    def extract_usage(self, raw_or_unified: Any, context: ProtocolContext | None = None) -> Usage | None:
        if isinstance(raw_or_unified, (UnifiedResponse, UnifiedStreamEvent)):
            return raw_or_unified.usage
        payload = _as_dict(raw_or_unified)
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else payload
        if not isinstance(usage, dict) or not any(key.endswith("tokens") for key in usage):
            return None
        input_details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
        output_details = usage.get("output_tokens_details") if isinstance(usage.get("output_tokens_details"), dict) else {}
        cost = None
        cost_details = usage.get("cost_details")
        if isinstance(cost_details, dict):
            provider_cost = cost_details.get("total_cost") or cost_details.get("request_cost_usd") or cost_details.get("cost")
            cost = CostDetails(
                provider_reported_cost=float(provider_cost) if provider_cost is not None else None,
                currency=str(cost_details.get("currency") or "USD"),
                source="usage.cost_details",
                metadata={k: deepcopy(v) for k, v in cost_details.items() if k not in {"total_cost", "cost", "currency"}},
            )
        return Usage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            cache_read_tokens=int(input_details.get("cached_tokens") or 0),
            cache_write_tokens=int(input_details.get("cache_creation_tokens") or usage.get("cache_creation_tokens") or 0),
            reasoning_tokens=int(output_details.get("reasoning_tokens") or 0),
            cost=cost,
            raw=deepcopy(usage),
        )

    def _parse_input(self, input_value: Any) -> list[UnifiedMessage]:
        if input_value is None:
            return []
        if isinstance(input_value, str):
            return [UnifiedMessage(role="user", content=text_blocks(input_value), raw=input_value)]
        if not isinstance(input_value, list):
            return [UnifiedMessage(role="user", content=[ContentBlock(type="unknown", raw=deepcopy(input_value))], raw=deepcopy(input_value))]
        messages = []
        for item in input_value:
            if isinstance(item, dict):
                messages.append(self._parse_input_item(item))
            else:
                messages.append(UnifiedMessage(role="user", content=text_blocks(str(item)), raw=deepcopy(item)))
        return messages

    def _parse_input_item(self, item: dict[str, Any]) -> UnifiedMessage:
        item_type = item.get("type")
        if item_type in {"message", None}:
            return UnifiedMessage(
                role=str(item.get("role") or "user"),
                content=self._parse_content(item.get("content")),
                raw=deepcopy(item),
                extra={k: deepcopy(v) for k, v in item.items() if k not in {"type", "role", "content"}},
            )
        if item_type == "function_call_output":
            return UnifiedMessage(
                role="tool",
                content=[ContentBlock(type="tool_result", tool_result=ToolResult(tool_call_id=item.get("call_id"), content=item.get("output")), raw=deepcopy(item))],
                tool_call_id=item.get("call_id"),
                raw=deepcopy(item),
            )
        return UnifiedMessage(role=str(item.get("role") or "user"), content=[ContentBlock(type=str(item_type or "unknown"), raw=deepcopy(item))], raw=deepcopy(item))

    def _format_input_message(self, message: UnifiedMessage) -> dict[str, Any]:
        if isinstance(message.raw, dict):
            payload = deepcopy(message.raw)
            if payload.get("type") == "function_call_output":
                payload["call_id"] = message.tool_call_id or payload.get("call_id")
                result = message.content[0].tool_result if message.content and message.content[0].tool_result else None
                if result:
                    payload["output"] = deepcopy(result.content)
                return payload
            payload["role"] = message.role
            payload["content"] = self._format_content(message.content)
            return payload
        return {"type": "message", "role": message.role, "content": self._format_content(message.content)}

    def _parse_output_item(self, item: dict[str, Any]) -> UnifiedMessage | None:
        item_type = item.get("type")
        if item_type == "message":
            return UnifiedMessage(
                role=str(item.get("role") or "assistant"),
                content=self._parse_content(item.get("content")),
                raw=deepcopy(item),
                extra={k: deepcopy(v) for k, v in item.items() if k not in {"type", "role", "content"}},
            )
        if item_type == "reasoning":
            reasoning = ReasoningBlock(type="reasoning", text=_reasoning_text(item), extra={k: deepcopy(v) for k, v in item.items() if k not in {"type", "summary"}})
            reasoning.raw = deepcopy(item)
            return UnifiedMessage(role="assistant", content=[ContentBlock(type="reasoning", reasoning=reasoning, raw=deepcopy(item))], reasoning=[reasoning], raw=deepcopy(item))
        if item_type in {"function_call", "custom_tool_call"}:
            call = ToolCall(id=item.get("call_id") or item.get("id"), name=item.get("name"), arguments=item.get("arguments") or item.get("input"), type=str(item_type), raw=deepcopy(item))
            return UnifiedMessage(role="assistant", content=[ContentBlock(type=str(item_type), tool_call=call, raw=deepcopy(item))], tool_calls=[call], raw=deepcopy(item))
        return None

    def _format_output_message(self, message: UnifiedMessage, index: int) -> dict[str, Any]:
        if isinstance(message.raw, dict):
            payload = deepcopy(message.raw)
            item_type = payload.get("type")
            if item_type == "message":
                payload["role"] = message.role
                payload["content"] = self._format_content(message.content)
                return payload
            if item_type == "reasoning" and message.reasoning:
                payload["summary"] = [{"type": "summary_text", "text": message.reasoning[0].text or ""}]
                return payload
            if item_type in {"function_call", "custom_tool_call"} and message.tool_calls:
                call = message.tool_calls[0]
                payload["call_id"] = call.id
                payload["name"] = call.name
                payload["arguments"] = deepcopy(call.arguments)
                return payload
        return {"id": f"msg_{index}", "type": "message", "role": message.role, "content": self._format_content(message.content)}

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
                blocks.append(ContentBlock(type="input_text", text=block, raw=block))
                continue
            if not isinstance(block, dict):
                blocks.append(ContentBlock(type="unknown", raw=deepcopy(block)))
                continue
            block_type = str(block.get("type") or "text")
            if block_type in {"input_text", "output_text", "text"}:
                blocks.append(ContentBlock(type=block_type, text=block.get("text", ""), raw=deepcopy(block), extra=_without(block, {"type", "text"})))
            elif block_type in {"input_image", "image_url"}:
                blocks.append(ContentBlock(type=block_type, source=deepcopy(block.get("image_url") or block.get("source")), raw=deepcopy(block), extra=_without(block, {"type", "image_url", "source"})))
            else:
                blocks.append(ContentBlock(type=block_type, raw=deepcopy(block), extra=_without(block, {"type"})))
        return blocks

    def _format_content(self, blocks: Iterable[ContentBlock]) -> list[dict[str, Any]]:
        formatted = []
        for block in blocks:
            if block.type in {"input_text", "output_text", "text"}:
                payload = deepcopy(block.raw) if isinstance(block.raw, dict) else {"type": block.type}
                payload["type"] = block.type
                payload["text"] = block.text or ""
                payload.update(deepcopy(block.extra))
                formatted.append(payload)
            elif block.type in {"input_image", "image_url"}:
                payload = deepcopy(block.raw) if isinstance(block.raw, dict) else {"type": block.type}
                payload["type"] = block.type
                payload["image_url"] = deepcopy(block.source)
                payload.update(deepcopy(block.extra))
                formatted.append(payload)
            else:
                payload = {"type": block.type}
                payload.update(deepcopy(block.extra))
                formatted.append(payload)
        return formatted

    def _parse_tool(self, tool: dict[str, Any]) -> ToolDefinition:
        payload = dict(tool or {})
        parameters = payload.get("parameters") or payload.get("input_schema") or {}
        return ToolDefinition(
            name=str(payload.get("name") or ""),
            description=payload.get("description"),
            input_schema=deepcopy(parameters),
            type=str(payload.get("type") or "function"),
            extra={k: deepcopy(v) for k, v in payload.items() if k not in {"type", "name", "description", "parameters", "input_schema"}},
        )

    def _format_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        payload = {"type": tool.type, "name": tool.name, "parameters": deepcopy(tool.input_schema)}
        if tool.description is not None:
            payload["description"] = tool.description
        payload.update(deepcopy(tool.extra))
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


def _reasoning_text(item: dict[str, Any]) -> str | None:
    summary = item.get("summary")
    if isinstance(summary, list):
        parts = []
        for part in summary:
            if isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts) if parts else None
    return str(summary) if summary else None


def _without(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: deepcopy(v) for k, v in payload.items() if k not in keys}


def _format_responses_usage(usage: Usage | None) -> dict[str, Any] | None:
    """Format normalized usage using OpenAI Responses public field names."""

    if usage is None:
        return None
    payload: dict[str, Any] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens or (usage.input_tokens + usage.output_tokens),
    }
    input_details: dict[str, Any] = {}
    if usage.cache_read_tokens:
        input_details["cached_tokens"] = usage.cache_read_tokens
    if usage.cache_write_tokens:
        # OpenAI Responses does not have a universal cache-write field, but this
        # extension keeps provider-reported cache creation visible without
        # leaking the unified internal `cache_write_tokens` key.
        input_details["cache_creation_tokens"] = usage.cache_write_tokens
    if input_details:
        payload["input_tokens_details"] = input_details
    output_details: dict[str, Any] = {}
    if usage.reasoning_tokens:
        output_details["reasoning_tokens"] = usage.reasoning_tokens
    if output_details:
        payload["output_tokens_details"] = output_details
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
