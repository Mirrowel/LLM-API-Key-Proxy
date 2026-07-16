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
from .canonical import (
    add_conversion_warning,
    canonical_stop_reason,
    canonical_structured_output,
    canonical_tool_arguments,
    canonical_tool_choice,
    coalesce_assistant_message,
    conversation_messages,
    format_stop_reason,
    format_structured_output,
    format_tool_choice,
    instruction_blocks,
    is_same_protocol,
    message_reasoning,
    message_tool_calls,
    message_tool_results,
    ordered_message_blocks,
    retain_supported_generation_params,
    resolve_tool_result_names,
    source_extensions,
    tool_arguments_text,
    tool_result_text,
)
from .operation import OPERATION_GENERATE, OPERATION_RESPONSES
from .validation import validate_generative_request, validate_generative_response
from .types import (
    ContentBlock,
    CostDetails,
    MediaSource,
    OutputItem,
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
    "background",
    "conversation",
    "include",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "parallel_tool_calls",
    "prompt",
    "prompt_cache_key",
    "reasoning",
    "safety_identifier",
    "service_tier",
    "store",
    "stream_options",
    "temperature",
    "text",
    "tool_choice",
    "top_p",
    "top_logprobs",
    "truncation",
    "user",
}

_REQUEST_CORE_FIELDS = {
    "model",
    "input",
    "metadata",
    "modalities",
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
        source_generation = {k: deepcopy(request[k]) for k in _GENERATION_PARAMS if k in request and k != "instructions"}
        generation_params = _parse_responses_generation_params(source_generation)
        if "tool_choice" in generation_params:
            generation_params["tool_choice"] = canonical_tool_choice(generation_params["tool_choice"], self.name)
        return UnifiedRequest(
            operation=OPERATION_RESPONSES,
            logical_operation=OPERATION_GENERATE,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            messages=resolve_tool_result_names(self._parse_input(request.get("input"))),
            system=text_blocks(request.get("instructions")) if request.get("instructions") is not None else [],
            tools=[self._parse_tool(tool) for tool in request.get("tools") or []],
            stream=bool(request.get("stream", False)),
            modalities=[str(value).lower() for value in request.get("modalities") or []],
            generation_params=generation_params,
            response_format=deepcopy(generation_params.get("structured_output")),
            previous_response_id=request.get("previous_response_id"),
            metadata=deepcopy(request.get("metadata") or {}),
            source_protocol=self.name,
            extensions={self.name: {"generation_params": source_generation}},
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        validate_generative_request(unified_request, self.name, context)
        preserve_source = is_same_protocol(context, self.name, unified_request.source_protocol)
        payload: dict[str, Any] = {
            "model": unified_request.model,
            "input": self._format_input(conversation_messages(unified_request), preserve_source=preserve_source),
        }
        instructions = "\n\n".join(block.text or "" for block in instruction_blocks(unified_request) if block.text)
        if instructions:
            payload["instructions"] = instructions
        if unified_request.previous_response_id:
            payload["previous_response_id"] = unified_request.previous_response_id
        if unified_request.tools:
            payload["tools"] = [self._format_tool(tool, preserve_source=preserve_source) for tool in unified_request.tools]
        if unified_request.stream:
            payload["stream"] = True
        if unified_request.modalities:
            payload["modalities"] = deepcopy(unified_request.modalities)
        if unified_request.metadata:
            payload["metadata"] = deepcopy(unified_request.metadata)
        payload.update(self._format_generation_params(unified_request, preserve_source=preserve_source))
        payload.update(source_extensions(unified_request.extra, context, self.name, unified_request.source_protocol))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = _as_dict(raw_response)
        output = deepcopy(response.get("output") or [])
        messages: list[UnifiedMessage] = []
        items: list[OutputItem] = []
        for index, item in enumerate(output):
            if isinstance(item, dict):
                parsed = self._parse_output_item(item)
                if parsed:
                    parsed.extra["_output_index"] = index
                    messages.append(parsed)
                    items.append(_output_item_from_message(parsed, item))
        stop_reason = canonical_stop_reason(response.get("status"))
        if stop_reason == "stop" and any(message_tool_calls(message) for message in messages):
            stop_reason = "tool_use"
        return UnifiedResponse(
            operation=OPERATION_RESPONSES,
            logical_operation=OPERATION_GENERATE,
            id=response.get("id"),
            model=response.get("model") or getattr(context, "model", None),
            messages=messages,
            items=items,
            output=output,
            stop_reason=stop_reason,
            usage=self.extract_usage(response, context),
            metadata={"object": response.get("object"), "created_at": response.get("created_at"), "native_status": response.get("status"), "incomplete_details": deepcopy(response.get("incomplete_details"))},
            source_protocol=self.name,
            raw=deepcopy(response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"id", "object", "created_at", "model", "output", "usage", "status"}},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:
        validate_generative_response(unified_response, self.name)
        preserve_source = is_same_protocol(context, self.name, unified_response.source_protocol)
        if preserve_source and unified_response.output:
            output = deepcopy(unified_response.output)
            for fallback_index, message in enumerate(unified_response.messages):
                output_index = message.extra.get("_output_index", fallback_index)
                if isinstance(output_index, int) and 0 <= output_index < len(output):
                    output[output_index] = self._format_output_message(message, output_index)
                else:
                    output.append(self._format_output_message(message, fallback_index))
        else:
            output = self._format_canonical_output(coalesce_assistant_message(unified_response.messages))
        payload = {
            "id": unified_response.id,
            "object": unified_response.metadata.get("object", "response"),
            "created_at": unified_response.metadata.get("created_at"),
            "model": unified_response.model,
            "status": format_stop_reason(unified_response.stop_reason, self.name),
            "output": output,
            "usage": _format_responses_usage(unified_response.usage),
        }
        payload.update(source_extensions(unified_response.extra, context, self.name, unified_response.source_protocol))
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
            provider_cost = cost_details.get("total_cost") or cost_details.get("request_cost_usd") or cost_details.get("cost") or cost_details.get("estimated_cost")
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
            result_content = canonical_tool_arguments(item.get("output"))
            return UnifiedMessage(
                role="tool",
                content=[ContentBlock(type="tool_result", tool_result=ToolResult(tool_call_id=item.get("call_id"), content=result_content), raw=deepcopy(item))],
                tool_call_id=item.get("call_id"),
                raw=deepcopy(item),
            )
        if item_type in {"function_call", "custom_tool_call"}:
            call = ToolCall(
                id=item.get("call_id") or item.get("id"),
                name=item.get("name"),
                arguments=canonical_tool_arguments(item.get("arguments") or item.get("input")),
                type="function" if item_type == "function_call" else str(item_type),
                raw=deepcopy(item),
            )
            return UnifiedMessage(
                role="assistant",
                content=[ContentBlock(type="tool_call", tool_call=call, raw=deepcopy(item))],
                tool_calls=[call],
                raw=deepcopy(item),
            )
        if item_type == "reasoning":
            reasoning = ReasoningBlock(type="reasoning", text=_reasoning_text(item), raw=deepcopy(item))
            return UnifiedMessage(
                role="assistant",
                content=[ContentBlock(type="reasoning", reasoning=reasoning, raw=deepcopy(item))],
                reasoning=[reasoning],
                raw=deepcopy(item),
            )
        return UnifiedMessage(role=str(item.get("role") or "user"), content=[ContentBlock(type=str(item_type or "unknown"), raw=deepcopy(item))], raw=deepcopy(item))

    def _format_input(self, messages: Iterable[UnifiedMessage], *, preserve_source: bool) -> list[dict[str, Any]]:
        """Format canonical turns into ordered Responses input items."""

        items: list[dict[str, Any]] = []
        for message in messages:
            visible: list[ContentBlock] = []

            def flush_visible() -> None:
                if not visible:
                    return
                residual_message = deepcopy(message)
                residual_message.content = list(visible)
                residual_message.tool_calls = []
                residual_message.reasoning = []
                items.append(self._format_input_message(residual_message, preserve_source=preserve_source))
                visible.clear()

            for block in ordered_message_blocks(message):
                if block.reasoning:
                    flush_visible()
                    if block.reasoning.text:
                        items.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": block.reasoning.text}]})
                elif block.tool_call:
                    flush_visible()
                    items.append(self._format_function_call(block.tool_call, preserve_source=preserve_source))
                elif block.tool_result:
                    flush_visible()
                    items.append(self._format_function_result(block.tool_result, preserve_source=preserve_source))
                else:
                    visible.append(block)
            flush_visible()
        return items

    def _format_input_message(self, message: UnifiedMessage, *, preserve_source: bool = True) -> dict[str, Any]:
        if preserve_source and isinstance(message.raw, dict):
            payload = deepcopy(message.raw)
            if payload.get("type") == "function_call_output":
                payload["call_id"] = message.tool_call_id or payload.get("call_id")
                result = message.content[0].tool_result if message.content and message.content[0].tool_result else None
                if result:
                    payload["output"] = deepcopy(result.content)
                return payload
            payload["role"] = message.role
            payload["content"] = self._format_content(message.content, role=message.role, preserve_source=preserve_source)
            return payload
        role = "assistant" if message.role in {"assistant", "model"} else "user"
        return {"type": "message", "role": role, "content": self._format_content(message.content, role=role, preserve_source=preserve_source)}

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
            call = ToolCall(id=item.get("call_id") or item.get("id"), name=item.get("name"), arguments=canonical_tool_arguments(item.get("arguments") or item.get("input")), type="function" if item_type == "function_call" else str(item_type), raw=deepcopy(item))
            return UnifiedMessage(role="assistant", content=[ContentBlock(type="tool_call", tool_call=call, raw=deepcopy(item))], tool_calls=[call], raw=deepcopy(item))
        return None

    def _format_output_message(self, message: UnifiedMessage, index: int) -> dict[str, Any]:
        if isinstance(message.raw, dict):
            payload = deepcopy(message.raw)
            item_type = payload.get("type")
            if item_type == "message":
                payload["role"] = message.role
                payload["content"] = self._format_content(message.content, role="assistant", output=True, preserve_source=True)
                return payload
            if item_type == "reasoning" and message.reasoning:
                payload["summary"] = [{"type": "summary_text", "text": message.reasoning[0].text or ""}]
                return payload
            if item_type in {"function_call", "custom_tool_call"} and message.tool_calls:
                call = message.tool_calls[0]
                payload["call_id"] = call.id
                payload["name"] = call.name
                payload["arguments"] = tool_arguments_text(call.arguments)
                return payload
        return {"id": f"msg_{index}", "type": "message", "role": message.role, "content": self._format_content(message.content, role=message.role, output=True, preserve_source=False)}

    def _format_canonical_output(self, message: UnifiedMessage) -> list[dict[str, Any]]:
        """Build ordered Responses output items from one canonical assistant turn."""

        output: list[dict[str, Any]] = []
        visible: list[ContentBlock] = []
        item_index = 0

        def flush_visible() -> None:
            nonlocal item_index
            if not visible:
                return
            output.append(
                {
                    "id": f"msg_{item_index}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": self._format_content(visible, role="assistant", output=True, preserve_source=False),
                }
            )
            visible.clear()
            item_index += 1

        for block in ordered_message_blocks(message):
            if block.reasoning:
                flush_visible()
                if block.reasoning.text:
                    output.append(
                        {
                            "id": f"rs_{item_index}",
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": block.reasoning.text}],
                            "status": "completed",
                        }
                    )
                    item_index += 1
            elif block.tool_call:
                flush_visible()
                item = self._format_function_call(block.tool_call, preserve_source=False)
                item.setdefault("id", f"fc_{item_index}")
                item["status"] = "completed"
                output.append(item)
                item_index += 1
            elif block.tool_result:
                flush_visible()
                output.append(self._format_function_result(block.tool_result, preserve_source=False))
                item_index += 1
            else:
                visible.append(block)
        flush_visible()
        return output

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
                blocks.append(ContentBlock(type="text", text=block.get("text", ""), raw=deepcopy(block), extra={"source_type": block_type, **_without(block, {"type", "text"})}))
            elif block_type in {"input_image", "image_url"}:
                source = _parse_responses_media_source(block)
                blocks.append(ContentBlock(type="image", source=source, raw=deepcopy(block), extra={"source_type": block_type, **_without(block, {"type", "image_url", "source"})}))
            elif block_type in {"input_file", "file"}:
                source = _parse_responses_media_source(block)
                blocks.append(ContentBlock(type="file", source=source, raw=deepcopy(block), extra={"source_type": block_type, **_without(block, {"type", "file_id", "file_data", "file_url"})}))
            else:
                blocks.append(ContentBlock(type=block_type, raw=deepcopy(block), extra=_without(block, {"type"})))
        return blocks

    def _format_content(self, blocks: Iterable[ContentBlock], *, role: str = "user", output: bool = False, preserve_source: bool = True) -> list[dict[str, Any]]:
        formatted = []
        for block in blocks:
            if block.type == "text":
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {}
                payload["type"] = "output_text" if output or role in {"assistant", "model"} else "input_text"
                payload["text"] = block.text or ""
                if preserve_source:
                    payload.update({k: deepcopy(v) for k, v in block.extra.items() if k != "source_type"})
                formatted.append(payload)
            elif block.type == "image":
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {"type": "input_image"}
                payload["type"] = "input_image"
                payload.update(_format_responses_image_source(block.source))
                if preserve_source:
                    payload.update({k: deepcopy(v) for k, v in block.extra.items() if k != "source_type"})
                formatted.append(payload)
            elif block.type in {"file", "document"}:
                payload = {"type": "input_file"}
                payload.update(_format_responses_file_source(block.source))
                formatted.append(payload)
            elif preserve_source and isinstance(block.raw, dict):
                formatted.append(deepcopy(block.raw))
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

    def _format_tool(self, tool: ToolDefinition, *, preserve_source: bool = True) -> dict[str, Any]:
        payload = {"type": "function" if tool.type == "function" else tool.type, "name": tool.name, "parameters": deepcopy(tool.input_schema)}
        if tool.description is not None:
            payload["description"] = tool.description
        if preserve_source:
            payload.update(deepcopy(tool.extra))
        return payload

    def _format_function_call(self, call: ToolCall, *, preserve_source: bool) -> dict[str, Any]:
        payload = deepcopy(call.raw) if preserve_source and isinstance(call.raw, dict) else {}
        payload.update(
            {
                "type": "function_call",
                "call_id": call.id or "",
                "name": call.name or "",
                "arguments": tool_arguments_text(call.arguments),
            }
        )
        return payload

    def _format_function_result(self, result: ToolResult, *, preserve_source: bool) -> dict[str, Any]:
        payload = deepcopy(result.raw) if preserve_source and isinstance(result.raw, dict) else {}
        result_content = {"error": result.content} if result.is_error else result.content
        payload.update({"type": "function_call_output", "call_id": result.tool_call_id or "", "output": tool_result_text(result_content)})
        return payload

    def _format_generation_params(self, request: UnifiedRequest, *, preserve_source: bool) -> dict[str, Any]:
        params = deepcopy(request.generation_params)
        original = request.extensions.get(self.name, {}).get("generation_params") if preserve_source else None
        payload = deepcopy(original) if isinstance(original, dict) else {}
        if "max_output_tokens" in params:
            payload["max_output_tokens"] = params.pop("max_output_tokens")
        if "stop_sequences" in params:
            # Responses currently has no universal stop field. Keep it only when
            # an explicitly compatible provider extension supplied one.
            params.pop("stop_sequences")
            add_conversion_warning(
                request,
                code="unsupported_optional_control",
                message="responses has no portable stop-sequence request field",
                field="stop_sequences",
                target_protocol=self.name,
            )
        reasoning = params.pop("reasoning", None)
        if isinstance(reasoning, dict):
            payload["reasoning"] = deepcopy(reasoning)
        structured = params.pop("structured_output", None)
        if isinstance(structured, dict):
            payload["text"] = {"format": format_structured_output(structured, self.name)}
        if "tool_choice" in params:
            payload["tool_choice"] = format_tool_choice(params.pop("tool_choice"), self.name)
        supported = {
            "background",
            "conversation",
            "include",
            "max_tool_calls",
            "parallel_tool_calls",
            "prompt",
            "prompt_cache_key",
            "safety_identifier",
            "service_tier",
            "store",
            "stream_options",
            "temperature",
            "top_logprobs",
            "top_p",
            "truncation",
            "user",
        }
        payload.update(
            retain_supported_generation_params(
                request,
                params,
                supported=supported,
                target_protocol=self.name,
            )
        )
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


def _parse_responses_generation_params(source: dict[str, Any]) -> dict[str, Any]:
    """Normalize Responses controls into canonical names."""

    params = deepcopy(source)
    text = params.pop("text", None)
    if isinstance(text, dict) and isinstance(text.get("format"), dict):
        params["structured_output"] = canonical_structured_output(text["format"], "responses")
    reasoning = params.get("reasoning")
    if isinstance(reasoning, dict):
        params["reasoning"] = deepcopy(reasoning)
    return params


def _parse_responses_media_source(block: dict[str, Any]) -> MediaSource:
    """Normalize Responses image and file content fields."""

    value = block.get("image_url") or block.get("file_url") or block.get("source")
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value:
            prefix, data = value.split(",", 1)
            return MediaSource(kind="base64", media_type=prefix[5:].split(";", 1)[0], data=data, raw=deepcopy(block))
        return MediaSource(kind="url", url=value, detail=block.get("detail"), raw=deepcopy(block))
    return MediaSource(
        kind="file" if block.get("file_id") else "base64" if block.get("file_data") else "url",
        media_type=block.get("mime_type") or block.get("media_type"),
        url=block.get("file_url"),
        data=block.get("file_data"),
        file_id=block.get("file_id"),
        detail=block.get("detail"),
        raw=deepcopy(block),
    )


def _coerce_media_source(value: Any) -> MediaSource:
    """Coerce legacy media dictionaries into canonical form."""

    if isinstance(value, MediaSource):
        return value
    if isinstance(value, str):
        return MediaSource(kind="url", url=value, raw=value)
    payload = value if isinstance(value, dict) else {}
    return MediaSource(
        kind="file" if payload.get("file_id") else "base64" if payload.get("data") or payload.get("file_data") else "url",
        media_type=payload.get("mime_type") or payload.get("media_type"),
        url=payload.get("url") or payload.get("file_url"),
        data=payload.get("data") or payload.get("file_data"),
        file_id=payload.get("file_id"),
        detail=payload.get("detail"),
        raw=deepcopy(value),
    )


def _format_responses_image_source(value: Any) -> dict[str, Any]:
    """Format a canonical image source for Responses input content."""

    source = _coerce_media_source(value)
    if source.url:
        image_url = source.url
    elif source.data:
        image_url = f"data:{source.media_type or 'application/octet-stream'};base64,{source.data}"
    else:
        image_url = source.file_id or ""
    payload: dict[str, Any] = {"image_url": image_url}
    if source.detail:
        payload["detail"] = source.detail
    return payload


def _format_responses_file_source(value: Any) -> dict[str, Any]:
    """Format a canonical file source for Responses input content."""

    source = _coerce_media_source(value)
    if source.file_id:
        return {"file_id": source.file_id}
    if source.data:
        return {"file_data": source.data}
    if source.url:
        return {"file_url": source.url}
    return {"file_data": ""}


def _output_item_from_message(message: UnifiedMessage, raw: dict[str, Any]) -> OutputItem:
    """Create an ordered canonical output item alongside compatibility messages."""

    item_type = str(raw.get("type") or "message")
    if item_type == "reasoning":
        reasoning = message_reasoning(message)
        return OutputItem(type="reasoning", id=raw.get("id"), reasoning=reasoning[0] if reasoning else None, status=raw.get("status"), raw=deepcopy(raw))
    if item_type in {"function_call", "custom_tool_call"}:
        calls = message_tool_calls(message)
        return OutputItem(type="tool_call", id=raw.get("id"), tool_call=calls[0] if calls else None, status=raw.get("status"), raw=deepcopy(raw))
    return OutputItem(type="message", id=raw.get("id"), role=message.role, content=deepcopy(message.content), status=raw.get("status"), raw=deepcopy(raw))
