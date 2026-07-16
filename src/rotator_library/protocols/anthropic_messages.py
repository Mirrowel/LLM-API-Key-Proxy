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
from .canonical import (
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
    may_emit_opaque_provider_state,
    normalize_tool_result_messages,
    ordered_message_blocks,
    retain_supported_generation_params,
    resolve_tool_result_names,
    source_extensions,
    tool_arguments_object,
    tool_result_text,
)
from .operation import OPERATION_COUNT_TOKENS, OPERATION_GENERATE, OPERATION_MESSAGES, OPERATION_UNKNOWN, normalize_operation
from .validation import validate_generative_request, validate_generative_response
from .types import (
    ContentBlock,
    MediaSource,
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
    "output_config",
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
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_MESSAGES, OPERATION_COUNT_TOKENS)

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        source_generation = {k: deepcopy(request[k]) for k in _GENERATION_PARAMS if k in request and k != "metadata"}
        generation_params = _parse_anthropic_generation_params(source_generation)
        if "tool_choice" in generation_params:
            generation_params["tool_choice"] = canonical_tool_choice(generation_params["tool_choice"], self.name)
        structured_output = canonical_structured_output(request.get("output_config"), self.name)
        if structured_output:
            generation_params["structured_output"] = structured_output
        messages = resolve_tool_result_names(
            normalize_tool_result_messages([self._parse_message(message) for message in request.get("messages") or []])
        )
        return UnifiedRequest(
            operation=_operation_from_context(context, OPERATION_MESSAGES),
            logical_operation=OPERATION_GENERATE,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            messages=messages,
            system=self._parse_system(request.get("system")),
            tools=[self._parse_tool_definition(tool) for tool in request.get("tools") or []],
            stream=bool(request.get("stream", False)),
            generation_params=generation_params,
            response_format=structured_output,
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
            "messages": self._format_messages(
                conversation_messages(unified_request),
                preserve_source=preserve_source,
                emit_opaque_state=may_emit_opaque_provider_state(context, preserve_source=preserve_source),
            ),
        }
        system = self._format_system(instruction_blocks(unified_request), preserve_source=preserve_source)
        if system is not None:
            payload["system"] = system
        if unified_request.tools:
            payload["tools"] = [self._format_tool_definition(tool, preserve_source=preserve_source) for tool in unified_request.tools]
        if unified_request.stream:
            payload["stream"] = True
        if unified_request.metadata:
            payload["metadata"] = deepcopy(unified_request.metadata)
        payload.update(self._format_generation_params(unified_request, preserve_source=preserve_source))
        payload.update(source_extensions(unified_request.extra, context, self.name, unified_request.source_protocol))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = _as_dict(raw_response)
        operation = _response_operation(response, context)
        message = UnifiedMessage(
            role=str(response.get("role") or "assistant"),
            content=self._parse_content(response.get("content")),
            raw=deepcopy(response),
            extra={"type": response.get("type")},
        )
        self._promote_message_blocks(message)
        return UnifiedResponse(
            operation=operation,
            logical_operation=OPERATION_GENERATE if operation != OPERATION_COUNT_TOKENS else OPERATION_UNKNOWN,
            id=response.get("id"),
            model=response.get("model") or getattr(context, "model", None),
            messages=[] if operation == OPERATION_COUNT_TOKENS else [message] if response else [],
            stop_reason=canonical_stop_reason(response.get("stop_reason")),
            usage=self.extract_usage(response, context),
            metadata={"stop_sequence": response.get("stop_sequence"), "type": response.get("type"), "native_stop_reason": response.get("stop_reason")},
            source_protocol=self.name,
            raw=deepcopy(response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"id", "type", "role", "content", "model", "stop_reason", "stop_sequence", "usage"}},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:
        if unified_response.operation == OPERATION_COUNT_TOKENS:
            usage = unified_response.usage
            payload = deepcopy(unified_response.extra)
            # Normalized usage wins over raw preserved fields so later adapters
            # can correct counts without stale provider keys shadowing them.
            payload["input_tokens"] = usage.input_tokens if usage else 0
            return payload
        validate_generative_response(unified_response, self.name)
        preserve_source = is_same_protocol(context, self.name, unified_response.source_protocol)
        message = unified_response.messages[0] if preserve_source and unified_response.messages else coalesce_assistant_message(unified_response.messages)
        payload = {
            "id": unified_response.id,
            "type": unified_response.metadata.get("type", "message"),
            "role": message.role,
            "content": self._format_assistant_content(message, preserve_source=preserve_source, emit_opaque_state=may_emit_opaque_provider_state(context, preserve_source=preserve_source)),
            "model": unified_response.model,
            "stop_reason": format_stop_reason(unified_response.stop_reason, self.name),
            "stop_sequence": unified_response.metadata.get("stop_sequence"),
            "usage": self._format_usage(unified_response.usage),
        }
        payload.update(source_extensions(unified_response.extra, context, self.name, unified_response.source_protocol))
        return {k: v for k, v in payload.items() if v is not None}

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        event = _decode_sse_data(raw_event)
        if event == "[DONE]":
            return UnifiedStreamEvent(type="done", operation=OPERATION_MESSAGES, raw=deepcopy(raw_event))
        data = _as_dict(event)
        event_type = str(data.get("type") or "chunk")

        if event_type == "error" or data.get("error") is not None:
            return UnifiedStreamEvent(type="error", operation=OPERATION_MESSAGES, error=deepcopy(data.get("error", data)), raw=deepcopy(raw_event), extra={"payload": data})
        if event_type == "message_start":
            response = self.parse_response(data.get("message") or {}, context)
            return UnifiedStreamEvent(type="message_start", operation=OPERATION_MESSAGES, message=response.messages[0] if response.messages else None, usage=response.usage, raw=deepcopy(raw_event), extra={"payload": data})
        if event_type == "message_delta":
            return UnifiedStreamEvent(type="message_delta", operation=OPERATION_MESSAGES, usage=self.extract_usage(data.get("usage") or {}, context), raw=deepcopy(raw_event), extra={"payload": data, "stop_reason": (data.get("delta") or {}).get("stop_reason")})
        if event_type in {"content_block_start", "content_block_delta", "content_block_stop"}:
            return self._parse_content_stream_event(data, raw_event)
        return UnifiedStreamEvent(type=event_type, operation=OPERATION_MESSAGES, raw=deepcopy(raw_event), extra={"payload": data})

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

    def _format_system(self, blocks: Iterable[ContentBlock], *, preserve_source: bool = True) -> Any:
        block_list = list(blocks)
        if not block_list:
            return None
        if preserve_source and all(block.type == "text" and not isinstance(block.raw, dict) and not block.extra for block in block_list):
            return first_text(block_list) or ""
        return self._format_content(block_list, preserve_source=preserve_source)

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

    def _format_messages(
        self,
        messages: Iterable[UnifiedMessage],
        *,
        preserve_source: bool,
        emit_opaque_state: bool,
    ) -> list[dict[str, Any]]:
        """Format canonical turns and merge adjacent Anthropic roles."""

        formatted: list[dict[str, Any]] = []
        for message in messages:
            role = "assistant" if message.role in {"assistant", "model"} else "user"
            content = self._format_assistant_content(message, preserve_source=preserve_source, emit_opaque_state=emit_opaque_state) if role == "assistant" else self._format_user_content(message, preserve_source=preserve_source)
            payload = {"role": role, "content": content}
            if preserve_source:
                payload.update(deepcopy(message.extra))
            if formatted and formatted[-1]["role"] == role:
                previous = formatted[-1].get("content")
                if not isinstance(previous, list):
                    previous = [{"type": "text", "text": str(previous or "")}]
                previous.extend(content)
                formatted[-1]["content"] = previous
            else:
                formatted.append(payload)
        return formatted

    def _format_assistant_content(
        self,
        message: UnifiedMessage,
        *,
        preserve_source: bool,
        emit_opaque_state: bool = True,
    ) -> list[dict[str, Any]]:
        """Format reasoning, visible content, and tool calls in Anthropic order."""

        return self._format_content(
            ordered_message_blocks(message),
            preserve_source=preserve_source,
            emit_opaque_state=emit_opaque_state,
        )

    def _format_user_content(self, message: UnifiedMessage, *, preserve_source: bool) -> list[dict[str, Any]]:
        """Format user content and canonical tool results."""

        return self._format_content(ordered_message_blocks(message), preserve_source=preserve_source)

    def _format_message(
        self,
        message: UnifiedMessage,
        *,
        preserve_source: bool = True,
        emit_opaque_state: bool = True,
    ) -> dict[str, Any]:
        role = "assistant" if message.role in {"assistant", "model"} else "user"
        content = self._format_assistant_content(message, preserve_source=preserve_source, emit_opaque_state=emit_opaque_state) if role == "assistant" else self._format_user_content(message, preserve_source=preserve_source)
        payload = {"role": role, "content": content}
        if preserve_source:
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
            source = _parse_anthropic_media_source(block.get("source"), kind=block_type)
            return ContentBlock(type=block_type, source=source, raw=deepcopy(block), extra=_without(block, {"type", "source"}))
        if block_type in {"thinking", "redacted_thinking"}:
            reasoning = ReasoningBlock(
                type=block_type,
                text=block.get("thinking"),
                signature=block.get("signature"),
                redacted=block_type == "redacted_thinking",
                raw=deepcopy(block),
                extra=_without(block, {"type", "thinking", "signature"}),
            )
            return ContentBlock(type="reasoning", reasoning=reasoning, raw=deepcopy(block))
        if block_type == "tool_use":
            return ContentBlock(
                type="tool_call",
                tool_call=ToolCall(id=block.get("id"), name=block.get("name"), arguments=canonical_tool_arguments(block.get("input")), type="function", raw=deepcopy(block)),
                raw=deepcopy(block),
                extra=_without(block, {"type", "id", "name", "input"}),
            )
        if block_type == "tool_result":
            return ContentBlock(
                type="tool_result",
                tool_result=ToolResult(tool_call_id=block.get("tool_use_id"), content=canonical_tool_arguments(block.get("content")), is_error=block.get("is_error"), raw=deepcopy(block)),
                raw=deepcopy(block),
                extra=_without(block, {"type", "tool_use_id", "content", "is_error"}),
            )
        return ContentBlock(type=block_type, raw=deepcopy(block), extra=_without(block, {"type"}))

    def _format_content(
        self,
        blocks: Iterable[ContentBlock],
        *,
        preserve_source: bool = True,
        emit_opaque_state: bool = True,
    ) -> list[dict[str, Any]]:
        formatted = []
        for block in blocks:
            if block.type == "text":
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {"type": "text"}
                payload["type"] = "text"
                payload["text"] = block.text or ""
                formatted.append(payload)
            elif block.reasoning:
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {"type": "redacted_thinking" if block.reasoning.redacted else "thinking"}
                payload["type"] = "redacted_thinking" if block.reasoning.redacted else "thinking"
                if block.reasoning.text is not None:
                    payload["thinking"] = block.reasoning.text
                if emit_opaque_state and block.reasoning.signature is not None:
                    payload["signature"] = block.reasoning.signature
                elif not emit_opaque_state:
                    payload.pop("signature", None)
                if preserve_source:
                    payload.update(deepcopy(block.reasoning.extra))
                formatted.append(payload)
            elif block.tool_call:
                formatted.append(self._format_tool_call(block.tool_call, preserve_source=preserve_source))
            elif block.tool_result:
                formatted.append(self._format_tool_result(block.tool_result, preserve_source=preserve_source))
            elif block.type in {"image", "document"}:
                formatted.append(_format_anthropic_media(block, preserve_source=preserve_source))
            elif preserve_source and isinstance(block.raw, dict):
                formatted.append(deepcopy(block.raw))
        return formatted

    def _parse_tool_definition(self, tool: dict[str, Any]) -> ToolDefinition:
        payload = dict(tool or {})
        return ToolDefinition(
            name=str(payload.get("name") or ""),
            description=payload.get("description"),
            input_schema=deepcopy(payload.get("input_schema") or {}),
            type="function",
            extra=_without(payload, {"name", "description", "input_schema"}),
        )

    def _format_tool_definition(self, tool: ToolDefinition, *, preserve_source: bool = True) -> dict[str, Any]:
        payload = {"name": tool.name, "input_schema": deepcopy(tool.input_schema)}
        if tool.description is not None:
            payload["description"] = tool.description
        if preserve_source:
            payload.update(deepcopy(tool.extra))
        return payload

    def _format_tool_call(self, call: ToolCall, *, preserve_source: bool) -> dict[str, Any]:
        payload = deepcopy(call.raw) if preserve_source and isinstance(call.raw, dict) else {}
        payload.update({"type": "tool_use", "id": call.id or "", "name": call.name or "", "input": tool_arguments_object(call.arguments)})
        return payload

    def _format_tool_result(self, result: ToolResult, *, preserve_source: bool) -> dict[str, Any]:
        payload = deepcopy(result.raw) if preserve_source and isinstance(result.raw, dict) else {}
        if preserve_source and isinstance(result.content, (str, list)):
            content = deepcopy(result.content)
        else:
            content = tool_result_text(result.content)
        payload.update({"type": "tool_result", "tool_use_id": result.tool_call_id or "", "content": content})
        if result.is_error is not None:
            payload["is_error"] = result.is_error
        return payload

    def _format_generation_params(self, request: UnifiedRequest, *, preserve_source: bool) -> dict[str, Any]:
        params = deepcopy(request.generation_params)
        original = request.extensions.get(self.name, {}).get("generation_params") if preserve_source else None
        payload = deepcopy(original) if isinstance(original, dict) else {}
        if "max_output_tokens" in params:
            payload["max_tokens"] = params.pop("max_output_tokens")
        if "stop_sequences" in params:
            payload["stop_sequences"] = params.pop("stop_sequences")
        if "tool_choice" in params:
            payload["tool_choice"] = format_tool_choice(params.pop("tool_choice"), self.name)
        if "structured_output" in params:
            payload["output_config"] = format_structured_output(params.pop("structured_output"), self.name)
        reasoning = params.pop("reasoning", None)
        if isinstance(reasoning, dict):
            thinking: dict[str, Any] = {}
            if reasoning.get("budget_tokens") is not None:
                thinking = {"type": "enabled", "budget_tokens": reasoning["budget_tokens"]}
            elif reasoning.get("enabled") is False:
                thinking = {"type": "disabled"}
            if thinking:
                payload["thinking"] = thinking
        supported = {"temperature", "top_k", "top_p"}
        payload.update(
            retain_supported_generation_params(
                request,
                params,
                supported=supported,
                target_protocol=self.name,
            )
        )
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
        return UnifiedStreamEvent(type=str(data.get("type") or "content_block_delta"), operation=OPERATION_MESSAGES, delta=message, raw=deepcopy(raw_event), extra={"payload": data, "index": data.get("index")})


def _operation_from_context(context: ProtocolContext | None, default: str) -> str:
    supported = {OPERATION_MESSAGES, OPERATION_COUNT_TOKENS}
    if context and isinstance(context.provider_options, dict):
        operation = normalize_operation(context.provider_options.get("operation"))
        if operation in supported:
            return operation
    if context and isinstance(context.metadata, dict):
        operation = normalize_operation(context.metadata.get("operation"))
        if operation in supported:
            return operation
    return default


def _response_operation(response: dict[str, Any], context: ProtocolContext | None) -> str:
    requested = _operation_from_context(context, OPERATION_MESSAGES)
    if requested == OPERATION_COUNT_TOKENS:
        return OPERATION_COUNT_TOKENS
    if "input_tokens" in response and not response.get("content") and not response.get("id"):
        return OPERATION_COUNT_TOKENS
    return OPERATION_MESSAGES


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


def _parse_anthropic_generation_params(source: dict[str, Any]) -> dict[str, Any]:
    """Normalize Anthropic controls into canonical names."""

    params = deepcopy(source)
    max_tokens = params.pop("max_tokens", None)
    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens
    stop_sequences = params.pop("stop_sequences", None)
    if stop_sequences is not None:
        params["stop_sequences"] = deepcopy(stop_sequences)
    thinking = params.pop("thinking", None)
    if isinstance(thinking, dict):
        params["reasoning"] = {
            "enabled": thinking.get("type") != "disabled",
            "budget_tokens": thinking.get("budget_tokens"),
        }
    return params


def _parse_anthropic_media_source(value: Any, *, kind: str) -> MediaSource:
    """Normalize Anthropic image/document source objects."""

    payload = value if isinstance(value, dict) else {}
    source_type = str(payload.get("type") or "")
    if source_type == "base64":
        source_kind = "base64"
    elif source_type in {"url", "text"}:
        source_kind = "url" if source_type == "url" else "text"
    elif payload.get("file_id"):
        source_kind = "file"
    else:
        source_kind = source_type or kind
    return MediaSource(
        kind=source_kind,
        media_type=payload.get("media_type"),
        url=payload.get("url"),
        data=payload.get("data") or payload.get("text"),
        file_id=payload.get("file_id"),
        raw=deepcopy(value),
        extra=_without(payload, {"type", "media_type", "url", "data", "text", "file_id"}),
    )


def _coerce_media_source(value: Any) -> MediaSource:
    """Coerce legacy media dictionaries into canonical form."""

    if isinstance(value, MediaSource):
        return value
    if isinstance(value, str):
        return MediaSource(kind="url", url=value, raw=value)
    payload = value if isinstance(value, dict) else {}
    url = payload.get("url")
    data = payload.get("data")
    file_id = payload.get("file_id")
    return MediaSource(
        kind="file" if file_id else "base64" if data else "url",
        media_type=payload.get("media_type") or payload.get("mime_type"),
        url=url,
        data=data,
        file_id=file_id,
        detail=payload.get("detail"),
        raw=deepcopy(value),
    )


def _format_anthropic_media(block: ContentBlock, *, preserve_source: bool) -> dict[str, Any]:
    """Format canonical media as an Anthropic content block."""

    source = _coerce_media_source(block.source)
    payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {"type": block.type}
    if source.kind == "base64":
        payload["source"] = {
            "type": "base64",
            "media_type": source.media_type or "application/octet-stream",
            "data": source.data or "",
        }
    elif source.file_id:
        payload["source"] = {"type": "file", "file_id": source.file_id}
    elif source.url:
        payload["source"] = {"type": "url", "url": source.url}
    elif source.data and block.type == "document":
        payload["source"] = {"type": "text", "media_type": source.media_type or "text/plain", "data": source.data}
    else:
        payload["source"] = {"type": source.kind, "data": source.data or ""}
    payload["type"] = block.type
    return payload
