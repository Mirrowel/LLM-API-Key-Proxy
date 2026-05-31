# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Gemini generateContent protocol adapter.

The adapter preserves Gemini-native content parts, thought signatures, safety
settings, tools, and generation configuration so later native providers can use
the same base without forcing an OpenAI-compatible intermediate shape.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, ClassVar, Iterable

from .base import ProtocolAdapter
from .operation import OPERATION_CHAT, OPERATION_COUNT_TOKENS, OPERATION_UNKNOWN, normalize_operation
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
)

_REQUEST_CORE_FIELDS = {
    "model",
    "contents",
    "systemInstruction",
    "system_instruction",
    "tools",
    "generationConfig",
    "generation_config",
    "safetySettings",
    "safety_settings",
    "stream",
}


class GeminiProtocol(ProtocolAdapter):
    """Adapter for Gemini ``generateContent`` and stream event shapes.

    Gemini parts are richer than simple chat messages. Unknown part fields remain
    in ``extra`` and raw payloads are preserved so provider-specific subclasses
    can refine behavior without losing data.
    """

    name: ClassVar[str] = "gemini"
    aliases: ClassVar[tuple[str, ...]] = ("google_gemini", "generate_content")
    supported_transports: ClassVar[tuple[str, ...]] = ("http", "sse")
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_CHAT, OPERATION_COUNT_TOKENS)

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        generation_config = deepcopy(request.get("generationConfig") or request.get("generation_config") or {})
        safety_settings = deepcopy(request.get("safetySettings") or request.get("safety_settings") or [])
        return UnifiedRequest(
            operation=_operation_from_context(context, OPERATION_CHAT),
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            messages=[self._parse_content(content) for content in request.get("contents") or []],
            system=self._parse_system(request.get("systemInstruction") or request.get("system_instruction")),
            tools=self._parse_tools(request.get("tools") or []),
            stream=bool(request.get("stream", False)),
            generation_params={"generationConfig": generation_config, "safetySettings": safety_settings},
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contents": [self._format_content(message) for message in unified_request.messages],
        }
        if unified_request.model:
            payload["model"] = unified_request.model
        if unified_request.system:
            payload["systemInstruction"] = {"parts": self._format_parts(unified_request.system)}
        generation_config = unified_request.generation_params.get("generationConfig")
        safety_settings = unified_request.generation_params.get("safetySettings")
        if generation_config:
            payload["generationConfig"] = deepcopy(generation_config)
        if safety_settings:
            payload["safetySettings"] = deepcopy(safety_settings)
        if unified_request.tools:
            payload["tools"] = self._format_tools(unified_request.tools)
        if unified_request.stream:
            payload["stream"] = True
        payload.update(deepcopy(unified_request.extra))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = _as_dict(raw_response)
        messages: list[UnifiedMessage] = []
        stop_reason = None
        for candidate in response.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            content = candidate.get("content") if isinstance(candidate.get("content"), dict) else {}
            message = self._parse_content(content)
            message.extra["candidate"] = _without(candidate, {"content"})
            messages.append(message)
            if candidate.get("finishReason") is not None:
                stop_reason = candidate.get("finishReason")
        return UnifiedResponse(
            operation=_response_operation(response, context),
            id=response.get("responseId") or response.get("id"),
            model=response.get("modelVersion") or getattr(context, "model", None),
            messages=messages,
            stop_reason=stop_reason,
            usage=self.extract_usage(response, context),
            metadata={"promptFeedback": deepcopy(response.get("promptFeedback")), "modelVersion": response.get("modelVersion")},
            raw=deepcopy(response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"responseId", "id", "modelVersion", "candidates", "usageMetadata", "promptFeedback"}},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:
        if unified_response.operation == OPERATION_COUNT_TOKENS:
            usage = unified_response.usage
            payload = deepcopy(unified_response.extra)
            # Normalized usage wins over raw preserved fields so later adapters
            # can correct counts without stale provider keys shadowing them.
            payload["totalTokens"] = usage.total_tokens if usage else 0
            return payload
        candidates = []
        for index, message in enumerate(unified_response.messages):
            candidate = {"index": index, "content": self._format_content(message)}
            if unified_response.stop_reason:
                candidate["finishReason"] = unified_response.stop_reason
            candidate.update(deepcopy(message.extra.get("candidate") or {}))
            candidates.append(candidate)
        payload = {
            "responseId": unified_response.id,
            "modelVersion": unified_response.model,
            "candidates": candidates,
            "usageMetadata": self._format_usage(unified_response.usage),
            "promptFeedback": deepcopy(unified_response.metadata.get("promptFeedback")),
        }
        payload.update(deepcopy(unified_response.extra))
        return {k: v for k, v in payload.items() if v is not None}

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        event = _decode_sse_data(raw_event)
        if event == "[DONE]":
            return UnifiedStreamEvent(type="done", operation=OPERATION_CHAT, raw=deepcopy(raw_event))
        data = _as_dict(event)
        response = self.parse_response(data, context)
        message = response.messages[0] if response.messages else None
        return UnifiedStreamEvent(
            type="message_delta" if message else "chunk",
            operation=response.operation,
            delta=message,
            usage=response.usage,
            raw=deepcopy(raw_event),
            extra={"payload": data, "finish_reason": response.stop_reason},
        )

    def extract_usage(self, raw_or_unified: Any, context: ProtocolContext | None = None) -> Usage | None:
        if isinstance(raw_or_unified, (UnifiedResponse, UnifiedStreamEvent)):
            return raw_or_unified.usage
        payload = _as_dict(raw_or_unified)
        usage = payload.get("usageMetadata") if isinstance(payload.get("usageMetadata"), dict) else payload
        if not isinstance(usage, dict) or (not any(key.endswith("TokenCount") for key in usage) and "totalTokens" not in usage):
            return None
        input_tokens = int(usage.get("promptTokenCount") or 0)
        output_tokens = int(usage.get("candidatesTokenCount") or 0)
        reasoning_tokens = int(usage.get("thoughtsTokenCount") or 0)
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=int(usage.get("totalTokenCount") or usage.get("totalTokens") or input_tokens + output_tokens + reasoning_tokens),
            cache_read_tokens=int(usage.get("cachedContentTokenCount") or 0),
            reasoning_tokens=reasoning_tokens,
            raw=deepcopy(usage),
        )

    def _parse_system(self, system: Any) -> list[ContentBlock]:
        if system is None:
            return []
        if isinstance(system, str):
            return [ContentBlock(type="text", text=system, raw=system)]
        if isinstance(system, dict):
            return self._parse_parts(system.get("parts") or [])
        return []

    def _parse_content(self, content: dict[str, Any]) -> UnifiedMessage:
        payload = dict(content or {})
        role = str(payload.get("role") or "model")
        # Gemini uses "model" where chat protocols usually say "assistant".
        normalized_role = "assistant" if role == "model" else role
        message = UnifiedMessage(
            role=normalized_role,
            content=self._parse_parts(payload.get("parts") or []),
            raw=deepcopy(content),
            extra={"gemini_role": role, **_without(payload, {"role", "parts"})},
        )
        for block in message.content:
            if block.tool_call:
                message.tool_calls.append(block.tool_call)
            if block.reasoning:
                message.reasoning.append(block.reasoning)
        return message

    def _format_content(self, message: UnifiedMessage) -> dict[str, Any]:
        role = message.extra.get("gemini_role") or ("model" if message.role == "assistant" else message.role)
        payload = {"role": role, "parts": self._format_parts(message.content)}
        payload.update({k: deepcopy(v) for k, v in message.extra.items() if k != "gemini_role"})
        return payload

    def _parse_parts(self, parts: Iterable[Any]) -> list[ContentBlock]:
        blocks = []
        for part in parts:
            blocks.append(self._parse_part(part))
        return blocks

    def _parse_part(self, part: Any) -> ContentBlock:
        if isinstance(part, str):
            return ContentBlock(type="text", text=part, raw=part)
        if not isinstance(part, dict):
            return ContentBlock(type="unknown", raw=deepcopy(part))
        if "text" in part:
            reasoning = None
            if part.get("thought") or part.get("thoughtSignature"):
                reasoning = ReasoningBlock(type="gemini_thought", text=part.get("text"), signature=part.get("thoughtSignature"), raw=deepcopy(part), extra=_without(part, {"text", "thought", "thoughtSignature"}))
            return ContentBlock(type="text", text=part.get("text", ""), reasoning=reasoning, raw=deepcopy(part), extra=_without(part, {"text"}))
        if "inlineData" in part or "inline_data" in part:
            source = part.get("inlineData") or part.get("inline_data")
            return ContentBlock(type="inline_data", source=deepcopy(source), raw=deepcopy(part), extra=_without(part, {"inlineData", "inline_data"}))
        if "fileData" in part or "file_data" in part:
            source = part.get("fileData") or part.get("file_data")
            return ContentBlock(type="file_data", source=deepcopy(source), raw=deepcopy(part), extra=_without(part, {"fileData", "file_data"}))
        if "functionCall" in part or "function_call" in part:
            call = part.get("functionCall") or part.get("function_call") or {}
            return ContentBlock(type="function_call", tool_call=ToolCall(name=call.get("name"), arguments=deepcopy(call.get("args")), type="function_call", raw=deepcopy(call)), raw=deepcopy(part), extra=_without(part, {"functionCall", "function_call"}))
        if "functionResponse" in part or "function_response" in part:
            response = part.get("functionResponse") or part.get("function_response") or {}
            return ContentBlock(type="function_response", tool_result=ToolResult(tool_call_id=response.get("name"), content=deepcopy(response.get("response")), raw=deepcopy(response)), raw=deepcopy(part), extra=_without(part, {"functionResponse", "function_response"}))
        return ContentBlock(type="unknown", raw=deepcopy(part), extra=deepcopy(part))

    def _format_parts(self, blocks: Iterable[ContentBlock]) -> list[dict[str, Any]]:
        parts = []
        for block in blocks:
            if block.tool_call:
                payload = deepcopy(block.raw) if isinstance(block.raw, dict) else {}
                payload["functionCall"] = {"name": block.tool_call.name, "args": deepcopy(block.tool_call.arguments)}
                parts.append(payload)
            elif block.tool_result:
                payload = deepcopy(block.raw) if isinstance(block.raw, dict) else {}
                payload["functionResponse"] = {"name": block.tool_result.tool_call_id, "response": deepcopy(block.tool_result.content)}
                parts.append(payload)
            elif block.type == "inline_data":
                payload = deepcopy(block.raw) if isinstance(block.raw, dict) else {}
                payload["inlineData"] = deepcopy(block.source)
                parts.append(payload)
            elif block.type == "file_data":
                payload = deepcopy(block.raw) if isinstance(block.raw, dict) else {}
                payload["fileData"] = deepcopy(block.source)
                parts.append(payload)
            else:
                payload = deepcopy(block.raw) if isinstance(block.raw, dict) else {}
                payload["text"] = block.text or ""
                if block.reasoning:
                    payload["thought"] = True
                    if block.reasoning.signature:
                        payload["thoughtSignature"] = block.reasoning.signature
                payload.update(deepcopy(block.extra))
                parts.append(payload)
        return parts

    def _parse_tools(self, tools: Iterable[dict[str, Any]]) -> list[ToolDefinition]:
        parsed: list[ToolDefinition] = []
        for container_index, tool in enumerate(tools):
            payload = dict(tool or {})
            declarations = payload.get("functionDeclarations") or payload.get("function_declarations") or []
            if declarations:
                for index, declaration in enumerate(declarations):
                    if not isinstance(declaration, dict):
                        continue
                    parsed.append(
                        ToolDefinition(
                            name=str(declaration.get("name") or ""),
                            description=declaration.get("description"),
                            input_schema=deepcopy(declaration.get("parameters") or {}),
                            type="function",
                            extra={"raw_container": deepcopy(tool), "container_index": container_index, "declaration_index": index},
                        )
                    )
                continue
            parsed.append(
                ToolDefinition(
                    name=str(payload.get("name") or payload.get("type") or "gemini_tool"),
                    description=payload.get("description"),
                    input_schema=deepcopy(payload.get("parameters") or {}),
                    type=str(payload.get("type") or next(iter(payload.keys()), "tool")),
                    extra={"raw": deepcopy(tool)},
                )
            )
        return parsed

    def _format_tools(self, tools: Iterable[ToolDefinition]) -> list[dict[str, Any]]:
        grouped: dict[int, dict[str, Any]] = {}
        ungrouped: list[dict[str, Any]] = []
        for tool in tools:
            raw_container = tool.extra.get("raw_container")
            container_index = tool.extra.get("container_index")
            declaration_index = tool.extra.get("declaration_index")
            if isinstance(raw_container, dict) and isinstance(container_index, int) and isinstance(declaration_index, int):
                container = grouped.setdefault(container_index, deepcopy(raw_container))
                declarations = container.setdefault("functionDeclarations", [])
                while len(declarations) <= declaration_index:
                    declarations.append({})
                declaration = deepcopy(declarations[declaration_index]) if isinstance(declarations[declaration_index], dict) else {}
                declaration["name"] = tool.name
                if tool.description is not None:
                    declaration["description"] = tool.description
                declaration["parameters"] = deepcopy(tool.input_schema)
                declarations[declaration_index] = declaration
                continue
            ungrouped.append(self._format_tool(tool))
        return [grouped[index] for index in sorted(grouped)] + ungrouped

    def _format_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        raw = tool.extra.get("raw")
        if isinstance(raw, dict):
            return deepcopy(raw)
        return {"functionDeclarations": [{"name": tool.name, "description": tool.description, "parameters": deepcopy(tool.input_schema)}]}

    def _format_usage(self, usage: Usage | None) -> dict[str, int] | None:
        if usage is None:
            return None
        payload = {
            "promptTokenCount": usage.input_tokens,
            "candidatesTokenCount": usage.output_tokens,
            "totalTokenCount": usage.total_tokens,
        }
        if usage.reasoning_tokens:
            payload["thoughtsTokenCount"] = usage.reasoning_tokens
        if usage.cache_read_tokens:
            payload["cachedContentTokenCount"] = usage.cache_read_tokens
        return payload


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return {}


def _operation_from_context(context: ProtocolContext | None, default: str) -> str:
    supported = {OPERATION_CHAT, OPERATION_COUNT_TOKENS}
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
    requested = _operation_from_context(context, OPERATION_CHAT)
    if requested == OPERATION_COUNT_TOKENS:
        return OPERATION_COUNT_TOKENS
    if "totalTokens" in response and "candidates" not in response:
        return OPERATION_COUNT_TOKENS
    return OPERATION_CHAT


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
