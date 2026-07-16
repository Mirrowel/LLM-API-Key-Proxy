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
from .canonical import (
    add_conversion_warning,
    canonical_stop_reason,
    canonical_structured_output,
    canonical_tool_arguments,
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
    tool_result_object,
)
from .operation import OPERATION_CHAT, OPERATION_COUNT_TOKENS, OPERATION_GENERATE, OPERATION_UNKNOWN, normalize_operation
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
    "toolConfig",
    "tool_config",
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
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_CHAT, OPERATION_COUNT_TOKENS, "generate", "stream_generate")

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        generation_config = deepcopy(request.get("generationConfig") or request.get("generation_config") or {})
        safety_settings = deepcopy(request.get("safetySettings") or request.get("safety_settings") or [])
        tool_config = deepcopy(request.get("toolConfig") or request.get("tool_config") or {})
        generation_params = _parse_gemini_generation_params(generation_config, tool_config)
        if safety_settings:
            generation_params["safety_settings"] = safety_settings
        return UnifiedRequest(
            operation=_operation_from_context(context, OPERATION_CHAT),
            logical_operation=OPERATION_GENERATE,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            messages=resolve_tool_result_names(
                normalize_tool_result_messages([self._parse_content(content) for content in request.get("contents") or []])
            ),
            system=self._parse_system(request.get("systemInstruction") or request.get("system_instruction")),
            tools=self._parse_tools(request.get("tools") or []),
            stream=bool(request.get("stream", False)),
            modalities=[str(value).lower() for value in generation_config.get("responseModalities") or []],
            generation_params=generation_params,
            response_format=deepcopy(generation_params.get("structured_output")),
            source_protocol=self.name,
            extensions={self.name: {"generationConfig": generation_config, "safetySettings": safety_settings, "toolConfig": tool_config}},
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        validate_generative_request(unified_request, self.name, context)
        preserve_source = is_same_protocol(context, self.name, unified_request.source_protocol)
        emit_opaque_state = may_emit_opaque_provider_state(context, preserve_source=preserve_source)
        payload: dict[str, Any] = {
            "contents": [
                self._format_content(message, preserve_source=preserve_source, emit_opaque_state=emit_opaque_state)
                for message in resolve_tool_result_names(deepcopy(conversation_messages(unified_request)))
            ],
        }
        if unified_request.model:
            payload["model"] = unified_request.model
        instructions = instruction_blocks(unified_request)
        if instructions:
            payload["systemInstruction"] = {
                "parts": self._format_parts(
                    instructions,
                    preserve_source=preserve_source,
                    emit_opaque_state=emit_opaque_state,
                )
            }
        generation_config, safety_settings, tool_config = self._format_generation_params(unified_request, preserve_source=preserve_source)
        if generation_config:
            payload["generationConfig"] = deepcopy(generation_config)
        if safety_settings:
            payload["safetySettings"] = deepcopy(safety_settings)
        if tool_config:
            payload["toolConfig"] = deepcopy(tool_config)
        if unified_request.tools:
            payload["tools"] = self._format_tools(unified_request.tools, preserve_source=preserve_source)
        if unified_request.stream:
            payload["stream"] = True
        payload.update(source_extensions(unified_request.extra, context, self.name, unified_request.source_protocol))
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
        canonical_reason = canonical_stop_reason(stop_reason)
        if canonical_reason == "stop" and any(message_tool_calls(message) for message in messages):
            canonical_reason = "tool_use"
        return UnifiedResponse(
            operation=_response_operation(response, context),
            logical_operation=OPERATION_GENERATE,
            id=response.get("responseId") or response.get("id"),
            model=response.get("modelVersion") or getattr(context, "model", None),
            messages=messages,
            stop_reason=canonical_reason,
            usage=self.extract_usage(response, context),
            metadata={"promptFeedback": deepcopy(response.get("promptFeedback")), "modelVersion": response.get("modelVersion"), "native_stop_reason": stop_reason},
            source_protocol=self.name,
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
        validate_generative_response(unified_response, self.name)
        preserve_source = is_same_protocol(context, self.name, unified_response.source_protocol)
        emit_opaque_state = may_emit_opaque_provider_state(context, preserve_source=preserve_source)
        messages = unified_response.messages if preserve_source else [coalesce_assistant_message(unified_response.messages)]
        candidates = []
        for index, message in enumerate(message for message in messages if message.content or message.reasoning or message.tool_calls):
            candidate = {"index": index, "content": self._format_content(message, preserve_source=preserve_source, emit_opaque_state=emit_opaque_state)}
            if unified_response.stop_reason:
                candidate["finishReason"] = format_stop_reason(unified_response.stop_reason, self.name)
            if preserve_source:
                candidate.update(deepcopy(message.extra.get("candidate") or {}))
            candidates.append(candidate)
        payload = {
            "responseId": unified_response.id,
            "modelVersion": unified_response.model,
            "candidates": candidates,
            "usageMetadata": self._format_usage(unified_response.usage),
            "promptFeedback": deepcopy(unified_response.metadata.get("promptFeedback")),
        }
        payload.update(source_extensions(unified_response.extra, context, self.name, unified_response.source_protocol))
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

    def _format_content(
        self,
        message: UnifiedMessage,
        *,
        preserve_source: bool = True,
        emit_opaque_state: bool = True,
    ) -> dict[str, Any]:
        role = message.extra.get("gemini_role") if preserve_source else None
        role = role or ("model" if message.role in {"assistant", "model"} else "user")
        parts = self._format_parts(
            ordered_message_blocks(message),
            preserve_source=preserve_source,
            emit_opaque_state=emit_opaque_state,
        )
        payload = {"role": role, "parts": parts}
        if preserve_source:
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
                reasoning = ReasoningBlock(type="reasoning", text=part.get("text"), signature=part.get("thoughtSignature"), raw=deepcopy(part), extra=_without(part, {"text", "thought", "thoughtSignature"}))
                return ContentBlock(type="reasoning", text=part.get("text", ""), reasoning=reasoning, raw=deepcopy(part), extra=_without(part, {"text"}))
            return ContentBlock(type="text", text=part.get("text", ""), raw=deepcopy(part), extra=_without(part, {"text"}))
        if "inlineData" in part or "inline_data" in part:
            source = part.get("inlineData") or part.get("inline_data")
            media = _parse_gemini_media_source(source, inline=True)
            return ContentBlock(type=_media_block_type(media.media_type), source=media, raw=deepcopy(part), extra=_without(part, {"inlineData", "inline_data"}))
        if "fileData" in part or "file_data" in part:
            source = part.get("fileData") or part.get("file_data")
            media = _parse_gemini_media_source(source, inline=False)
            return ContentBlock(type=_media_block_type(media.media_type), source=media, raw=deepcopy(part), extra=_without(part, {"fileData", "file_data"}))
        if "functionCall" in part or "function_call" in part:
            call = part.get("functionCall") or part.get("function_call") or {}
            return ContentBlock(type="tool_call", tool_call=ToolCall(id=call.get("id"), name=call.get("name"), arguments=canonical_tool_arguments(call.get("args")), type="function", raw=deepcopy(call)), raw=deepcopy(part), extra=_without(part, {"functionCall", "function_call"}))
        if "functionResponse" in part or "function_response" in part:
            response = part.get("functionResponse") or part.get("function_response") or {}
            result_content = canonical_tool_arguments(response.get("response"))
            return ContentBlock(type="tool_result", tool_result=ToolResult(tool_call_id=response.get("id") or response.get("name"), name=response.get("name"), content=result_content, raw=deepcopy(response)), raw=deepcopy(part), extra=_without(part, {"functionResponse", "function_response"}))
        return ContentBlock(type="unknown", raw=deepcopy(part), extra=deepcopy(part))

    def _format_parts(
        self,
        blocks: Iterable[ContentBlock],
        *,
        preserve_source: bool = True,
        emit_opaque_state: bool = True,
    ) -> list[dict[str, Any]]:
        parts = []
        for block in blocks:
            if block.tool_call:
                parts.append(self._format_tool_call(block.tool_call, preserve_source=preserve_source))
            elif block.tool_result:
                parts.append(self._format_tool_result(block.tool_result, preserve_source=preserve_source))
            elif block.type in {"image", "audio", "video", "file", "document"}:
                parts.append(_format_gemini_media(block, preserve_source=preserve_source))
            elif block.reasoning:
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {}
                payload["text"] = block.reasoning.text or ""
                payload["thought"] = True
                if emit_opaque_state and block.reasoning.signature:
                    payload["thoughtSignature"] = block.reasoning.signature
                elif not emit_opaque_state:
                    payload.pop("thoughtSignature", None)
                parts.append(payload)
            else:
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {}
                payload["text"] = block.text or ""
                if preserve_source:
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

    def _format_tools(self, tools: Iterable[ToolDefinition], *, preserve_source: bool = True) -> list[dict[str, Any]]:
        grouped: dict[int, dict[str, Any]] = {}
        ungrouped: list[dict[str, Any]] = []
        for tool in tools:
            raw_container = tool.extra.get("raw_container")
            container_index = tool.extra.get("container_index")
            declaration_index = tool.extra.get("declaration_index")
            if preserve_source and isinstance(raw_container, dict) and isinstance(container_index, int) and isinstance(declaration_index, int):
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
            ungrouped.append(self._format_tool(tool, preserve_source=preserve_source))
        return [grouped[index] for index in sorted(grouped)] + ungrouped

    def _format_tool(self, tool: ToolDefinition, *, preserve_source: bool = True) -> dict[str, Any]:
        raw = tool.extra.get("raw")
        if preserve_source and isinstance(raw, dict):
            return deepcopy(raw)
        return {"functionDeclarations": [{"name": tool.name, "description": tool.description, "parameters": deepcopy(tool.input_schema)}]}

    def _format_tool_call(self, call: ToolCall, *, preserve_source: bool) -> dict[str, Any]:
        payload = deepcopy(call.raw) if preserve_source and isinstance(call.raw, dict) else {}
        function_call = {"name": call.name or "", "args": tool_arguments_object(call.arguments)}
        if call.id and not (preserve_source and call.extra.get("synthetic_id")):
            function_call["id"] = call.id
        payload["functionCall"] = function_call
        return payload

    def _format_tool_result(self, result: ToolResult, *, preserve_source: bool) -> dict[str, Any]:
        payload = deepcopy(result.raw) if preserve_source and isinstance(result.raw, dict) else {}
        result_content = {"error": result.content} if result.is_error else result.content
        response = {"name": result.name or result.tool_call_id or "", "response": tool_result_object(result_content)}
        if result.tool_call_id and result.name and not (preserve_source and result.extra.get("synthetic_tool_call_id")):
            response["id"] = result.tool_call_id
        payload["functionResponse"] = response
        return payload

    def _format_generation_params(self, request: UnifiedRequest, *, preserve_source: bool) -> tuple[dict[str, Any], list[Any], dict[str, Any]]:
        source = request.extensions.get(self.name, {}) if preserve_source else {}
        generation = deepcopy(source.get("generationConfig") or {})
        safety = deepcopy(source.get("safetySettings") or [])
        tool_config = deepcopy(source.get("toolConfig") or {})
        params = deepcopy(request.generation_params)
        canonical_safety = params.pop("safety_settings", None)
        if canonical_safety is not None:
            safety = deepcopy(canonical_safety)
        if request.modalities:
            generation["responseModalities"] = [str(value).upper() for value in request.modalities]
        mapping = {
            "max_output_tokens": "maxOutputTokens",
            "stop_sequences": "stopSequences",
            "top_p": "topP",
            "top_k": "topK",
            "temperature": "temperature",
            "candidate_count": "candidateCount",
            "seed": "seed",
            "frequency_penalty": "frequencyPenalty",
            "presence_penalty": "presencePenalty",
        }
        for canonical, wire in mapping.items():
            if canonical in params:
                generation[wire] = params.pop(canonical)
        structured = params.pop("structured_output", None)
        if isinstance(structured, dict):
            if structured.get("strict") is False:
                add_conversion_warning(
                    request,
                    code="structured_output_strictness_strengthened",
                    message="Gemini enforces its response schema; explicit strict=false was strengthened",
                    target_protocol=self.name,
                    field="structured_output.strict",
                )
            generation.update(
                {
                    key: value
                    for key, value in format_structured_output(structured, self.name).items()
                    if value is not None
                }
            )
        reasoning = params.pop("reasoning", None)
        if isinstance(reasoning, dict):
            thinking: dict[str, Any] = {}
            if reasoning.get("budget_tokens") is not None:
                thinking["thinkingBudget"] = reasoning["budget_tokens"]
            if reasoning.get("include_thoughts") is not None:
                thinking["includeThoughts"] = reasoning["include_thoughts"]
            if thinking:
                generation["thinkingConfig"] = thinking
        tool_choice = params.pop("tool_choice", None)
        if tool_choice is not None:
            tool_config = format_tool_choice(tool_choice, self.name)
        retain_supported_generation_params(
            request,
            params,
            supported=set(),
            target_protocol=self.name,
        )
        return generation, safety, tool_config

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
    supported = {OPERATION_CHAT, OPERATION_COUNT_TOKENS, "generate", "stream_generate"}
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
    return requested if requested in {"generate", "stream_generate"} else OPERATION_CHAT


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


def _parse_gemini_generation_params(generation: dict[str, Any], tool_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize Gemini generation and tool controls."""

    params: dict[str, Any] = {}
    mapping = {
        "maxOutputTokens": "max_output_tokens",
        "stopSequences": "stop_sequences",
        "topP": "top_p",
        "topK": "top_k",
        "temperature": "temperature",
        "candidateCount": "candidate_count",
        "seed": "seed",
        "frequencyPenalty": "frequency_penalty",
        "presencePenalty": "presence_penalty",
    }
    for wire, canonical in mapping.items():
        if wire in generation:
            params[canonical] = deepcopy(generation[wire])
    if generation.get("responseMimeType") is not None or generation.get("responseJsonSchema") is not None or generation.get("responseSchema") is not None:
        params["structured_output"] = canonical_structured_output(
            {
                "type": "json_schema" if generation.get("responseJsonSchema") is not None or generation.get("responseSchema") is not None else "json_object",
                "schema": deepcopy(generation.get("responseJsonSchema") or generation.get("responseSchema")),
            },
            "gemini",
        )
    thinking = generation.get("thinkingConfig")
    if isinstance(thinking, dict):
        params["reasoning"] = {
            "budget_tokens": thinking.get("thinkingBudget"),
            "include_thoughts": thinking.get("includeThoughts"),
        }
    if tool_config:
        params["tool_choice"] = _parse_gemini_tool_choice(tool_config)
    return params


def _parse_gemini_tool_choice(tool_config: dict[str, Any]) -> Any:
    """Normalize Gemini function-calling mode and allow-list."""

    config = tool_config.get("functionCallingConfig") or tool_config.get("function_calling_config") or {}
    if not isinstance(config, dict):
        return deepcopy(tool_config)
    mode = str(config.get("mode") or "AUTO").lower()
    names = deepcopy(config.get("allowedFunctionNames") or config.get("allowed_function_names") or [])
    if mode == "none":
        return {"mode": "none"}
    if mode == "any" and len(names) == 1:
        return {"mode": "named", "name": names[0]}
    if mode == "any":
        return {"mode": "required", "allowed_names": names}
    return {"mode": "auto"}


def _parse_gemini_media_source(value: Any, *, inline: bool) -> MediaSource:
    """Normalize Gemini inlineData and fileData sources."""

    payload = value if isinstance(value, dict) else {}
    return MediaSource(
        kind="base64" if inline else "url",
        media_type=payload.get("mimeType") or payload.get("mime_type"),
        url=payload.get("fileUri") or payload.get("file_uri"),
        data=payload.get("data"),
        file_id=payload.get("fileId") or payload.get("file_id"),
        raw=deepcopy(value),
        extra=_without(payload, {"mimeType", "mime_type", "fileUri", "file_uri", "data", "fileId", "file_id"}),
    )


def _media_block_type(media_type: Any) -> str:
    """Return the canonical media category for a MIME type."""

    value = str(media_type or "").lower()
    if value.startswith("image/"):
        return "image"
    if value.startswith("audio/"):
        return "audio"
    if value.startswith("video/"):
        return "video"
    return "file"


def _coerce_media_source(value: Any) -> MediaSource:
    """Coerce legacy dictionaries into a canonical media source."""

    if isinstance(value, MediaSource):
        return value
    if isinstance(value, str):
        return MediaSource(kind="url", url=value, raw=value)
    payload = value if isinstance(value, dict) else {}
    return MediaSource(
        kind="base64" if payload.get("data") else "url",
        media_type=payload.get("mimeType") or payload.get("mime_type") or payload.get("media_type"),
        url=payload.get("fileUri") or payload.get("file_uri") or payload.get("url"),
        data=payload.get("data"),
        file_id=payload.get("fileId") or payload.get("file_id"),
        raw=deepcopy(value),
    )


def _format_gemini_media(block: ContentBlock, *, preserve_source: bool) -> dict[str, Any]:
    """Format canonical media as a Gemini content part."""

    source = _coerce_media_source(block.source)
    payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {}
    if source.data:
        payload["inlineData"] = {"mimeType": source.media_type or "application/octet-stream", "data": source.data}
    else:
        file_data: dict[str, Any] = {
            "mimeType": source.media_type or "application/octet-stream",
            "fileUri": source.url or source.file_id or "",
        }
        payload["fileData"] = file_data
    return payload
