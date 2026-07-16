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
from typing import Any, ClassVar, Iterable, Optional

from .base import ProtocolAdapter
from .canonical import (
    canonical_stop_reason,
    canonical_structured_output,
    canonical_tool_arguments,
    coalesce_assistant_message,
    canonical_tool_choice,
    format_stop_reason,
    format_structured_output,
    format_tool_choice,
    conversation_messages,
    instruction_messages,
    is_same_protocol,
    retain_supported_generation_params,
    resolve_tool_result_names,
    source_extensions,
    tool_arguments_text,
)
from .operation import OPERATION_CHAT, OPERATION_GENERATE
from .validation import validate_generative_request, validate_generative_response
from .types import (
    ContentBlock,
    CostDetails,
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
    "modalities",
    "audio",
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
        messages = resolve_tool_result_names([self._parse_message(message) for message in request.get("messages") or []])
        tools = [self._parse_tool_definition(tool) for tool in request.get("tools") or []]
        source_generation_params = {k: deepcopy(request[k]) for k in _GENERATION_PARAMS if k in request}
        generation_params = _parse_openai_generation_params(source_generation_params)
        structured_output = canonical_structured_output(request.get("response_format"), self.name)
        if structured_output:
            generation_params["structured_output"] = structured_output
        if "tool_choice" in generation_params:
            generation_params["tool_choice"] = canonical_tool_choice(generation_params["tool_choice"], self.name)
        if request.get("audio") is not None:
            generation_params["audio_output"] = deepcopy(request["audio"])
        extra = {k: deepcopy(v) for k, v in request.items() if k not in _REQUEST_CORE_FIELDS}

        return UnifiedRequest(
            operation=OPERATION_CHAT,
            logical_operation=OPERATION_GENERATE,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            messages=messages,
            tools=tools,
            stream=bool(request.get("stream", False)),
            modalities=[str(value).lower() for value in request.get("modalities") or []],
            generation_params=generation_params,
            response_format=structured_output,
            metadata=deepcopy(request.get("metadata") or {}),
            source_protocol=self.name,
            extensions={self.name: {"generation_params": source_generation_params, "response_format": deepcopy(request.get("response_format"))}},
            raw=deepcopy(raw_request),
            extra=extra,
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
        validate_generative_request(unified_request, self.name, context)
        preserve_source = is_same_protocol(context, self.name, unified_request.source_protocol)
        payload: dict[str, Any] = {
            "model": unified_request.model,
            "messages": self._format_request_messages(
                [*instruction_messages(unified_request), *conversation_messages(unified_request)],
                preserve_source=preserve_source,
            ),
        }
        if unified_request.tools:
            payload["tools"] = [self._format_tool_definition(tool, preserve_source=preserve_source) for tool in unified_request.tools]
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
            logical_operation=OPERATION_GENERATE,
            id=response.get("id"),
            model=response.get("model") or getattr(context, "model", None),
            messages=messages,
            stop_reason=canonical_stop_reason(stop_reason),
            usage=self.extract_usage(response, context),
            metadata={
                "object": response.get("object"),
                "created": response.get("created"),
                "system_fingerprint": response.get("system_fingerprint"),
                "native_stop_reason": stop_reason,
            },
            source_protocol=self.name,
            raw=deepcopy(response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in {"id", "object", "created", "model", "choices", "usage", "system_fingerprint"}},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None) -> dict[str, Any]:
        validate_generative_response(unified_response, self.name)
        preserve_source = is_same_protocol(context, self.name, unified_response.source_protocol)
        messages = unified_response.messages if preserve_source else [coalesce_assistant_message(unified_response.messages)]
        choices = []
        for index, message in enumerate(messages):
            choices.append(
                {
                    "index": index,
                    "message": _format_response_message(
                        self._format_message(message, preserve_source=preserve_source),
                        message,
                    ),
                    "finish_reason": format_stop_reason(unified_response.stop_reason, self.name),
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
        payload.update(source_extensions(unified_response.extra, context, self.name, unified_response.source_protocol))
        return {k: v for k, v in payload.items() if v is not None}

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        event = _decode_sse_data(raw_event)
        if event == "[DONE]":
            return UnifiedStreamEvent(type="done", operation=OPERATION_CHAT, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type="done", raw=deepcopy(raw_event))
        data = _as_dict(event)
        if data.get("error") is not None:
            return UnifiedStreamEvent(type="error", operation=OPERATION_CHAT, logical_operation=OPERATION_GENERATE, source_protocol=self.name, native_type="error", error=deepcopy(data["error"]), raw=deepcopy(raw_event), extra={"payload": data})

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
            logical_operation=OPERATION_GENERATE,
            source_protocol=self.name,
            native_type="chat.completion.chunk",
            delta=delta_message,
            usage=usage,
            raw=deepcopy(raw_event),
            extra={
                "id": data.get("id"),
                "model": data.get("model"),
                "finish_reason": canonical_stop_reason(finish_reason),
                "payload": data,
            },
        )

    def format_stream_event(self, unified_event: UnifiedStreamEvent, context: ProtocolContext | None = None) -> Any:
        if unified_event.type == "done":
            return "data: [DONE]\n\n"
        if unified_event.raw is not None and (context is None or context.source_protocol in {None, "openai_chat"}):
            payload = deepcopy(unified_event.raw)
            if unified_event.delta is not None and isinstance(payload, dict) and isinstance(payload.get("choices"), list) and payload["choices"]:
                choice = payload["choices"][0]
                if isinstance(choice, dict):
                    formatted_delta = self._format_message(unified_event.delta, preserve_source=True)
                    original_delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                    if "role" not in original_delta:
                        formatted_delta.pop("role", None)
                    choice["delta"] = formatted_delta
            return payload
        if unified_event.delta is not None:
            delta = _format_response_message(self._format_message(unified_event.delta, preserve_source=False), unified_event.delta)
            if unified_event.extra.get("finish_reason") is None:
                delta.pop("role", None)
            payload = {
                "id": unified_event.extra.get("id"),
                "object": "chat.completion.chunk",
                "model": unified_event.extra.get("model"),
                "choices": [{"index": 0, "delta": delta, "finish_reason": format_stop_reason(unified_event.stop_reason or unified_event.extra.get("finish_reason"), self.name)}],
                "usage": _format_openai_usage(unified_event.usage),
            }
            return f"data: {json.dumps({k: v for k, v in payload.items() if v is not None})}\n\n"
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
            provider_cost = cost_details.get("total_cost") or cost_details.get("request_cost_usd") or cost_details.get("cost") or cost_details.get("estimated_cost")
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
        role = str(payload.get("role") or "assistant")
        content = self._parse_content(payload.get("content"))
        if role == "tool":
            result_content = canonical_tool_arguments(payload.get("content"))
            content = [
                ContentBlock(
                    type="tool_result",
                    tool_result=ToolResult(
                        tool_call_id=payload.get("tool_call_id"),
                        name=payload.get("name"),
                        content=result_content,
                        raw=deepcopy(message),
                    ),
                    raw=deepcopy(message),
                )
            ]
        return UnifiedMessage(
            role=role,
            content=content,
            name=payload.get("name"),
            tool_call_id=payload.get("tool_call_id"),
            tool_calls=self._parse_message_tool_calls(payload),
            reasoning=reasoning,
            raw=deepcopy(message),
            extra={k: deepcopy(v) for k, v in payload.items() if k not in {"role", "content", "name", "tool_call_id", "tool_calls", "reasoning", "reasoning_content"}},
        )

    def _format_request_messages(self, messages: Iterable[UnifiedMessage], *, preserve_source: bool) -> list[dict[str, Any]]:
        """Format messages, expanding protocols that embed tool results in user turns."""

        formatted: list[dict[str, Any]] = []
        for message in messages:
            result_blocks = [block for block in message.content if block.tool_result]
            if result_blocks and not (message.role == "tool" and len(result_blocks) == 1):
                residual = [block for block in message.content if not block.tool_result]
                if residual:
                    residual_message = deepcopy(message)
                    residual_message.content = residual
                    residual_message.tool_call_id = None
                    formatted.append(self._format_message(residual_message, preserve_source=False))
                for block in result_blocks:
                    result = block.tool_result
                    if result is None:
                        continue
                    formatted.append(
                        {
                            "role": "tool",
                            "tool_call_id": result.tool_call_id,
                            "content": _tool_result_text(result.content),
                        }
                    )
                continue
            formatted.append(self._format_message(message, preserve_source=preserve_source))
        return formatted

    def _format_message(self, message: UnifiedMessage, *, preserve_source: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role}
        if message.name:
            payload["name"] = message.name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        result_blocks = [block.tool_result for block in message.content if block.tool_result]
        if message.role == "tool" and result_blocks:
            result = result_blocks[0]
            payload["tool_call_id"] = result.tool_call_id or message.tool_call_id
            if result.name:
                payload["name"] = result.name
            content = _tool_result_text({"error": result.content} if result.is_error else result.content)
        else:
            content = self._format_content(message.content, preserve_source=preserve_source)
        if content is not None:
            payload["content"] = content
        extra = deepcopy(message.extra) if preserve_source else {}
        legacy_function_call = extra.get("function_call")
        tool_calls = _message_tool_calls(message)
        if tool_calls and not legacy_function_call:
            payload["tool_calls"] = [self._format_tool_call(call, preserve_source=preserve_source) for call in tool_calls]
        elif tool_calls and legacy_function_call:
            call = tool_calls[0]
            extra["function_call"] = {"name": call.name or "", "arguments": tool_arguments_text(call.arguments)}
        if message.reasoning:
            # OpenAI-compatible providers use multiple names for reasoning text.
            # Prefer the common extension field while keeping all blocks in extra.
            text = "".join(block.text or "" for block in message.reasoning if block.text)
            if text:
                payload["reasoning_content"] = text
        payload.update(extra)
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
                raw_source = deepcopy(block.get("image_url") or block.get("source"))
                source = _openai_media_source(raw_source, kind="image")
                blocks.append(ContentBlock(type="image", source=source, raw=deepcopy(block), extra=_without(block, {"type", "image_url", "source"})))
            elif block_type in {"input_audio", "audio"}:
                raw_source = deepcopy(block.get("input_audio") or block.get("audio") or block.get("source"))
                source = _openai_media_source(raw_source, kind="audio")
                blocks.append(ContentBlock(type="audio", source=source, raw=deepcopy(block), extra=_without(block, {"type", "input_audio", "audio", "source"})))
            elif block_type in {"file", "input_file"}:
                source = _openai_media_source(block, kind="file")
                blocks.append(ContentBlock(type="file", source=source, raw=deepcopy(block), extra=_without(block, {"type", "file_id", "file_data", "filename"})))
            else:
                blocks.append(ContentBlock(type=str(block_type), raw=deepcopy(block), extra=_without(block, {"type"})))
        return blocks

    def _format_content(self, blocks: Iterable[ContentBlock], *, preserve_source: bool = True) -> Any:
        block_list = list(blocks)
        if not block_list:
            return None
        if all(block.type == "text" and (not preserve_source or not isinstance(block.raw, dict)) and (not preserve_source or not block.extra) for block in block_list):
            return first_text(block_list) or ""
        formatted = []
        for block in block_list:
            if block.type == "text":
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {"type": "text"}
                payload["type"] = "text"
                payload["text"] = block.text or ""
                if preserve_source:
                    payload.update(deepcopy(block.extra))
                formatted.append(payload)
            elif block.type == "image":
                payload = deepcopy(block.raw) if preserve_source and isinstance(block.raw, dict) else {"type": "image_url"}
                payload["type"] = "image_url"
                payload["image_url"] = _format_openai_image_source(block.source)
                if preserve_source:
                    payload.update(deepcopy(block.extra))
                formatted.append(payload)
            elif block.type == "audio":
                source = _media_source(block.source)
                payload = {"type": "input_audio", "input_audio": {"data": source.data or "", "format": _audio_format(source.media_type)}}
                formatted.append(payload)
            elif block.type in {"file", "document"}:
                source = _media_source(block.source)
                file_payload: dict[str, Any] = {"type": "file"}
                if source.file_id:
                    file_payload["file_id"] = source.file_id
                elif source.data:
                    file_payload["file_data"] = source.data
                elif source.url:
                    file_payload["file_url"] = source.url
                formatted.append(file_payload)
            elif preserve_source and isinstance(block.raw, dict):
                formatted.append(deepcopy(block.raw))
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

    def _format_tool_definition(self, tool: ToolDefinition, *, preserve_source: bool = True) -> dict[str, Any]:
        raw = tool.extra.get("raw")
        payload = deepcopy(raw) if preserve_source and isinstance(raw, dict) else {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": deepcopy(tool.input_schema),
            },
        }
        payload["type"] = "function"
        function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
        function.update({"name": tool.name, "description": tool.description, "parameters": deepcopy(tool.input_schema)})
        payload["function"] = {k: v for k, v in function.items() if v is not None}
        return payload

    def _parse_tool_call(self, call: dict[str, Any]) -> ToolCall:
        payload = dict(call or {})
        function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
        arguments: Any = canonical_tool_arguments(function.get("arguments"))
        return ToolCall(
            id=payload.get("id"),
            name=function.get("name") or payload.get("name"),
            arguments=arguments,
            type=str(payload.get("type") or "function"),
            index=payload.get("index"),
            raw=deepcopy(call),
            extra={**_without(function, {"name", "arguments"}), **_without(payload, {"id", "function", "type", "index", "name"})},
        )

    def _format_tool_call(self, call: ToolCall, *, preserve_source: bool = True) -> dict[str, Any]:
        payload = deepcopy(call.raw) if preserve_source and isinstance(call.raw, dict) else {}
        payload["type"] = "function"
        if call.id:
            payload["id"] = call.id
        if call.index is not None:
            payload["index"] = call.index
        function = deepcopy(payload.get("function")) if isinstance(payload.get("function"), dict) else {}
        function["name"] = call.name or ""
        function["arguments"] = tool_arguments_text(call.arguments)
        payload["function"] = function
        return payload

    def _format_generation_params(self, request: UnifiedRequest, *, preserve_source: bool) -> dict[str, Any]:
        """Format canonical controls into OpenAI Chat field names."""

        params = deepcopy(request.generation_params)
        if preserve_source:
            original = request.extensions.get(self.name, {}).get("generation_params")
            payload = deepcopy(original) if isinstance(original, dict) else {}
        else:
            payload = {}
        if "max_output_tokens" in params:
            payload["max_completion_tokens"] = params.pop("max_output_tokens")
        if "stop_sequences" in params:
            payload["stop"] = params.pop("stop_sequences")
        reasoning = params.pop("reasoning", None)
        if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
            payload["reasoning_effort"] = reasoning["effort"]
        if "structured_output" in params:
            payload["response_format"] = format_structured_output(params.pop("structured_output"), self.name)
        if "tool_choice" in params:
            payload["tool_choice"] = format_tool_choice(params.pop("tool_choice"), self.name)
        if "audio_output" in params:
            payload["audio"] = deepcopy(params.pop("audio_output"))
        supported = {
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "n",
            "parallel_tool_calls",
            "presence_penalty",
            "seed",
            "service_tier",
            "stream_options",
            "temperature",
            "top_logprobs",
            "top_p",
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


def _extract_reasoning(payload: dict[str, Any]) -> list[ReasoningBlock]:
    blocks = []
    for field_name in ("reasoning_content", "reasoning"):
        value = payload.get(field_name)
        if value:
            blocks.append(ReasoningBlock(type=field_name, text=str(value), extra={"source_field": field_name}))
    return blocks


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


def _format_response_message(payload: dict[str, Any], message: UnifiedMessage) -> dict[str, Any]:
    """Return Chat Completions response-message shape.

    Request messages may legitimately preserve content-part arrays. Assistant
    response messages from non-chat native protocols often arrive as text parts;
    Chat Completions clients expect the final message content to be a string in
    that common case.
    """

    if payload.get("content") is not None and message.content:
        if all(block.type in {"text", "input_text", "output_text"} and not block.extra for block in message.content):
            payload = dict(payload)
            payload["content"] = _first_response_text(message.content) or ""
    return payload


def _first_response_text(blocks: Iterable[ContentBlock]) -> Optional[str]:
    parts = [block.text for block in blocks if block.type in {"text", "input_text", "output_text"} and block.text]
    return "".join(parts) if parts else first_text(blocks)


def _parse_openai_generation_params(source: dict[str, Any]) -> dict[str, Any]:
    """Normalize OpenAI Chat controls into protocol-independent field names."""

    params = deepcopy(source)
    max_tokens = params.pop("max_completion_tokens", params.pop("max_tokens", None))
    if max_tokens is not None:
        params["max_output_tokens"] = max_tokens
    stop = params.pop("stop", None)
    if stop is not None:
        params["stop_sequences"] = [stop] if isinstance(stop, str) else deepcopy(stop)
    reasoning_effort = params.pop("reasoning_effort", None)
    if reasoning_effort is not None:
        params["reasoning"] = {"effort": reasoning_effort}
    return params


def _openai_media_source(value: Any, *, kind: str) -> MediaSource:
    """Parse OpenAI media fields into a canonical media source."""

    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value:
            prefix, data = value.split(",", 1)
            return MediaSource(kind="base64", media_type=prefix[5:].split(";", 1)[0], data=data, raw=value)
        return MediaSource(kind="url", url=value, raw=value)
    payload = value if isinstance(value, dict) else {}
    url = payload.get("url") or payload.get("file_url")
    data = payload.get("data") or payload.get("file_data")
    media_type = payload.get("media_type") or payload.get("mime_type") or payload.get("format")
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        prefix, encoded = url.split(",", 1)
        media_type = media_type or prefix[5:].split(";", 1)[0]
        data = encoded
        url = None
    file_id = payload.get("file_id")
    source_kind = "file" if file_id else "base64" if data else "url"
    return MediaSource(
        kind=source_kind,
        media_type=media_type,
        url=url,
        data=data,
        file_id=file_id,
        detail=payload.get("detail"),
        raw=deepcopy(value),
        extra=_without(payload, {"url", "file_url", "data", "file_data", "file_id", "media_type", "mime_type", "format", "detail"}),
    )


def _media_source(value: Any) -> MediaSource:
    """Coerce legacy dictionary media sources into the canonical type."""

    if isinstance(value, MediaSource):
        return value
    return _openai_media_source(value, kind="media")


def _format_openai_image_source(value: Any) -> dict[str, Any]:
    """Format a canonical image source for Chat Completions."""

    source = _media_source(value)
    if source.url:
        url = source.url
    elif source.data:
        media_type = source.media_type or "application/octet-stream"
        url = f"data:{media_type};base64,{source.data}"
    elif source.file_id:
        url = source.file_id
    else:
        url = ""
    payload: dict[str, Any] = {"url": url}
    if source.detail:
        payload["detail"] = source.detail
    return payload


def _audio_format(media_type: Optional[str]) -> str:
    """Return OpenAI's compact audio format label from a MIME type."""

    if not media_type:
        return "wav"
    return media_type.rsplit("/", 1)[-1].lower()


def _message_tool_calls(message: UnifiedMessage) -> list[ToolCall]:
    """Return de-duplicated calls from both canonical message representations."""

    calls = list(message.tool_calls)
    seen = {(call.id, call.name, tool_arguments_text(call.arguments)) for call in calls}
    for block in message.content:
        if not block.tool_call:
            continue
        key = (block.tool_call.id, block.tool_call.name, tool_arguments_text(block.tool_call.arguments))
        if key not in seen:
            calls.append(block.tool_call)
            seen.add(key)
    return calls


def _tool_result_text(value: Any) -> str:
    """Serialize a canonical tool result for Chat's tool message content."""

    if isinstance(value, str):
        return value
    return json.dumps(serialize_value(value), separators=(",", ":"))


def _without(payload: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {k: deepcopy(v) for k, v in payload.items() if k not in keys}
