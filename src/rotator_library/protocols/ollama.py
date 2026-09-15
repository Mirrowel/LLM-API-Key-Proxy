# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Ollama-native chat, generate, and embeddings protocol adapter.

Wire reference: docs.ollama.com ``/api/chat``, ``/api/generate``, ``/api/embed``.
Streaming is newline-delimited JSON (one object per line, ``done:true`` on the
terminal object) — not SSE. Durations are nanoseconds; usage counts are
``prompt_eval_count`` / ``eval_count``. Assistant messages carry ``images``
(base64 strings), ``thinking``, and ``tool_calls`` whose ``function.arguments``
is a JSON OBJECT (never the OpenAI string spelling). Request ``think`` is a
bool or one of low/medium/high/max; ``options`` is the Modelfile vocabulary.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, ClassVar, Mapping, Optional

from .base import ProtocolAdapter
from .canonical import (
    STOP_REASON_ERROR,
    STOP_REASON_MAX_TOKENS,
    STOP_REASON_STOP,
    add_conversion_warning,
    canonical_tool_arguments,
    message_reasoning,
    message_tool_calls,
    message_tool_results,
    normalize_reasoning_controls,
    tool_result_text,
)
from .operation import OPERATION_EMBEDDINGS, OPERATION_OLLAMA_CHAT, OPERATION_OLLAMA_GENERATE, normalize_operation
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

# Request fields with a dedicated canonical home; everything else is preserved
# verbatim in ``extra``.
_OPTION_FIELDS = {"options", "format", "keep_alive", "template", "context", "raw", "suffix"}
_CORE_FIELDS = {"operation", "model", "messages", "prompt", "input", "stream", "system", "tools", "think", *_OPTION_FIELDS}

# Ollama option key <-> canonical control. Canonical homes follow the shared
# generation-params vocabulary; the remaining Modelfile-only keys stay nested
# under ``options`` verbatim (disclosed by the generic conversion warnings).
_OLLAMA_OPTION_TO_CANONICAL = {
    "num_predict": "max_output_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "stop": "stop_sequences",
}
_CANONICAL_TO_OLLAMA_OPTION = {value: key for key, value in _OLLAMA_OPTION_TO_CANONICAL.items()}

# The four native thinking levels Ollama documents (request ``think`` string).
_OLLAMA_THINK_LEVELS = ("low", "medium", "high", "max")

# Message members consumed into canonical blocks; never replayed via ``extra``.
_MESSAGE_HANDLED_KEYS = {"role", "content", "images", "thinking", "tool_calls", "tool_name", "tool_call_id"}

# Response envelope members represented on UnifiedResponse; the rest is extra.
# ``done``/``done_reason`` stay in ``extra`` so the native terminal spelling is
# preserved for same-protocol replay (including load/unload error reasons).
_RESPONSE_HANDLED_KEYS = {
    "model",
    "message",
    "response",
    "embeddings",
    "embedding",
    "prompt_eval_count",
    "eval_count",
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
}


class OllamaProtocol(ProtocolAdapter):
    """Adapter for Ollama `/api/chat`, `/api/generate`, and embeddings shapes."""

    name: ClassVar[str] = "ollama"
    aliases: ClassVar[tuple[str, ...]] = ("ollama_native",)
    supported_operations: ClassVar[tuple[str, ...]] = (OPERATION_OLLAMA_CHAT, OPERATION_OLLAMA_GENERATE, OPERATION_EMBEDDINGS)
    supported_transports: ClassVar[tuple[str, ...]] = ("http", "jsonl")

    def parse_request(self, raw_request: dict[str, Any], context: ProtocolContext | None = None) -> UnifiedRequest:
        request = dict(raw_request or {})
        operation = _ollama_operation(request, context)
        input_field = "input"
        if operation == OPERATION_EMBEDDINGS and "input" not in request and "prompt" in request:
            # Older Ollama embeddings endpoints use `prompt`; newer endpoints use
            # `input`. Preserve the original spelling so round-trips are lossless.
            input_field = "prompt"
        generation_params = _parse_options(request.get("options"))
        for key in ("format", "keep_alive", "template", "context", "raw", "suffix"):
            if key in request:
                generation_params[key] = deepcopy(request[key])
        if "think" in request:
            generation_params["reasoning"] = _reasoning_from_think(request["think"])
        return UnifiedRequest(
            operation=operation,
            model=str(request.get("model") or getattr(context, "model", None) or ""),
            messages=[_message_from_ollama(message) for message in request.get("messages") or []],
            system=[ContentBlock(type="text", text=str(request["system"]))] if "system" in request else [],
            tools=[_tool_from_ollama(tool) for tool in request.get("tools") or [] if isinstance(tool, dict)],
            stream=bool(request.get("stream", False)),
            input=deepcopy(request.get(input_field) if operation == OPERATION_EMBEDDINGS else request.get("prompt")),
            generation_params=generation_params,
            metadata={**({"embedding_input_field": input_field} if operation == OPERATION_EMBEDDINGS else {}), "has_stream": "stream" in request},
            raw=deepcopy(raw_request),
            extra={k: deepcopy(v) for k, v in request.items() if k not in _CORE_FIELDS},
        )

    def build_request(self, unified_request: UnifiedRequest, context: ProtocolContext | None = None, capabilities: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": unified_request.model}
        if unified_request.stream or unified_request.metadata.get("has_stream"):
            payload["stream"] = unified_request.stream
        if unified_request.operation == OPERATION_OLLAMA_CHAT:
            payload["messages"] = [_message_to_ollama(message) for message in unified_request.messages]
        elif unified_request.operation == OPERATION_EMBEDDINGS:
            input_field = str(unified_request.metadata.get("embedding_input_field") or "input")
            payload[input_field] = deepcopy(unified_request.input)
        else:
            payload["prompt"] = deepcopy(unified_request.input)
        if unified_request.system:
            payload["system"] = "".join(block.text or "" for block in unified_request.system if block.type == "text")
        if unified_request.tools:
            payload["tools"] = [_tool_to_ollama(tool) for tool in unified_request.tools]
        payload.update(_build_options(unified_request, context))
        payload.update(deepcopy(unified_request.extra))
        return payload

    def parse_response(self, raw_response: Any, context: ProtocolContext | None = None) -> UnifiedResponse:
        response = raw_response if isinstance(raw_response, dict) else {}
        output = []
        messages: list[UnifiedMessage] = []
        if isinstance(response.get("message"), dict):
            message = _message_from_ollama(response["message"])
            messages.append(message)
            output.append(message.to_dict())
        elif "response" in response:
            output.append(response.get("response"))
        top_thinking = response.get("thinking")
        if top_thinking and not messages:
            # /api/generate surfaces thinking as a top-level sibling.
            reasoning = ReasoningBlock(type="reasoning", text=str(top_thinking))
            messages.append(
                UnifiedMessage(
                    role="assistant",
                    content=[ContentBlock(type="reasoning", text=str(top_thinking), reasoning=reasoning)],
                    reasoning=[reasoning],
                )
            )
        operation = _ollama_operation(response, context)
        stop_reason = _stop_reason_from_done(response.get("done_reason"), response, done=response.get("done"))
        return UnifiedResponse(
            operation=operation,
            model=response.get("model") or getattr(context, "model", None),
            messages=messages,
            output=output,
            data=deepcopy(response.get("embeddings") or response.get("embedding") or []),
            stop_reason=stop_reason,
            usage=_ollama_usage(response),
            raw=deepcopy(raw_response),
            extra={k: deepcopy(v) for k, v in response.items() if k not in _RESPONSE_HANDLED_KEYS},
        )

    def format_response(self, unified_response: UnifiedResponse, context: ProtocolContext | None = None, capabilities: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Format a unified response back to an Ollama response shape.

        Ollama responses are often mutated by adapters after parsing. Do not
        return `raw` wholesale here; rebuild the public fields from unified state
        and then merge preserved extras/timing values.
        """

        payload = deepcopy(unified_response.extra)
        if unified_response.model:
            payload["model"] = unified_response.model
        operation = unified_response.operation or _ollama_operation(payload, context)
        if operation == OPERATION_OLLAMA_CHAT:
            if unified_response.messages:
                payload["message"] = _message_to_ollama(unified_response.messages[0])
        elif operation == OPERATION_EMBEDDINGS:
            raw = unified_response.raw if isinstance(unified_response.raw, dict) else {}
            key = "embedding" if "embedding" in raw and "embeddings" not in raw else "embeddings"
            payload[key] = deepcopy(unified_response.data)
        else:
            payload["response"] = _ollama_response_text(unified_response)
        if "done" not in payload:
            payload["done"] = True
        if "done_reason" not in payload:
            mapped = _done_reason_from_stop(unified_response.stop_reason)
            if mapped is not None:
                payload["done_reason"] = mapped
        if unified_response.usage and isinstance(unified_response.usage.raw, dict):
            for key, value in unified_response.usage.raw.items():
                if key.endswith("duration"):
                    payload[key] = deepcopy(value)
            if unified_response.usage.input_tokens:
                payload["prompt_eval_count"] = unified_response.usage.input_tokens
            if unified_response.usage.output_tokens:
                payload["eval_count"] = unified_response.usage.output_tokens
        return {k: v for k, v in payload.items() if v is not None}

    def parse_stream_event(self, raw_event: Any, context: ProtocolContext | None = None) -> UnifiedStreamEvent:
        """Parse one NDJSON frame.

        Ollama streams assistant tool calls incrementally — the same loss the
        OpenAI-compat surface suffers. Native parsing must not repeat it: partial
        ``tool_calls`` fragments are buffered on the stream context and only
        emitted as complete calls on ``done:true``.
        """

        data = _json_event(raw_event)
        if not isinstance(data, dict):
            return UnifiedStreamEvent(type="metadata", operation=OPERATION_OLLAMA_GENERATE, raw=deepcopy(raw_event), extra={"unparsed": True})
        is_done = bool(data.get("done"))
        delta = None
        if isinstance(data.get("message"), dict):
            message_payload = data["message"]
            delta = _message_from_ollama(message_payload)
            if is_done:
                if delta.tool_calls:
                    # The terminal snapshot already carries the complete calls;
                    # drop any buffered fragments without re-emitting them.
                    _finalize_tool_fragments(context, None)
                else:
                    _attach_completed_calls(delta, _finalize_tool_fragments(context, None))
            else:
                _accumulate_tool_fragments(context, message_payload.get("tool_calls"))
                delta.tool_calls = []
                delta.content = [block for block in delta.content if block.type != "tool_call"]
        elif data.get("response") is not None:
            delta = UnifiedMessage(role="assistant", content=[ContentBlock(type="text", text=str(data.get("response")))])
            thinking = data.get("thinking")
            if thinking:
                reasoning = ReasoningBlock(type="reasoning", text=str(thinking))
                delta.reasoning.append(reasoning)
                delta.content.append(ContentBlock(type="reasoning", text=str(thinking), reasoning=reasoning))
        event_type = "done" if is_done else "message_delta"
        return UnifiedStreamEvent(
            type=event_type,
            operation=_ollama_operation(data, context),
            delta=delta,
            stop_reason=_stop_reason_from_done(data.get("done_reason"), data, done=is_done),
            usage=_ollama_usage(data),
            raw=deepcopy(raw_event),
            extra=deepcopy(data),
        )


def _ollama_operation(request: dict[str, Any], context: ProtocolContext | None = None) -> str:
    explicit = normalize_operation(request.get("operation"))
    if explicit in {OPERATION_OLLAMA_CHAT, OPERATION_OLLAMA_GENERATE, OPERATION_EMBEDDINGS}:
        return explicit
    if context and isinstance(context.provider_options, dict):
        operation = normalize_operation(context.provider_options.get("operation"))
        if operation in {OPERATION_OLLAMA_CHAT, OPERATION_OLLAMA_GENERATE, OPERATION_EMBEDDINGS}:
            return operation
    if context and isinstance(context.metadata, dict):
        operation = normalize_operation(context.metadata.get("operation"))
        if operation in {OPERATION_OLLAMA_CHAT, OPERATION_OLLAMA_GENERATE, OPERATION_EMBEDDINGS}:
            return operation
    if "messages" in request or "message" in request:
        return OPERATION_OLLAMA_CHAT
    if "prompt" in request and request.get("endpoint") != "embeddings":
        return OPERATION_OLLAMA_GENERATE
    if "embeddings" in request or "embedding" in request or "input" in request or request.get("endpoint") == "embeddings":
        return OPERATION_EMBEDDINGS
    return OPERATION_OLLAMA_GENERATE


def _parse_options(raw_options: Any) -> dict[str, Any]:
    """Split Ollama ``options`` into canonical controls plus verbatim extras."""

    generation_params: dict[str, Any] = {}
    options = deepcopy(raw_options) if isinstance(raw_options, dict) else {}
    for option_key, canonical_key in _OLLAMA_OPTION_TO_CANONICAL.items():
        if option_key in options:
            generation_params[canonical_key] = options.pop(option_key)
    if options:
        # Modelfile-only keys (num_ctx, repeat_penalty, ...) have no canonical
        # home: they ride verbatim so same-protocol round-trips are lossless.
        generation_params["options"] = options
    return generation_params


def _build_options(unified_request: UnifiedRequest, context: ProtocolContext | None = None) -> dict[str, Any]:
    """Fold canonical controls back into the native ``options`` object."""

    params = deepcopy(unified_request.generation_params)
    options: dict[str, Any] = {}
    existing = params.pop("options", None)
    if isinstance(existing, dict):
        options.update(existing)
    for canonical_key, option_key in _CANONICAL_TO_OLLAMA_OPTION.items():
        if canonical_key in params:
            options[option_key] = params.pop(canonical_key)
    reasoning = params.pop("reasoning", None)
    if reasoning is not None:
        think = _think_from_reasoning(reasoning, unified_request)
        if think is not None:
            params["think"] = think
    if options:
        params["options"] = options
    return params


def _reasoning_from_think(value: Any) -> dict[str, Any]:
    """Map an Ollama ``think`` control onto canonical reasoning state."""

    if isinstance(value, bool):
        return {"enabled": value}
    if isinstance(value, str):
        level = value.strip().lower()
        if level in _OLLAMA_THINK_LEVELS:
            return {"enabled": True, "effort": level}
        if level in {"true", "on", "yes"}:
            return {"enabled": True}
        if level in {"false", "off", "no"}:
            return {"enabled": False}
    return {"enabled": bool(value)}


def _think_from_reasoning(reasoning: Any, request: UnifiedRequest) -> Any:
    """Map canonical reasoning controls onto Ollama's ``think`` field."""

    normalized = normalize_reasoning_controls(reasoning)
    effort = normalized.get("effort")
    enabled = normalized.get("enabled")
    if enabled is False or effort == "none":
        return False
    if effort in _OLLAMA_THINK_LEVELS:
        return effort
    if effort == "minimal":
        add_conversion_warning(
            request,
            code="reasoning_effort_approximated",
            message="reasoning effort 'minimal' has no Ollama think level; approximated to 'low'",
            field="reasoning.effort",
            target_protocol="ollama",
        )
        return "low"
    if effort == "xhigh":
        add_conversion_warning(
            request,
            code="reasoning_effort_approximated",
            message="reasoning effort 'xhigh' has no Ollama think level; approximated to 'max'",
            field="reasoning.effort",
            target_protocol="ollama",
        )
        return "max"
    if effort is not None:
        add_conversion_warning(
            request,
            code="reasoning_effort_unknown",
            message=f"reasoning effort '{effort}' is not an Ollama think level; reasoning enabled with the default level",
            field="reasoning.effort",
            target_protocol="ollama",
        )
        return True
    if enabled is True or normalized:
        return True
    return None


def _stop_reason_from_done(value: Any, data: dict[str, Any], *, done: bool = True) -> str | None:
    """Map Ollama ``done_reason`` onto the canonical completion vocabulary.

    Only the two documented clean reasons map exactly; every other value
    (load/unload included) is a provider-level failure with the native spelling
    preserved in ``extra`` (the caller stores the raw frame).
    """

    if value is None:
        return STOP_REASON_STOP if done and data.get("done") else None
    normalized = str(value).strip().lower()
    if normalized == "stop":
        return STOP_REASON_STOP
    if normalized == "length":
        return STOP_REASON_MAX_TOKENS
    return STOP_REASON_ERROR


def _done_reason_from_stop(stop_reason: Any) -> str | None:
    if stop_reason == STOP_REASON_STOP:
        return "stop"
    if stop_reason == STOP_REASON_MAX_TOKENS:
        return "length"
    return None


def _message_from_ollama(message: dict[str, Any]) -> UnifiedMessage:
    role = str(message.get("role") or "assistant")
    blocks: list[ContentBlock] = []
    reasoning_blocks: list[ReasoningBlock] = []
    tool_calls: list[ToolCall] = []

    for image in message.get("images") or []:
        source = _media_source_from_image(image)
        if source is not None:
            blocks.append(ContentBlock(type="image", source=source, raw=deepcopy(image)))

    if message.get("content") is not None:
        blocks.append(ContentBlock(type="text", text=str(message.get("content") or "")))

    thinking = message.get("thinking")
    if thinking:
        reasoning = ReasoningBlock(type="reasoning", text=str(thinking))
        reasoning_blocks.append(reasoning)
        blocks.append(ContentBlock(type="reasoning", text=str(thinking), reasoning=reasoning))

    for index, raw_call in enumerate(message.get("tool_calls") or []):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
        arguments = canonical_tool_arguments(function.get("arguments"))
        # Ollama tool calls carry no id: mint a synthetic correlation id so the
        # canonical object satisfies cross-protocol validation; it is never
        # emitted back onto the Ollama wire.
        call = ToolCall(
            id=f"call_{index}",
            name=function.get("name"),
            arguments=arguments,
            type="function",
            index=index,
            raw=deepcopy(raw_call),
            extra={"synthetic_id": True},
        )
        tool_calls.append(call)
        blocks.append(ContentBlock(type="tool_call", tool_call=call, raw=deepcopy(raw_call)))

    if role == "tool":
        result = ToolResult(
            tool_call_id=message.get("tool_call_id"),
            name=message.get("tool_name"),
            content=message.get("content"),
        )
        # The tool result IS the content: never duplicate it as a plain text
        # block (Chat/Anthropic normalize results to a dedicated block family).
        blocks = [ContentBlock(type="tool_result", tool_result=result, raw=deepcopy(message))]

    return UnifiedMessage(
        role=role,
        content=blocks,
        tool_calls=tool_calls,
        reasoning=reasoning_blocks,
        tool_call_id=message.get("tool_call_id"),
        raw=deepcopy(message),
        extra={k: deepcopy(v) for k, v in message.items() if k not in _MESSAGE_HANDLED_KEYS},
    )


def _message_to_ollama(message: UnifiedMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role}
    images: list[str] = []
    content = ""
    for block in message.content:
        if block.type == "text" and block.text is not None:
            content += str(block.text)
        elif block.type == "image" and block.source is not None:
            data = getattr(block.source, "data", None)
            url = getattr(block.source, "url", None)
            if data:
                images.append(str(data))
            elif url:
                images.append(str(url))
    if message.role == "tool":
        results = message_tool_results(message)
        if results:
            content = tool_result_text(results[0].content)
            name = results[0].name or message.name
            if name:
                payload["tool_name"] = name
    payload["content"] = content
    if images:
        payload["images"] = images
    reasoning = message_reasoning(message)
    if reasoning and reasoning[0].text:
        payload["thinking"] = reasoning[0].text
    calls = message_tool_calls(message)
    if calls:
        payload["tool_calls"] = [
            {"function": {"name": call.name, "arguments": canonical_tool_arguments(call.arguments)}}
            for call in calls
        ]
    payload.update(deepcopy(message.extra))
    return payload


def _media_source_from_image(image: Any) -> MediaSource | None:
    if isinstance(image, str):
        return MediaSource(kind="data", data=image)
    if isinstance(image, dict):
        data = image.get("data") or image.get("base64")
        url = image.get("url")
        if data or url:
            return MediaSource(
                kind="url" if url and not data else "data",
                url=url,
                data=data,
                media_type=image.get("media_type"),
            )
    return None


def _tool_from_ollama(tool: dict[str, Any]) -> ToolDefinition:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    schema = function.get("parameters")
    return ToolDefinition(
        name=str(function.get("name") or ""),
        description=function.get("description"),
        input_schema=deepcopy(schema) if isinstance(schema, dict) else {},
        type=str(tool.get("type") or "function"),
        extra={k: deepcopy(v) for k, v in tool.items() if k not in {"type", "function"}},
    )


def _tool_to_ollama(tool: ToolDefinition) -> dict[str, Any]:
    function: dict[str, Any] = {"name": tool.name}
    if tool.description is not None:
        function["description"] = tool.description
    if tool.input_schema:
        function["parameters"] = deepcopy(tool.input_schema)
    payload: dict[str, Any] = {"type": tool.type or "function", "function": function}
    payload.update(deepcopy(tool.extra))
    return payload


def _accumulate_tool_fragments(context: ProtocolContext | None, raw_calls: Any) -> None:
    if not raw_calls or context is None or not isinstance(context.metadata, dict):
        return
    buffer = context.metadata.setdefault("_ollama_stream_tool_calls", {})
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function") if isinstance(raw_call.get("function"), dict) else {}
        entry = buffer.setdefault(index, {"name": "", "parts": [], "object": None})
        if function.get("name"):
            entry["name"] = function["name"]
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            entry["parts"].append(arguments)
        elif arguments is not None:
            entry["object"] = deepcopy(arguments)


def _finalize_tool_fragments(context: ProtocolContext | None, raw_calls: Any) -> list[ToolCall]:
    _accumulate_tool_fragments(context, raw_calls)
    if context is None or not isinstance(context.metadata, dict):
        return []
    buffer = context.metadata.pop("_ollama_stream_tool_calls", {}) or {}
    calls: list[ToolCall] = []
    for index in sorted(buffer):
        entry = buffer[index]
        arguments: Any = entry.get("object")
        if arguments is None:
            text = "".join(entry.get("parts") or [])
            arguments = canonical_tool_arguments(text) if text else {}
        calls.append(
            ToolCall(
                id=f"call_{index}",
                name=entry.get("name") or "",
                arguments=arguments,
                type="function",
                index=index,
                extra={"synthetic_id": True},
            )
        )
    return calls


def _attach_completed_calls(message: UnifiedMessage, calls: list[ToolCall]) -> None:
    if not calls:
        return
    existing = {(call.name, json.dumps(call.arguments, sort_keys=True, default=str)) for call in message.tool_calls}
    for call in calls:
        key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
        if key in existing:
            continue
        message.tool_calls.append(call)
        message.content.append(ContentBlock(type="tool_call", tool_call=call))


def _ollama_response_text(response: UnifiedResponse) -> str:
    if response.output:
        return "".join(str(item) for item in response.output if item is not None)
    if response.messages:
        blocks = [block for block in response.messages[0].content if block.type == "text"]
        joined = "".join(block.text or "" for block in blocks)
        if joined:
            return joined
    return ""


def _json_event(raw_event: Any) -> Any:
    if isinstance(raw_event, dict):
        return raw_event
    if not isinstance(raw_event, str):
        return None
    text = raw_event.strip()
    if text.startswith("data:"):
        text = text[5:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _ollama_usage(response: dict[str, Any]) -> Usage | None:
    prompt_tokens = int(response.get("prompt_eval_count") or 0)
    output_tokens = int(response.get("eval_count") or 0)
    cached_tokens = int(
        response.get("prompt_eval_count_cached")
        or response.get("cache_read_count")
        or response.get("cached_tokens")
        or 0
    )
    if not prompt_tokens and not output_tokens and not cached_tokens:
        return None
    return Usage(
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cached_tokens,
        raw={k: deepcopy(v) for k, v in response.items() if k.endswith("count") or k.endswith("duration")},
    )
