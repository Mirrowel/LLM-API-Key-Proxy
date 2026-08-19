"""Stateful canonical conversion for generative streaming protocols.

Protocol adapters parse provider frames into :class:`UnifiedStreamEvent`.  This
module owns the inverse operation because destination protocols have different
lifecycle requirements: one canonical delta can expand into several SSE frames,
and terminal events must close every destination-owned content block exactly
once.  Operational concerns such as timeouts, retries, cancellation, and
heartbeats remain in ``client.streaming``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import uuid
from typing import Any, AsyncIterator

from .canonical import (
    STOP_REASON_UNKNOWN,
    format_stop_reason,
    ordered_message_blocks,
    tool_arguments_text,
)
from .types import (
    ContentBlock,
    ProtocolContext,
    ProtocolError,
    ToolCall,
    UnifiedMessage,
    UnifiedStreamEvent,
    Usage,
    serialize_value,
)


@dataclass
class StreamFormatState:
    """Destination-owned lifecycle state for one converted stream."""

    protocol: str
    response_id: str
    model: str
    started: bool = False
    terminal: bool = False
    completion_emitted: bool = False
    role_emitted: bool = False
    stop_reason: str | None = None
    usage: Usage | None = None
    next_index: int = 0
    open_blocks: dict[str, int] = field(default_factory=dict)
    block_order: list[str] = field(default_factory=list)
    item_ids: dict[str, str] = field(default_factory=dict)
    item_kinds: dict[str, str] = field(default_factory=dict)
    text_by_key: dict[str, str] = field(default_factory=dict)
    tool_arguments: dict[str, str] = field(default_factory=dict)
    tool_names: dict[str, str] = field(default_factory=dict)
    tool_ids: dict[str, str] = field(default_factory=dict)
    emitted_tools: set[str] = field(default_factory=set)


class ProtocolStreamConverter:
    """Convert raw source frames to one independently selected output protocol."""

    def __init__(
        self,
        source_protocol: Any,
        output_protocol: Any,
        context: ProtocolContext,
    ) -> None:
        self.source_protocol = source_protocol
        self.output_protocol = output_protocol
        self.context = context
        self.state = stream_format_state(context, output_protocol.name)

    def convert(self, raw_event: Any) -> list[Any]:
        """Parse and format one source frame, expanding destination lifecycle frames."""

        event = self.source_protocol.parse_stream_event(raw_event, self.context)
        return format_canonical_stream_event(
            event,
            self.output_protocol.name,
            self.context,
            state=self.state,
        )


async def convert_protocol_stream(
    stream: AsyncIterator[Any],
    *,
    source_protocol: Any,
    output_protocol: Any,
    context: ProtocolContext,
) -> AsyncIterator[Any]:
    """Convert a resilient source stream while preserving transport heartbeats."""

    converter = ProtocolStreamConverter(source_protocol, output_protocol, context)
    async for raw_event in stream:
        if isinstance(raw_event, str) and raw_event.lstrip().startswith(":"):
            yield raw_event
            continue
        for frame in converter.convert(raw_event):
            yield frame


def stream_format_state(
    context: ProtocolContext | None,
    protocol: str,
) -> StreamFormatState:
    """Return persistent formatter state stored on the stream protocol context."""

    metadata = context.metadata if context is not None else {}
    states = metadata.setdefault("_stream_format_states", {})
    state = states.get(protocol)
    if isinstance(state, StreamFormatState):
        return state
    model = str((context.model if context else None) or metadata.get("model") or "")
    request_id = str((context.request_id if context else None) or metadata.get("request_id") or uuid.uuid4().hex)
    prefix = {"openai_chat": "chatcmpl", "anthropic_messages": "msg", "responses": "resp"}.get(protocol, "stream")
    state = StreamFormatState(protocol=protocol, response_id=f"{prefix}_{request_id}", model=model)
    states[protocol] = state
    return state


def format_canonical_stream_event(
    event: UnifiedStreamEvent,
    target_protocol: str,
    context: ProtocolContext | None = None,
    *,
    state: StreamFormatState | None = None,
) -> list[Any]:
    """Format one canonical event into zero or more destination wire frames."""

    state = state or stream_format_state(context, target_protocol)
    if state.terminal:
        return []
    if event.usage is not None:
        state.usage = event.usage
    if event.stop_reason:
        state.stop_reason = event.stop_reason
    elif event.extra.get("stop_reason"):
        state.stop_reason = str(event.extra["stop_reason"])

    if target_protocol == "openai_chat":
        return _format_openai(event, state)
    if target_protocol == "anthropic_messages":
        return _format_anthropic(event, state)
    if target_protocol == "responses":
        return _format_responses(event, state)
    if target_protocol == "gemini":
        return _format_gemini(event, state)
    raise ProtocolError(
        f"Canonical streaming is not supported for {target_protocol}",
        protocol=target_protocol,
        pass_name="format_stream_event",
    )


def _format_openai(event: UnifiedStreamEvent, state: StreamFormatState) -> list[str]:
    if event.type == "error" or event.error is not None:
        state.terminal = True
        return [_data_frame({"error": _error_payload(event.error)}), "data: [DONE]\n\n"]

    frames: list[str] = []
    delta = _openai_delta(event.delta or event.message)
    if delta:
        if not state.role_emitted:
            delta.setdefault("role", "assistant")
            state.role_emitted = True
        frames.append(_data_frame(_openai_chunk(state, delta=delta, finish_reason=None, usage=event.usage)))

    terminal = _is_terminal(event)
    if event.stop_reason or event.extra.get("stop_reason"):
        reason = event.stop_reason or event.extra.get("stop_reason")
        state.stop_reason = str(reason)
        frames.append(_data_frame(_openai_chunk(state, delta={}, finish_reason=format_stop_reason(state.stop_reason, "openai_chat"), usage=event.usage)))
    if terminal:
        frames.append("data: [DONE]\n\n")
        state.terminal = True
    elif event.usage is not None and not delta:
        frames.append(_data_frame(_openai_chunk(state, delta={}, finish_reason=None, usage=event.usage)))
    return frames


def _format_anthropic(event: UnifiedStreamEvent, state: StreamFormatState) -> list[str]:
    if event.type == "error" or event.error is not None:
        state.terminal = True
        return [_event_frame("error", {"type": "error", "error": _error_payload(event.error)})]

    frames = _anthropic_start(state)
    for block in _event_blocks(event):
        key, block_type = _block_key(block, state)
        if key not in state.open_blocks:
            index = state.next_index
            state.next_index += 1
            state.open_blocks[key] = index
            state.block_order.append(key)
            frames.append(_event_frame("content_block_start", {
                "type": "content_block_start",
                "index": index,
                "content_block": _anthropic_block_start(block, key, state),
            }))
        index = state.open_blocks[key]
        delta = _anthropic_block_delta(block, key, state)
        if delta is not None:
            frames.append(_event_frame("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": delta,
            }))

    if _is_terminal(event):
        for key in state.block_order:
            frames.append(_event_frame("content_block_stop", {
                "type": "content_block_stop",
                "index": state.open_blocks[key],
            }))
        reason = format_stop_reason(state.stop_reason, "anthropic_messages")
        frames.append(_event_frame("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": None},
            "usage": _anthropic_usage(state.usage, output_only=True),
        }))
        frames.append(_event_frame("message_stop", {"type": "message_stop"}))
        state.terminal = True
    return frames


def _format_responses(event: UnifiedStreamEvent, state: StreamFormatState) -> list[str]:
    if event.type == "error" or event.error is not None:
        frames = _responses_start(state)
        error = _error_payload(event.error)
        frames.append(_event_frame("response.failed", {
            "type": "response.failed",
            "response": _responses_object(state, status="failed", error=error),
        }))
        frames.append("data: [DONE]\n\n")
        state.terminal = True
        return frames

    frames = _responses_start(state)
    for block in _event_blocks(event):
        key, kind = _block_key(block, state)
        if key not in state.item_ids:
            item_id = _responses_item_id(kind, state.next_index)
            state.next_index += 1
            state.item_ids[key] = item_id
            state.item_kinds[key] = kind
            frames.extend(_responses_item_start(block, key, item_id, state))
        frames.extend(_responses_item_delta(block, key, state.item_ids[key], state))

    if _is_terminal(event):
        status = format_stop_reason(state.stop_reason or STOP_REASON_UNKNOWN, "responses") or "incomplete"
        item_status = "completed" if status == "completed" else "incomplete"
        for key, item_id in state.item_ids.items():
            frames.extend(_responses_item_done(key, item_id, state, item_status=item_status))
        event_name = "response.failed" if status == "failed" else "response.completed" if status == "completed" else "response.incomplete"
        frames.append(_event_frame(event_name, {
            "type": event_name,
            "response": _responses_object(state, status=status),
        }))
        frames.append("data: [DONE]\n\n")
        state.terminal = True
    return frames


def _format_gemini(event: UnifiedStreamEvent, state: StreamFormatState) -> list[str]:
    if event.type == "error" or event.error is not None:
        state.terminal = True
        return [_data_frame({"error": _error_payload(event.error)})]

    parts: list[dict[str, Any]] = []
    for block in _event_blocks(event):
        if block.tool_call:
            key, _ = _block_key(block, state)
            call = block.tool_call
            state.tool_names[key] = call.name or state.tool_names.get(key, "")
            state.tool_ids[key] = call.id or state.tool_ids.get(key, "")
            fragment = tool_arguments_text(call.arguments)
            if key in state.emitted_tools:
                if fragment:
                    raise ProtocolError(
                        "Gemini tool-call arguments continued after a complete object was emitted",
                        protocol="gemini",
                        pass_name="format_stream_event",
                        payload={"tool_call": key},
                    )
                continue
            state.tool_arguments[key] = state.tool_arguments.get(key, "") + fragment
            if not state.tool_arguments[key]:
                continue
            try:
                arguments = json.loads(state.tool_arguments[key])
            except json.JSONDecodeError:
                continue
            parts.append(_gemini_function_call_part(key, arguments, state))
            state.emitted_tools.add(key)
        elif block.reasoning:
            parts.append({"text": block.reasoning.text or "", "thought": True})
        elif block.type == "text":
            parts.append({"text": block.text or ""})

    if _is_terminal(event):
        for key in state.tool_names:
            if key in state.emitted_tools:
                continue
            arguments_text = state.tool_arguments.get(key, "")
            if arguments_text and not _is_json(arguments_text):
                raise ProtocolError(
                    "Gemini cannot emit an incomplete streamed tool-call argument object",
                    protocol="gemini",
                    pass_name="format_stream_event",
                    payload={"tool_call": key},
                )
            arguments = json.loads(arguments_text) if arguments_text else {}
            parts.append(_gemini_function_call_part(key, arguments, state))
            state.emitted_tools.add(key)

    frames: list[str] = []
    finish_reason = None
    if event.stop_reason or event.extra.get("stop_reason"):
        state.stop_reason = str(event.stop_reason or event.extra.get("stop_reason"))
        finish_reason = format_stop_reason(state.stop_reason, "gemini")
    if parts or finish_reason or event.usage is not None:
        candidate: dict[str, Any] = {"index": 0}
        if parts:
            candidate["content"] = {"role": "model", "parts": parts}
        if finish_reason:
            candidate["finishReason"] = finish_reason
        payload: dict[str, Any] = {"candidates": [candidate]}
        if state.model:
            payload["modelVersion"] = state.model
        usage = _gemini_usage(event.usage)
        if usage:
            payload["usageMetadata"] = usage
        frames.append(_data_frame(payload))
        if finish_reason:
            state.completion_emitted = True
    if _is_terminal(event):
        state.terminal = True
    return frames


def _gemini_function_call_part(
    key: str,
    arguments: Any,
    state: StreamFormatState,
) -> dict[str, Any]:
    """Build one single-shot Gemini function call from buffered fragments."""

    function_call: dict[str, Any] = {
        "name": state.tool_names.get(key, ""),
        "args": arguments,
    }
    if state.tool_ids.get(key):
        function_call["id"] = state.tool_ids[key]
    return {"functionCall": function_call}


def _event_blocks(event: UnifiedStreamEvent) -> list[ContentBlock]:
    if _is_terminal(event) or event.type.endswith(".done"):
        # Provider terminal snapshots repeat content already delivered as deltas.
        # They remain available on the event for accounting/storage but are not
        # emitted as a second client-visible delta.
        return []
    message = event.delta or event.message
    if message is None:
        return []
    return ordered_message_blocks(message)


def _block_key(block: ContentBlock, state: StreamFormatState) -> tuple[str, str]:
    if block.tool_call:
        call = block.tool_call
        identity = call.index if call.index is not None else call.id or call.name or "default"
        return f"tool:{identity}", "tool"
    if block.reasoning:
        return "reasoning:0", "reasoning"
    return "text:0", "text"


def _openai_delta(message: UnifiedMessage | None) -> dict[str, Any]:
    if message is None:
        return {}
    delta: dict[str, Any] = {}
    text = "".join(block.text or "" for block in ordered_message_blocks(message) if block.type == "text" and not block.reasoning)
    reasoning = "".join(block.reasoning.text or "" for block in ordered_message_blocks(message) if block.reasoning)
    if text:
        delta["content"] = text
    if reasoning:
        delta["reasoning_content"] = reasoning
    calls = [block.tool_call for block in ordered_message_blocks(message) if block.tool_call]
    if calls:
        delta["tool_calls"] = [
            {
                "index": call.index if call.index is not None else index,
                "id": call.id,
                "type": call.type or "function",
                "function": {
                    "name": call.name,
                    "arguments": tool_arguments_text(call.arguments),
                },
            }
            for index, call in enumerate(calls)
        ]
        delta["tool_calls"] = [
            {key: value for key, value in call.items() if value is not None}
            for call in delta["tool_calls"]
        ]
    return delta


def _openai_chunk(
    state: StreamFormatState,
    *,
    delta: dict[str, Any],
    finish_reason: str | None,
    usage: Usage | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": state.response_id,
        "object": "chat.completion.chunk",
        "model": state.model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    formatted_usage = _openai_usage(usage)
    if formatted_usage:
        payload["usage"] = formatted_usage
    return payload


def _anthropic_start(state: StreamFormatState) -> list[str]:
    if state.started:
        return []
    state.started = True
    return [_event_frame("message_start", {
        "type": "message_start",
        "message": {
            "id": state.response_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": state.model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": _anthropic_usage(state.usage),
        },
    })]


def _anthropic_block_start(block: ContentBlock, key: str, state: StreamFormatState) -> dict[str, Any]:
    if block.tool_call:
        call = block.tool_call
        state.tool_names[key] = call.name or state.tool_names.get(key, "")
        state.tool_ids[key] = call.id or state.tool_ids.get(key, f"call_{state.open_blocks[key]}")
        return {"type": "tool_use", "id": state.tool_ids[key], "name": state.tool_names[key], "input": {}}
    if block.reasoning:
        return {"type": "thinking", "thinking": ""}
    return {"type": "text", "text": ""}


def _anthropic_block_delta(block: ContentBlock, key: str, state: StreamFormatState) -> dict[str, Any] | None:
    if block.tool_call:
        call = block.tool_call
        if call.name:
            state.tool_names[key] = call.name
        fragment = tool_arguments_text(call.arguments)
        state.tool_arguments[key] = state.tool_arguments.get(key, "") + fragment
        return {"type": "input_json_delta", "partial_json": fragment} if fragment else None
    if block.reasoning:
        text = block.reasoning.text or ""
        return {"type": "thinking_delta", "thinking": text} if text else None
    text = block.text or ""
    return {"type": "text_delta", "text": text} if text else None


def _responses_start(state: StreamFormatState) -> list[str]:
    if state.started:
        return []
    state.started = True
    return [_event_frame("response.created", {
        "type": "response.created",
        "response": _responses_object(state, status="in_progress"),
    })]


def _responses_item_id(kind: str, index: int) -> str:
    return f"{'fc' if kind == 'tool' else 'rs' if kind == 'reasoning' else 'msg'}_{index}"


def _responses_item_start(block: ContentBlock, key: str, item_id: str, state: StreamFormatState) -> list[str]:
    kind = state.item_kinds[key]
    if kind == "tool" and block.tool_call:
        call = block.tool_call
        state.tool_names[key] = call.name or ""
        state.tool_ids[key] = call.id or item_id
        item = {"id": item_id, "type": "function_call", "call_id": state.tool_ids[key], "name": state.tool_names[key], "arguments": "", "status": "in_progress"}
        return [_event_frame("response.output_item.added", {"type": "response.output_item.added", "output_index": state.next_index - 1, "item": item})]
    if kind == "reasoning":
        item = {"id": item_id, "type": "reasoning", "summary": [], "status": "in_progress"}
        return [
            _event_frame("response.output_item.added", {"type": "response.output_item.added", "output_index": state.next_index - 1, "item": item}),
            _event_frame("response.reasoning_summary_part.added", {"type": "response.reasoning_summary_part.added", "item_id": item_id, "output_index": state.next_index - 1, "summary_index": 0, "part": {"type": "summary_text", "text": ""}}),
        ]
    item = {"id": item_id, "type": "message", "role": "assistant", "content": [], "status": "in_progress"}
    return [
        _event_frame("response.output_item.added", {"type": "response.output_item.added", "output_index": state.next_index - 1, "item": item}),
        _event_frame("response.content_part.added", {"type": "response.content_part.added", "item_id": item_id, "output_index": state.next_index - 1, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
    ]


def _responses_item_delta(block: ContentBlock, key: str, item_id: str, state: StreamFormatState) -> list[str]:
    kind = state.item_kinds[key]
    output_index = list(state.item_ids).index(key)
    if kind == "tool" and block.tool_call:
        fragment = tool_arguments_text(block.tool_call.arguments)
        state.tool_arguments[key] = state.tool_arguments.get(key, "") + fragment
        return [_event_frame("response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta", "item_id": item_id, "output_index": output_index, "delta": fragment})] if fragment else []
    text = block.reasoning.text if block.reasoning else block.text
    if not text:
        return []
    state.text_by_key[key] = state.text_by_key.get(key, "") + text
    event_name = "response.reasoning_summary_text.delta" if kind == "reasoning" else "response.output_text.delta"
    payload = {"type": event_name, "item_id": item_id, "output_index": output_index, "delta": text}
    payload["summary_index" if kind == "reasoning" else "content_index"] = 0
    return [_event_frame(event_name, payload)]


def _responses_item_done(
    key: str,
    item_id: str,
    state: StreamFormatState,
    *,
    item_status: str,
) -> list[str]:
    kind = state.item_kinds[key]
    output_index = list(state.item_ids).index(key)
    if kind == "tool":
        arguments = state.tool_arguments.get(key, "")
        item = {"id": item_id, "type": "function_call", "call_id": state.tool_ids.get(key, item_id), "name": state.tool_names.get(key, ""), "arguments": arguments, "status": item_status}
        return [
            _event_frame("response.function_call_arguments.done", {"type": "response.function_call_arguments.done", "item_id": item_id, "output_index": output_index, "arguments": arguments}),
            _event_frame("response.output_item.done", {"type": "response.output_item.done", "output_index": output_index, "item": item}),
        ]
    text = state.text_by_key.get(key, "")
    if kind == "reasoning":
        item = {"id": item_id, "type": "reasoning", "summary": [{"type": "summary_text", "text": text}], "status": item_status}
        return [
            _event_frame("response.reasoning_summary_text.done", {"type": "response.reasoning_summary_text.done", "item_id": item_id, "output_index": output_index, "summary_index": 0, "text": text}),
            _event_frame("response.reasoning_summary_part.done", {"type": "response.reasoning_summary_part.done", "item_id": item_id, "output_index": output_index, "summary_index": 0, "part": item["summary"][0]}),
            _event_frame("response.output_item.done", {"type": "response.output_item.done", "output_index": output_index, "item": item}),
        ]
    part = {"type": "output_text", "text": text, "annotations": []}
    item = {"id": item_id, "type": "message", "role": "assistant", "content": [part], "status": item_status}
    return [
        _event_frame("response.output_text.done", {"type": "response.output_text.done", "item_id": item_id, "output_index": output_index, "content_index": 0, "text": text}),
        _event_frame("response.content_part.done", {"type": "response.content_part.done", "item_id": item_id, "output_index": output_index, "content_index": 0, "part": part}),
        _event_frame("response.output_item.done", {"type": "response.output_item.done", "output_index": output_index, "item": item}),
    ]


def _responses_object(state: StreamFormatState, *, status: str, error: Any = None) -> dict[str, Any]:
    output = []
    item_status = "completed" if status == "completed" else "in_progress" if status == "in_progress" else "incomplete"
    for key, item_id in state.item_ids.items():
        kind = state.item_kinds[key]
        if kind == "tool":
            output.append({"id": item_id, "type": "function_call", "call_id": state.tool_ids.get(key, item_id), "name": state.tool_names.get(key, ""), "arguments": state.tool_arguments.get(key, ""), "status": item_status})
        elif kind == "reasoning":
            output.append({"id": item_id, "type": "reasoning", "summary": [{"type": "summary_text", "text": state.text_by_key.get(key, "")}], "status": item_status})
        else:
            output.append({"id": item_id, "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": state.text_by_key.get(key, ""), "annotations": []}], "status": item_status})
    payload: dict[str, Any] = {"id": state.response_id, "object": "response", "status": status, "model": state.model, "output": output}
    usage = _responses_usage(state.usage)
    if usage:
        payload["usage"] = usage
    if error is not None:
        payload["error"] = error
    return payload


def _is_terminal(event: UnifiedStreamEvent) -> bool:
    return event.type in {
        "done",
        "message_stop",
        "response.completed",
        "response.failed",
        "response.incomplete",
        "completed",
    }


def _data_frame(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(serialize_value(payload), ensure_ascii=False)}\n\n"


def decode_sse_data(raw_event: Any) -> Any:
    """Decode one SSE frame while accepting optional event and comment lines."""

    if not isinstance(raw_event, str):
        return raw_event
    text = raw_event.strip()
    data_lines = [
        line[len("data:") :].strip()
        for line in text.splitlines()
        if line.strip().startswith("data:")
    ]
    if data_lines:
        text = "\n".join(data_lines).strip()
    if text == "[DONE]":
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw_event


def _event_frame(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(serialize_value(payload), ensure_ascii=False)}\n\n"


def _error_payload(error: Any) -> dict[str, Any]:
    if isinstance(error, dict):
        return serialize_value(error)
    return {"type": "server_error", "message": str(error or "Provider stream failed")}


def _openai_usage(usage: Usage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    payload: dict[str, Any] = {
        "prompt_tokens": usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }
    if usage.cache_read_tokens or usage.cache_write_tokens:
        payload["prompt_tokens_details"] = {"cached_tokens": usage.cache_read_tokens, "cache_creation_tokens": usage.cache_write_tokens}
    if usage.reasoning_tokens:
        payload["completion_tokens_details"] = {"reasoning_tokens": usage.reasoning_tokens}
    return payload


def _anthropic_usage(usage: Usage | None, *, output_only: bool = False) -> dict[str, int]:
    if usage is None:
        return {"output_tokens": 0} if output_only else {"input_tokens": 0, "output_tokens": 0}
    payload = {"output_tokens": usage.output_tokens}
    if not output_only:
        payload["input_tokens"] = usage.input_tokens
        if usage.cache_read_tokens:
            payload["cache_read_input_tokens"] = usage.cache_read_tokens
        if usage.cache_write_tokens:
            payload["cache_creation_input_tokens"] = usage.cache_write_tokens
    return payload


def _responses_usage(usage: Usage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    return {
        "input_tokens": usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens,
        "input_tokens_details": {"cached_tokens": usage.cache_read_tokens},
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
        "total_tokens": usage.total_tokens,
    }


def _gemini_usage(usage: Usage | None) -> dict[str, int] | None:
    if usage is None:
        return None
    return {
        "promptTokenCount": usage.input_tokens + usage.cache_read_tokens,
        "candidatesTokenCount": usage.output_tokens,
        "totalTokenCount": usage.total_tokens,
        "cachedContentTokenCount": usage.cache_read_tokens,
        "thoughtsTokenCount": usage.reasoning_tokens,
    }


def _is_json(value: str) -> bool:
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return False
    return True
