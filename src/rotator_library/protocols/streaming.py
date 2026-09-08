"""Stateful canonical conversion for generative streaming protocols.

Protocol adapters parse provider frames into :class:`UnifiedStreamEvent`.  This
module owns the inverse operation because destination protocols have different
lifecycle requirements: one canonical delta can expand into several SSE frames,
and terminal events must close every destination-owned content block exactly
once.  Operational concerns such as timeouts, retries, cancellation, and
heartbeats remain in ``client.streaming``.
"""

from __future__ import annotations

from copy import deepcopy

from dataclasses import dataclass, field
import json
import time
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
    ConversionWarning,
    ProtocolContext,
    ProtocolError,
    ToolCall,
    UnifiedMessage,
    UnifiedStreamEvent,
    Usage,
    serialize_value,
)

# Anthropic-native server-tool content-block types (mirrors the anthropic
# adapter's whitelist; streaming emits these verbatim for anthropic clients).
_ANTHROPIC_SERVER_TOOL_BLOCK_TYPES = {
    "server_tool_use",
    # Current server-tool catalog result families; unknown *_tool_result /
    # *_tool_result_error shapes pass through as raw-union blocks (the
    # suffix grammar is stable across versioned tool names).
    "web_search_tool_result",
    "web_fetch_tool_result",
    "code_execution_tool_result",
    "text_editor_tool_result",
    "bash_tool_result",
    "bash_code_execution_tool_result",
    "text_editor_code_execution_tool_result",
    "memory_tool_result",
    "tool_search_tool_result",
    "advisor_tool_result",
    "mcp_toolset_tool_result",
}


def _is_anthropic_server_tool_block(raw_type: str) -> bool:
    """Known families + versioned spellings of the same stems (see the
    adapter's _is_server_tool_result_type — a user's own *_tool_result
    block never fabricates a server builtin)."""

    from .anthropic_messages import _is_server_tool_result_type

    return _is_server_tool_result_type(raw_type)


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
    finish_emitted: bool = False
    # Stream-side conversion drops are SILENT on the wire by construction
    # (in-band frames have no summary header): they accumulate here for the
    # pipeline tail to trace/log — never lost, never in-band.
    warnings: list["ConversionWarning"] = field(default_factory=list)
    created: int = field(default_factory=lambda: int(time.time()))
    finished_choices: set[int] = field(default_factory=set)
    block_signatures: dict[str, str] = field(default_factory=dict)
    emitted_signatures: set[str] = field(default_factory=set)
    reasoning_encrypted: dict[str, str] = field(default_factory=dict)
    sequence: int = 0
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: Usage | None = None
    next_index: int = 0
    open_blocks: dict[str, int] = field(default_factory=dict)
    block_order: list[str] = field(default_factory=list)
    item_ids: dict[str, str] = field(default_factory=dict)
    item_kinds: dict[str, str] = field(default_factory=dict)
    builtin_items: dict[str, dict] = field(default_factory=dict)
    refusal_by_key: dict[str, bool] = field(default_factory=dict)
    text_by_key: dict[str, str] = field(default_factory=dict)
    tool_arguments: dict[str, str] = field(default_factory=dict)
    tool_names: dict[str, str] = field(default_factory=dict)
    tool_ids: dict[str, str] = field(default_factory=dict)
    tool_signatures: dict[str, str] = field(default_factory=dict)
    source_protocol: str | None = None
    emitted_tools: set[str] = field(default_factory=set)
    # Block-identity bookkeeping (defect 8): events that carry explicit
    # content/output indexes use them directly; identity-less wires (chat
    # chunks) mint a family epoch that reopens when a different block family
    # intervenes, so `text -> tool -> text` stays three distinct blocks.
    family_epoch: dict[str, int] = field(default_factory=dict)
    last_family: str | None = None


class ProtocolStreamConverter:
    """Convert raw source frames to the client's own protocol."""

    def __init__(
        self,
        source_protocol: Any,
        client_protocol: Any,
        context: ProtocolContext,
    ) -> None:
        self.source_protocol = source_protocol
        self.client_protocol = client_protocol
        self.context = context
        self.state = stream_format_state(context, client_protocol.name)

    def convert(self, raw_event: Any) -> list[Any]:
        """Parse and format one source frame, expanding destination lifecycle frames."""

        events = self.source_protocol.parse_stream_events(raw_event, self.context)
        frames: list[Any] = []
        for event in events:
            frames.extend(
                format_canonical_stream_event(
                    event,
                    self.client_protocol.name,
                    self.context,
                    state=self.state,
                )
            )
        return frames


async def convert_protocol_stream(
    stream: AsyncIterator[Any],
    *,
    source_protocol: Any,
    client_protocol: Any,
    context: ProtocolContext,
) -> AsyncIterator[Any]:
    """Convert a resilient source stream while preserving transport heartbeats."""

    converter = ProtocolStreamConverter(source_protocol, client_protocol, context)
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
    if state.source_protocol is None and getattr(event, "source_protocol", None):
        # Same-protocol signature replay gating (D8): opaque provider state
        # streams only back to its own protocol.
        state.source_protocol = event.source_protocol
    if event.usage is not None:
        # Cumulative stream usage semantics (Anthropic message_delta and
        # friends): later events refine, they never erase earlier facts —
        # input_tokens arrives at message_start, output accumulates to the
        # terminal event. Field-wise merge, event wins when non-zero.
        if state.usage is None:
            state.usage = event.usage
        else:
            merged_input = event.usage.input_tokens or state.usage.input_tokens
            merged_output = event.usage.output_tokens or state.usage.output_tokens
            state.usage = Usage(
                input_tokens=merged_input,
                output_tokens=merged_output,
                # Cumulative deltas legitimately recompute the total: an
                # output-only delta's post_init total (0+42) must never
                # clobber the message_start total.
                total_tokens=max(event.usage.total_tokens or 0, state.usage.total_tokens or 0, merged_input + merged_output),
                cache_read_tokens=event.usage.cache_read_tokens or state.usage.cache_read_tokens,
                cache_write_tokens=event.usage.cache_write_tokens or state.usage.cache_write_tokens,
                reasoning_tokens=event.usage.reasoning_tokens or state.usage.reasoning_tokens,
                audio_tokens=event.usage.audio_tokens or state.usage.audio_tokens,
                output_audio_tokens=event.usage.output_audio_tokens or state.usage.output_audio_tokens,
                accepted_prediction_tokens=event.usage.accepted_prediction_tokens or state.usage.accepted_prediction_tokens,
                rejected_prediction_tokens=event.usage.rejected_prediction_tokens or state.usage.rejected_prediction_tokens,
                cost=event.usage.cost or state.usage.cost,
                raw=event.usage.raw,
                extra=event.usage.extra or state.usage.extra,
            )
    if event.stop_reason:
        state.stop_reason = event.stop_reason
    elif event.extra.get("stop_reason"):
        state.stop_reason = str(event.extra["stop_reason"])
    if event.extra.get("stop_sequence"):
        state.stop_sequence = str(event.extra["stop_sequence"])

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


def _stream_warn(state: "StreamFormatState", code: str, message: str, field: str) -> None:
    """Record a stream-side conversion drop (traced at the pipeline tail)."""

    if state.warnings is None:
        state.warnings = []
    if not any(w.code == code and w.message == message for w in state.warnings):
        state.warnings.append(
            ConversionWarning(
                code=code,
                message=message,
                field=field,
                source_protocol=state.source_protocol,
                target_protocol=state.protocol,
            )
        )


def _format_openai(event: UnifiedStreamEvent, state: StreamFormatState) -> list[str]:
    if event.type == "error" or event.error is not None:
        state.terminal = True
        return [_data_frame({"error": _error_payload(event.error)}), "data: [DONE]\n\n"]

    frames: list[str] = []
    delta = _openai_delta(event.delta or event.message)
    choice_index = event.output_index or 0
    choice_logprobs = event.extra.get("logprobs")
    if delta:
        if not state.role_emitted:
            delta.setdefault("role", "assistant")
            state.role_emitted = True
        frames.append(_data_frame(_openai_chunk(state, delta=delta, finish_reason=None, choice_index=choice_index, logprobs=choice_logprobs)))

    reason = event.stop_reason or event.extra.get("stop_reason")
    if reason and choice_index not in state.finished_choices:
        state.stop_reason = str(reason)
        frames.append(_data_frame(_openai_chunk(
            state,
            delta={},
            finish_reason=format_stop_reason(state.stop_reason, "openai_chat"),
            choice_index=choice_index,
            logprobs=choice_logprobs,
        )))
        state.finished_choices.add(choice_index)
    if _is_terminal(event):
        if state.usage is not None:
            frames.append(_data_frame(_openai_chunk(state, delta=None, finish_reason=None, usage=state.usage, empty_choices=True)))
        frames.append("data: [DONE]\n\n")
        state.terminal = True
    return frames


def _format_anthropic(event: UnifiedStreamEvent, state: StreamFormatState) -> list[str]:
    if event.type == "error" or event.error is not None:
        state.terminal = True
        return [_event_frame("error", {"type": "error", "error": _error_payload(event.error)})]
    if event.type == "heartbeat" or event.native_type == "ping":
        # Keep-alives never open message lifecycle frames.
        return []

    frames = _anthropic_start(state)
    visible = _client_visible_blocks(event, include_builtins=True)
    for block in visible:
        if block.type == "citations_delta":
            # Citations attach to the OPEN text block — no new lifecycle.
            open_indexes = [state.open_blocks[k] for k in state.block_order if k in state.open_blocks and k.startswith("text:")]
            if open_indexes and block.annotations:
                annotation = block.annotations[0]
                frames.append(_event_frame("content_block_delta", {
                    "type": "content_block_delta",
                    "index": open_indexes[-1],
                    "delta": {"type": "citations_delta", "citation": deepcopy(annotation.raw) if isinstance(annotation.raw, dict) else {"type": "citation", "cited_text": annotation.citation, "url": annotation.url}},
                }))
            continue
        key, block_type = _block_key(block, state, event)
        if key not in state.open_blocks:
            # Documented grammar: each open block closes before the next
            # opens (index-keyed accumulation tolerates otherwise, but the
            # wire order stays spec-conformant).
            if state.open_blocks:
                for open_key in list(state.open_blocks):
                    if open_key == key:
                        continue
                    frames.extend(_anthropic_close_block(open_key, state))
            if block.type == "builtin_tool":
                raw_type = str(block.raw.get("type") or "") if isinstance(block.raw, dict) else ""
                if _is_anthropic_server_tool_block(raw_type):
                    # Anthropic-native server-tool blocks arrive complete:
                    # the full payload rides content_block_start, then closes
                    # (no delta phase exists for them).
                    index = state.next_index
                    state.next_index += 1
                    state.block_order.append(f"builtin:{index}")
                    frames.append(_event_frame("content_block_start", {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": deepcopy(block.raw),
                    }))
                    frames.append(_event_frame("content_block_stop", {
                        "type": "content_block_stop",
                        "index": index,
                    }))
                else:
                    # Foreign-shaped builtin records stay omitted entirely
                    # (no legal anthropic wire shape, no fabricated empty
                    # blocks) — disclosed, never silent.
                    _stream_warn(
                        state,
                        "builtin_tool_output_dropped",
                        f"builtin tool record ({getattr(getattr(block, 'builtin_tool', None), 'kind', '') or 'unknown'}) has no Anthropic stream shape; omitted",
                        "content",
                    )
                continue
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
        signature = getattr(block.reasoning, "signature", None) if block.reasoning else None
        if signature:
            state.block_signatures[key] = signature
        delta = _anthropic_block_delta(block, key, state)
        if delta is not None:
            frames.append(_event_frame("content_block_delta", {
                "type": "content_block_delta",
                "index": index,
                "delta": delta,
            }))

    if _is_terminal(event):
        for key in list(state.open_blocks):
            frames.extend(_anthropic_close_block(key, state))
        # A concrete stop_reason only when one was actually observed (the
        # completion-evidence contract: bare EOF never fabricates a reason);
        # unmappable reasons degrade to end_turn upstream in format_response
        # where a warning can be recorded.
        reason = format_stop_reason(state.stop_reason, "anthropic_messages")
        terminal_stop_sequence = event.extra.get("stop_sequence") or state.stop_sequence
        if reason == "end_turn" and terminal_stop_sequence:
            # The matched sequence distinguishes stop_sequence from end_turn
            # (mirrors the non-stream path).
            reason = "stop_sequence"
        frames.append(_event_frame("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": terminal_stop_sequence},
            "usage": _anthropic_usage(state.usage, output_only=True),
        }))
        frames.append(_event_frame("message_stop", {"type": "message_stop"}))
        state.terminal = True
    return frames


def _anthropic_close_block(key: str, state: StreamFormatState) -> list[str]:
    """Close one open block, flushing an unemitted thinking signature first
    (signature_delta must precede content_block_stop)."""

    frames: list[str] = []
    index = state.open_blocks.get(key)
    if index is None:
        return frames
    signature = state.block_signatures.get(key)
    if (
        signature
        and signature != "__redacted__"
        and key not in state.emitted_signatures
        and state.source_protocol == "anthropic_messages"
    ):
        # signature_delta flushes before content_block_stop — same-protocol
        # clients only (foreign-source signatures stay cache-owned, D8).
        # The "__redacted__" sentinel is an internal marker for
        # redacted_thinking blocks, NEVER a signature value: emitting it
        # would fabricate a cryptographic claim clients replay upstream.
        frames.append(_event_frame("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "signature_delta", "signature": signature},
        }))
        state.emitted_signatures.add(key)
    frames.append(_event_frame("content_block_stop", {
        "type": "content_block_stop",
        "index": index,
    }))
    state.open_blocks.pop(key, None)
    return frames


def _format_responses(event: UnifiedStreamEvent, state: StreamFormatState) -> list[str]:
    if event.type == "error" or event.error is not None:
        frames = _responses_start(state)
        error = _error_payload(event.error)
        # Open items close before the failure terminal (grammar parity with
        # the completed path).
        for key, item_id in state.item_ids.items():
            frames.extend(_responses_item_done(key, item_id, state, item_status="incomplete"))
        frames.append(_responses_frame("response.failed", {
            "type": "response.failed",
            "response": _responses_object(state, status="failed", error=error),
        }, state))
        frames.append("data: [DONE]\n\n")
        state.terminal = True
        return frames

    frames = _responses_start(state)
    # Done/terminal snapshots are the only carrier of bound opaque state
    # (encrypted reasoning) on real upstreams — harvest before the delta
    # skip so synthesized frames carry it (D8).
    _harvest_done_state(event, state)
    for block in _event_blocks(event):
        if block.type == "builtin_tool" and block.builtin_tool is not None:
            # Provider-executed tool records stream as native output items.
            # Responses-native raw shapes round-trip verbatim; foreign raw
            # shapes (e.g. Anthropic server_tool_use) synthesize the
            # equivalent native call item — never leak foreign item types.
            # Registered in item_ids so downstream index derivation and the
            # terminal response object include them (no index collisions,
            # no identity collapse with interleaved text items).
            builtin = block.builtin_tool
            raw_item = builtin.raw if isinstance(builtin.raw, dict) and str(builtin.raw.get("type", "")).endswith("_call") else None
            item = raw_item or {
                "type": f"{str(builtin.kind).replace('-', '_')}_call",
                "id": builtin.call_id or _responses_item_id("builtin", state.next_index),
                "status": builtin.status or "completed",
            }
            builtin_key = f"builtin:{item.get('id') or state.next_index}:{state.next_index}"
            state.item_ids[builtin_key] = item.get("id") or _responses_item_id("builtin", state.next_index)
            state.item_kinds[builtin_key] = "builtin"
            state.builtin_items[builtin_key] = deepcopy(item)
            state.last_family = "builtin"
            frames.append(_responses_frame("response.output_item.added", {"type": "response.output_item.added", "output_index": state.next_index, "item": deepcopy(item)}, state))
            frames.append(_responses_frame("response.output_item.done", {"type": "response.output_item.done", "output_index": state.next_index, "item": deepcopy(item)}, state))
            state.next_index += 1
            continue
        key, kind = _block_key(block, state, event)
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
        frames.append(_responses_frame(event_name, {
            "type": event_name,
            "response": _responses_object(state, status=status),
        }, state))
        frames.append("data: [DONE]\n\n")
        state.terminal = True
    return frames


def _format_gemini(event: UnifiedStreamEvent, state: StreamFormatState) -> list[str]:
    if event.type == "error" or event.error is not None:
        state.terminal = True
        return [_data_frame({"error": _error_payload(event.error)})]

    parts: list[dict[str, Any]] = []
    for block in _client_visible_blocks(event):
        if block.tool_call:
            key, _ = _block_key(block, state, event)
            call = block.tool_call
            state.tool_names[key] = call.name or state.tool_names.get(key, "")
            if getattr(call, "signature", None):
                state.tool_signatures[key] = call.signature
            # Synthetic correlation ids never reach the wire (same rule as
            # the non-stream path — the call_ prefix alone is NOT the test).
            if call.id and not call.extra.get("synthetic_id"):
                state.tool_ids[key] = call.id
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
            thought_part: dict[str, Any] = {"text": block.reasoning.text or "", "thought": True}
            if block.reasoning.signature and event.source_protocol == "gemini":
                # thoughtSignature rides the thought part for SAME-PROTOCOL
                # clients (multi-turn replay contract); foreign-source
                # signatures stay suppressed (cache-owned, D8).
                thought_part["thoughtSignature"] = block.reasoning.signature
            parts.append(thought_part)
        elif block.type == "refusal":
            # Gemini has no refusal part: the text survives as a plain part
            # (same degradation as the non-stream path).
            _stream_warn(
                state,
                "refusal_downgraded",
                "refusal has no Gemini stream part; degraded to plain text",
                "content",
            )
            parts.append({"text": block.refusal or ""})
        elif block.type == "text":
            text_part: dict[str, Any] = {"text": block.text or ""}
            signature_on_text = (block.extra or {}).get("thoughtSignature") or (block.extra or {}).get("thought_signature")
            if signature_on_text and event.source_protocol == "gemini":
                text_part["thoughtSignature"] = signature_on_text
            parts.append(text_part)
        elif block.type in {"image", "audio", "video", "file", "document"}:
            # Media parts stream as inlineData/fileData exactly like the
            # non-stream path — never silently dropped.
            source = getattr(block, "source", None)
            media_part = _gemini_media_part(block, source)
            if media_part is not None:
                parts.append(media_part)
        elif block.type == "builtin_tool":
            # Native union members (executableCode, server toolCall, ...)
            # replay their raw part verbatim on the stream.
            if isinstance(block.raw, dict) and any(k in block.raw for k in ("executableCode", "codeExecutionResult", "toolCall", "toolResponse")):
                parts.append(deepcopy(block.raw))

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
    if finish_reason and state.completion_emitted:
        # Duplicate completion (e.g. synthetic terminal after a finish frame):
        # never re-emit the finish reason.
        finish_reason = None
    if parts or finish_reason or event.usage is not None:
        candidate: dict[str, Any] = {"index": event.output_index or 0}
        if parts:
            candidate["content"] = {"role": "model", "parts": parts}
        if finish_reason:
            candidate["finishReason"] = finish_reason
        payload: dict[str, Any] = {"candidates": [candidate]}
        if state.model:
            payload["modelVersion"] = state.model
        usage = _gemini_usage(state.usage or event.usage)
        if usage and not state.completion_emitted:
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
    tool_id = state.tool_ids.get(key)
    if tool_id:
        # `id` is Gemini-3+ only; synthetic ids are filtered at record time.
        function_call["id"] = tool_id
    signature = state.tool_signatures.get(key)
    if signature and state.source_protocol == "gemini":
        function_call_payload = {"functionCall": function_call}
        function_call_payload["thoughtSignature"] = signature
        return function_call_payload
    return {"functionCall": function_call}


def _gemini_media_part(block: Any, source: Any) -> dict[str, Any] | None:
    """Render a canonical media block as a Gemini inlineData/fileData part."""

    if source is None:
        return None
    data = getattr(source, "data", None)
    if data:
        mime = getattr(source, "media_type", None)
        if not mime:
            # Fabricating a mimeType the source never declared is a guess —
            # plain data without mime never reaches the wire (Gemini requires
            # a concrete inlineData.mimeType).
            return None
        return {"inlineData": {"mimeType": mime, "data": data}}
    file_uri = getattr(source, "file_uri", None) or getattr(source, "url", None)
    file_id = getattr(source, "file_id", None)
    if file_uri or file_id:
        file_data: dict[str, Any] = {}
        if file_uri:
            file_data["fileUri"] = file_uri
        if file_id:
            file_data["fileId"] = file_id
        return {"fileData": file_data}
    return None


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


def _harvest_done_state(event: UnifiedStreamEvent, state: "StreamFormatState") -> None:
    """Harvest provider-owned state from done/terminal snapshots.

    Real upstreams deliver ``encrypted_content`` (and tool signatures) only
    on ``output_item.done`` / terminal objects — the delta path never sees
    them. Harvest here so the synthesized done frames and the terminal
    response object carry the bound opaque state (D8).
    """

    encrypted_blocks = [
        block
        for block in _client_visible_blocks(event, include_builtins=True)
        if block.reasoning and block.reasoning.encrypted_content
    ]
    if not encrypted_blocks:
        return
    raw = event.extra.get("payload") if isinstance(event.extra, dict) else None
    raw_item = raw.get("item") if isinstance(raw, dict) and isinstance(raw.get("item"), dict) else None
    raw_item_id = raw_item.get("id") if isinstance(raw_item, dict) else None
    # Match by upstream item id when present; else the (single) reasoning
    # item open on this state.
    if isinstance(raw_item_id, str):
        for key, item_id in state.item_ids.items():
            if item_id == raw_item_id and state.item_kinds.get(key) == "reasoning":
                state.reasoning_encrypted[key] = encrypted_blocks[0].reasoning.encrypted_content
                return
    for key, kind in state.item_kinds.items():
        if kind == "reasoning":
            state.reasoning_encrypted[key] = encrypted_blocks[0].reasoning.encrypted_content
            break


def _client_visible_blocks(event: UnifiedStreamEvent, *, include_builtins: bool = False) -> list[ContentBlock]:
    """Blocks a NON-responses client target may render.

    Builtin tool records are provider-internal: the Responses stream
    formatter emits them as native output items; Anthropic targets render
    anthropic-native server-tool blocks verbatim (they are legal wire
    content); every other target omits them (no fabricated empty text
    blocks).
    """

    if include_builtins:
        return list(_event_blocks(event))
    return [block for block in _event_blocks(event) if block.type != "builtin_tool"]


def _block_key(block: ContentBlock, state: StreamFormatState, event: UnifiedStreamEvent | None = None) -> tuple[str, str]:
    if block.tool_call:
        call = block.tool_call
        identity = call.index if call.index is not None else call.id or call.name or "default"
        # Tool blocks also advance the family sequence so a following text
        # block reopens as a new block instead of merging with the earlier one.
        state.last_family = "tool"
        return f"tool:{identity}", "tool"
    if block.type == "refusal":
        # Refusal keeps its own block family: ordinary text streams never
        # merge into a refusal block (or vice versa) on any target.
        family = "refusal"
    else:
        family = "reasoning" if block.reasoning else "text"
    explicit_index = _event_block_index(event)
    if explicit_index is not None:
        return f"{family}:{explicit_index}", family
    epoch = state.family_epoch.get(family, 0)
    if state.last_family is not None and state.last_family != family:
        epoch += 1
        state.family_epoch[family] = epoch
    else:
        state.family_epoch.setdefault(family, epoch)
    state.last_family = family
    return f"{family}:{epoch}", family


def _event_block_index(event: UnifiedStreamEvent | None) -> str | None:
    """Composite explicit block identity: item/output index + content index.

    Responses-style events set both (content_index is per-item, so two output
    items sharing content_index 0 must NOT merge); identity-less wires (chat)
    return None and fall back to the family-epoch heuristic.
    """

    if event is None:
        return None
    content_index = getattr(event, "content_index", None)
    output_index = getattr(event, "output_index", None)
    if output_index is None and content_index is None:
        return None
    output_part = f"o{output_index}" if isinstance(output_index, int) else "o-"
    content_part = f"c{content_index}" if isinstance(content_index, int) else "c-"
    return f"{output_part}:{content_part}"


def _openai_delta(message: UnifiedMessage | None) -> dict[str, Any]:
    if message is None:
        return {}
    delta: dict[str, Any] = {}
    text = "".join(block.text or "" for block in ordered_message_blocks(message) if block.type == "text" and not block.reasoning)
    reasoning = "".join(block.reasoning.text or "" for block in ordered_message_blocks(message) if block.reasoning)
    refusal = "".join(block.refusal or "" for block in ordered_message_blocks(message) if block.type == "refusal")
    if text:
        delta["content"] = text
    if reasoning:
        delta["reasoning_content"] = reasoning
    if refusal:
        # Chat natively carries refusals on the delta (exact mapping).
        delta["refusal"] = refusal
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
    delta: dict[str, Any] | None,
    finish_reason: str | None,
    usage: Usage | None = None,
    choice_index: int = 0,
    empty_choices: bool = False,
    logprobs: Any = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": state.response_id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.model,
        # Documented include_usage grammar: intermediate chunks carry an
        # explicit null usage; the terminal usage chunk carries an EMPTY
        # choices array.
        "usage": _openai_usage(usage) if usage is not None else None,
    }
    if empty_choices:
        payload["choices"] = []
    else:
        choice: dict[str, Any] = {"index": choice_index, "delta": delta if delta is not None else {}, "finish_reason": finish_reason}
        if logprobs is not None:
            choice["logprobs"] = deepcopy(logprobs)
        payload["choices"] = [choice]
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
        if block.reasoning.redacted:
            # Redacted blocks never stream deltas — one full start payload.
            state.block_signatures[key] = "__redacted__"
            return {"type": "redacted_thinking", "data": block.reasoning.encrypted_content or ""}
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
    if block.type == "refusal":
        # Anthropic has no refusal part: the text survives as a text delta
        # (same degradation as the non-stream path).
        text = block.refusal or ""
        return {"type": "text_delta", "text": text} if text else None
    text = block.text or ""
    return {"type": "text_delta", "text": text} if text else None


def _responses_start(state: StreamFormatState) -> list[str]:
    if state.started:
        return []
    state.started = True
    # Documented lifecycle: created -> in_progress -> items -> terminal.
    return [
        _responses_frame("response.created", {
            "type": "response.created",
            "response": _responses_object(state, status="in_progress"),
        }, state),
        _responses_frame("response.in_progress", {
            "type": "response.in_progress",
            "response": _responses_object(state, status="in_progress"),
        }, state),
    ]


def _responses_frame(event_name: str, payload: dict[str, Any], state: StreamFormatState) -> str:
    """SSE frame with the spec-mandated monotonic sequence_number."""
    payload = dict(payload)
    payload.setdefault("sequence_number", state.sequence)
    state.sequence += 1
    return _event_frame(event_name, payload)


def _responses_item_id(kind: str, index: int) -> str:
    return f"{'fc' if kind == 'tool' else 'rs' if kind == 'reasoning' else 'msg'}_{index}"


def _responses_item_start(block: ContentBlock, key: str, item_id: str, state: StreamFormatState) -> list[str]:
    kind = state.item_kinds[key]
    if kind == "tool" and block.tool_call:
        call = block.tool_call
        state.tool_names[key] = call.name or ""
        state.tool_ids[key] = call.id or item_id
        item = {"id": item_id, "type": "function_call", "call_id": state.tool_ids[key], "name": state.tool_names[key], "arguments": "", "status": "in_progress"}
        return [_responses_frame("response.output_item.added", {"type": "response.output_item.added", "output_index": state.next_index - 1, "item": item}, state)]
    if kind == "reasoning":
        item = {"id": item_id, "type": "reasoning", "summary": [], "status": "in_progress"}
        return [
            _responses_frame("response.output_item.added", {"type": "response.output_item.added", "output_index": state.next_index - 1, "item": item}, state),
            _responses_frame("response.reasoning_summary_part.added", {"type": "response.reasoning_summary_part.added", "item_id": item_id, "output_index": state.next_index - 1, "summary_index": 0, "part": {"type": "summary_text", "text": ""}}, state),
        ]
    if block.type == "refusal":
        item = {"id": item_id, "type": "message", "role": "assistant", "content": [], "status": "in_progress"}
        return [
            _responses_frame("response.output_item.added", {"type": "response.output_item.added", "output_index": state.next_index - 1, "item": item}, state),
            _responses_frame("response.content_part.added", {"type": "response.content_part.added", "item_id": item_id, "output_index": state.next_index - 1, "content_index": 0, "part": {"type": "refusal", "refusal": ""}}, state),
        ]
    item = {"id": item_id, "type": "message", "role": "assistant", "content": [], "status": "in_progress"}
    return [
        _responses_frame("response.output_item.added", {"type": "response.output_item.added", "output_index": state.next_index - 1, "item": item}, state),
        _responses_frame("response.content_part.added", {"type": "response.content_part.added", "item_id": item_id, "output_index": state.next_index - 1, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}, state),
    ]


def _responses_item_delta(block: ContentBlock, key: str, item_id: str, state: StreamFormatState) -> list[str]:
    kind = state.item_kinds[key]
    output_index = list(state.item_ids).index(key)
    if kind == "tool" and block.tool_call:
        fragment = tool_arguments_text(block.tool_call.arguments)
        state.tool_arguments[key] = state.tool_arguments.get(key, "") + fragment
        return [_responses_frame("response.function_call_arguments.delta", {"type": "response.function_call_arguments.delta", "item_id": item_id, "output_index": output_index, "delta": fragment}, state)] if fragment else []
    if block.type == "refusal":
        # Responses refusal parts stream on the refusal variant of the
        # content delta (never dropped).
        text = block.refusal or ""
        if not text:
            return []
        state.text_by_key[key] = state.text_by_key.get(key, "") + text
        state.refusal_by_key[key] = True
        return [_responses_frame("response.refusal.delta", {"type": "response.refusal.delta", "item_id": item_id, "output_index": output_index, "content_index": 0, "delta": text}, state)]
    text = block.reasoning.text if block.reasoning else block.text
    if block.reasoning and block.reasoning.encrypted_content:
        # D8: bound opaque reasoning state rides the streamed item for
        # continuation replay (output_item.done + terminal object).
        state.reasoning_encrypted[key] = block.reasoning.encrypted_content
    if not text:
        return []
    state.text_by_key[key] = state.text_by_key.get(key, "") + text
    event_name = "response.reasoning_summary_text.delta" if kind == "reasoning" else "response.output_text.delta"
    payload = {"type": event_name, "item_id": item_id, "output_index": output_index, "delta": text}
    payload["summary_index" if kind == "reasoning" else "content_index"] = 0
    return [_responses_frame(event_name, payload, state)]


def _responses_item_done(
    key: str,
    item_id: str,
    state: StreamFormatState,
    *,
    item_status: str,
) -> list[str]:
    kind = state.item_kinds[key]
    if kind == "builtin":
        # added+done were already emitted back-to-back at registration.
        return []
    output_index = list(state.item_ids).index(key)
    if kind == "tool":
        arguments = state.tool_arguments.get(key, "")
        item = {"id": item_id, "type": "function_call", "call_id": state.tool_ids.get(key, item_id), "name": state.tool_names.get(key, ""), "arguments": arguments, "status": item_status}
        return [
            _responses_frame("response.function_call_arguments.done", {"type": "response.function_call_arguments.done", "item_id": item_id, "output_index": output_index, "arguments": arguments}, state),
            _responses_frame("response.output_item.done", {"type": "response.output_item.done", "output_index": output_index, "item": item}, state),
        ]
    text = state.text_by_key.get(key, "")
    if kind == "reasoning":
        item = {"id": item_id, "type": "reasoning", "summary": [{"type": "summary_text", "text": text}], "status": item_status}
        encrypted = state.reasoning_encrypted.get(key)
        if encrypted:
            item["encrypted_content"] = encrypted
        return [
            _responses_frame("response.reasoning_summary_text.done", {"type": "response.reasoning_summary_text.done", "item_id": item_id, "output_index": output_index, "summary_index": 0, "text": text}, state),
            _responses_frame("response.reasoning_summary_part.done", {"type": "response.reasoning_summary_part.done", "item_id": item_id, "output_index": output_index, "summary_index": 0, "part": item["summary"][0]}, state),
            _responses_frame("response.output_item.done", {"type": "response.output_item.done", "output_index": output_index, "item": item}, state),
        ]
    if state.refusal_by_key.get(key):
        part = {"type": "refusal", "refusal": text}
        item = {"id": item_id, "type": "message", "role": "assistant", "content": [part], "status": item_status}
        return [
            _responses_frame("response.refusal.done", {"type": "response.refusal.done", "item_id": item_id, "output_index": output_index, "content_index": 0, "refusal": text}, state),
            _responses_frame("response.content_part.done", {"type": "response.content_part.done", "item_id": item_id, "output_index": output_index, "content_index": 0, "part": part}, state),
            _responses_frame("response.output_item.done", {"type": "response.output_item.done", "output_index": output_index, "item": item}, state),
        ]
    part = {"type": "output_text", "text": text, "annotations": []}
    item = {"id": item_id, "type": "message", "role": "assistant", "content": [part], "status": item_status}
    return [
        _responses_frame("response.output_text.done", {"type": "response.output_text.done", "item_id": item_id, "output_index": output_index, "content_index": 0, "text": text}, state),
        _responses_frame("response.content_part.done", {"type": "response.content_part.done", "item_id": item_id, "output_index": output_index, "content_index": 0, "part": part}, state),
        _responses_frame("response.output_item.done", {"type": "response.output_item.done", "output_index": output_index, "item": item}, state),
    ]


def _responses_object(state: StreamFormatState, *, status: str, error: Any = None) -> dict[str, Any]:
    output = []
    item_status = "completed" if status == "completed" else "in_progress" if status == "in_progress" else "incomplete"
    for key, item_id in state.item_ids.items():
        kind = state.item_kinds[key]
        if kind == "builtin":
            output.append(deepcopy(state.builtin_items.get(key) or {"id": item_id, "type": "builtin_call", "status": item_status}))
        elif kind == "tool":
            output.append({"id": item_id, "type": "function_call", "call_id": state.tool_ids.get(key, item_id), "name": state.tool_names.get(key, ""), "arguments": state.tool_arguments.get(key, ""), "status": item_status})
        elif kind == "reasoning":
            reasoning_item: dict[str, Any] = {"id": item_id, "type": "reasoning", "summary": [{"type": "summary_text", "text": state.text_by_key.get(key, "")}], "status": item_status}
            encrypted = state.reasoning_encrypted.get(key)
            if encrypted:
                reasoning_item["encrypted_content"] = encrypted
            output.append(reasoning_item)
        elif state.refusal_by_key.get(key):
            output.append({"id": item_id, "type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": state.text_by_key.get(key, "")}], "status": item_status})
        else:
            output.append({"id": item_id, "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": state.text_by_key.get(key, ""), "annotations": []}], "status": item_status})
    payload: dict[str, Any] = {"id": state.response_id, "object": "response", "status": status, "model": state.model, "output": output}
    if status == "incomplete":
        # Documented reason field on incomplete terminals.
        payload["incomplete_details"] = {"reason": "max_output_tokens" if state.stop_reason == "max_tokens" else "content_filter" if state.stop_reason == "content_filter" else "max_output_tokens"}
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
    # Canonical input_tokens is inclusive of cache reads/writes (H2) —
    # identical formula to the non-streaming formatter, never double-counted.
    payload: dict[str, Any] = {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }
    if usage.cache_read_tokens or usage.cache_write_tokens:
        details: dict[str, Any] = {}
        if usage.cache_read_tokens:
            details["cached_tokens"] = usage.cache_read_tokens
        if usage.cache_write_tokens:
            details["cache_creation_tokens"] = usage.cache_write_tokens
        if usage.audio_tokens:
            details["audio_tokens"] = usage.audio_tokens
        payload["prompt_tokens_details"] = details
    elif usage.audio_tokens:
        payload["prompt_tokens_details"] = {"audio_tokens": usage.audio_tokens}
    completion_details: dict[str, Any] = {}
    if usage.reasoning_tokens:
        completion_details["reasoning_tokens"] = usage.reasoning_tokens
    if usage.output_audio_tokens:
        completion_details["audio_tokens"] = usage.output_audio_tokens
    if usage.accepted_prediction_tokens:
        completion_details["accepted_prediction_tokens"] = usage.accepted_prediction_tokens
    if usage.rejected_prediction_tokens:
        completion_details["rejected_prediction_tokens"] = usage.rejected_prediction_tokens
    if completion_details:
        payload["completion_tokens_details"] = completion_details
    return payload


def _anthropic_usage(usage: Usage | None, *, output_only: bool = False) -> dict[str, int]:
    if usage is None:
        return {"output_tokens": 0} if output_only else {"input_tokens": 0, "output_tokens": 0}
    payload: dict[str, Any] = {"output_tokens": usage.output_tokens}
    if not output_only:
        # Canonical input_tokens is cache-inclusive (H2): unfold to the
        # Anthropic sibling convention, never negative.
        payload["input_tokens"] = max(0, usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens)
        if usage.cache_read_tokens:
            payload["cache_read_input_tokens"] = usage.cache_read_tokens
        if usage.cache_write_tokens:
            payload["cache_creation_input_tokens"] = usage.cache_write_tokens
    if isinstance(usage.extra.get("server_tool_use"), dict):
        payload["server_tool_use"] = deepcopy(usage.extra["server_tool_use"])
    if usage.reasoning_tokens:
        payload["output_tokens_details"] = {"thinking_tokens": usage.reasoning_tokens}
    return payload


def _responses_usage(usage: Usage | None) -> dict[str, Any] | None:
    if usage is None:
        return None
    # Canonical input_tokens is cache-INCLUSIVE (H2): emitted verbatim;
    # details are standard spellings, zero-valued keys omitted.
    payload: dict[str, Any] = {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens or (usage.input_tokens + usage.output_tokens),
    }
    input_details: dict[str, Any] = {}
    if usage.cache_read_tokens:
        input_details["cached_tokens"] = usage.cache_read_tokens
    if usage.cache_write_tokens:
        input_details["cache_write_tokens"] = usage.cache_write_tokens
    if usage.audio_tokens:
        input_details["audio_tokens"] = usage.audio_tokens
    if input_details:
        payload["input_tokens_details"] = input_details
    output_details: dict[str, Any] = {}
    if usage.reasoning_tokens:
        output_details["reasoning_tokens"] = usage.reasoning_tokens
    if usage.output_audio_tokens:
        output_details["audio_tokens"] = usage.output_audio_tokens
    if output_details:
        payload["output_tokens_details"] = output_details
    return payload


def _gemini_usage(usage: Usage | None) -> dict[str, int] | None:
    if usage is None:
        return None
    # Canonical input_tokens is cache-INCLUSIVE (H2) — matching Gemini's own
    # convention (promptTokenCount includes cachedContentTokenCount).
    # Detail keys emit only when non-zero (symmetric with the non-stream
    # formatter; zero-valued detail counts are not wire facts).
    payload: dict[str, int] = {
        "promptTokenCount": usage.input_tokens,
        "candidatesTokenCount": usage.output_tokens,
        "totalTokenCount": usage.total_tokens,
    }
    if usage.cache_read_tokens:
        payload["cachedContentTokenCount"] = usage.cache_read_tokens
    if usage.reasoning_tokens:
        payload["thoughtsTokenCount"] = usage.reasoning_tokens
    return payload


def _is_json(value: str) -> bool:
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return False
    return True