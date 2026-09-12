"""Stateful canonical conversion for generative streaming protocols.

Protocol adapters parse provider frames into :class:`UnifiedStreamEvent`.  This
module owns the inverse operation because destination protocols have different
lifecycle requirements: one canonical delta can expand into several SSE frames,
and terminal events must close every destination-owned content block exactly
once.  Operational concerns (timeouts, retries, cancellation, heartbeats,
relay/rotation) live in ``client/stream_ops.py`` (:class:`NeutralStreamPipeline`)
and ``streaming/relay.py``; ``client/streaming.py`` keeps only buffer/parse
helpers.
"""

from __future__ import annotations

from copy import deepcopy

from dataclasses import dataclass, field
import json
import time
import uuid
from typing import Any

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
    UnifiedMessage,
    UnifiedStreamEvent,
    Usage,
    serialize_value,
)

def _is_anthropic_server_tool_block(raw_type: str) -> bool:
    """Known families + versioned spellings of the same stems (see the
    adapter's _is_server_tool_result_type — a user's own *_tool_result
    block never fabricates a server builtin)."""

    from .anthropic_messages import _is_server_tool_result_type

    return _is_server_tool_result_type(raw_type)


# Canonical block families a NON-Gemini stream target has no wire shape for
# (Gemini renders media as inlineData/fileData). Every target records the
# drop through the shared disclosure helper — never a phantom empty block.
_UNREPRESENTABLE_MEDIA_BLOCK_TYPES = {"image", "audio", "video", "file", "document"}


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
    # G13: choices that have produced at least one frame — the EOF repair
    # closes every SEEN choice, not just index 0 (gomodel Terminate shape).
    seen_choices: set[int] = field(default_factory=set)
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
    # G4: the CLIENT's own stream_options.include_usage request. None keeps
    # legacy behavior (emit the terminal usage frame whenever usage exists);
    # False suppresses the extra usage-only chunk per the official chat
    # streaming grammar (intermediate null-usage also drops with False).
    include_usage: bool | None = None
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
    tool_occurrences: dict[str, int] = field(default_factory=dict)
    source_protocol: str | None = None
    # G13 D8 identity: provider-owned id/model observed on the wire. When
    # present they win over the locally minted identity; the minted value is
    # only a fallback for wires that never carry their own.
    provider_id_seen: str = ""
    provider_model_seen: str = ""
    system_fingerprint: str | None = None
    service_tier: str | None = None
    # G14: response-level Gemini metadata that must survive the stream once
    # (blocked-prompt promptFeedback and modelStatus). Latched on the first
    # chunk that carries it, emitted on the first formatted frame.
    prompt_feedback: dict[str, Any] | None = None
    model_status: dict[str, Any] | None = None
    gemini_meta_emitted: bool = False
    emitted_tools: set[str] = field(default_factory=set)
    # Block-identity bookkeeping (defect 8): events that carry explicit
    # content/output indexes use them directly; identity-less wires (chat
    # chunks) mint a family epoch that reopens when a different block family
    # intervenes, so `text -> tool -> text` stays three distinct blocks.
    family_epoch: dict[str, int] = field(default_factory=dict)
    last_family: str | None = None
    # G13: dense 0-based tool-call index per message for chat targets —
    # source block/item indexes (anthropic content-block index, responses
    # output_index) are NOT chat tool-array positions.
    chat_tool_index_by_key: dict[str, int] = field(default_factory=dict)
    # G13: per-choice role emission (n>1 choices each need role: assistant).
    _role_choices: set[int] = field(default_factory=set)
    # G13 relay shadow-state: when True the state accumulates (usage, tool
    # buffers, identities, latches) but produced frames are discarded —
    # a relayed stream keeps the formatter warm so mid-stream disengage
    # continues from live state instead of a virgin one.
    observe_only: bool = False

    def reset_for_attempt(self) -> None:
        """Rotation survival contract (G13 D1).

        Called by the executor at the start of every credential/attempt
        rotation on the SAME stream. Survives: lifecycle continuity
        (``started`` — the client already saw message_start/created),
        identity (``response_id``/``created``), spent reality (``usage``,
        ``warnings`` — append-only), ``role_emitted``, and ``seen_choices``
        (frames the client already received). Resets: every per-attempt
        latch and buffer — most critically ``terminal`` (an error-frame
        terminal on attempt 1 must not zero out attempt 2's frames),
        finishes, open blocks, tool fragments, opaque-state caches.
        """

        self.terminal = False
        self.completion_emitted = False
        self.finished_choices = set()
        self.stop_reason = None
        self.stop_sequence = None
        self.next_index = 0
        self.open_blocks = {}
        self.block_order = []
        self.item_ids = {}
        self.item_kinds = {}
        self.builtin_items = {}
        self.refusal_by_key = {}
        self.text_by_key = {}
        self.tool_arguments = {}
        self.tool_names = {}
        self.tool_ids = {}
        self.tool_signatures = {}
        self.tool_occurrences = {}
        self.emitted_tools = set()
        self.family_epoch = {}
        self.last_family = None
        self.chat_tool_index_by_key = {}
        self._role_choices = set(self.seen_choices) if self.role_emitted else set()
        self.block_signatures = {}
        self.emitted_signatures = set()
        self.reasoning_encrypted = {}


class ProtocolStreamConverter:
    """Convert raw source frames to the client's own protocol.

    Test/convenience seam only: it pairs one source parser with the shared
    :func:`format_canonical_stream_event` and keeps a warm
    :class:`StreamFormatState`. Production streaming goes through
    ``client/stream_ops.py`` (:class:`NeutralStreamPipeline`) which calls
    ``format_canonical_stream_event`` directly and owns rotation/relay.
    """

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
        # Latch the stream's source protocol once: it is the stream-wide
        # provenance the D8 opaque-state predicate consults (see
        # ``_may_emit_opaque``) so per-event and latched provenance can never
        # disagree.
        state.source_protocol = event.source_protocol
    _lift_provider_identity(state, event)
    if target_protocol != "openai_chat" and (event.extra or {}).get("logprobs") is not None:
        # Logprobs are a Chat-only wire member: any other destination must
        # disclose the loss rather than silently vanish the evidence.
        _stream_warn(
            state,
            "logprobs_dropped",
            f"logprobs have no {target_protocol} stream representation; dropped",
            "logprobs",
        )
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
        frames = _format_openai(event, state)
    elif target_protocol == "anthropic_messages":
        frames = _format_anthropic(event, state)
    elif target_protocol == "responses":
        frames = _format_responses(event, state)
    elif target_protocol == "gemini":
        frames = _format_gemini(event, state)
    elif target_protocol == "ollama":
        frames = _format_ollama(event, state)
    else:
        raise ProtocolError(
            f"Canonical streaming is not supported for {target_protocol}",
            protocol=target_protocol,
            pass_name="format_stream_event",
        )
    # G13 relay shadow: state accumulated, frames discarded. A relayed
    # stream stays format-warm so a mid-stream disengage continues from
    # live state instead of a virgin one.
    if state.observe_only:
        return []
    return frames


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


def _stream_disclose_unrepresentable(state: "StreamFormatState", kind: str, detail: str) -> None:
    """Record the one shared warning for content a target cannot represent.

    The stream formatter never opens a phantom block/item for a canonical
    family the destination has no legal wire shape for: the block is skipped
    outright and the loss is disclosed through the state warning sink (the
    pipeline tail traces it). ``kind`` names the content family and
    ``detail`` the target-specific gap.
    """

    _stream_warn(
        state,
        "unrepresentable_content_dropped",
        f"{kind} content has no {state.protocol} stream representation; omitted ({detail})",
        "content",
    )


def _may_emit_opaque(
    state: "StreamFormatState",
    event: UnifiedStreamEvent | None = None,
    context: ProtocolContext | None = None,
) -> bool:
    """D8 gate: may opaque provider state leave the stream cache?

    Opaque state (anthropic thinking signatures, Gemini thought signatures,
    Responses ``encrypted_content``) is provider-owned and only meaningful to
    the provider that minted it. The stream formatter has no provider identity
    on events, so protocol equality is the gate: the destination protocol must
    agree with BOTH the latched stream source and the current event's source.
    Either mismatch suppresses the value (kept in ``state`` for cache/extract
    use) and records one ``opaque_state_suppressed`` disclosure — cross-
    protocol state is never translated, only extracted.
    """

    target = state.protocol
    latched = state.source_protocol
    event_source = getattr(event, "source_protocol", None) if event is not None else None
    if event_source is None and context is not None:
        event_source = getattr(context, "source_protocol", None)
    mismatches = [source for source in (latched, event_source) if source and source != target]
    if mismatches:
        _stream_warn(
            state,
            "opaque_state_suppressed",
            f"opaque provider state from {mismatches[0]} is not re-emitted to a {target} stream; kept for cache only",
            "content",
        )
        return False
    return True


def _lift_provider_identity(state: "StreamFormatState", event: UnifiedStreamEvent) -> None:
    """Lift provider-owned identity/metadata off the event onto the state.

    Chat chunks carry ``id``/``model``/``system_fingerprint``/``service_tier``
    directly in ``extra``; Gemini and Responses nest them under
    ``extra["payload"]`` (``responseId``/``modelVersion`` or a ``response``
    object). Only the first observed value wins so later frames cannot churn
    the identity the client already received.
    """

    extra = event.extra if isinstance(event.extra, dict) else {}
    payload = extra.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}

    provider_id = extra.get("id") or payload.get("id") or payload.get("responseId") or response.get("id")
    provider_model = extra.get("model") or payload.get("model") or payload.get("modelVersion") or response.get("model")
    if not state.provider_id_seen and isinstance(provider_id, str) and provider_id:
        state.provider_id_seen = provider_id
    if not state.provider_model_seen and isinstance(provider_model, str) and provider_model:
        state.provider_model_seen = provider_model

    fingerprint = extra.get("system_fingerprint") or payload.get("system_fingerprint")
    if state.system_fingerprint is None and isinstance(fingerprint, str) and fingerprint:
        state.system_fingerprint = fingerprint
    service_tier = extra.get("service_tier") or payload.get("service_tier")
    if state.service_tier is None and isinstance(service_tier, str) and service_tier:
        state.service_tier = service_tier
    prompt_feedback = payload.get("promptFeedback")
    if state.prompt_feedback is None and isinstance(prompt_feedback, dict):
        state.prompt_feedback = deepcopy(prompt_feedback)
    model_status = payload.get("modelStatus")
    if state.model_status is None and isinstance(model_status, dict):
        state.model_status = deepcopy(model_status)


def _gemini_identity(state: "StreamFormatState") -> dict[str, Any]:
    """Gemini stream identity fields (provider-owned wins over the fallback)."""

    fields: dict[str, Any] = {}
    if state.provider_id_seen:
        fields["responseId"] = state.provider_id_seen
    model = state.provider_model_seen or state.model
    if model:
        fields["modelVersion"] = model
    return fields


_GEMINI_PART_RESIDUAL_KEYS = ("videoMetadata", "partMetadata", "mediaResolution", "audioTranscription")


def _gemini_meta_fields(state: "StreamFormatState") -> dict[str, Any]:
    """Response-level Gemini metadata emitted once on the first frame.

    ``promptFeedback`` (a blocked prompt's blockReason) and ``modelStatus``
    live at the response root, not on a candidate; the stream must carry them
    or a blocked prompt reads as an empty success.
    """

    fields: dict[str, Any] = {}
    if state.gemini_meta_emitted:
        return fields
    if state.prompt_feedback is not None:
        fields["promptFeedback"] = deepcopy(state.prompt_feedback)
    if state.model_status is not None:
        fields["modelStatus"] = deepcopy(state.model_status)
    if fields:
        state.gemini_meta_emitted = True
    return fields


def _gemini_same_protocol(state: "StreamFormatState", event: UnifiedStreamEvent | None) -> bool:
    """Whether the stream source and destination are both Gemini."""

    source = (getattr(event, "source_protocol", None) if event is not None else None) or state.source_protocol
    return source == "gemini" and state.protocol == "gemini"


def _merge_gemini_part_residuals(part: dict[str, Any], block: Any) -> None:
    """Replay part-level Gemini residuals (same-protocol via raw extras).

    videoMetadata/partMetadata/mediaResolution/audioTranscription are legal
    Part members with no cross-protocol home; a same-protocol stream must not
    silently drop them.
    """

    extra = getattr(block, "extra", None)
    if not isinstance(extra, dict):
        return
    for key in _GEMINI_PART_RESIDUAL_KEYS:
        if key in extra and key not in part:
            part[key] = deepcopy(extra[key])


def _format_openai(event: UnifiedStreamEvent, state: StreamFormatState) -> list[str]:
    if event.type == "error" or event.error is not None:
        state.terminal = True
        return [_data_frame({"error": _error_payload(event.error, "openai_chat")}), "data: [DONE]\n\n"]

    frames: list[str] = []
    delta = _openai_delta(event.delta or event.message, state)
    choice_index = event.output_index or 0
    choice_logprobs = event.extra.get("logprobs")
    obfuscation = event.extra.get("obfuscation")
    if obfuscation is not None:
        # Chat->Chat only (this function is the chat target): the provider's
        # obfuscation payload rides the delta frame verbatim.
        delta["obfuscation"] = deepcopy(obfuscation)
    if delta:
        state.seen_choices.add(choice_index)
        role_needed = choice_index not in state._role_choices
        if role_needed:
            delta.setdefault("role", "assistant")
            state._role_choices.add(choice_index)
        frames.append(_data_frame(_openai_chunk(state, delta=delta, finish_reason=None, choice_index=choice_index, logprobs=choice_logprobs)))

    reason = event.stop_reason or event.extra.get("stop_reason")
    if reason and choice_index not in state.finished_choices:
        state.stop_reason = str(reason)
        state.seen_choices.add(choice_index)
        frames.append(_data_frame(_openai_chunk(
            state,
            delta={},
            finish_reason=format_stop_reason(state.stop_reason, "openai_chat"),
            choice_index=choice_index,
            logprobs=choice_logprobs,
        )))
        state.finished_choices.add(choice_index)
    if _is_terminal(event):
        # G13 close-all: the EOF repair must finish every SEEN choice the
        # provider never finished (gomodel Terminate shape) — a strict SDK
        # assembling choice 1..n-1 hangs forever on a missing finish.
        for open_choice in sorted(state.seen_choices - state.finished_choices):
            frames.append(_data_frame(_openai_chunk(
                state,
                delta={},
                finish_reason=format_stop_reason(state.stop_reason or "stop", "openai_chat"),
                choice_index=open_choice,
            )))
            state.finished_choices.add(open_choice)
        if state.usage is not None and state.include_usage is not False:
            frames.append(_data_frame(_openai_chunk(state, delta=None, finish_reason=None, usage=state.usage, empty_choices=True)))
        frames.append("data: [DONE]\n\n")
        state.terminal = True
    return frames


def _format_anthropic(event: UnifiedStreamEvent, state: StreamFormatState) -> list[str]:
    if event.type == "error" or event.error is not None:
        state.terminal = True
        return [_event_frame("error", {"type": "error", "error": _error_payload(event.error, "anthropic_messages")})]
    if event.type == "heartbeat" or event.native_type == "ping":
        # Keep-alives never open message lifecycle frames.
        return []

    frames = _anthropic_start(state)
    visible = _client_visible_blocks(event, include_builtins=True)
    for block in visible:
        if block.type == "citations_delta":
            # Citations attach to the OPEN text block — no new lifecycle.
            # Every annotation renders its own frame (a foreign block may
            # batch several); with no open text block there is no legal
            # attachment point, so the drop is disclosed.
            open_indexes = [state.open_blocks[k] for k in state.block_order if k in state.open_blocks and k.startswith("text:")]
            if open_indexes and block.annotations:
                for annotation in block.annotations:
                    frames.append(_event_frame("content_block_delta", {
                        "type": "content_block_delta",
                        "index": open_indexes[-1],
                        "delta": {"type": "citations_delta", "citation": deepcopy(annotation.raw) if isinstance(annotation.raw, dict) else {"type": "citation", "cited_text": annotation.citation, "url": annotation.url}},
                    }))
            else:
                _stream_disclose_unrepresentable(state, "citations", "no open text block to attach the citation stream")
            continue
        if block.type in _UNREPRESENTABLE_MEDIA_BLOCK_TYPES or block.type == "unknown":
            # No Anthropic content-block start shape exists for media or
            # unknown families: skip the block outright (the old fall-through
            # opened a phantom empty text block with zero deltas).
            _stream_disclose_unrepresentable(state, block.type, "no Anthropic content-block stream shape")
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
                    frames.extend(_anthropic_close_block(open_key, state, event))
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
                "content_block": _anthropic_block_start(block, key, state, event),
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
            frames.extend(_anthropic_close_block(key, state, event))
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


def _anthropic_close_block(
    key: str,
    state: StreamFormatState,
    event: UnifiedStreamEvent | None = None,
) -> list[str]:
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
        and _may_emit_opaque(state, event)
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
        error = _error_payload(event.error, "responses")
        # Open items close before the failure terminal (grammar parity with
        # the completed path).
        for key, item_id in state.item_ids.items():
            frames.extend(_responses_item_done(key, item_id, state, item_status="incomplete", event=event))
        frames.append(_responses_frame("response.failed", {
            "type": "response.failed",
            "response": _responses_object(state, status="failed", error=error, event=event),
        }, state))
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
        if block.type in _UNREPRESENTABLE_MEDIA_BLOCK_TYPES or block.type in {"unknown", "citations_delta"}:
            # Responses output items exist for text/reasoning/refusal/tools;
            # media, citations, and unknown families would mint an empty
            # message item with zero deltas — skip + disclose instead.
            _stream_disclose_unrepresentable(state, block.type, "no Responses output-item stream shape")
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
            frames.extend(_responses_item_done(key, item_id, state, item_status=item_status, event=event))
        event_name = "response.failed" if status == "failed" else "response.completed" if status == "completed" else "response.incomplete"
        frames.append(_responses_frame(event_name, {
            "type": event_name,
            "response": _responses_object(state, status=status, event=event),
        }, state))
        # Official Responses SSE grammar ends on the terminal event — the
        # chat-grammar `data: [DONE]` sentinel does not belong here.
        state.terminal = True
    return frames


def _format_gemini(event: UnifiedStreamEvent, state: StreamFormatState) -> list[str]:
    if event.type == "error" or event.error is not None:
        state.terminal = True
        return [_data_frame({"error": _error_payload(event.error, "gemini")})]

    parts: list[dict[str, Any]] = []
    for block in _client_visible_blocks(event, include_builtins=True):
        if block.type == "builtin_tool" and not (
            isinstance(block.raw, dict)
            and any(k in block.raw for k in ("executableCode", "codeExecutionResult", "toolCall", "toolResponse"))
        ):
            # Foreign-shaped builtin records (no Gemini part home) drop
            # disclosed — never silently, never fabricated.
            _stream_warn(
                state,
                "builtin_tool_output_dropped",
                f"builtin tool record ({getattr(getattr(block, 'builtin_tool', None), 'kind', '') or 'unknown'}) has no Gemini stream part; omitted",
                "content",
            )
            continue
        if block.type == "citations_delta":
            # Gemini has no part-level annotation slot: a citation cannot
            # ride a text part, and a standalone citation part has no union
            # member — disclose the drop (never a phantom part).
            _stream_disclose_unrepresentable(state, "citations", "no Gemini part-level annotation shape")
            continue
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
                    # G13: never raise in-band — a post-emit fragment
                    # degrades to a disclosed drop (the emitted complete
                    # object stays authoritative).
                    _stream_warn(
                        state,
                        "tool_arguments_late_fragment",
                        "Gemini tool-call fragment arrived after the complete object was emitted; dropped",
                        "tool_calls",
                    )
                continue
            state.tool_arguments[key] = state.tool_arguments.get(key, "") + fragment
            if not state.tool_arguments[key]:
                continue
            try:
                arguments = json.loads(state.tool_arguments[key])
            except json.JSONDecodeError:
                continue
            parts.append(_gemini_function_call_part(key, arguments, state, event))
            state.emitted_tools.add(key)
        elif block.reasoning:
            thought_part: dict[str, Any] = {"text": block.reasoning.text or "", "thought": True}
            if block.reasoning.signature and _may_emit_opaque(state, event):
                # thoughtSignature rides the thought part for SAME-PROTOCOL
                # clients (multi-turn replay contract); foreign-source
                # signatures stay suppressed (cache-owned, D8).
                thought_part["thoughtSignature"] = block.reasoning.signature
            if _gemini_same_protocol(state, event):
                _merge_gemini_part_residuals(thought_part, block)
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
            if signature_on_text and _may_emit_opaque(state, event):
                text_part["thoughtSignature"] = signature_on_text
            if block.annotations:
                # No Gemini part carries annotations; the text survives, the
                # citation evidence is disclosed as dropped.
                _stream_disclose_unrepresentable(state, "citations", "no Gemini part-level annotation shape")
            if _gemini_same_protocol(state, event):
                _merge_gemini_part_residuals(text_part, block)
            parts.append(text_part)
        elif block.type in {"image", "audio", "video", "file", "document"}:
            # Media parts stream as inlineData/fileData exactly like the
            # non-stream path — never silently dropped.
            source = getattr(block, "source", None)
            media_part = _gemini_media_part(block, source, state=state)
            if media_part is not None:
                media_signature = (block.extra or {}).get("thoughtSignature") or (block.extra or {}).get("thought_signature")
                if media_signature and _may_emit_opaque(state, event):
                    # Opaque part-level signature on the media extra: same
                    # D8 gate as text/thought parts.
                    media_part["thoughtSignature"] = media_signature
                if _gemini_same_protocol(state, event):
                    _merge_gemini_part_residuals(media_part, block)
                parts.append(media_part)
        elif block.type == "builtin_tool":
            # Native union members (executableCode, server toolCall, ...)
            # replay their raw part verbatim on the stream (foreign shapes
            # were filtered + disclosed at the loop top).
            parts.append(deepcopy(block.raw))
        else:
            # Unknown/future part families have no Gemini union member to
            # replay (native raws rode their own branches above): disclose
            # rather than vanish.
            _stream_disclose_unrepresentable(state, block.type, "no Gemini stream part shape")

    if _is_terminal(event):
        for key in state.tool_names:
            if key in state.emitted_tools:
                continue
            arguments_text = state.tool_arguments.get(key, "")
            arguments: Any = {}
            if arguments_text:
                if _is_json(arguments_text):
                    arguments = json.loads(arguments_text)
                else:
                    # G13: the formatter NEVER raises in-band (a raise kills
                    # the whole stream through the generic-exception path).
                    # Incomplete fragments degrade to their accumulated
                    # object-or-empty form, disclosed via a stream warning.
                    _stream_warn(
                        state,
                        "tool_arguments_incomplete",
                        "Gemini stream ended with an incomplete tool-call argument fragment; emitted the accumulated prefix as-is",
                        "tool_calls",
                    )
                    try:
                        arguments = json.loads(arguments_text)
                    except json.JSONDecodeError:
                        arguments = {"_incomplete": arguments_text}
            parts.append(_gemini_function_call_part(key, arguments, state, event))
            state.emitted_tools.add(key)

    frames: list[str] = []
    finish_reason = None
    if event.stop_reason or event.extra.get("stop_reason"):
        state.stop_reason = str(event.stop_reason or event.extra.get("stop_reason"))
        finish_reason = format_stop_reason(state.stop_reason, "gemini")
    candidate_index = event.output_index or 0
    if parts:
        state.seen_choices.add(candidate_index)
    # Duplicate completion suppresses ONLY the current candidate's frame —
    # the stream-global flag is usage framing, sibling candidates must each
    # close on the wire even after candidate 0 finished.
    current_finish = finish_reason if candidate_index not in state.finished_choices else None
    usage_ready = event.usage is not None and not state.completion_emitted
    if _is_terminal(event) and state.stop_reason:
        # G13 close-all: finish every SEEN candidate the provider never
        # finished on the wire (gomodel Terminate shape) — an SDK assembling
        # candidate 1..n-1 waits forever on a missing finishReason.
        _open_candidates = sorted(state.seen_choices - state.finished_choices - {candidate_index})
    else:
        _open_candidates = []
    if parts or current_finish or usage_ready or _open_candidates:
        if not parts and not current_finish and usage_ready:
            # Pre-finish usage-only frame: the clean usageMetadata chunk —
            # never an empty-candidate husk (undocumented shape).
            usage = _gemini_usage(state.usage or event.usage, preserve_source=_gemini_same_protocol(state, event))
            if usage:
                payload = {"usageMetadata": usage}
                payload.update(_gemini_identity(state))
                payload.update(_gemini_meta_fields(state))
                frames.append(_data_frame(payload))
            state.completion_emitted = True
            if _is_terminal(event):
                state.terminal = True
                for open_candidate in _open_candidates:
                    state.finished_choices.add(open_candidate)
            return frames
        candidate: dict[str, Any] = {"index": candidate_index}
        if parts:
            candidate["content"] = {"role": "model", "parts": parts}
        if current_finish:
            candidate["finishReason"] = current_finish
        payload: dict[str, Any] = {"candidates": [candidate]}
        payload.update(_gemini_identity(state))
        payload.update(_gemini_meta_fields(state))
        usage = _gemini_usage(state.usage or event.usage, preserve_source=_gemini_same_protocol(state, event))
        if usage and not state.completion_emitted:
            payload["usageMetadata"] = usage
        frames.append(_data_frame(payload))
        if current_finish:
            state.finished_choices.add(candidate_index)
            state.completion_emitted = True
        for open_candidate in _open_candidates:
            closing: dict[str, Any] = {
                "index": open_candidate,
                "finishReason": format_stop_reason(state.stop_reason, "gemini") or "STOP",
            }
            closing_payload: dict[str, Any] = {"candidates": [closing]}
            closing_payload.update(_gemini_identity(state))
            closing_payload.update(_gemini_meta_fields(state))
            frames.append(_data_frame(closing_payload))
            state.finished_choices.add(open_candidate)
    elif event.usage is not None and state.completion_emitted and not parts and not finish_reason:
        # Terminal usage after the last finish frame: a usage-only chunk —
        # never an empty-candidate husk (undocumented shape).
        usage = _gemini_usage(state.usage or event.usage, preserve_source=_gemini_same_protocol(state, event))
        if usage:
            payload = {"usageMetadata": usage}
            payload.update(_gemini_identity(state))
            payload.update(_gemini_meta_fields(state))
            frames.append(_data_frame(payload))
    if not frames and not state.gemini_meta_emitted and (state.prompt_feedback is not None or state.model_status is not None):
        # Blocked-prompt / status-only chunk: the response root metadata has
        # no candidate to ride — emit it on its own frame rather than losing
        # the blockReason (or modelStatus) entirely.
        meta_payload = _gemini_meta_fields(state)
        if meta_payload:
            meta_payload.update(_gemini_identity(state))
            frames.append(_data_frame(meta_payload))
    if _is_terminal(event):
        state.terminal = True
    return frames


def _ollama_json_line(payload: dict[str, Any]) -> str:
    """One newline-terminated NDJSON frame (Ollama's streaming wire)."""

    return json.dumps(serialize_value(payload), ensure_ascii=False) + "\n"


def _ollama_timestamp(created: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created))


def _ollama_stream_usage(usage: Usage | None) -> dict[str, Any]:
    """Terminal statistics block: counts plus verbatim nanosecond durations."""

    if usage is None:
        return {}
    payload: dict[str, Any] = {}
    if isinstance(usage.raw, dict):
        for key, value in usage.raw.items():
            if key.endswith("duration"):
                payload[key] = deepcopy(value)
    if usage.input_tokens:
        payload["prompt_eval_count"] = usage.input_tokens
    if usage.output_tokens:
        payload["eval_count"] = usage.output_tokens
    if usage.cache_read_tokens:
        payload["prompt_eval_count_cached"] = usage.cache_read_tokens
    return payload


def _ollama_parse_arguments(text: str) -> Any:
    if not text:
        return {} if text == "" else None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _ollama_stream_tool_calls(
    blocks: list[ContentBlock],
    state: StreamFormatState,
    terminal: bool,
) -> list[dict[str, Any]]:
    """Accumulate tool-call fragments into complete Ollama function calls.

    Ollama requires ``function.arguments`` to be a JSON object, so fragments
    are buffered until they parse (or the stream terminates) — partial JSON is
    never emitted as an object.
    """

    calls: list[dict[str, Any]] = []
    for block in blocks:
        call = block.tool_call
        if call is None:
            continue
        identity = call.index if call.index is not None else (call.id or call.name or "default")
        key = f"tool:{identity}"
        state.tool_names[key] = call.name or state.tool_names.get(key, "")
        if call.id and not call.extra.get("synthetic_id"):
            state.tool_ids[key] = call.id
        if key in state.emitted_tools:
            continue
        fragment = tool_arguments_text(call.arguments)
        state.tool_arguments[key] = state.tool_arguments.get(key, "") + fragment
        arguments = _ollama_parse_arguments(state.tool_arguments[key])
        if arguments is None:
            if not terminal:
                continue
            accumulated = state.tool_arguments[key]
            _stream_warn(
                state,
                "tool_arguments_incomplete",
                "Ollama stream ended with an incomplete tool-call argument fragment; emitted the accumulated prefix as-is",
                "tool_calls",
            )
            arguments = json.loads(accumulated) if _is_json(accumulated) else {"_incomplete": accumulated}
        calls.append({"function": {"name": state.tool_names.get(key, ""), "arguments": arguments}})
        state.emitted_tools.add(key)
    return calls


def _format_ollama(event: UnifiedStreamEvent, state: StreamFormatState) -> list[str]:
    """Format one canonical event as an Ollama NDJSON line.

    Wire shape: ``{"message": {...}, "done": false}`` per delta and a terminal
    ``{"done": true, "done_reason": ..., ...counts/durations}`` object. Ollama
    has no official mid-stream error frame; an ``{"error": {...}}`` object line
    is emitted (plexus reference) and the stream terminates.
    """

    if event.type == "error" or event.error is not None:
        state.terminal = True
        return [_ollama_json_line({"error": _error_payload(event.error, "ollama")})]

    message = event.delta or event.message
    terminal = _is_terminal(event)
    blocks = ordered_message_blocks(message) if message is not None else []
    if terminal:
        # Terminal snapshots repeat content already delivered as deltas; only
        # tool calls still awaiting emission ride the final frame.
        content_text = ""
        reasoning_text = ""
        blocks = [block for block in blocks if block.tool_call]
    else:
        content_text = "".join(block.text or "" for block in blocks if block.type == "text" and not block.reasoning)
        reasoning_text = "".join(block.reasoning.text or "" for block in blocks if block.reasoning)

    if event.stop_reason:
        state.stop_reason = event.stop_reason
    elif event.extra.get("stop_reason"):
        state.stop_reason = str(event.extra["stop_reason"])

    tool_calls = _ollama_stream_tool_calls(blocks, state, terminal)
    model = state.provider_model_seen or state.model
    created = _ollama_timestamp(state.created)

    if terminal:
        message_payload: dict[str, Any] = {"role": "assistant", "content": content_text}
        if reasoning_text:
            message_payload["thinking"] = reasoning_text
        if tool_calls:
            message_payload["tool_calls"] = tool_calls
        payload: dict[str, Any] = {
            "model": model,
            "created_at": created,
            "message": message_payload,
            "done": True,
        }
        native_done = event.extra.get("done_reason")
        if isinstance(native_done, str) and native_done:
            payload["done_reason"] = native_done
        else:
            mapped = _ollama_done_reason(event.stop_reason or state.stop_reason)
            if mapped is not None:
                payload["done_reason"] = mapped
        payload.update(_ollama_stream_usage(state.usage or event.usage))
        state.terminal = True
        return [_ollama_json_line(payload)]

    if not content_text and not reasoning_text and not tool_calls:
        return []
    message_payload = {"role": "assistant", "content": content_text}
    if reasoning_text:
        message_payload["thinking"] = reasoning_text
    if tool_calls:
        message_payload["tool_calls"] = tool_calls
    return [_ollama_json_line({"model": model, "created_at": created, "message": message_payload, "done": False})]


def _ollama_done_reason(stop_reason: str | None) -> str | None:
    if stop_reason == "stop":
        return "stop"
    if stop_reason == "max_tokens":
        return "length"
    return None


def _gemini_skip_signature_sentinel() -> dict[str, Any]:
    """Gemini request-side sentinel for a call with no recoverable signature.

    Gemini-3 validates thought signatures on function-call parts; when a
    conversation is known to carry signatures (``state.tool_signatures`` saw
    them for sibling calls) but this call never got one, the caller must say
    so explicitly rather than emit an unsigned part the provider rejects.
    Wire-legal shape: the LITERAL STRING value on ``thoughtSignature`` —
    verified against the official docs and reference implementations
    (a sibling boolean key is not a shape Gemini accepts).
    """

    return {"thoughtSignature": "skip_thought_signature_validator"}


def _gemini_function_call_part(
    key: str,
    arguments: Any,
    state: StreamFormatState,
    event: UnifiedStreamEvent | None = None,
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
    if signature and _may_emit_opaque(state, event):
        function_call_payload = {"functionCall": function_call}
        function_call_payload["thoughtSignature"] = signature
        return function_call_payload
    source = (getattr(event, "source_protocol", None) if event is not None else None) or state.source_protocol
    if (
        not signature
        and source == "gemini"
        and any(other_key != key for other_key in state.tool_signatures)
    ):
        # Conservative: only when signatures were known to be in play for the
        # conversation, never fabricated for an all-unsigned stream.
        function_call_payload = {"functionCall": function_call}
        function_call_payload.update(_gemini_skip_signature_sentinel())
        return function_call_payload
    return {"functionCall": function_call}


def _gemini_media_part(block: Any, source: Any, state: "StreamFormatState | None" = None) -> dict[str, Any] | None:
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
            if state is not None:
                _stream_warn(
                    state,
                    "media_dropped",
                    "media without a declared mimeType has no Gemini inlineData shape; dropped",
                    "content",
                )
            return None
        return {"inlineData": {"mimeType": mime, "data": data}}
    file_uri = getattr(source, "file_uri", None) or getattr(source, "url", None)
    if file_uri:
        # fileUri is the wire member (external HTTPS included, ≤100MB fetch);
        # file_id is an OpenAI-side identity with no Gemini spelling.
        return {"fileData": {"fileUri": file_uri}}
    if state is not None:
        _stream_warn(
            state,
            "media_dropped",
            "media without inline data or a fileUri has no Gemini part shape; dropped",
            "content",
        )
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
    response object carry the bound opaque state (D8). Reads the event's
    message DIRECTLY: the delta-skip filter returns empty for exactly
    these events.
    """

    message = event.delta or event.message
    if message is None:
        return
    encrypted_blocks = [
        block
        for block in ordered_message_blocks(message)
        if block.reasoning and block.reasoning.encrypted_content
    ]
    if not encrypted_blocks:
        return
    raw = event.extra.get("payload") if isinstance(event.extra, dict) else None
    raw_item = raw.get("item") if isinstance(raw, dict) and isinstance(raw.get("item"), dict) else None
    raw_item_id = raw_item.get("id") if isinstance(raw_item, dict) else None
    # Attribution, in order of precision: the event's output_index (the
    # real wire carries it as a SIBLING of item, and the parser populates
    # event.output_index), then a lenient inside-item read, then the
    # upstream item id, then the first open reasoning item. Without this,
    # multi-reasoning responses mis-bind opaque state (upstream ids never
    # match the synthesized rs_N ids).
    raw_output_index = event.output_index if isinstance(event.output_index, int) else (raw_item.get("output_index") if isinstance(raw_item, dict) else None)
    encrypted_value = encrypted_blocks[0].reasoning.encrypted_content
    if isinstance(raw_output_index, int):
        candidate_keys = [key for key, _ in state.item_ids.items() if state.item_kinds.get(key) == "reasoning"]
        candidate_keys.sort(key=lambda key: list(state.item_ids).index(key))
        if 0 <= raw_output_index < len(candidate_keys):
            state.reasoning_encrypted[candidate_keys[raw_output_index]] = encrypted_value
            return
    if isinstance(raw_item_id, str):
        for key, item_id in state.item_ids.items():
            if item_id == raw_item_id and state.item_kinds.get(key) == "reasoning":
                state.reasoning_encrypted[key] = encrypted_value
                return
    for key, kind in state.item_kinds.items():
        if kind == "reasoning":
            state.reasoning_encrypted[key] = encrypted_value
            return
    # The item never streamed deltas (complete-on-arrival): register it now
    # so the done frames + terminal object include it.
    key = f"reasoning:{state.next_index}"
    state.next_index += 1
    item_id = raw_item_id if isinstance(raw_item_id, str) else _responses_item_id("reasoning", state.next_index)
    state.item_ids[key] = item_id
    state.item_kinds[key] = "reasoning"
    state.reasoning_encrypted[key] = encrypted_blocks[0].reasoning.encrypted_content
    for block in encrypted_blocks:
        if block.reasoning and block.reasoning.text:
            state.text_by_key[key] = (state.text_by_key.get(key, "") or "") + block.reasoning.text
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
        if call.index is None and call.id is None:
            # Id-less, index-less (legal on Gemini): an occurrence counter
            # separates genuinely NEW calls — but never splits fragments of
            # one call. Continuation fragments keep the open occurrence; a
            # COMPLETE object after a complete buffer is a snapshot
            # correction of the SAME call (replace, not a second call); an
            # incomplete fragment after a complete buffer starts a new
            # parallel occurrence.
            fragment = tool_arguments_text(call.arguments)
            open_occurrence = state.tool_occurrences.get(f"{identity}#open")
            if open_occurrence is None:
                open_occurrence = state.tool_occurrences.get(identity, 0)
                state.tool_occurrences[identity] = open_occurrence + 1
                state.tool_occurrences[f"{identity}#open"] = open_occurrence
            else:
                buffered = state.tool_occurrences.get(f"{identity}#buffered", "")
                if buffered and _is_json(buffered):
                    if fragment and not _is_json(fragment):
                        # complete buffer + incomplete fragment = genuinely
                        # the next parallel call — new occurrence
                        open_occurrence = state.tool_occurrences.get(identity, 0)
                        state.tool_occurrences[identity] = open_occurrence + 1
                        state.tool_occurrences[f"{identity}#open"] = open_occurrence
                        state.tool_occurrences[f"{identity}#buffered"] = fragment
                    else:
                        # complete + complete = snapshot correction: replace
                        state.tool_occurrences[f"{identity}#buffered"] = fragment
                else:
                    state.tool_occurrences[f"{identity}#buffered"] = buffered + fragment
            identity = f"{identity}#{open_occurrence}"
        # Tool blocks also advance the family sequence so a following text
        # block reopens as a new block instead of merging with the earlier one.
        state.last_family = "tool"
        state.family_epoch[f"#last|{_event_output_scope(event)}"] = "tool"
        return f"tool:{identity}", "tool"
    if block.type == "refusal":
        # Refusal keeps its own block family: ordinary text streams never
        # merge into a refusal block (or vice versa) on any target.
        family = "refusal"
    else:
        family = "reasoning" if block.reasoning else "text"
    explicit_index = _event_block_index(event)
    content_part_is_real = (
        event is not None and isinstance(getattr(event, "content_index", None), int)
    )
    if explicit_index is not None and content_part_is_real:
        # Genuine per-block identity (anthropic content_index, responses
        # dual-index): the explicit key IS the block. Record the family so
        # a later identity-less block still sequences correctly.
        state.last_family = family
        return f"{family}:{explicit_index}", family
    # Choice-scoped family epoch (G13): output_index identifies the CHOICE,
    # never the block within it — chat/gemini sources carry output_index on
    # every event with content_index always None, so the epoch must run
    # PER CHOICE for `text -> tool -> text` to stay three blocks.
    scope = _event_output_scope(event)
    epoch_key = f"{family}|{scope}"
    epoch = state.family_epoch.get(epoch_key, 0)
    last_for_scope = state.family_epoch.get(f"#last|{scope}")
    if last_for_scope is not None and last_for_scope != family:
        epoch += 1
        state.family_epoch[epoch_key] = epoch
    else:
        state.family_epoch.setdefault(epoch_key, epoch)
    state.family_epoch[f"#last|{scope}"] = family
    if scope == "o-":
        state.last_family = family
    return f"{family}:{scope}:{epoch}", family


def _event_output_scope(event: UnifiedStreamEvent | None) -> str:
    """Choice scope for family epochs: output_index when present, else '-'."""

    if event is None:
        return "o-"
    output_index = getattr(event, "output_index", None)
    return f"o{output_index}" if isinstance(output_index, int) else "o-"


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


def _openai_delta(message: UnifiedMessage | None, state: StreamFormatState | None = None) -> dict[str, Any]:
    if message is None:
        return {}
    delta: dict[str, Any] = {}
    blocks = ordered_message_blocks(message)
    text = "".join(block.text or "" for block in blocks if block.type == "text" and not block.reasoning)
    reasoning = "".join(block.reasoning.text or "" for block in blocks if block.reasoning)
    refusal = "".join(block.refusal or "" for block in blocks if block.type == "refusal")
    if text:
        delta["content"] = text
    if reasoning:
        delta["reasoning_content"] = reasoning
    if refusal:
        # Chat natively carries refusals on the delta (exact mapping).
        delta["refusal"] = refusal
    calls = [block.tool_call for block in blocks if block.tool_call]
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
    # Annotations ride the delta (the chat stream grammar allows annotation
    # arrays): text-part annotations AND foreign citations_delta blocks both
    # normalize through the same Chat annotation formatter.
    annotations = [annotation for block in blocks for annotation in block.annotations]
    if annotations:
        from .openai_chat import _format_openai_annotations

        delta["annotations"] = _format_openai_annotations(annotations)
    # Audio chunks use the documented Chat response `audio` object on the
    # delta (id/data/transcript/format) — mirroring the non-stream message
    # level, never a content part.
    for block in blocks:
        if block.type != "audio":
            continue
        audio_entry = _openai_audio_delta(block, state)
        if audio_entry is not None:
            delta["audio"] = audio_entry
        break
    # Every remaining canonical family without a Chat delta shape is
    # disclosed, never silently vanished.
    for block in blocks:
        if block.type == "builtin_tool":
            _stream_disclose_unrepresentable(state, "builtin_tool", "no Chat delta shape")
        elif block.type == "unknown":
            _stream_disclose_unrepresentable(state, "unknown", "no Chat delta shape")
        elif block.type in _UNREPRESENTABLE_MEDIA_BLOCK_TYPES and block.type != "audio":
            _stream_disclose_unrepresentable(state, block.type, "no Chat assistant delta content-part shape")
    return delta


def _openai_audio_delta(block: ContentBlock, state: StreamFormatState | None) -> dict[str, Any] | None:
    """Build the Chat streaming ``delta.audio`` object for one audio block.

    Mirrors the non-stream response-level synthesis: inline data plus the
    canonical identity/transcript when present. Keys that are absent stay
    absent (no null placeholders); a URL/file-only source has no documented
    stream delta shape and is disclosed.
    """

    source = getattr(block, "source", None)
    data = getattr(source, "data", None) if source is not None else None
    if not data:
        if state is not None:
            _stream_disclose_unrepresentable(state, "audio", "audio without inline data has no Chat delta.audio shape")
        return None
    entry: dict[str, Any] = {"data": data}
    file_id = getattr(source, "file_id", None)
    if isinstance(file_id, str) and file_id:
        entry["id"] = file_id
    transcript = getattr(source, "transcript", None)
    if isinstance(transcript, str) and transcript:
        entry["transcript"] = transcript
    media_type = getattr(source, "media_type", None)
    if media_type:
        from .openai_chat import _audio_format

        audio_format = _audio_format(media_type)
        if audio_format:
            entry["format"] = audio_format
    return entry



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
        "id": state.provider_id_seen or state.response_id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.provider_model_seen or state.model,
        # Documented include_usage grammar: intermediate chunks carry an
        # explicit null usage; the terminal usage chunk carries an EMPTY
        # choices array. G4: when the client explicitly disabled usage
        # frames the key drops entirely (the official no-usage shape).
        "usage": _openai_usage(usage) if usage is not None else None,
    }
    if state.system_fingerprint is not None:
        payload["system_fingerprint"] = state.system_fingerprint
    if state.service_tier is not None:
        payload["service_tier"] = state.service_tier
    if usage is None and state.include_usage is False:
        payload.pop("usage", None)
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


def _anthropic_block_start(
    block: ContentBlock,
    key: str,
    state: StreamFormatState,
    event: UnifiedStreamEvent | None = None,
) -> dict[str, Any]:
    if block.tool_call:
        call = block.tool_call
        state.tool_names[key] = call.name or state.tool_names.get(key, "")
        state.tool_ids[key] = call.id or state.tool_ids.get(key, f"call_{state.open_blocks[key]}")
        return {"type": "tool_use", "id": state.tool_ids[key], "name": state.tool_names[key], "input": {}}
    if block.reasoning:
        if block.reasoning.redacted:
            # Redacted blocks never stream deltas — one full start payload.
            state.block_signatures[key] = "__redacted__"
            if _may_emit_opaque(state, event):
                return {"type": "redacted_thinking", "data": block.reasoning.encrypted_content or ""}
            # Cross-protocol bound opaque state is cache-owned: never leak the
            # encrypted blob; degrade to an empty thinking block (legal
            # lifecycle) and let the predicate disclose the suppression.
            return {"type": "thinking", "thinking": ""}
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
    event: UnifiedStreamEvent | None = None,
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
        if encrypted and _may_emit_opaque(state, event):
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


def _responses_object(
    state: StreamFormatState,
    *,
    status: str,
    error: Any = None,
    event: UnifiedStreamEvent | None = None,
) -> dict[str, Any]:
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
            if encrypted and _may_emit_opaque(state, event):
                reasoning_item["encrypted_content"] = encrypted
            output.append(reasoning_item)
        elif state.refusal_by_key.get(key):
            output.append({"id": item_id, "type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": state.text_by_key.get(key, "")}], "status": item_status})
        else:
            output.append({"id": item_id, "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": state.text_by_key.get(key, ""), "annotations": []}], "status": item_status})
    payload: dict[str, Any] = {"id": state.response_id, "object": "response", "status": status, "model": state.model, "output": output}
    if status == "incomplete":
        # Documented reason field on incomplete terminals.
        if state.stop_reason in ("max_tokens", "content_filter"):
            # Only genuine causes carry incomplete_details — an unknown/
            # repaired reason never fabricates a speculative one.
            payload["incomplete_details"] = {"reason": "max_output_tokens" if state.stop_reason == "max_tokens" else "content_filter"}
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


# --- Per-dialect terminal error vocabulary ---------------------------------
#
# Stream errors arrive as provider payloads (dict) or arbitrary exceptions.
# HTTP status is the weakest signal; structured body fields win, then the
# status ladder, then narrow token checks (docs/experimental/error-reference
# G1 §5.9). The index is local and dependency-free by design: the formatter
# must stay import-light for adapter format_stream_event paths.
_GEMINI_STATUS_FAMILIES = {
    "INVALID_ARGUMENT": "invalid_request",
    "DEADLINE_EXCEEDED": "timeout",
    "NOT_FOUND": "not_found",
    "PERMISSION_DENIED": "forbidden",
    "RESOURCE_EXHAUSTED": "quota_exceeded",
    "INTERNAL": "server_error",
    "UNAVAILABLE": "overloaded",
    "UNAUTHENTICATED": "authentication",
    "ALREADY_EXISTS": "conflict",
    "CANCELLED": "server_error",
}
_GOOGLE_RPC_CODE_FAMILIES = {
    1: "server_error",
    3: "invalid_request",
    4: "timeout",
    5: "not_found",
    7: "forbidden",
    8: "quota_exceeded",
    13: "server_error",
    14: "overloaded",
    16: "authentication",
}
_HTTP_STATUS_FAMILIES = {
    400: "invalid_request",
    401: "authentication",
    403: "forbidden",
    404: "not_found",
    408: "timeout",
    409: "conflict",
    413: "request_too_large",
    422: "invalid_request",
    429: "rate_limit",
    499: "server_error",
    500: "server_error",
    502: "overloaded",
    503: "overloaded",
    504: "timeout",
    529: "overloaded",
}
# Ordered (family, markers): quota/context before the generic rate/invalid
# families so a mixed blob classifies on its strongest signal. Markers are
# specific phrases — never bare tokens that misroute (e.g. "rate" ⊂ "generate").
_ERROR_TOKEN_FAMILIES = (
    ("quota_exceeded", ("insufficient_quota", "quota", "billing_error", "enforced_spend_limit_reached", "credit_balance_exhausted", "spend_limit", "resource_exhausted")),
    ("context_window_exceeded", ("context_length_exceeded", "context window", "prompt is too long", "exceeds the maximum number of tokens", "input length and max_tokens")),
    ("rate_limit", ("rate_limit", "rate limit", "ratelimit", "too many requests", "slow down")),
    ("timeout", ("timeout", "timed out", "deadline_exceeded", "deadline exceeded")),
    ("overloaded", ("overloaded", "unavailable", "model is overloaded")),
    ("authentication", ("authentication", "unauthenticated", "unauthorized", "invalid_api_key", "invalid api key", "api key not valid", "reported as leaked")),
    ("forbidden", ("permission_error", "permission", "forbidden", "permission_denied")),
    ("not_found", ("not_found", "model_not_found", "does not exist")),
    ("conflict", ("conflict", "already_exists")),
    ("request_too_large", ("request_too_large", "too large")),
    ("invalid_request", ("invalid_request", "invalid_prompt", "invalid_argument", "bad request")),
    ("server_error", ("server_error", "api_error", "internal")),
)
_OPENAI_ERROR_TYPES = {
    "invalid_request": "invalid_request_error",
    "context_window_exceeded": "invalid_request_error",
    "request_too_large": "invalid_request_error",
    "authentication": "authentication_error",
    "forbidden": "permission_error",
    "not_found": "not_found_error",
    "rate_limit": "rate_limit_error",
    "quota_exceeded": "insufficient_quota",
    "conflict": "invalid_request_error",
}
_ANTHROPIC_ERROR_TYPES = {
    "invalid_request": "invalid_request_error",
    "context_window_exceeded": "invalid_request_error",
    "request_too_large": "request_too_large",
    "authentication": "authentication_error",
    "forbidden": "permission_error",
    "not_found": "not_found_error",
    "rate_limit": "rate_limit_error",
    "quota_exceeded": "billing_error",
    "conflict": "conflict_error",
    "timeout": "timeout_error",
    "overloaded": "overloaded_error",
}
# Responses stream errors use the ResponseError.code enum (never the HTTP
# body vocabulary): rate_limit_exceeded ≠ chat's rate_limit_error spelling.
_RESPONSES_ERROR_CODES = {
    "invalid_request": "invalid_prompt",
    "context_window_exceeded": "invalid_prompt",
    "request_too_large": "invalid_prompt",
    "rate_limit": "rate_limit_exceeded",
    "quota_exceeded": "rate_limit_exceeded",
}
_GEMINI_ERROR_ENVELOPES = {
    "invalid_request": (400, "INVALID_ARGUMENT"),
    "context_window_exceeded": (400, "INVALID_ARGUMENT"),
    "request_too_large": (400, "INVALID_ARGUMENT"),
    "authentication": (401, "UNAUTHENTICATED"),
    "forbidden": (403, "PERMISSION_DENIED"),
    "not_found": (404, "NOT_FOUND"),
    "conflict": (409, "ALREADY_EXISTS"),
    "rate_limit": (429, "RESOURCE_EXHAUSTED"),
    "quota_exceeded": (429, "RESOURCE_EXHAUSTED"),
    "timeout": (504, "DEADLINE_EXCEEDED"),
    "overloaded": (503, "UNAVAILABLE"),
}


def _error_inner(error: Any) -> dict[str, Any]:
    """Unwrap up to two ``{"error": {...}}`` envelopes to the body dict."""

    payload = error
    for _ in range(2):
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            payload = payload["error"]
        else:
            break
    return payload if isinstance(payload, dict) else {}


def _error_family(error: Any) -> str:
    """Classify a stream error payload into the G1 family vocabulary."""

    payload = _error_inner(error)
    status = payload.get("status")
    if isinstance(status, str):
        family = _GEMINI_STATUS_FAMILIES.get(status.strip().upper())
        if family:
            return family
    tokens: list[str] = []
    for key in ("type", "code"):
        value = payload.get(key)
        if isinstance(value, str):
            tokens.append(value.lower())
    if isinstance(error, str):
        tokens.append(error.lower())
    blob = " ".join(tokens)
    for family, markers in _ERROR_TOKEN_FAMILIES:
        if any(marker in blob for marker in markers):
            return family
    message = payload.get("message")
    if isinstance(message, str):
        lowered = message.lower()
        for family, markers in _ERROR_TOKEN_FAMILIES:
            if any(marker in lowered for marker in markers):
                return family
    numeric = payload.get("status_code") or payload.get("http_status")
    if numeric is None and isinstance(status, int):
        numeric = status
    if numeric is None and isinstance(payload.get("code"), int):
        numeric = payload["code"]
    if isinstance(numeric, int):
        if 0 < numeric < 100:
            return _GOOGLE_RPC_CODE_FAMILIES.get(numeric, "server_error")
        if numeric in _HTTP_STATUS_FAMILIES:
            return _HTTP_STATUS_FAMILIES[numeric]
    return "server_error"


def _error_message(error: Any, payload: dict[str, Any]) -> str:
    message = payload.get("message")
    if isinstance(message, str) and message:
        return message
    if isinstance(error, BaseException):
        return str(error) or "Provider stream failed"
    if isinstance(error, str) and error:
        return error
    return "Provider stream failed"


def _error_payload(error: Any, target_protocol: str) -> dict[str, Any]:
    """Render a stream error inside the target dialect's official envelope.

    ``request_id`` is deliberately omitted from the Anthropic shape: the wire
    field must echo the provider's ``request-id`` HEADER, which the formatter
    never sees — fabricating it from the response body would be a false claim.
    """

    payload = _error_inner(error)
    family = _error_family(error)
    message = _error_message(error, payload)
    if target_protocol == "anthropic_messages":
        return {"type": _ANTHROPIC_ERROR_TYPES.get(family, "api_error"), "message": message}
    if target_protocol == "responses":
        return {"code": _RESPONSES_ERROR_CODES.get(family, "server_error"), "message": message}
    if target_protocol == "gemini":
        code, status = _GEMINI_ERROR_ENVELOPES.get(family, (500, "INTERNAL"))
        return {"code": code, "message": message, "status": status}
    # openai_chat: keep the provider body, normalize the envelope keys.
    result: dict[str, Any] = deepcopy(payload)
    result["message"] = message
    result["type"] = _OPENAI_ERROR_TYPES.get(family, "server_error")
    result["param"] = result.get("param") if isinstance(result.get("param"), str) else None
    provider_code = payload.get("code")
    result["code"] = provider_code if isinstance(provider_code, (str, int)) else None
    return result



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


def _gemini_usage(usage: Usage | None, *, preserve_source: bool = False) -> dict[str, Any] | None:
    if usage is None:
        return None
    # Canonical input_tokens is cache-INCLUSIVE (H2) — matching Gemini's own
    # convention (promptTokenCount includes cachedContentTokenCount).
    # Detail keys emit only when non-zero (symmetric with the non-stream
    # formatter; zero-valued detail counts are not wire facts).
    payload: dict[str, Any] = {
        "promptTokenCount": usage.input_tokens,
        "candidatesTokenCount": usage.output_tokens,
        "totalTokenCount": usage.total_tokens,
    }
    if usage.cache_read_tokens:
        payload["cachedContentTokenCount"] = usage.cache_read_tokens
    if usage.reasoning_tokens:
        payload["thoughtsTokenCount"] = usage.reasoning_tokens
    tool_prompt = (usage.extra or {}).get("tool_use_prompt_tokens")
    if tool_prompt:
        # The tool-prompt bucket is a real Gemini usage member; the stream
        # must not drop it (non-stream already emits it).
        payload["toolUsePromptTokenCount"] = int(tool_prompt)
    if preserve_source and isinstance(usage.raw, dict):
        # Same-protocol stream replay of the raw detail arrays and tier
        # fields (mirror of the non-stream formatter).
        for key in (
            "promptTokensDetails",
            "cacheTokensDetails",
            "candidatesTokensDetails",
            "toolUsePromptTokensDetails",
            "trafficType",
            "serviceTier",
        ):
            if key in usage.raw and key not in payload:
                payload[key] = deepcopy(usage.raw[key])
    return payload


def _is_json(value: str) -> bool:
    try:
        json.loads(value)
    except json.JSONDecodeError:
        return False
    return True