# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Neutral-event stream operations (W5a/W5b/W5c/W5d).

All provider streams are consumed as ``UnifiedStreamEvent`` objects:

- native protocol execution yields events directly (single parse per frame);
- LiteLLM/custom chat-wire streams pass through :class:`ChatWireStreamAdapter`,
  which parses each chunk once and normalizes chat finish semantics.

:class:`NeutralStreamPipeline` is the operational layer: retry/error gating,
usage and cost accounting, session-anchor evidence, stream metrics, TTFB/stall
enforcement, heartbeat emission, client-disconnect handling, and completion
gating all run on neutral events. Client-protocol formatting happens exactly
once, at the tail, via ``format_canonical_stream_event``. There is no Chat SSE
intermediate between provider parsing and client formatting.
"""

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field, replace
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from ..core.errors import StreamedAPIError, CredentialNeedsReauthError
from ..protocols.streaming import format_canonical_stream_event, stream_format_state
from ..protocols.types import ProtocolContext, UnifiedStreamEvent, Usage, serialize_value
from ..streaming import StreamEvent, StreamMonitor
from ..streaming.relay import RelayStreamItem, StreamRepairState
from ..streaming.transport import SSEStreamFormatter
from ..usage.accounting import UsageRecord, extract_usage_record
from ..usage.costs import CostBreakdown, CostCalculator

__all__ = [
    "ChatWireStreamAdapter",
    "NeutralStreamPipeline",
    "StreamUsageTracker",
    "RelayStreamItem",
    "StreamRepairState",
]

if TYPE_CHECKING:
    from ..usage.manager import CredentialContext

lib_logger = logging.getLogger("rotator_library")

_CHAT_TERMINAL_EVENT_TYPES = {
    "done",
    "message_stop",
    "response.completed",
    "response.failed",
    "response.incomplete",
    "completed",
}


def _is_terminal_event(event: UnifiedStreamEvent) -> bool:
    return event.type in _CHAT_TERMINAL_EVENT_TYPES


def _event_has_meaningful_usage(event: UnifiedStreamEvent) -> bool:
    usage = event.usage
    if usage is None:
        return False
    if isinstance(usage, dict):
        return any(
            isinstance(value, (int, float)) and value > 0
            for key, value in usage.items()
            if key in {"prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens", "input_tokens", "output_tokens"}
        )
    for attr in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens"):
        value = getattr(usage, attr, 0)
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


def _completion_signal(event: UnifiedStreamEvent) -> bool:
    """Completion-evidence gate (phase-11 contract) on neutral events.

    A provider completion signal is an explicit terminal event, a usage-backed
    final event, or a finish reason paired with usage. Bare iterator EOF is a
    transport fact and never satisfies this gate.
    """

    if _is_terminal_event(event):
        return True
    if event.stop_reason and _event_has_meaningful_usage(event):
        return True
    if _event_has_meaningful_usage(event) and event.delta is None and event.message is None:
        return True
    return False


class StreamUsageTracker:
    """Accumulate normalized usage from neutral events and chat-wire cost frames."""

    def __init__(self, model: str, provider: Optional[str] = None) -> None:
        self.model = model
        self.provider = provider
        self.usage_record = UsageRecord(source="stream", model=model, provider=provider)

    def merge_event(self, event: UnifiedStreamEvent, *, source: str = "stream_event") -> None:
        usage = getattr(event, "usage", None)
        if usage is None:
            return
        candidate = extract_usage_record(
            serialize_value(event),
            provider=self.provider,
            model=self.model,
            source=source,
        )
        self.usage_record = self._reduce(self.usage_record, candidate)

    def merge_usage_payload(self, payload: Optional[Dict[str, Any]], *, source: str = "stream_final_chunk") -> None:
        if not isinstance(payload, dict):
            return
        usage_dict = payload.get("usage") if isinstance(payload.get("usage"), dict) else None
        if usage_dict is not None:
            # Exclusive-reasoning providers (reasoning outside completion) must
            # be normalized to the inclusive convention before accounting.
            from ..core.utils import normalize_usage_for_response

            normalize_usage_for_response(usage_dict, self.model)
        candidate = extract_usage_record(payload, provider=self.provider, model=self.model, source=source)
        self.usage_record = self._reduce(self.usage_record, candidate)

    def merge_cost_record(self, cost_record: UsageRecord) -> None:
        if cost_record.provider_reported_cost is None:
            return
        self.usage_record = replace(
            self.usage_record,
            provider_reported_cost=cost_record.provider_reported_cost,
            cost_currency=cost_record.cost_currency,
            cost_source=cost_record.cost_source,
        )

    def adopt(self, record: Optional[UsageRecord]) -> None:
        """Adopt a wire-exhaustive record (native executor usage) when richer."""

        if record is None:
            return
        self.usage_record = self._reduce(self.usage_record, record)

    @staticmethod
    def _reduce(base: UsageRecord, candidate: UsageRecord) -> UsageRecord:
        """Hybrid merge: a valueless late record never zeroes accumulation.

        A provider's trailing ``{"usage": {}}`` decodes to an all-zero
        record; REPLACE semantics would wipe everything accumulated from
        earlier frames. Port of the native executor's token-value guard:
        a candidate without token values keeps the base (cost carries
        over); a candidate WITH values wins, with base cost preserved.
        """

        if candidate is None:
            return base
        if not _record_has_token_values(candidate):
            merged = base
        else:
            merged = candidate
        if merged.provider_reported_cost is None:
            for source in (base, candidate):
                if source is not None and source.provider_reported_cost is not None:
                    merged = replace(
                        merged,
                        provider_reported_cost=source.provider_reported_cost,
                        cost_currency=source.cost_currency,
                        cost_source=source.cost_source,
                    )
                    break
        return merged


def _record_has_token_values(record: UsageRecord) -> bool:
    for attr in (
        "input_tokens",
        "completion_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
    ):
        value = getattr(record, attr, None)
        if isinstance(value, (int, float)) and value > 0:
            return True
    return False


class ChatWireStreamAdapter:
    """Parse chat-wire stream chunks (LiteLLM objects/dicts) into neutral events.

    Responsibilities (event-space equivalents of the legacy chunk handler):

    - one parse per chunk via the openai_chat protocol adapter;
    - in-band provider errors raise ``StreamedAPIError`` before formatting;
    - intermediate finish reasons are held back and emitted only with the
      completion signal, with tool-call finish taking priority;
    - provider-reported cost siblings and ``: cost`` / ``event: cost`` frames
      merge into the usage tracker;
    - DiffusionGemma-style deltas that mix reasoning and final text are split.
    """

    def __init__(self, model: str, repair_state: Optional[StreamRepairState] = None) -> None:
        self.model = model
        self._seen_tool_calls = False
        self._held_reason: Optional[str] = None
        self._held_reason_canonical: Optional[str] = None
        self.repair_state = repair_state if repair_state is not None else StreamRepairState()

    async def events(
        self,
        stream: AsyncIterator[Any],
        usage: StreamUsageTracker,
    ) -> AsyncGenerator[UnifiedStreamEvent, None]:
        from ..protocols import get_protocol
        from .streaming import StreamingHandler, _usage_record_from_sse_cost_chunk

        chat = get_protocol("openai_chat")
        try:
            async for chunk in stream:
                if isinstance(chunk, (str, bytes)):
                    # Custom plugins may still yield chat SSE strings; parse
                    # each data frame once through the same chunk path.
                    text = chunk.decode() if isinstance(chunk, bytes) else chunk
                    stripped = text.strip()
                    if not stripped:
                        continue
                    if stripped in {"[DONE]", "data: [DONE]"}:
                        yield self._terminal_event()
                        return
                    for payload_str in StreamingHandler._sse_data_payloads(text):
                        if payload_str == "[DONE]":
                            yield self._terminal_event()
                            return
                        try:
                            payload = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(payload, dict):
                            for event in self._dict_chunk_events(payload, chat, usage):
                                yield event
                    cost_record = _usage_record_from_sse_cost_chunk(text, model=self.model)
                    if cost_record.provider_reported_cost is not None:
                        usage.merge_cost_record(cost_record)
                    continue
                chunk_dict = self._as_dict(chunk)
                if chunk_dict is None:
                    continue
                for event in self._dict_chunk_events(chunk_dict, chat, usage):
                    yield event
        finally:
            # Closing this adapter (client disconnect, timeout, rotation) must
            # also close the wrapped provider stream, not just this generator.
            closer = getattr(stream, "aclose", None) or getattr(stream, "close", None)
            if closer is not None:
                try:
                    result = closer()
                    if result is not None and hasattr(result, "__await__"):
                        await result
                except Exception as exc:
                    lib_logger.debug("Failed to close upstream chat-wire stream: %s", exc)

    def _dict_chunk_events(
        self,
        chunk_dict: Dict[str, Any],
        chat: Any,
        usage: StreamUsageTracker,
    ) -> List[UnifiedStreamEvent]:
        from .streaming import StreamingHandler

        error_payload = StreamingHandler._in_band_error_payload(chunk_dict)
        if error_payload is not None:
            raise StreamedAPIError(
                str(error_payload.get("message") or error_payload.get("type") or "Provider stream failed"),
                data={"error": error_payload},
            )

        events: List[UnifiedStreamEvent] = []
        for piece in self._split_mixed_reasoning(chunk_dict):
            for event in self._events_for_chunk(piece, chat):
                if event is None:
                    continue
                if event.type == "error":
                    error = event.error if isinstance(event.error, dict) else {"message": str(event.error)}
                    raise StreamedAPIError(
                        str(error.get("message") or error.get("type") or "Provider stream failed"),
                        data={"error": dict(error)},
                    )
                usage.merge_usage_payload(_usage_with_cost_siblings(piece))
                events.append(event)
        return events

    def _events_for_chunk(self, chunk_dict: Dict[str, Any], chat: Any) -> List[UnifiedStreamEvent]:
        # Plural parse: one wire chunk may carry several choices (n>1) and
        # every candidate must survive to the client.
        events: List[UnifiedStreamEvent] = list(chat.parse_stream_events(chunk_dict))
        if not events:
            return []
        if any(
            item.delta is not None and getattr(item.delta, "tool_calls", None)
            for item in events
        ):
            self._seen_tool_calls = True
            self.repair_state.tools_seen = True

        # Repair evidence: the provider's own reasons (per choice) are
        # recorded for the pipeline tail even when held back.
        for item in events:
            if item.stop_reason:
                self._held_reason_canonical = item.stop_reason
                self.repair_state.last_provider_reason = item.stop_reason
                output_index = getattr(item, "output_index", None)
                try:
                    choice_key = int(output_index) if output_index is not None else 0
                except (TypeError, ValueError):
                    choice_key = 0
                self.repair_state.held_reasons[choice_key] = item.stop_reason

        if _is_terminal_event(events[0]):
            return [self._with_final_reason(events[0]), *events[1:]]

        first = events[0]
        delta_message = first.delta
        meaningful_usage = _event_has_meaningful_usage(first)
        finish_seen = first.stop_reason is not None
        if finish_seen and not meaningful_usage:
            # Intermediate finish frame: hold back every choice's reason,
            # keep every candidate's delta flowing (never drop siblings).
            kept = [replace(item, stop_reason=None) for item in events if item.delta is not None]
            return kept or []

        if meaningful_usage and (delta_message is None or finish_seen):
            # Usage-bearing final frame: every sibling survives, each with
            # its own repaired reason (provider's own reason wins).
            return [self._with_final_reason(item) for item in events]

        if delta_message is None and first.message is None and not meaningful_usage:
            return []

        if first.stop_reason is not None and not meaningful_usage and delta_message is not None:
            return [replace(item, stop_reason=None) for item in events]

        return events

    def _with_final_reason(self, event: UnifiedStreamEvent) -> UnifiedStreamEvent:
        # Provider's own final-frame reason wins; tools-seen only fills in
        # when the provider never stated one (G4 ruling killed the override).
        if event.stop_reason:
            return event
        if self._held_reason_canonical:
            return replace(event, stop_reason=self._held_reason_canonical)
        if self._seen_tool_calls:
            return replace(event, stop_reason="tool_calls")
        return event

    def _terminal_event(self) -> UnifiedStreamEvent:
        reason = self.repair_state.repaired_reason(None) or "stop"
        return UnifiedStreamEvent(
            type="done",
            source_protocol="openai_chat",
            native_type="done",
            stop_reason=reason,
        )

    @staticmethod
    def _as_dict(chunk: Any) -> Optional[Dict[str, Any]]:
        if isinstance(chunk, dict):
            return chunk
        if hasattr(chunk, "model_dump"):
            return chunk.model_dump()
        if hasattr(chunk, "dict"):
            return chunk.dict()
        return None

    def _split_mixed_reasoning(self, chunk_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split deltas that mix reasoning and final text (DiffusionGemma)."""

        choices = chunk_dict.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            return [chunk_dict]
        delta = choices[0].get("delta")
        if not isinstance(delta, dict):
            return [chunk_dict]
        content = delta.get("content")
        reasoning_fields = [
            field
            for field in ("reasoning_content", "reasoning")
            if isinstance(delta.get(field), str) and delta[field]
        ]
        if not isinstance(content, str) or not content or not reasoning_fields:
            return [chunk_dict]

        import copy

        reasoning_piece = copy.deepcopy(chunk_dict)
        reasoning_delta = reasoning_piece["choices"][0]["delta"]
        for field in ("content", "tool_calls", "function_call", "audio", "refusal"):
            reasoning_delta.pop(field, None)
        reasoning_piece["choices"][0]["finish_reason"] = None
        reasoning_piece.pop("usage", None)

        content_piece = copy.deepcopy(chunk_dict)
        content_delta = content_piece["choices"][0]["delta"]
        for field in reasoning_fields:
            content_delta.pop(field, None)

        return [reasoning_piece, content_piece]


class NeutralStreamPipeline:
    """Operational stream layer over neutral events with client formatting tail.

    G4 conditional re-serialization: when ``relay_eligible`` is set and a
    transport frame arrives with raw wire text (``RelayStreamItem``), the
    provider's bytes are forwarded to the client untouched while the
    observation side (usage, anchors, metrics, repair evidence) runs on the
    parsed copy. Any edit signal (hook/adapter modification, error frames,
    repair needs) permanently disengages the relay for the stream — the
    formatter path takes over. Protocol equality alone never implies relay.
    """

    def __init__(
        self,
        *,
        client_protocol_name: str,
        protocol_context: ProtocolContext,
        model: str,
        request: Optional[Any] = None,
        cred_context: Optional["CredentialContext"] = None,
        skip_cost_calculation: bool = False,
        response_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        success_callback: Optional[Callable[[], None]] = None,
        transaction_logger: Optional[Any] = None,
        repair_state: Optional[StreamRepairState] = None,
        client_include_usage: Optional[bool] = None,
        relay_eligible: bool = False,
        deadline: Optional[float] = None,
        provider_plugin: Any = None,
    ) -> None:
        self.client_protocol_name = client_protocol_name
        self.protocol_context = protocol_context
        self.model = model
        self.request = request
        self.cred_context = cred_context
        self.skip_cost_calculation = skip_cost_calculation
        self.response_callback = response_callback
        self.success_callback = success_callback
        self.transaction_logger = transaction_logger
        self.repair_state = repair_state if repair_state is not None else StreamRepairState()
        self.client_include_usage = client_include_usage
        self.relay_eligible = relay_eligible
        self.deadline = deadline
        self.provider_plugin = provider_plugin
        self.usage = StreamUsageTracker(model, provider=protocol_context.provider)

    async def run(
        self,
        event_source: AsyncIterator[UnifiedStreamEvent],
        *,
        usage_provider: Optional[Callable[[], Optional[UsageRecord]]] = None,
    ) -> AsyncGenerator[str, None]:
        from ..config.experimental import get_stream_runtime_settings
        from .streaming import StreamBuffer

        stream_settings = get_stream_runtime_settings()
        formatter = SSEStreamFormatter()
        monitor = StreamMonitor(clock=time.monotonic)
        state = stream_format_state(self.protocol_context, self.client_protocol_name)
        # G4: the client's own usage-frame preference reaches the formatter
        # state (None preserves legacy always-emit behavior).
        state.include_usage = self.client_include_usage

        stream_completed = False
        completion_signaled = False
        upstream_closed = False
        stream_cancelled = False
        relay_active = self.relay_eligible
        last_heartbeat_at = monitor.metrics.started_at
        error_buffer = StreamBuffer()
        # G4: yield-suspend accounting — time spent blocked delivering to a
        # slow client must not charge the upstream stall/TTFB clocks.
        paused_at: Optional[float] = None
        paused_since_first_byte = 0.0
        paused_since_last_chunk = 0.0

        def _note_yield_suspension() -> None:
            nonlocal paused_at
            paused_at = time.monotonic()

        def _note_yield_resumed() -> None:
            nonlocal paused_at, paused_since_first_byte, paused_since_last_chunk
            if paused_at is not None:
                delta = time.monotonic() - paused_at
                paused_since_first_byte += delta
                paused_since_last_chunk += delta
                paused_at = None

        assistant_parts: List[str] = []
        tool_call_ids: List[str] = []
        tool_call_events: Dict[Tuple[int, int], Dict[str, str]] = {}

        self._log_lifecycle("stream_started", monitor, "started")

        stream_iterator = event_source.__aiter__()

        async def close_upstream(reason: str, *, force: bool = False) -> None:
            nonlocal upstream_closed
            if upstream_closed or (not force and not stream_settings.cancel_upstream_on_disconnect):
                return
            upstream_closed = True
            for candidate in (stream_iterator, event_source):
                try:
                    closer = getattr(candidate, "aclose", None)
                    if closer:
                        await closer()
                        self._log_lifecycle("stream_upstream_cancelled", monitor, "cancelled", {"reason": reason})
                        return
                    closer = getattr(candidate, "close", None)
                    if closer:
                        closer()
                        self._log_lifecycle("stream_upstream_cancelled", monitor, "cancelled", {"reason": reason})
                        return
                except Exception as exc:
                    lib_logger.debug("Failed to close upstream stream: %s", exc)
                    self._log_lifecycle("stream_upstream_close_failed", monitor, "error", {"reason": reason, "error_type": type(exc).__name__})
                    return

        try:
            while True:
                try:
                    _note_yield_resumed()
                    if self.request and await self.request.is_disconnected():
                        lib_logger.info(
                            "Client disconnected. Aborting stream for model %s.", self.model
                        )
                        break

                    next_task = asyncio.create_task(stream_iterator.__anext__())
                    try:
                        while True:
                            wait_seconds = _next_stream_wait_seconds(monitor, stream_settings, last_heartbeat_at, deadline=self.deadline)
                            wait_tasks = {next_task}
                            disconnect_task = None
                            if self.request is not None:
                                disconnect_task = asyncio.create_task(self.request.is_disconnected())
                                wait_tasks.add(disconnect_task)
                            done, _ = await asyncio.wait(wait_tasks, timeout=wait_seconds)
                            if disconnect_task is not None:
                                if disconnect_task in done and disconnect_task.result():
                                    stream_cancelled = True
                                    next_task.cancel()
                                    with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                                        await next_task
                                    await close_upstream("client_disconnect")
                                    return
                                if not disconnect_task.done():
                                    disconnect_task.cancel()
                                    with contextlib.suppress(asyncio.CancelledError):
                                        await disconnect_task
                            if next_task in done:
                                event = next_task.result()
                                break

                            timeout_error = _stream_timeout_error(
                                monitor,
                                stream_settings,
                                paused_since_first_byte=paused_since_first_byte,
                                paused_since_last_chunk=paused_since_last_chunk,
                                deadline=self.deadline,
                            )
                            if timeout_error:
                                next_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                                    await next_task
                                await close_upstream(timeout_error[0], force=True)
                                self._log_lifecycle(timeout_error[2], monitor, "error", {"error": timeout_error[1]})
                                raise StreamedAPIError(timeout_error[1]["message"], data={"error": timeout_error[1]})

                            if _heartbeat_due(monitor, stream_settings, last_heartbeat_at):
                                heartbeat = formatter.format_heartbeat()
                                last_heartbeat_at = time.monotonic()
                                self._log_lifecycle("stream_heartbeat", monitor, "heartbeat")
                                _note_yield_suspension()
                                yield heartbeat
                    except Exception:
                        if not next_task.done():
                            next_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError, StopAsyncIteration):
                                await next_task
                        raise

                    if monitor.metrics.first_byte_at is None:
                        self._log_lifecycle("stream_first_byte", monitor, "raw_chunk")

                    # Upstream progress resets the per-chunk pause discount —
                    # only client-side suspension since the last event counts.
                    paused_since_last_chunk = 0.0
                    if monitor.metrics.first_byte_at is None:
                        paused_since_first_byte = 0.0
                    error_buffer.reset()

                    # G4: sources may deliver transport frames (raw + parsed)
                    # or bare events.
                    if isinstance(event, RelayStreamItem):
                        if event.is_comment:
                            # Provider heartbeat evidence relays verbatim and
                            # counts as liveness for the stall detector.
                            monitor.record_event(
                                StreamEvent("metadata", protocol=self.client_protocol_name, data={"comment": True})
                            )
                            if relay_active and event.raw is not None:
                                _note_yield_suspension()
                                yield event.raw + "\n\n"
                            continue
                        frame_events = event.events
                        frame_raw = event.raw
                    else:
                        frame_events = [event]
                        frame_raw = None

                    terminal_seen = False
                    error_in_frame = False
                    for item_event in frame_events:
                        self.usage.merge_event(item_event)
                        self._collect_anchors(item_event, assistant_parts, tool_call_ids, tool_call_events)
                        if _completion_signal(item_event):
                            completion_signaled = True
                        if item_event.type == "error":
                            error_in_frame = True
                        if _is_terminal_event(item_event):
                            terminal_seen = True

                    wire_event = StreamEvent(
                        "message" if any(self._event_visible(e) for e in frame_events) else "metadata",
                        protocol=self.client_protocol_name,
                        visible_output=any(self._event_visible(e) for e in frame_events),
                    )
                    first_visible = (
                        wire_event.visible_output
                        and monitor.metrics.first_visible_output_at is None
                    )
                    monitor.record_event(wire_event)
                    if first_visible:
                        self._log_lifecycle("stream_first_visible_output", monitor, "message")

                    # Relay decision (per frame, sticky disengage): raw bytes
                    # exist, no edit signal, no error, no repair need on the
                    # frame itself. Hook edits flip repair_state.edited_by_hook.
                    if (
                        relay_active
                        and frame_raw is not None
                        and not error_in_frame
                        and not self.repair_state.edited_by_hook
                    ):
                        self._trace_frame(frame_raw)
                        _note_yield_suspension()
                        yield frame_raw + "\n\n"
                        if terminal_seen:
                            # The provider's own terminal bytes were relayed —
                            # the tail must not synthesize a second one.
                            state.terminal = True
                            stream_completed = True
                            break
                        continue

                    if error_in_frame or self.repair_state.edited_by_hook:
                        # Disengage relay permanently: edits/errors require the
                        # formatter path for the rest of the stream.
                        relay_active = False

                    for item_event in frame_events:
                        if item_event.type == "error":
                            error = item_event.error if isinstance(item_event.error, dict) else {"message": str(item_event.error)}
                            raise StreamedAPIError(
                                str(error.get("message") or error.get("type") or "Provider stream failed"),
                                data={"error": dict(error)},
                            )
                        for frame in format_canonical_stream_event(
                            item_event,
                            self.client_protocol_name,
                            self.protocol_context,
                            state=state,
                        ):
                            self._trace_frame(frame)
                            _note_yield_suspension()
                            yield frame

                    if terminal_seen:
                        stream_completed = True
                        break

                except StopAsyncIteration:
                    stream_completed = True
                    break

                except CredentialNeedsReauthError as e:
                    if self.cred_context:
                        from ..error_handler import classify_error

                        self.cred_context.mark_failure(classify_error(e))
                    raise StreamedAPIError("Credential needs re-authentication", data=e)

                except json.JSONDecodeError as e:
                    error_buffer.append(str(e))
                    if error_buffer.is_complete:
                        raise StreamedAPIError("Provider error", data=error_buffer.content)
                    continue

                except Exception as e:
                    error_str = str(e)
                    error_buffer.append(error_str)
                    if error_buffer.is_complete:
                        if self.cred_context:
                            from ..error_handler import classify_error

                            self.cred_context.mark_failure(classify_error(e))
                        raise StreamedAPIError("Provider error in stream", data=error_buffer.content)
                    extracted = _try_extract_error(e, error_buffer.content)
                    if extracted:
                        if self.cred_context:
                            from ..error_handler import classify_error

                            self.cred_context.mark_failure(classify_error(e))
                        raise StreamedAPIError("Provider error in stream", data=extracted)
                    monitor.metrics.error_count += 1
                    await close_upstream("stream_exception", force=True)
                    raise

        except StreamedAPIError:
            await close_upstream("streamed_api_error", force=True)
            raise

        except asyncio.CancelledError:
            stream_cancelled = True
            monitor.cancel()
            self._log_lifecycle("stream_cancelled", monitor, "cancelled", {"reason": "task_cancelled"})
            await close_upstream("task_cancelled", force=True)
            raise

        finally:
            if stream_completed:
                if state.terminal is False:
                    # G4 repair tail: missing finish/usage is recoverable,
                    # never fatal. Reason ladder: provider's own final-frame
                    # reason > held intermediate > tools-seen > stop. Usage is
                    # always present (zeros when unknown) so downstream
                    # parsers never meet a missing field.
                    repaired_reason = self.repair_state.repaired_reason(state.stop_reason)
                    terminal_event = UnifiedStreamEvent(
                        type="done",
                        source_protocol=self.client_protocol_name,
                        native_type="done",
                        stop_reason=repaired_reason,
                        usage=state.usage if state.usage is not None else Usage(),
                    )
                    for frame in format_canonical_stream_event(
                        terminal_event,
                        self.client_protocol_name,
                        self.protocol_context,
                        state=state,
                    ):
                        self._trace_frame(frame)
                        yield frame

                if usage_provider is not None:
                    self.usage.adopt(usage_provider())

                # Stream-side conversion drops (foreign builtins omitted,
                # refusal degradations, media drops) accumulate on the
                # formatter state — traced once at the tail (streams carry
                # no in-band summary header by construction).
                if state.warnings:
                    for warning in state.warnings:
                        lib_logger.info(
                            "stream conversion warning: [%s] %s (field=%s, target=%s)",
                            warning.code,
                            warning.message,
                            warning.field,
                            warning.target_protocol,
                        )
                    self._log_lifecycle(
                        "stream_conversion_warnings",
                        monitor,
                        "completed",
                        {"warnings": [vars(w) for w in state.warnings]},
                    )

                if self.cred_context:
                    cost_breakdown = self._cost_breakdown(self.usage.usage_record)
                    self._log_usage_accounting(self.usage.usage_record, cost_breakdown)
                    record = self.usage.usage_record
                    self.cred_context.mark_success(
                        prompt_tokens=record.prompt_tokens_for_mark_success,
                        completion_tokens=record.completion_tokens,
                        thinking_tokens=record.reasoning_tokens,
                        prompt_tokens_cache_read=record.cache_read_tokens,
                        prompt_tokens_cache_write=record.cache_write_tokens,
                        approx_cost=cost_breakdown.total_cost,
                    )
                if self.success_callback:
                    self.success_callback()

                if (
                    self.response_callback
                    and completion_signaled
                    and (assistant_parts or tool_call_ids)
                ):
                    # Phase-11 gate: anchors are recorded only after an explicit
                    # provider completion signal; bare EOF never qualifies.
                    self.response_callback(
                        {
                            "messages": [
                                {
                                    "role": "assistant",
                                    "content": "".join(assistant_parts),
                                    "tool_calls": _assembled_tool_calls(
                                        tool_call_ids,
                                        tool_call_events,
                                    ),
                                }
                            ]
                        }
                    )

                monitor.complete()
                self._log_lifecycle("stream_completed", monitor, "completed")
                self._log_lifecycle("stream_metrics_final", monitor, "metadata")

            elif self.request and await self.request.is_disconnected():
                stream_cancelled = True
                monitor.cancel()
                self._log_lifecycle("stream_cancelled", monitor, "cancelled")
                await close_upstream("client_disconnect")
            elif stream_cancelled:
                await close_upstream("stream_cancelled", force=True)

    # ------------------------------------------------------------------
    # helpers

    @staticmethod
    def _event_visible(event: UnifiedStreamEvent) -> bool:
        """Read the REAL neutral attributes (G4): text lives in content
        blocks, reasoning is a list of ReasoningBlock — the old
        ``delta.text``/``delta.reasoning.text`` reads never existed, so
        TTFT/visible metrics only ever fired for tool calls. Reasoning-only
        deltas count as visible output (operator ruling: they lock the
        fallback route exactly like text)."""

        if _is_terminal_event(event):
            return False
        if event.type == "error":
            return False
        delta = event.delta or event.message
        if delta is None:
            return False
        for block in getattr(delta, "content", None) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                return True
        for reasoning_block in getattr(delta, "reasoning", None) or []:
            text = getattr(reasoning_block, "text", None)
            if isinstance(text, str) and text:
                return True
        tool_calls = getattr(delta, "tool_calls", None)
        return bool(tool_calls)

    def _collect_anchors(
        self,
        event: UnifiedStreamEvent,
        assistant_parts: List[str],
        tool_call_ids: List[str],
        tool_call_events: Dict[Tuple[int, int], Dict[str, str]],
    ) -> None:
        delta = event.delta or event.message
        if delta is None:
            return
        choice_key = 0
        output_index = getattr(event, "output_index", None)
        try:
            choice_key = int(output_index) if output_index is not None else 0
        except (TypeError, ValueError):
            choice_key = 0
        for block in getattr(delta, "content", None) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                assistant_parts.append(text)
        for position, call in enumerate(getattr(delta, "tool_calls", None) or []):
            call_id = getattr(call, "id", None)
            if call_id and str(call_id) not in tool_call_ids:
                tool_call_ids.append(str(call_id))
            index = getattr(call, "index", None)
            try:
                event_index = int(index) if index is not None else position
            except (TypeError, ValueError):
                event_index = position
            entry = tool_call_events.setdefault((choice_key, event_index), {})
            if call_id:
                entry["id"] = str(call_id)
            name = getattr(call, "name", None)
            if name:
                entry["name"] = str(name)
            # ToolCall carries .arguments directly (the old .function read
            # never existed — streamed tool arguments were erased).
            arguments = getattr(call, "arguments", None)
            if isinstance(arguments, str) and arguments:
                entry["arguments"] = _merge_streamed_tool_arguments(
                    entry.get("arguments", ""),
                    arguments,
                )

    def _cost_breakdown(self, usage_record: UsageRecord) -> CostBreakdown:
        if self.skip_cost_calculation:
            return CostBreakdown(pricing_source="skipped")
        # G4: provider plugin pricing applies on streams exactly like the
        # non-streaming path (the bare calculator skipped it entirely).
        return CostCalculator(provider_plugin=self.provider_plugin).calculate(
            usage_record, model=self.model, provider=self.protocol_context.provider
        )

    def _log_usage_accounting(self, usage_record: UsageRecord, cost_breakdown: CostBreakdown) -> None:
        if not self.transaction_logger:
            return
        try:
            self.transaction_logger.log_transform_pass(
                "usage_accounting_summary",
                {"usage": usage_record.to_dict(), "cost": cost_breakdown.to_dict()},
                direction="metadata",
                stage="final",
                transport="sse",
                metadata={
                    "source": usage_record.source,
                    "pricing_source": cost_breakdown.pricing_source,
                },
                snapshot=False,
            )
        except Exception as exc:
            lib_logger.debug("Stream usage trace failed: %s", exc)

    def _trace_frame(self, frame: str) -> None:
        if not self.transaction_logger:
            return
        try:
            self.transaction_logger.log_transform_pass(
                "formatted_client_stream_event",
                frame,
                direction="stream_out",
                stage="final",
                protocol=self.client_protocol_name,
                transport="sse",
                snapshot=False,
            )
        except Exception as exc:
            lib_logger.debug("Stream frame trace failed: %s", exc)

    def _log_lifecycle(
        self,
        pass_name: str,
        monitor: StreamMonitor,
        event_type: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.transaction_logger:
            return
        try:
            event = StreamEvent(event_type, protocol=self.client_protocol_name, data=data or {})
            self.transaction_logger.log_transform_pass(
                pass_name,
                {"event": event.to_dict(), "metrics": monitor.metrics.to_dict()},
                direction="stream",
                stage="client",
                protocol=self.client_protocol_name,
                transport="sse",
                metadata={"event_type": event_type},
                snapshot=False,
            )
        except Exception as exc:
            lib_logger.debug("Stream lifecycle trace failed: %s", exc)


def _next_stream_wait_seconds(
    monitor: StreamMonitor,
    settings: Any,
    last_heartbeat_at: float,
    deadline: Optional[float] = None,
) -> Optional[float]:
    candidates: List[float] = []
    now = time.monotonic()
    if deadline is not None:
        candidates.append(max(0.0, deadline - now))
    if settings.heartbeat_seconds:
        candidates.append(max(0.0, last_heartbeat_at + settings.heartbeat_seconds - now))
    if settings.ttfb_timeout_seconds and monitor.metrics.first_byte_at is None:
        candidates.append(max(0.0, monitor.metrics.started_at + settings.ttfb_timeout_seconds - now))
    if settings.stall_timeout_seconds and monitor.metrics.first_byte_at is not None:
        last_chunk_at = monitor.metrics.last_chunk_at or monitor.metrics.first_byte_at
        candidates.append(max(0.0, last_chunk_at + settings.stall_timeout_seconds - now))
    return min(candidates) if candidates else None


def _heartbeat_due(monitor: StreamMonitor, settings: Any, last_heartbeat_at: float) -> bool:
    if not settings.heartbeat_seconds:
        return False
    return time.monotonic() - last_heartbeat_at >= settings.heartbeat_seconds


def _stream_timeout_error(
    monitor: StreamMonitor,
    settings: Any,
    *,
    paused_since_first_byte: float = 0.0,
    paused_since_last_chunk: float = 0.0,
    deadline: Optional[float] = None,
) -> Optional[Tuple[str, Dict[str, Any], str]]:
    now = time.monotonic()
    if deadline is not None:
        remaining = deadline - now
        if remaining <= 0:
            return (
                "deadline_exceeded",
                {
                    "message": "Request deadline exceeded while streaming",
                    "type": "api_connection",
                    "details": {"timeout_type": "deadline"},
                },
                "stream_deadline_exceeded",
            )
    if settings.ttfb_timeout_seconds and monitor.metrics.first_byte_at is None:
        if now - monitor.metrics.started_at - paused_since_first_byte >= settings.ttfb_timeout_seconds:
            return (
                "ttfb_timeout",
                {
                    "message": "Stream timed out before first byte",
                    "type": "api_connection",
                    "details": {"timeout_type": "ttfb", "timeout_seconds": settings.ttfb_timeout_seconds},
                },
                "stream_ttfb_timeout",
            )
    if settings.stall_timeout_seconds and monitor.metrics.first_byte_at is not None:
        last_chunk_at = monitor.metrics.last_chunk_at or monitor.metrics.first_byte_at
        if last_chunk_at is not None and now - last_chunk_at - paused_since_last_chunk >= settings.stall_timeout_seconds:
            return (
                "stall_timeout",
                {
                    "message": "Stream stalled while waiting for provider data",
                    "type": "api_connection",
                    "details": {"timeout_type": "stall", "timeout_seconds": settings.stall_timeout_seconds},
                },
                "stream_stall_timeout",
            )
    return None


def _try_extract_error(exception: Exception, buffer: str) -> Optional[Dict[str, Any]]:
    import codecs
    import re

    error_str = str(exception)
    match = re.search(r"b'(\{.*\})'", error_str, re.DOTALL)
    if match:
        try:
            decoded = codecs.decode(match.group(1), "unicode_escape")
            return json.loads(decoded)
        except (json.JSONDecodeError, ValueError):
            pass
    if "Received chunk:" in error_str:
        chunk = error_str.split("Received chunk:")[-1].strip()
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            pass
    if buffer:
        try:
            return json.loads(buffer)
        except json.JSONDecodeError:
            pass
    return None


def _usage_with_cost_siblings(chunk_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    usage = chunk_dict.get("usage")
    if not isinstance(usage, dict):
        return None
    merged = dict(usage)
    for key in ("cost_details", "cost", "total_cost", "estimated_cost", "provider_reported_cost", "request_cost_usd", "currency", "costMetadata"):
        if key in chunk_dict and key not in merged:
            merged[key] = chunk_dict[key]
    return {"usage": merged}


def _usage_from_sse_string(chunk: str) -> Optional[Dict[str, Any]]:
    data_lines: List[str] = []
    for line in chunk.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("data:"):
            data_lines.append(stripped[5:].strip())
    if not data_lines:
        return None
    payload = "\n".join(data_lines).strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    merged = dict(usage)
    for key in ("cost_details", "cost", "total_cost", "estimated_cost", "provider_reported_cost", "request_cost_usd", "currency", "costMetadata"):
        if key in data and key not in merged:
            merged[key] = data[key]
    return {"usage": merged}


def _merge_streamed_tool_arguments(current: str, incoming: str) -> str:
    if not current:
        return incoming
    if incoming.startswith(current):
        return incoming
    if current.startswith(incoming):
        return current
    return current + incoming


def _assembled_tool_calls(
    tool_call_ids: List[str],
    tool_call_events: Dict[Tuple[int, int], Dict[str, str]],
) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    emitted_ids: set = set()
    for index in sorted(tool_call_events):
        event = tool_call_events[index]
        call_id = event.get("id")
        if not call_id:
            continue
        call: Dict[str, Any] = {"id": call_id}
        if event.get("name"):
            call["function"] = {
                "name": event["name"],
                "arguments": event.get("arguments", ""),
            }
        calls.append(call)
        emitted_ids.add(call_id)
    calls.extend(
        {"id": call_id}
        for call_id in tool_call_ids
        if call_id not in emitted_ids
    )
    return calls
