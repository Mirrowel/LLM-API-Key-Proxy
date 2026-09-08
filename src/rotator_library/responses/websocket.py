# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""WebSocket Mode for the Responses API compatibility layer.

Official wire contract (developers.openai.com/api/docs/guides/websocket-mode,
pinned 2026-09-07 — the sequential revision: "a single WebSocket connection
can receive multiple response.create messages, but it runs them
sequentially"; a newer revision describes multiplexed stream_id lanes with
16 in-flight / 32 named lanes, which this implementation deliberately does
NOT adopt until the skew settles):

- The client opens one persistent connection (``wss://.../v1/responses``,
  ``Authorization: Bearer`` header) and starts every turn with a
  ``response.create`` frame whose payload mirrors the HTTP create body —
  except transport-specific fields (``stream``, ``background``, and the
  WebSocket-only ``stream_id`` / ``generate``).
- ``stream_id`` (1-256 chars of ``[A-Za-z0-9_.-]``) labels a lane; omitted
  means the implicit default lane. Named-lane server frames carry the same
  ``stream_id``. Turns on one connection process strictly in arrival order
  (FIFO) — one in-flight response at a time, no multiplexing.
- ``generate: false`` warms request state: the connection-local cache
  records the tools/instructions/input for the next turn and returns a
  chainable response ID with no model output.
- Continuation uses ``previous_response_id`` with incremental input. The
  previous-response state lives in a CONNECTION-LOCAL IN-MEMORY cache (the
  most recent responses), consulted before the global store — this is what
  keeps ``store=false`` / ZDR chains working. Failed turns evict the
  referenced parent id (stale state is never reused).
- Server frames are the SAME event objects the HTTP streaming API emits
  (``type`` + ``sequence_number`` + payload); terminal
  ``response.completed`` / ``response.failed`` / ``response.incomplete``
  close the turn — there is no ``[DONE]`` sentinel on this transport.
- Errors arrive as ``{"type":"error","status":...,"error":{...}}`` frames,
  including ``previous_response_not_found`` and
  ``websocket_connection_limit_reached`` after the configured connection
  lifetime (default 60 minutes).

In scope: response.create (+ warmup), named lanes with FIFO ordering,
connection-local continuation cache, store=true passthrough persistence,
spec-shaped error frames, connection lifetime limit, disconnect cleanup.

Deliberately NOT implemented (rejected or deferred, never silent):
- ``response.cancel`` — not in the official Responses WebSocket client
  vocabulary (it belongs to the Realtime API); frames of this type receive
  a clean ``invalid_request_error``.
- steering (``response.steer`` family) and multiplexed lanes — deferred
  until the spec revision settles; not accepted silently.
- Application-level keepalive frames — none exist in the spec; deployments
  behind idle-timeout proxies must rely on WebSocket protocol pings
  (uvicorn's ``ws_ping_interval``, default 20s).
- Binary frames — this transport is text-only (every documented frame is a
  JSON object); binary messages terminate the connection.
- Lifetime-limit interruption mid-turn — the limit fires between turns: an
  in-flight turn always runs to its terminal event, then the error frame
  and close arrive (sequential turns cannot be preempted without
  cancelling provider work mid-flight).
- Warmup inheritance of tools/instructions — clients resend tools each
  turn (the guide's own continuation examples do); warmup replays input
  items only.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, MutableMapping, Optional

from .streaming import ResponsesStreamEvent
from .types import StoredResponse, generate_response_id

# stream_id grammar per the official guide.
_STREAM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,256}$")

# Default connection lifetime (the documented service limit: 60 minutes).
DEFAULT_MAX_CONNECTION_SECONDS = 60 * 60

# Bounded connection-local cache: the guide keeps "the most recent" state;
# 32 ids cover deep tool loops without unbounded memory per connection.
_LOCAL_CACHE_MAX_ENTRIES = 32

# Turn-closing terminal events (no [DONE] sentinel on this transport).
_TERMINAL_EVENT_TYPES = {"response.completed", "response.failed", "response.incomplete"}

# Keepalive: covered by WebSocket protocol pings (see module docstring).


def error_frame(
    code: str,
    message: str,
    *,
    status: int = 400,
    param: Optional[str] = None,
    stream_id: Optional[str] = None,
    err_type: Optional[str] = None,
) -> dict[str, Any]:
    """Build a documented error frame.

    Shape per the guide's verbatim examples: ``{"type": "error", "status":
    ..., "error": {"code", "message", "param"?}}`` — the connection-limit
    example additionally carries ``error.type: "invalid_request_error"``;
    ``err_type`` emits that inner type when provided.
    """

    error: dict[str, Any] = {"code": code, "message": message}
    if err_type is not None:
        # Preserve the documented key order of the limit example.
        error = {"type": err_type, "code": code, "message": message}
    if param is not None:
        error["param"] = param
    frame: dict[str, Any] = {"type": "error", "status": status, "error": error}
    if stream_id:
        frame["stream_id"] = stream_id
    return frame


@dataclass
class ClientFrame:
    """One parsed client frame (response.create)."""

    payload: dict[str, Any] = field(default_factory=dict)
    stream_id: Optional[str] = None
    warmup: bool = False


@dataclass
class LaneState:
    """Per-lane bookkeeping on one connection."""

    stream_id: Optional[str] = None
    latest_response_id: Optional[str] = None


def parse_client_frame(raw: str | bytes | dict[str, Any]) -> ClientFrame | dict[str, Any]:
    """Parse one client frame into a turn request.

    Returns :class:`ClientFrame` on success or a spec-shaped error frame
    dict on rejection. Only ``response.create`` is in the official client
    vocabulary; anything else (including the Realtime-API ``response.cancel``)
    is rejected cleanly.
    """

    if isinstance(raw, (str, bytes)):
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            return error_frame("invalid_request_error", "frame is not valid JSON", err_type="invalid_request_error")
    else:
        payload = raw
    if not isinstance(payload, dict):
        return error_frame("invalid_request_error", "frame must be a JSON object", err_type="invalid_request_error")
    frame_type = payload.get("type")
    if frame_type is None:
        return error_frame("invalid_request_error", "frame requires a 'type' field", err_type="invalid_request_error")
    if frame_type != "response.create":
        return error_frame(
            "invalid_request_error",
            f"frame type {frame_type!r} is not part of the Responses WebSocket vocabulary (only response.create is supported)",
            err_type="invalid_request_error",
        )
    if "background" in payload and payload["background"]:
        # The events reference pins background as unsupported on this
        # transport — a TRUTHY background (the requested lifecycle) is
        # rejected; explicit `background: false` is a no-op mirroring the
        # HTTP body and strips like every other unused transport field.
        return error_frame(
            "invalid_request_error",
            "background mode is not supported on the WebSocket transport",
            param="background",
            err_type="invalid_request_error",
        )
    if payload.get("conversation") is not None and payload.get("previous_response_id") is not None:
        return error_frame(
            "invalid_request_error",
            "conversation and previous_response_id are mutually exclusive",
            param="conversation",
            err_type="invalid_request_error",
        )
    # Transport-specific fields never apply on this transport.
    body = {k: v for k, v in payload.items() if k not in {"type", "stream", "background", "generate"}}
    lane = body.pop("stream_id", None)
    if lane is None and "stream_id" in payload and payload["stream_id"] is None:
        # Explicit null is not omission — reject rather than silently
        # coercing to the default lane.
        return error_frame(
            "invalid_request_error",
            "stream_id must not be null; omit the field for the default lane",
            param="stream_id",
            err_type="invalid_request_error",
        )
    if lane == "":
        return error_frame(
            "invalid_request_error",
            "stream_id must not be empty; omit the field for the default lane",
            param="stream_id",
            err_type="invalid_request_error",
        )
    if lane is not None and not isinstance(lane, str):
        return error_frame("invalid_request_error", "stream_id must be a string", param="stream_id", err_type="invalid_request_error")
    if lane is not None and not _STREAM_ID_PATTERN.match(lane):
        return error_frame(
            "invalid_request_error",
            "stream_id must be 1-256 characters of letters, numbers, underscores, hyphens, and periods",
            param="stream_id",
            err_type="invalid_request_error",
        )
    return ClientFrame(payload=body, stream_id=lane, warmup=payload.get("generate") is False)


class ResponsesWebSocketFormatter:
    """Serialize transport-neutral events as WebSocket JSON frames.

    A frame IS the streaming event object (``type`` + ``sequence_number`` +
    payload fields) — the same objects the HTTP streaming API emits, with
    the lane's ``stream_id`` stamped on named lanes. SSE-only artifacts
    (comment heartbeats, the ``[DONE]`` sentinel) have no equivalent here
    and are dropped.
    """

    transport = "websocket"

    def format_event(self, event_name: str, payload: dict[str, Any]) -> str:
        frame = dict(payload) if isinstance(payload, dict) else {}
        frame.setdefault("type", event_name)
        return json.dumps(frame, ensure_ascii=False)

    def format_stream_event(self, event: ResponsesStreamEvent, *, stream_id: Optional[str] = None) -> Optional[str]:
        if event.heartbeat or event.event_name == "done":
            return None
        frame = dict(event.payload) if isinstance(event.payload, dict) else {}
        frame["type"] = event.event_name or frame.get("type") or "response.event"
        if stream_id:
            frame["stream_id"] = stream_id
        return json.dumps(frame, ensure_ascii=False)


class _LocalResponsesCache(OrderedDict):
    """Connection-local continuation cache (bounded, most-recent eviction).

    In-memory only — the guide's ZDR-compatible fast continuation path.
    The ResponsesService consults it before the global store for
    previous_response_id resolution and lineage walks, and stores each
    completed turn in it (even when ``store=false``).
    """

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > _LOCAL_CACHE_MAX_ENTRIES:
            self.popitem(last=False)


def _warmup_input_items(raw_input: Any) -> list[Any]:
    """Normalize warmup input for lineage replay — never raises.

    Mirrors the service's ``_input_items`` semantics: lists pass through,
    anything else wraps atomically (strings included); absent -> empty.
    """

    if isinstance(raw_input, list):
        return list(raw_input)
    if raw_input is None:
        return []
    return [raw_input]


def _is_previous_response_failure(exc: BaseException) -> bool:
    """Structured detection of continuation misses — never loose substrings.

    The service raises ``not_found_error`` ResponsesServiceErrors with
    "Previous/Response not found: <id>" messages (strict prefixes);
    provider-side failures that merely CONTAIN the phrase stay ordinary
    failures because their ``error_type`` differs.
    """

    if getattr(exc, "error_type", None) != "not_found_error":
        return False
    message = str(exc).lower()
    return message.startswith("previous response") or message.startswith("response not found")


def _failed_event_is_continuation_miss(response_obj: dict[str, Any]) -> bool:
    """Terminal response.failed miss detection (structured type + prefix)."""

    error = response_obj.get("error")
    if not isinstance(error, dict):
        return False
    code = str(error.get("code") or "")
    if code == "previous_response_not_found":
        return True
    if str(error.get("type") or "") != "not_found_error":
        return False
    message = str(error.get("message") or "").lower()
    return message.startswith("previous response") or message.startswith("response not found")


def _service_error_frame(exc: BaseException, stream_id: Optional[str]) -> dict[str, Any]:
    """Map a ResponsesServiceError to its spec-shaped frame.

    ``error.type`` mirrors the service's own classification verbatim (a
    502 upstream error stays an upstream error; only the guide-pinned
    examples carry fixed inner types). The continuation-miss frame matches
    the guide example exactly: no inner ``type``.
    """

    status = int(getattr(exc, "status_code", 400) or 400)
    err_type = str(getattr(exc, "error_type", None) or "invalid_request_error")
    message = str(exc)
    if _is_previous_response_failure(exc):
        return error_frame(
            "previous_response_not_found",
            message,
            status=400,
            param="previous_response_id",
            stream_id=stream_id,
        )
    return error_frame(err_type, message, status=status, stream_id=stream_id)


class ResponsesWebSocketSession:
    """Drive one WebSocket connection over the Responses service.

    Structure (per the gatekeeper design): ``run()`` reads frames and
    processes turns strictly sequentially — a turn must reach its terminal
    event before the next ``response.create`` is read (FIFO, matching the
    pinned sequential revision). Every turn's event generator is closed
    cooperatively (``aclose``) on completion, cancellation, or disconnect,
    so upstream provider streams never leak. The connection-local
    continuation cache is bounded and dies with this object.
    """

    def __init__(
        self,
        *,
        service: Any,
        client: Any,
        max_connection_seconds: float = DEFAULT_MAX_CONNECTION_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        transaction_logger_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._service = service
        self._client = client
        self._max_connection_seconds = max_connection_seconds
        self._clock = clock
        self._transaction_logger_factory = transaction_logger_factory
        self._lanes: dict[str, LaneState] = {}
        self.local_cache: MutableMapping[str, StoredResponse] = _LocalResponsesCache()
        self._formatter = ResponsesWebSocketFormatter()
        # NOTE: locally synthesized frames draw sequence numbers from the
        # shared module domain (see _next_sequence) — no session counter.

    @property
    def lanes(self) -> dict[str, LaneState]:
        return self._lanes

    # -- sequencing for locally synthesized frames --------------------------

    def _next_sequence(self) -> int:
        # Shared module domain (the same counter native-turn events use):
        # per-connection counters would collide with turn event sequences —
        # every frame on the connection must be monotonic together.
        from .streaming import next_sequence_value

        return next_sequence_value()

    def _lane(self, stream_id: Optional[str]) -> LaneState:
        key = stream_id or ""
        if key not in self._lanes:
            self._lanes[key] = LaneState(stream_id=stream_id)
        return self._lanes[key]

    # -- frame handling ------------------------------------------------------

    async def handle_frame(self, raw: str | bytes | dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Handle one client frame, yielding server frames as dicts."""

        parsed = parse_client_frame(raw)
        if isinstance(parsed, dict):
            yield parsed
            return
        if parsed.warmup:
            async for frame in self._warmup(parsed):
                yield frame
            return
        # GeneratorExit at OUR yield must unwind the turn generator too —
        # without this try/finally, closing this chain abandons _turn one
        # level down and its upstream aclose is deferred to GC.
        turn_gen = self._turn(parsed)
        try:
            async for frame in turn_gen:
                yield frame
        finally:
            await turn_gen.aclose()

    async def _warmup(self, frame: ClientFrame) -> AsyncGenerator[dict[str, Any], None]:
        """``generate: false`` — prepare request state, return a chainable id.

        State is connection-local ONLY (the guide's in-memory cache): the
        warmup body's input items persist in this session's cache with the
        scope the service resolves for this connection (``public`` — the
        WS transport carries no scope headers), so the next turn chaining
        from the returned id replays them through the standard lineage
        expansion. Tools/instructions are NOT inherited — the guide's own
        continuation examples resend tools every turn; clients do the same
        after warmup. Nothing is written to the global store or disk.
        """

        response_id = generate_response_id()
        stored = StoredResponse(
            id=response_id,
            model=str(frame.payload.get("model") or ""),
            status="completed",
            response={
                "id": response_id,
                "object": "response",
                "status": "completed",
                "model": str(frame.payload.get("model") or ""),
                "output": [],
            },
            request=dict(frame.payload),
            input_items=_warmup_input_items(frame.payload.get("input")),
            scope_key="public",
        )
        self.local_cache[response_id] = stored
        lane = self._lane(frame.stream_id)
        lane.latest_response_id = response_id
        warmup_frame: dict[str, Any] = {
            "type": "response.completed",
            "sequence_number": self._next_sequence(),
            "response": dict(stored.response),
        }
        if frame.stream_id:
            warmup_frame["stream_id"] = frame.stream_id
        yield warmup_frame

    async def _turn(self, frame: ClientFrame) -> AsyncGenerator[dict[str, Any], None]:
        """Run one response.create turn to its terminal event."""

        lane = self._lane(frame.stream_id)
        body = dict(frame.payload)
        previous_id = body.get("previous_response_id")
        transaction_logger = (
            self._transaction_logger_factory(str(body.get("model") or "unknown"))
            if self._transaction_logger_factory is not None
            else None
        )
        events: Optional[AsyncGenerator[ResponsesStreamEvent, None]] = None
        terminal_seen = False
        turn_error_status: Optional[int] = None
        try:
            events = self._service.stream_turn_events(
                body,
                self._client,
                transaction_logger=transaction_logger,
                local_cache=self.local_cache,
            )
            async for event in events:
                if not isinstance(event, ResponsesStreamEvent):
                    continue
                if event.heartbeat:
                    continue
                payload = dict(event.payload) if isinstance(event.payload, dict) else {}
                payload["type"] = event.event_name or payload.get("type") or "response.event"
                if frame.stream_id:
                    payload["stream_id"] = frame.stream_id
                response_obj = payload.get("response")
                if isinstance(response_obj, dict) and isinstance(response_obj.get("id"), str):
                    lane.latest_response_id = response_obj["id"]
                if (
                    event.event_name == "response.failed"
                    and isinstance(response_obj, dict)
                    and _failed_event_is_continuation_miss(response_obj)
                ):
                    # The service converts pre-stream failures into terminal
                    # response.failed events; the guide documents
                    # continuation misses as top-level error frames.
                    if isinstance(previous_id, str):
                        self.local_cache.pop(previous_id, None)
                    yield error_frame(
                        "previous_response_not_found",
                        str((response_obj.get("error") or {}).get("message") or "previous response not found"),
                        status=400,
                        param="previous_response_id",
                        stream_id=frame.stream_id,
                    )
                    terminal_seen = True
                    turn_error_status = 400
                    break
                yield payload
                if event.event_name in _TERMINAL_EVENT_TYPES:
                    terminal_seen = True
                    if event.event_name == "response.failed":
                        # Failure status derives from the failure payload when
                        # the provider classified it (4xx request classes),
                        # else 500 — never a blanket 500.
                        failure_payload = event.payload.get("response") if isinstance(event.payload, dict) else None
                        failure_status = (
                            failure_payload.get("error", {}).get("status")
                            if isinstance(failure_payload, dict) and isinstance(failure_payload.get("error"), dict)
                            else None
                        )
                        try:
                            turn_error_status = int(failure_status) if failure_status else 500
                        except (TypeError, ValueError):
                            turn_error_status = 500
                        # Failed turns evict the referenced parent from the
                        # connection-local cache (never reuse stale state).
                        if isinstance(previous_id, str):
                            self.local_cache.pop(previous_id, None)
                    break
        except Exception as exc:
            frame_out = _service_error_frame(exc, frame.stream_id)
            if _is_previous_response_failure(exc) and isinstance(previous_id, str):
                self.local_cache.pop(previous_id, None)
            turn_error_status = int(frame_out.get("status") or 500)
            yield frame_out
        finally:
            if events is not None:
                try:
                    await events.aclose()
                except Exception:
                    pass
            if transaction_logger is not None and hasattr(transaction_logger, "finalize_metadata"):
                try:
                    transaction_logger.finalize_metadata(
                        status_code=turn_error_status if turn_error_status is not None else (200 if terminal_seen else 500)
                    )
                except Exception:
                    pass

    # -- connection driver ---------------------------------------------------

    async def run(self, websocket: Any) -> None:
        """Serve the connection until close or the lifetime limit."""

        started = self._clock()
        while True:
            elapsed = self._clock() - started
            remaining = self._max_connection_seconds - elapsed
            if remaining <= 0:
                await websocket.send_json(error_frame(
                    "websocket_connection_limit_reached",
                    f"Responses websocket connection limit reached ({_human_duration(self._max_connection_seconds)}). Create a new websocket connection to continue.",
                    err_type="invalid_request_error",
                ))
                break
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=remaining)
            except asyncio.TimeoutError:
                continue
            except Exception:
                # Client disconnect: the in-flight turn (if any) is closed
                # cooperatively by handle_frame's finally paths.
                break
            turn_failed = False
            handle_gen = None
            try:
                handle_gen = self.handle_frame(raw)
                async for server_frame in handle_gen:
                    await websocket.send_json(server_frame)
            except Exception:
                # Send failures terminate the connection; closing the frame
                # handler chain unwinds _turn's finally, which acloses the
                # service event stream (and the upstream generator).
                turn_failed = True
            finally:
                if handle_gen is not None:
                    try:
                        await handle_gen.aclose()
                    except Exception:
                        pass
            if turn_failed:
                break
        try:
            await websocket.close(code=1000)
        except Exception:
            pass

    # seconds/minutes rendering for the limit message (guide says "(60 minutes)")


def _human_duration(seconds: float) -> str:
    if seconds > 0 and abs(seconds - round(seconds)) < 1e-9 and int(round(seconds)) % 60 == 0:
        return f"{int(round(seconds)) // 60} minutes"
    return f"{int(seconds)} seconds"
