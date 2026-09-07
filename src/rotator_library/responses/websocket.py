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


def parse_client_frame(raw: str | bytes | dict[str, Any], *, stream_id: Optional[str] = None) -> ClientFrame | dict[str, Any]:
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
    # Transport-specific fields never apply on this transport.
    body = {k: v for k, v in payload.items() if k not in {"type", "stream", "background", "generate"}}
    lane = body.pop("stream_id", stream_id)
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


def _is_previous_response_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "previous response" in message or "response not found" in message


def _service_error_frame(exc: BaseException, stream_id: Optional[str]) -> dict[str, Any]:
    """Map a ResponsesServiceError to its spec-shaped frame."""

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
            err_type="invalid_request_error",
        )
    return error_frame(err_type, message, status=status, stream_id=stream_id, err_type="invalid_request_error")


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
        self._sequence = 0

    @property
    def lanes(self) -> dict[str, LaneState]:
        return self._lanes

    # -- sequencing for locally synthesized frames --------------------------

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

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
        async for frame in self._turn(parsed):
            yield frame

    async def _warmup(self, frame: ClientFrame) -> AsyncGenerator[dict[str, Any], None]:
        """``generate: false`` — prepare request state, return a chainable id.

        State is connection-local ONLY (the guide's in-memory cache): the
        warmup body's input/tools/instructions persist in this session's
        cache so the next turn chaining from the returned id replays them.
        Nothing is written to the global store or disk.
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
                    and _is_previous_response_failure(str((response_obj.get("error") or {}).get("message") or ""))
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
                        err_type="invalid_request_error",
                    )
                    terminal_seen = True
                    break
                yield payload
                if event.event_name in _TERMINAL_EVENT_TYPES:
                    terminal_seen = True
                    if event.event_name == "response.failed":
                        # Failed turns evict the referenced parent from the
                        # connection-local cache (never reuse stale state).
                        if isinstance(previous_id, str):
                            self.local_cache.pop(previous_id, None)
                    break
        except Exception as exc:
            frame_out = _service_error_frame(exc, frame.stream_id)
            if _is_previous_response_failure(exc) and isinstance(previous_id, str):
                self.local_cache.pop(previous_id, None)
            yield frame_out
        finally:
            if events is not None:
                try:
                    await events.aclose()
                except Exception:
                    pass
            if transaction_logger is not None and hasattr(transaction_logger, "finalize_metadata"):
                try:
                    transaction_logger.finalize_metadata(status_code=200 if terminal_seen else 500)
                except Exception:
                    pass

    # -- connection driver ---------------------------------------------------

    async def run(self, websocket: Any) -> None:
        """Serve the connection until close or the lifetime limit."""

        started = self._clock()
        limit_error_sent = False
        while True:
            elapsed = self._clock() - started
            remaining = self._max_connection_seconds - elapsed
            if remaining <= 0:
                await websocket.send_json(error_frame(
                    "websocket_connection_limit_reached",
                    f"Responses websocket connection limit reached ({_human_duration(self._max_connection_seconds)}). Create a new websocket connection to continue.",
                    err_type="invalid_request_error",
                ))
                limit_error_sent = True
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
            try:
                async for server_frame in self.handle_frame(raw):
                    await websocket.send_json(server_frame)
            except Exception:
                # Send failures terminate the connection; the turn generator
                # chain is closed by _turn's finally on unwind.
                turn_failed = True
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
