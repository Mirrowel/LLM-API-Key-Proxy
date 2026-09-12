# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""WebSocket Mode for the Responses API compatibility layer.

Official wire contract (developers.openai.com/api/docs/guides/websocket-mode,
pinned 2026-09-12 — the multiplexed revision):

- The client opens one persistent connection (``wss://.../v1/responses``,
  ``Authorization: Bearer`` header) and starts every turn with a
  ``response.create`` frame whose payload mirrors the HTTP create body —
  except transport-specific fields (``stream``, ``background``, and the
  WebSocket-only ``stream_id`` / ``generate``).
- ``stream_id`` (1-256 chars of ``[A-Za-z0-9_.-]``) labels a lane; omitted
  means the implicit default lane. Named-lane server frames carry the same
  ``stream_id``. Turns on one lane run strictly FIFO and never overlap;
  turns on different lanes run concurrently. Reusing a ``stream_id`` without
  ``previous_response_id`` starts a fresh response on that lane.
- Service limits: up to 16 active in-flight responses per connection
  (excess ``response.create`` frames queue in their lane); up to 32 distinct
  named ``stream_id`` lanes (the default lane is not counted); a 60 minute
  connection lifetime (``max_connection_seconds`` stays configurable).
- ``generate: false`` warms request state: the connection-local cache
  records the turn's tools/instructions/input and returns a chainable
  response ID with no model output. Warmups validate exactly like a real
  turn (including continuation resolution and the frame's routing-derived
  scope) and are never sent to a provider.
- Continuation uses ``previous_response_id`` with incremental input. The
  previous-response state lives in a CONNECTION-LOCAL IN-MEMORY cache (the
  most recent responses), consulted before the global store — this is what
  keeps ``store=false`` / ZDR chains working; a ``store=true`` chain may
  hydrate from the global store on a local miss.
- ``response.steer`` carries ONLY ``type`` + ``previous_response_id`` +
  ``input``. ``response.steer.accepted`` acknowledges a queued steer;
  ``response.steer.failed`` reports ``invalid_input``, ``response_not_found``,
  ``response_already_completed``, ``response_not_active``, or
  ``too_many_pending_steers`` (cap 8 per response).
- Server frames are the SAME event objects the HTTP streaming API emits
  (``type`` + ``sequence_number`` + payload) and every event frame is built
  by :class:`ResponsesWebSocketFormatter`; terminal
  ``response.completed`` / ``response.failed`` / ``response.incomplete``
  close the turn — there is no ``[DONE]`` sentinel on this transport.
  ``sequence_number`` is a per-lane monotonic authority: the session overlays
  the lane's own counter on every frame it emits (provider events,
  synthesized steer/warmup frames, terminal failures), replacing whatever
  number the payload carried, so one lane's stream is strictly increasing.
- Errors arrive as ``{"type":"error","status":...,"error":{...}}`` frames,
  including ``invalid_stream_id``, ``websocket_stream_limit_reached``,
  ``websocket_connection_limit_reached``, and ``previous_response_not_found``.
  Outbound frames larger than 4 MiB are rejected with ``invalid_request_error``
  and close code 1008; inbound frames are capped by the configurable limit in
  the divergences below. Non-finite numeric values in a frame are rejected as
  ``invalid_request_error`` (JSON ``NaN`` / ``Infinity`` are not valid).

Concurrency model: ONE reader loop parses frames and routes each
``response.create`` into a per-lane FIFO queue; a per-lane worker drains its
queue with a single in-flight turn, and a global 16-slot semaphore throttles
across lanes (a create arriving at the cap waits in its lane queue, exactly
the documented "queued" behavior). Every frame leaves through one
lock-guarded send path, so concurrent lane turns interleave without tearing
frames. Steering is routed through a response-id -> lane index; a steer
targeting the lane's in-flight response is queued against that response, and
one targeting its just-completed response is queued for the lane.

Deliberate divergences from the pinned revision (never silent):

- **Bounded steering model.** The official successor lifecycle auto-creates a
  replacement response immediately after a steered turn. This proxy does NOT
  interrupt an in-flight provider turn and does not synthesize a successor:
  an accepted steer's input is prepended to the NEXT ``response.create`` on
  the target's lane. ``steer.accepted`` / ``steer.failed`` are emitted
  honestly, but the client still owns issuing that next create. Consequently
  a steered in-flight turn is never interrupted and therefore never ends as
  ``response.incomplete`` with ``incomplete_details.reason: "steered"``.
  ``response.steer.pending`` is not implemented — an accepted steer is only
  observed through the create that consumes it.
- **Inbound frame cap raised for official payloads.** The guide's 4 MiB
  client-frame cap cannot carry the official input schema (``file_data``
  payloads reach tens of MB), so this transport defaults inbound frames to
  32 MiB (``RESPONSES_WEBSOCKET_MAX_FRAME_BYTES``). Outbound frames keep the
  documented 4 MiB cap. The HTTP Responses transport has no comparable
  frame-size cap.
- **Per-lane backlog is unbounded.** The documented 16 in-flight and 32-lane
  caps hold, but creates queued behind them accumulate in their lane FIFO
  without a second bound; only the lane worker drains that queue.
- **Warmup validation runs inline in the reader.** A ``generate: false``
  frame validates (model, lifecycle, continuation resolution) on the reader
  loop rather than in a lane worker; validation is I/O-light by construction,
  which keeps the lane-state mutation race-free.
- **Steer index may outlive the continuation cache.** The steer response
  index holds 256 ids while the connection-local continuation cache holds 32;
  under churn an accepted steer's target can be evicted from the cache before
  the next create. A late ``response_not_found`` is the honest outcome.
- **Oldest-insertion cache eviction.** The connection-local cache evicts the
  oldest inserted entry (not an access-LRU); re-inserting a key refreshes its
  insertion position.
- **No eviction on failure.** The official same-lane-failure rule drops the
  referenced parent's memory. Operator ruling overrides it here: a failed
  turn NEVER deletes conversation memory (cross-lane parents obviously also
  stay). Stale-state reuse is accepted in exchange for never losing a ZDR
  chain to a transient provider failure.
- Application-level keepalive frames — none exist in the spec; deployments
  behind idle-timeout proxies must rely on WebSocket protocol pings
  (uvicorn's ``ws_ping_interval``, default 20s).
- Binary frames — this transport is text-only (every documented frame is a
  JSON object).
- ``response.cancel`` — Realtime-API-only; frames of this type receive a
  clean ``invalid_request_error``.
- Warmup inheritance of tools/instructions is proxy-side only: the warmup
  row's request state seeds the next turn's provider request when the turn
  omits it. Normal (non-warmup) continuations keep their own request fields.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, MutableMapping, Optional

from ..protocols.canonical import complete_responses_object
from .service import _safe_stored_request
from .streaming import ResponsesStreamEvent
from .types import StoredResponse, generate_response_id

# stream_id grammar per the official guide.
_STREAM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,256}$")

# Default connection lifetime (the documented service limit: 60 minutes).
DEFAULT_MAX_CONNECTION_SECONDS = 60 * 60

# Bounded connection-local cache: the guide keeps "the most recent" state;
# 32 ids cover deep tool loops without unbounded memory per connection.
_LOCAL_CACHE_MAX_ENTRIES = 32

# Official service limits for the multiplexed revision.
MAX_NAMED_LANES = 32
MAX_CONCURRENT_RESPONSES = 16
MAX_PENDING_STEERS = 8

# Outbound frames keep the documented 4 MiB cap. Inbound frames default to
# 32 MiB because the official input schema admits multimodal `file_data`
# payloads far larger than 4 MiB (the HTTP transport has no such cap);
# deployments tune this via RESPONSES_WEBSOCKET_MAX_FRAME_BYTES.
MAX_FRAME_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_INBOUND_FRAME_BYTES = 32 * 1024 * 1024

# Steering lookups are bounded alongside the local cache so a long-lived
# connection cannot accumulate response ids without limit.
_RESPONSE_INDEX_MAX = 256

# The reader awaits the socket directly (so container cancellation propagates
# cleanly); a short poll cap lets it notice worker-side fatal sends promptly.
_RECEIVE_POLL_SECONDS = 0.5

# Turn-closing terminal events (no [DONE] sentinel on this transport).
_TERMINAL_EVENT_TYPES = {"response.completed", "response.failed", "response.incomplete"}

# The only client fields the steering grammar admits.
_STEER_ALLOWED_FIELDS = frozenset({"type", "previous_response_id", "input"})

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


def _non_finite_path(value: Any, path: str = "") -> Optional[str]:
    """Return the first JSON path holding a non-finite number, else ``None``.

    Python's ``json`` parser accepts ``NaN`` / ``Infinity`` although the JSON
    spec does not; those values must never reach a provider body or a stored
    row, so the transport rejects the whole frame.
    """

    if isinstance(value, float):
        if not math.isfinite(value):
            return path or "frame"
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            found = _non_finite_path(item, f"{path}.{key}" if path else str(key))
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _non_finite_path(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


@dataclass
class ClientFrame:
    """One parsed ``response.create`` client frame."""

    payload: dict[str, Any] = field(default_factory=dict)
    stream_id: Optional[str] = None
    warmup: bool = False


@dataclass
class SteerFrame:
    """One parsed ``response.steer`` client frame.

    ``error`` is ``(code, message, param?)`` for grammar failures the
    transport must report as ``response.steer.failed`` rather than a
    top-level error frame.
    """

    previous_response_id: Optional[str] = None
    input: Any = None
    steer_id: str = ""
    error: Optional[tuple[str, str, Optional[str]]] = None


@dataclass
class LaneState:
    """Per-lane bookkeeping on one connection.

    ``latest_response_id`` is the fork/parent pointer: it tracks the newest
    response observed on the lane (active or completed) so steering and
    continuation can resolve the lane's turn without re-walking the cache.
    ``pending_steers`` holds accepted steer entries (``{"target", "input"}``)
    that a future ``response.create`` chaining from their target prepends.
    ``sequence`` is the lane's own monotonic frame counter: every frame
    emitted on the lane is stamped with the next value, so one lane's
    observable stream is strictly increasing regardless of provider-supplied
    numbers.
    """

    stream_id: Optional[str] = None
    latest_response_id: Optional[str] = None
    queue: "asyncio.Queue[ClientFrame]" = field(default_factory=asyncio.Queue)
    in_flight: int = 0
    pending_steers: list[dict[str, Any]] = field(default_factory=list)
    sequence: int = 0
    worker: Optional[asyncio.Task] = None


def _parse_steer_frame(payload: dict[str, Any]) -> SteerFrame:
    """Parse one ``response.steer`` body into a :class:`SteerFrame`."""

    steer_id = f"steer_{generate_response_id()[len('resp_'):]}"
    unexpected = set(payload) - _STEER_ALLOWED_FIELDS
    if unexpected:
        return SteerFrame(
            steer_id=steer_id,
            error=(
                "invalid_input",
                f"response.steer accepts only type, previous_response_id, input (unexpected: {sorted(unexpected)})",
                None,
            ),
        )
    previous_id = payload.get("previous_response_id")
    if not isinstance(previous_id, str) or not previous_id:
        return SteerFrame(
            steer_id=steer_id,
            input=payload.get("input"),
            error=("invalid_input", "response.steer requires a non-empty previous_response_id", "previous_response_id"),
        )
    if "input" not in payload:
        return SteerFrame(
            previous_response_id=previous_id,
            steer_id=steer_id,
            error=("invalid_input", "response.steer requires input", "input"),
        )
    steer_input = payload.get("input")
    if isinstance(steer_input, str):
        valid = bool(steer_input)
    elif isinstance(steer_input, list):
        valid = bool(steer_input)
    else:
        valid = False
    if not valid:
        return SteerFrame(
            previous_response_id=previous_id,
            steer_id=steer_id,
            error=(
                "invalid_input",
                "response.steer input must be a non-empty string or a non-empty list",
                "input",
            ),
        )
    return SteerFrame(previous_response_id=previous_id, input=steer_input, steer_id=steer_id)


def parse_client_frame(raw: str | bytes | dict[str, Any]) -> ClientFrame | SteerFrame | dict[str, Any]:
    """Parse one client frame into a turn or steer request.

    Returns :class:`ClientFrame` / :class:`SteerFrame` on success or a
    spec-shaped error frame dict on rejection. Only ``response.create`` and
    ``response.steer`` are in the official client vocabulary; anything else
    (including the Realtime-API ``response.cancel``) is rejected cleanly.
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
    bad_number = _non_finite_path(payload)
    if bad_number is not None:
        return error_frame(
            "invalid_request_error",
            f"non-finite number at {bad_number!r} is not allowed in WebSocket frames",
            param=bad_number,
            err_type="invalid_request_error",
        )
    frame_type = payload.get("type")
    if frame_type is None:
        return error_frame("invalid_request_error", "frame requires a 'type' field", err_type="invalid_request_error")
    if frame_type == "response.steer":
        return _parse_steer_frame(payload)
    if frame_type != "response.create":
        return error_frame(
            "invalid_request_error",
            f"frame type {frame_type!r} is not part of the Responses WebSocket vocabulary (only response.create and response.steer are supported)",
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
            "invalid_stream_id",
            "stream_id must not be null; omit the field for the default lane",
            param="stream_id",
            err_type="invalid_request_error",
        )
    if lane == "":
        return error_frame(
            "invalid_stream_id",
            "stream_id must not be empty; omit the field for the default lane",
            param="stream_id",
            err_type="invalid_request_error",
        )
    if lane is not None and not isinstance(lane, str):
        return error_frame("invalid_stream_id", "stream_id must be a string", param="stream_id", err_type="invalid_request_error")
    if lane is not None and not _STREAM_ID_PATTERN.match(lane):
        return error_frame(
            "invalid_stream_id",
            "stream_id must be 1-256 characters of letters, numbers, underscores, hyphens, and periods",
            param="stream_id",
            err_type="invalid_request_error",
        )
    return ClientFrame(payload=body, stream_id=lane, warmup=payload.get("generate") is False)


class ResponsesWebSocketFormatter:
    """Build transport-neutral events as WebSocket JSON frames.

    A frame IS the streaming event object (``type`` + ``sequence_number`` +
    payload fields) — the same objects the HTTP streaming API emits, with the
    lane's ``stream_id`` stamped on named lanes. SSE-only artifacts (comment
    heartbeats, the ``[DONE]`` sentinel) have no equivalent here and are
    dropped. ``event_frame`` is the single frame-construction seam every
    transport event flows through; ``format_stream_event`` serializes it.
    """

    transport = "websocket"

    def event_frame(self, event: ResponsesStreamEvent, *, stream_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Return the JSON-ready frame for one neutral event, or ``None``."""

        if event.heartbeat or event.event_name == "done":
            return None
        frame = dict(event.payload) if isinstance(event.payload, dict) else {}
        frame["type"] = event.event_name or frame.get("type") or "response.event"
        if stream_id:
            frame["stream_id"] = stream_id
        return frame

    def format_event(self, event_name: str, payload: dict[str, Any]) -> str:
        frame = dict(payload) if isinstance(payload, dict) else {}
        frame.setdefault("type", event_name)
        return json.dumps(frame, ensure_ascii=False)

    def format_stream_event(self, event: ResponsesStreamEvent, *, stream_id: Optional[str] = None) -> Optional[str]:
        frame = self.event_frame(event, stream_id=stream_id)
        if frame is None:
            return None
        return json.dumps(frame, ensure_ascii=False)


class _LocalResponsesCache(OrderedDict):
    """Connection-local continuation cache (bounded, oldest-insertion eviction).

    In-memory only — the guide's ZDR-compatible fast continuation path.
    The ResponsesService consults it before the global store for
    previous_response_id resolution and lineage walks, and stores each
    completed turn in it (even when ``store=false``). Re-inserting a key
    refreshes its insertion position.
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


def _failed_event_is_continuation_miss(response_obj: Any) -> bool:
    """Terminal response.failed miss detection (structured type + prefix)."""

    if not isinstance(response_obj, dict):
        return False
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


def _frame_bytes(raw: Any) -> int:
    """Return the UTF-8 byte length of a raw client frame."""

    if isinstance(raw, str):
        return len(raw.encode("utf-8"))
    if isinstance(raw, (bytes, bytearray)):
        return len(raw)
    return len(json.dumps(raw, ensure_ascii=False).encode("utf-8"))


class ResponsesWebSocketSession:
    """Drive one multiplexed Responses WebSocket connection.

    ``run()`` is the production driver: one reader loop parses frames and
    routes creates into per-lane FIFO queues drained by per-lane worker
    tasks; a global semaphore caps concurrent in-flight turns at 16 and a
    single lock guards the send path. ``handle_frame()`` is the single-turn
    convenience seam tests use to collect one turn's frames inline. In both
    paths every provider event is closed cooperatively (``aclose``) on
    completion, cancellation, or disconnect, so upstream streams never leak.
    The connection-local continuation cache is bounded and dies with this
    object; no-turn-failure eviction ever removes a parent (operator ruling).
    """

    def __init__(
        self,
        *,
        service: Any,
        client: Any,
        max_connection_seconds: float = DEFAULT_MAX_CONNECTION_SECONDS,
        max_inbound_frame_bytes: int = DEFAULT_MAX_INBOUND_FRAME_BYTES,
        clock: Callable[[], float] = time.monotonic,
        transaction_logger_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._service = service
        self._client = client
        self._max_connection_seconds = max_connection_seconds
        self._max_inbound_frame_bytes = max_inbound_frame_bytes
        self._clock = clock
        self._started = clock()
        self._transaction_logger_factory = transaction_logger_factory
        self._lanes: dict[str, LaneState] = {}
        self._response_lane: "OrderedDict[str, LaneState]" = OrderedDict()
        self._response_status: dict[str, str] = {}
        self._in_flight_semaphore = asyncio.Semaphore(MAX_CONCURRENT_RESPONSES)
        self._send_lock = asyncio.Lock()
        self._worker_tasks: set[asyncio.Task] = set()
        self._closed = False
        self._close_code: Optional[int] = None
        self._sequence = 0
        self.local_cache: MutableMapping[str, StoredResponse] = _LocalResponsesCache()
        self._formatter = ResponsesWebSocketFormatter()

    @property
    def lanes(self) -> dict[str, LaneState]:
        return self._lanes

    def _next_sequence(self) -> int:
        """Connection-local sequence for frames with no lane.

        Lane-attached frames use the lane's own counter (``_stamp_lane``):
        every provider-supplied number is replaced so each lane's observable
        stream stays strictly monotonic on its own. Lane-less grammar/steer
        frames draw this connection counter — never a process-global.
        """

        self._sequence += 1
        return self._sequence

    @staticmethod
    def _stamp_lane(frame: dict[str, Any], lane: LaneState) -> dict[str, Any]:
        """Overlay the lane's next monotonic sequence number onto ``frame``."""

        lane.sequence += 1
        frame["sequence_number"] = lane.sequence
        return frame

    def _format_event(
        self,
        event_name: str,
        payload: dict[str, Any],
        stream_id: Optional[str],
        *,
        lane: Optional[LaneState] = None,
    ) -> dict[str, Any]:
        """Build one server event frame through the WebSocket formatter.

        With a ``lane`` the frame is stamped using that lane's monotonic
        counter (replacing whatever sequence number the payload carried);
        without one (grammar-level failures before lane resolution) the
        connection-local counter is used.
        """

        frame = self._formatter.event_frame(ResponsesStreamEvent(event_name, payload), stream_id=stream_id)
        if frame is None:  # pragma: no cover - callers never pass heartbeat/done
            frame = dict(payload)
        if lane is not None:
            return self._stamp_lane(frame, lane)
        frame["sequence_number"] = self._next_sequence()
        return frame

    def _scope_key_for(self, body: dict[str, Any]) -> str:
        """Derive the connection scope from the frame's routing fields.

        The WS transport has no scope headers, so the turn body's routing
        fields (``api_keys`` / ``providers`` / ``classifier`` / ``private``)
        are authoritative — the same resolution ``stream_turn_events`` uses.
        """

        resolver = getattr(self._service, "request_scope_key", None)
        if callable(resolver):
            try:
                return str(resolver(body))
            except Exception:
                return "public"
        return "public"

    async def _validate_turn(self, body: dict[str, Any]) -> None:
        """Validate a turn preconditions-only (no provider call), incl. lineage."""

        validator = getattr(self._service, "validate_stream_request", None)
        if validator is None:
            return
        await validator(body, local_cache=self.local_cache)

    def _lane(self, stream_id: Optional[str]) -> LaneState:
        key = stream_id or ""
        lane = self._lanes.get(key)
        if lane is None:
            lane = LaneState(stream_id=stream_id)
            self._lanes[key] = lane
        return lane

    def _named_lane_count(self) -> int:
        return sum(1 for key in self._lanes if key)

    def _ensure_lane(self, stream_id: Optional[str]) -> tuple[Optional[LaneState], Optional[dict[str, Any]]]:
        """Return the lane for ``stream_id`` or the 32-lane limit error frame."""

        key = stream_id or ""
        if key and key not in self._lanes and self._named_lane_count() >= MAX_NAMED_LANES:
            return None, error_frame(
                "websocket_stream_limit_reached",
                f"Responses websocket stream limit reached ({MAX_NAMED_LANES} named streams). Reuse an existing stream_id or create a new connection to continue.",
                param="stream_id",
                stream_id=stream_id,
                err_type="invalid_request_error",
            )
        return self._lane(stream_id), None

    def _register_response(self, response_id: str, lane: LaneState, status: str) -> None:
        """Index a response id to its lane and status (bounded, FIFO prune)."""

        self._response_lane[response_id] = lane
        self._response_lane.move_to_end(response_id)
        while len(self._response_lane) > _RESPONSE_INDEX_MAX:
            stale, _ = self._response_lane.popitem(last=False)
            self._response_status.pop(stale, None)
        self._response_status[response_id] = status

    # -- frame handling (single-turn convenience seam) -----------------------

    async def handle_frame(self, raw: str | bytes | dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Handle one client frame, yielding server frames as dicts.

        Single-turn convenience for tests and embedding: it runs a create
        inline instead of routing through the lane workers.
        """

        parsed = parse_client_frame(raw)
        if isinstance(parsed, dict):
            yield parsed
            return
        if isinstance(parsed, SteerFrame):
            async for frame in self._steer(parsed):
                yield frame
            return
        if parsed.warmup:
            async for frame in self._warmup(parsed):
                yield frame
            return
        lane, lane_error = self._ensure_lane(parsed.stream_id)
        if lane_error is not None:
            yield lane_error
            return
        # GeneratorExit at OUR yield must unwind the turn generator too —
        # without this try/finally, closing this chain abandons _turn one
        # level down and its upstream aclose is deferred to GC.
        turn_gen = self._turn(parsed, lane)
        try:
            async for frame in turn_gen:
                yield frame
        finally:
            await turn_gen.aclose()

    # -- warmup --------------------------------------------------------------

    async def _warmup(self, frame: ClientFrame) -> AsyncGenerator[dict[str, Any], None]:
        """``generate: false`` — prepare request state, return a chainable id.

        State is connection-local ONLY (the guide's in-memory cache): the
        warmup body's input items, tools, and instructions persist in this
        session's cache under the scope derived FROM THE FRAME's routing
        fields, so the next turn chaining from the returned id replays the
        input through lineage expansion and seeds omitted tools/instructions
        from the warmed request. Nothing is written to the global store or
        disk. The warmup validates exactly like a real turn (model,
        lifecycle, and continuation resolution) before returning its id.
        """

        body = dict(frame.payload)
        try:
            await self._validate_turn(body)
        except Exception as exc:
            yield _service_error_frame(exc, frame.stream_id)
            return
        lane, lane_error = self._ensure_lane(frame.stream_id)
        if lane_error is not None:
            yield lane_error
            return
        response_id = generate_response_id()
        model = str(body.get("model") or "")
        stored = StoredResponse(
            id=response_id,
            model=model,
            status="completed",
            # One shared builder keeps the warmup object SDK-shaped.
            response=complete_responses_object(
                {},
                response_id=response_id,
                model=model,
                status="completed",
            ),
            request=_safe_stored_request(body),
            input_items=_warmup_input_items(body.get("input")),
            metadata={"warmup": True},
            scope_key=self._scope_key_for(body),
        )
        self.local_cache[response_id] = stored
        lane.latest_response_id = response_id
        self._register_response(response_id, lane, "completed")
        yield self._format_event(
            "response.completed",
            {
                "type": "response.completed",
                "response": dict(stored.response),
            },
            frame.stream_id,
            lane=lane,
        )

    # -- turns ---------------------------------------------------------------

    def _prepare_body(self, frame: ClientFrame, lane: LaneState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Merge warmup state and the target's queued steers into a turn body.

        Returns the body plus the steer entries consumed into it; the caller
        restores those entries on failure so a retry create can re-prepend
        them. Warmup seeding is scope-gated: a warmed row only seeds a turn
        whose routing scope matches. Steers are partitioned by target, so an
        accepted steer for a different response stays queued.
        """

        body = dict(frame.payload)
        previous_id = body.get("previous_response_id")
        if isinstance(previous_id, str):
            parent = self.local_cache.get(previous_id)
            if (
                parent is not None
                and parent.metadata.get("warmup")
                and isinstance(parent.request, dict)
                and parent.scope_key == self._scope_key_for(body)
            ):
                for key in ("tools", "tool_choice", "instructions"):
                    if key not in body and key in parent.request:
                        body[key] = deepcopy(parent.request[key])
        consumed: list[dict[str, Any]] = []
        if isinstance(previous_id, str) and lane.pending_steers:
            consumed = [entry for entry in lane.pending_steers if entry.get("target") == previous_id]
            if consumed:
                lane.pending_steers = [entry for entry in lane.pending_steers if entry.get("target") != previous_id]
                items: list[Any] = []
                for entry in consumed:
                    items.extend(_warmup_input_items(entry.get("input")))
                items.extend(_warmup_input_items(body.get("input")))
                body["input"] = items
        return body, consumed

    async def _turn(self, frame: ClientFrame, lane: LaneState) -> AsyncGenerator[dict[str, Any], None]:
        """Run one response.create turn to its terminal event.

        Provider events pass through the WebSocket formatter (never a
        hand-built frame); a continuation miss is surfaced as the documented
        top-level ``previous_response_not_found`` frame. Every frame is
        stamped with the lane's monotonic sequence. No failure path evicts
        the referenced parent (the no-eviction ruling), and a failed turn
        returns any steer input it consumed so a retry create re-prepends it.
        """

        body, consumed_steers = self._prepare_body(frame, lane)
        transaction_logger = (
            self._transaction_logger_factory(str(body.get("model") or "unknown"))
            if self._transaction_logger_factory is not None
            else None
        )
        events: Optional[AsyncGenerator[ResponsesStreamEvent, None]] = None
        terminal_seen = False
        turn_failed = False
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
                event_name = event.event_name or payload.get("type") or "response.event"
                response_obj = payload.get("response")
                response_id = response_obj.get("id") if isinstance(response_obj, dict) else None
                if isinstance(response_id, str):
                    lane.latest_response_id = response_id
                    self._register_response(response_id, lane, "active")
                if event_name == "response.failed" and _failed_event_is_continuation_miss(response_obj):
                    # The service converts pre-stream failures into terminal
                    # response.failed events; the guide documents continuation
                    # misses as top-level error frames. The parent is NOT
                    # evicted (the no-eviction ruling).
                    if isinstance(response_id, str):
                        self._response_status[response_id] = "failed"
                    yield self._stamp_lane(
                        error_frame(
                            "previous_response_not_found",
                            str((response_obj.get("error") or {}).get("message") or "previous response not found"),
                            status=400,
                            param="previous_response_id",
                            stream_id=frame.stream_id,
                        ),
                        lane,
                    )
                    terminal_seen = True
                    turn_failed = True
                    turn_error_status = 400
                    break
                if event_name in _TERMINAL_EVENT_TYPES:
                    terminal_seen = True
                    if isinstance(response_id, str):
                        # Terminal status is its own vocabulary: a failed
                        # response can no longer be steered.
                        self._response_status[response_id] = {
                            "response.completed": "completed",
                            "response.failed": "failed",
                        }.get(event_name, "incomplete")
                    if event_name == "response.failed":
                        turn_failed = True
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
                    yield self._format_event(event_name, payload, frame.stream_id, lane=lane)
                    break
                yield self._format_event(event_name, payload, frame.stream_id, lane=lane)
        except Exception as exc:
            turn_failed = True
            frame_out = _service_error_frame(exc, frame.stream_id)
            turn_error_status = int(frame_out.get("status") or 500)
            yield self._stamp_lane(frame_out, lane)
        finally:
            if turn_failed and consumed_steers:
                # A failed turn must not swallow accepted steers: return them
                # to the front of the queue so a retry create re-prepends them.
                lane.pending_steers = consumed_steers + lane.pending_steers
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

    # -- steering ------------------------------------------------------------

    def _steer_accepted_frame(self, frame: SteerFrame, lane: Optional[LaneState] = None) -> dict[str, Any]:
        return self._format_event(
            "response.steer.accepted",
            {
                "type": "response.steer.accepted",
                "steer": {"id": frame.steer_id, "previous_response_id": frame.previous_response_id},
            },
            lane.stream_id if lane is not None else None,
            lane=lane,
        )

    def _steer_failed_frame(
        self,
        frame: SteerFrame,
        code: str,
        message: str,
        param: Optional[str],
        lane: Optional[LaneState] = None,
    ) -> dict[str, Any]:
        error: dict[str, Any] = {"type": "invalid_request_error", "code": code, "message": message}
        if param is not None:
            error["param"] = param
        return self._format_event(
            "response.steer.failed",
            {
                "type": "response.steer.failed",
                "steer": {
                    "id": frame.steer_id,
                    "input": frame.input,
                    "previous_response_id": frame.previous_response_id,
                },
                "error": error,
            },
            lane.stream_id if lane is not None else None,
            lane=lane,
        )

    def _pending_steer_count(self, lane: LaneState, target: str) -> int:
        return sum(1 for entry in lane.pending_steers if entry.get("target") == target)

    async def _steer(self, frame: SteerFrame) -> AsyncGenerator[dict[str, Any], None]:
        """Accept, queue, or fail one ``response.steer``.

        Bounded model: an accepted steer is queued on its target's lane and
        prepended to the NEXT create on that lane; an in-flight provider turn
        is never interrupted and no successor is auto-created.
        """

        if frame.error is not None:
            code, message, param = frame.error
            yield self._steer_failed_frame(frame, code, message, param)
            return
        target = frame.previous_response_id or ""
        lane = self._response_lane.get(target)
        if lane is None:
            yield self._steer_failed_frame(frame, "response_not_found", f"Response {target!r} not found on this connection", "previous_response_id")
            return
        status = self._response_status.get(target)
        if status == "active":
            if self._pending_steer_count(lane, target) >= MAX_PENDING_STEERS:
                yield self._steer_failed_frame(frame, "too_many_pending_steers", f"Response {target!r} already has {MAX_PENDING_STEERS} pending steers", None, lane)
                return
            lane.pending_steers.append({"target": target, "input": frame.input})
            yield self._steer_accepted_frame(frame, lane)
            return
        if status == "completed":
            if lane.latest_response_id != target:
                yield self._steer_failed_frame(frame, "response_already_completed", f"Response {target!r} already completed and is no longer the lane's active response", None, lane)
                return
            if self._pending_steer_count(lane, target) >= MAX_PENDING_STEERS:
                yield self._steer_failed_frame(frame, "too_many_pending_steers", f"Response {target!r} already has {MAX_PENDING_STEERS} pending steers", None, lane)
                return
            lane.pending_steers.append({"target": target, "input": frame.input})
            yield self._steer_accepted_frame(frame, lane)
            return
        if status == "failed":
            yield self._steer_failed_frame(frame, "response_not_active", f"Response {target!r} failed and can no longer be steered", None, lane)
            return
        yield self._steer_failed_frame(frame, "response_not_active", f"Response {target!r} is not active and cannot be steered", None, lane)

    # -- connection driver ---------------------------------------------------

    async def _send(self, websocket: Any, frame: dict[str, Any]) -> None:
        """Serialize and send one frame under the single send lock.

        Outbound frames over the 4 MiB cap are replaced by an error frame and
        tear the connection down with code 1008; a raising ``send_json``
        marks the connection dead so the driver stops. Once the connection is
        dead, later sends are silently dropped.
        """

        if self._closed:
            return
        async with self._send_lock:
            data = json.dumps(frame, ensure_ascii=False)
            if len(data.encode("utf-8")) > MAX_FRAME_BYTES:
                self._close_code = 1008
                self._closed = True
                await websocket.send_json(
                    error_frame(
                        "invalid_request_error",
                        f"serialized frame exceeds the {MAX_FRAME_BYTES} byte WebSocket frame limit",
                        err_type="invalid_request_error",
                    )
                )
                return
            try:
                await websocket.send_json(frame)
            except Exception:
                self._closed = True
                raise

    def _ensure_worker(self, lane: LaneState, websocket: Any) -> None:
        if lane.worker is None or lane.worker.done():
            task = asyncio.create_task(self._lane_worker(lane, websocket))
            lane.worker = task
            self._worker_tasks.add(task)
            task.add_done_callback(self._worker_tasks.discard)

    async def _lane_worker(self, lane: LaneState, websocket: Any) -> None:
        """Drain one lane's FIFO queue, one in-flight turn at a time.

        The connection lifetime is enforced here, between turns: a turn that
        dequeued before the deadline runs to its terminal, and the next queued
        create after the deadline gets the documented limit error frame and
        closes the connection. An in-flight stream is never truncated.
        """

        while not self._closed:
            try:
                frame = await lane.queue.get()
            except asyncio.CancelledError:
                raise
            if self._clock() - self._started >= self._max_connection_seconds:
                await self._send(
                    websocket,
                    self._stamp_lane(
                        error_frame(
                            "websocket_connection_limit_reached",
                            f"Responses websocket connection limit reached ({_human_duration(self._max_connection_seconds)}). Create a new websocket connection to continue.",
                            err_type="invalid_request_error",
                            stream_id=lane.stream_id,
                        ),
                        lane,
                    ),
                )
                lane.queue.task_done()
                self._closed = True
                break
            async with self._in_flight_semaphore:
                lane.in_flight += 1
                turn_gen = self._turn(frame, lane)
                try:
                    async for out in turn_gen:
                        await self._send(websocket, out)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # _turn already converts service failures into error
                    # frames; a raised exception here is a send/transport
                    # failure, which _send marks fatal.
                    pass
                finally:
                    try:
                        await turn_gen.aclose()
                    except Exception:
                        pass
                    lane.in_flight -= 1
            lane.queue.task_done()

    async def _dispatch(self, raw: Any, websocket: Any) -> None:
        """Parse one client frame and route it without awaiting the turn."""

        parsed = parse_client_frame(raw)
        if isinstance(parsed, dict):
            await self._send(websocket, parsed)
            return
        if isinstance(parsed, SteerFrame):
            async for frame in self._steer(parsed):
                await self._send(websocket, frame)
            return
        if parsed.warmup:
            async for frame in self._warmup(parsed):
                await self._send(websocket, frame)
            return
        lane, lane_error = self._ensure_lane(parsed.stream_id)
        if lane_error is not None:
            await self._send(websocket, lane_error)
            return
        await lane.queue.put(parsed)
        self._ensure_worker(lane, websocket)

    async def _shutdown(self) -> None:
        """Cancel lane workers and cooperatively close their upstream streams."""

        self._closed = True
        tasks = list(self._worker_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._worker_tasks.clear()

    async def _finalize(self, websocket: Any, close_code: int) -> None:
        """Cancel workers and close the socket (cooperative cleanup)."""

        await self._shutdown()
        try:
            await websocket.close(code=close_code)
        except Exception:
            pass

    async def run(self, websocket: Any) -> None:
        """Serve the connection until close, fatal error, or the lifetime limit.

        The reader awaits the socket directly (no detached receive task) so a
        container-driven cancellation propagates cleanly; a short poll cap is
        the only concession, letting worker-side fatal sends interrupt the
        reader promptly.
        """

        started = self._clock()
        self._started = started
        close_code = 1000
        cancelled = False
        try:
            while not self._closed:
                # The reader never enforces the lifetime itself: a turn may be
                # in flight. `_lane_worker` sees the deadline between turns.
                remaining = self._max_connection_seconds - (self._clock() - started)
                timeout = min(remaining, _RECEIVE_POLL_SECONDS) if remaining > 0 else _RECEIVE_POLL_SECONDS
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    # Client disconnect: workers are closed cooperatively below.
                    break
                if self._closed:
                    close_code = self._close_code if self._close_code is not None else close_code
                    break
                if _frame_bytes(raw) > self._max_inbound_frame_bytes:
                    await self._send(
                        websocket,
                        error_frame(
                            "invalid_request_error",
                            f"client frame exceeds the {self._max_inbound_frame_bytes} byte WebSocket frame limit",
                            err_type="invalid_request_error",
                        ),
                    )
                    close_code = 1008
                    break
                try:
                    await self._dispatch(raw, websocket)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # A transport bug must never kill the connection, except
                    # where the spec demands a close (already handled above).
                    if not self._closed:
                        try:
                            await self._send(
                                websocket,
                                error_frame("internal_error", "internal websocket processing error", status=500, err_type="server_error"),
                            )
                        except Exception:
                            self._closed = True
                if self._closed:
                    close_code = self._close_code if self._close_code is not None else close_code
                    break
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            self._closed = True
            finalizer = self._finalize(websocket, self._close_code if self._close_code is not None else close_code)
            if cancelled:
                # The container is tearing the connection down: never block the
                # cancellation with awaits (that can surface as a cancelled
                # task); schedule the cooperative cleanup instead.
                asyncio.ensure_future(finalizer)
            else:
                await finalizer


def _human_duration(seconds: float) -> str:
    if seconds > 0 and abs(seconds - round(seconds)) < 1e-9 and int(round(seconds)) % 60 == 0:
        return f"{int(round(seconds)) // 60} minutes"
    return f"{int(seconds)} seconds"
