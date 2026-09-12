"""G11 Phase C — first-class WebSocket Responses mode.

Concurrent lanes, same-lane FIFO, 16 in-flight cap, 32-lane cap, default
lane, forks, the no-eviction ruling, the bounded steering model, warmup
request-state preservation + frame-derived scope, stream_id grammar, the
4 MiB frame cap, and non-finite rejection.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Optional

from rotator_library.responses import InMemoryResponsesStore, ResponsesService
from rotator_library.responses.streaming import ResponsesStreamEvent
from rotator_library.responses.types import StoredResponse
from rotator_library.responses.websocket import (
    MAX_CONCURRENT_RESPONSES,
    MAX_FRAME_BYTES,
    MAX_NAMED_LANES,
    MAX_PENDING_STEERS,
    ResponsesWebSocketSession,
    parse_client_frame,
)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class CollectingSocket:
    def __init__(self, incoming: Optional[list[str]] = None):
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        for frame in incoming or []:
            self._queue.put_nowait(frame)
        self.sent: list[dict[str, Any]] = []
        self.closed_with: Optional[int] = None

    async def receive_text(self) -> str:
        return await self._queue.get()

    async def push(self, raw: str) -> None:
        await self._queue.put(raw)

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def close(self, code: int = 1000) -> None:
        self.closed_with = code


async def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


class Harness:
    """Runs a session against a controllable socket and tears it down."""

    def __init__(self, session: ResponsesWebSocketSession, incoming: Optional[list[str]] = None):
        self.session = session
        self.ws = CollectingSocket(incoming)
        self.task = asyncio.create_task(session.run(self.ws))

    async def wait(self, predicate: Callable[[list[dict[str, Any]]], bool], timeout: float = 5.0) -> None:
        await _wait_until(lambda: predicate(self.ws.sent), timeout)

    async def aclose(self) -> None:
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


async def _collect(agen) -> list[dict[str, Any]]:
    return [item async for item in agen]


def _create(*, model: str = "m", stream_id: Optional[str] = None, **fields: Any) -> str:
    payload: dict[str, Any] = {"type": "response.create", "model": model, **fields}
    if stream_id is not None:
        payload["stream_id"] = stream_id
    return json.dumps(payload)


def _types(frames: list[dict[str, Any]]) -> list[str]:
    return [frame.get("type") for frame in frames]


class ScriptedService:
    """Real-ish service seam: created -> gate -> completed/failed per turn."""

    def __init__(self, gate_count: int = 0, fail_ids: tuple[str, ...] = ()):
        self.requests: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0
        self.gates = [asyncio.Event() for _ in range(gate_count)]
        self.release_all = asyncio.Event()
        self.store = InMemoryResponsesStore()
        self.fail_ids = set(fail_ids)

    def release_all_gates(self) -> None:
        self.release_all.set()
        for gate in self.gates:
            gate.set()

    async def stream_turn_events(self, raw_request, client, *, transaction_logger=None, local_cache=None, **kwargs):
        index = len(self.requests)
        self.requests.append(dict(raw_request))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        response_id = f"resp_script_{index + 1}"
        try:
            yield ResponsesStreamEvent(
                "response.created",
                {"type": "response.created", "sequence_number": index, "response": {"id": response_id, "status": "in_progress"}},
            )
            gate = self.gates[index] if index < len(self.gates) else self.release_all
            await gate.wait()
            if response_id in self.fail_ids:
                yield ResponsesStreamEvent(
                    "response.failed",
                    {"type": "response.failed", "response": {"id": response_id, "status": "failed", "error": {"message": "boom"}}},
                )
            else:
                yield ResponsesStreamEvent(
                    "response.completed",
                    {"type": "response.completed", "response": {"id": response_id, "status": "completed"}},
                )
        finally:
            self.active -= 1


class ChainClient:
    """Emits a created/completed pair and records the provider payload."""

    def __init__(self, prefix: str = "chain"):
        self.payloads: list[dict[str, Any]] = []
        self.prefix = prefix

    async def agenerate(self, payload, *, input_protocol, request=None, **kwargs):
        self.payloads.append(dict(payload))
        response_id = f"resp_{self.prefix}_{len(self.payloads)}"

        async def frames():
            yield f'event: response.created\ndata: {{"type":"response.created","response":{{"id":"{response_id}","object":"response","status":"in_progress","model":"m","output":[]}}}}\n\n'
            yield f'event: response.completed\ndata: {{"type":"response.completed","response":{{"id":"{response_id}","object":"response","status":"completed","model":"m","output":[]}}}}\n\n'

        return frames()


def _events_for(frames: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [frame for frame in frames if frame.get("type") == name]


# ---------------------------------------------------------------------------
# Concurrency + lane limits
# ---------------------------------------------------------------------------


async def test_two_named_lanes_interleave_concurrently() -> None:
    service = ScriptedService(gate_count=2)
    session = ResponsesWebSocketSession(service=service, client=object())
    harness = Harness(session, [_create(stream_id="alpha"), _create(stream_id="beta")])
    try:
        await harness.wait(lambda sent: len(service.requests) == 2)
        # Both lanes started before either gate released: true parallelism.
        assert service.max_active == 2
        service.release_all_gates()
        await harness.wait(lambda sent: len(_events_for(sent, "response.completed")) == 2)
    finally:
        await harness.aclose()

    created = _events_for(harness.ws.sent, "response.created")
    assert {frame["stream_id"] for frame in created} == {"alpha", "beta"}
    terminals = _events_for(harness.ws.sent, "response.completed")
    assert {frame["stream_id"] for frame in terminals} == {"alpha", "beta"}


async def test_same_lane_creates_are_fifo_non_overlapping() -> None:
    service = ScriptedService(gate_count=2)
    session = ResponsesWebSocketSession(service=service, client=object())
    harness = Harness(session, [_create(stream_id="lane-a"), _create(stream_id="lane-a")])
    try:
        await harness.wait(lambda sent: len(_events_for(sent, "response.created")) == 1)
        # The second create must wait: the lane worker cannot overlap turns.
        await asyncio.sleep(0.1)
        assert len(service.requests) == 1
        assert service.max_active == 1
        service.gates[0].set()
        await harness.wait(lambda sent: len(service.requests) == 2)
        service.gates[1].set()
        await harness.wait(lambda sent: len(_events_for(sent, "response.completed")) == 2)
    finally:
        await harness.aclose()

    assert service.max_active == 1
    assert sorted(frame["stream_id"] for frame in _events_for(harness.ws.sent, "response.completed")) == ["lane-a", "lane-a"]


async def test_sixteen_in_flight_cap_queues_the_seventeenth() -> None:
    service = ScriptedService(gate_count=MAX_CONCURRENT_RESPONSES)
    session = ResponsesWebSocketSession(service=service, client=object())
    incoming = [_create(stream_id=f"lane-{index}") for index in range(MAX_CONCURRENT_RESPONSES + 1)]
    harness = Harness(session, incoming)
    try:
        await harness.wait(lambda sent: len(service.requests) == MAX_CONCURRENT_RESPONSES)
        await asyncio.sleep(0.1)
        # The 17th wait sits in its lane queue; no error frame is emitted.
        assert len(service.requests) == MAX_CONCURRENT_RESPONSES
        assert not _events_for(harness.ws.sent, "error")
        assert len(_events_for(harness.ws.sent, "response.created")) == MAX_CONCURRENT_RESPONSES
        service.release_all_gates()
        await harness.wait(lambda sent: len(_events_for(sent, "response.completed")) == MAX_CONCURRENT_RESPONSES + 1, timeout=10.0)
    finally:
        await harness.aclose()

    assert service.max_active == MAX_CONCURRENT_RESPONSES
    assert len(_events_for(harness.ws.sent, "response.completed")) == MAX_CONCURRENT_RESPONSES + 1


async def test_thirty_second_named_lane_is_rejected() -> None:
    service = ScriptedService(gate_count=0)
    session = ResponsesWebSocketSession(service=service, client=object())
    incoming = [_create(stream_id=f"lane-{index}") for index in range(MAX_NAMED_LANES + 1)]
    harness = Harness(session, incoming)
    try:
        await harness.wait(lambda sent: bool(_events_for(sent, "error")), timeout=10.0)
    finally:
        await harness.aclose()

    errors = _events_for(harness.ws.sent, "error")
    assert len(errors) == 1
    assert errors[0]["error"]["code"] == "websocket_stream_limit_reached"
    assert errors[0]["error"]["type"] == "invalid_request_error"


async def test_default_lane_omits_stream_id() -> None:
    service = ScriptedService(gate_count=1)
    session = ResponsesWebSocketSession(service=service, client=object())
    harness = Harness(session, [_create()])
    try:
        await harness.wait(lambda sent: len(_events_for(sent, "response.created")) == 1)
        service.gates[0].set()
        await harness.wait(lambda sent: len(_events_for(sent, "response.completed")) == 1)
    finally:
        await harness.aclose()

    assert all("stream_id" not in frame for frame in harness.ws.sent)
    assert session.lanes[""].latest_response_id == "resp_script_1"


# ---------------------------------------------------------------------------
# Continuation, forks, no-eviction
# ---------------------------------------------------------------------------


async def test_fork_lane_b_continues_from_lane_a_completed_id() -> None:
    client = ChainClient()
    service = ResponsesService(store=InMemoryResponsesStore())
    session = ResponsesWebSocketSession(service=service, client=client)

    first = await _collect(session.handle_frame(_create(stream_id="lane-a", store=False, input="alpha")))
    fork_id = first[-1]["response"]["id"]
    assert fork_id in session.local_cache
    assert session.lanes["lane-a"].latest_response_id == fork_id

    second = await _collect(session.handle_frame(_create(stream_id="lane-b", store=False, previous_response_id=fork_id, input="beta")))
    assert second[-1]["type"] == "response.completed", second
    # Lane A's memory is intact after the fork; the fork replayed its input.
    assert fork_id in session.local_cache
    assert session.lanes["lane-a"].latest_response_id == fork_id
    assert "alpha" in json.dumps(client.payloads[1].get("input"))


async def test_failure_never_evicts_the_referenced_parent() -> None:
    service = ScriptedService(gate_count=1, fail_ids=("resp_script_1",))
    session = ResponsesWebSocketSession(service=service, client=object())
    session.local_cache["resp_parent"] = StoredResponse(id="resp_parent", model="m", status="completed", response={})
    harness = Harness(session, [_create(stream_id="lane-a", previous_response_id="resp_parent")])
    try:
        await harness.wait(lambda sent: len(_events_for(sent, "response.created")) == 1)
        service.gates[0].set()
        await harness.wait(lambda sent: len(_events_for(sent, "response.failed")) == 1)
    finally:
        await harness.aclose()

    # The operator ruling rejects official same-lane-failure eviction.
    assert "resp_parent" in session.local_cache
    assert harness.ws.sent[-1]["type"] == "response.failed"


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------


async def test_steer_accepted_and_input_prepended_to_next_create() -> None:
    service = ScriptedService(gate_count=2)
    session = ResponsesWebSocketSession(service=service, client=object())
    harness = Harness(session, [_create(stream_id="lane-a", input="base")])
    try:
        await harness.wait(lambda sent: len(_events_for(sent, "response.created")) == 1)
        await harness.ws.push(json.dumps({"type": "response.steer", "previous_response_id": "resp_script_1", "input": "steer text"}))
        await harness.wait(lambda sent: len(_events_for(sent, "response.steer.accepted")) == 1)
        accepted = _events_for(harness.ws.sent, "response.steer.accepted")[0]
        # Grammar: the steer frame must NOT carry stream_id.
        assert "stream_id" not in accepted
        assert accepted["steer"]["previous_response_id"] == "resp_script_1"

        service.gates[0].set()
        await harness.wait(lambda sent: len(_events_for(sent, "response.completed")) == 1)
        await harness.ws.push(
            json.dumps(
                {"type": "response.create", "model": "m", "stream_id": "lane-a", "previous_response_id": "resp_script_1", "input": "next"}
            )
        )
        await harness.wait(lambda sent: len(service.requests) == 2)
        service.gates[1].set()
        await harness.wait(lambda sent: len(_events_for(sent, "response.completed")) == 2)
    finally:
        await harness.aclose()

    assert service.requests[1]["input"] == ["steer text", "next"]


async def test_unknown_steer_id_fails_without_stream_id() -> None:
    session = ResponsesWebSocketSession(service=ScriptedService(), client=object())
    frames = await _collect(session.handle_frame(json.dumps({"type": "response.steer", "previous_response_id": "resp_nope", "input": "x"})))
    assert len(frames) == 1
    assert frames[0]["type"] == "response.steer.failed"
    assert frames[0]["error"]["code"] == "response_not_found"
    assert frames[0]["error"]["param"] == "previous_response_id"
    assert "stream_id" not in frames[0]


async def test_too_many_pending_steers_cap() -> None:
    service = ScriptedService(gate_count=1)
    session = ResponsesWebSocketSession(service=service, client=object())
    harness = Harness(session, [_create(stream_id="lane-a")])
    try:
        await harness.wait(lambda sent: len(_events_for(sent, "response.created")) == 1)
        for index in range(MAX_PENDING_STEERS + 1):
            await harness.ws.push(json.dumps({"type": "response.steer", "previous_response_id": "resp_script_1", "input": f"s{index}"}))
        await harness.wait(lambda sent: len(_events_for(sent, "response.steer.failed")) == 1)
        await harness.wait(lambda sent: len(_events_for(sent, "response.steer.accepted")) == MAX_PENDING_STEERS)
    finally:
        await harness.aclose()

    failed = _events_for(harness.ws.sent, "response.steer.failed")[0]
    assert failed["error"]["code"] == "too_many_pending_steers"
    assert len(failed["steer"]["input"]) > 0


async def test_invalid_steer_grammar_is_invalid_input() -> None:
    session = ResponsesWebSocketSession(service=ScriptedService(), client=object())
    frames = await _collect(session.handle_frame(json.dumps({"type": "response.steer", "input": "x"})))
    assert frames[0]["type"] == "response.steer.failed"
    assert frames[0]["error"]["code"] == "invalid_input"


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


async def test_warmup_preserves_tools_instructions_and_derives_scope_from_frame() -> None:
    tools = [{"type": "function", "name": "lookup", "parameters": {"type": "object"}}]
    service = ResponsesService(store=InMemoryResponsesStore())
    client = ChainClient()
    session = ResponsesWebSocketSession(service=service, client=client)
    warmup_body = {
        "model": "m",
        "generate": False,
        "classifier": "scope-a",
        "instructions": "Be terse.",
        "tools": tools,
        "input": "warm context",
    }
    warmup = await _collect(session.handle_frame(json.dumps({"type": "response.create", **warmup_body})))
    warmup_id = warmup[0]["response"]["id"]
    cached = session.local_cache[warmup_id]
    assert cached.metadata.get("warmup") is True
    assert cached.request["tools"] == tools
    assert cached.request["instructions"] == "Be terse."
    # Scope is derived FROM THE FRAME, not pinned to public.
    expected_scope = service.request_scope_key({key: value for key, value in warmup_body.items() if key != "type"})
    assert cached.scope_key == expected_scope
    assert cached.scope_key.startswith("classifier:")

    turn = await _collect(
        session.handle_frame(
            json.dumps(
                {
                    "type": "response.create",
                    "model": "m",
                    "classifier": "scope-a",
                    "previous_response_id": warmup_id,
                    "input": "real question",
                }
            )
        )
    )
    assert turn[-1]["type"] == "response.completed", turn
    # The warmed tools/instructions seeded the provider request.
    assert client.payloads[0]["tools"] == tools
    assert client.payloads[0]["instructions"] == "Be terse."
    assert "warm context" in json.dumps(client.payloads[0].get("input"))


# ---------------------------------------------------------------------------
# Grammar + frame limits
# ---------------------------------------------------------------------------


def test_invalid_stream_id_grammar_code() -> None:
    for raw in (
        {"type": "response.create", "stream_id": ""},
        {"type": "response.create", "stream_id": None},
        {"type": "response.create", "stream_id": "bad id!"},
        {"type": "response.create", "stream_id": "a" * 257},
    ):
        frame = parse_client_frame(raw)
        assert frame["type"] == "error"
        assert frame["error"]["code"] == "invalid_stream_id"
        assert frame["error"]["param"] == "stream_id"


def test_non_finite_numbers_are_rejected() -> None:
    nan = parse_client_frame('{"type":"response.create","model":"m","temperature":NaN}')
    assert nan["type"] == "error"
    assert nan["error"]["code"] == "invalid_request_error"
    assert nan["error"]["param"] == "temperature"
    infinity = parse_client_frame(json.loads('{"type":"response.create","model":"m","top_p":Infinity}'))
    assert infinity["error"]["code"] == "invalid_request_error"


async def test_frame_over_four_mib_is_rejected_and_closes_1008() -> None:
    raw = '{"type":"response.create","model":"m","input":"' + ("x" * MAX_FRAME_BYTES) + '"}'
    session = ResponsesWebSocketSession(service=ScriptedService(), client=object())
    harness = Harness(session, [raw])
    try:
        await harness.wait(lambda sent: bool(_events_for(sent, "error")))
        await _wait_until(lambda: harness.ws.closed_with is not None)
    finally:
        await harness.aclose()

    error = _events_for(harness.ws.sent, "error")[0]
    assert error["error"]["code"] == "invalid_request_error"
    assert harness.ws.closed_with == 1008


class HugeFrameService:
    """Emits one valid turn whose terminal frame blows the outbound cap."""

    def __init__(self):
        self.store = InMemoryResponsesStore()

    async def stream_turn_events(self, raw_request, client, *, transaction_logger=None, local_cache=None, **kwargs):
        yield ResponsesStreamEvent(
            "response.created",
            {"type": "response.created", "response": {"id": "resp_big", "status": "in_progress"}},
        )
        yield ResponsesStreamEvent(
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_big",
                    "status": "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "x" * MAX_FRAME_BYTES}]}],
                },
            },
        )


async def test_outbound_frame_over_four_mib_closes_1008() -> None:
    session = ResponsesWebSocketSession(service=HugeFrameService(), client=object())
    harness = Harness(session, [_create(stream_id="lane-big")])
    try:
        await harness.wait(lambda sent: bool(_events_for(sent, "error")), timeout=10.0)
        await _wait_until(lambda: harness.ws.closed_with is not None, timeout=10.0)
    finally:
        await harness.aclose()

    assert _events_for(harness.ws.sent, "error")[0]["error"]["code"] == "invalid_request_error"
    assert harness.ws.closed_with == 1008

