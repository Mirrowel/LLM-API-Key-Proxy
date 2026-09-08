from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from proxy_app import main as proxy_main
from rotator_library.responses import InMemoryResponsesStore, ResponsesService, ResponsesServiceError
from rotator_library.responses.streaming import ResponsesStreamEvent
from rotator_library.responses.types import StoredResponse
from rotator_library.responses.websocket import (
    DEFAULT_MAX_CONNECTION_SECONDS,
    LaneState,
    ResponsesWebSocketFormatter,
    ResponsesWebSocketSession,
    error_frame,
    parse_client_frame,
)


# ---------------------------------------------------------------------------
# Frame parsing
# ---------------------------------------------------------------------------


def test_create_frame_strips_transport_fields_and_reads_lane() -> None:
    frame = parse_client_frame(
        json.dumps(
            {
                "type": "response.create",
                "model": "gpt-test",
                "stream": True,
                "generate": True,
                "stream_id": "agent.lane_1",
                "input": [{"type": "message", "role": "user", "content": "hi"}],
            }
        )
    )
    assert isinstance(frame, object)
    assert frame.stream_id == "agent.lane_1"
    for stripped in ("stream", "background", "generate", "stream_id", "type"):
        assert stripped not in frame.payload
    assert frame.payload["model"] == "gpt-test"
    assert frame.warmup is False


def test_background_and_conversation_conflicts_rejected_pre_flight() -> None:
    background = parse_client_frame(json.dumps({"type": "response.create", "model": "m", "background": True}))
    assert background["error"]["param"] == "background"
    conflict = parse_client_frame(
        json.dumps({"type": "response.create", "model": "m", "conversation": "c1", "previous_response_id": "resp_1"})
    )
    assert conflict["error"]["param"] == "conversation"


def test_explicit_null_stream_id_rejected() -> None:
    null_lane = parse_client_frame(json.dumps({"type": "response.create", "model": "m", "stream_id": None}))
    assert null_lane["error"]["param"] == "stream_id"


def test_warmup_flag_detected_from_generate_false() -> None:
    frame = parse_client_frame(json.dumps({"type": "response.create", "model": "m", "generate": False}))
    assert frame.warmup is True
    assert "generate" not in frame.payload


def test_malformed_unknown_and_cancel_frames_rejected_with_spec_shapes() -> None:
    malformed = parse_client_frame("not json")
    assert malformed["type"] == "error"
    assert malformed["error"]["type"] == "invalid_request_error"
    unknown = parse_client_frame({"type": "response.ponies"})
    assert unknown["error"]["code"] == "invalid_request_error"
    # response.cancel belongs to the Realtime API, not Responses WebSocket.
    cancel = parse_client_frame({"type": "response.cancel"})
    assert cancel["error"]["code"] == "invalid_request_error"
    empty_lane = parse_client_frame({"type": "response.create", "stream_id": ""})
    assert empty_lane["error"]["param"] == "stream_id"
    bad_lane = parse_client_frame({"type": "response.create", "stream_id": "bad id!"})
    assert bad_lane["error"]["param"] == "stream_id"
    long_lane = parse_client_frame({"type": "response.create", "stream_id": "a" * 257})
    assert long_lane["error"]["param"] == "stream_id"
    legal_lane = parse_client_frame({"type": "response.create", "stream_id": "a.-_9"})
    assert legal_lane.stream_id == "a.-_9"


# ---------------------------------------------------------------------------
# Error frame shapes (guide examples verbatim)
# ---------------------------------------------------------------------------


def test_error_frames_match_documented_examples() -> None:
    not_found = error_frame(
        "previous_response_not_found",
        "Previous response with id 'resp_abc' not found.",
        status=400,
        param="previous_response_id",
    )
    assert not_found == {
        "type": "error",
        "status": 400,
        "error": {"code": "previous_response_not_found", "message": "Previous response with id 'resp_abc' not found.", "param": "previous_response_id"},
    }
    limit = error_frame(
        "websocket_connection_limit_reached",
        "Responses websocket connection limit reached (60 minutes). Create a new websocket connection to continue.",
        err_type="invalid_request_error",
    )
    assert limit["error"]["type"] == "invalid_request_error"
    assert limit["error"]["code"] == "websocket_connection_limit_reached"
    assert "(60 minutes)" in limit["error"]["message"]


def test_named_lane_errors_echo_stream_id() -> None:
    frame = error_frame("x", "msg", stream_id="lane-9")
    assert frame["stream_id"] == "lane-9"


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def test_formatter_emits_event_objects_and_drops_sse_artifacts() -> None:
    formatter = ResponsesWebSocketFormatter()
    event = ResponsesStreamEvent(
        "response.created",
        {"type": "response.created", "sequence_number": 3, "response": {"id": "resp_1"}},
    )
    frame = json.loads(formatter.format_stream_event(event, stream_id="lane-a"))
    assert frame["type"] == "response.created"
    assert frame["sequence_number"] == 3
    assert frame["stream_id"] == "lane-a"
    # Named lanes ASSIGN (provider echo never wins).
    assert formatter.format_stream_event(ResponsesStreamEvent("heartbeat", {})) is None
    assert formatter.format_stream_event(ResponsesStreamEvent("done", {}, terminal=True)) is None


# ---------------------------------------------------------------------------
# Session turns
# ---------------------------------------------------------------------------


class FakeService:
    """Serves neutral events via stream_turn_events (the production seam)."""

    def __init__(self, events=None, error=None):
        self.event_lists = [list(events or [])]
        self.error = error
        self.store = InMemoryResponsesStore()
        self.turn_requests: list[dict] = []

    def queue_events(self, events) -> None:
        self.event_lists.append(list(events))

    async def stream_turn_events(self, raw_request, client, *, transaction_logger=None, local_cache=None, **kwargs):
        self.turn_requests.append(dict(raw_request))
        if self.error is not None:
            raise self.error
        if self.event_lists:
            events = self.event_lists.pop(0)
        else:
            events = []
        for event in events:
            yield event


async def _collect(agen):
    return [item async for item in agen]


def _event(name: str, payload: dict, **kwargs) -> ResponsesStreamEvent:
    return ResponsesStreamEvent(name, {"type": name, **payload}, **kwargs)


@pytest.mark.asyncio
async def test_warmup_returns_chainable_id_and_caches_state_session_locally() -> None:
    service = FakeService()
    session = ResponsesWebSocketSession(service=service, client=object())
    frames = await _collect(
        session.handle_frame(
            json.dumps(
                {
                    "type": "response.create",
                    "model": "gpt-test",
                    "generate": False,
                    "instructions": "You are a coder.",
                    "input": [{"type": "message", "role": "user", "content": "ctx"}],
                }
            )
        )
    )
    assert len(frames) == 1
    frame = frames[0]
    assert frame["type"] == "response.completed"
    assert frame["response"]["object"] == "response"
    assert frame["response"]["output"] == []
    response_id = frame["response"]["id"]
    # State is connection-local ONLY: nothing in the global store.
    assert await service.store.get(response_id) is None
    cached = session.local_cache.get(response_id)
    assert cached is not None
    assert cached.request["instructions"] == "You are a coder."
    # Warmup sequences never collide with 0 (per-connection counter).
    # Shared sequence domain: any non-negative monotonic value is legal
    # (the session counter is unified with the turn-event domain).
    assert frame["sequence_number"] >= 0


@pytest.mark.asyncio
async def test_store_false_chain_continues_via_connection_local_cache() -> None:
    """The flagship WebSocket workflow: ZDR chains must not 404 on turn 2.

    Real service, real cache seam: turn 1 (store=false) lands in the
    connection-local cache only; turn 2 continues from it with lineage
    expansion; a FRESH session (connection closed) gets the documented
    previous_response_not_found error.
    """

    captured: list[dict] = []

    class ChainClient:
        async def agenerate(self, payload, *, input_protocol, request=None, **kwargs):
            captured.append(dict(payload))

            async def frames():
                yield 'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_ws_chain","object":"response","status":"in_progress","model":"m","output":[]}}\n\n'
                yield 'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_ws_chain","object":"response","status":"completed","model":"m","output":[{"id":"msg_0","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok"}]}]}}\n\n'

            return frames()

    service = ResponsesService(store=InMemoryResponsesStore())
    session = ResponsesWebSocketSession(service=service, client=ChainClient())
    first = await _collect(
        session.handle_frame(json.dumps({"type": "response.create", "model": "m", "store": False, "input": "first turn"}))
    )
    assert first[-1]["type"] == "response.completed"
    response_id = first[-1]["response"]["id"]
    # store=false: NOT in the global store, IS in the connection-local cache.
    assert await service.store.get(response_id) is None
    assert response_id in session.local_cache

    second = await _collect(
        session.handle_frame(
            json.dumps({"type": "response.create", "model": "m", "store": False, "previous_response_id": response_id, "input": "second turn"})
        )
    )
    assert second[-1]["type"] == "response.completed", second
    # Lineage expansion replayed turn 1's input ahead of turn 2's.
    assert len(captured) == 2
    import json as _json

    second_input = captured[1].get("input")
    second_text = _json.dumps(second_input)
    assert "first turn" in second_text
    assert "second turn" in second_text

    # A FRESH connection (cache died with the session) 404s per the guide.
    fresh = ResponsesWebSocketSession(service=service, client=ChainClient())
    denied = await _collect(
        fresh.handle_frame(json.dumps({"type": "response.create", "model": "m", "store": False, "previous_response_id": response_id, "input": "x"}))
    )
    assert denied[-1]["type"] == "error"
    assert denied[-1]["error"]["code"] == "previous_response_not_found"
    assert denied[-1]["error"]["param"] == "previous_response_id"


@pytest.mark.asyncio
async def test_full_turn_yields_event_objects_with_terminal_discipline() -> None:
    service = FakeService(
        events=[
            _event("response.created", {"sequence_number": 0, "response": {"id": "resp_ws_1", "status": "in_progress"}}),
            _event("response.in_progress", {"sequence_number": 1, "response": {"id": "resp_ws_1"}}),
            _event("response.completed", {"sequence_number": 2, "response": {"id": "resp_ws_1", "status": "completed"}}),
            # Post-terminal junk must never leak (terminal discipline).
            _event("response.output_text.delta", {"delta": "stale"}),
        ]
    )
    session = ResponsesWebSocketSession(service=service, client=object())
    frames = await _collect(
        session.handle_frame(
            json.dumps({"type": "response.create", "model": "gpt-test", "stream_id": "lane-1", "input": "hi"})
        )
    )
    types = [frame["type"] for frame in frames]
    assert types == ["response.created", "response.in_progress", "response.completed"]
    assert all(frame["stream_id"] == "lane-1" for frame in frames)
    assert not any(frame["type"] == "done" for frame in frames)
    assert session.lanes["lane-1"].latest_response_id == "resp_ws_1"
    assert service.turn_requests[0].get("input") == "hi"


@pytest.mark.asyncio
async def test_previous_response_not_found_maps_to_spec_frame_with_param_and_eviction() -> None:
    service = FakeService(error=ResponsesServiceError("Previous response not found: resp_missing", status_code=404, error_type="not_found_error"))
    session = ResponsesWebSocketSession(service=service, client=object())
    session.local_cache["resp_missing"] = StoredResponse(id="resp_missing", model="m", status="completed", response={})
    frames = await _collect(
        session.handle_frame(
            json.dumps({"type": "response.create", "model": "m", "previous_response_id": "resp_missing", "stream_id": "lane-x"})
        )
    )
    assert len(frames) == 1
    frame = frames[0]
    assert frame["type"] == "error"
    assert frame["error"]["code"] == "previous_response_not_found"
    assert frame["error"]["param"] == "previous_response_id"
    assert frame["stream_id"] == "lane-x"
    # Referenced parent evicted from the connection-local cache.
    assert "resp_missing" not in session.local_cache


@pytest.mark.asyncio
async def test_store_failed_policy_covers_all_four_cells() -> None:
    """The failed-turn policy gates the global store AND the ZDR local
    cache — the store=false x store_failed=false cell is the flagship
    WebSocket configuration and must MISS on a chained failed id."""

    from rotator_library.responses.types import ResponsesStoreSettings

    async def run_cell(store: bool, store_failed: bool):
        fail_next = {"state": True}

        class CellClient:
            async def agenerate(self, payload, *, input_protocol, request=None, **kwargs):
                if fail_next["state"]:
                    fail_next["state"] = False

                    async def failing():
                        yield 'event: response.failed\ndata: {"type":"response.failed","response":{"id":"resp_cell","object":"response","status":"failed","model":"m","output":[],"error":{"message":"boom"}}}\n\n'

                    return failing()

                async def ok():
                    yield 'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_cell2","object":"response","status":"completed","model":"m","output":[]}}\n\n'

                return ok()

        service = ResponsesService(
            store=InMemoryResponsesStore(),
            store_settings=ResponsesStoreSettings(store_failed=store_failed),
        )
        session = ResponsesWebSocketSession(service=service, client=CellClient())
        await _collect(
            session.handle_frame(json.dumps({"type": "response.create", "model": "m", "store": store}))
        )
        failed_cached = any(stored.id == "resp_cell" for stored in session.local_cache.values())
        frames = await _collect(
            session.handle_frame(
                json.dumps({"type": "response.create", "model": "m", "store": store, "previous_response_id": "resp_cell"})
            )
        )
        return failed_cached, frames

    # store=false, store_failed=False: the ZDR flagship — failed ids never
    # cached anywhere; the chained turn MISSES.
    failed_cached, frames = await run_cell(False, False)
    assert failed_cached is False
    assert frames[-1]["type"] == "error"

    # store=false, store_failed=True: local cache keeps the failed id (ZDR
    # chains continue past failures).
    failed_cached, frames = await run_cell(False, True)
    assert failed_cached is True
    assert frames[-1]["type"] != "error"

    # store=true, store_failed=False: no store, no local cache — miss.
    failed_cached, frames = await run_cell(True, False)
    assert failed_cached is False
    assert frames[-1]["type"] == "error"

    # store=true, store_failed=True: stored (global), chainable.
    failed_cached, frames = await run_cell(True, True)
    assert failed_cached is True
    assert frames[-1]["type"] != "error"


@pytest.mark.asyncio
async def test_failed_turn_evicts_referenced_parent() -> None:
    service = FakeService(
        events=[
            _event("response.created", {"response": {"id": "resp_f", "status": "in_progress"}}),
            _event("response.failed", {"response": {"id": "resp_f", "status": "failed", "error": {"message": "boom"}}}),
        ]
    )
    session = ResponsesWebSocketSession(service=service, client=object())
    session.local_cache["resp_parent"] = StoredResponse(id="resp_parent", model="m", status="completed", response={})
    frames = await _collect(
        session.handle_frame(json.dumps({"type": "response.create", "model": "m", "previous_response_id": "resp_parent"}))
    )
    assert frames[-1]["type"] == "response.failed"
    assert "resp_parent" not in session.local_cache


@pytest.mark.asyncio
async def test_warmup_chains_into_next_turn_via_real_service() -> None:
    """Warmup ids must chain through the REAL lineage walk (scope + items)."""

    captured: list[dict] = []

    class ChainClient:
        async def agenerate(self, payload, *, input_protocol, request=None, **kwargs):
            captured.append(dict(payload))

            async def frames():
                yield 'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_after_warmup","object":"response","status":"completed","model":"m","output":[]}}\n\n'

            return frames()

    service = ResponsesService(store=InMemoryResponsesStore())
    session = ResponsesWebSocketSession(service=service, client=ChainClient())
    warmup = await _collect(
        session.handle_frame(
            json.dumps({"type": "response.create", "model": "m", "generate": False, "input": "warm context"})
        )
    )
    warmup_id = warmup[0]["response"]["id"]
    turn = await _collect(
        session.handle_frame(
            json.dumps({"type": "response.create", "model": "m", "previous_response_id": warmup_id, "input": "real question"})
        )
    )
    assert turn[-1]["type"] == "response.completed", turn
    # The warmup input replayed ahead of the real question.
    import json as _json

    text = _json.dumps(captured[0].get("input"))
    assert "warm context" in text
    assert "real question" in text


@pytest.mark.asyncio
async def test_local_cache_eviction_under_churn_keeps_most_recent() -> None:
    session = ResponsesWebSocketSession(service=FakeService(), client=object())
    from rotator_library.responses.websocket import _LOCAL_CACHE_MAX_ENTRIES

    for i in range(_LOCAL_CACHE_MAX_ENTRIES + 5):
        session.local_cache[f"resp_{i}"] = StoredResponse(id=f"resp_{i}", model="m", status="completed", response={})
    assert len(session.local_cache) == _LOCAL_CACHE_MAX_ENTRIES
    assert "resp_0" not in session.local_cache
    assert f"resp_{_LOCAL_CACHE_MAX_ENTRIES + 4}" in session.local_cache


@pytest.mark.asyncio
async def test_provider_failure_containing_not_found_phrase_stays_response_failed() -> None:
    """R1 guard: 'DB response not found for query' is an ordinary failure."""

    service = FakeService(
        events=[
            _event("response.created", {"response": {"id": "resp_x", "status": "in_progress"}}),
            _event(
                "response.failed",
                {"response": {"id": "resp_x", "status": "failed", "error": {"type": "server_error", "message": "DB response not found for query"}}},
            ),
        ]
    )
    session = ResponsesWebSocketSession(service=service, client=object())
    frames = await _collect(session.handle_frame(json.dumps({"type": "response.create", "model": "m", "previous_response_id": "resp_parent"})))
    assert frames[-1]["type"] == "response.failed"
    assert frames[-1]["type"] != "error"


@pytest.mark.asyncio
async def test_multi_turn_connection_chains_sequentially() -> None:
    service = FakeService(
        events=[
            _event("response.completed", {"response": {"id": "resp_a", "status": "completed"}}),
        ]
    )
    service.queue_events([_event("response.completed", {"response": {"id": "resp_b", "status": "completed"}})])
    session = ResponsesWebSocketSession(service=service, client=object())
    first = await _collect(session.handle_frame(json.dumps({"type": "response.create", "model": "m"})))
    second = await _collect(session.handle_frame(json.dumps({"type": "response.create", "model": "m", "previous_response_id": "resp_a"})))
    assert first[-1]["response"]["id"] == "resp_a"
    assert second[-1]["response"]["id"] == "resp_b"
    assert service.turn_requests[1]["previous_response_id"] == "resp_a"


@pytest.mark.asyncio
async def test_send_failure_mid_turn_closes_upstream_before_return() -> None:
    """F1: a send failure mid-turn must close the service stream
    cooperatively before run() returns — never deferred to GC."""

    import asyncio

    closed = {"events": False}

    class SlowService:
        # Production shape: an async GENERATOR method (calling it yields
        # the generator directly — never a coroutine).
        async def stream_turn_events(self, raw_request, client, *, transaction_logger=None, local_cache=None, **kwargs):
            try:
                yield _event("response.created", {"response": {"id": "resp_leak", "status": "in_progress"}})
                await asyncio.sleep(60)
            finally:
                closed["events"] = True

    class FailingSocket:
        async def receive_text(self):
            await asyncio.sleep(0)
            return json.dumps({"type": "response.create", "model": "m"})

        async def send_json(self, payload):
            # First send fails: the send-failure path (RuntimeError, not
            # cancellation) must unwind the aclose chain synchronously.
            raise RuntimeError("client went away")

        async def close(self, code=1000):
            pass

    session = ResponsesWebSocketSession(service=SlowService(), client=object())
    await asyncio.wait_for(session.run(FailingSocket()), timeout=5)
    assert closed["events"] is True


def test_warmup_input_items_never_raise_on_odd_shapes() -> None:
    from rotator_library.responses.websocket import _warmup_input_items

    assert _warmup_input_items("text") == ["text"]
    assert _warmup_input_items({"type": "message", "role": "user"}) == [{"type": "message", "role": "user"}]
    assert _warmup_input_items(123) == [123]
    assert _warmup_input_items(["a", "b"]) == ["a", "b"]
    assert _warmup_input_items(None) == []


def test_background_false_strips_but_truthy_rejects() -> None:
    falsy = parse_client_frame(json.dumps({"type": "response.create", "model": "m", "background": False}))
    assert not isinstance(falsy, dict), falsy
    assert "background" not in falsy.payload
    truthy = parse_client_frame(json.dumps({"type": "response.create", "model": "m", "background": True}))
    assert truthy["error"]["param"] == "background"





class FakeWebSocket:
    def __init__(self, incoming: list[str], on_receive=None):
        self._incoming = list(incoming)
        self.sent: list[dict] = []
        self.closed_with = None
        self._on_receive = on_receive

    async def receive_text(self):
        if self._on_receive is not None:
            self._on_receive()
        if not self._incoming:
            raise RuntimeError("closed")
        return self._incoming.pop(0)

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code=1000):
        self.closed_with = code


@pytest.mark.asyncio
async def test_connection_limit_error_frame_and_explicit_close() -> None:
    now = {"t": 0.0}

    def clock():
        return now["t"]

    calls = {"n": 0}

    def on_receive():
        calls["n"] += 1
        if calls["n"] >= 1:
            now["t"] = 3601.0

    ws = FakeWebSocket(
        incoming=[json.dumps({"type": "response.create", "model": "m", "generate": False})],
        on_receive=on_receive,
    )
    session = ResponsesWebSocketSession(service=FakeService(), client=object(), max_connection_seconds=3600.0, clock=clock)
    await session.run(ws)
    limit_frames = [f for f in ws.sent if f.get("error", {}).get("code") == "websocket_connection_limit_reached"]
    assert limit_frames, ws.sent
    assert limit_frames[0]["error"]["type"] == "invalid_request_error"
    assert "(60 minutes)" in limit_frames[0]["error"]["message"]
    assert ws.closed_with == 1000


# ---------------------------------------------------------------------------
# Route-level (TestClient websocket)
# ---------------------------------------------------------------------------


class RouteFakeClient:
    async def agenerate(self, payload, *, input_protocol, request=None, **kwargs):
        async def frames():
            yield 'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_ws_route","object":"response","status":"in_progress","model":"gpt-test","output":[]}}\n\n'
            yield 'event: response.in_progress\ndata: {"type":"response.in_progress","response":{"id":"resp_ws_route","object":"response","status":"in_progress","model":"gpt-test","output":[]}}\n\n'
            yield 'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_ws_route","object":"response","status":"completed","model":"gpt-test","output":[]}}\n\n'

        return frames()


def _ws_client():
    proxy_main.PROXY_API_KEY = None
    proxy_main.ENABLE_RAW_LOGGING = False
    proxy_main.ENABLE_REQUEST_LOGGING = False
    proxy_main.app.state.rotating_client = RouteFakeClient()
    proxy_main.app.state.responses_service = ResponsesService(store=InMemoryResponsesStore())
    return TestClient(proxy_main.app)


def test_websocket_route_streams_a_full_turn() -> None:
    # Bare TestClient (no lifespan): the fake client on app.state must
    # survive — lifespan startup would re-initialize the real one.
    client = _ws_client()
    with client.websocket_connect("/v1/responses") as ws:
        ws.send_json({"type": "response.create", "model": "gpt-test", "input": "hi"})
        first = ws.receive_json()
        assert first["type"] == "response.created", first
        terminal = None
        for _ in range(6):
            frame = ws.receive_json()
            if frame["type"] in {"response.completed", "response.failed", "error"}:
                terminal = frame
                break
        assert terminal is not None and terminal["type"] == "response.completed", terminal


def test_websocket_route_denies_unauthenticated_upgrade() -> None:
    proxy_main.PROXY_API_KEY = "secret-key"
    proxy_main.app.state.rotating_client = RouteFakeClient()
    proxy_main.app.state.responses_service = ResponsesService(store=InMemoryResponsesStore())
    client = TestClient(proxy_main.app)
    with pytest.raises(Exception):
        with client.websocket_connect("/v1/responses"):
            pass
    proxy_main.PROXY_API_KEY = None


def test_websocket_route_accepts_authenticated_upgrade() -> None:
    proxy_main.PROXY_API_KEY = "secret-key"
    proxy_main.app.state.rotating_client = RouteFakeClient()
    proxy_main.app.state.responses_service = ResponsesService(store=InMemoryResponsesStore())
    client = TestClient(proxy_main.app)
    with client.websocket_connect("/v1/responses", headers={"Authorization": "Bearer secret-key"}) as ws:
        ws.send_json({"type": "response.create", "model": "gpt-test", "input": "hi", "generate": False})
        frame = ws.receive_json()
        assert frame["type"] == "response.completed"
    proxy_main.PROXY_API_KEY = None
