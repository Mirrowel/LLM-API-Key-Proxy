"""G2 executor integration: declared stages fire on the real native paths."""

from __future__ import annotations

import pytest

from rotator_library.hooks.types import HookAction, HookResult, PipelineHook, TransportView
from rotator_library.native_provider.context import NativeProviderContext
from rotator_library.native_provider.executor import NativeProviderExecutor
from rotator_library.native_provider.http import NativeHTTPTransport


class StageRecorder(PipelineHook):
    """Records every stage it sees; optionally rewrites payload payloads."""

    name = "stage_recorder"
    stages = ("request_received", "routing_resolved", "credential_selected", "session_resolved",
              "parsed_canonical", "canonical_state_inject_a", "canonical_state_inject_b",
              "transport_basis_selected", "provider_built", "finalizer", "mutated",
              "state_inject_a", "state_inject_b", "validated", "transport_ready", "sent",
              "response_received", "response_state_extract_a", "response_state_extract_b",
              "response_parsed", "response_formatted", "usage_recorded",
              "stream_opened", "stream_event", "stream_assembled", "stream_closed")

    def __init__(self, mutate_at: str = "", marker: dict | None = None):
        self.seen: list[tuple[str, str, bool]] = []
        self._mutate_at = mutate_at
        self._marker = marker or {"_hook": True}

    async def __call__(self, invocation, context):
        self.seen.append((invocation.stage, invocation.direction, invocation.is_terminal))
        if invocation.stage == self._mutate_at and isinstance(invocation.payload, dict):
            payload = dict(invocation.payload)
            payload.update(self._marker)
            return payload
        return None


def _context(**overrides) -> NativeProviderContext:
    base = dict(
        provider="openai_test",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://provider.example/v1/chat/completions",
        input_protocol_name="openai_chat",
        client_protocol_name="openai_chat",
        operation="chat",
        headers={"Authorization": "Bearer test"},
    )
    base.update(overrides)
    return NativeProviderContext(**base)


class _FakeResponse:
    def __init__(self, body: dict, status: int = 200):
        self._body = body
        self.status_code = status

    def json(self):
        return self._body


class _FakeClient:
    def __init__(self, response_body: dict):
        self._response_body = response_body
        self.calls: list[dict] = []

    async def post(self, endpoint, headers=None, json=None, **kwargs):
        self.calls.append({"endpoint": endpoint, "headers": dict(headers or {}), "json": json})
        return _FakeResponse(self._response_body)


def _chat_response() -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-test",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


@pytest.mark.asyncio
async def test_non_streaming_request_stages_fire_in_order():
    recorder = StageRecorder(mutate_at="provider_built")
    context = _context(hook_class_declarations=(recorder,))
    client = _FakeClient(_chat_response())
    executor = NativeProviderExecutor()
    result = await executor.execute(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]},
        context,
        NativeHTTPTransport(client),
    )
    stages = [s for s, d, t in recorder.seen]
    # request side ordering (the stage-correction: finalizer AFTER cache band,
    # validate after finalizer, transport last)
    assert stages.index("provider_built") < stages.index("mutated")
    assert stages.index("mutated") < stages.index("state_inject_a")
    assert stages.index("state_inject_b") < stages.index("validated")
    assert stages.index("validated") < stages.index("transport_ready")
    assert stages.index("transport_ready") < stages.index("sent")
    # response side
    assert stages.index("response_received") < stages.index("response_state_extract_a")
    assert stages.index("response_state_extract_b") < stages.index("response_parsed")
    assert stages.index("response_parsed") < stages.index("response_formatted")
    assert stages.index("response_formatted") < stages.index("usage_recorded")
    # the mutation reached the wire
    assert client.calls[0]["json"].get("_hook") is True
    # and the response survived
    assert result["choices"][0]["message"]["content"] == "hi"


@pytest.mark.asyncio
async def test_transport_slot_rewrites_endpoint_and_headers():
    class UrlFlipper(PipelineHook):
        name = "url_flipper"
        stages = ("transport_ready",)

        async def __call__(self, invocation, context):
            view: TransportView = invocation.transport
            view.endpoint = "https://shadow.example/v1/chat/completions"
            view.headers = dict(view.headers or {})
            view.headers["x-shadow"] = "yes"
            view.changed = True
            return None

    context = _context(hook_class_declarations=(UrlFlipper(),))
    client = _FakeClient(_chat_response())
    await NativeProviderExecutor().execute(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]},
        context,
        NativeHTTPTransport(client),
    )
    assert client.calls[0]["endpoint"] == "https://shadow.example/v1/chat/completions"
    assert client.calls[0]["headers"]["x-shadow"] == "yes"
    # overlay recorded — nothing invisible
    overlays = context.request_transport_overlays
    assert any(o.get("kind") == "transport_rewrite" and "shadow" in str(o.get("endpoint", ""))
               for o in overlays)


@pytest.mark.asyncio
async def test_block_verdict_raises_structured_error():
    from rotator_library.core.errors import StructuredAPIResponseError

    class Blocker(PipelineHook):
        name = "blocker"
        stages = ("provider_built",)

        async def __call__(self, invocation, context):
            return HookResult(action=HookAction.BLOCK, message="not allowed here")

    context = _context(hook_class_declarations=(Blocker(),))
    executor = NativeProviderExecutor()
    with pytest.raises(StructuredAPIResponseError) as excinfo:
        await executor.execute(
            {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]},
            context,
            NativeHTTPTransport(_FakeClient(_chat_response())),
        )
    assert "not allowed here" in str(excinfo.value)


class _FakeStreamClient:
    """Yields chat SSE chunks: two deltas, then a usage-carrying done chunk."""

    def __init__(self, chunks: list):
        self._chunks = chunks

    def stream(self, method, endpoint, headers=None, json=None, **kwargs):
        lines = []
        for chunk in self._chunks:
            import json as _json
            lines.append("data: " + _json.dumps(chunk))
        lines.append("data: [DONE]")
        text = "\n\n".join(lines) + "\n\n"

        class _Resp:
            status_code = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            def aiter_lines(self):
                async def _gen():
                    for line in text.splitlines():
                        yield line
                return _gen()

        return _Resp()


def _stream_chunks() -> list[dict]:
    return [
        {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test",
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": "he"}, "finish_reason": None}]},
        {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test",
         "choices": [{"index": 0, "delta": {"content": "y"}, "finish_reason": "stop"}]},
        {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test",
         "choices": [],
         "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}},
    ]


@pytest.mark.asyncio
async def test_stream_stages_fire_including_terminal_and_cleanup():
    recorder = StageRecorder()
    context = _context(hook_class_declarations=(recorder,))
    executor = NativeProviderExecutor()
    events = []
    async for event in executor.stream(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}], "stream": True},
        context,
        NativeHTTPTransport(_FakeStreamClient(_stream_chunks())),
    ):
        events.append(event)
    stages = [s for s, d, t in recorder.seen]
    # terminal flows through the S2 slot (D3 — no more done-bypass)
    stream_events = [(s, t) for s, d, t in recorder.seen if s == "stream_event"]
    assert any(t for _, t in stream_events), "terminal event must pass through stream_event slot"
    # lifecycle stages
    assert "stream_opened" in stages
    assert "stream_assembled" in stages
    assert "stream_closed" in stages
    assert stages.index("stream_opened") < stages.index("stream_event") < stages.index("stream_assembled")
    assert stages.index("stream_assembled") < stages.index("stream_closed")
    # events actually streamed to the client
    assert [e.type for e in events][-1] == "done"


@pytest.mark.asyncio
async def test_stream_event_drop_and_replace():
    class Filter(PipelineHook):
        name = "filter"
        stages = ("stream_event",)

        def __init__(self):
            self.dropped = 0
            self.replaced = 0

        @staticmethod
        def _text_of(payload):
            delta = getattr(payload, "delta", None)
            blocks = list(getattr(delta, "content", None) or [])
            return "".join(b.text or "" for b in blocks if getattr(b, "type", "") == "text")

        async def __call__(self, invocation, context):
            payload = invocation.payload
            text = self._text_of(payload)
            if "secret" in text:
                self.dropped += 1
                return HookResult(action=HookAction.DROP)
            if text == "he":
                self.replaced += 1
                from dataclasses import replace as _replace
                delta = payload.delta
                blocks = [ _replace(b, text="HE") if getattr(b, "type", "") == "text" else b for b in delta.content ]
                new_event = _replace(payload, delta=_replace(delta, content=blocks))
                return HookResult(action=HookAction.REPLACE, payload=new_event)
            return None

    filt = Filter()
    context = _context(hook_class_declarations=(filt,))
    executor = NativeProviderExecutor()
    chunks = _stream_chunks()
    chunks[0]["choices"][0]["delta"]["content"] = "he"
    chunks.insert(1, {"id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test",
                      "choices": [{"index": 0, "delta": {"content": "secret-bit"}, "finish_reason": None}]})
    texts = []
    async for event in executor.stream(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}], "stream": True},
        context,
        NativeHTTPTransport(_FakeStreamClient(chunks)),
    ):
        if event.type == "message_delta" and event.delta is not None:
            texts.append("".join(b.text or "" for b in (event.delta.content or []) if getattr(b, "type", "") == "text"))
    assert "HE" in texts and "secret-bit" not in texts
    assert filt.dropped == 1 and filt.replaced == 1


@pytest.mark.asyncio
async def test_stream_closed_fires_on_error():
    recorder = StageRecorder()
    context = _context(hook_class_declarations=(recorder,))

    class _BoomStream:
        def stream(self, method, endpoint, headers=None, json=None, **kwargs):
            class _Resp:
                status_code = 200

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *exc):
                    return False

                def aiter_lines(self):
                    async def _gen():
                        yield 'data: {"error": {"message": "boom", "type": "server_error"}}'
                    return _gen()

            return _Resp()

    executor = NativeProviderExecutor()
    with pytest.raises(Exception):
        async for _ in executor.stream(
            {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}], "stream": True},
            context,
            NativeHTTPTransport(_BoomStream()),
        ):
            pass
    assert any(s == "stream_closed" for s, d, t in recorder.seen)


@pytest.mark.asyncio
async def test_fast_path_flips_to_rebuild_on_canonical_hook_edit():
    """A canonical edit at R5 must disable the raw fast path (sent wire = edited)."""

    class CanonicalEditor(PipelineHook):
        name = "canonical_editor"
        stages = ("parsed_canonical",)

        async def __call__(self, invocation, context):
            req = invocation.payload
            from dataclasses import replace as _replace
            return _replace(req, metadata={**dict(getattr(req, "metadata", {}) or {}), "hook_edit": True})

    context = _context(
        raw_client_request={"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]},
        hook_class_declarations=(CanonicalEditor(),),
    )
    client = _FakeClient(_chat_response())
    await NativeProviderExecutor().execute(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}]},
        context,
        NativeHTTPTransport(client),
    )
    # canonical edit -> rebuild basis -> overlay records it
    overlays = context.request_transport_overlays
    assert overlays and overlays[0].get("kind") == "canonical_rebuild"


@pytest.mark.asyncio
async def test_isolation_two_concurrent_runs_share_nothing():
    r1_events: list = []
    r2_events: list = []

    def make_hook(sink):
        class Sink(PipelineHook):
            name = "sink"
            stages = ("parsed_canonical", "response_formatted")

            async def __call__(self, invocation, context):
                sink.append((id(context), context.state))
                context.state["owner"] = id(context)
                return None
        return Sink()

    async def drive(tag: str, sink):
        context = _context(hook_class_declarations=(make_hook(sink),))
        await NativeProviderExecutor().execute(
            {"model": "gpt-test", "messages": [{"role": "user", "content": tag}]},
            context,
            NativeHTTPTransport(_FakeClient(_chat_response())),
        )

    import asyncio as _aio
    await _aio.gather(drive("one", r1_events), drive("two", r2_events))
    ctx1 = {c for c, _ in r1_events}
    ctx2 = {c for c, _ in r2_events}
    assert ctx1 and ctx2 and ctx1.isdisjoint(ctx2), "runs must not share context objects"
