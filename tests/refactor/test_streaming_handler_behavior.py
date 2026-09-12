"""Live pipeline stream-policy behavior (G4 stream core).

These tests exercise ``NeutralStreamPipeline.run`` — the operational stream
layer that replaced the retired ``StreamingHandler.wrap_stream``. They pin the
same behavioral contracts the legacy handler owned:

- TTFB timeout raises ``StreamedAPIError`` and closes upstream;
- stall timeout after the first byte raises ``StreamedAPIError``;
- configured heartbeat intervals emit SSE comment frames;
- client disconnects cancel and close the upstream source;
- completion accounting (finish reason + usage) survives the neutral events.

Sources are async-iterator objects yielding ``UnifiedStreamEvent`` so an
unstarted source can still be closed (mirroring the real upstream seam). Raw
chat-wire chunks are driven through ``ChatWireStreamAdapter`` exactly as the
client executor composes them.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.client.stream_ops import (
    ChatWireStreamAdapter,
    NeutralStreamPipeline,
)
from rotator_library.core.errors import StreamedAPIError
from rotator_library.protocols.types import (
    ContentBlock,
    ProtocolContext,
    UnifiedMessage,
    UnifiedStreamEvent,
)
from tests.refactor.helpers import FakeCredentialContext


def _protocol_context(model: str = "mock/model") -> ProtocolContext:
    return ProtocolContext(
        provider="mock",
        model=model,
        source_protocol="openai_chat",
        target_protocol="openai_chat",
        input_protocol="openai_chat",
        provider_protocol="openai_chat",
        client_protocol="openai_chat",
        transport="sse",
    )


def _pipeline(model: str = "mock/model", **kwargs: Any) -> NeutralStreamPipeline:
    return NeutralStreamPipeline(
        client_protocol_name="openai_chat",
        protocol_context=_protocol_context(model),
        model=model,
        **kwargs,
    )


def _delta(text: str) -> UnifiedStreamEvent:
    return UnifiedStreamEvent(
        type="message_delta",
        source_protocol="openai_chat",
        delta=UnifiedMessage(
            role="assistant",
            content=[ContentBlock(type="text", text=text)],
        ),
    )


async def _chunks():
    yield {"choices": [{"delta": {"tool_calls": [{"id": "tool-1"}]}, "finish_reason": "stop"}]}
    yield {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }


class _HangingSource:
    """An async source that never produces an event until cancelled."""

    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(1)
        return _delta("late")

    async def aclose(self) -> None:
        self.closed = True


class _FirstThenHangSource(_HangingSource):
    """Emits one event (locking the first byte) and then hangs."""

    def __init__(self) -> None:
        super().__init__()
        self.index = 0

    async def __anext__(self):
        if self.index == 0:
            self.index += 1
            return _delta("hi")
        return await super().__anext__()


class _DelayedSource:
    """Emits one event after a delay, then completes."""

    def __init__(self, delay: float = 0.03) -> None:
        self.delay = delay
        self.index = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index == 0:
            self.index += 1
            await asyncio.sleep(self.delay)
            return _delta("hi")
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class _DisconnectedRequest:
    async def is_disconnected(self) -> bool:
        return True


class _DelayedDisconnectedRequest:
    def __init__(self) -> None:
        self.calls = 0

    async def is_disconnected(self) -> bool:
        self.calls += 1
        await asyncio.sleep(0.01)
        return self.calls >= 1


@pytest.mark.asyncio
async def test_pipeline_finish_reason_and_usage() -> None:
    context = FakeCredentialContext("key")
    pipeline = _pipeline(cred_context=context)
    adapter = ChatWireStreamAdapter("mock/model", repair_state=pipeline.repair_state)

    chunks = [frame async for frame in pipeline.run(adapter.events(_chunks(), pipeline.usage))]

    assert chunks[-1] == "data: [DONE]\n\n"
    assert '"finish_reason": null' in chunks[0]
    # G4 ruling: the provider's own final-frame reason wins over tools-seen.
    assert '"finish_reason": "stop"' in chunks[1]
    assert context.success_tokens == {
        "prompt": 3,
        "completion": 2,
        "thinking": 0,
        "prompt_cached": 0,
        "prompt_cache_write": 0,
    }


@pytest.mark.asyncio
async def test_pipeline_ttfb_timeout_raises_and_closes_upstream(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_TTFB_TIMEOUT_SECONDS", "0.01")
    monkeypatch.delenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", raising=False)
    source = _HangingSource()

    with pytest.raises(StreamedAPIError) as exc:
        _ = [frame async for frame in _pipeline().run(source)]

    assert source.closed is True
    assert exc.value.data["error"]["details"]["timeout_type"] == "ttfb"


@pytest.mark.asyncio
async def test_pipeline_timeout_closes_upstream_even_when_disconnect_close_disabled(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_TTFB_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("STREAM_CANCEL_UPSTREAM_ON_DISCONNECT", "false")
    source = _HangingSource()

    with pytest.raises(StreamedAPIError):
        _ = [frame async for frame in _pipeline().run(source)]

    assert source.closed is True


@pytest.mark.asyncio
async def test_pipeline_stall_timeout_after_first_byte(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_STALL_TIMEOUT_SECONDS", "0.01")
    monkeypatch.delenv("STREAM_TTFB_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", raising=False)
    source = _FirstThenHangSource()
    chunks: List[str] = []

    with pytest.raises(StreamedAPIError) as exc:
        async for chunk in _pipeline().run(source):
            chunks.append(chunk)

    assert chunks and chunks[0].startswith("data: ")
    assert source.closed is True
    assert exc.value.data["error"]["details"]["timeout_type"] == "stall"


@pytest.mark.asyncio
async def test_pipeline_emits_configured_heartbeats(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", "0.01")
    monkeypatch.delenv("STREAM_TTFB_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("STREAM_STALL_TIMEOUT_SECONDS", raising=False)

    chunks = [frame async for frame in _pipeline().run(_DelayedSource())]

    assert any(chunk.startswith(": heartbeat") for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_pipeline_closes_upstream_on_client_disconnect(monkeypatch) -> None:
    monkeypatch.delenv("STREAM_TTFB_TIMEOUT_SECONDS", raising=False)
    source = _HangingSource()

    chunks = [
        frame
        async for frame in _pipeline(request=_DisconnectedRequest()).run(source)
    ]

    assert chunks == []
    assert source.closed is True


@pytest.mark.asyncio
async def test_pipeline_closes_upstream_when_disconnect_happens_during_wait(monkeypatch) -> None:
    monkeypatch.delenv("STREAM_TTFB_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", raising=False)
    source = _HangingSource()

    chunks = [
        frame
        async for frame in _pipeline(request=_DelayedDisconnectedRequest()).run(source)
    ]

    assert chunks == []
    assert source.closed is True
