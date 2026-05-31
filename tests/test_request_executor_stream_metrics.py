from __future__ import annotations

import json
import asyncio

import pytest

from rotator_library.client.streaming import StreamingHandler
from rotator_library.core.errors import StreamedAPIError
from rotator_library.transaction_logger import TransactionLogger


async def _chunks():
    yield {"id": "chunk_1", "choices": [{"delta": {"content": "hi"}}]}
    yield {"id": "chunk_2", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class HangingStream:
    def __init__(self) -> None:
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(1)
        return {"choices": [{"delta": {"content": "late"}}]}

    async def aclose(self) -> None:
        self.closed = True


class DelayedStream:
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
            return {"id": "chunk_1", "choices": [{"delta": {"content": "hi"}}]}
        if self.index == 1:
            self.index += 1
            return {"id": "chunk_2", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class FirstThenHangStream(HangingStream):
    def __init__(self) -> None:
        super().__init__()
        self.index = 0

    async def __anext__(self):
        if self.index == 0:
            self.index += 1
            return {"id": "chunk_1", "choices": [{"delta": {}}]}
        return await super().__anext__()


class DisconnectedRequest:
    async def is_disconnected(self) -> bool:
        return True


class DelayedDisconnectedRequest:
    def __init__(self) -> None:
        self.calls = 0

    async def is_disconnected(self) -> bool:
        self.calls += 1
        await asyncio.sleep(0.01)
        return self.calls >= 1


def _trace_passes(log_dir):
    return [json.loads(line)["pass_name"] for line in (log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_streaming_handler_emits_lifecycle_metrics_without_changing_output(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)

    chunks = [chunk async for chunk in StreamingHandler().wrap_stream(_chunks(), "cred", "openai/gpt-test", transaction_logger=logger)]

    assert chunks[0].startswith("data: ")
    assert chunks[-1] == "data: [DONE]\n\n"
    pass_names = _trace_passes(logger.log_dir)
    assert "stream_started" in pass_names
    assert "stream_first_byte" in pass_names
    assert "stream_first_visible_output" in pass_names
    assert "stream_completed" in pass_names
    assert "stream_metrics_final" in pass_names


@pytest.mark.asyncio
async def test_stream_trace_metrics_can_be_disabled_without_changing_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STREAM_TRACE_METRICS", "false")
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)

    chunks = [chunk async for chunk in StreamingHandler().wrap_stream(_chunks(), "cred", "openai/gpt-test", transaction_logger=logger)]

    assert chunks[0].startswith("data: ")
    assert chunks[-1] == "data: [DONE]\n\n"
    if (logger.log_dir / "transform_trace.jsonl").exists():
        pass_names = _trace_passes(logger.log_dir)
        assert "stream_started" not in pass_names
        assert "stream_metrics_final" not in pass_names


@pytest.mark.asyncio
async def test_streaming_handler_closes_upstream_on_client_disconnect(monkeypatch) -> None:
    monkeypatch.delenv("STREAM_TTFB_TIMEOUT_SECONDS", raising=False)
    stream = HangingStream()

    chunks = [chunk async for chunk in StreamingHandler().wrap_stream(stream, "cred", "openai/gpt-test", request=DisconnectedRequest())]

    assert chunks == []
    assert stream.closed is True


@pytest.mark.asyncio
async def test_streaming_handler_closes_upstream_when_disconnect_happens_during_wait(monkeypatch) -> None:
    monkeypatch.delenv("STREAM_TTFB_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", raising=False)
    stream = HangingStream()

    chunks = [chunk async for chunk in StreamingHandler().wrap_stream(stream, "cred", "openai/gpt-test", request=DelayedDisconnectedRequest())]

    assert chunks == []
    assert stream.closed is True


@pytest.mark.asyncio
async def test_streaming_handler_emits_configured_heartbeats(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", "0.01")
    monkeypatch.delenv("STREAM_TTFB_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("STREAM_STALL_TIMEOUT_SECONDS", raising=False)

    chunks = [chunk async for chunk in StreamingHandler().wrap_stream(DelayedStream(), "cred", "openai/gpt-test")]

    assert any(chunk.startswith(": heartbeat") for chunk in chunks)
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_streaming_handler_ttfb_timeout_closes_upstream(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_TTFB_TIMEOUT_SECONDS", "0.01")
    monkeypatch.delenv("STREAM_HEARTBEAT_INTERVAL_SECONDS", raising=False)
    stream = HangingStream()

    with pytest.raises(StreamedAPIError) as exc:
        _ = [chunk async for chunk in StreamingHandler().wrap_stream(stream, "cred", "openai/gpt-test")]

    assert stream.closed is True
    assert exc.value.data["error"]["details"]["timeout_type"] == "ttfb"


@pytest.mark.asyncio
async def test_stream_timeout_closes_upstream_even_when_disconnect_close_disabled(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_TTFB_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setenv("STREAM_CANCEL_UPSTREAM_ON_DISCONNECT", "false")
    stream = HangingStream()

    with pytest.raises(StreamedAPIError):
        _ = [chunk async for chunk in StreamingHandler().wrap_stream(stream, "cred", "openai/gpt-test")]

    assert stream.closed is True


@pytest.mark.asyncio
async def test_streaming_handler_stall_timeout_after_first_byte(monkeypatch) -> None:
    monkeypatch.setenv("STREAM_STALL_TIMEOUT_SECONDS", "0.01")
    monkeypatch.delenv("STREAM_TTFB_TIMEOUT_SECONDS", raising=False)
    stream = FirstThenHangStream()
    chunks = []

    with pytest.raises(StreamedAPIError) as exc:
        async for chunk in StreamingHandler().wrap_stream(stream, "cred", "openai/gpt-test"):
            chunks.append(chunk)

    assert chunks and chunks[0].startswith("data: ")
    assert stream.closed is True
    assert exc.value.data["error"]["details"]["timeout_type"] == "stall"


@pytest.mark.asyncio
async def test_streaming_handler_passes_through_formatted_sse_chunks(monkeypatch) -> None:
    async def formatted_stream():
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'

    chunks = [chunk async for chunk in StreamingHandler().wrap_stream(formatted_stream(), "cred", "openai/gpt-test")]

    assert chunks[0] == 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
    assert chunks[-1] == "data: [DONE]\n\n"
