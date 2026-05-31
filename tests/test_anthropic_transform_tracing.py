from __future__ import annotations

import json

import pytest

import rotator_library.client.anthropic as anthropic_client_module
from rotator_library.anthropic_compat import AnthropicMessagesRequest
from rotator_library.anthropic_compat.streaming import anthropic_streaming_wrapper
from rotator_library.client.anthropic import AnthropicHandler
from rotator_library.transaction_logger import TransactionLogger


def _trace_entries(log_dir):
    return [json.loads(line) for line in (log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]


class FakeRotatingClient:
    enable_request_logging = True

    async def acompletion(self, **kwargs):
        return {
            "id": "chat_1",
            "model": kwargs["model"],
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }


@pytest.mark.asyncio
async def test_anthropic_handler_traces_conversion_boundaries(tmp_path, monkeypatch) -> None:
    created: list[TransactionLogger] = []

    def logger_factory(provider, model, enabled=True, api_format="ant", parent_dir=None):
        logger = TransactionLogger(provider, model, enabled=enabled, api_format=api_format, parent_dir=tmp_path)
        created.append(logger)
        return logger

    monkeypatch.setattr(anthropic_client_module, "TransactionLogger", logger_factory)
    request = AnthropicMessagesRequest(model="openai/gpt-test", max_tokens=16, messages=[{"role": "user", "content": "hi"}])

    response = await AnthropicHandler(FakeRotatingClient()).messages(request)

    assert response["content"][0]["text"] == "ok"
    pass_names = [entry["pass_name"] for entry in _trace_entries(created[0].log_dir)]
    assert "anthropic_raw_request" in pass_names
    assert "anthropic_to_openai_request" in pass_names
    assert "anthropic_openai_response" in pass_names
    assert "openai_to_anthropic_response" in pass_names
    assert "anthropic_final_response" in pass_names


class ClosingOpenAIStream:
    def __init__(self) -> None:
        self.closed = False
        self._chunks = iter(['data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


class IteratorOnlyCloseStream:
    def __init__(self) -> None:
        self.iterator = IteratorOnlyCloseStreamIterator()

    def __aiter__(self):
        return self.iterator


class IteratorOnlyCloseStreamIterator:
    def __init__(self) -> None:
        self.closed = False
        self._chunks = iter(['data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'])

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_anthropic_stream_traces_and_closes_on_disconnect(tmp_path) -> None:
    logger = TransactionLogger("anthropic", "claude-test", parent_dir=tmp_path)
    stream = ClosingOpenAIStream()

    async def disconnected() -> bool:
        return True

    chunks = [chunk async for chunk in anthropic_streaming_wrapper(stream, "claude-test", is_disconnected=disconnected, transaction_logger=logger)]

    assert chunks == []
    assert stream.closed is True
    pass_names = [entry["pass_name"] for entry in _trace_entries(logger.log_dir)]
    assert "anthropic_stream_source_chunk" in pass_names
    assert "anthropic_stream_disconnected" in pass_names
    assert "anthropic_stream_upstream_closed" in pass_names


@pytest.mark.asyncio
async def test_anthropic_stream_closes_iterator_only_upstream() -> None:
    stream = IteratorOnlyCloseStream()

    async def disconnected() -> bool:
        return True

    chunks = [chunk async for chunk in anthropic_streaming_wrapper(stream, "claude-test", is_disconnected=disconnected)]

    assert chunks == []
    assert stream.iterator.closed is True


@pytest.mark.asyncio
async def test_anthropic_stream_traces_emitted_frames(tmp_path) -> None:
    logger = TransactionLogger("anthropic", "claude-test", parent_dir=tmp_path)

    async def stream():
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        yield "data: [DONE]\n\n"

    chunks = [chunk async for chunk in anthropic_streaming_wrapper(stream(), "claude-test", transaction_logger=logger)]

    assert any("event: message_start" in chunk for chunk in chunks)
    assert any("event: message_stop" in chunk for chunk in chunks)
    pass_names = [entry["pass_name"] for entry in _trace_entries(logger.log_dir)]
    assert "anthropic_stream_message_start" in pass_names
    assert "anthropic_stream_content_delta" in pass_names
    assert "anthropic_stream_message_delta" in pass_names
    assert "anthropic_stream_message_stop" in pass_names
