from __future__ import annotations

import json
from types import MethodType

import pytest

from rotator_library.client.executor import RequestExecutor
from rotator_library.core.types import RequestContext
from rotator_library.routing import parse_route_target
from rotator_library.transaction_logger import TransactionLogger


class StreamFailure(Exception):
    def __init__(self, error_type: str) -> None:
        super().__init__(error_type)
        self.error_type = error_type


def _context(*, logger=None) -> RequestContext:
    return RequestContext(
        model="code",
        provider="requested",
        kwargs={"model": "code", "messages": [], "stream": True},
        streaming=True,
        credentials=["cred-a"],
        deadline=9999999999.0,
        transaction_logger=logger,
        routing_targets=(parse_route_target("codex/gpt-5.1-codex"), parse_route_target("openai/gpt-5.1")),
        routing_group_name="code_chain",
    )


@pytest.mark.asyncio
async def test_streaming_fallback_tries_next_target_before_output() -> None:
    executor = RequestExecutor.__new__(RequestExecutor)
    attempts = []

    async def fake_stream(self, context):
        attempts.append(context.provider)
        if len(attempts) == 1:
            raise StreamFailure("rate_limit")
        yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        yield "data: [DONE]\n\n"

    executor._execute_streaming = MethodType(fake_stream, executor)

    chunks = [chunk async for chunk in executor._execute_streaming_with_fallback(_context())]

    assert attempts == ["codex", "openai"]
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_streaming_fallback_blocks_after_visible_output() -> None:
    executor = RequestExecutor.__new__(RequestExecutor)

    async def fake_stream(self, context):
        yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise StreamFailure("rate_limit")

    executor._execute_streaming = MethodType(fake_stream, executor)

    with pytest.raises(StreamFailure):
        [chunk async for chunk in executor._execute_streaming_with_fallback(_context())]


@pytest.mark.asyncio
async def test_streaming_fallback_trace_records_blocked_after_output(tmp_path) -> None:
    executor = RequestExecutor.__new__(RequestExecutor)
    logger = TransactionLogger("routing", "code", parent_dir=tmp_path)

    async def fake_stream(self, context):
        yield 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise StreamFailure("rate_limit")

    executor._execute_streaming = MethodType(fake_stream, executor)

    with pytest.raises(StreamFailure):
        [chunk async for chunk in executor._execute_streaming_with_fallback(_context(logger=logger))]

    pass_names = [json.loads(line)["pass_name"] for line in (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "routing_stream_target_attempt_started" in pass_names
    assert "routing_stream_target_attempt_failed" in pass_names
    assert "routing_stream_fallback_blocked_after_output" in pass_names


@pytest.mark.asyncio
async def test_streaming_fallback_treats_error_chunk_as_not_visible_output() -> None:
    executor = RequestExecutor.__new__(RequestExecutor)
    attempts = []

    async def fake_stream(self, context):
        attempts.append(context.provider)
        if len(attempts) == 1:
            yield 'data: {"error":{"type":"rate_limit"}}\n\n'
            raise StreamFailure("rate_limit")
        yield "data: [DONE]\n\n"

    executor._execute_streaming = MethodType(fake_stream, executor)

    chunks = [chunk async for chunk in executor._execute_streaming_with_fallback(_context())]

    assert attempts == ["codex", "openai"]
    assert chunks[-1] == "data: [DONE]\n\n"
