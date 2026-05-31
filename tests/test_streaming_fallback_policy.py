from __future__ import annotations

import json
from types import MethodType

import pytest

from rotator_library.client import executor as executor_module
from rotator_library.client.executor import RequestExecutor
from rotator_library.core.types import RequestContext
from rotator_library.routing import parse_route_target
from rotator_library.routing.types import FallbackGroup
from rotator_library.transaction_logger import TransactionLogger


class StreamFailure(Exception):
    def __init__(self, error_type: str) -> None:
        super().__init__(error_type)
        self.error_type = error_type


def _context(*, logger=None) -> RequestContext:
    targets = (parse_route_target("codex/gpt-5.1-codex"), parse_route_target("openai/gpt-5.1"))
    return RequestContext(
        model="code",
        provider="requested",
        kwargs={"model": "code", "messages": [], "stream": True},
        streaming=True,
        credentials=["cred-a"],
        deadline=9999999999.0,
        transaction_logger=logger,
        routing_targets=targets,
        routing_group_name="code_chain",
        routing_group=FallbackGroup(name="code_chain", targets=targets, failover_on=frozenset({"authentication", "rate_limit"}), stop_on=frozenset({"validation"})),
    )


def _context_never_streaming_fallback(*, logger=None) -> RequestContext:
    context = _context(logger=logger)
    context.routing_group = FallbackGroup(name="code_chain", targets=context.routing_targets, failover_on=frozenset({"rate_limit"}), streaming_policy="never")
    return context


@pytest.mark.asyncio
async def test_streaming_fallback_hard_stops_auth_even_with_group_override() -> None:
    executor = RequestExecutor.__new__(RequestExecutor)
    attempts = []

    async def fake_stream(self, context):
        attempts.append(context.provider)
        if len(attempts) == 1:
            raise StreamFailure("authentication")
        yield "data: [DONE]\n\n"

    executor._execute_streaming = MethodType(fake_stream, executor)

    with pytest.raises(StreamFailure):
        [chunk async for chunk in executor._execute_streaming_with_fallback(_context())]

    assert attempts == ["codex"]


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

    context = _context()
    chunks = [chunk async for chunk in executor._execute_streaming_with_fallback(context)]

    assert attempts == ["codex", "openai"]
    assert chunks[-1] == "data: [DONE]\n\n"
    assert context.routing_attempt_history[0]["error_type"] == "rate_limit"
    assert context.routing_attempt_history[0]["fallback_allowed"] is True
    assert context.routing_attempt_history[1]["success"] is True


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
            yield "data: [DONE]\n\n"
            return
        yield "data: [DONE]\n\n"

    executor._execute_streaming = MethodType(fake_stream, executor)

    chunks = [chunk async for chunk in executor._execute_streaming_with_fallback(_context())]

    assert attempts == ["codex", "openai"]
    assert chunks == ["data: [DONE]\n\n"]


@pytest.mark.asyncio
async def test_streaming_fallback_respects_never_policy_before_output() -> None:
    executor = RequestExecutor.__new__(RequestExecutor)
    attempts = []

    async def fake_stream(self, context):
        attempts.append(context.provider)
        raise StreamFailure("rate_limit")
        yield ""

    executor._execute_streaming = MethodType(fake_stream, executor)

    with pytest.raises(StreamFailure):
        [chunk async for chunk in executor._execute_streaming_with_fallback(_context_never_streaming_fallback())]

    assert attempts == ["codex"]


@pytest.mark.asyncio
async def test_streaming_fallback_exhaustion_trace_uses_sanitized_summaries(tmp_path) -> None:
    executor = RequestExecutor.__new__(RequestExecutor)
    logger = TransactionLogger("routing", "code", parent_dir=tmp_path)

    async def fake_stream(self, context):
        raise StreamFailure("rate_limit")
        yield ""

    executor._execute_streaming = MethodType(fake_stream, executor)

    with pytest.raises(StreamFailure):
        [chunk async for chunk in executor._execute_streaming_with_fallback(_context_never_streaming_fallback(logger=logger))]

    entries = [json.loads(line) for line in (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    exhausted = [entry for entry in entries if entry["pass_name"] == "routing_fallback_exhausted"][-1]
    assert exhausted["metadata"]["fallback_targets"][0]["message"] == ""
    assert exhausted["metadata"]["streaming_policy"] == "never"


def test_stream_timeout_details_merge_into_aggregate_error() -> None:
    error_data = {"error": {"type": "proxy_error", "details": {"attempts": 1}}}
    stream_error = {"error": {"type": "api_connection", "details": {"timeout_type": "ttfb", "timeout_seconds": 0.1}}}

    executor_module._merge_stream_error_details(error_data, stream_error)

    assert error_data["error"]["type"] == "api_connection"
    assert error_data["error"]["details"]["attempts"] == 1
    assert error_data["error"]["details"]["timeout_type"] == "ttfb"
