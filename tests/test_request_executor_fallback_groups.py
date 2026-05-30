from __future__ import annotations

import json
from types import MethodType

import pytest

from rotator_library.client.executor import RequestExecutor
from rotator_library.core.types import RequestContext
from rotator_library.routing import parse_route_target
from rotator_library.transaction_logger import TransactionLogger


class ClassifiedFailure(Exception):
    def __init__(self, error_type: str) -> None:
        super().__init__(error_type)
        self.error_type = error_type


def _context(*, routing_targets=None, logger=None) -> RequestContext:
    return RequestContext(
        model="code",
        provider="requested",
        kwargs={"model": "code", "messages": []},
        streaming=False,
        credentials=["cred-a"],
        deadline=9999999999.0,
        transaction_logger=logger,
        routing_targets=routing_targets,
        routing_group_name="code_chain" if routing_targets else None,
    )


def _executor_with_attempts(attempts):
    executor = RequestExecutor.__new__(RequestExecutor)

    async def fake_execute(self, context):
        attempts.append(context)
        if len(attempts) == 1:
            raise ClassifiedFailure("rate_limit")
        return {"id": "ok", "model": context.model}

    executor._execute_non_streaming = MethodType(fake_execute, executor)
    return executor


@pytest.mark.asyncio
async def test_non_streaming_fallback_group_tries_next_target_on_retryable_error() -> None:
    attempts = []
    targets = (parse_route_target("codex/gpt-5.1-codex"), parse_route_target("openai/gpt-5.1"))

    result = await _executor_with_attempts(attempts)._execute_non_streaming_with_fallback(_context(routing_targets=targets))

    assert result == {"id": "ok", "model": "openai/gpt-5.1"}
    assert [attempt.provider for attempt in attempts] == ["codex", "openai"]
    assert [attempt.kwargs["model"] for attempt in attempts] == ["codex/gpt-5.1-codex", "openai/gpt-5.1"]


@pytest.mark.asyncio
async def test_non_streaming_fallback_group_stops_on_permanent_error() -> None:
    executor = RequestExecutor.__new__(RequestExecutor)
    attempts = []

    async def fake_execute(self, context):
        attempts.append(context)
        raise ClassifiedFailure("validation")

    executor._execute_non_streaming = MethodType(fake_execute, executor)

    with pytest.raises(ClassifiedFailure):
        await executor._execute_non_streaming_with_fallback(
            _context(routing_targets=(parse_route_target("codex/gpt-5.1-codex"), parse_route_target("openai/gpt-5.1")))
        )

    assert len(attempts) == 1


@pytest.mark.asyncio
async def test_non_streaming_fallback_group_handles_structured_error_response() -> None:
    executor = RequestExecutor.__new__(RequestExecutor)
    attempts = []

    async def fake_execute(self, context):
        attempts.append(context)
        if len(attempts) == 1:
            return {"error": {"type": "proxy_all_credentials_exhausted", "details": {"normal_error_summary": "2 rate_limit"}}}
        return {"id": "ok", "model": context.model}

    executor._execute_non_streaming = MethodType(fake_execute, executor)

    result = await executor._execute_non_streaming_with_fallback(
        _context(routing_targets=(parse_route_target("codex/gpt-5.1-codex"), parse_route_target("openai/gpt-5.1")))
    )

    assert result == {"id": "ok", "model": "openai/gpt-5.1"}


@pytest.mark.asyncio
async def test_non_streaming_fallback_group_emits_routing_trace(tmp_path) -> None:
    attempts = []
    logger = TransactionLogger("routing", "code", parent_dir=tmp_path)
    targets = (parse_route_target("codex/gpt-5.1-codex"), parse_route_target("openai/gpt-5.1"))

    await _executor_with_attempts(attempts)._execute_non_streaming_with_fallback(_context(routing_targets=targets, logger=logger))

    pass_names = [json.loads(line)["pass_name"] for line in (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "routing_decision" in pass_names
    assert pass_names.count("routing_target_attempt_started") == 2
    assert "routing_target_attempt_failed" in pass_names
    assert "routing_fallback_selected" in pass_names
    assert "routing_target_attempt_succeeded" in pass_names
