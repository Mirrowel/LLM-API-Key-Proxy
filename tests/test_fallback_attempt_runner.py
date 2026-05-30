from __future__ import annotations

import pytest

from rotator_library.routing import FallbackAttemptRunner, FallbackExhaustedError, RoutingDecision, parse_route_target
from rotator_library.routing.types import FallbackGroup


class ClassifiedFailure(Exception):
    def __init__(self, error_type: str, *, emitted_output: bool = False) -> None:
        super().__init__(error_type)
        self.error_type = error_type
        self.emitted_output = emitted_output


def _decision() -> RoutingDecision:
    return RoutingDecision(
        requested_model="code",
        group_name="code_chain",
        targets=(parse_route_target("codex/gpt-5.1-codex"), parse_route_target("openai/gpt-5.1")),
        reason="model_route_group",
    )


@pytest.mark.asyncio
async def test_attempt_runner_returns_first_success_without_fallback() -> None:
    calls = []

    async def attempt(target, index):
        calls.append(target.prefixed_model)
        return {"target": target.prefixed_model}

    result = await FallbackAttemptRunner().run(_decision(), attempt)

    assert result == {"target": "codex/gpt-5.1-codex"}
    assert calls == ["codex/gpt-5.1-codex"]


@pytest.mark.asyncio
async def test_attempt_runner_falls_back_on_retryable_error() -> None:
    calls = []

    async def attempt(target, index):
        calls.append(target.prefixed_model)
        if index == 0:
            raise ClassifiedFailure("rate_limit")
        return {"target": target.prefixed_model}

    result = await FallbackAttemptRunner().run(_decision(), attempt)

    assert result == {"target": "openai/gpt-5.1"}
    assert calls == ["codex/gpt-5.1-codex", "openai/gpt-5.1"]


@pytest.mark.asyncio
async def test_attempt_runner_stops_on_permanent_error() -> None:
    async def attempt(target, index):
        raise ClassifiedFailure("validation")

    with pytest.raises(FallbackExhaustedError) as exc:
        await FallbackAttemptRunner().run(_decision(), attempt)

    assert len(exc.value.attempts) == 1
    assert exc.value.attempts[0].error_type == "validation"


@pytest.mark.asyncio
async def test_attempt_runner_blocks_stream_fallback_after_output() -> None:
    async def attempt(target, index):
        raise ClassifiedFailure("rate_limit", emitted_output=True)

    with pytest.raises(FallbackExhaustedError) as exc:
        await FallbackAttemptRunner().run(_decision(), attempt, stream=True)

    assert len(exc.value.attempts) == 1
    assert exc.value.attempts[0].emitted_output is True


@pytest.mark.asyncio
async def test_attempt_runner_honors_group_policy_overrides() -> None:
    group = FallbackGroup(
        name="custom",
        targets=_decision().targets,
        failover_on=frozenset({"authentication"}),
        stop_on=frozenset({"validation"}),
    )
    calls = []

    async def attempt(target, index):
        calls.append(index)
        if index == 0:
            raise ClassifiedFailure("authentication")
        return {"target": target.prefixed_model}

    result = await FallbackAttemptRunner().run_group(_decision(), group, attempt)

    assert result == {"target": "openai/gpt-5.1"}
    assert calls == [0, 1]
