from __future__ import annotations

import asyncio

import pytest

from rotator_library.client.executor import RequestExecutor, RoutingExecutionError, _can_start_stream_provider_cooldown
from rotator_library.cooldown_manager import CooldownManager
from rotator_library.core.types import RequestContext
from rotator_library.error_handler import ClassifiedError
from rotator_library.transaction_logger import TransactionLogger


class FakeCooldown:
    def __init__(self) -> None:
        self.started = []
        self.scoped_started = []
        self.waits = []

    async def start_cooldown(self, provider, duration):
        self.started.append((provider, duration))

    async def start_scoped_cooldown(self, provider, duration, *, model=None, scope="provider", reason=None):
        self.scoped_started.append((provider, duration, model, scope, reason))

    async def get_max_remaining(self, provider, *, model=None):
        self.waits.append((provider, model))
        return 0


class BudgetCooldown(FakeCooldown):
    def __init__(self, remaining: float) -> None:
        super().__init__()
        self.remaining = remaining

    async def get_max_remaining(self, provider, *, model=None):
        self.waits.append((provider, model))
        return self.remaining


@pytest.mark.asyncio
async def test_start_cooldown_extends_but_does_not_shorten() -> None:
    manager = CooldownManager()

    await manager.start_cooldown("provider", 30)
    initial = await manager.get_remaining_cooldown("provider")
    await asyncio.sleep(0.01)
    await manager.start_cooldown("provider", 1)
    after_shorter = await manager.get_remaining_cooldown("provider")
    await manager.start_cooldown("provider", 60)
    after_longer = await manager.get_remaining_cooldown("provider")

    assert after_shorter > 25
    assert after_shorter <= initial
    assert after_longer > after_shorter


@pytest.mark.asyncio
async def test_model_cooldown_is_independent_from_provider_cooldown() -> None:
    manager = CooldownManager()

    await manager.start_scoped_cooldown("provider", 30, model="model-a", scope="model", reason="capacity")

    assert await manager.is_scoped_cooling_down("provider", model="model-a", scope="model") is True
    assert await manager.is_scoped_cooling_down("provider", model="model-b", scope="model") is False
    assert await manager.is_cooling_down("provider") is False


@pytest.mark.asyncio
async def test_max_remaining_uses_provider_or_model_scope() -> None:
    manager = CooldownManager()

    await manager.start_cooldown("provider", 1)
    await manager.start_scoped_cooldown("provider", 30, model="model-a", scope="model")

    assert await manager.get_max_remaining("provider", model="model-a") > 20
    assert 0 < await manager.get_max_remaining("provider", model="model-b") <= 1


@pytest.mark.asyncio
async def test_cooldown_snapshot_reports_scopes() -> None:
    manager = CooldownManager()

    await manager.start_scoped_cooldown("provider", 30, model="model-a", scope="model", reason="capacity")
    snapshot = await manager.snapshot()

    assert snapshot[0].provider == "provider"
    assert snapshot[0].scope == "model"
    assert snapshot[0].model == "model-a"
    assert snapshot[0].reason == "capacity"


def _classified(error_type: str, retry_after=None) -> ClassifiedError:
    return ClassifiedError(error_type, original_exception=Exception(error_type), retry_after=retry_after)


def _executor(cooldown) -> RequestExecutor:
    executor = RequestExecutor.__new__(RequestExecutor)
    executor._cooldown = cooldown
    from rotator_library.retry_policy import FailureHistory

    executor._failure_history = FailureHistory()
    return executor


def _context(logger) -> RequestContext:
    return RequestContext(
        model="openai/gpt-test",
        provider="openai",
        kwargs={"model": "openai/gpt-test"},
        streaming=False,
        credentials=["cred"],
        deadline=9999999999.0,
        transaction_logger=logger,
    )


@pytest.mark.asyncio
async def test_large_retry_after_starts_provider_cooldown_and_traces(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SMALL_COOLDOWN_RETRY_THRESHOLD", "10")
    monkeypatch.setenv("PROVIDER_COOLDOWN_MIN_SECONDS", "10")
    cooldown = FakeCooldown()
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)

    await _executor(cooldown)._maybe_start_provider_cooldown(
        "openai",
        _classified("rate_limit", retry_after=60),
        context=_context(logger),
    )

    assert cooldown.scoped_started == [("openai", 60, None, "provider", "retry_after")]
    trace = (logger.log_dir / "transform_trace.jsonl").read_text(encoding="utf-8")
    assert "provider_cooldown_started" in trace


@pytest.mark.asyncio
async def test_small_retry_after_skips_provider_cooldown(monkeypatch) -> None:
    monkeypatch.setenv("SMALL_COOLDOWN_RETRY_THRESHOLD", "10")
    cooldown = FakeCooldown()

    await _executor(cooldown)._maybe_start_provider_cooldown(
        "openai",
        _classified("rate_limit", retry_after=3),
        context=None,
    )

    assert cooldown.started == []
    assert cooldown.scoped_started == []


@pytest.mark.asyncio
async def test_model_capacity_starts_model_scoped_cooldown(monkeypatch) -> None:
    monkeypatch.setenv("PROVIDER_COOLDOWN_DEFAULT_SECONDS", "30")
    monkeypatch.setenv("PROVIDER_COOLDOWN_MIN_SECONDS", "10")
    cooldown = FakeCooldown()
    executor = _executor(cooldown)

    await executor._maybe_start_provider_cooldown(
        "openai",
        _classified("server_error"),
        context=None,
        model="gpt-5",
        original_error=Exception("MODEL_CAPACITY_EXHAUSTED"),
    )

    assert cooldown.scoped_started == [("openai", 30, "gpt-5", "model", "model_capacity_cooldown")]
    assert executor._failure_history.snapshot()[0].scope == "model"


@pytest.mark.asyncio
async def test_wait_for_cooldown_uses_model_scope_when_available() -> None:
    cooldown = FakeCooldown()

    await _executor(cooldown)._wait_for_cooldown("openai", 9999999999.0, model="gpt-5")

    assert cooldown.waits == [("openai", "gpt-5")]


@pytest.mark.asyncio
async def test_wait_for_cooldown_exceeding_budget_fails_fast() -> None:
    cooldown = BudgetCooldown(remaining=60)

    with pytest.raises(RoutingExecutionError) as exc:
        await _executor(cooldown)._wait_for_cooldown("openai", 1.0, model="gpt-5")

    assert exc.value.error_type == "rate_limit"
    assert cooldown.waits == [("openai", "gpt-5")]


@pytest.mark.asyncio
async def test_generic_transient_records_history_before_starting_cooldown(monkeypatch) -> None:
    monkeypatch.setenv("PROVIDER_BACKOFF_THRESHOLD", "2")
    monkeypatch.setenv("PROVIDER_COOLDOWN_DEFAULT_SECONDS", "10")
    monkeypatch.setenv("PROVIDER_COOLDOWN_MIN_SECONDS", "10")
    cooldown = FakeCooldown()
    executor = _executor(cooldown)

    await executor._maybe_start_provider_cooldown("openai", _classified("server_error"), context=None, model="gpt-5")
    assert cooldown.scoped_started == []
    assert executor._failure_history.snapshot()[0].reason == "transient_backoff_threshold_not_met"

    await executor._maybe_start_provider_cooldown("openai", _classified("server_error"), context=None, model="gpt-5")
    assert cooldown.scoped_started == [("openai", 10, None, "provider", "default_transient_cooldown")]


def test_streaming_provider_cooldown_gate_allows_only_pre_output_failures() -> None:
    assert _can_start_stream_provider_cooldown(None) is True
    assert _can_start_stream_provider_cooldown('data: {"error":{"type":"rate_limit"}}\n\n') is True
    assert _can_start_stream_provider_cooldown('data: {"choices":[{"delta":{"content":"visible"}}]}\n\n') is False
    assert _can_start_stream_provider_cooldown('data: {"usage":{"total_tokens":1}}\n\n', emitted_output=True) is False
