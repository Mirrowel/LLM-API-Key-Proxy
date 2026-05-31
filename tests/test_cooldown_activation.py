from __future__ import annotations

import asyncio

import pytest

from rotator_library.client.executor import RequestExecutor, _can_start_stream_provider_cooldown
from rotator_library.cooldown_manager import CooldownManager
from rotator_library.core.types import RequestContext
from rotator_library.error_handler import ClassifiedError
from rotator_library.transaction_logger import TransactionLogger


class FakeCooldown:
    def __init__(self) -> None:
        self.started = []

    async def start_cooldown(self, provider, duration):
        self.started.append((provider, duration))


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

    assert cooldown.started == [("openai", 60)]
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


def test_streaming_provider_cooldown_gate_allows_only_pre_output_failures() -> None:
    assert _can_start_stream_provider_cooldown(None) is True
    assert _can_start_stream_provider_cooldown('data: {"error":{"type":"rate_limit"}}\n\n') is True
    assert _can_start_stream_provider_cooldown('data: {"choices":[{"delta":{"content":"visible"}}]}\n\n') is False
