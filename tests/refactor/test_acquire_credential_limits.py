import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.usage.manager import UsageManager
from rotator_library.usage import config as usage_config
from rotator_library.usage.config import load_provider_usage_config
from rotator_library.providers.provider_interface import ProviderInterface
from rotator_library.error_handler import NoAvailableKeysError


class LimitProvider(ProviderInterface):
    provider_env_name = "limit"

    async def get_models(self, api_key, client):
        return []


class BalancedLimitProvider(LimitProvider):
    default_rotation_mode = "balanced"


class OptimalMultiplierProvider(LimitProvider):
    default_optimal_priority_multipliers = {1: 2}


class ModeDefaultLimitProvider(LimitProvider):
    default_rotation_mode = "balanced"
    default_max_concurrent_per_key_balanced = 1
    default_max_concurrent_per_key_sequential = 1
    default_optimal_concurrent_per_key_balanced = 1
    default_optimal_concurrent_per_key_sequential = 1


def test_acquire_blocks_on_active_concurrency():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    manager = UsageManager(
        provider="limit",
        file_path=temp_path,
        provider_plugins={"limit": LimitProvider},
        max_concurrent_per_key=1,
    )

    async def run_test():
        await manager.initialize(["key-1"])
        stable_id = manager._registry.get_stable_id("key-1", "limit")
        state = manager._states[stable_id]
        state.active_requests = 1
        state.max_concurrent = 1

        deadline = time.time() + 0.05
        try:
            await manager.acquire_credential("limit/model", deadline=deadline)
        except NoAvailableKeysError:
            return True
        return False

    assert asyncio.run(run_test()) is True


def test_acquire_fails_when_all_on_cooldown():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    manager = UsageManager(
        provider="limit",
        file_path=temp_path,
        provider_plugins={"limit": LimitProvider},
    )

    async def run_test():
        await manager.initialize(["key-2"])
        stable_id = manager._registry.get_stable_id("key-2", "limit")
        state = manager._states[stable_id]
        await manager._tracking.apply_cooldown(
            state=state,
            reason="rate_limit",
            until=time.time() + 5.0,
            model_or_group="limit/model",
            source="test",
        )

        deadline = time.time() + 0.05
        try:
            await manager.acquire_credential("limit/model", deadline=deadline)
        except NoAvailableKeysError:
            return True
        return False

    assert asyncio.run(run_test()) is True


def test_acquire_prefers_below_optimal_capacity():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    manager = UsageManager(
        provider="limit",
        file_path=temp_path,
        provider_plugins={"limit": BalancedLimitProvider},
        max_concurrent_per_key=-1,
        optimal_concurrent_per_key=1,
    )

    async def run_test():
        await manager.initialize(["key-1", "key-2"])
        stable_1 = manager._registry.get_stable_id("key-1", "limit")
        manager._states[stable_1].active_requests = 1

        ctx = await manager.acquire_credential(
            "limit/model", deadline=time.time() + 0.2
        )
        return ctx.credential

    assert asyncio.run(run_test()) == "key-2"


def test_acquire_stacks_when_all_credentials_at_optimal_capacity():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    manager = UsageManager(
        provider="limit",
        file_path=temp_path,
        provider_plugins={"limit": BalancedLimitProvider},
        max_concurrent_per_key=-1,
        optimal_concurrent_per_key=1,
    )

    async def run_test():
        await manager.initialize(["key-1", "key-2"])
        for accessor in ("key-1", "key-2"):
            stable_id = manager._registry.get_stable_id(accessor, "limit")
            manager._states[stable_id].active_requests = 1

        ctx = await manager.acquire_credential(
            "limit/model", deadline=time.time() + 0.2
        )
        return ctx.credential in {"key-1", "key-2"}

    assert asyncio.run(run_test()) is True


def test_concurrent_acquire_spreads_before_stacking():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    manager = UsageManager(
        provider="limit",
        file_path=temp_path,
        provider_plugins={"limit": BalancedLimitProvider},
        max_concurrent_per_key=-1,
        optimal_concurrent_per_key=1,
    )

    async def run_test():
        await manager.initialize(["key-1", "key-2"])
        contexts = await asyncio.gather(
            manager.acquire_credential("limit/model", deadline=time.time() + 0.5),
            manager.acquire_credential("limit/model", deadline=time.time() + 0.5),
        )
        return {ctx.credential for ctx in contexts}

    assert asyncio.run(run_test()) == {"key-1", "key-2"}


def test_explicit_optimal_concurrency_uses_priority_multiplier():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    config = load_provider_usage_config(
        "limit", {"limit": OptimalMultiplierProvider}
    )
    manager = UsageManager(
        provider="limit",
        file_path=temp_path,
        provider_plugins={"limit": OptimalMultiplierProvider},
        config=config,
        optimal_concurrent_per_key=2,
    )

    async def run_test():
        await manager.initialize(["key-1"], priorities={"key-1": 1})
        stable_id = manager._registry.get_stable_id("key-1", "limit")
        return manager._states[stable_id].optimal_concurrent

    assert asyncio.run(run_test()) == 4


def test_mode_specific_env_concurrency_overrides(monkeypatch):
    monkeypatch.setenv("ROTATION_MODE_LIMIT", "balanced")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS_PER_KEY_LIMIT", "5")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS_PER_KEY_LIMIT_BALANCED", "2")
    monkeypatch.setenv("OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_LIMIT", "4")
    monkeypatch.setenv("OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_LIMIT_BALANCED", "1")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    config = load_provider_usage_config("limit", {"limit": LimitProvider})
    manager = UsageManager(
        provider="limit",
        file_path=temp_path,
        provider_plugins={"limit": LimitProvider},
        config=config,
    )

    async def run_test():
        await manager.initialize(["key-1"])
        stable_id = manager._registry.get_stable_id("key-1", "limit")
        state = manager._states[stable_id]
        return state.max_concurrent, state.optimal_concurrent

    assert asyncio.run(run_test()) == (2, 1)


def test_provider_generic_concurrency_applies_when_no_mode_override(monkeypatch):
    monkeypatch.setenv("ROTATION_MODE_LIMIT", "sequential")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS_PER_KEY_LIMIT", "5")
    monkeypatch.setenv("OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_LIMIT", "4")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    config = load_provider_usage_config("limit", {"limit": LimitProvider})
    manager = UsageManager(
        provider="limit",
        file_path=temp_path,
        provider_plugins={"limit": LimitProvider},
        config=config,
    )

    async def run_test():
        await manager.initialize(["key-1"])
        stable_id = manager._registry.get_stable_id("key-1", "limit")
        state = manager._states[stable_id]
        return state.max_concurrent, state.optimal_concurrent

    assert asyncio.run(run_test()) == (5, 4)


def test_provider_wide_env_overrides_provider_mode_defaults(monkeypatch):
    monkeypatch.setenv("ROTATION_MODE_LIMIT", "balanced")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS_PER_KEY_LIMIT", "-1")
    monkeypatch.setenv("OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_LIMIT", "3")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    config = load_provider_usage_config("limit", {"limit": ModeDefaultLimitProvider})
    manager = UsageManager(
        provider="limit",
        file_path=temp_path,
        provider_plugins={"limit": ModeDefaultLimitProvider},
        config=config,
    )

    async def run_test():
        await manager.initialize(["key-1"])
        stable_id = manager._registry.get_stable_id("key-1", "limit")
        state = manager._states[stable_id]
        return state.max_concurrent, state.optimal_concurrent

    assert asyncio.run(run_test()) == (-1, 3)


def test_mode_specific_env_overrides_provider_wide_env_with_provider_mode_defaults(monkeypatch):
    monkeypatch.setenv("ROTATION_MODE_LIMIT", "balanced")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS_PER_KEY_LIMIT", "-1")
    monkeypatch.setenv("OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_LIMIT", "3")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS_PER_KEY_LIMIT_BALANCED", "2")
    monkeypatch.setenv("OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_LIMIT_BALANCED", "1")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    config = load_provider_usage_config("limit", {"limit": ModeDefaultLimitProvider})
    manager = UsageManager(
        provider="limit",
        file_path=temp_path,
        provider_plugins={"limit": ModeDefaultLimitProvider},
        config=config,
    )

    async def run_test():
        await manager.initialize(["key-1"])
        stable_id = manager._registry.get_stable_id("key-1", "limit")
        state = manager._states[stable_id]
        return state.max_concurrent, state.optimal_concurrent

    assert asyncio.run(run_test()) == (2, 1)


def test_invalid_concurrency_env_logs_warning_and_keeps_provider_default(
    monkeypatch,
):
    warnings = []
    monkeypatch.setattr(usage_config.lib_logger, "warning", warnings.append)
    monkeypatch.setenv("ROTATION_MODE_LIMIT", "balanced")
    monkeypatch.setenv("MAX_CONCURRENT_REQUESTS_PER_KEY_LIMIT", "invalid")
    monkeypatch.setenv("OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_LIMIT", "also-invalid")

    config = load_provider_usage_config("limit", {"limit": ModeDefaultLimitProvider})

    assert config.get_base_max_concurrent() == 1
    assert config.get_base_optimal_concurrent() == 1
    assert any(
        "Invalid MAX_CONCURRENT_REQUESTS_PER_KEY_LIMIT='invalid'" in message
        for message in warnings
    )
    assert any(
        "Invalid OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_LIMIT='also-invalid'" in message
        for message in warnings
    )
