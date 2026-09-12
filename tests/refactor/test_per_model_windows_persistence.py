import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.usage.manager import UsageManager
from rotator_library.providers.provider_interface import ProviderInterface


class MockProvider(ProviderInterface):
    provider_env_name = "mock"

    async def get_models(self, api_key, client):
        return []


def test_per_model_windows_persist_independently():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    manager = UsageManager(
        provider="mock",
        file_path=temp_path,
        provider_plugins={"mock": MockProvider},
    )

    async def run_test():
        await manager.initialize(["key-1"])

        with (
            patch(
                "rotator_library.usage.tracking.engine.time.time", return_value=1000.0
            ),
            patch(
                "rotator_library.usage.tracking.windows.time.time", return_value=1000.0
            ),
        ):
            await manager.record_usage("key-1", "mock/model-a", success=True)

        with (
            patch(
                "rotator_library.usage.tracking.engine.time.time", return_value=2000.0
            ),
            patch(
                "rotator_library.usage.tracking.windows.time.time", return_value=2000.0
            ),
        ):
            await manager.record_usage("key-1", "mock/model-b", success=True)

        await manager.save(force=True)

    asyncio.run(run_test())

    stable_id = manager._registry.get_stable_id("key-1", "mock")
    state = manager._states[stable_id]

    model_a = state.model_usage["mock/model-a"].windows["daily"].started_at
    model_b = state.model_usage["mock/model-b"].windows["daily"].started_at
    assert model_a == 1000.0
    assert model_b == 2000.0

    manager_reload = UsageManager(
        provider="mock",
        file_path=temp_path,
        provider_plugins={"mock": MockProvider},
    )

    async def reload():
        await manager_reload.initialize(["key-1"])

    asyncio.run(reload())

    stable_id_reload = manager_reload._registry.get_stable_id("key-1", "mock")
    state_reload = manager_reload._states[stable_id_reload]
    assert state_reload.model_usage["mock/model-a"].windows["daily"].started_at == 1000.0
    assert state_reload.model_usage["mock/model-b"].windows["daily"].started_at == 2000.0
