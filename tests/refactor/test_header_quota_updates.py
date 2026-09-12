import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.usage.manager import UsageManager
from rotator_library.providers.provider_interface import ProviderInterface


class HeaderProvider(ProviderInterface):
    provider_env_name = "header"

    async def get_models(self, api_key, client):
        return []


def test_headers_update_primary_window_limit_and_reset():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    manager = UsageManager(
        provider="header",
        file_path=temp_path,
        provider_plugins={"header": HeaderProvider},
    )

    async def run_test():
        await manager.initialize(["key-2"])

        with (
            patch(
                "rotator_library.usage.tracking.engine.time.time", return_value=1000.0
            ),
            patch(
                "rotator_library.usage.tracking.windows.time.time", return_value=1000.0
            ),
        ):
            stable_id = manager._registry.get_stable_id("key-2", "header")
            await manager._record_success(
                stable_id,
                "model-x",
                None,
                prompt_tokens=0,
                completion_tokens=0,
                prompt_tokens_cache_read=0,
                approx_cost=0.0,
                response_headers={
                    "x-ratelimit-limit": "10",
                    "x-ratelimit-reset": "60",
                },
            )

    asyncio.run(run_test())

    stable_id = manager._registry.get_stable_id("key-2", "header")
    state = manager._states[stable_id]
    window = state.model_usage["model-x"].windows["daily"]
    assert window.limit == 10
    assert window.reset_at == 1060.0
    assert state.model_usage["model-x"].totals.request_count == 1
