import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.usage.manager import UsageManager
from rotator_library.core.types import RequestCompleteResult
from rotator_library.providers.provider_interface import ProviderInterface


class HookProvider(ProviderInterface):
    provider_env_name = "hook"

    async def get_models(self, api_key, client):
        return []

    def on_request_complete(self, credential, model, success, response, error):
        return RequestCompleteResult(
            count_override=3,
            cooldown_override=10.0,
            force_exhausted=True,
        )


def test_hook_overrides_usage_and_cooldown():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    manager = UsageManager(
        provider="hook",
        file_path=temp_path,
        provider_plugins={"hook": HookProvider},
    )

    async def run_test():
        await manager.initialize(["key-1"])
        ctx = await manager.acquire_credential("hook/model", deadline=9999999999.0)
        async with ctx:
            ctx.mark_success(prompt_tokens=1, completion_tokens=1)
        return ctx

    ctx = asyncio.run(run_test())

    stable_id = manager._registry.get_stable_id("key-1", "hook")
    state = manager._states[stable_id]
    assert state.totals.success_count == 3
    assert state.get_cooldown("hook/model") is not None
    assert state.fair_cycle["hook/model"].exhausted is True
