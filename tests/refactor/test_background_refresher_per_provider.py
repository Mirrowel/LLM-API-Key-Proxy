import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.background_refresher import BackgroundRefresher


class JobProvider:
    provider_env_name = "job"

    def __init__(self):
        self.calls = []

    def get_background_job_config(self):
        return {"name": "quota_refresh", "interval": 1, "run_on_start": True}

    async def run_background_job(self, usage_manager, credentials):
        self.calls.append((usage_manager, tuple(credentials)))


class FakeClient:
    def __init__(self, provider, usage_manager):
        self._provider = provider
        self.all_credentials = {"job": ["cred-1"]}
        self.usage_managers = {"job": usage_manager}

    def _get_provider_instance(self, provider_name):
        return self._provider


def test_background_job_uses_per_provider_manager():
    provider = JobProvider()
    client = FakeClient(provider, usage_manager=object())
    refresher = BackgroundRefresher(client)

    config = provider.get_background_job_config()

    async def run_test():
        task = asyncio.create_task(
            refresher._run_provider_background_job(
                "job", provider, client.all_credentials["job"], config
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_test())

    assert provider.calls
    assert provider.calls[0][0] is client.usage_managers["job"]
