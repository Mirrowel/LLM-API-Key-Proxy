import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rotator_library.usage.manager import UsageManager
from rotator_library.providers.provider_interface import ProviderInterface


class GroupedProvider(ProviderInterface):
    provider_env_name = "grouped"
    model_quota_groups = {"family": ["model-a", "model-b"]}

    async def get_models(self, api_key, client):
        return []


def test_quota_group_syncs_counts_and_windows():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        temp_path = f.name

    manager = UsageManager(
        provider="grouped",
        file_path=temp_path,
        provider_plugins={"grouped": GroupedProvider},
    )

    async def run_test():
        await manager.initialize(["key-1"])
        with (
            patch(
                "rotator_library.usage.tracking.engine.time.time", return_value=1234.0
            ),
            patch(
                "rotator_library.usage.tracking.windows.time.time", return_value=1234.0
            ),
        ):
            await manager.record_usage("key-1", "grouped/model-a", success=True)

    asyncio.run(run_test())

    stable_id = manager._registry.get_stable_id("key-1", "grouped")
    state = manager._states[stable_id]

    # Sibling models no longer get mirrored per-model counts; the quota
    # group's own stats are the shared aggregation point.
    assert state.model_usage["grouped/model-a"].totals.request_count == 1
    assert state.group_usage["family"].totals.request_count == 1

    group_window = state.group_usage["family"].windows["daily"]
    assert group_window.request_count == 1
    assert group_window.started_at == 1234.0

    # Group is authoritative: the requesting model's window timing is
    # synced from the group window.
    model_a_window = state.model_usage["grouped/model-a"].windows["daily"]
    assert model_a_window.started_at == 1234.0
