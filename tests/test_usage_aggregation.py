import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.usage.config import ProviderUsageConfig, WindowDefinition
from rotator_library.usage.manager import UsageManager
from rotator_library.usage.types import CredentialState, WindowStats
from rotator_library.providers.provider_interface import UsageResetConfigDef


class DummyProvider:
    def get_model_quota_group(self, model: str):
        if model in {"test/model-a", "test/model-b"}:
            return "grp"
        return None

    def get_models_in_quota_group(self, group: str):
        if group == "grp":
            return ["model-a", "model-b"]
        return []


class PerModelWindowProvider:
    """Provider config that stores per-model quota data under the models key."""

    usage_reset_configs = {
        "default": UsageResetConfigDef(
            window_seconds=5 * 60 * 60,
            mode="per_model",
            description="5-hour per-model window",
            field_name="models",
        )
    }
    tier_priorities = {"standard": 2}
    default_tier_priority = 10


class UsageAggregationTests(unittest.IsolatedAsyncioTestCase):
    async def test_group_and_credential_window_aggregation(self):
        config = ProviderUsageConfig(
            windows=[
                WindowDefinition.rolling(
                    "5h",
                    18000,
                    is_primary=True,
                    applies_to="model",
                )
            ]
        )
        manager = UsageManager(
            provider="test",
            provider_plugins={"test": DummyProvider},
            config=config,
        )
        state = CredentialState(stable_id="id", provider="test", accessor="acc")
        state.window_definitions = config.windows

        await manager._tracking.record_success(
            state,
            "test/model-a",
            quota_group="grp",
            request_count=3,
            prompt_tokens=30,
            approx_cost=0.3,
        )
        await manager._tracking.record_success(
            state,
            "test/model-b",
            quota_group="grp",
            request_count=4,
            prompt_tokens=20,
            approx_cost=0.2,
        )

        group_window = state.group_usage["grp"].windows["5h"]
        self.assertEqual(group_window.request_count, 7)
        self.assertEqual(group_window.success_count, 7)
        self.assertEqual(group_window.failure_count, 0)
        self.assertEqual(group_window.total_tokens, 50)
        self.assertAlmostEqual(group_window.approx_cost, 0.5)

        self.assertEqual(state.totals.request_count, 7)
        self.assertEqual(state.totals.success_count, 7)
        self.assertEqual(state.totals.failure_count, 0)
        self.assertEqual(state.totals.total_tokens, 50)
        self.assertAlmostEqual(state.totals.approx_cost, 0.5)


class UsageWindowReconcileTests(unittest.TestCase):
    def test_per_model_reset_config_uses_singular_applies_to(self):
        manager = UsageManager(
            provider="test",
            provider_plugins={"test": PerModelWindowProvider},
            config=ProviderUsageConfig(),
        )
        state = CredentialState(
            stable_id="id",
            provider="test",
            accessor="acc",
            tier="standard",
        )

        definitions = manager._get_window_definitions_for_state(state)

        self.assertEqual(definitions[0].applies_to, "model")

    def test_reconcile_window_counts_increases_successes(self):
        config = ProviderUsageConfig(
            windows=[
                WindowDefinition.rolling(
                    "5h",
                    18000,
                    is_primary=True,
                    applies_to="model",
                )
            ]
        )
        manager = UsageManager(provider="test", config=config)
        window = WindowStats(
            name="5h",
            request_count=3,
            success_count=2,
            failure_count=1,
        )

        manager._reconcile_window_counts(window, 5)

        self.assertEqual(window.request_count, 5)
        self.assertEqual(window.success_count, 4)
        self.assertEqual(window.failure_count, 1)

    def test_reconcile_window_counts_preserves_failures(self):
        config = ProviderUsageConfig(
            windows=[
                WindowDefinition.rolling(
                    "5h",
                    18000,
                    is_primary=True,
                    applies_to="model",
                )
            ]
        )
        manager = UsageManager(provider="test", config=config)
        window = WindowStats(
            name="5h",
            request_count=5,
            success_count=3,
            failure_count=2,
        )

        manager._reconcile_window_counts(window, 4)

        self.assertEqual(window.request_count, 4)
        self.assertEqual(window.success_count, 2)
        self.assertEqual(window.failure_count, 2)
