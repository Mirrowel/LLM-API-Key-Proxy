import unittest
import os
import sys
import time
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rotator_library.usage.config import ProviderUsageConfig, FairCycleConfig, load_provider_usage_config
from rotator_library.usage.selection.engine import SelectionEngine
from rotator_library.usage.selection.strategies.sequential import SequentialStrategy
from rotator_library.usage.types import SelectionContext
from rotator_library.usage.types import CredentialState, RotationMode, LimitResult, LimitCheckResult


class SelectionEngineTests(unittest.TestCase):
    def _make_state(self, stable_id: str, accessor: str, active_requests: int = 0):
        state = CredentialState(stable_id=stable_id, provider="gemini", accessor=accessor)
        state.active_requests = active_requests
        return state

    def test_fair_cycle_retry_uses_keyword_args(self):
        config = ProviderUsageConfig(
            rotation_mode=RotationMode.SEQUENTIAL,
            fair_cycle=FairCycleConfig(enabled=True),
        )
        limit_engine = Mock()
        window_manager = Mock()
        engine = SelectionEngine(config, limit_engine, window_manager)

        states = {
            "cred_a": self._make_state("cred_a", "a"),
            "cred_b": self._make_state("cred_b", "b"),
        }

        call_count = {"n": 0}

        def check_all(state, model, quota_group):
            if call_count["n"] == 0:
                return LimitCheckResult.blocked(LimitResult.BLOCKED_FAIR_CYCLE, "fair cycle")
            return LimitCheckResult.ok()

        def try_reset(provider, model, quota_group, states_arg, candidates, priorities):
            call_count["n"] += 1
            return True

        limit_engine.check_all.side_effect = check_all
        engine._try_fair_cycle_reset = Mock(side_effect=try_reset)

        selected = engine.select(
            provider="gemini",
            model="gemini-2.5-pro",
            states=states,
            quota_group="group-a",
            session_id="session-1",
        )

        self.assertIn(selected, states)
        self.assertEqual(engine._try_fair_cycle_reset.call_count, 1)

    def test_sequential_uses_group_sticky_when_no_session(self):
        config = ProviderUsageConfig(rotation_mode=RotationMode.SEQUENTIAL)
        limit_engine = Mock()
        limit_engine.check_all.return_value = LimitCheckResult.ok()
        window_manager = Mock()
        engine = SelectionEngine(config, limit_engine, window_manager)

        states = {
            "cred_a": self._make_state("cred_a", "a"),
            "cred_b": self._make_state("cred_b", "b"),
        }

        selected_1 = engine.select(
            provider="gemini",
            model="gemini-2.5-pro",
            states=states,
            quota_group="group-a",
            session_id=None,
        )

        selected_2 = engine.select(
            provider="gemini",
            model="gemini-2.5-pro",
            states=states,
            quota_group="group-a",
            session_id=None,
        )

        self.assertIn(selected_1, states)
        self.assertEqual(selected_1, selected_2)

    def test_sequential_sticky_scope_uses_model_only_with_session(self):
        strategy = SequentialStrategy()
        states = {
            "cred_a": self._make_state("cred_a", "a"),
            "cred_b": self._make_state("cred_b", "b"),
        }

        no_session_context = SelectionContext(
            provider="gemini",
            model="gemini-2.5-pro",
            quota_group="shared-group",
            candidates=["cred_a", "cred_b"],
            priorities={"cred_a": 1, "cred_b": 1},
            usage_counts={"cred_a": 0, "cred_b": 0},
            rotation_mode=RotationMode.SEQUENTIAL,
            rotation_tolerance=0.0,
            deadline=0.0,
            session_id=None,
        )
        strategy.select(no_session_context, states)

        self.assertIsNotNone(
            strategy.get_current("gemini", "shared-group", None)
        )
        self.assertIsNone(
            strategy.get_current("gemini", "gemini-2.5-pro", None)
        )

        session_context = SelectionContext(
            provider="gemini",
            model="gemini-2.5-pro",
            quota_group="shared-group",
            candidates=["cred_a", "cred_b"],
            priorities={"cred_a": 1, "cred_b": 1},
            usage_counts={"cred_a": 0, "cred_b": 0},
            rotation_mode=RotationMode.SEQUENTIAL,
            rotation_tolerance=0.0,
            deadline=0.0,
            session_id="session-1",
        )
        strategy.select(session_context, states)

        self.assertIsNotNone(
            strategy.get_current("gemini", "gemini-2.5-pro", "session-1")
        )
        self.assertIsNone(
            strategy.get_current("gemini", "shared-group", "session-1")
        )

    def test_sequential_spreads_new_sessions_across_equal_credentials(self):
        strategy = SequentialStrategy()
        states = {
            f"cred_{idx}": self._make_state(f"cred_{idx}", str(idx))
            for idx in range(4)
        }
        selected = set()

        for idx in range(12):
            context = SelectionContext(
                provider="gemini",
                model="gemini-2.5-pro",
                quota_group="shared-group",
                candidates=list(states.keys()),
                priorities={key: 1 for key in states},
                usage_counts={key: 0 for key in states},
                rotation_mode=RotationMode.SEQUENTIAL,
                rotation_tolerance=0.0,
                deadline=0.0,
                session_id=f"session-{idx}",
            )
            selected.add(strategy.select(context, states))

        self.assertGreater(len(selected), 1)

    def test_sequential_session_first_pick_honors_priority(self):
        strategy = SequentialStrategy()
        states = {
            "paid_a": self._make_state("paid_a", "paid-a"),
            "paid_b": self._make_state("paid_b", "paid-b"),
            "free_a": self._make_state("free_a", "free-a"),
        }
        selected = set()

        for idx in range(8):
            context = SelectionContext(
                provider="gemini",
                model="gemini-2.5-pro",
                quota_group="shared-group",
                candidates=list(states.keys()),
                priorities={"paid_a": 1, "paid_b": 1, "free_a": 3},
                usage_counts={key: 0 for key in states},
                rotation_mode=RotationMode.SEQUENTIAL,
                rotation_tolerance=0.0,
                deadline=0.0,
                session_id=f"session-{idx}",
            )
            selected.add(strategy.select(context, states))

        self.assertTrue(selected <= {"paid_a", "paid_b"})

    def test_sequential_initial_pick_uses_affinity_key_when_present(self):
        strategy = SequentialStrategy()
        states = {
            f"cred_{idx}": self._make_state(f"cred_{idx}", str(idx))
            for idx in range(4)
        }

        first = SelectionContext(
            provider="gemini",
            model="gemini-2.5-pro",
            quota_group="shared-group",
            candidates=list(states.keys()),
            priorities={key: 1 for key in states},
            usage_counts={key: 0 for key in states},
            rotation_mode=RotationMode.SEQUENTIAL,
            rotation_tolerance=0.0,
            deadline=0.0,
            session_id="live-session-a",
            session_affinity_key="stable-affinity",
        )
        second = SelectionContext(
            provider="gemini",
            model="gemini-2.5-pro",
            quota_group="shared-group",
            candidates=list(states.keys()),
            priorities={key: 1 for key in states},
            usage_counts={key: 0 for key in states},
            rotation_mode=RotationMode.SEQUENTIAL,
            rotation_tolerance=0.0,
            deadline=0.0,
            session_id="live-session-b",
            session_affinity_key="stable-affinity",
        )

        self.assertEqual(strategy.select(first, states), strategy.select(second, states))

    def test_sequential_prunes_expired_session_sticky_entries(self):
        strategy = SequentialStrategy(sticky_entry_ttl_seconds=1)
        states = {
            "cred_a": self._make_state("cred_a", "a"),
            "cred_b": self._make_state("cred_b", "b"),
        }
        context = SelectionContext(
            provider="gemini",
            model="gemini-2.5-pro",
            quota_group="shared-group",
            candidates=list(states.keys()),
            priorities={key: 1 for key in states},
            usage_counts={key: 0 for key in states},
            rotation_mode=RotationMode.SEQUENTIAL,
            rotation_tolerance=0.0,
            deadline=0.0,
            session_id="session-1",
        )

        strategy.select(context, states)
        self.assertIsNotNone(strategy.get_current("gemini", "gemini-2.5-pro", "session-1"))

        for entry in strategy._current.values():
            entry.last_seen = time.time() - 2

        self.assertIsNone(strategy.get_current("gemini", "gemini-2.5-pro", "session-1"))

    def test_sequential_trims_oldest_sticky_entries_when_over_max(self):
        strategy = SequentialStrategy()
        strategy.max_sticky_entries = 2
        states = {
            f"cred_{idx}": self._make_state(f"cred_{idx}", str(idx))
            for idx in range(3)
        }

        for idx in range(3):
            context = SelectionContext(
                provider="gemini",
                model="gemini-2.5-pro",
                quota_group="shared-group",
                candidates=list(states.keys()),
                priorities={key: 1 for key in states},
                usage_counts={key: 0 for key in states},
                rotation_mode=RotationMode.SEQUENTIAL,
                rotation_tolerance=0.0,
                deadline=0.0,
                session_id=f"session-{idx}",
            )
            strategy.select(context, states)
            for entry in strategy._current.values():
                entry.last_seen -= 10

        self.assertLessEqual(len(strategy._current), 2)
        self.assertIsNone(strategy.get_current("gemini", "gemini-2.5-pro", "session-0"))

    def test_session_sticky_env_config_is_parsed(self):
        with patch.dict(
            os.environ,
            {
                "SESSION_STICKY_ENTRY_TTL_SECONDS_GEMINI": "123",
                "SESSION_STICKY_MAX_ENTRIES_GEMINI": "456",
            },
        ):
            config = load_provider_usage_config("gemini", {})

        self.assertEqual(config.session_sticky_entry_ttl_seconds, 123)
        self.assertEqual(config.session_sticky_max_entries, 456)


if __name__ == "__main__":
    unittest.main()
