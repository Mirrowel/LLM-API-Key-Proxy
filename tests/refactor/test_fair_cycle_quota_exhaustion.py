"""Comprehensive tests for fair cycle quota-based exhaustion."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from rotator_library.usage.config import (
    FairCycleConfig,
    ProviderUsageConfig,
    get_default_windows,
)
from rotator_library.usage.limits.fair_cycle import FairCycleChecker
from rotator_library.usage.tracking.windows import WindowManager
from rotator_library.usage.types import (
    CredentialState,
    FairCycleState,
    ModelStats,
    WindowStats,
    LimitResult,
    TrackingMode,
)


# =============================================================================
# QUOTA-BASED EXHAUSTION TESTS
# =============================================================================


class TestQuotaBasedExhaustion:
    """Tests for fair cycle quota threshold exhaustion."""

    def _make_state_with_usage(
        self, stable_id: str, model: str, request_count: int, window_limit: int
    ) -> CredentialState:
        """Helper to create credential state with usage."""
        state = CredentialState(stable_id=stable_id, provider="test", accessor="key")
        usage = ModelStats()
        usage.windows["daily"] = WindowStats(
            name="daily",
            request_count=request_count,
            started_at=time.time() - 1000,
            reset_at=time.time() + 80000,
            limit=window_limit,
        )
        state.model_usage[model] = usage

        # Initialize fair cycle state with cycle_request_count
        state.fair_cycle[model] = FairCycleState(
            model_or_group=model,
            cycle_request_count=request_count,  # Proxy-only usage
        )
        return state

    def test_below_threshold_not_exhausted(self):
        """Credential below quota threshold is not exhausted."""
        config = FairCycleConfig(enabled=True, quota_threshold=1.0)
        windows = WindowManager(get_default_windows())
        checker = FairCycleChecker(config, windows)

        # Usage: 5/10 (50%) with threshold 1.0 (100%)
        state = self._make_state_with_usage("cred-1", "model-x", 5, 10)

        result = checker.check(state, "model-x", None)
        assert result.allowed is True
        assert state.fair_cycle["model-x"].exhausted is False

    def test_at_threshold_becomes_exhausted(self):
        """Credential at quota threshold becomes exhausted."""
        config = FairCycleConfig(enabled=True, quota_threshold=1.0)
        windows = WindowManager(get_default_windows())
        checker = FairCycleChecker(config, windows)

        # Usage: 10/10 (100%) with threshold 1.0 (100%)
        state = self._make_state_with_usage("cred-1", "model-x", 10, 10)

        result = checker.check(state, "model-x", None)

        # Should be marked exhausted
        assert state.fair_cycle["model-x"].exhausted is True
        assert state.fair_cycle["model-x"].exhausted_reason == "quota_threshold"
        # But blocked (waiting for other credentials)
        assert result.allowed is False
        assert result.result == LimitResult.BLOCKED_FAIR_CYCLE

    def test_above_threshold_exhausted(self):
        """Credential above quota threshold is exhausted."""
        config = FairCycleConfig(enabled=True, quota_threshold=1.0)
        windows = WindowManager(get_default_windows())
        checker = FairCycleChecker(config, windows)

        # Usage: 15/10 (150%) with threshold 1.0 (100%)
        state = self._make_state_with_usage("cred-1", "model-x", 15, 10)

        result = checker.check(state, "model-x", None)

        assert state.fair_cycle["model-x"].exhausted is True
        assert result.allowed is False

    def test_half_threshold(self):
        """Quota threshold of 0.5 exhausts at 50% usage."""
        config = FairCycleConfig(enabled=True, quota_threshold=0.5)
        windows = WindowManager(get_default_windows())
        checker = FairCycleChecker(config, windows)

        # Usage: 5/10 (50%) with threshold 0.5 (50%)
        state = self._make_state_with_usage("cred-1", "model-x", 5, 10)

        result = checker.check(state, "model-x", None)

        assert state.fair_cycle["model-x"].exhausted is True
        assert result.allowed is False

    def test_double_threshold(self):
        """Quota threshold of 2.0 allows 200% of window limit."""
        config = FairCycleConfig(enabled=True, quota_threshold=2.0)
        windows = WindowManager(get_default_windows())
        checker = FairCycleChecker(config, windows)

        # Usage: 15/10 (150%) with threshold 2.0 (200%)
        state = self._make_state_with_usage("cred-1", "model-x", 15, 10)

        result = checker.check(state, "model-x", None)

        # Not yet at 200%, so not exhausted
        assert state.fair_cycle["model-x"].exhausted is False
        assert result.allowed is True

        # Now at 200%
        state.fair_cycle["model-x"].cycle_request_count = 20
        result = checker.check(state, "model-x", None)

        assert state.fair_cycle["model-x"].exhausted is True
        assert result.allowed is False

    def test_disabled_fair_cycle_always_allows(self):
        """Disabled fair cycle always allows."""
        config = FairCycleConfig(enabled=False, quota_threshold=1.0)
        windows = WindowManager(get_default_windows())
        checker = FairCycleChecker(config, windows)

        # Even with 200% usage, should be allowed
        state = self._make_state_with_usage("cred-1", "model-x", 20, 10)

        result = checker.check(state, "model-x", None)
        assert result.allowed is True


# =============================================================================
# FAIR CYCLE RESET TESTS
# =============================================================================


class TestFairCycleReset:
    """Tests for fair cycle reset behavior."""

    def test_reset_clears_exhaustion(self):
        """Reset clears exhaustion state."""
        config = FairCycleConfig(enabled=True)
        checker = FairCycleChecker(config)

        state = CredentialState(stable_id="cred-1", provider="test", accessor="key")
        state.fair_cycle["model-x"] = FairCycleState(
            model_or_group="model-x",
            exhausted=True,
            exhausted_reason="quota_threshold",
            cycle_request_count=100,
        )

        checker.reset(state, "model-x", None)

        assert state.fair_cycle["model-x"].exhausted is False
        assert state.fair_cycle["model-x"].exhausted_reason is None
        assert state.fair_cycle["model-x"].cycle_request_count == 0

    def test_reset_cycle_clears_all_states(self):
        """reset_cycle() clears all credential states."""
        config = FairCycleConfig(enabled=True)
        checker = FairCycleChecker(config)

        states = []
        for i in range(3):
            state = CredentialState(
                stable_id=f"cred-{i}", provider="test", accessor=f"key-{i}"
            )
            state.fair_cycle["model-x"] = FairCycleState(
                model_or_group="model-x",
                exhausted=True,
                cycle_request_count=50,
            )
            states.append(state)

        checker.reset_cycle("test", "model-x", states)

        for state in states:
            assert state.fair_cycle["model-x"].exhausted is False
            assert state.fair_cycle["model-x"].cycle_request_count == 0


# =============================================================================
# TRACKING MODE TESTS
# =============================================================================


class TestTrackingModes:
    """Tests for different fair cycle tracking modes."""

    def test_model_group_mode_tracks_per_model(self):
        """MODEL_GROUP mode tracks separately per model."""
        config = FairCycleConfig(enabled=True, tracking_mode=TrackingMode.MODEL_GROUP)
        checker = FairCycleChecker(config)

        state = CredentialState(stable_id="cred-1", provider="test", accessor="key")
        state.fair_cycle["model-a"] = FairCycleState(
            model_or_group="model-a", exhausted=True
        )
        state.fair_cycle["model-b"] = FairCycleState(
            model_or_group="model-b", exhausted=False
        )

        # model-a should be blocked
        result_a = checker.check(state, "model-a", None)
        assert result_a.allowed is False

        # model-b should be allowed
        result_b = checker.check(state, "model-b", None)
        assert result_b.allowed is True

    def test_credential_mode_tracks_globally(self):
        """CREDENTIAL mode tracks across all models."""
        config = FairCycleConfig(enabled=True, tracking_mode=TrackingMode.CREDENTIAL)
        checker = FairCycleChecker(config)

        state = CredentialState(stable_id="cred-1", provider="test", accessor="key")
        # Global key "_credential_" is used for credential mode (from FAIR_CYCLE_GLOBAL_KEY)
        state.fair_cycle["_credential_"] = FairCycleState(
            model_or_group="_credential_", exhausted=True
        )

        # Both models should be blocked (global exhaustion)
        result_a = checker.check(state, "model-a", None)
        result_b = checker.check(state, "model-b", None)

        assert result_a.allowed is False
        assert result_b.allowed is False

    def test_quota_group_overrides_model(self):
        """When quota_group is provided, it's used instead of model."""
        config = FairCycleConfig(enabled=True, tracking_mode=TrackingMode.MODEL_GROUP)
        checker = FairCycleChecker(config)

        state = CredentialState(stable_id="cred-1", provider="test", accessor="key")
        state.fair_cycle["flash-group"] = FairCycleState(
            model_or_group="flash-group", exhausted=True
        )

        # Even though model is different, quota_group matches
        result = checker.check(state, "gemini-2.0-flash", quota_group="flash-group")
        assert result.allowed is False
