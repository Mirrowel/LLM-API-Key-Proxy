"""Comprehensive tests for custom cap limits - advanced features."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from rotator_library.usage.config import (
    CustomCapConfig,
    ProviderUsageConfig,
    get_default_windows,
    CapMode,
    CooldownMode,
)
from rotator_library.usage.limits.custom_caps import CustomCapChecker
from rotator_library.usage.limits.engine import LimitEngine
from rotator_library.usage.tracking.windows import WindowManager
from rotator_library.usage.types import (
    CredentialState,
    ModelStats,
    WindowStats,
    LimitResult,
)


def make_state_with_usage(
    stable_id: str,
    model: str,
    request_count: int,
    window_limit: int,
    window_reset_at: float = None,
    priority: int = 1,
) -> CredentialState:
    """Helper to create credential state with usage."""
    state = CredentialState(stable_id=stable_id, provider="test", accessor="key")
    state.priority = priority
    usage = ModelStats()
    reset_at = window_reset_at or (time.time() + 3600)  # Default: reset in 1 hour
    usage.windows["daily"] = WindowStats(
        name="daily",
        request_count=request_count,
        started_at=time.time() - 1000,
        reset_at=reset_at,
        limit=window_limit,
    )
    state.model_usage[model] = usage
    return state


# =============================================================================
# CAPS HIGHER THAN API LIMITS
# =============================================================================


class TestCapsHigherThanApiLimits:
    """Tests for custom caps that are HIGHER than API limits."""

    def test_cap_higher_than_limit_allows_beyond_api(self):
        """Cap of 200 with API limit of 100 allows up to 200."""
        cap = CustomCapConfig(
            tier_key="1",
            model_or_group="model-x",
            max_requests=200,  # Higher than API's 100
            cooldown_mode=CooldownMode.FIXED,
            cooldown_value=300,
        )
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker([cap], windows)

        # At 100 (API limit) - should still be allowed by custom cap
        state = make_state_with_usage("cred-1", "model-x", 100, 100)
        result = checker.check(state, "model-x", None)
        assert result.allowed is True

        # At 150 - still under custom cap of 200
        state = make_state_with_usage("cred-1", "model-x", 150, 100)
        result = checker.check(state, "model-x", None)
        assert result.allowed is True

        # At 200 - hits custom cap
        state = make_state_with_usage("cred-1", "model-x", 200, 100)
        result = checker.check(state, "model-x", None)
        assert result.allowed is False
        assert result.result == LimitResult.BLOCKED_CUSTOM_CAP

    def test_percentage_cap_over_100(self):
        """Percentage cap of 150% allows 1.5x the API limit."""
        cap = CustomCapConfig(
            tier_key="1",
            model_or_group="model-x",
            max_requests=150,  # 150%
            max_requests_mode=CapMode.PERCENTAGE,
            cooldown_mode=CooldownMode.QUOTA_RESET,
            cooldown_value=0,
        )
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker([cap], windows)

        # API limit is 100, 150% = 150 requests allowed
        state = make_state_with_usage("cred-1", "model-x", 100, 100)
        result = checker.check(state, "model-x", None)
        assert result.allowed is True

        state = make_state_with_usage("cred-1", "model-x", 149, 100)
        result = checker.check(state, "model-x", None)
        assert result.allowed is True

        state = make_state_with_usage("cred-1", "model-x", 150, 100)
        result = checker.check(state, "model-x", None)
        assert result.allowed is False


# =============================================================================
# COOLDOWN MODE TESTS
# =============================================================================


class TestCooldownModes:
    """Tests for different cooldown modes."""

    def test_fixed_cooldown(self):
        """FIXED mode: cooldown is fixed duration from now."""
        now = time.time()
        cap = CustomCapConfig(
            tier_key="1",
            model_or_group="model-x",
            max_requests=10,
            cooldown_mode=CooldownMode.FIXED,
            cooldown_value=300,  # 5 minutes
        )
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker([cap], windows)

        # Reset in 1 hour, but cooldown should be ~5 minutes from now
        reset_at = now + 3600
        state = make_state_with_usage("cred-1", "model-x", 10, 100, reset_at)

        result = checker.check(state, "model-x", None)
        assert result.allowed is False
        assert result.blocked_until is not None
        # Should be approximately now + 300 (within a second tolerance)
        assert abs(result.blocked_until - (now + 300)) < 2

    def test_quota_reset_cooldown(self):
        """QUOTA_RESET mode: cooldown until natural window reset."""
        now = time.time()
        reset_at = now + 3600  # 1 hour from now

        cap = CustomCapConfig(
            tier_key="1",
            model_or_group="model-x",
            max_requests=10,
            cooldown_mode=CooldownMode.QUOTA_RESET,
            cooldown_value=0,
        )
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker([cap], windows)

        state = make_state_with_usage("cred-1", "model-x", 10, 100, reset_at)

        result = checker.check(state, "model-x", None)
        assert result.allowed is False
        assert result.blocked_until == reset_at

    def test_offset_positive_cooldown(self):
        """OFFSET mode with positive value: wait AFTER natural reset."""
        now = time.time()
        reset_at = now + 3600  # 1 hour from now

        cap = CustomCapConfig(
            tier_key="1",
            model_or_group="model-x",
            max_requests=10,
            cooldown_mode=CooldownMode.OFFSET,
            cooldown_value=600,  # +10 minutes after reset
        )
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker([cap], windows)

        state = make_state_with_usage("cred-1", "model-x", 10, 100, reset_at)

        result = checker.check(state, "model-x", None)
        assert result.allowed is False
        # Should be reset_at + 600 = 1h 10m from now
        assert result.blocked_until == reset_at + 600

    def test_offset_negative_clamped_to_reset(self):
        """OFFSET mode with negative value: clamped to >= natural reset."""
        now = time.time()
        reset_at = now + 3600  # 1 hour from now

        cap = CustomCapConfig(
            tier_key="1",
            model_or_group="model-x",
            max_requests=10,
            cooldown_mode=CooldownMode.OFFSET,
            cooldown_value=-300,  # -5 minutes (before reset)
        )
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker([cap], windows)

        state = make_state_with_usage("cred-1", "model-x", 10, 100, reset_at)

        result = checker.check(state, "model-x", None)
        assert result.allowed is False
        # Should be clamped to natural reset (can't end before quota resets)
        assert result.blocked_until == reset_at


# =============================================================================
# TIER/PRIORITY MATCHING TESTS
# =============================================================================


class TestTierPriorityMatching:
    """Tests for cap matching based on tier/priority."""

    def test_exact_tier_match(self):
        """Cap with exact tier match is used."""
        caps = [
            CustomCapConfig(
                tier_key="1",
                model_or_group="model-x",
                max_requests=50,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            ),
            CustomCapConfig(
                tier_key="2",
                model_or_group="model-x",
                max_requests=100,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            ),
        ]
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker(caps, windows)

        # Priority 1 -> uses cap with limit 50
        state1 = make_state_with_usage("cred-1", "model-x", 50, 1000, priority=1)
        result1 = checker.check(state1, "model-x", None)
        assert result1.allowed is False  # At limit

        # Priority 2 -> uses cap with limit 100
        state2 = make_state_with_usage("cred-2", "model-x", 50, 1000, priority=2)
        result2 = checker.check(state2, "model-x", None)
        assert result2.allowed is True  # Under limit

    def test_default_tier_fallback(self):
        """Default tier is used when no exact match."""
        caps = [
            CustomCapConfig(
                tier_key="default",
                model_or_group="model-x",
                max_requests=25,
                cooldown_mode=CooldownMode.QUOTA_RESET,
                cooldown_value=0,
            ),
        ]
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker(caps, windows)

        # Priority 5 (no exact match) -> uses default
        state = make_state_with_usage("cred-1", "model-x", 25, 1000, priority=5)
        result = checker.check(state, "model-x", None)
        assert result.allowed is False

    def test_no_cap_allows_unlimited(self):
        """No applicable cap means no restriction."""
        caps = [
            CustomCapConfig(
                tier_key="1",
                model_or_group="model-y",  # Different model
                max_requests=10,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            ),
        ]
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker(caps, windows)

        # model-x has no cap defined
        state = make_state_with_usage("cred-1", "model-x", 9999, 1000, priority=1)
        result = checker.check(state, "model-x", None)
        assert result.allowed is True


# =============================================================================
# QUOTA GROUP TESTS
# =============================================================================


class TestQuotaGroupCaps:
    """Tests for caps applied to quota groups."""

    def test_cap_on_quota_group(self):
        """Cap applied to quota group affects all models in group."""
        caps = [
            CustomCapConfig(
                tier_key="1",
                model_or_group="flash-group",
                max_requests=100,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            ),
        ]
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker(caps, windows)

        # Create state with group usage
        state = CredentialState(stable_id="cred-1", provider="test", accessor="key")
        state.priority = 1
        group_stats = state.get_group_stats("flash-group", create=True)
        group_stats.windows["daily"] = WindowStats(
            name="daily",
            request_count=100,
            started_at=time.time() - 1000,
            reset_at=time.time() + 3600,
            limit=1000,
        )

        result = checker.check(state, "gemini-2.0-flash", quota_group="flash-group")
        assert result.allowed is False
        assert result.result == LimitResult.BLOCKED_CUSTOM_CAP


# =============================================================================
# LAYERED CAPS TESTS (MODEL + GROUP INDEPENDENT)
# =============================================================================


class TestLayeredCaps:
    """Tests for layered caps - model and group caps checked independently."""

    def test_model_cap_blocks_model_not_group(self):
        """Model cap exceeded blocks only that model, group still available."""
        caps = [
            CustomCapConfig(
                tier_key="1",
                model_or_group="gemini-2.0-flash",  # Model cap
                max_requests=50,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            ),
            CustomCapConfig(
                tier_key="1",
                model_or_group="flash-group",  # Group cap
                max_requests=200,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            ),
        ]
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker(caps, windows)

        # Create state with model usage at 50 (model cap hit)
        state = CredentialState(stable_id="cred-1", provider="test", accessor="key")
        state.priority = 1

        # Model usage: 50 (at model cap)
        model_stats = state.get_model_stats("gemini-2.0-flash", create=True)
        model_stats.windows["daily"] = WindowStats(
            name="daily",
            request_count=50,
            started_at=time.time() - 1000,
            reset_at=time.time() + 3600,
            limit=1000,
        )

        # Group usage: 70 (under group cap of 200)
        group_stats = state.get_group_stats("flash-group", create=True)
        group_stats.windows["daily"] = WindowStats(
            name="daily",
            request_count=70,
            started_at=time.time() - 1000,
            reset_at=time.time() + 3600,
            limit=1000,
        )

        # Request for gemini-2.0-flash with group -> should be blocked by MODEL cap
        result = checker.check(state, "gemini-2.0-flash", quota_group="flash-group")
        assert result.allowed is False
        assert "model" in result.reason.lower()
        assert "gemini-2.0-flash" in result.reason

    def test_group_cap_blocks_when_model_under(self):
        """Group cap exceeded blocks even when model is under its cap."""
        caps = [
            CustomCapConfig(
                tier_key="1",
                model_or_group="gemini-2.0-flash",  # Model cap
                max_requests=100,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            ),
            CustomCapConfig(
                tier_key="1",
                model_or_group="flash-group",  # Group cap
                max_requests=50,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            ),
        ]
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker(caps, windows)

        # Create state
        state = CredentialState(stable_id="cred-1", provider="test", accessor="key")
        state.priority = 1

        # Model usage: 30 (under model cap of 100)
        model_stats = state.get_model_stats("gemini-2.0-flash", create=True)
        model_stats.windows["daily"] = WindowStats(
            name="daily",
            request_count=30,
            started_at=time.time() - 1000,
            reset_at=time.time() + 3600,
            limit=1000,
        )

        # Group usage: 50 (at group cap)
        group_stats = state.get_group_stats("flash-group", create=True)
        group_stats.windows["daily"] = WindowStats(
            name="daily",
            request_count=50,
            started_at=time.time() - 1000,
            reset_at=time.time() + 3600,
            limit=1000,
        )

        # Request -> model is checked first (OK), then group (BLOCKED)
        result = checker.check(state, "gemini-2.0-flash", quota_group="flash-group")
        assert result.allowed is False
        assert "group" in result.reason.lower()
        assert "flash-group" in result.reason

    def test_both_caps_under_limit_allows(self):
        """Both model and group caps under limit -> allowed."""
        caps = [
            CustomCapConfig(
                tier_key="1",
                model_or_group="gemini-2.0-flash",
                max_requests=100,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            ),
            CustomCapConfig(
                tier_key="1",
                model_or_group="flash-group",
                max_requests=200,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            ),
        ]
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker(caps, windows)

        state = CredentialState(stable_id="cred-1", provider="test", accessor="key")
        state.priority = 1

        # Both under limits
        model_stats = state.get_model_stats("gemini-2.0-flash", create=True)
        model_stats.windows["daily"] = WindowStats(
            name="daily",
            request_count=50,
            started_at=time.time() - 1000,
            reset_at=time.time() + 3600,
            limit=1000,
        )

        group_stats = state.get_group_stats("flash-group", create=True)
        group_stats.windows["daily"] = WindowStats(
            name="daily",
            request_count=100,
            started_at=time.time() - 1000,
            reset_at=time.time() + 3600,
            limit=1000,
        )

        result = checker.check(state, "gemini-2.0-flash", quota_group="flash-group")
        assert result.allowed is True

    def test_only_model_cap_defined(self):
        """Only model cap defined, no group cap -> only model checked."""
        caps = [
            CustomCapConfig(
                tier_key="1",
                model_or_group="gemini-2.0-flash",
                max_requests=50,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            ),
        ]
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker(caps, windows)

        state = CredentialState(stable_id="cred-1", provider="test", accessor="key")
        state.priority = 1

        model_stats = state.get_model_stats("gemini-2.0-flash", create=True)
        model_stats.windows["daily"] = WindowStats(
            name="daily",
            request_count=50,
            started_at=time.time() - 1000,
            reset_at=time.time() + 3600,
            limit=1000,
        )

        # Group has high usage but no cap defined for it
        group_stats = state.get_group_stats("flash-group", create=True)
        group_stats.windows["daily"] = WindowStats(
            name="daily",
            request_count=9999,
            started_at=time.time() - 1000,
            reset_at=time.time() + 3600,
            limit=1000,
        )

        result = checker.check(state, "gemini-2.0-flash", quota_group="flash-group")
        assert result.allowed is False
        assert "model" in result.reason.lower()

    def test_only_group_cap_defined(self):
        """Only group cap defined, no model cap -> only group checked."""
        caps = [
            CustomCapConfig(
                tier_key="1",
                model_or_group="flash-group",
                max_requests=100,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            ),
        ]
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker(caps, windows)

        state = CredentialState(stable_id="cred-1", provider="test", accessor="key")
        state.priority = 1

        # Model has high usage but no cap defined for it
        model_stats = state.get_model_stats("gemini-2.0-flash", create=True)
        model_stats.windows["daily"] = WindowStats(
            name="daily",
            request_count=9999,
            started_at=time.time() - 1000,
            reset_at=time.time() + 3600,
            limit=1000,
        )

        group_stats = state.get_group_stats("flash-group", create=True)
        group_stats.windows["daily"] = WindowStats(
            name="daily",
            request_count=100,
            started_at=time.time() - 1000,
            reset_at=time.time() + 3600,
            limit=1000,
        )

        result = checker.check(state, "gemini-2.0-flash", quota_group="flash-group")
        assert result.allowed is False
        assert "group" in result.reason.lower()

    def test_different_models_same_group_cap(self):
        """Different models in same group share group cap."""
        caps = [
            CustomCapConfig(
                tier_key="1",
                model_or_group="flash-group",
                max_requests=100,
                cooldown_mode=CooldownMode.FIXED,
                cooldown_value=60,
            ),
        ]
        windows = WindowManager(get_default_windows())
        checker = CustomCapChecker(caps, windows)

        state = CredentialState(stable_id="cred-1", provider="test", accessor="key")
        state.priority = 1

        # Group at cap
        group_stats = state.get_group_stats("flash-group", create=True)
        group_stats.windows["daily"] = WindowStats(
            name="daily",
            request_count=100,
            started_at=time.time() - 1000,
            reset_at=time.time() + 3600,
            limit=1000,
        )

        # Both models in same group should be blocked
        result1 = checker.check(state, "gemini-2.0-flash", quota_group="flash-group")
        result2 = checker.check(
            state, "gemini-2.0-flash-lite", quota_group="flash-group"
        )

        assert result1.allowed is False
        assert result2.allowed is False
        assert "flash-group" in result1.reason
        assert "flash-group" in result2.reason
