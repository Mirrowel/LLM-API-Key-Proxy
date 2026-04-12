# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for usage tracking: windows, quota groups, custom caps, fair cycle.

Usage tracking bugs cause:
- Over-use: burning through paid credentials too fast
- Under-use: leaving free quota on the table
- Wrong cooldowns: starving the pool of available credentials
- Fair cycle bugs: same credential used repeatedly while others sit idle

NO network calls, NO API keys needed.
"""

import time
from dataclasses import dataclass

import pytest

from rotator_library.usage.types import (
    WindowStats,
    RotationMode,
    ResetMode,
    TrackingMode,
    CooldownMode,
)
from rotator_library.usage.config import WindowDefinition, load_provider_usage_config


class TestUsageConfigLoading:
    """Test usage config loading from environment variables."""

    def test_default_config_no_plugins(self):
        """Default config when no plugins are available."""
        config = load_provider_usage_config("nonexistent_provider", {})
        assert config is not None

    def test_rolling_reset_mode(self):
        """ROLLING mode for continuous rolling windows."""
        wd = WindowDefinition.rolling(name="5h", duration_seconds=18000)
        assert wd.reset_mode == ResetMode.ROLLING
        assert wd.duration_seconds == 18000

    def test_daily_reset_mode(self):
        """FIXED_DAILY mode for daily fixed windows."""
        wd = WindowDefinition.daily()
        assert wd.reset_mode == ResetMode.FIXED_DAILY

    def test_api_authoritative_reset_mode(self):
        """API_AUTHORITATIVE mode when provider determines reset."""
        assert ResetMode.API_AUTHORITATIVE.value == "api_authoritative"


class TestWindowStats:
    """Test WindowStats data structures."""

    def test_window_stats_creation(self):
        """WindowStats can be created with basic fields."""
        stats = WindowStats(name="5h")
        assert stats.name == "5h"
        assert stats.request_count == 0

    def test_window_stats_with_quota_reset(self):
        """WindowStats with authoritative reset timestamp."""
        future_time = time.time() + 3600
        stats = WindowStats(name="5h", reset_at=future_time)
        assert stats.reset_at is not None
        assert stats.reset_at > time.time()

    def test_window_stats_remaining(self):
        """remaining property calculates correctly."""
        stats = WindowStats(name="5h", request_count=50, limit=100)
        assert stats.remaining == 50

    def test_window_stats_remaining_unlimited(self):
        """remaining is None when no limit set."""
        stats = WindowStats(name="5h", request_count=50)
        assert stats.remaining is None


class TestQuotaGroups:
    """Test model quota group logic from ProviderInterface."""

    def test_quota_group_resolution(self):
        """Models in the same quota group share cooldown timing."""
        from rotator_library.providers.provider_interface import ProviderInterface

        class TestProvider(ProviderInterface):
            provider_env_name = "test"
            model_quota_groups = {
                "pro": ["gemini-2.5-pro", "gemini-3-pro-preview"],
                "flash": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
            }

            async def get_models(self, api_key, client):
                return []

        provider = TestProvider()
        group = provider.get_model_quota_group("gemini-2.5-pro")
        assert group == "pro"

        group = provider.get_model_quota_group("gemini-2.5-flash")
        assert group == "flash"

    def test_ungrouped_model(self):
        """Models not in any group return None."""
        from rotator_library.providers.provider_interface import ProviderInterface

        class TestProvider(ProviderInterface):
            provider_env_name = "test"
            model_quota_groups = {
                "pro": ["gemini-2.5-pro"],
            }

            async def get_models(self, api_key, client):
                return []

        provider = TestProvider()
        group = provider.get_model_quota_group("some-random-model")
        assert group is None

    def test_provider_prefix_stripped(self):
        """Provider prefix is stripped before group lookup."""
        from rotator_library.providers.provider_interface import ProviderInterface

        class TestProvider(ProviderInterface):
            provider_env_name = "test"
            model_quota_groups = {
                "pro": ["gemini-2.5-pro"],
            }

            async def get_models(self, api_key, client):
                return []

        provider = TestProvider()
        group = provider.get_model_quota_group("test/gemini-2.5-pro")
        assert group == "pro"


class TestCustomCaps:
    """Test custom cap configuration parsing."""

    def test_custom_cap_absolute_value(self):
        """Absolute custom cap values are parsed correctly."""
        from rotator_library.providers.provider_interface import ProviderInterface

        class TestProvider(ProviderInterface):
            provider_env_name = "test"
            default_custom_caps = {
                2: {
                    "claude": {
                        "max_requests": 100,
                        "cooldown_mode": "quota_reset",
                        "cooldown_value": 0,
                    }
                }
            }

            async def get_models(self, api_key, client):
                return []

        provider = TestProvider()
        caps = provider.default_custom_caps
        assert caps[2]["claude"]["max_requests"] == 100

    def test_custom_cap_percentage_value(self):
        """Percentage custom cap values are stored as strings."""
        from rotator_library.providers.provider_interface import ProviderInterface

        class TestProvider(ProviderInterface):
            provider_env_name = "test"
            default_custom_caps = {
                2: {
                    "claude": {
                        "max_requests": "80%",
                        "cooldown_mode": "offset",
                        "cooldown_value": 3600,
                    }
                }
            }

            async def get_models(self, api_key, client):
                return []

        provider = TestProvider()
        cap_value = provider.default_custom_caps[2]["claude"]["max_requests"]
        assert cap_value == "80%"


class TestRotationModes:
    """Test rotation mode types."""

    def test_balanced_mode(self):
        """Balanced mode distributes load evenly."""
        assert RotationMode.BALANCED.value == "balanced"

    def test_sequential_mode(self):
        """Sequential mode uses credentials until exhausted."""
        assert RotationMode.SEQUENTIAL.value == "sequential"

    def test_fair_cycle_tracking_modes(self):
        """Fair cycle tracking modes exist."""
        assert TrackingMode.MODEL_GROUP.value == "model_group"
        assert TrackingMode.CREDENTIAL.value == "credential"

    def test_cooldown_modes(self):
        """Custom cap cooldown modes exist."""
        assert CooldownMode.QUOTA_RESET.value == "quota_reset"
        assert CooldownMode.OFFSET.value == "offset"
        assert CooldownMode.FIXED.value == "fixed"
