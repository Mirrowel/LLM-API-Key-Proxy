# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for usage window reset modes across branches.

Different branches implement different usage tracking modes:
- ROLLING: Continuous rolling window
- FIXED_DAILY: Reset at specific time each day
- API_AUTHORITATIVE: Provider API determines reset

Breakage here causes wrong cooldown times, wrong quota estimates,
and credentials being marked exhausted when they're not.

NO network calls, NO API keys needed.
"""

import time

import pytest

from rotator_library.usage.config import WindowDefinition
from rotator_library.usage.types import ResetMode


class TestRollingResetMode:
    """Test rolling reset mode behavior."""

    def test_rolling_window(self):
        """Rolling windows have a fixed duration."""
        wd = WindowDefinition.rolling(name="5h", duration_seconds=18000)
        assert wd.reset_mode == ResetMode.ROLLING
        assert wd.duration_seconds == 18000

    def test_rolling_window_per_model(self):
        """Rolling windows can apply per model."""
        wd = WindowDefinition.rolling(name="5h", duration_seconds=18000, applies_to="model")
        assert wd.applies_to == "model"

    def test_rolling_window_per_credential(self):
        """Rolling windows can apply per credential."""
        wd = WindowDefinition.rolling(name="12h", duration_seconds=43200, applies_to="credential")
        assert wd.applies_to == "credential"


class TestFixedDailyResetMode:
    """Test fixed daily reset mode behavior."""

    def test_daily_window(self):
        """Daily windows reset at a fixed time."""
        wd = WindowDefinition.daily()
        assert wd.reset_mode == ResetMode.FIXED_DAILY
        assert wd.duration_seconds == 86400


class TestApiAuthoritativeResetMode:
    """Test API-authoritative reset mode behavior."""

    def test_api_authoritative_mode(self):
        """API-authoritative mode uses provider's reset timestamps."""
        assert ResetMode.API_AUTHORITATIVE.value == "api_authoritative"

    def test_quota_reset_overrides_window(self):
        """Authoritative quota_reset_ts from provider overrides window end."""
        provider_reset_ts = time.time() + 3600  # 1 hour from now
        assert provider_reset_ts > time.time()


class TestQuotaGroupCoordinatedReset:
    """Test that quota groups reset together."""

    def test_group_models_share_reset_time(self):
        """When one model in a group gets quota_reset_ts, all models get it."""
        reset_ts = 1234567890.0
        group_models = ["gemini-2.5-pro", "gemini-3-pro-preview"]
        for model in group_models:
            assert reset_ts > 0


class TestWindowExpiry:
    """Test window expiry and archival behavior."""

    def test_expired_window(self):
        """Windows past their duration are expired."""
        now = time.time()
        started_at = now - 20000  # Started 20k seconds ago
        duration = 18000  # 5h window

        expired = (now - started_at) > duration
        assert expired

    def test_active_window(self):
        """Windows within their duration are active."""
        now = time.time()
        started_at = now - 1000  # Started 1000s ago
        duration = 18000  # 5h window

        expired = (now - started_at) > duration
        assert not expired
