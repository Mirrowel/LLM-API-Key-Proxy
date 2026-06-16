# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Tests for cooldown_manager.CooldownManager.

CooldownManager tracks per-provider cooldown periods to handle
IP-based rate limiting. When a 429 is received, all requests to
that provider are paused for a configurable duration.
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from cooldown_manager import CooldownManager


class TestCooldownBasics:
    """Basic functionality tests."""

    @pytest.mark.asyncio
    async def test_provider_not_cooling_down_initially(self):
        cm = CooldownManager()
        assert await cm.is_cooling_down("openai") is False

    @pytest.mark.asyncio
    async def test_start_cooldown_makes_provider_cooling(self):
        cm = CooldownManager()
        await cm.start_cooldown("openai", 60)
        assert await cm.is_cooling_down("openai") is True

    @pytest.mark.asyncio
    async def test_different_providers_independent(self):
        cm = CooldownManager()
        await cm.start_cooldown("openai", 60)
        assert await cm.is_cooling_down("openai") is True
        assert await cm.is_cooling_down("anthropic") is False


class TestCooldownRemaining:
    """Tests for remaining time calculations."""

    @pytest.mark.asyncio
    async def test_remaining_cooldown_positive(self):
        cm = CooldownManager()
        await cm.start_cooldown("openai", 100)
        remaining = await cm.get_cooldown_remaining("openai")
        # Should be close to 100 but slightly less due to execution time
        assert 90 <= remaining <= 100

    @pytest.mark.asyncio
    async def test_remaining_cooldown_zero_if_not_cooling(self):
        cm = CooldownManager()
        remaining = await cm.get_cooldown_remaining("openai")
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_remaining_cooldown_after_expiry(self):
        cm = CooldownManager()
        await cm.start_cooldown("openai", 0)
        # With duration=0, the cooldown is set to exactly now
        # By the time we check, time has advanced
        await asyncio.sleep(0.01)
        remaining = await cm.get_cooldown_remaining("openai")
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_get_remaining_cooldown_alias(self):
        """get_remaining_cooldown is a backward-compatible alias."""
        cm = CooldownManager()
        await cm.start_cooldown("openai", 50)
        r1 = await cm.get_cooldown_remaining("openai")
        r2 = await cm.get_remaining_cooldown("openai")
        assert abs(r1 - r2) < 1  # Should be nearly identical


class TestCooldownExpiry:
    """Tests for cooldown expiration behavior."""

    @pytest.mark.asyncio
    async def test_cooldown_expires(self):
        cm = CooldownManager()
        await cm.start_cooldown("openai", 1)
        assert await cm.is_cooling_down("openai") is True
        # Patch the clock to advance past expiry without real sleeping
        with patch("cooldown_manager.time.time", return_value=time.time() + 1.1):
            assert await cm.is_cooling_down("openai") is False

    @pytest.mark.asyncio
    async def test_cooldown_extends_on_new_start(self):
        cm = CooldownManager()
        await cm.start_cooldown("openai", 5)
        r1 = await cm.get_cooldown_remaining("openai")

        await cm.start_cooldown("openai", 100)
        r2 = await cm.get_cooldown_remaining("openai")

        assert r2 > r1


class TestCooldownConcurrency:
    """Tests for concurrent access safety."""

    @pytest.mark.asyncio
    async def test_concurrent_start_and_check(self):
        cm = CooldownManager()
        # Multiple concurrent operations should not cause issues
        await asyncio.gather(
            cm.start_cooldown("openai", 10),
            cm.start_cooldown("anthropic", 20),
            cm.is_cooling_down("openai"),
            cm.is_cooling_down("anthropic"),
            cm.is_cooling_down("gemini"),
        )
        assert await cm.is_cooling_down("openai") is True
        assert await cm.is_cooling_down("anthropic") is True
        assert await cm.is_cooling_down("gemini") is False

    @pytest.mark.asyncio
    async def test_concurrent_reads(self):
        cm = CooldownManager()
        await cm.start_cooldown("openai", 10)
        # Many concurrent reads
        results = await asyncio.gather(*[cm.is_cooling_down("openai") for _ in range(100)])
        assert all(results)
