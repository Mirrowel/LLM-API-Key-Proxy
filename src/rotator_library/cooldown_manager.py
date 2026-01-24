# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import asyncio
import logging
import time
from typing import Dict, Tuple

lib_logger = logging.getLogger("rotator_library")


class CooldownManager:
    """
    Manages global cooldown periods for API providers to handle IP-based rate limiting.
    This ensures that once a 429 error is received for a provider, all subsequent
    requests to that provider are paused for a specified duration.

    Also manages provider/model-level cooldowns for 503 errors (capacity exhaustion),
    where rotating credentials is pointless because all credentials are equally affected.
    """

    def __init__(self):
        self._cooldowns: Dict[str, float] = {}  # provider -> end_time
        self._model_cooldowns: Dict[
            Tuple[str, str], float
        ] = {}  # (provider, model) -> end_time
        self._lock = asyncio.Lock()

    async def is_cooling_down(self, provider: str) -> bool:
        """Checks if a provider is currently in a cooldown period."""
        async with self._lock:
            return (
                provider in self._cooldowns and time.time() < self._cooldowns[provider]
            )

    async def start_cooldown(self, provider: str, duration: int):
        """
        Initiates or extends a cooldown period for a provider.
        The cooldown is set to the current time plus the specified duration.
        """
        async with self._lock:
            self._cooldowns[provider] = time.time() + duration

    async def get_cooldown_remaining(self, provider: str) -> float:
        """
        Returns the remaining cooldown time in seconds for a provider.
        Returns 0 if the provider is not in a cooldown period.
        """
        async with self._lock:
            if provider in self._cooldowns:
                remaining = self._cooldowns[provider] - time.time()
                return max(0, remaining)
            return 0

    async def get_remaining_cooldown(self, provider: str) -> float:
        """Backward-compatible alias for get_cooldown_remaining."""
        return await self.get_cooldown_remaining(provider)

    # =========================================================================
    # MODEL-LEVEL COOLDOWNS (for 503 errors)
    # =========================================================================

    async def start_model_cooldown(
        self, provider: str, model: str, duration: float
    ) -> None:
        """
        Apply cooldown to a specific provider/model combination.

        Used for 503 errors (capacity exhaustion) where rotating credentials
        is pointless - all credentials are equally affected.

        Args:
            provider: Provider name (e.g., "antigravity")
            model: Model name (e.g., "claude-opus-4.5")
            duration: Cooldown duration in seconds
        """
        async with self._lock:
            key = (provider, model)
            self._model_cooldowns[key] = time.time() + duration
            lib_logger.info(
                f"Provider/model 503 cooldown: {provider}/{model} for {duration:.1f}s"
            )

    async def get_model_cooldown_remaining(self, provider: str, model: str) -> float:
        """
        Get remaining cooldown time for a provider/model combination.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            Remaining cooldown time in seconds, or 0 if not cooling down
        """
        async with self._lock:
            key = (provider, model)
            if key in self._model_cooldowns:
                remaining = self._model_cooldowns[key] - time.time()
                return max(0, remaining)
            return 0

    async def is_model_cooling_down(self, provider: str, model: str) -> bool:
        """
        Check if a provider/model combination is currently on cooldown.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            True if on cooldown, False otherwise
        """
        return await self.get_model_cooldown_remaining(provider, model) > 0
