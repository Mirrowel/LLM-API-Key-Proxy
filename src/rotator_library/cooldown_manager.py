# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import asyncio
import time
from typing import Dict

class CooldownManager:
    """
    Manages cooldown periods for API credentials to handle rate limiting.
    Cooldowns are applied per-credential, allowing other credentials from the
    same provider to be used while one is cooling down.
    """

    def __init__(self):
        self._cooldowns: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def is_cooling_down(self, credential: str) -> bool:
        """Checks if a credential is currently in a cooldown period."""
        async with self._lock:
            return credential in self._cooldowns and time.time() < self._cooldowns[credential]

    async def start_cooldown(self, credential: str, duration: int):
        """
        Initiates or extends a cooldown period for a credential.
        The cooldown is set to the current time plus the specified duration.
        """
        async with self._lock:
            self._cooldowns[credential] = time.time() + duration

    async def get_cooldown_remaining(self, credential: str) -> float:
        """
        Returns the remaining cooldown time in seconds for a credential.
        Returns 0 if the credential is not in a cooldown period.
        """
        async with self._lock:
            if credential in self._cooldowns:
                remaining = self._cooldowns[credential] - time.time()
                return max(0, remaining)
            return 0