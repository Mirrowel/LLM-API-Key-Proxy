# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import asyncio
import time
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CooldownSnapshot:
    """Read-only view of one active cooldown scope for tests/observability."""

    provider: str
    scope: str
    model: str | None
    remaining: float
    reason: str | None = None


class CooldownManager:
    """
    Manages global cooldown periods for API providers to handle IP-based rate limiting.
    This ensures that once a 429 error is received for a provider, all subsequent
    requests to that provider are paused for a specified duration.
    """

    def __init__(self):
        self._cooldowns: Dict[str, float] = {}
        self._metadata: Dict[str, dict[str, str | None]] = {}
        self._lock = asyncio.Lock()

    async def is_cooling_down(self, provider: str) -> bool:
        """Checks if a provider is currently in a cooldown period."""
        return await self.is_scoped_cooling_down(provider, scope="provider")

    async def start_cooldown(self, provider: str, duration: int):
        """
        Initiates or extends a cooldown period for a provider.
        The cooldown is set to the current time plus the specified duration.

        A shorter new cooldown must not shorten an existing longer cooldown;
        provider-wide throttles often arrive concurrently from several requests.
        """
        await self.start_scoped_cooldown(provider, duration, scope="provider")

    async def get_cooldown_remaining(self, provider: str) -> float:
        """
        Returns the remaining cooldown time in seconds for a provider.
        Returns 0 if the provider is not in a cooldown period.
        """
        return await self.get_scoped_remaining(provider, scope="provider")

    async def get_remaining_cooldown(self, provider: str) -> float:
        """Backward-compatible alias for get_cooldown_remaining."""
        return await self.get_cooldown_remaining(provider)

    async def start_scoped_cooldown(
        self,
        provider: str,
        duration: int,
        *,
        model: str | None = None,
        scope: str = "provider",
        reason: str | None = None,
    ) -> None:
        """Start or extend a provider/model cooldown scope.

        Model scopes are intentionally separate from provider scopes because
        capacity failures often belong to one model deployment rather than the
        entire provider. Provider scopes remain available for provider-wide
        throttles and block every model for the provider.
        """

        key = _cooldown_key(provider, scope=scope, model=model)
        async with self._lock:
            new_expiry = time.time() + max(0, duration)
            current_expiry = self._cooldowns.get(key, 0)
            if new_expiry > current_expiry:
                self._cooldowns[key] = new_expiry
                self._metadata[key] = {"provider": provider, "scope": scope, "model": model, "reason": reason}

    async def get_scoped_remaining(self, provider: str, *, model: str | None = None, scope: str = "provider") -> float:
        """Return remaining seconds for one cooldown scope."""

        key = _cooldown_key(provider, scope=scope, model=model)
        async with self._lock:
            return self._remaining_for_key(key, time.time())

    async def get_max_remaining(self, provider: str, *, model: str | None = None) -> float:
        """Return the max provider/model cooldown remaining for a request."""

        async with self._lock:
            now = time.time()
            provider_remaining = self._remaining_for_key(_cooldown_key(provider, scope="provider"), now)
            model_remaining = self._remaining_for_key(_cooldown_key(provider, scope="model", model=model), now) if model else 0
            return max(provider_remaining, model_remaining)

    async def is_scoped_cooling_down(self, provider: str, *, model: str | None = None, scope: str = "provider") -> bool:
        """Return whether one scoped cooldown is currently active."""

        return (await self.get_scoped_remaining(provider, model=model, scope=scope)) > 0

    async def snapshot(self) -> tuple[CooldownSnapshot, ...]:
        """Return active cooldown scopes for tests and future observability."""

        async with self._lock:
            now = time.time()
            snapshots = []
            for key, expires_at in list(self._cooldowns.items()):
                remaining = max(0, expires_at - now)
                if remaining <= 0:
                    continue
                metadata = self._metadata.get(key, {})
                snapshots.append(
                    CooldownSnapshot(
                        provider=str(metadata.get("provider") or key),
                        scope=str(metadata.get("scope") or "provider"),
                        model=metadata.get("model"),
                        remaining=remaining,
                        reason=metadata.get("reason"),
                    )
                )
            return tuple(snapshots)

    def _remaining_for_key(self, key: str, now: float) -> float:
        expires_at = self._cooldowns.get(key)
        if expires_at is None:
            return 0
        remaining = expires_at - now
        if remaining <= 0:
            self._cooldowns.pop(key, None)
            self._metadata.pop(key, None)
            return 0
        return remaining


def _cooldown_key(provider: str, *, scope: str, model: str | None = None) -> str:
    if scope == "model" and model:
        return f"provider:{provider}:model:{model}"
    return f"provider:{provider}"
