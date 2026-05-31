# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Async stores for field-cache values."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any, Callable, Protocol

from ..protocols import serialize_value


class FieldCacheStore(Protocol):
    """Minimal async store interface used by `FieldCacheEngine`."""

    async def get(self, key: str) -> Any: ...

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None: ...

    async def append(self, key: str, values: list[Any], *, ttl_seconds: int | None = None) -> list[Any]: ...

    async def clear(self) -> None: ...


class InMemoryFieldCacheStore:
    """Simple process-local store with optional per-key TTL.

    This is the default native runtime store. It intentionally persists only for
    the Python process and avoids a database while still preserving protocol
    state across requests handled by the same executor instance.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._values: dict[str, Any] = {}
        self._expires_at: dict[str, float] = {}
        self._clock = clock or time.monotonic

    async def get(self, key: str) -> Any:
        if self._is_expired(key):
            self._values.pop(key, None)
            self._expires_at.pop(key, None)
            return None
        return deepcopy(self._values.get(key))

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        self._values[key] = deepcopy(value)
        self._set_expiry(key, ttl_seconds)

    async def append(self, key: str, values: list[Any], *, ttl_seconds: int | None = None) -> list[Any]:
        current = await self.get(key)
        if not isinstance(current, list):
            current = []
        current = deepcopy(current) + deepcopy(values)
        self._values[key] = current
        self._set_expiry(key, ttl_seconds)
        return deepcopy(current)

    async def clear(self) -> None:
        self._values.clear()
        self._expires_at.clear()

    def _set_expiry(self, key: str, ttl_seconds: int | None) -> None:
        if ttl_seconds is None or ttl_seconds <= 0:
            self._expires_at.pop(key, None)
            return
        self._expires_at[key] = self._clock() + ttl_seconds

    def _is_expired(self, key: str) -> bool:
        expires_at = self._expires_at.get(key)
        return expires_at is not None and expires_at <= self._clock()


class ProviderCacheFieldStore:
    """Field-cache store backed by an injected `ProviderCache` instance.

    The wrapper does not create `ProviderCache` itself because that class starts
    background async tasks during initialization. Providers or later config code
    should own that lifecycle and pass an initialized cache here.
    """

    def __init__(self, provider_cache: Any) -> None:
        self._cache = provider_cache

    async def get(self, key: str) -> Any:
        raw = await self._cache.retrieve_async(key)
        if raw is None:
            return None
        value = json.loads(raw)
        if isinstance(value, dict) and value.get("__field_cache_wrapped__") is True:
            expires_at = value.get("expires_at")
            if isinstance(expires_at, (int, float)) and expires_at <= time.time():
                return None
            return value.get("value")
        return value

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        payload = serialize_value(value)
        if ttl_seconds is not None and ttl_seconds > 0:
            payload = {"__field_cache_wrapped__": True, "expires_at": time.time() + ttl_seconds, "value": payload}
        await self._cache.store_async(key, json.dumps(payload, ensure_ascii=False))

    async def append(self, key: str, values: list[Any], *, ttl_seconds: int | None = None) -> list[Any]:
        current = await self.get(key)
        if not isinstance(current, list):
            current = []
        current = current + serialize_value(values)
        await self.set(key, current, ttl_seconds=ttl_seconds)
        return current

    async def clear(self) -> None:
        await self._cache.clear()
