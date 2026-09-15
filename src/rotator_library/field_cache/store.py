# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Async stores for field-cache values."""

from __future__ import annotations

import asyncio
import json
import time
from copy import deepcopy
from typing import Any, Callable, Protocol

from ..protocols import serialize_value

_TTL_ENVELOPE_MARKER = "__llm_proxy_field_cache_ttl_v1__"
_APPEND_LOCKS: dict[int, asyncio.Lock] = {}


class FieldCacheStore(Protocol):
    """Minimal async store interface used by `FieldCacheEngine`."""

    async def get(self, key: str) -> Any: ...

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None: ...

    async def append(
        self,
        key: str,
        values: list[Any],
        *,
        ttl_seconds: int | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
    ) -> list[Any]: ...

    async def clear(self) -> None: ...


class InMemoryFieldCacheStore:
    """Simple process-local store with optional per-key TTL.

    This is the default native runtime store. It intentionally persists only for
    the Python process and avoids a database while still preserving protocol
    state across requests handled by the same executor instance.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] | None = None,
        max_entries: int = 10_000,
    ) -> None:
        self._values: dict[str, Any] = {}
        self._expires_at: dict[str, float] = {}
        self._last_access: dict[str, float] = {}
        self._clock = clock or time.monotonic
        self._max_entries = max(1, int(max_entries))

    async def get(self, key: str) -> Any:
        if self._is_expired(key):
            self._remove(key)
            return None
        if key in self._values:
            self._last_access[key] = self._clock()
        return deepcopy(self._values.get(key))

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        self._prune()
        self._values[key] = deepcopy(value)
        self._last_access[key] = self._clock()
        self._set_expiry(key, ttl_seconds)
        self._trim()

    async def append(
        self,
        key: str,
        values: list[Any],
        *,
        ttl_seconds: int | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
    ) -> list[Any]:
        current = await self.get(key)
        if not isinstance(current, list):
            current = []
        current = _bounded_append_values(
            deepcopy(current),
            deepcopy(values),
            max_values=max_values,
            max_bytes=max_bytes,
        )
        self._values[key] = current
        self._last_access[key] = self._clock()
        self._set_expiry(key, ttl_seconds)
        self._trim()
        return deepcopy(current)

    async def clear(self) -> None:
        self._values.clear()
        self._expires_at.clear()
        self._last_access.clear()

    def _set_expiry(self, key: str, ttl_seconds: int | None) -> None:
        if ttl_seconds is None or ttl_seconds <= 0:
            self._expires_at.pop(key, None)
            return
        self._expires_at[key] = self._clock() + ttl_seconds

    def _is_expired(self, key: str) -> bool:
        expires_at = self._expires_at.get(key)
        return expires_at is not None and expires_at <= self._clock()

    def _prune(self) -> None:
        for key in tuple(self._expires_at):
            if self._is_expired(key):
                self._remove(key)

    def _trim(self) -> None:
        overflow = len(self._values) - self._max_entries
        if overflow <= 0:
            return
        oldest = sorted(self._values, key=lambda key: (self._last_access.get(key, 0.0), key))
        for key in oldest[:overflow]:
            self._remove(key)

    def _remove(self, key: str) -> None:
        self._values.pop(key, None)
        self._expires_at.pop(key, None)
        self._last_access.pop(key, None)


class ProviderCacheFieldStore:
    """Field-cache store backed by an injected `ProviderCache` instance.

    The wrapper does not create `ProviderCache` itself because that class starts
    background async tasks during initialization. Providers or later config code
    should own that lifecycle and pass an initialized cache here.
    """

    def __init__(self, provider_cache: Any) -> None:
        self._cache = provider_cache
        self._append_lock = _shared_append_lock(provider_cache)

    async def get(self, key: str) -> Any:
        raw = await self._cache.retrieve_async(key)
        return _decode_provider_cache_value(raw)

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        await self._cache.store_async(
            key,
            _encode_provider_cache_value(value, ttl_seconds=ttl_seconds),
        )

    async def append(
        self,
        key: str,
        values: list[Any],
        *,
        ttl_seconds: int | None = None,
        max_values: int | None = None,
        max_bytes: int | None = None,
    ) -> list[Any]:
        if hasattr(self._cache, "update_async"):
            result: list[Any] = []

            def update(raw: str | None) -> str:
                current = _decode_provider_cache_value(raw)
                if not isinstance(current, list):
                    current = []
                bounded = _bounded_append_values(
                    current,
                    serialize_value(values),
                    max_values=max_values,
                    max_bytes=max_bytes,
                )
                result[:] = bounded
                return _encode_provider_cache_value(
                    bounded,
                    ttl_seconds=ttl_seconds,
                )

            await self._cache.update_async(key, update)
            return result
        async with self._append_lock:
            current = await self.get(key)
            if not isinstance(current, list):
                current = []
            current = _bounded_append_values(
                current,
                serialize_value(values),
                max_values=max_values,
                max_bytes=max_bytes,
            )
            await self.set(key, current, ttl_seconds=ttl_seconds)
            return current

    async def clear(self) -> None:
        await self._cache.clear()


def _shared_append_lock(provider_cache: Any) -> asyncio.Lock:
    lock = getattr(provider_cache, "_field_cache_append_lock", None)
    if isinstance(lock, asyncio.Lock):
        return lock
    lock = asyncio.Lock()
    try:
        setattr(provider_cache, "_field_cache_append_lock", lock)
    except (AttributeError, TypeError):
        lock = _APPEND_LOCKS.setdefault(id(provider_cache), lock)
    return lock


def _decode_provider_cache_value(raw: str | None) -> Any:
    if raw is None:
        return None
    value = json.loads(raw)
    if isinstance(value, dict) and value.get(_TTL_ENVELOPE_MARKER) is True:
        expires_at = value.get("expires_at")
        if isinstance(expires_at, (int, float)) and expires_at <= time.time():
            return None
        return value.get("value")
    return value


def _encode_provider_cache_value(value: Any, *, ttl_seconds: int | None) -> str:
    payload = serialize_value(value)
    if ttl_seconds is not None and ttl_seconds > 0:
        payload = {
            _TTL_ENVELOPE_MARKER: True,
            "expires_at": time.time() + ttl_seconds,
            "value": payload,
        }
    return json.dumps(payload, ensure_ascii=False)


def _bounded_append_values(
    current: list[Any],
    values: list[Any],
    *,
    max_values: int | None,
    max_bytes: int | None,
) -> list[Any]:
    """Append newest state while enforcing provider-rule memory bounds."""

    combined = current + values
    if max_values is not None and len(combined) > max_values:
        combined = combined[-max_values:]
    if max_bytes is not None:
        while combined and len(json.dumps(serialize_value(combined), ensure_ascii=False).encode("utf-8")) > max_bytes:
            if len(combined) == 1:
                raise ValueError("Field-cache value exceeds max_bytes")
            combined.pop(0)
    return combined


def _bounded_set_value(
    value: Any,
    *,
    max_values: int | None,
    max_bytes: int | None,
    trim_collections: bool,
) -> Any:
    """Bound scalar or correlated-map state before replacing a cache value."""

    bounded = deepcopy(value)
    if trim_collections and max_values is not None:
        if isinstance(bounded, dict) and len(bounded) > max_values:
            bounded = dict(list(bounded.items())[-max_values:])
        elif isinstance(bounded, list) and len(bounded) > max_values:
            bounded = bounded[-max_values:]
    if max_bytes is None:
        return bounded
    while len(json.dumps(serialize_value(bounded), ensure_ascii=False).encode("utf-8")) > max_bytes:
        if not trim_collections:
            raise ValueError("Field-cache value exceeds max_bytes")
        if isinstance(bounded, dict) and len(bounded) > 1:
            bounded.pop(next(iter(bounded)))
            continue
        if isinstance(bounded, list) and len(bounded) > 1:
            bounded.pop(0)
            continue
        raise ValueError("Field-cache value exceeds max_bytes")
    return bounded
