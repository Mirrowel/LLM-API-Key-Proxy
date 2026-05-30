# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Async stores for field-cache values."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Protocol

from ..protocols import serialize_value


class FieldCacheStore(Protocol):
    """Minimal async store interface used by `FieldCacheEngine`."""

    async def get(self, key: str) -> Any: ...

    async def set(self, key: str, value: Any) -> None: ...

    async def append(self, key: str, values: list[Any]) -> list[Any]: ...

    async def clear(self) -> None: ...


class InMemoryFieldCacheStore:
    """Simple process-local store for tests and lightweight runtime use."""

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return deepcopy(self._values.get(key))

    async def set(self, key: str, value: Any) -> None:
        self._values[key] = deepcopy(value)

    async def append(self, key: str, values: list[Any]) -> list[Any]:
        current = self._values.get(key)
        if not isinstance(current, list):
            current = []
        current = deepcopy(current) + deepcopy(values)
        self._values[key] = current
        return deepcopy(current)

    async def clear(self) -> None:
        self._values.clear()


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
        return json.loads(raw)

    async def set(self, key: str, value: Any) -> None:
        await self._cache.store_async(key, json.dumps(serialize_value(value), ensure_ascii=False))

    async def append(self, key: str, values: list[Any]) -> list[Any]:
        current = await self.get(key)
        if not isinstance(current, list):
            current = []
        current = current + serialize_value(values)
        await self.set(key, current)
        return current

    async def clear(self) -> None:
        await self._cache.clear()
