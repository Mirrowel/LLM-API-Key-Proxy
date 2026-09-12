# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Storage backends for Responses API objects."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Protocol

from .types import StoredResponse


class ResponsesStore(Protocol):
    """Minimal async store for response retrieval and continuation."""

    async def save(self, response: StoredResponse) -> None: ...

    async def get(
        self,
        response_id: str,
        scope_key: str = "public",
    ) -> Optional[StoredResponse]: ...

    async def delete(self, response_id: str, scope_key: str = "public") -> bool: ...

    async def list_input_items(
        self,
        response_id: str,
        scope_key: str = "public",
    ) -> Optional[list[Any]]: ...

    async def close(self) -> None: ...


class InMemoryResponsesStore:
    """Process-local Responses store.

    This is the Phase 4 default because it has no async lifecycle and avoids a
    new persistence dependency. A provider-cache-backed store can be injected by
    later configuration code when disk persistence is desired.
    """

    def __init__(self, *, max_items: int | None = None) -> None:
        self._responses: dict[tuple[str, str], StoredResponse] = {}
        self.max_items = max_items if max_items and max_items > 0 else None

    async def save(self, response: StoredResponse) -> None:
        self._prune_expired()
        key = (response.scope_key or "public", response.id)
        self._responses[key] = StoredResponse.from_dict(response.to_dict())
        self._prune_overflow()

    async def get(
        self,
        response_id: str,
        scope_key: str = "public",
    ) -> Optional[StoredResponse]:
        key = (scope_key, response_id)
        response = self._responses.get(key)
        if response is None:
            return None
        try:
            if response.is_expired():
                self._responses.pop(key, None)
                return None
            return StoredResponse.from_dict(response.to_dict())
        except (ValueError, TypeError, KeyError):
            # A corrupt row is a miss, never an error on every read.
            self._responses.pop(key, None)
            return None

    async def delete(self, response_id: str, scope_key: str = "public") -> bool:
        return self._responses.pop((scope_key, response_id), None) is not None

    async def list_input_items(
        self,
        response_id: str,
        scope_key: str = "public",
    ) -> Optional[list[Any]]:
        response = await self.get(response_id, scope_key)
        if response is None:
            return None
        return deepcopy(response.input_items)

    async def close(self) -> None:
        """Memory storage owns no external resources."""

    def _prune_expired(self) -> None:
        for key, response in list(self._responses.items()):
            try:
                expired = response.is_expired()
            except Exception:
                # Poison row: same containment as reads — drop it, never
                # fail the save that triggered pruning.
                expired = True
            if expired:
                self._responses.pop(key, None)

    def _prune_overflow(self) -> None:
        if not self.max_items:
            return
        while len(self._responses) > self.max_items:
            oldest_key = min(
                self._responses,
                key=lambda key: self._responses[key].created_at,
            )
            self._responses.pop(oldest_key, None)


class ProviderCacheResponsesStore:
    """Responses store backed by an injected `ProviderCache` instance.

    The wrapper is given a `ProviderCache` instance or a factory. The factory
    form defers construction to the first async use because `ProviderCache`
    schedules background tasks and therefore requires a running event loop;
    the service can be built from a sync dependency. Rows obey their own
    ``expires_at`` and the wrapper bounds the process-local index to
    ``max_items`` (oldest ``created_at`` evicted first), mirroring the
    in-memory backend so the durable default has the same policy surface.
    """

    def __init__(
        self,
        provider_cache: Any = None,
        *,
        prefix: str = "responses",
        max_items: int | None = None,
        cache_factory: Any = None,
    ) -> None:
        self._cache = provider_cache
        self._cache_factory = cache_factory
        self._prefix = prefix
        self.max_items = max_items if max_items and max_items > 0 else None
        self._index: dict[tuple[str, str], StoredResponse] = {}

    def _backend(self) -> Any:
        """Return the cache, constructing it on first use when lazy."""

        if self._cache is None:
            if self._cache_factory is None:
                raise RuntimeError("ProviderCacheResponsesStore has no cache backend")
            self._cache = self._cache_factory()
        return self._cache

    async def save(self, response: StoredResponse) -> None:
        cache = self._backend()
        self._index[(response.scope_key or "public", response.id)] = StoredResponse.from_dict(response.to_dict())
        await cache.store_async(
            self._key(response.id, response.scope_key or "public"),
            json.dumps(response.to_dict(), ensure_ascii=False),
        )
        flush = getattr(cache, "_save_to_disk", None)
        if callable(flush):
            await flush()
        await self._prune_expired()
        await self._prune_overflow()

    async def get(
        self,
        response_id: str,
        scope_key: str = "public",
    ) -> Optional[StoredResponse]:
        cache = self._backend()
        raw = await cache.retrieve_async(self._key(response_id, scope_key))
        legacy_key = None
        if raw is None:
            legacy_key = self._legacy_key(response_id)
            raw = await cache.retrieve_async(legacy_key)
        if raw is None:
            return None
        try:
            response = StoredResponse.from_dict(json.loads(raw))
            response_scope = response.scope_key or "public"
            if response.id != response_id or response_scope != scope_key:
                return None
            # Expiry runs inside the corrupt-row guard: a poisoned
            # ``expires_at`` is a miss, never a TypeError on every read.
            expired = response.is_expired()
        except (ValueError, TypeError, KeyError):
            # Corrupt cache rows are cache misses, never 500s.
            return None
        if expired:
            await self.delete(response_id, scope_key)
            return None
        if legacy_key is not None:
            response.scope_key = response_scope
            await self.save(response)
            delete = getattr(cache, "delete_async", None)
            if delete:
                await delete(legacy_key)
        else:
            self._index[(response_scope, response.id)] = StoredResponse.from_dict(response.to_dict())
        return response

    async def delete(self, response_id: str, scope_key: str = "public") -> bool:
        self._index.pop((scope_key, response_id), None)
        delete = getattr(self._backend(), "delete_async", None)
        if delete:
            return bool(await delete(self._key(response_id, scope_key)))
        # Third-party cache adapters may not support key-level deletion. Avoid
        # clearing unrelated provider state when that capability is absent.
        return False

    async def list_input_items(
        self,
        response_id: str,
        scope_key: str = "public",
    ) -> Optional[list[Any]]:
        response = await self.get(response_id, scope_key)
        if response is None:
            return None
        return deepcopy(response.input_items)

    async def close(self) -> None:
        """Stop the injected cache's writer and cleanup tasks."""

        if self._cache is None:
            return
        shutdown = getattr(self._cache, "shutdown", None)
        if shutdown:
            await shutdown()

    async def _prune_expired(self) -> None:
        for key, response in list(self._index.items()):
            if response.is_expired():
                self._index.pop(key, None)
                await self.delete(response.id, response.scope_key or "public")

    async def _prune_overflow(self) -> None:
        if not self.max_items:
            return
        while len(self._index) > self.max_items:
            oldest_key = min(
                self._index,
                key=lambda key: self._index[key].created_at,
            )
            response = self._index.pop(oldest_key, None)
            if response is not None:
                await self.delete(response.id, response.scope_key or "public")

    def _key(self, response_id: str, scope_key: str) -> str:
        digest = hashlib.sha256(
            f"{scope_key}\x00{response_id}".encode("utf-8")
        ).hexdigest()
        return f"{self._prefix}:{digest}"

    def _legacy_key(self, response_id: str) -> str:
        safe_id = response_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        return f"{self._prefix}:{safe_id}"


def create_configured_responses_store(*, config: Any = None, env: Any = None) -> ResponsesStore:
    """Create the configured Responses store backend.

    The provider-cache backend is the durable default; ``memory`` remains an
    opt-in. The durable backend keeps the configured row TTL authoritative and
    uses the ProviderCache disk TTL only as a floor so a longer configured TTL
    cannot be clipped by the cache's own retention.
    """

    from ..config.experimental import get_responses_store_runtime_settings, get_responses_store_settings
    from ..providers.provider_cache import create_provider_cache

    runtime = get_responses_store_runtime_settings(config=config, env=env)
    settings = get_responses_store_settings(config=config, env=env)
    if runtime.backend == "memory":
        return InMemoryResponsesStore(max_items=settings.max_items)
    cache_dir = Path(runtime.cache_dir) if runtime.cache_dir else None
    disk_ttl_seconds = runtime.cache_disk_ttl_seconds
    if settings.ttl_seconds:
        disk_ttl_seconds = max(disk_ttl_seconds, settings.ttl_seconds)
    return ProviderCacheResponsesStore(
        prefix=runtime.cache_prefix,
        max_items=settings.max_items,
        cache_factory=lambda: create_provider_cache(
            runtime.cache_name,
            cache_dir=cache_dir,
            memory_ttl_seconds=runtime.cache_memory_ttl_seconds,
            disk_ttl_seconds=disk_ttl_seconds,
            env_prefix=f"{runtime.cache_name.upper().replace('-', '_')}_CACHE",
        ),
    )
