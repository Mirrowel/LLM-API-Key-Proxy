# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Storage backends for Responses API objects."""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
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
    """Responses store backed directly by the ``cache`` storage engine.

    One row per stored response, keyed by the scoped SHA-256 of
    ``(scope_key, response_id)`` and valued as the response JSON. The row TTL
    is ``max(configured TTL, retention floor)`` so a longer configured TTL is
    never clipped. Rows obey their own ``expires_at`` on read and the engine
    bounds retention to ``max_items`` (oldest ``created_at`` evicted first),
    mirroring the in-memory backend's policy surface.
    """

    def __init__(
        self,
        *,
        prefix: str = "responses",
        max_items: int | None = None,
        ttl_seconds: Optional[float] = None,
        engine: Any = None,
    ) -> None:
        self._prefix = prefix
        self.max_items = max_items if max_items and max_items > 0 else None
        self._ttl_seconds = ttl_seconds if ttl_seconds and ttl_seconds > 0 else None
        self._engine = engine

    def _backend(self) -> Any:
        """Return the process-wide cache engine, resolving it lazily."""

        if self._engine is None:
            from ..storage.engine import get_engine

            self._engine = get_engine("cache")
        return self._engine

    async def save(self, response: StoredResponse) -> None:
        payload = json.dumps(response.to_dict(), ensure_ascii=False).encode("utf-8")
        engine = self._backend()
        await engine.aset(
            self._key(response.id, response.scope_key or "public"),
            payload,
            ttl_seconds=self._ttl_seconds,
            created_at=response.created_at,
        )
        await self._prune_expired()
        await self._prune_overflow()

    async def get(
        self,
        response_id: str,
        scope_key: str = "public",
    ) -> Optional[StoredResponse]:
        engine = self._backend()
        raw = await engine.aget(self._key(response_id, scope_key))
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
            # Corrupt engine rows are cache misses, never 500s.
            return None
        if expired:
            await self.delete(response_id, scope_key)
            return None
        return response

    async def delete(self, response_id: str, scope_key: str = "public") -> bool:
        return bool(await self._backend().adelete(self._key(response_id, scope_key)))

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
        """Shared engine lifecycle is owned by the app lifespan."""

    async def _prune_expired(self) -> None:
        # Engine TTL owns expiry; the indexed sweep is the light safety reaper.
        await self._backend().asweep()

    async def _prune_overflow(self) -> None:
        if not self.max_items:
            return
        engine = self._backend()
        prefix = f"{self._prefix}:"
        overflow = await asyncio.to_thread(engine.count_prefix, prefix) - self.max_items
        if overflow <= 0:
            return
        # Metadata-only trim: oldest-created rows beyond the cap, no blob
        # reads — a save stays O(overflow), never O(all rows).
        for key in await asyncio.to_thread(
            engine.oldest_keys, prefix=prefix, skip=self.max_items, take=overflow
        ):
            await engine.adelete(key)

    def _key(self, response_id: str, scope_key: str) -> str:
        digest = hashlib.sha256(
            f"{scope_key}\x00{response_id}".encode("utf-8")
        ).hexdigest()
        return f"{self._prefix}:{digest}"


def create_configured_responses_store(*, config: Any = None, env: Any = None) -> ResponsesStore:
    """Create the configured Responses store backend.

    The engine backend is the durable default; ``memory`` remains an opt-in.
    The engine row TTL keeps the configured TTL authoritative and uses the
    retention floor so a longer configured TTL cannot be clipped by the
    cache engine's own retention.
    """

    from ..config.experimental import get_responses_store_runtime_settings, get_responses_store_settings

    runtime = get_responses_store_runtime_settings(config=config, env=env)
    settings = get_responses_store_settings(config=config, env=env)
    if runtime.backend == "memory":
        return InMemoryResponsesStore(max_items=settings.max_items)
    ttl_seconds = float(runtime.cache_disk_ttl_seconds)
    if settings.ttl_seconds:
        ttl_seconds = max(ttl_seconds, float(settings.ttl_seconds))
    return ProviderCacheResponsesStore(
        prefix=runtime.cache_prefix,
        max_items=settings.max_items,
        ttl_seconds=ttl_seconds,
    )
