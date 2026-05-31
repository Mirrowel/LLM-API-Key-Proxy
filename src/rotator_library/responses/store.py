# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Storage backends for Responses API objects."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Optional, Protocol

from .types import StoredResponse


class ResponsesStore(Protocol):
    """Minimal async store for response retrieval and continuation."""

    async def save(self, response: StoredResponse) -> None: ...

    async def get(self, response_id: str) -> Optional[StoredResponse]: ...

    async def delete(self, response_id: str) -> bool: ...

    async def list_input_items(self, response_id: str) -> Optional[list[Any]]: ...


class InMemoryResponsesStore:
    """Process-local Responses store.

    This is the Phase 4 default because it has no async lifecycle and avoids a
    new persistence dependency. A provider-cache-backed store can be injected by
    later configuration code when disk persistence is desired.
    """

    def __init__(self, *, max_items: int | None = None) -> None:
        self._responses: dict[str, StoredResponse] = {}
        self.max_items = max_items if max_items and max_items > 0 else None

    async def save(self, response: StoredResponse) -> None:
        self._prune_expired()
        self._responses[response.id] = StoredResponse.from_dict(response.to_dict())
        self._prune_overflow()

    async def get(self, response_id: str) -> Optional[StoredResponse]:
        response = self._responses.get(response_id)
        if response is None:
            return None
        if response.is_expired():
            self._responses.pop(response_id, None)
            return None
        return StoredResponse.from_dict(response.to_dict())

    async def delete(self, response_id: str) -> bool:
        return self._responses.pop(response_id, None) is not None

    async def list_input_items(self, response_id: str) -> Optional[list[Any]]:
        response = await self.get(response_id)
        if response is None:
            return None
        return deepcopy(response.input_items)

    def _prune_expired(self) -> None:
        for response_id, response in list(self._responses.items()):
            if response.is_expired():
                self._responses.pop(response_id, None)

    def _prune_overflow(self) -> None:
        if not self.max_items:
            return
        while len(self._responses) > self.max_items:
            oldest_id = min(self._responses.values(), key=lambda response: response.created_at).id
            self._responses.pop(oldest_id, None)


class ProviderCacheResponsesStore:
    """Responses store backed by an injected `ProviderCache` instance.

    The wrapper does not instantiate `ProviderCache`, because that class starts
    background tasks. The caller owns cache lifecycle and shutdown.
    """

    def __init__(self, provider_cache: Any, *, prefix: str = "responses") -> None:
        self._cache = provider_cache
        self._prefix = prefix

    async def save(self, response: StoredResponse) -> None:
        await self._cache.store_async(self._key(response.id), json.dumps(response.to_dict(), ensure_ascii=False))

    async def get(self, response_id: str) -> Optional[StoredResponse]:
        raw = await self._cache.retrieve_async(self._key(response_id))
        if raw is None:
            return None
        response = StoredResponse.from_dict(json.loads(raw))
        if response.is_expired():
            await self.delete(response_id)
            return None
        return response

    async def delete(self, response_id: str) -> bool:
        delete = getattr(self._cache, "delete_async", None)
        if delete:
            return bool(await delete(self._key(response_id)))
        # ProviderCache currently exposes clear(), not key-level deletion. When
        # key deletion is unavailable, avoid clearing unrelated provider state.
        return False

    async def list_input_items(self, response_id: str) -> Optional[list[Any]]:
        response = await self.get(response_id)
        if response is None:
            return None
        return deepcopy(response.input_items)

    def _key(self, response_id: str) -> str:
        safe_id = response_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        return f"{self._prefix}:{safe_id}"
