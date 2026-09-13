# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/provider_cache.py
"""
Shared cache utility for providers.

A modular, async-capable cache system backed by the G17 storage engine
(``rotator_library.storage.engine.get_engine("cache")``). Rows live while
they are used and die when they go idle or expire: one row TTL plus engine
access-time tracking replaces the former dual memory/disk split. The
``memory_ttl_seconds`` constructor argument is kept for call-site
compatibility but no longer creates a second retention tier.

Usage examples:
- Gemini 3: thoughtSignatures (tool_call_id → encrypted signature)
- Claude: Thinking content (composite_key → thinking text + signature)
- General: Any transient data that benefits from persistence across requests
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..storage.engine import get_engine

lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    return os.getenv(key, str(default).lower()).lower() in ("true", "1", "yes")


def _env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    return int(os.getenv(key, str(default)))


# =============================================================================
# PROVIDER CACHE CLASS
# =============================================================================


class ProviderCache:
    """
    Server-side cache for provider conversation state preservation.

    A generic key-value cache over the shared ``cache`` storage engine.
    Features:

    - Single-TTL rows: every value lives until its TTL expires, and the
      engine refreshes ``last_access`` on every read so hot keys survive
      idle pruning. The former memory/disk split is collapsed into this
      one row lifetime (rows live while used, die when idle/expired).
    - Atomic per-key writes through the engine's dedupe-aware upsert.
    - Key-level deletion and namespace-scoped clearing.
    - Statistics derived from the engine counters.

    Args:
        cache_file: Path used for naming/logging only (engine holds data)
        memory_ttl_seconds: Retained for call-site compatibility; unused
        disk_ttl_seconds: Row lifetime in seconds (default: 48 hours)
        enable_disk: Whether durable storage is enabled (default: env/True)
        write_interval: Retained for call-site compatibility; unused
        cleanup_interval: Retained for call-site compatibility; unused
        env_prefix: Environment variable prefix for configuration overrides

    Environment Variables (with default prefix "PROVIDER_CACHE"):
        {PREFIX}_ENABLE: Enable/disable persistence
    """

    def __init__(
        self,
        cache_file: Path,
        memory_ttl_seconds: int = 3600,
        disk_ttl_seconds: int = 172800,  # 48 hours
        enable_disk: Optional[bool] = None,
        write_interval: Optional[int] = None,
        cleanup_interval: Optional[int] = None,
        env_prefix: str = "PROVIDER_CACHE",
    ):
        self._cache_file = cache_file
        self._memory_ttl = memory_ttl_seconds
        self._disk_ttl = disk_ttl_seconds
        self._lock = asyncio.Lock()
        self._enable_disk = (
            enable_disk
            if enable_disk is not None
            else _env_bool(f"{env_prefix}_ENABLE", True)
        )
        self._write_interval = write_interval or _env_int(
            f"{env_prefix}_WRITE_INTERVAL", 60
        )
        self._cleanup_interval = cleanup_interval or _env_int(
            f"{env_prefix}_CLEANUP_INTERVAL", 1800
        )
        self._cache_name = cache_file.stem if cache_file else "unnamed"
        identity = str(cache_file) if cache_file else "unnamed"
        self._namespace = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        self._engine = get_engine("cache")
        lib_logger.debug(
            f"ProviderCache[{self._cache_name}]: Engine-backed "
            f"(row_ttl={disk_ttl_seconds}s, namespace={self._namespace})"
        )

    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================

    def _row_key(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    def _write(self, key: str, value: str) -> bool:
        if not self._enable_disk:
            return False
        try:
            raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        except (TypeError, ValueError):
            return False
        ttl = self._disk_ttl if self._disk_ttl and self._disk_ttl > 0 else None
        return self._engine.set(self._row_key(key), raw, ttl_seconds=ttl)

    def _read(self, key: str, *, touch: bool = True) -> Optional[str]:
        if not self._enable_disk:
            return None
        raw = self._engine.get(self._row_key(key), touch=touch)
        if raw is None:
            return None
        try:
            return raw.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            return None

    def store(self, key: str, value: str) -> None:
        """
        Store a value synchronously (schedules async storage).

        Args:
            key: Cache key
            value: Value to store (typically JSON-serialized data)
        """
        asyncio.create_task(self._async_store(key, value))

    async def _async_store(self, key: str, value: str) -> None:
        """Async implementation of store."""
        async with self._lock:
            self._write(key, value)

    async def store_async(self, key: str, value: str) -> None:
        """
        Store a value asynchronously (awaitable).

        Use this when you need to ensure the value is stored before continuing.
        """
        await self._async_store(key, value)

    async def update_async(
        self,
        key: str,
        update: Callable[[Optional[str]], str],
    ) -> str:
        """Atomically update one value within this cache backend instance."""

        async with self._lock:
            updated = update(self._read(key))
            self._write(key, updated)
            return updated

    def retrieve(self, key: str) -> Optional[str]:
        """
        Retrieve a value by key synchronously.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        return self._read(key)

    async def retrieve_async(self, key: str) -> Optional[str]:
        """
        Retrieve a value asynchronously.

        Use this when you can await and need guaranteed access-time refresh.
        """
        raw = await self._engine.aget(self._row_key(key), touch=True)
        if raw is None:
            return None
        try:
            return raw.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            return None

    async def delete_async(self, key: str) -> bool:
        """Delete one key from durable storage."""

        return await self._engine.adelete(self._row_key(key))

    def contains(self, key: str) -> bool:
        """Check if key exists without updating access time or stats."""
        return self._engine.get(self._row_key(key), touch=False) is not None

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics mapped from the shared engine counters."""
        counters = self._engine.stats_counters
        hits = max(0, counters["gets"] - counters["misses"])
        return {
            "memory_hits": hits,
            "disk_hits": 0,
            "misses": counters["misses"],
            "writes": counters["sets"],
            "disk_errors": 0,
            "memory_entries": self._engine.size(),
            "dirty": False,
            "disk_enabled": self._enable_disk,
            "disk_available": True,
        }

    async def clear(self) -> None:
        """Clear all values owned by this cache namespace."""

        prefix = f"{self._namespace}:"
        for key in self._engine.keys(prefix=prefix):
            self._engine.delete(key)

    async def shutdown(self) -> None:
        """No-op: the process-wide engine is closed by the app lifespan."""
        if self._enable_disk:
            await self._engine.asweep()


# =============================================================================
# CONVENIENCE FACTORY
# =============================================================================


def create_provider_cache(
    name: str,
    cache_dir: Optional[Path] = None,
    memory_ttl_seconds: int = 3600,
    disk_ttl_seconds: int = 172800,  # 48 hours
    env_prefix: Optional[str] = None,
) -> ProviderCache:
    """
    Factory function to create a provider cache with sensible defaults.

    Args:
        name: Cache name (used as filename and for logging)
        cache_dir: Directory for cache file (default: project_root/cache/provider_name)
        memory_ttl_seconds: Retained for call-site compatibility; unused
        disk_ttl_seconds: Row TTL
        env_prefix: Environment variable prefix (default: derived from name)

    Returns:
        Configured ProviderCache instance
    """
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent.parent.parent.parent / "cache"

    cache_file = cache_dir / f"{name}.json"

    if env_prefix is None:
        # Convert name to env prefix: "gemini3_signatures" -> "GEMINI3_SIGNATURES_CACHE"
        env_prefix = f"{name.upper().replace('-', '_')}_CACHE"

    return ProviderCache(
        cache_file=cache_file,
        memory_ttl_seconds=memory_ttl_seconds,
        disk_ttl_seconds=disk_ttl_seconds,
        env_prefix=env_prefix,
    )
