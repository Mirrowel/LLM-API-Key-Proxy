# src/rotator_library/providers/provider_cache.py
"""
Shared cache utility for providers.

A modular, async-capable cache system supporting:
- Dual-TTL: short-lived memory cache, longer-lived disk persistence
- Background persistence with batched writes
- Automatic cleanup of expired entries
- Generic key-value storage for any provider-specific needs

Usage examples:
- Gemini 3: thoughtSignatures (tool_call_id → encrypted signature)
- Claude: Thinking content (composite_key → thinking text + signature)
- General: Any transient data that benefits from persistence across requests
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..utils.resilient_io import safe_write_json

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

    A generic, modular cache supporting any key-value data that providers need
    to persist across requests. Features:

    - Dual-TTL system: entries live in memory for memory_ttl, but persist on
      disk for the longer disk_ttl. Memory cleanup does NOT affect disk entries.
    - Merge-on-save: disk writes merge current memory with existing disk entries,
      preserving disk-only entries until they exceed disk_ttl
    - Async disk persistence with batched writes
    - Background cleanup task for memory-expired entries (disk untouched)
    - Statistics tracking (hits, misses, writes, disk preservation)

    Args:
        cache_file: Path to disk cache file
        memory_ttl_seconds: In-memory entry lifetime (default: 1 hour)
        disk_ttl_seconds: Disk entry lifetime (default: 48 hours)
        enable_disk: Whether to enable disk persistence (default: from env or True)
        write_interval: Seconds between background disk writes (default: 60)
        cleanup_interval: Seconds between expired entry cleanup (default: 30 min)
        env_prefix: Environment variable prefix for configuration overrides

    Environment Variables (with default prefix "PROVIDER_CACHE"):
        {PREFIX}_ENABLE: Enable/disable disk persistence
        {PREFIX}_WRITE_INTERVAL: Background write interval in seconds
        {PREFIX}_CLEANUP_INTERVAL: Cleanup interval in seconds
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
        # In-memory cache: {cache_key: (data, timestamp)}
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._memory_ttl = memory_ttl_seconds
        self._disk_ttl = disk_ttl_seconds
        self._lock = asyncio.Lock()
        self._disk_lock = asyncio.Lock()

        # Disk persistence configuration
        self._cache_file = cache_file
        self._enable_disk = (
            enable_disk
            if enable_disk is not None
            else _env_bool(f"{env_prefix}_ENABLE", True)
        )
        self._dirty = False
        self._write_interval = write_interval or _env_int(
            f"{env_prefix}_WRITE_INTERVAL", 60
        )
        self._cleanup_interval = cleanup_interval or _env_int(
            f"{env_prefix}_CLEANUP_INTERVAL", 1800
        )

        # Background tasks
        self._writer_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self._stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "writes": 0,
            "disk_errors": 0,
        }

        # Track disk health for monitoring
        self._disk_available = True

        # Metadata about this cache instance
        self._cache_name = cache_file.stem if cache_file else "unnamed"

        if self._enable_disk:
            lib_logger.debug(
                f"ProviderCache[{self._cache_name}]: Disk enabled "
                f"(memory_ttl={memory_ttl_seconds}s, disk_ttl={disk_ttl_seconds}s)"
            )
            asyncio.create_task(self._async_init())
        else:
            lib_logger.debug(f"ProviderCache[{self._cache_name}]: Memory-only mode")

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    async def _async_init(self) -> None:
        """Async initialization: load from disk and start background tasks."""
        try:
            await self._load_from_disk()
            await self._start_background_tasks()
        except Exception as e:
            lib_logger.error(
                f"ProviderCache[{self._cache_name}] async init failed: {e}"
            )

    async def _load_from_disk(self) -> None:
        """Load cache from disk file with TTL validation."""
        if not self._enable_disk or not self._cache_file.exists():
            return

        try:
            async with self._disk_lock:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get("version") != "1.0":
                    lib_logger.warning(
                        f"ProviderCache[{self._cache_name}]: Version mismatch, starting fresh"
                    )
                    return

                now = time.time()
                entries = data.get("entries", {})
                loaded = expired = 0

                for cache_key, entry in entries.items():
                    age = now - entry.get("timestamp", 0)
                    if age <= self._disk_ttl:
                        value = entry.get(
                            "value", entry.get("signature", "")
                        )  # Support both formats
                        if value:
                            self._cache[cache_key] = (value, entry["timestamp"])
                            loaded += 1
                    else:
                        expired += 1

                lib_logger.debug(
                    f"ProviderCache[{self._cache_name}]: Loaded {loaded} entries ({expired} expired)"
                )
        except json.JSONDecodeError as e:
            lib_logger.warning(
                f"ProviderCache[{self._cache_name}]: File corrupted: {e}"
            )
        except Exception as e:
            lib_logger.error(f"ProviderCache[{self._cache_name}]: Load failed: {e}")

    # =========================================================================
    # DISK PERSISTENCE
    # =========================================================================

    async def _save_to_disk(self) -> bool:
        """Persist cache to disk using atomic write with health tracking.

        Implements dual-TTL preservation: merges current memory state with
        existing disk entries that haven't exceeded disk_ttl. This ensures
        entries persist on disk for the full disk_ttl even after they expire
        from memory (which uses the shorter memory_ttl).

        Returns:
            True if write succeeded, False otherwise.
        """
        if not self._enable_disk:
            return True  # Not an error if disk is disabled

        async with self._disk_lock:
            now = time.time()

            # Step 1: Load existing disk entries (if any)
            existing_entries: Dict[str, Dict[str, Any]] = {}
            if self._cache_file.exists():
                try:
                    with open(self._cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    existing_entries = data.get("entries", {})
                except (json.JSONDecodeError, IOError, OSError):
                    pass  # Start fresh if corrupted or unreadable

            # Step 2: Filter existing disk entries by disk_ttl (not memory_ttl)
            # This preserves entries that expired from memory but are still valid on disk
            valid_disk_entries = {
                k: v
                for k, v in existing_entries.items()
                if now - v.get("timestamp", 0) <= self._disk_ttl
            }

            # Step 3: Merge - memory entries take precedence (fresher timestamps)
            merged_entries = valid_disk_entries.copy()
            for key, (val, ts) in self._cache.items():
                merged_entries[key] = {"value": val, "timestamp": ts}

            # Count entries that were preserved from disk (not in memory)
            memory_keys = set(self._cache.keys())
            preserved_from_disk = len(
                [k for k in valid_disk_entries if k not in memory_keys]
            )

            # Step 4: Build and save merged cache data
            cache_data = {
                "version": "1.0",
                "memory_ttl_seconds": self._memory_ttl,
                "disk_ttl_seconds": self._disk_ttl,
                "entries": merged_entries,
                "statistics": {
                    "total_entries": len(merged_entries),
                    "memory_entries": len(self._cache),
                    "disk_preserved": preserved_from_disk,
                    "last_write": now,
                    **self._stats,
                },
            }

            if safe_write_json(
                self._cache_file, cache_data, lib_logger, secure_permissions=True
            ):
                self._stats["writes"] += 1
                self._disk_available = True
                # Log merge info only when we preserved disk-only entries (infrequent)
                if preserved_from_disk > 0:
                    lib_logger.debug(
                        f"ProviderCache[{self._cache_name}]: Saved {len(merged_entries)} entries "
                        f"(memory={len(self._cache)}, preserved_from_disk={preserved_from_disk})"
                    )
                return True
            else:
                self._stats["disk_errors"] += 1
                self._disk_available = False
                return False

    # =========================================================================
    # BACKGROUND TASKS
    # =========================================================================

    async def _start_background_tasks(self) -> None:
        """Start background writer and cleanup tasks."""
        if not self._enable_disk or self._running:
            return

        self._running = True
        self._writer_task = asyncio.create_task(self._writer_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        lib_logger.debug(f"ProviderCache[{self._cache_name}]: Started background tasks")

    async def _writer_loop(self) -> None:
        """Background task: periodically flush dirty cache to disk."""
        try:
            while self._running:
                await asyncio.sleep(self._write_interval)
                if self._dirty:
                    try:
                        success = await self._save_to_disk()
                        if success:
                            self._dirty = False
                        # If save failed, _dirty remains True so we retry next interval
                    except Exception as e:
                        lib_logger.error(
                            f"ProviderCache[{self._cache_name}]: Writer error: {e}"
                        )
        except asyncio.CancelledError:
            pass

    async def _cleanup_loop(self) -> None:
        """Background task: periodically clean up expired entries."""
        try:
            while self._running:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
        except asyncio.CancelledError:
            pass

    async def _cleanup_expired(self) -> None:
        """Remove expired entries from memory cache.

        Only cleans memory - disk entries are preserved and cleaned during
        _save_to_disk() based on their own disk_ttl.
        """
        async with self._lock:
            now = time.time()
            expired = [
                k for k, (_, ts) in self._cache.items() if now - ts > self._memory_ttl
            ]
            for k in expired:
                del self._cache[k]
            # Don't set dirty flag: memory cleanup shouldn't trigger disk write
            # Disk entries are cleaned separately in _save_to_disk() by disk_ttl
            if expired:
                lib_logger.debug(
                    f"ProviderCache[{self._cache_name}]: Cleaned {len(expired)} expired entries from memory"
                )

    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================

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
            self._cache[key] = (value, time.time())
            self._dirty = True

    async def store_async(self, key: str, value: str) -> None:
        """
        Store a value asynchronously (awaitable).

        Use this when you need to ensure the value is stored before continuing.
        """
        await self._async_store(key, value)

    def retrieve(self, key: str) -> Optional[str]:
        """
        Retrieve a value by key (synchronous, with optional async disk fallback).

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp <= self._memory_ttl:
                self._stats["memory_hits"] += 1
                return value
            else:
                # Entry expired from memory - remove from memory only
                # Don't set dirty flag: disk copy should persist until disk_ttl
                del self._cache[key]

        self._stats["misses"] += 1
        if self._enable_disk:
            # Schedule async disk lookup for next time
            asyncio.create_task(self._check_disk_fallback(key))
        return None

    async def retrieve_async(self, key: str) -> Optional[str]:
        """
        Retrieve a value asynchronously (checks disk if not in memory).

        Use this when you can await and need guaranteed disk fallback.
        """
        # Check memory first
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp <= self._memory_ttl:
                self._stats["memory_hits"] += 1
                return value
            else:
                # Entry expired from memory - remove from memory only
                # Don't set dirty flag: disk copy should persist until disk_ttl
                async with self._lock:
                    if key in self._cache:
                        del self._cache[key]

        # Check disk
        if self._enable_disk:
            return await self._disk_retrieve(key)

        self._stats["misses"] += 1
        return None

    async def _check_disk_fallback(self, key: str) -> None:
        """Check disk for key and load into memory if found (background)."""
        try:
            if not self._cache_file.exists():
                return

            async with self._disk_lock:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                entries = data.get("entries", {})
                if key in entries:
                    entry = entries[key]
                    ts = entry.get("timestamp", 0)
                    if time.time() - ts <= self._disk_ttl:
                        value = entry.get("value", entry.get("signature", ""))
                        if value:
                            async with self._lock:
                                self._cache[key] = (value, ts)
                                self._stats["disk_hits"] += 1
                            lib_logger.debug(
                                f"ProviderCache[{self._cache_name}]: Loaded {key} from disk"
                            )
        except Exception as e:
            lib_logger.debug(
                f"ProviderCache[{self._cache_name}]: Disk fallback failed: {e}"
            )

    async def _disk_retrieve(self, key: str) -> Optional[str]:
        """Direct disk retrieval with loading into memory."""
        try:
            if not self._cache_file.exists():
                self._stats["misses"] += 1
                return None

            async with self._disk_lock:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                entries = data.get("entries", {})
                if key in entries:
                    entry = entries[key]
                    ts = entry.get("timestamp", 0)
                    if time.time() - ts <= self._disk_ttl:
                        value = entry.get("value", entry.get("signature", ""))
                        if value:
                            async with self._lock:
                                self._cache[key] = (value, ts)
                            self._stats["disk_hits"] += 1
                            return value

            self._stats["misses"] += 1
            return None
        except Exception as e:
            lib_logger.debug(
                f"ProviderCache[{self._cache_name}]: Disk retrieve failed: {e}"
            )
            self._stats["misses"] += 1
            return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def contains(self, key: str) -> bool:
        """Check if key exists in memory cache (without updating stats)."""
        if key in self._cache:
            _, timestamp = self._cache[key]
            return time.time() - timestamp <= self._memory_ttl
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics including disk health."""
        return {
            **self._stats,
            "memory_entries": len(self._cache),
            "dirty": self._dirty,
            "disk_enabled": self._enable_disk,
            "disk_available": self._disk_available,
        }

    async def clear(self) -> None:
        """Clear all cached data."""
        async with self._lock:
            self._cache.clear()
            self._dirty = True
        if self._enable_disk:
            await self._save_to_disk()

    async def shutdown(self) -> None:
        """Graceful shutdown: flush pending writes and stop background tasks."""
        lib_logger.info(f"ProviderCache[{self._cache_name}]: Shutting down...")
        self._running = False

        # Cancel background tasks
        for task in (self._writer_task, self._cleanup_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Final save
        if self._dirty and self._enable_disk:
            await self._save_to_disk()

        lib_logger.info(
            f"ProviderCache[{self._cache_name}]: Shutdown complete "
            f"(stats: mem_hits={self._stats['memory_hits']}, "
            f"disk_hits={self._stats['disk_hits']}, misses={self._stats['misses']})"
        )


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
        memory_ttl_seconds: In-memory TTL
        disk_ttl_seconds: Disk TTL
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
