# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""The storage engine (G17): one SQLite-backed KV store per data family.

Replaces the whole-file-rewrite JSON caches with indexed row storage:

- WAL + ``synchronous=NORMAL``: append-only writes, no per-commit fsync
  stall (the documented production balance for WAL mode).
- Per-row ``expires_at`` (TTL) and ``last_access`` (idle pruning) with
  indexes — pruning becomes indexed DELETEs instead of full scans.
- Values stored as zstd blobs through :mod:`rotator_library.utils.zstd_io`
  (plain bytes when ``zstandard`` is unavailable — the read side sniffs
  the magic, same graceful degradation as transaction archives).
- Light dedupe: writing the same content under the same key is a no-op
  for the blob (only ``expires_at``/``last_access`` refresh — a recorded
  value that comes again is not recorded again).
- One connection guarded by an RLock; async callers wrap ops in
  ``asyncio.to_thread``. A background sweeper expires rows and runs
  ``incremental_vacuum`` on a cadence.
- Corrupt rows are misses, never errors (the containment doctrine every
  store already follows).

No migration from the old JSON stores (operator ruling): fresh start,
old files are untouched leftovers until the orphan cleanup pass.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Tuple

from ..utils import zstd_io

_engine_logger = __import__("logging").getLogger("rotator_library.storage")

DEFAULT_BUSY_TIMEOUT_MS = 5000
DEFAULT_SWEEP_INTERVAL_SECONDS = 300


class StorageEngine:
    """One SQLite-backed KV store (one file, one data family)."""

    def __init__(
        self,
        path: Path,
        *,
        sweep_interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
        idle_prune_seconds: float | None = None,
        start_sweeper: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            timeout=DEFAULT_BUSY_TIMEOUT_MS / 1000.0,
            isolation_level=None,  # autocommit; explicit BEGIN for batches
        )
        self._closed = False
        self._init_schema()
        self.stats_counters: dict[str, int] = {"gets": 0, "misses": 0, "sets": 0, "dedupe_skips": 0, "deletes": 0, "swept": 0}
        self._sweep_interval = sweep_interval_seconds
        self._idle_prune = idle_prune_seconds
        self._sweeper: Optional[threading.Thread] = None
        self._stop = threading.Event()
        if start_sweeper:
            self._sweeper = threading.Thread(target=self._sweep_loop, name=f"storage-sweep-{self.path.stem}", daemon=True)
            self._sweeper.start()

    # -- schema ----------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock:
            # auto_vacuum must be set before the first table exists on a
            # fresh database file to take effect.
            self._conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.execute("PRAGMA cache_size=-64000")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entries(
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    last_access REAL NOT NULL,
                    raw_bytes INTEGER NOT NULL,
                    hash TEXT NOT NULL
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_expires ON entries(expires_at)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_access ON entries(last_access)")

    # -- primitives (thread-safe, blocking) -------------------------------

    def get(self, key: str, *, touch: bool = True) -> Optional[bytes]:
        """One row by key; expired/corrupt/missing are None (never raise)."""

        now = time.time()
        with self._lock:
            self.stats_counters["gets"] += 1
            try:
                row = self._conn.execute(
                    "SELECT value, expires_at, hash FROM entries WHERE key = ?", (key,)
                ).fetchone()
            except sqlite3.Error:
                self.stats_counters["misses"] += 1
                return None
            if row is None:
                self.stats_counters["misses"] += 1
                return None
            blob, expires_at, _hash = row
            if expires_at is not None and expires_at <= now:
                self.stats_counters["misses"] += 1
                try:
                    self._conn.execute("DELETE FROM entries WHERE key = ?", (key,))
                except sqlite3.Error:
                    pass
                return None
            try:
                value = zstd_io.decompress_bytes(bytes(blob))
            except Exception:
                value = None
            if value is None or hashlib.sha256(value).hexdigest() != _hash:
                # Undecodable or hash-mismatched (corrupt or tampered) row:
                # a miss, and the row is dropped so it cannot poison
                # stats or sweeps.
                self.stats_counters["misses"] += 1
                try:
                    self._conn.execute("DELETE FROM entries WHERE key = ?", (key,))
                except sqlite3.Error:
                    pass
                return None
            if touch:
                try:
                    self._conn.execute("UPDATE entries SET last_access = ? WHERE key = ?", (now, key))
                except sqlite3.Error:
                    pass
            return value

    def set(
        self,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[float] = None,
        created_at: Optional[float] = None,
    ) -> bool:
        """Store one row. Returns True when the blob was written, False on
        dedupe skip (same content hash — only TTL/access refresh).
        ``created_at`` lets callers stamp the row's logical creation time
        (eviction order follows it, not the write clock)."""

        now = time.time()
        stamp = float(created_at) if created_at is not None else now
        expires_at = (now + ttl_seconds) if ttl_seconds is not None and ttl_seconds > 0 else None
        raw = bytes(value)
        digest = hashlib.sha256(raw).hexdigest()
        compressed = zstd_io.compress_bytes(raw)
        with self._lock:
            self.stats_counters["sets"] += 1
            try:
                existing = self._conn.execute("SELECT hash FROM entries WHERE key = ?", (key,)).fetchone()
                if existing is not None and existing[0] == digest:
                    # Recorded before, byte-identical: refresh lifecycle
                    # fields only — no blob write, no page churn.
                    self.stats_counters["dedupe_skips"] += 1
                    self._conn.execute(
                        "UPDATE entries SET expires_at = ?, last_access = ? WHERE key = ?",
                        (expires_at, now, key),
                    )
                    return False
                self._conn.execute(
                    """
                    INSERT INTO entries(key, value, created_at, expires_at, last_access, raw_bytes, hash)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        expires_at = excluded.expires_at,
                        last_access = excluded.last_access,
                        raw_bytes = excluded.raw_bytes,
                        hash = excluded.hash
                    """,
                    (key, compressed, stamp, expires_at, now, len(raw), digest),
                )
                return True
            except sqlite3.Error as exc:
                self.stats_counters["write_errors"] = self.stats_counters.get("write_errors", 0) + 1
                _engine_logger.warning("storage engine write failed for %s: %s", self.path.name, exc)
                return False

    def delete(self, key: str) -> bool:
        with self._lock:
            self.stats_counters["deletes"] += 1
            try:
                cursor = self._conn.execute("DELETE FROM entries WHERE key = ?", (key,))
                return cursor.rowcount > 0
            except sqlite3.Error as exc:
                self.stats_counters["write_errors"] = self.stats_counters.get("write_errors", 0) + 1
                _engine_logger.warning("storage engine write failed for %s: %s", self.path.name, exc)
                return False

    def clear(self) -> None:
        with self._lock:
            try:
                self._conn.execute("DELETE FROM entries")
            except sqlite3.Error:
                pass

    def size(self) -> int:
        with self._lock:
            try:
                return int(self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
            except sqlite3.Error:
                return 0

    def iterate(self, *, prefix: Optional[str] = None) -> Iterator[Tuple[str, bytes, dict[str, Any]]]:
        """Yield (key, decompressed value, lifecycle metadata) rows."""

        now = time.time()
        with self._lock:
            try:
                if prefix:
                    rows = self._conn.execute(
                        "SELECT key, value, created_at, expires_at, last_access, raw_bytes, hash "
                        "FROM entries WHERE key LIKE ? ORDER BY key",
                        (prefix + "%",),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT key, value, created_at, expires_at, last_access, raw_bytes, hash "
                        "FROM entries ORDER BY key"
                    ).fetchall()
            except sqlite3.Error:
                return
        verify = zstd_io.compression_available()
        for key, blob, created_at, expires_at, last_access, raw_bytes, digest in rows:
            if expires_at is not None and expires_at <= now:
                continue
            try:
                value = zstd_io.decompress_bytes(bytes(blob))
            except Exception:
                continue
            if verify and hashlib.sha256(value).hexdigest() != digest:
                continue
            yield (
                str(key),
                value,
                {
                    "created_at": created_at,
                    "expires_at": expires_at,
                    "last_access": last_access,
                    "raw_bytes": raw_bytes,
                    "hash": digest,
                },
            )

    def oldest_keys(self, *, prefix: Optional[str] = None, skip: int = 0, take: int = 1) -> list[str]:
        """Eviction candidates under a prefix, metadata only (no blob reads):
        rows BEYOND the newest ``skip`` (skip=cap keeps the newest cap and
        yields the overflow, oldest-evicted order)."""

        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT key FROM entries WHERE key LIKE ? AND (expires_at IS NULL OR expires_at > ?) "
                    "ORDER BY created_at DESC, last_access DESC, key LIMIT ? OFFSET ?",
                    (str(prefix or "") + "%", time.time(), int(take), int(skip)),
                ).fetchall()
                return [str(r[0]) for r in rows]
            except sqlite3.Error:
                return []

    def count_prefix(self, prefix: str) -> int:
        with self._lock:
            try:
                return int(
                    self._conn.execute(
                        "SELECT COUNT(*) FROM entries WHERE key LIKE ? AND (expires_at IS NULL OR expires_at > ?)",
                        (str(prefix) + "%", time.time()),
                    ).fetchone()[0]
                )
            except sqlite3.Error:
                return 0

    def sync_keys(self, prefix: str, keep: Iterable[str]) -> int:
        """Delete rows under the prefix that are NOT in `keep` (compaction
        for stores whose authoritative set lives in memory)."""

        keep_set = set(keep)
        removed = 0
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT key FROM entries WHERE key LIKE ?", (str(prefix) + "%",)
                ).fetchall()
            except sqlite3.Error:
                return 0
        for (key,) in rows:
            if str(key) not in keep_set:
                if self.delete(str(key)):
                    removed += 1
        return removed

    def keys(self, *, prefix: Optional[str] = None) -> list[str]:
        with self._lock:
            try:
                if prefix:
                    rows = self._conn.execute(
                        "SELECT key FROM entries WHERE key LIKE ? ORDER BY key", (prefix + "%",)
                    ).fetchall()
                else:
                    rows = self._conn.execute("SELECT key FROM entries ORDER BY key").fetchall()
                return [str(r[0]) for r in rows]
            except sqlite3.Error:
                return []

    # -- maintenance ------------------------------------------------------

    def sweep(self, *, max_idle_seconds: Optional[float] = None) -> int:
        """Delete expired rows (and optionally idle ones); return count."""

        now = time.time()
        removed = 0
        with self._lock:
            try:
                cursor = self._conn.execute("DELETE FROM entries WHERE expires_at IS NOT NULL AND expires_at <= ?", (now,))
                removed += max(0, cursor.rowcount or 0)
                if max_idle_seconds is not None and max_idle_seconds > 0:
                    cursor = self._conn.execute(
                        "DELETE FROM entries WHERE last_access <= ?", (now - max_idle_seconds,)
                    )
                    removed += max(0, cursor.rowcount or 0)
            except sqlite3.Error:
                pass
            else:
                if removed:
                    self.stats_counters["swept"] += removed
        if removed:
            self.vacuum_incremental()
        return removed

    def vacuum_incremental(self, pages: int = 1000) -> None:
        with self._lock:
            try:
                self._conn.execute(f"PRAGMA incremental_vacuum({int(pages)})")
            except sqlite3.Error:
                pass

    def stats(self) -> dict[str, Any]:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(raw_bytes), 0), COALESCE(SUM(LENGTH(value)), 0), "
                    "COALESCE(MIN(created_at), 0), COALESCE(MAX(created_at), 0), "
                    "COALESCE(SUM(expires_at IS NOT NULL AND expires_at <= ?), 0), "
                    "COALESCE(SUM(last_access <= ?), 0) "
                    "FROM entries",
                    (time.time(), time.time() - 86400.0),
                ).fetchone()
                wal = self._conn.execute("PRAGMA journal_mode").fetchone()[0]
                page_count = self._conn.execute("PRAGMA page_count").fetchone()[0]
                page_size = self._conn.execute("PRAGMA page_size").fetchone()[0]
            except sqlite3.Error:
                return {"rows": 0}
        rows, raw_total, stored_total, oldest, newest, expired, idle_day = row
        return {
            "rows": int(rows),
            "raw_bytes": int(raw_total),
            "stored_bytes": int(stored_total),
            "compression_ratio": round(float(raw_total) / float(stored_total), 2) if stored_total else None,
            "oldest_created": float(oldest) if rows else None,
            "newest_created": float(newest) if rows else None,
            "expired_rows": int(expired),
            "idle_over_day": int(idle_day),
            "journal_mode": str(wal),
            "file_bytes": int(page_count) * int(page_size),
            "counters": dict(self.stats_counters),
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def _sweep_loop(self) -> None:
        while not self._stop.wait(self._sweep_interval):
            try:
                self.sweep(max_idle_seconds=self._idle_prune)
            except Exception:
                pass

    # -- async bridge ------------------------------------------------------

    async def aget(self, key: str, *, touch: bool = True) -> Optional[bytes]:
        return await asyncio.to_thread(self.get, key, touch=touch)

    async def aset(
        self,
        key: str,
        value: bytes,
        *,
        ttl_seconds: Optional[float] = None,
        created_at: Optional[float] = None,
    ) -> bool:
        return await asyncio.to_thread(
            self.set, key, value, ttl_seconds=ttl_seconds, created_at=created_at
        )

    async def adelete(self, key: str) -> bool:
        return await asyncio.to_thread(self.delete, key)

    async def asweep(self, *, max_idle_seconds: Optional[float] = None) -> int:
        return await asyncio.to_thread(self.sweep, max_idle_seconds=max_idle_seconds)


_ENGINE_CACHE: dict[str, "StorageEngine"] = {}
_ENGINE_CACHE_LOCK = threading.Lock()


def get_engine(family: str, *, directory: Optional[Path] = None) -> StorageEngine:
    """The process-wide engine for one data family ('cache' | 'usage' |
    'session'). One file per family keeps corruption isolated and
    retention/vacuum independent."""

    with _ENGINE_CACHE_LOCK:
        engine = _ENGINE_CACHE.get(family)
        if engine is None or engine._closed:
            if directory is not None:
                base = Path(directory)
            else:
                from ..utils.paths import get_default_root

                base = Path(get_default_root()) / "store"
            # usage is accounting, not cache: no idle pruning, no TTL —
            # rows live until their credential is structurally removed.
            # cache idle-default is 3 days of inactivity (the global
            # default now that per-rule TTLs are gone).
            idle_defaults = {"cache": 3 * 86400.0, "session": None}
            engine = StorageEngine(
                base / f"{family}.db",
                idle_prune_seconds=idle_defaults.get(family),
            )
            _ENGINE_CACHE[family] = engine
        return engine


def close_all_engines() -> None:
    with _ENGINE_CACHE_LOCK:
        for engine in _ENGINE_CACHE.values():
            engine.close()
        _ENGINE_CACHE.clear()
