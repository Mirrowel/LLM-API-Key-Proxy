# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G17 storage engine tests: round-trip, TTL, touch, dedupe, corruption,
sweeps, stats, WAL pragmas."""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.storage.engine import StorageEngine  # noqa: E402


def make_engine(tmp_path, **kwargs) -> StorageEngine:
    return StorageEngine(tmp_path / "test.db", sweep_interval_seconds=3600, **kwargs)


def test_round_trip_bytes_and_json(tmp_path):
    engine = make_engine(tmp_path, start_sweeper=False)
    try:
        engine.set("k1", b"hello world")
        assert engine.get("k1") == b"hello world"

        payload = json.dumps({"model": "llama-3", "messages": [{"role": "user"}] * 50}).encode()
        engine.set("k2", payload, ttl_seconds=60)
        got = json.loads(engine.get("k2"))
        assert got["model"] == "llama-3"
        assert engine.size() == 2
    finally:
        engine.close()


def test_ttl_expiry(tmp_path):
    engine = make_engine(tmp_path)
    try:
        engine.set("short", b"x", ttl_seconds=0.05)
        assert engine.get("short", touch=False) == b"x"
        time.sleep(0.08)
        assert engine.get("short") is None  # expired row is a miss
        assert engine.size() == 0  # and is reaped on read
    finally:
        engine.close()


def test_no_ttl_rows_persist(tmp_path):
    engine = make_engine(tmp_path, start_sweeper=False)
    try:
        engine.set("forever", b"y")
        assert engine.get("forever") is not None
        assert engine.sweep() == 0  # nothing to expire
    finally:
        engine.close()


def test_touch_updates_last_access_and_idle_sweep(tmp_path):
    engine = make_engine(tmp_path, start_sweeper=False)
    try:
        now = time.time()
        engine.set("touched", b"v")
        engine.set("stale", b"v")
        with engine._lock:
            engine._conn.execute(
                "UPDATE entries SET last_access = ? WHERE key = 'stale'", (now - 100.0,)
            )
        removed = engine.sweep(max_idle_seconds=50.0)
        assert removed == 1
        assert engine.get("touched") == b"v"  # recently accessed survives
        assert engine.get("stale") is None
    finally:
        engine.close()


def test_dedupe_skip_same_content(tmp_path):
    engine = make_engine(tmp_path, start_sweeper=False)
    try:
        assert engine.set("k", b"same-bytes") is True
        with engine._lock:
            before = engine._conn.total_changes
        assert engine.set("k", b"same-bytes") is False  # dedupe skip
        with engine._lock:
            after = engine._conn.total_changes
        assert after - before == 1  # only the lifecycle UPDATE, no blob write
        assert engine.stats_counters["dedupe_skips"] == 1
        # Different content still writes.
        assert engine.set("k", b"other-bytes") is True
        assert engine.get("k") == b"other-bytes"
    finally:
        engine.close()


def test_corrupt_blob_is_miss_and_reaped(tmp_path):
    engine = make_engine(tmp_path, start_sweeper=False)
    try:
        engine.set("bad", b"good")
        with engine._lock:
            engine._conn.execute("UPDATE entries SET value = ? WHERE key = 'bad'", (b"\x00garbage",))
        assert engine.get("bad") is None  # miss, never raises
        assert engine.size() == 0  # row reaped
    finally:
        engine.close()


def test_delete_and_clear(tmp_path):
    engine = make_engine(tmp_path, start_sweeper=False)
    try:
        engine.set("a", b"1")
        engine.set("b", b"2")
        assert engine.delete("a") is True
        assert engine.delete("missing") is False
        engine.clear()
        assert engine.size() == 0
    finally:
        engine.close()


def test_iterate_and_keys_with_prefix(tmp_path):
    engine = make_engine(tmp_path, start_sweeper=False)
    try:
        engine.set("resp:one", b"{\"id\": 1}")
        engine.set("resp:two", b"2")
        engine.set("other:three", b"3")
        keys = engine.keys(prefix="resp:")
        assert keys == ["resp:one", "resp:two"]
        rows = list(engine.iterate(prefix="resp:"))
        assert rows[0][0] == "resp:one" and json.loads(rows[0][1])["id"] == 1
        meta = rows[0][2]
        assert "created_at" in meta and "hash" in meta
    finally:
        engine.close()


def test_stats_shape_and_compression(tmp_path):
    engine = make_engine(tmp_path, start_sweeper=False)
    try:
        payload = (json.dumps({"messages": [{"role": "user", "content": "x" * 200}] * 20})).encode()
        engine.set("big", payload)
        stats = engine.stats()
        assert stats["rows"] == 1
        assert stats["journal_mode"].lower() == "wal"
        assert stats["raw_bytes"] == len(payload)
        if engine_stats_ratio_possible():
            assert stats["compression_ratio"] is None or stats["compression_ratio"] >= 1.0
    finally:
        engine.close()


def engine_stats_ratio_possible() -> bool:
    # zstd may be unavailable (plain passthrough) — ratio then equals ~1.
    from rotator_library.utils import zstd_io

    return zstd_io.compression_available()


def test_wal_pragmas_on_open(tmp_path):
    engine = make_engine(tmp_path, start_sweeper=False)
    try:
        mode = engine._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() == "wal"
        sync = engine._conn.execute("PRAGMA synchronous").fetchone()[0]
        assert int(sync) == 1  # NORMAL
    finally:
        engine.close()


def test_concurrent_thread_access(tmp_path):
    engine = make_engine(tmp_path, start_sweeper=False)
    try:
        errors: list[str] = []

        def worker(n: int) -> None:
            try:
                for i in range(50):
                    key = f"w{n}:k{i}"
                    engine.set(key, f"payload-{n}-{i}".encode())
                    assert engine.get(key) == f"payload-{n}-{i}".encode()
            except Exception as exc:  # pragma: no cover
                errors.append(f"{n}: {exc}")

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        assert engine.size() == 200
    finally:
        engine.close()


def test_reopen_persists_rows(tmp_path):
    path = tmp_path / "reopen.db"
    engine = StorageEngine(path, sweep_interval_seconds=3600, start_sweeper=False)
    engine.set("persist", b"survives", ttl_seconds=600)
    engine.close()
    engine2 = StorageEngine(path, sweep_interval_seconds=3600, start_sweeper=False)
    try:
        assert engine2.get("persist") == b"survives"
    finally:
        engine2.close()


def test_close_is_idempotent_and_checkpointed(tmp_path):
    engine = make_engine(tmp_path, start_sweeper=False)
    engine.set("k", b"v")
    engine.close()
    engine.close()  # idempotent
    wal = tmp_path / "test.db-wal"
    if wal.exists():
        assert wal.stat().st_size == 0  # TRUNCATE checkpoint ran
