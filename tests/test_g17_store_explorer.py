# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Store explorer tests: rendering, listing, value display."""

from __future__ import annotations

import ast
import inspect
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_module_level_is_stdlib_only():
    import proxy_app.store_explorer as module

    tree = ast.parse(inspect.getsource(module))
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith(("rotator_library", "proxy_app")) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("rotator_library", "proxy_app"))


def test_fmt_helpers():
    from proxy_app.store_explorer import _fmt_age, _fmt_bytes

    assert _fmt_age(None) == "-"
    assert _fmt_age(30) == "30s"
    assert _fmt_age(7200) == "2.0h"
    assert _fmt_bytes(None) == "-"
    assert _fmt_bytes(2048) == "2.0KB"


def test_family_row_and_listing(tmp_path, monkeypatch):
    from rotator_library.storage.engine import StorageEngine
    import proxy_app.store_explorer as se

    engine = StorageEngine(tmp_path / "cache.db", start_sweeper=False)
    monkeypatch.setattr(se, "_engine_for", lambda family: engine)
    try:
        engine.set("resp:one", b'{"id": 1}', ttl_seconds=600)
        engine.set("resp:two", b'{"id": 2}', ttl_seconds=600)
        row = se._family_row(1, "cache")
        assert "cache" in row and "rows=2" in row and "expired=0" in row

        rows, total = se._list_rows(engine, "cache", 0)
        assert total == 2
        keys = {k for k, _ in rows}
        assert keys == {"resp:one", "resp:two"}
        meta = dict(rows)[next(iter(keys))]
        assert meta["raw_bytes"] == len(b'{"id": 1}') and "hash" in meta

        se._show_row(engine, "resp:one", meta)  # renders without raising
    finally:
        engine.close()


def test_write_amplification_row_level(tmp_path):
    """The G17 benchmark pin: saving N+1 rows touches exactly N+1 sets —
    one row per credential, never a whole-store rewrite."""

    from rotator_library.storage.engine import StorageEngine

    engine = StorageEngine(tmp_path / "bench.db", start_sweeper=False)
    try:
        for i in range(50):
            engine.set(f"k{i}", b"x" * 100, ttl_seconds=600)
        with engine._lock:
            before = engine._conn.total_changes
        engine.set("k25", b"y" * 100, ttl_seconds=600)  # one more row write
        with engine._lock:
            after = engine._conn.total_changes
        assert after - before == 1  # exactly one INSERT — no whole-store churn
        # A same-content rewrite is lifecycle-only (one UPDATE, no blob).
        engine.set("k25", b"y" * 100, ttl_seconds=600)
        with engine._lock:
            after2 = engine._conn.total_changes
        assert after2 - after == 1
        # A read touch is one UPDATE.
        assert engine.get("k25") == b"y" * 100
        with engine._lock:
            after3 = engine._conn.total_changes
        assert after3 - after2 == 1
    finally:
        engine.close()
