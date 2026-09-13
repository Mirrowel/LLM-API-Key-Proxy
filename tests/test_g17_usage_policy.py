# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Usage-store policy pin: accounting rows are never auto-pruned."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_usage_family_sweeper_is_expiry_only(tmp_path):
    from rotator_library.storage.engine import StorageEngine

    engine = StorageEngine(tmp_path / "usage.db", start_sweeper=False)
    try:
        # The default engine has no idle policy: sweep() is expiry-only.
        assert engine._idle_prune is None
        engine.set("acct:old-key", b'{"requests": 42}', ttl_seconds=None)
        with engine._lock:
            engine._conn.execute(
                "UPDATE entries SET last_access = ? WHERE key = 'acct:old-key'",
                (time.time() - 400 * 86400.0,),
            )
        removed = engine.sweep()  # what the background sweeper runs
        assert removed == 0
        assert engine.get("acct:old-key", touch=False) == b'{"requests": 42}'
    finally:
        engine.close()


def test_usage_engine_default_policy_is_none():
    from rotator_library.storage.engine import _ENGINE_CACHE

    engine = _ENGINE_CACHE.get("usage")
    assert engine is None or engine._idle_prune is None
