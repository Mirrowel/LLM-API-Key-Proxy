# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Store explorer — the basic TUI over the SQLite storage-engine DBs.

Launched from the proxy launcher's main menu (same pattern as the
transaction explorer). Lists every engine database with its vital signs,
drills into rows, shows decompressed values, and exposes the
maintenance actions (delete row, expiry sweep, incremental vacuum).

Scoped deliberately basic per the operator ruling; the polished explorer
comes with the TUI rework after this PR merges.

Module boundary: stdlib-only imports at module level (launcher fast
path); the library loads lazily inside the functions that need it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PAGE_SIZE = 15

FAMILIES = ("cache", "usage", "session")


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _fmt_bytes(count: int | None) -> str:
    if count is None:
        return "-"
    value = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


def _engine_for(family: str):
    from rotator_library.storage.engine import get_engine

    return get_engine(family)


def _family_row(index: int, family: str) -> str:
    try:
        stats = _engine_for(family).stats()
    except Exception as exc:
        return f"{index:>4}. {family:<10} (unavailable: {exc})"
    ratio = stats.get("compression_ratio")
    ratio_text = f"{ratio}x" if ratio else "-"
    return (
        f"{index:>4}. {family:<10} rows={stats.get('rows', 0):<7} "
        f"raw={_fmt_bytes(stats.get('raw_bytes')):<9} stored={_fmt_bytes(stats.get('stored_bytes')):<9} "
        f"ratio={ratio_text:<6} expired={stats.get('expired_rows', 0):<5} "
        f"db={_fmt_bytes(stats.get('file_bytes'))}"
    )


def _pretty(value: bytes, limit: int = 4000) -> str:
    try:
        text = json.dumps(json.loads(value.decode("utf-8")), indent=2, ensure_ascii=False)
    except Exception:
        try:
            text = value.decode("utf-8", errors="replace")
        except Exception:
            text = repr(value)
    if len(text) > limit:
        return text[:limit] + f"\n... ({len(text) - limit} more characters)"
    return text


def _list_rows(engine, family: str, page: int, needle: str = "") -> tuple[list[tuple[str, dict]], int]:
    import time

    now = time.time()
    keys = engine.keys(prefix=needle or None)
    total = len(keys)
    start = page * PAGE_SIZE
    window = keys[start : start + PAGE_SIZE]
    rows = []
    for key in window:
        raw_meta = None
        with engine._lock:
            try:
                found = engine._conn.execute(
                    "SELECT created_at, expires_at, last_access, raw_bytes, hash FROM entries WHERE key = ?",
                    (key,),
                ).fetchone()
            except Exception:
                found = None
        if found:
            raw_meta = {
                "created_at": found[0],
                "expires_at": found[1],
                "last_access": found[2],
                "raw_bytes": found[3],
                "hash": found[4],
            }
        if raw_meta is None:
            continue
        idle = now - (raw_meta.get("last_access") or now)
        raw_meta["idle"] = idle
        rows.append((key, raw_meta))
    return rows, total


def _show_row(engine, key: str, meta: dict) -> None:
    print("=" * 70)
    print(f"ROW: {key}")
    print("=" * 70)
    import time as _time

    now = _time.time()
    print(f"size:    {_fmt_bytes(meta.get('raw_bytes'))} raw")
    print(f"created: {_fmt_age(now - meta['created_at']) if meta.get('created_at') else '-'}")
    expires = meta.get("expires_at")
    print(f"expires: {_fmt_age(expires - now) if expires else 'never'}")
    print(f"idle:    {_fmt_age(meta.get('idle'))}")
    print(f"sha256:  {str(meta.get('hash'))[:16]}…")
    value = engine.get(key, touch=False)
    print("\n--- value (decompressed) ---")
    print(_pretty(value) if value is not None else "(unreadable or missing)")


def run_store_explorer() -> None:
    """The interactive loop: family list → rows → value/actions."""

    page = 0
    while True:
        print("=" * 70)
        print("STORE EXPLORER — engine databases")
        print("=" * 70)
        for i, family in enumerate(FAMILIES, start=1):
            print(_family_row(i, family))
        print()
        print("<family number> open | q quit")
        choice = input("> ").strip().lower()
        if choice in ("q", "quit", ""):
            return
        if not choice.isdigit() and not choice.startswith("/"):
            continue
        needle = ""
        if choice.startswith("/"):
            needle = choice[1:]
            family = FAMILIES[0]
        else:
            index = int(choice) - 1
            if not 0 <= index < len(FAMILIES):
                print("No such family.")
                continue
            family = FAMILIES[index]
        try:
            engine = _engine_for(family)
        except Exception as exc:
            print(f"Cannot open {family}: {exc}")
            continue
        _browse_family(engine, family, needle)


def _browse_family(engine, family: str, needle: str) -> None:
    page = 0
    while True:
        rows, total = _list_rows(engine, family, page, needle)
        max_page = max(0, (total - 1) // PAGE_SIZE)
        print("=" * 70)
        print(f"{family.upper()} — {total} rows (page {page + 1}/{max_page + 1})")
        print("=" * 70)
        for i, (key, meta) in enumerate(rows, start=page * PAGE_SIZE + 1):
            print(f"{i:>5}. {key[:52]:<52} {_fmt_bytes(meta.get('raw_bytes')):<8} idle {_fmt_age(meta.get('idle')):<7}")
        print()
        print("<number> open | /text search | n/p page | s sweep | v vacuum | b back | q quit")
        choice = input("> ").strip()
        low = choice.lower()
        if low in ("q", "quit"):
            raise SystemExit(0)
        if low == "b":
            return
        if low in ("n", "next"):
            page += 1
            continue
        if low in ("p", "prev"):
            page -= 1
            continue
        if low == "s":
            removed = engine.sweep()
            print(f"swept {removed} expired rows")
            continue
        if low == "v":
            engine.vacuum_incremental()
            print("incremental vacuum done")
            continue
        if low.startswith("/"):
            return  # search restarts at family list in this basic version
        if not choice.isdigit():
            continue
        index = int(choice) - 1
        if not 0 <= index < total:
            print("No such row.")
            continue
        rows2, _ = _list_rows(engine, family, index // PAGE_SIZE, needle)
        entry = rows2[index % PAGE_SIZE] if index % PAGE_SIZE < len(rows2) else None
        if entry is None:
            print("Row not readable.")
            continue
        key, meta = entry
        _show_row(engine, key, meta)
        print("\n[d] delete this row | [Enter] back")
        sub = input("> ").strip().lower()
        if sub == "d":
            engine.delete(key)
            print("deleted")


def main(argv: list[str] | None = None) -> int:
    try:
        run_store_explorer()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
