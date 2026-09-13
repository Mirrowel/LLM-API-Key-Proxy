"""Archive format, naming, and compression (G10 Phase A).

One sealed transaction record becomes exactly one file on disk:

    ``MMDD_HHMMSS_{protocol}[_{profile}]_{provider}_{model}_{request_id}.transaction.zst``

The envelope is a single zstd frame (JSON payload), plain level-3 zstd.
(Trained per-family dictionaries are a deferred follow-up — noted in the
session ledger; they slot into ``compress_envelope``/``decompress_archive``
without touching the record or writer layers.)
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from ..utils.paths import get_default_root
from ..utils import zstd_io

ARCHIVE_SUFFIX = ".transaction.zst"
ARCHIVE_FORMAT = "proxy-transaction/2"

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", "COM1", "LPT1"}


def transactions_dir() -> Path:
    root = get_default_root()
    base = Path(root) / "logs" / "transactions"
    base.mkdir(parents=True, exist_ok=True)
    return base


def sanitize_component(value: str, fallback: str = "x") -> str:
    cleaned = _UNSAFE_NAME.sub("-", str(value or "").strip())[:48].strip("-.")
    if not cleaned or cleaned.upper() in _WINDOWS_RESERVED:
        return fallback
    return cleaned


def archive_filename(
    *,
    protocol: str,
    provider: str,
    model: str,
    request_id: str,
    profile: Optional[str] = None,
    when: Optional[float] = None,
) -> str:
    """Build the archive filename: timestamp + identity, path-length safe."""

    moment = time.localtime(when if when is not None else time.time())
    parts = [
        # Year leads: newest-N sorts by filename and a year-less stamp
        # inverts across New Year (December would sort before January).
        time.strftime("%Y%m%d_%H%M%S", moment),
        sanitize_component(protocol, "proto"),
    ]
    if profile:
        parts.append(sanitize_component(profile, "prof"))
    parts.extend(
        [
            sanitize_component(provider, "prov"),
            sanitize_component(model, "model"),
            sanitize_component(request_id, "req"),
        ]
    )
    name = "_".join(parts) + ARCHIVE_SUFFIX
    # Windows MAX_PATH guard: leave headroom for the directory.
    if len(name) > 180:
        import hashlib

        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[:150]}_{digest}{ARCHIVE_SUFFIX}"
    return name


def compress_envelope(envelope: dict[str, Any]) -> bytes:
    """Serialize + compress one envelope as a single plain-zstd frame."""

    payload = json.dumps(envelope, default=str, ensure_ascii=False).encode("utf-8")
    return zstd_io.compress_bytes(payload)


def decompress_archive(data: bytes) -> dict[str, Any]:
    """Read one archive back into its envelope (zstd frame or plain JSON)."""

    blob = zstd_io.decompress_bytes(data)
    payload = json.loads(blob.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("archive payload is not an envelope")
    return payload


def read_archive(path: Path) -> dict[str, Any]:
    """Read an archive file from disk (TUI/read tool entry point)."""

    return decompress_archive(Path(path).read_bytes())


def write_archive_atomic(target_dir: Path, filename: str, data: bytes) -> Path:
    """Write one archive atomically: temp file in the same dir + rename."""

    target = target_dir / filename
    tmp = target_dir / (filename + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return target


def iter_archives(base: Optional[Path] = None):
    """Yield archive paths oldest→newest (filename sort = time order)."""

    directory = base or transactions_dir()
    return iter(sorted(path for path in directory.glob(f"*{ARCHIVE_SUFFIX}")))


def _approx_size_json(envelope: dict[str, Any]) -> int:
    """Cheap size estimate for queue budgeting (never serialized twice)."""

    try:
        return len(json.dumps(envelope, default=str))
    except Exception:
        return 65536


def prune_archives(retention: int, base: Optional[Path] = None) -> int:
    """Keep only the newest ``retention`` archives; return removed count.

    Also sweeps stale ``*.tmp`` spill/crash leftovers — they never match
    the archive glob and would otherwise leak forever.
    """

    if retention <= 0:
        return 0
    directory = base or transactions_dir()
    for stale in directory.glob("*.tmp"):
        try:
            stale.unlink()
        except OSError:
            pass
    paths = list(iter_archives(base))
    excess = len(paths) - retention
    removed = 0
    for path in paths[: max(0, excess)]:
        try:
            path.unlink()
            removed += 1
        except OSError:
            pass
    return removed
