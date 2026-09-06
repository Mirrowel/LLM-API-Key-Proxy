# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Zstd-compressed JSON/JSONL writing with graceful degradation.

One utility owns the compression behavior (D15); call sites hand off data
and destination and stay unchanged whether compression is available or
not. Missing ``zstandard`` degrades to plain files with a metadata flag —
logging never fails a request and never blocks startup.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Iterable

lib_logger = logging.getLogger("rotator_library.zstd_io")

_LOCK = threading.Lock()
try:  # pragma: no cover - import availability is environment-dependent
    import zstandard as _zstd

    _COMPRESSION_AVAILABLE = True
    _COMPRESSOR = _zstd.ZstdCompressor(level=3)
    _DECOMPRESSOR = _zstd.ZstdDecompressor()
except Exception:  # pragma: no cover
    _zstd = None
    _COMPRESSION_AVAILABLE = False
    _COMPRESSOR = None
    _DECOMPRESSOR = None


def compression_available() -> bool:
    return _COMPRESSION_AVAILABLE


def _write_bytes(path: Path, payload: bytes, *, compressed: bool) -> None:
    if compressed and _COMPRESSION_AVAILABLE:
        with _LOCK:
            data = _COMPRESSOR.compress(payload)
    else:
        data = payload
    path.write_bytes(data)


def write_json(path: Path, payload: Any, *, indent: int | None = 2) -> bool:
    """Serialize ``payload`` to ``path`` (``.zst`` suffix when compressing).

    Returns True when the bytes were zstd-compressed.
    """

    text = json.dumps(payload, indent=indent, ensure_ascii=False, default=str)
    compressed = _COMPRESSION_AVAILABLE
    target = Path(str(path) + ".zst") if compressed else path
    try:
        _write_bytes(target, text.encode("utf-8"), compressed=compressed)
        return compressed
    except Exception:
        lib_logger.debug("zstd json write failed for %s; falling back", path, exc_info=True)
        try:
            Path(path).write_text(text, encoding="utf-8")
        except Exception:
            lib_logger.debug("fallback write also failed for %s", path, exc_info=True)
        return False


def append_jsonl(path: Path, entries: Iterable[Any], *, flush: bool = True) -> bool:
    """Append JSONL entries (compressed rewrite when compressing).

    zstd streams are not appendable; the file is decompressed, extended,
    and recompressed. Batch callers should pass many entries per call —
    the rewrite cost is per-call, not per-entry.
    Returns True when the stored bytes are zstd-compressed.
    """

    lines = "".join(
        json.dumps(entry, ensure_ascii=False, default=str) + "\n" for entry in entries
    )
    compressed = _COMPRESSION_AVAILABLE
    target = Path(str(path) + ".zst") if compressed else path
    try:
        if compressed and target.exists():
            with _LOCK:
                existing = _DECOMPRESSOR.decompress(target.read_bytes()).decode("utf-8")
            text = existing + lines
        else:
            text = lines
        _write_bytes(target, text.encode("utf-8"), compressed=compressed)
        return compressed
    except Exception:
        lib_logger.debug("zstd jsonl append failed for %s; falling back", path, exc_info=True)
        try:
            # Fallback keeps ONE readable stream: if a compressed file
            # already exists, merge into a plain side file the reader also
            # merges (never a silent split the reader cannot see).
            fallback = Path(str(path) + ".fallback.jsonl")
            if compressed and target.exists():
                try:
                    with _LOCK:
                        prior = _DECOMPRESSOR.decompress(target.read_bytes()).decode("utf-8")
                    fallback.write_text(prior, encoding="utf-8")
                    target.unlink(missing_ok=True)
                except Exception:
                    pass
            with open(fallback if fallback.exists() else (path if not compressed else fallback), "a", encoding="utf-8") as handle:
                handle.write(lines)
        except Exception:
            lib_logger.debug("fallback append also failed for %s", path, exc_info=True)
        return False


class JsonlBuffer:
    """Bounded in-memory JSONL accumulator (memory-safe L1 chunk capture)."""

    def __init__(self, max_entries: int = 4096, max_chars: int = 4_000_000):
        self.max_entries = max_entries
        self.max_chars = max_chars
        self._entries: list[str] = []
        self._chars = 0
        self.dropped = 0

    def add(self, entry: Any) -> None:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        if len(line) > self.max_chars:
            self._entries.clear()
            self._chars = 0
            self.dropped += 1
            return
        self._entries.append(line)
        self._chars += len(line)
        while self._entries and (
            len(self._entries) > self.max_entries or self._chars > self.max_chars
        ):
            self._chars -= len(self._entries.pop(0))
            self.dropped += 1

    def flush_to(self, path: Path) -> bool:
        if not self._entries:
            return False
        compressed = _COMPRESSION_AVAILABLE
        target = Path(str(path) + ".zst") if compressed else path
        text = "\n".join(self._entries) + "\n"
        try:
            _write_bytes(target, text.encode("utf-8"), compressed=compressed)
            return compressed
        except Exception:
            lib_logger.debug("zstd buffer flush failed for %s", path, exc_info=True)
            return False


def read_json_any(path: Path) -> Any:
    """Read a JSON file whether or not it is zstd-compressed."""

    for candidate in (Path(str(path) + ".zst"), path):
        if not candidate.exists():
            continue
        raw = candidate.read_bytes()
        if candidate.suffix == ".zst":
            if not _COMPRESSION_AVAILABLE:
                raise RuntimeError(f"{candidate} is zstd-compressed but the zstandard package is unavailable")
            with _LOCK:
                raw = _DECOMPRESSOR.decompress(raw)
        return json.loads(raw.decode("utf-8"))
    raise FileNotFoundError(path)


def read_jsonl_any(path: Path) -> list[Any]:
    """Read a JSONL file whether or not it is zstd-compressed (merging any
    fallback side file written after a compression failure)."""

    entries: list[Any] = []
    found = False
    for candidate in (Path(str(path) + ".zst"), path):
        if not candidate.exists():
            continue
        found = True
        raw = candidate.read_bytes()
        if candidate.suffix == ".zst":
            if not _COMPRESSION_AVAILABLE:
                raise RuntimeError(f"{candidate} is zstd-compressed but the zstandard package is unavailable")
            with _LOCK:
                raw = _DECOMPRESSOR.decompress(raw)
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    fallback = Path(str(path) + ".fallback.jsonl")
    if fallback.exists():
        found = True
        for line in fallback.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    if not found:
        raise FileNotFoundError(path)
    return entries
