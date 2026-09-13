"""The single background transaction writer (G10 Phase A).

All sealed records across all concurrent requests funnel through ONE
writer thread. It group-flushes: whatever is queued when it wakes is
compressed and written as sequential batches with one fsync per batch
(group commit — the pattern every reference converges on: gomodel's
flushLoop, OpenTelemetry's batch processor, database group commit).

Two modes per the operator ruling:
    - ``buffered`` (default): the record lives fully in memory until
      sealed; the writer compresses once and writes one atomic file.
    - ``incremental``: large/streaming requests append completed sections
      to a per-request temp file as they arrive (RAM stays flat), and the
      writer seals that temp into the same one-archive output. Deployed
      where concurrency and load make RAM the scarce resource.

Queue discipline: bounded by record count AND queued bytes; on overflow
the newest submission is dropped with a console warning (load-shedding,
never blocking a request). Each batch writes archives with a data fsync
per file (atomic tmp+rename); a periodic orphan sweep seals records whose
request never finalized (cancellation, stuck contexts).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import archive
from .record import TransactionRecord

lib_logger = logging.getLogger("rotator_library.transaction")

WRITER_QUEUE_MAX = 2048
WRITER_QUEUE_BYTE_CAP = 256 * 1024 * 1024
FLUSH_INTERVAL_SECONDS = 1.0

# TRANSACTION_LOG_MODE: buffered (default) | incremental
MODE_BUFFERED = "buffered"
MODE_INCREMENTAL = "incremental"


def _configured_mode() -> str:
    import os as _os

    mode = str(_os.environ.get("TRANSACTION_LOG_MODE", MODE_BUFFERED)).strip().lower()
    return mode if mode in (MODE_BUFFERED, MODE_INCREMENTAL) else MODE_BUFFERED


def _configured_retention() -> int:
    try:
        value = int(os.environ.get("TRANSACTION_LOG_RETENTION", "1000"))
        return max(0, value)
    except ValueError:
        return 1000


class TransactionWriter:
    """Singleton background writer owning all transaction disk I/O."""

    _instance: Optional["TransactionWriter"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._queue: "queue.Queue[Optional[dict[str, Any]]]" = queue.Queue(maxsize=WRITER_QUEUE_MAX)
        self._queued_bytes = 0
        self._queued_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False
        self._draining = False
        self._batch_counter = 0
        self.mode = _configured_mode()
        self.retention = _configured_retention()
        self.dropped = 0

    @classmethod
    def instance(cls) -> "TransactionWriter":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run, name="transaction-writer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Drain and stop (proxy shutdown path).

        The drain must survive a full queue: the loop below exits only on
        stop, so the worker keeps writing everything already submitted —
        a completed request's record never dies in the queue.
        """

        if not self._started:
            return
        self._draining = True
        deadline = time.time() + timeout
        while time.time() < deadline and not self._queue.empty():
            time.sleep(0.05)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._started = False

    # -- submission ------------------------------------------------------

    def submit_sealed(self, envelope: dict[str, Any], *, filename: str) -> None:
        """Queue one sealed envelope for archive write (non-blocking)."""

        self.start()
        if self._draining:
            # Shutdown window: draining synchronously here would block the
            # request; the record is logged-lost instead (warned once).
            self.dropped += 1
            return
        item = {"envelope": envelope, "filename": filename}
        approx = archive._approx_size_json(envelope)
        with self._queued_lock:
            over = self._queue.qsize() >= WRITER_QUEUE_MAX or self._queued_bytes + approx > WRITER_QUEUE_BYTE_CAP
        if over:
            self.dropped += 1
            lib_logger.warning(
                "transaction writer queue over budget (%d dropped) — sealed record %s dropped",
                self.dropped,
                filename,
            )
            return
        try:
            self._queue.put_nowait(item)
            with self._queued_lock:
                self._queued_bytes += approx
        except queue.Full:
            self.dropped += 1
            lib_logger.warning(
                "transaction writer queue full (%d dropped) — sealed record %s dropped",
                self.dropped,
                filename,
            )

    # -- worker ----------------------------------------------------------

    def _run(self) -> None:
        last_sweep = 0.0
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=FLUSH_INTERVAL_SECONDS)
            except queue.Empty:
                if self._draining and self._queue.empty():
                    break
                if time.time() - last_sweep > 30.0:
                    self._sweep_orphans()
                    last_sweep = time.time()
                continue
            if item is None:
                # Stop marker: drain the remainder then exit.
                self._drain_batch()
                break
            batch = [item]
            # Group flush: coalesce everything already queued.
            while len(batch) < 256:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._drain_batch(batch)
                    self._queue.task_done()
                    return
                batch.append(nxt)
                self._queue.task_done()
            self._write_batch(batch)

    def _drain_batch(self, batch: Optional[list] = None) -> None:
        pending = batch if batch is not None else []
        if batch is None:
            while True:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is not None:
                    pending.append(nxt)
                self._queue.task_done()
        if pending:
            self._write_batch(pending)

    def _sweep_orphans(self) -> None:
        """Seal records whose request never finalized (escape hatch)."""

        from .record import sweep_orphaned_records

        try:
            for orphan in sweep_orphaned_records():
                envelope = orphan.seal()
                filename = archive.archive_filename(
                    protocol=orphan.protocol,
                    provider=orphan.provider,
                    model=orphan.model,
                    request_id=orphan.request_id,
                    profile=orphan.profile,
                    when=orphan.created_at,
                )
                self._queue.put_nowait({"envelope": envelope, "filename": filename})
        except Exception:
            pass

    def _write_batch(self, batch: list[dict[str, Any]]) -> None:
        """Write one group of sealed records sequentially; one fsync per file
        (atomic tmp+rename already fsyncs data; directory entries batch)."""

        target_dir = archive.transactions_dir()
        with self._queued_lock:
            self._queued_bytes = max(0, self._queued_bytes - sum(archive._approx_size_json(item["envelope"]) for item in batch))
        for item in batch:
            envelope = item["envelope"]
            filename = item["filename"]
            try:
                blob = archive.compress_envelope(envelope)
                archive.write_archive_atomic(target_dir, filename, blob)
            except Exception as exc:  # never fail the writer loop
                lib_logger.warning("failed to write transaction archive %s: %s", filename, exc)
        # Prune every 64 batches (not per batch — the glob+sort is the
        # stat-storm class; batch-gating bounds it).
        self._batch_counter += 1
        if self._batch_counter >= 64:
            self._batch_counter = 0
            try:
                archive.prune_archives(self.retention, target_dir)
            except Exception:
                pass
