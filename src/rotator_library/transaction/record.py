"""The per-request transaction record accumulator (G10 Phase A).

One :class:`TransactionRecord` per request holds everything that will be
sealed into the on-disk archive: the wire boundaries, the recipe (the
deterministic-reconstruction inputs), the change log (value-level events
for every runtime mutation), metadata, and error evidence.

Memory contract:
    - boundaries are stored as the exact JSON-serializable payloads handed
      over (raw bytes semantics: never re-serialized from parsed forms);
    - a byte budget caps the whole record; overflow sets explicit
      truncation flags in :attr:`truncation` — never silent loss of shape;
    - in ``buffered`` mode nothing touches disk until the record is sealed
      and handed to the writer; in ``incremental`` mode completed sections
      spill to a per-request append file (RAM stays flat for huge streams).

The record is append-only: once sealed it is immutable.
"""

from __future__ import annotations

import json
import os
import threading
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, IO

# Default per-record budget before truncation/spill kicks in (16 MiB —
# matches the G10 design; oversized streams degrade, never balloon).
DEFAULT_RECORD_BUDGET_BYTES = 16 * 1024 * 1024
# Per-boundary soft cap before the boundary itself is truncated.
DEFAULT_BOUNDARY_CAP_BYTES = 8 * 1024 * 1024


@dataclass
class ChangeEvent:
    """One value-level entry in the change log.

    ``kind`` is the closed vocabulary (hook_edit, adapter_edit, cache_injection,
    opaque_strip, repair, routing_substitution, fallback, relay_disengage,
    minted_id, escalation, disclosure, ...). ``value`` carries the mutated
    value / resulting payload fragment — the change log is value-level, not
    event-level: an entry must be enough to replay the mutation offline.
    """

    seq: int
    stage: str
    kind: str
    detail: str = ""
    value: Any = None
    code: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "stage": self.stage,
            "kind": self.kind,
            "code": self.code,
            "detail": self.detail[:400],
            "value": self.value,
            "ts": round(self.timestamp, 6),
        }


# Live unsealed records (weak — the registry never keeps them alive).
# The writer thread sweeps this set: a record older than the orphan
# timeout that never sealed (cancelled request, stuck context) is sealed
# as incomplete so completed work is never silently lost.
_LIVE_RECORDS: "weakref.WeakSet[TransactionRecord]" = weakref.WeakSet()
_ORPHAN_TIMEOUT_SECONDS = 3600.0
# Hard cap on change-log events: a million-frame stream must not mint a
# million dataclass objects. Over the cap, events drop with a flag.
MAX_CHANGE_EVENTS = 20000


def sweep_orphaned_records() -> list["TransactionRecord"]:
    """Seal records idle past the orphan timeout (activity-based: a live
    long stream keeps touching its record and is never reaped)."""

    now = time.time()
    try:
        live = list(_LIVE_RECORDS)
    except RuntimeError:  # WeakSet mutated during iteration — skip cycle
        return []
    orphans = [
        r
        for r in live
        if r.sealed_at is None and now - max(r.created_at, r.last_activity) > _ORPHAN_TIMEOUT_SECONDS
    ]
    for record in orphans:
        record.truncation.setdefault("orphan", "record idle past the orphan timeout; auto-sealed incomplete")
    return orphans


class TransactionRecord:
    """Accumulator for one request's transaction archive."""

    def __init__(
        self,
        *,
        request_id: str,
        protocol: str,
        provider: str,
        model: str,
        profile: Optional[str] = None,
        operation: str = "",
        execution_mode: str = "",
        budget_bytes: int = DEFAULT_RECORD_BUDGET_BYTES,
        boundary_cap_bytes: int = DEFAULT_BOUNDARY_CAP_BYTES,
    ) -> None:
        self.request_id = request_id
        self.protocol = protocol
        self.provider = provider
        self.model = model
        self.profile = profile
        self.operation = operation
        self.execution_mode = execution_mode
        self.budget_bytes = budget_bytes
        self.boundary_cap_bytes = boundary_cap_bytes

        self.created_at = time.time()
        self.last_activity = self.created_at
        self.sealed_at: Optional[float] = None
        # One lock per record: sealing (request thread OR orphan sweeper)
        # and mutation (event loop) never interleave — the envelope is
        # built under the lock and snapshots, never aliases, the lists.
        self._lock = threading.Lock()
        self.status_code: Optional[int] = None

        # The two external inputs are always captured in full (subject to
        # caps); the derived boundaries (provider-bound payload, client
        # egress) become escalation-only once the value-level change log
        # covers their mutations (G10 Phase B).
        self.stream_chunks: list[Any] = []
        self.client_chunks: list[Any] = []
        self.change_log: list[ChangeEvent] = []
        self.metadata: dict[str, Any] = {}
        self.errors: list[dict[str, Any]] = []
        self.attempts: list[dict[str, Any]] = []
        self.routing: dict[str, Any] = {}
        self.escalations: list[str] = []
        self.truncation: dict[str, str] = {}
        self._seq = 0
        self._approx_bytes = 0
        # Incremental mode: sections spill to a per-request append file
        # (RAM stays flat for huge streams); seal() reads it back once.
        self._spill_path: Optional[Path] = None
        self._spill_handle: Optional[IO[str]] = None
        self._dropped_events = 0
        self._boundaries_ram: dict[str, Any] = {}
        self._boundaries_fold: Optional[dict[str, Any]] = None
        self._boundary_order: list[str] = []
        _LIVE_RECORDS.add(self)

    # -- boundaries ------------------------------------------------------

    def set_boundary(self, name: str, payload: Any) -> None:
        """Record one wire boundary (``client_request``, ``provider_request``,
        ``provider_response``, ``client_egress``).

        Payloads are stored as given; a boundary already set is overwritten
        only by a strictly newer capture of the same boundary (last write
        wins — the executor retries reuse one record per request).
        """

        with self._lock:
            if self.sealed_at is not None:
                return
            self.last_activity = time.time()
            payload = _bounded_head(payload, self.boundary_cap_bytes, self.truncation, name)
            approx = _approx_size(payload)
            if name in self.boundary_order:
                # Overwrite: refund the previous size, add the new (retries
                # rewrite provider_request/client_egress; the ledger must
                # track the replacement, not double-count).
                previous = self.boundaries.get(name)
                self._approx_bytes -= min(_approx_size(previous), self.boundary_cap_bytes)
            else:
                self._boundary_order.append(name)
            self._boundaries_ram[name] = payload if self._spill_handle is None else None
            if self._spill_handle is not None:
                self._spill_write("boundary", name, payload)
            self._approx_bytes += approx
            self._check_budget()

    def add_stream_chunk(self, chunk: Any) -> None:
        """Append one provider stream chunk (the glued provider-arrival form)."""

        with self._lock:
            if self.sealed_at is not None:
                return
            self.last_activity = time.time()
            approx = _approx_size(chunk)
            if self._approx_bytes + approx > self.budget_bytes:
                self.truncation["stream_chunks"] = "record budget exceeded; later chunks dropped"
                return
            if self._spill_handle is not None:
                self._spill_write("stream_chunk", None, chunk)
            else:
                self.stream_chunks.append(chunk)
            self._approx_bytes += approx

    def add_client_chunk(self, chunk: Any) -> None:
        """Append one client-egress stream chunk (the glued client-departure form)."""

        with self._lock:
            if self.sealed_at is not None:
                return
            self.last_activity = time.time()
            approx = _approx_size(chunk)
            if self._approx_bytes + approx > self.budget_bytes:
                self.truncation["client_chunks"] = "record budget exceeded; later chunks dropped"
                return
            if self._spill_handle is not None:
                self._spill_write("client_chunk", None, chunk)
            else:
                self.client_chunks.append(chunk)
            self._approx_bytes += approx

    # -- change log ------------------------------------------------------

    def record_change(
        self,
        stage: str,
        kind: str,
        *,
        detail: str = "",
        value: Any = None,
        code: str = "",
    ) -> None:
        """Append one value-level change-log event.

        ``code`` carries the disclosure code for G10 Phase B warning events
        (the closed vocabulary); ``value`` is required to be replayable for
        mutating kinds.
        """

        with self._lock:
            if self.sealed_at is not None:
                return
            self.last_activity = time.time()
            if len(self.change_log) >= MAX_CHANGE_EVENTS:
                # Event-count cap: the counter survives, the event object
                # does not (a million-frame stream must not mint a million
                # events regardless of value sizes).
                self.truncation["change_log_events"] = "change-log event cap reached; later events dropped"
                self._dropped_events += 1
                return
            approx_value = _approx_size(value) + 96
            if self._approx_bytes + approx_value > self.budget_bytes:
                # Budget gate: the EVENT survives (shape), the VALUE is dropped
                # with an explicit flag — bounded RAM, no silent event loss.
                self.truncation["change_log_values"] = "record budget exceeded; later change values dropped"
                value = None
                approx_value = 96
            self._seq += 1
            self.change_log.append(ChangeEvent(self._seq, stage, kind, detail=detail, value=value, code=code))
            self._approx_bytes += approx_value

    def mark_escalation(self, reason: str) -> None:
        """Flag the record as non-derivable in some aspect.

        Escalation is what makes the minimal footprint honest: any request
        touched by a hook edit, cache injection, repair, or other runtime
        decision that offline reconstruction cannot reproduce records the
        affected boundary too (the caller decides which via
        :meth:`set_boundary`; the flag travels in the archive so readers
        know why it is there).
        """

        if self.sealed_at is not None:
            return
        if reason not in self.escalations:
            self.escalations.append(reason)

    # -- metadata --------------------------------------------------------

    def update_metadata(self, **fields: Any) -> None:
        with self._lock:
            if self.sealed_at is not None:
                return
            self.last_activity = time.time()
            self.metadata.update(fields)

    def record_attempt(self, entry: dict[str, Any]) -> None:
        if self.sealed_at is not None:
            return
        self.attempts.append(entry)

    def record_routing(self, record: dict[str, Any]) -> None:
        if self.sealed_at is not None:
            return
        self.routing.update(record)

    def record_error(self, error_type: str, message: str, raw: Any = None) -> None:
        if self.sealed_at is not None:
            return
        self.errors.append({"type": error_type, "message": str(message)[:2000], "raw": raw})

    # -- sealing ---------------------------------------------------------

    def seal(self, status_code: Optional[int] = None) -> dict[str, Any]:
        """Freeze the record into the archive envelope (idempotent)."""

        with self._lock:
            already_sealed = self.sealed_at is not None
            if not already_sealed:
                self.sealed_at = time.time()
                if status_code is not None:
                    self.status_code = status_code
                self._spill_drain_locked()
            truncation = dict(self.truncation)
            if self._dropped_events:
                truncation["change_log_events"] = f"{self._dropped_events} change-log events dropped (cap)"
            envelope = {
                "format": "proxy-transaction/2",
                "sealed_at": self.sealed_at,
                "recipe": {
                    "request_id": self.request_id,
                    "protocol": self.protocol,
                    "provider": self.provider,
                    "profile": self.profile,
                    "model": self.model,
                    "operation": self.operation,
                    "execution_mode": self.execution_mode,
                    "created_at": self.created_at,
                },
                # Snapshots, never aliases: post-seal mutation of the live
                # lists cannot corrupt a returned envelope.
                "boundaries": {name: self.boundaries.get(name) for name in list(self.boundary_order)},
                "stream_chunks": list(self.stream_chunks),
                "client_chunks": list(self.client_chunks),
                "change_log": [event.to_dict() for event in list(self.change_log)],
                "metadata": dict(self.metadata),
                "attempts": list(self.attempts),
                "routing": dict(self.routing),
                "errors": list(self.errors),
                "escalations": list(self.escalations),
                "truncation": truncation,
            }
            if self.status_code is not None:
                envelope["status_code"] = self.status_code
        return envelope

    # -- internals -------------------------------------------------------

    def _check_budget(self) -> None:
        if self._approx_bytes > self.budget_bytes:
            self.truncation.setdefault("record", f"record budget {self.budget_bytes} exceeded")


    # -- incremental spill -------------------------------------------------

    def enable_spill(self, directory: "Path") -> None:
        """Switch to incremental mode: sections append to a per-request
        temp file instead of accumulating in RAM.

        Called lazily — once the record crosses half its budget — so tiny
        requests never pay a file create/unlink. Boundaries already in RAM
        migrate to the spill on next overwrite; chunks spill from now on.
        Peak RAM at seal still includes the fold-back; what stays flat is
        the mid-request footprint for huge streams.
        """

        with self._lock:
            if self._spill_handle is not None or self.sealed_at is not None:
                return
            self._spill_path = directory / f".spill-{self.request_id}.jsonl"
            self._spill_handle = open(self._spill_path, "w", encoding="utf-8")
            for name in list(self._boundary_order):
                payload = self._boundaries_ram.pop(name, None)
                if payload is not None:
                    self._spill_write("boundary", name, payload)
            self.stream_chunks.clear()
            self.client_chunks.clear()

    def _spill_write(self, section: str, name: Optional[str], payload: Any) -> None:
        if self._spill_handle is None:
            return
        try:
            self._spill_handle.write(json.dumps({"s": section, "n": name, "v": payload}, default=str, ensure_ascii=False) + "\n")
        except Exception:
            self.truncation.setdefault("spill", "spill write failed; section lost")

    @property
    def boundaries(self) -> dict[str, Any]:
        """Boundary view: RAM map (no spill) or fold-back map (after drain)."""

        if self._spill_handle is None and self._boundaries_fold is not None:
            return self._boundaries_fold
        if self._spill_handle is None:
            return self._boundaries_ram
        return self._boundaries_ram

    @property
    def boundary_order(self) -> list[str]:
        return self._boundary_order

    def maybe_enable_spill(self, directory: "Path") -> None:
        """Lazy spill switch once the record crosses half its budget."""

        if self._spill_handle is None and self._approx_bytes > self.budget_bytes // 2:
            self.enable_spill(directory)

    def _spill_drain_locked(self) -> None:
        """Close the spill file and fold its sections back for sealing."""

        if self._spill_handle is None:
            return
        try:
            self._spill_handle.close()
        except Exception:
            pass
        self._spill_handle = None
        self._boundaries_fold = {}
        if self._spill_path is None or not self._spill_path.exists():
            self._boundaries_fold = {}
            return
        try:
            with open(self._spill_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    section = entry.get("s")
                    name = entry.get("n")
                    value = entry.get("v")
                    if section == "boundary" and name:
                        self._boundaries_fold[name] = value
                        if name not in self._boundary_order:
                            self._boundary_order.append(name)
                    elif section == "stream_chunk":
                        self.stream_chunks.append(value)
                    elif section == "client_chunk":
                        self.client_chunks.append(value)
        except Exception:
            self.truncation.setdefault("spill", "spill read-back failed")
        finally:
            try:
                self._spill_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._spill_path = None


def _bounded_head(payload: Any, cap_bytes: int, truncation: dict[str, str], name: str) -> Any:
    """Bound ANY payload type by serialized head, with an explicit flag.

    JSON is serialized once and sliced; the retained head carries a marker
    so readers know shape was lost — the previous dict-branch kept the
    whole payload while claiming truncation, which was worse than no cap.
    """

    approx = _approx_size(payload)
    if approx <= cap_bytes:
        return payload
    truncation[name] = f"boundary exceeded {cap_bytes} bytes; serialized head kept only"
    try:
        text = json.dumps(payload, default=str, ensure_ascii=False)
    except Exception:
        return {"__truncated__": True, "reason": "unserializable payload"}
    return {
        "__truncated__": True,
        "original_bytes": approx,
        "head": text[:cap_bytes],
    }


def _approx_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (str, bytes)):
        return len(value)
    if isinstance(value, (dict, list)):
        try:
            import json

            return len(json.dumps(value, default=str))
        except Exception:
            return 4096
    return 64


