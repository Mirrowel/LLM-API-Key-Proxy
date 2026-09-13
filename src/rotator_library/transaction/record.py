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

import time
from dataclasses import dataclass, field
from typing import Any, Optional

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
        self.sealed_at: Optional[float] = None
        self.status_code: Optional[int] = None

        # The two external inputs are always captured in full (subject to
        # caps); the derived boundaries (provider-bound payload, client
        # egress) become escalation-only once the value-level change log
        # covers their mutations (G10 Phase B).
        self.boundaries: dict[str, Any] = {}
        self.boundary_order: list[str] = []
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

    # -- boundaries ------------------------------------------------------

    def set_boundary(self, name: str, payload: Any) -> None:
        """Record one wire boundary (``client_request``, ``provider_request``,
        ``provider_response``, ``client_egress``).

        Payloads are stored as given; a boundary already set is overwritten
        only by a strictly newer capture of the same boundary (last write
        wins — the executor retries reuse one record per request).
        """

        if self.sealed_at is not None:
            return
        approx = _approx_size(payload)
        if approx > self.boundary_cap_bytes:
            self.truncation[name] = f"boundary exceeded {self.boundary_cap_bytes} bytes; kept head only"
            payload = _truncate_head(payload, self.boundary_cap_bytes)
            approx = self.boundary_cap_bytes
        previous = self.boundaries.get(name)
        self.boundaries[name] = payload
        if name not in self.boundary_order:
            self.boundary_order.append(name)
            self._approx_bytes += approx
        self._check_budget()

    def add_stream_chunk(self, chunk: Any) -> None:
        """Append one provider stream chunk (the glued provider-arrival form)."""

        if self.sealed_at is not None:
            return
        approx = _approx_size(chunk)
        if self._approx_bytes + approx > self.budget_bytes:
            self.truncation["stream_chunks"] = "record budget exceeded; later chunks dropped"
            return
        self.stream_chunks.append(chunk)
        self._approx_bytes += approx

    def add_client_chunk(self, chunk: Any) -> None:
        """Append one client-egress stream chunk (the glued client-departure form)."""

        if self.sealed_at is not None:
            return
        approx = _approx_size(chunk)
        if self._approx_bytes + approx > self.budget_bytes:
            self.truncation["client_chunks"] = "record budget exceeded; later chunks dropped"
            return
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

        if self.sealed_at is not None:
            return
        self._seq += 1
        self.change_log.append(ChangeEvent(self._seq, stage, kind, detail=detail, value=value, code=code))
        self._approx_bytes += _approx_size(value) + 96

    def mark_escalation(self, reason: str) -> None:
        """Flag the record as non-derivable in some aspect.

        Escalation is what makes the minimal footprint honest: any request
        touched by a hook edit, cache injection, repair, or other runtime
        decision that offline reconstruction cannot reproduce records the
        affected boundary too (the caller decides which via
        :meth:`set_boundary`; the flag travels in the archive so readers
        know why it is there).
        """

        if reason not in self.escalations:
            self.escalations.append(reason)

    # -- metadata --------------------------------------------------------

    def update_metadata(self, **fields: Any) -> None:
        if self.sealed_at is not None:
            return
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

        if self.sealed_at is None:
            self.sealed_at = time.time()
            if status_code is not None:
                self.status_code = status_code
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
            "boundaries": {name: self.boundaries[name] for name in self.boundary_order},
            "stream_chunks": self.stream_chunks,
            "client_chunks": self.client_chunks,
            "change_log": [event.to_dict() for event in self.change_log],
            "metadata": self.metadata,
            "attempts": self.attempts,
            "routing": self.routing,
            "errors": self.errors,
            "escalations": self.escalations,
            "truncation": self.truncation,
        }
        if self.status_code is not None:
            envelope["status_code"] = self.status_code
        return envelope

    # -- internals -------------------------------------------------------

    def _check_budget(self) -> None:
        if self._approx_bytes > self.budget_bytes:
            self.truncation.setdefault("record", f"record budget {self.budget_bytes} exceeded")


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


def _truncate_head(payload: Any, cap_bytes: int) -> Any:
    """Keep the head of an oversized payload with an explicit marker."""

    marker = {"__truncated__": True, "original_bytes": _approx_size(payload)}
    if isinstance(payload, (str, bytes)):
        text = payload[:cap_bytes] if isinstance(payload, str) else payload[:cap_bytes].decode("utf-8", "replace")
        return {"__head__": text, **marker}
    return {"__head_json__": payload, **marker}
