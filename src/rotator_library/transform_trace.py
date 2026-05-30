# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Transform-pass trace primitives for transaction logging.

The trace layer is observability-only: failures here must never change request
execution. Transaction logging uses this module to snapshot each meaningful
request, response, and stream state as later protocol/adapters mutate payloads.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from .protocols import serialize_value

lib_logger = logging.getLogger("rotator_library")

REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = frozenset(
    {
        "api-key",
        "credential-identifier",
        "authorization",
        "x-api-key",
        "x-goog-api-key",
        "access-token",
        "refresh-token",
        "client-secret",
        "password",
        "secret",
        "token",
    }
)


def _normalise_key(key: Any) -> str:
    return str(key).strip().lower().replace("_", "-")


def sanitize_for_trace(value: Any) -> Any:
    """Return a JSON-safe, recursively redacted value for trace files.

    Redaction is intentionally key-based rather than value-based. Model text may
    legitimately mention tokens, passwords, or secrets; hiding by value would make
    debugging transformations unreliable. Sensitive framework fields are redacted
    only when their key name is known to carry credentials.
    """

    serialized = serialize_value(value)
    if isinstance(serialized, dict):
        sanitized = {}
        for key, item in serialized.items():
            if _normalise_key(key) in _SENSITIVE_KEYS:
                sanitized[str(key)] = REDACTED
            else:
                sanitized[str(key)] = sanitize_for_trace(item)
        return sanitized
    if isinstance(serialized, list):
        return [sanitize_for_trace(item) for item in serialized]
    return serialized


def sanitize_filename(value: str) -> str:
    """Return a stable, filesystem-safe name component for snapshot files."""

    safe = value.strip().lower().replace(" ", "_") or "trace"
    for char in '/\\:*?"<>|':
        safe = safe.replace(char, "_")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in safe)


@dataclass
class TransformTraceEntry:
    """A single transform-pass observation.

    Entries are deliberately protocol-neutral. Later phases can add protocol,
    adapter, field-cache, routing, and transport passes without changing the file
    format used by this phase.
    """

    sequence: int
    component: str
    pass_name: str
    direction: str
    stage: str
    timestamp_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    protocol: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    credential_id: Optional[str] = None
    transport: Optional[str] = None
    changed_from_previous: Optional[bool] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    data: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "component": self.component,
            "pass_name": self.pass_name,
            "direction": self.direction,
            "stage": self.stage,
            "timestamp_utc": self.timestamp_utc,
            "protocol": self.protocol,
            "provider": self.provider,
            "model": self.model,
            "credential_id": self.credential_id,
            "transport": self.transport,
            "changed_from_previous": self.changed_from_previous,
            "metadata": sanitize_for_trace(self.metadata),
            "data": sanitize_for_trace(self.data),
        }


class TransformTraceWriter:
    """Append-only writer for transform trace entries and snapshots.

    One writer owns one local sequence counter. Phase 2 intentionally does not
    promise a global ordering across client and provider writers; entries include
    component and timestamps so interleaved logs remain understandable.
    """

    def __init__(
        self,
        log_dir: Path,
        *,
        component: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.log_dir = log_dir
        self.component = component
        self.provider = provider
        self.model = model
        self.enabled = enabled
        self._sequence = 0
        self.trace_file = log_dir / "transform_trace.jsonl"
        self.snapshot_dir = log_dir / "transforms"

    def record(
        self,
        pass_name: str,
        data: Any,
        *,
        direction: str,
        stage: str,
        protocol: Optional[str] = None,
        credential_id: Optional[str] = None,
        transport: Optional[str] = None,
        changed_from_previous: Optional[bool] = None,
        metadata: Optional[dict[str, Any]] = None,
        snapshot: bool = True,
    ) -> Optional[TransformTraceEntry]:
        """Record a transform pass, swallowing logging failures."""

        if not self.enabled:
            return None
        self._sequence += 1
        entry = TransformTraceEntry(
            sequence=self._sequence,
            component=self.component,
            pass_name=pass_name,
            direction=direction,
            stage=stage,
            protocol=protocol,
            provider=self.provider,
            model=self.model,
            credential_id=credential_id,
            transport=transport,
            changed_from_previous=changed_from_previous,
            metadata=metadata or {},
            data=data,
        )
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.trace_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            if snapshot and direction != "stream":
                self.snapshot_dir.mkdir(parents=True, exist_ok=True)
                snapshot_name = f"{entry.sequence:04d}_{sanitize_filename(pass_name)}.json"
                with open(self.snapshot_dir / snapshot_name, "w", encoding="utf-8") as handle:
                    json.dump(entry.to_dict(), handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            lib_logger.debug("Transform trace write failed for %s: %s", pass_name, exc)
        return entry
