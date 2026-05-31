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
import re
import uuid
from dataclasses import dataclass, field
from dataclasses import is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from .protocols import serialize_value

lib_logger = logging.getLogger("rotator_library")

REDACTED = "[REDACTED]"

_SENSITIVE_KEYS = frozenset(
    {
        "api-key",
        "credential-identifier",
        "authorization",
        "proxy-authorization",
        "cookie",
        "set-cookie",
        "x-api-key",
        "x-goog-api-key",
        "openai-api-key",
        "access-token",
        "refresh-token",
        "client-secret",
        "password",
        "secret",
        "token",
    }
)

_SENSITIVE_TEXT_RE = re.compile(
    r"(?im)\b(authorization|proxy-authorization|x-api-key|x-goog-api-key|openai-api-key|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|cookie|set-cookie)\b(['\"]?\s*[:=]\s*['\"]?)([^'\"\r\n,}]+)"
)


def _normalise_key(key: Any) -> str:
    return str(key).strip().lower().replace("_", "-")


def scrub_sensitive_text(value: str) -> str:
    """Scrub obvious credential-bearing header/query fragments from text.

    General trace payloads use key-based redaction to avoid hiding model text.
    Error strings and object reprs are different: providers often embed HTTP
    headers or query strings in exception text, so this targeted scrub only
    applies when callers opt into string scrubbing.
    """

    return _SENSITIVE_TEXT_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", value)


def _object_mapping(value: Any) -> Optional[dict[str, Any]]:
    """Extract structured data from common SDK objects before using repr()."""

    for method_name in ("model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                result = method()
                if isinstance(result, Mapping):
                    return dict(result)
            except Exception:
                pass
    structured: dict[str, Any] = {"type": f"{type(value).__module__}.{type(value).__name__}"}
    found = False
    for attr in ("status_code", "headers", "url", "method", "text", "content"):
        if hasattr(value, attr):
            try:
                structured[attr] = getattr(value, attr)
                found = True
            except Exception:
                pass
    if found:
        return structured
    if hasattr(value, "__dict__") and not is_dataclass(value):
        try:
            return {"type": structured["type"], "attributes": vars(value)}
        except Exception:
            return None
    return None


def sanitize_for_trace(value: Any, *, scrub_strings: bool = False) -> Any:
    """Return a JSON-safe, recursively redacted value for trace files.

    Redaction is intentionally key-based rather than value-based. Model text may
    legitimately mention tokens, passwords, or secrets; hiding by value would make
    debugging transformations unreliable. Sensitive framework fields are redacted
    only when their key name is known to carry credentials.
    """

    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            if _normalise_key(key) in _SENSITIVE_KEYS:
                sanitized[str(key)] = REDACTED
            else:
                sanitized[str(key)] = sanitize_for_trace(item, scrub_strings=scrub_strings)
        return sanitized
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize_for_trace(item, scrub_strings=scrub_strings) for item in value]
    if isinstance(value, str):
        return scrub_sensitive_text(value) if scrub_strings else value

    object_mapping = _object_mapping(value)
    if object_mapping is not None:
        return sanitize_for_trace(object_mapping, scrub_strings=scrub_strings)

    serialized = serialize_value(value)
    if isinstance(serialized, dict):
        return sanitize_for_trace(serialized, scrub_strings=scrub_strings)
    if isinstance(serialized, list):
        return [sanitize_for_trace(item, scrub_strings=scrub_strings) for item in serialized]
    if isinstance(serialized, str):
        return scrub_sensitive_text(serialized) if scrub_strings else serialized
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
    request_id: Optional[str] = None
    timestamp_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    protocol: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    credential_id: Optional[str] = None
    transport: Optional[str] = None
    changed_from_previous: Optional[bool] = None
    session_id: Optional[str] = None
    scope_key: Optional[str] = None
    classifier: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    data: Any = None
    scrub_strings: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "component": self.component,
            "pass_name": self.pass_name,
            "direction": self.direction,
            "stage": self.stage,
            "request_id": self.request_id,
            "timestamp_utc": self.timestamp_utc,
            "protocol": self.protocol,
            "provider": self.provider,
            "model": self.model,
            "credential_id": self.credential_id,
            "transport": self.transport,
            "changed_from_previous": self.changed_from_previous,
            "session_id": self.session_id,
            "scope_key": self.scope_key,
            "classifier": self.classifier,
            "metadata": sanitize_for_trace(self.metadata, scrub_strings=self.scrub_strings),
            "data": sanitize_for_trace(self.data, scrub_strings=self.scrub_strings),
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
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        scope_key: Optional[str] = None,
        classifier: Optional[str] = None,
        snapshot_namespace: Optional[str] = None,
        enabled: bool = True,
    ) -> None:
        self.log_dir = log_dir
        self.component = component
        self.provider = provider
        self.model = model
        self.request_id = request_id
        self.session_id = session_id
        self.scope_key = scope_key
        self.classifier = classifier
        self.snapshot_namespace = snapshot_namespace
        self.enabled = enabled
        self._sequence = 0
        self.trace_file = log_dir / "transform_trace.jsonl"
        self.snapshot_dir = log_dir / "transforms"

    def update_context(
        self,
        *,
        session_id: Optional[str] = None,
        scope_key: Optional[str] = None,
        classifier: Optional[str] = None,
    ) -> None:
        """Update immutable-ish correlation fields discovered after creation."""

        if session_id is not None:
            self.session_id = session_id
        if scope_key is not None:
            self.scope_key = scope_key
        if classifier is not None:
            self.classifier = classifier

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
        session_id: Optional[str] = None,
        scope_key: Optional[str] = None,
        classifier: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        scrub_strings: bool = False,
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
            request_id=self.request_id,
            protocol=protocol,
            provider=self.provider,
            model=self.model,
            credential_id=credential_id,
            transport=transport,
            changed_from_previous=changed_from_previous,
            session_id=session_id if session_id is not None else self.session_id,
            scope_key=scope_key if scope_key is not None else self.scope_key,
            classifier=classifier if classifier is not None else self.classifier,
            metadata=metadata or {},
            data=data,
            scrub_strings=scrub_strings,
        )
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.trace_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            if snapshot and direction != "stream":
                self.snapshot_dir.mkdir(parents=True, exist_ok=True)
                namespace = f"{sanitize_filename(self.snapshot_namespace)}_" if self.snapshot_namespace else ""
                snapshot_name = f"{entry.sequence:04d}_{namespace}{sanitize_filename(pass_name)}.json"
                with open(self.snapshot_dir / snapshot_name, "w", encoding="utf-8") as handle:
                    json.dump(entry.to_dict(), handle, indent=2, ensure_ascii=False)
        except Exception as exc:
            lib_logger.debug("Transform trace write failed for %s: %s", pass_name, exc)
        return entry


def provider_snapshot_namespace() -> str:
    """Return a short namespace that prevents provider snapshot collisions."""

    return f"provider_{uuid.uuid4().hex[:8]}"
