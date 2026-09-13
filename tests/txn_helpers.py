"""Helpers for re-pinned G10 transaction-record tests.

The old directory layout (request.json / transform_trace.jsonl / provider/)
is retired. These helpers read the in-memory ``TransactionRecord`` /
``sealed_envelope`` instead, exposing the same conceptual fields the old
tests asserted (a pass name, its payload) plus the new value-level
change-log shape.
"""

from __future__ import annotations

import json
from typing import Any


def changes(logger: Any) -> list[dict[str, Any]]:
    """Return the change log as plain dicts (newest shape, old keys)."""

    record = getattr(logger, "_record", None)
    events = list(getattr(record, "change_log", []) or [])
    return [
        {
            "pass_name": event.code,
            "code": event.code,
            "stage": event.stage,
            "kind": event.kind,
            "detail": event.detail,
            "data": event.value,
        }
        for event in events
    ]


def pass_names(logger: Any) -> list[str]:
    return [entry["pass_name"] for entry in changes(logger)]


def by_pass(logger: Any, name: str) -> list[dict[str, Any]]:
    return [entry for entry in changes(logger) if entry["pass_name"] == name]


def change_text(logger: Any) -> str:
    return "\n".join(
        json.dumps(entry, ensure_ascii=False, default=str) for entry in changes(logger)
    )


def boundaries(logger: Any) -> dict[str, Any]:
    record = getattr(logger, "_record", None)
    return dict(getattr(record, "boundaries", {}) or {})


def client_chunks(logger: Any) -> list[Any]:
    record = getattr(logger, "_record", None)
    return list(getattr(record, "client_chunks", []) or [])


def stream_chunks(logger: Any) -> list[Any]:
    record = getattr(logger, "_record", None)
    return list(getattr(record, "stream_chunks", []) or [])


def record_errors(logger: Any) -> list[dict[str, Any]]:
    """Errors accumulated on the record (type/message/raw)."""

    record = getattr(logger, "_record", None)
    return list(getattr(record, "errors", []) or [])


def error_records(logger: Any) -> list[dict[str, Any]]:
    """Facade error records (failed_pass_name/error_type/message/stage)."""

    return list(getattr(logger, "_error_records", []) or [])


def sealed(logger: Any) -> dict[str, Any]:
    return getattr(logger, "sealed_envelope", None) or {}


def sealed_errors(logger: Any) -> list[dict[str, Any]]:
    envelope = sealed(logger)
    metadata = envelope.get("metadata") or {}
    return list(metadata.get("errors") or [])


def change_text_with_errors(logger: Any) -> str:
    """Full record text: change log + boundary + accumulated error messages."""

    payloads: list[Any] = [changes(logger), boundaries(logger), record_errors(logger), error_records(logger)]
    return json.dumps(payloads, ensure_ascii=False, default=str)
