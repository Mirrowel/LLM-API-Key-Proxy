# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G10 transaction-record pins: one envelope per request.

The directory-layout era (request.json / transform_trace.jsonl / provider/,
capture/) is retired. A request accumulates one in-memory ``TransactionRecord``
and seals exactly one envelope submitted to the single background
``TransactionWriter``. This file pins naming, sealing, capture escalation,
budgets, archive roundtrip/retention, JSON safety, aggregation, and the
buffered/incremental mode flag.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest

from rotator_library.transaction import TransactionRecord, TransactionWriter, archive
from rotator_library.transaction_logger import (
    TransactionLogger,
    _make_json_safe,
)


# ---------------------------------------------------------------------------
# (a) archive naming
# ---------------------------------------------------------------------------


def test_archive_filename_order_and_sanitization() -> None:
    when = 1_700_000_000
    stamp = time.strftime("%m%d_%H%M%S", time.localtime(when))
    name = archive.archive_filename(
        protocol="openai_chat",
        provider="openai",
        model="gpt-4",
        request_id="abc123",
        when=when,
    )
    assert name == f"{stamp}_openai_chat_openai_gpt-4_abc123.transaction.zst"

    with_profile = archive.archive_filename(
        protocol="openai_chat",
        provider="openai",
        model="gpt-4",
        request_id="abc123",
        profile="fast lane",
        when=when,
    )
    assert with_profile == f"{stamp}_openai_chat_fast-lane_openai_gpt-4_abc123.transaction.zst"

    # Unsafe characters collapse to dashes.
    assert archive.sanitize_component("open/ai chat") == "open-ai-chat"


def test_archive_filename_windows_reserved_and_overlong() -> None:
    assert archive.sanitize_component("CON") == "x"
    assert archive.sanitize_component("con", "proto") == "proto"
    reserved = archive.archive_filename(
        protocol="CON",
        provider="openai",
        model="m",
        request_id="r",
        when=1_700_000_000,
    )
    assert "_proto_" in reserved

    untruncated = (
        "_".join(["0101_000000", "openai_chat", "p" * 60, "m" * 60, "r" * 60])
        + archive.ARCHIVE_SUFFIX
    )
    long_name = archive.archive_filename(
        protocol="openai_chat",
        provider="p" * 60,
        model="m" * 60,
        request_id="r" * 60,
        when=1_700_000_000,
    )
    # Overlong names are truncated (a hash suffix replaces the tail) and keep
    # the archive suffix.
    assert long_name != untruncated
    assert len(long_name) < len(untruncated)
    assert long_name.endswith(archive.ARCHIVE_SUFFIX)


# ---------------------------------------------------------------------------
# (b) one archive per request
# ---------------------------------------------------------------------------


def test_one_sealed_envelope_per_request(monkeypatch, tmp_path) -> None:
    captured: list[tuple[dict, str]] = []
    writer = TransactionWriter.instance()
    monkeypatch.setattr(
        writer,
        "submit_sealed",
        lambda envelope, *, filename: captured.append((envelope, filename)),
    )

    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path, protocol="openai_chat")
    logger.log_request({"model": "gpt-test", "stream": False, "messages": [{"role": "user", "content": "hi"}]})
    logger.log_transformed_request(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high"},
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
    )
    logger.log_response({"model": "gpt-test", "choices": [], "usage": {"total_tokens": 2}})

    assert len(captured) == 1
    envelope, filename = captured[0]
    assert envelope["format"] == "proxy-transaction/2"
    assert envelope["recipe"]["request_id"] == logger.request_id
    assert envelope["recipe"]["protocol"] == "openai_chat"
    assert envelope["recipe"]["provider"] == "openai"
    assert list(envelope["boundaries"].keys()) == [
        "client_request",
        "provider_request",
        "client_egress",
    ]
    assert envelope["status_code"] == 200
    assert filename.endswith(archive.ARCHIVE_SUFFIX)
    # Only one seal: a second finalize must not submit again.
    logger.finalize_metadata(status_code=200)
    assert len(captured) == 1


# ---------------------------------------------------------------------------
# (c) error capture escalation
# ---------------------------------------------------------------------------


def _logger(tmp_path) -> TransactionLogger:
    return TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)


def test_flush_capture_escalates_for_request_errors(tmp_path) -> None:
    logger = _logger(tmp_path)

    class BadRequest(Exception):
        error_type = "invalid_request"

    assert logger.flush_capture_on_error(BadRequest("bad payload")) is True
    assert "error_capture" in logger._record.escalations
    # Drained: a second flush does not duplicate the escalation.
    assert logger.flush_capture_on_error(BadRequest("again")) is False


@pytest.mark.parametrize(
    "error_type",
    ["rate_limit", "authentication_error", "timeout_error", "api_connection"],
)
def test_flush_capture_never_escalates_for_rotation_errors(tmp_path, error_type) -> None:
    logger = _logger(tmp_path)

    class RotationError(Exception):
        pass

    error = RotationError("rotate me")
    error.error_type = error_type
    assert logger.flush_capture_on_error(error) is False
    assert "error_capture" not in logger._record.escalations


# ---------------------------------------------------------------------------
# (d) record budget and boundary cap
# ---------------------------------------------------------------------------


def test_stream_chunk_budget_sets_truncation_and_stops_growth() -> None:
    record = TransactionRecord(request_id="r", protocol="p", provider="pr", model="m", budget_bytes=20)
    record.add_stream_chunk("0123456789")  # 10 bytes: fits
    record.add_stream_chunk("0123456789ABCDEF")  # 16 bytes: over budget
    record.add_stream_chunk("0123456789ABCDEF")  # still over

    assert len(record.stream_chunks) == 1
    assert "stream_chunks" in record.truncation
    assert "budget" in record.truncation["stream_chunks"]


def test_boundary_cap_truncates_with_head_marker() -> None:
    record = TransactionRecord(request_id="r", protocol="p", provider="pr", model="m", boundary_cap_bytes=10)
    record.set_boundary("client_request", "x" * 100)

    payload = record.boundaries["client_request"]
    assert payload["__truncated__"] is True
    assert payload["original_bytes"] == 100
    assert payload["__head__"] == "x" * 10
    assert "client_request" in record.truncation


# ---------------------------------------------------------------------------
# (e) archive roundtrip
# ---------------------------------------------------------------------------


def test_archive_roundtrip(tmp_path: Path) -> None:
    envelope = {
        "format": "proxy-transaction/2",
        "recipe": {"request_id": "r1", "protocol": "openai_chat"},
        "boundaries": {"client_request": {"model": "m"}},
        "change_log": [{"seq": 1, "code": "x"}],
        "metadata": {"duration_ms": 1.5},
    }

    blob = archive.compress_envelope(envelope)
    assert archive.decompress_archive(blob) == envelope

    path = archive.write_archive_atomic(tmp_path, "one.transaction.zst", blob)
    assert path.exists()
    assert archive.read_archive(path) == envelope
    # No leftover temp file.
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# (f) retention
# ---------------------------------------------------------------------------


def test_prune_archives_keeps_newest_by_name(tmp_path: Path) -> None:
    names = [f"2024010{i}_000000_p_pr_m_r.transaction.zst" for i in range(1, 6)]
    for name in names:
        (tmp_path / name).write_bytes(b"x")

    removed = archive.prune_archives(3, tmp_path)

    assert removed == 2
    remaining = sorted(path.name for path in archive.iter_archives(tmp_path))
    assert remaining == names[2:]


# ---------------------------------------------------------------------------
# (g) JSON safety
# ---------------------------------------------------------------------------


@dataclass
class _Nested:
    count: int
    path: Path


@dataclass
class _Node:
    name: str
    child: object | None = None


def test_make_json_safe_handles_dataclasses_paths_datetimes_and_cycles(tmp_path) -> None:
    assert _make_json_safe(_Nested(2, Path("a/b.json"))) == {"count": 2, "path": str(Path("a/b.json"))}
    assert _make_json_safe(datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02T03:04:05"
    node = _Node("root")
    node.child = node
    assert _make_json_safe(node) == {"name": "root", "child": "<circular>"}


def test_log_request_and_response_are_json_safe(tmp_path) -> None:
    logger = TransactionLogger("openai", "gpt-test", parent_dir=tmp_path)
    node = _Node("root")
    node.child = node

    logger.log_request({"model": "gpt-test", "api_key": "secret", "messages": []})
    logger.log_response(
        {
            "model": "gpt-test",
            "created_at": datetime(2026, 1, 2, 3, 4, 5),
            "path": Path("a/b.json"),
            "node": node,
        }
    )

    request = logger._record.boundaries["client_request"]
    assert request["api_key"] == "[REDACTED]"
    response = logger._record.boundaries["client_egress"]
    assert response["created_at"] == "2026-01-02T03:04:05"
    assert response["path"] == str(Path("a/b.json"))
    assert response["node"] == {"name": "root", "child": "<circular>"}
    # The sealed envelope is serializable end to end.
    json.dumps(logger.sealed_envelope, default=str)


# ---------------------------------------------------------------------------
# (h) assemble_streaming_response unchanged
# ---------------------------------------------------------------------------


def test_assemble_streaming_response_concatenates_and_keeps_usage() -> None:
    chunks = [
        {"id": "c1", "created": 1, "model": "gpt-test", "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
    ]

    assembled = TransactionLogger.assemble_streaming_response(chunks)

    assert assembled["object"] == "chat.completion"
    assert assembled["choices"][0]["message"]["content"] == "Hello"
    assert assembled["choices"][0]["finish_reason"] == "stop"
    assert assembled["usage"] == {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    assert TransactionLogger.assemble_streaming_response([]) == {}


# ---------------------------------------------------------------------------
# (i) incremental vs buffered mode flag
# ---------------------------------------------------------------------------


def test_writer_mode_reads_env(monkeypatch) -> None:
    monkeypatch.delenv("TRANSACTION_LOG_MODE", raising=False)
    assert TransactionWriter().mode == "buffered"

    monkeypatch.setenv("TRANSACTION_LOG_MODE", "incremental")
    assert TransactionWriter().mode == "incremental"

    monkeypatch.setenv("TRANSACTION_LOG_MODE", "bogus")
    assert TransactionWriter().mode == "buffered"
