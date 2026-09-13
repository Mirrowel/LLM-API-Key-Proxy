# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Explorer tests: list/row rendering and offline reconstruction."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _seal(tmp_path, **kwargs) -> Path:
    from rotator_library.transaction import TransactionRecord
    from rotator_library.transaction.archive import (
        archive_filename,
        compress_envelope,
        write_archive_atomic,
    )

    record = TransactionRecord(
        request_id=kwargs.get("request_id", "t1"),
        protocol=kwargs.get("protocol", "openai_chat"),
        provider=kwargs.get("provider", "groq"),
        model=kwargs.get("model", "llama-3"),
    )
    record.set_boundary(
        "client_request",
        kwargs.get(
            "client_request",
            {"model": "llama-3", "messages": [{"role": "user", "content": "hi"}]},
        ),
    )
    for event in kwargs.get("changes", []):
        record.record_change(**event)
    envelope = record.seal(status_code=kwargs.get("status_code", 200))
    name = archive_filename(
        protocol=record.protocol,
        provider=record.provider,
        model=record.model,
        request_id=record.request_id,
    )
    return write_archive_atomic(tmp_path, name, compress_envelope(envelope))


def test_load_archives_reads_and_orders(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSACTION_LOG_DIR", str(tmp_path))
    from proxy_app.transaction_explorer import _load_archives, _row

    _seal(tmp_path, request_id="older")
    _seal(tmp_path, request_id="newer")

    entries = _load_archives()
    assert len(entries) == 2
    ids = {entry[1]["recipe"]["request_id"] for entry in entries}
    assert ids == {"older", "newer"}
    row = _row(1, *entries[0])
    assert "openai_chat" in row and "groq" in row and "200" in row


def test_row_flags_errors_and_escalations(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSACTION_LOG_DIR", str(tmp_path))
    from proxy_app.transaction_explorer import _row
    from rotator_library.transaction import TransactionRecord

    record = TransactionRecord(request_id="r", protocol="p", provider="pr", model="m")
    record.mark_escalation("hook_edit")
    record.record_error("bad", "boom")
    row = _row(1, Path("20260101_000000_p_pr_m_r.transaction.zst"), record.seal(500))
    assert "errors" in row and "escalated" in row and "500" in row


def test_reconstruct_transaction_replays_neutral_parse(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSACTION_LOG_DIR", str(tmp_path))
    from proxy_app.transaction_explorer import _load_archives, reconstruct_transaction
    from rotator_library.transaction.archive import read_archive

    _seal(
        tmp_path,
        request_id="t1",
        changes=[
            {"stage": "mutated", "kind": "hook_edit", "detail": "tools stripped", "value": {"removed": 2}},
            {"stage": "stream", "kind": "repair", "detail": "synthesized finish reason"},
        ],
    )
    entries = _load_archives()
    artifacts = reconstruct_transaction(entries[0][1])
    assert artifacts["neutral_parse"]["model"] == "llama-3"
    assert artifacts["provider_target"] == "groq/llama-3"
    kinds = {note["kind"] for note in artifacts["replay_notes"]}
    assert kinds == {"hook_edit", "repair"}
    # Non-derivables are surfaced honestly.
    assert any("provider response" in item for item in artifacts["unreconstructable"])


def test_reconstruct_transaction_without_client_request(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSACTION_LOG_DIR", str(tmp_path))
    from proxy_app.transaction_explorer import reconstruct_transaction
    from rotator_library.transaction import TransactionRecord
    from rotator_library.transaction.archive import (
        archive_filename,
        compress_envelope,
        write_archive_atomic,
    )

    record = TransactionRecord(request_id="empty", protocol="openai_chat", provider="x", model="y")
    envelope = record.seal(200)
    name = archive_filename(protocol="openai_chat", provider="x", model="y", request_id="empty")
    write_archive_atomic(tmp_path, name, compress_envelope(envelope))

    artifacts = reconstruct_transaction(envelope)
    assert "error" in artifacts
    assert "nothing to replay" in artifacts["error"]


def test_unreadable_archive_is_listed_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSACTION_LOG_DIR", str(tmp_path))
    from proxy_app.transaction_explorer import _load_archives

    (tmp_path / "20260101_000000_x_x_x_broken.transaction.zst").write_bytes(b"garbage")
    entries = _load_archives()
    assert len(entries) == 1
    assert entries[0][1].get("format") == "unreadable"


def test_module_level_is_stdlib_only():
    import ast
    import inspect

    import proxy_app.transaction_explorer as module

    tree = ast.parse(inspect.getsource(module))
    for node in tree.body:  # top-level statements only (docstring included)
        if isinstance(node, ast.Import):
            names = {alias.name for alias in node.names}
            assert not any(n.startswith("rotator_library") or n.startswith("proxy_app") for n in names)
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith(("rotator_library", "proxy_app"))
