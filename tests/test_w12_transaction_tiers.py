# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""W12 acceptance fixtures: leveled transaction logging (D15).

L1 = the four boundaries + metadata + buffered stream chunks, zstd when
available. L2 = + intermediates (trace JSONL + snapshots). L3 = verbose.
Capture-on-error archives the buffered intermediates for request-relevant
failures even at L1; rotation-class failures never trigger it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rotator_library.transaction_logger import TransactionLogger
from rotator_library.transform_trace import REDACTED
from rotator_library.utils import zstd_io


def _artifact(log_dir: Path, name: str) -> Path:
    compressed = log_dir / (name + ".zst")
    return compressed if compressed.exists() else log_dir / name


def _boundary_flow(logger: TransactionLogger) -> None:
    logger.log_request({"model": "gpt-test", "api_key": "sk-secret", "messages": [{"role": "user", "content": "hi"}]})
    logger.log_stream_chunk({"choices": [{"delta": {"content": "ok"}}]})
    logger.log_response({"model": "gpt-test", "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


def test_l1_default_writes_boundaries_metadata_and_no_intermediates(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TRANSACTION_LOG_LEVEL", raising=False)
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    assert logger.trace_level == 1

    _boundary_flow(logger)

    assert _artifact(logger.log_dir, "request.json").exists()
    assert _artifact(logger.log_dir, "response.json").exists()
    assert _artifact(logger.log_dir, "streaming_chunks.jsonl").exists()
    assert not (logger.log_dir / "transform_trace.jsonl").exists()
    assert not (logger.log_dir / "transform_trace.jsonl.zst").exists()
    assert not (logger.log_dir / "transforms").exists()

    metadata = zstd_io.read_json_any(logger.log_dir / "metadata.json")
    assert metadata["schema"] == "2"
    assert metadata["trace_level"] == 1
    assert metadata["compressed"] is zstd_io.compression_available()
    assert metadata["reconstruct"]["script"] == "tools/reconstruct_traces.py"
    assert metadata["usage"]["total_tokens"] == 2


def test_l1_boundary_redaction_never_writes_secrets(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TRANSACTION_LOG_LEVEL", raising=False)
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    _boundary_flow(logger)

    request_payload = zstd_io.read_json_any(_artifact(logger.log_dir, "request.json"))
    assert request_payload["data"]["api_key"] == REDACTED
    response_payload = zstd_io.read_json_any(_artifact(logger.log_dir, "response.json"))
    assert "sk-secret" not in json.dumps(response_payload)


def test_l2_enables_intermediates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRANSACTION_LOG_LEVEL", "2")
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    assert logger.trace_level == 2

    _boundary_flow(logger)

    entries = zstd_io.read_jsonl_any(logger.log_dir / "transform_trace.jsonl")
    pass_names = [entry["pass_name"] for entry in entries]
    assert "raw_client_request" in pass_names
    assert "final_client_response" in pass_names
    assert any((logger.log_dir / "transforms").iterdir())


def test_capture_on_error_archives_buffered_trace_at_l1(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TRANSACTION_LOG_LEVEL", raising=False)
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    _boundary_flow(logger)

    class FakeRequestError(Exception):
        error_type = "invalid_request"

    captured = logger.flush_capture_on_error(FakeRequestError("bad payload"))

    assert captured is True
    capture_file = logger.log_dir / "capture" / "captured_trace.json"
    assert capture_file.exists() or (logger.log_dir / "capture" / "captured_trace.json.zst").exists()
    drained = zstd_io.read_json_any(logger.log_dir / "capture" / "captured_trace.json")
    assert isinstance(drained, list) and drained
    assert {entry["pass_name"] for entry in drained} >= {"raw_client_request", "final_client_response"}
    # The ring is drained: a second flush does not duplicate.
    assert logger.flush_capture_on_error(FakeRequestError("again")) is False


def test_rotation_class_failures_never_capture(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TRANSACTION_LOG_LEVEL", raising=False)
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    _boundary_flow(logger)

    class FakeRateLimit(Exception):
        error_type = "rate_limit"

    assert logger.flush_capture_on_error(FakeRateLimit("429")) is False
    assert not (logger.log_dir / "capture").exists()


def test_capture_bucket_covers_classifier_aliases() -> None:
    """The exclusion set stays aligned with the classifier vocabulary:
    every rotation-class error_type classify_error produces must be
    excluded from capture; request-relevant classes must qualify."""

    from rotator_library.error_handler import classify_error
    from rotator_library.transaction_logger import _error_qualifies_for_capture

    class _Fake(Exception):
        def __init__(self, message: str, error_type: str) -> None:
            super().__init__(message)
            self.error_type = error_type

    rotation_inputs = [
        _Fake("Rate limit exceeded", "rate_limit"),
        _Fake("quota exceeded on tier", "quota_exceeded"),
        _Fake("authentication error", "authentication_error"),
        _Fake("permission denied", "permission_error"),
        _Fake("timeout", "proxy_timeout"),
    ]
    for exc in rotation_inputs:
        assert _error_qualifies_for_capture(exc) is False, exc.error_type

    # Native provider spellings normalize to classifier stems.
    assert _error_qualifies_for_capture(_Fake("rl", "rate_limit_error")) is False
    assert _error_qualifies_for_capture(_Fake("q", "Quota-Exceeded")) is False
    # Structured status codes win over type labels.
    class _Structured(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__("structured")
            self.status_code = status_code

    assert _error_qualifies_for_capture(_Structured(429)) is False
    assert _error_qualifies_for_capture(_Structured(504)) is False
    assert _error_qualifies_for_capture(_Structured(400)) is True
    # Mid-stream provider error payloads.
    class _Streamed(Exception):
        def __init__(self, data: dict) -> None:
            super().__init__("streamed")
            self.data = data

    assert _error_qualifies_for_capture(_Streamed({"error": {"type": "rate_limit_error"}})) is False
    assert _error_qualifies_for_capture(_Streamed({"error": {"type": "invalid_request_error"}})) is True
    # The real classifier agrees on the split for representative inputs.
    classified = classify_error(_Fake("Rate limit exceeded", "rate_limit"))
    assert classified.error_type in {"rate_limit", "rate_limit_error"}


def test_log_transform_error_triggers_capture_and_records_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TRANSACTION_LOG_LEVEL", raising=False)
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    logger.log_request({"model": "gpt-test", "messages": []})

    logger.log_transform_error("adapter_chain", ValueError("boom"), payload={}, stage="adapter")

    assert (logger.log_dir / "capture").exists()
    logger.finalize_metadata(status_code=500)
    metadata = zstd_io.read_json_any(logger.log_dir / "metadata.json")
    assert metadata["errors"]
    assert metadata["errors"][0]["failed_pass_name"] == "adapter_chain"


def test_attempt_and_routing_records_land_in_metadata(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TRANSACTION_LOG_LEVEL", raising=False)
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    logger.record_attempt({
        "attempt": 1,
        "provider": "openai",
        "execution": "native",
        "incoming_protocol": "openai_chat",
        "outgoing_protocol": "openai_chat",
        "passthrough": True,
        "status": "success",
    })
    logger.record_routing({"event": "fallback_selected", "from": "openai", "to": "groq"})
    logger.update_metadata(fast_path="raw", overlays=["model"])

    _boundary_flow(logger)

    metadata = zstd_io.read_json_any(logger.log_dir / "metadata.json")
    assert metadata["attempts"][0]["execution"] == "native"
    assert metadata["attempts"][0]["passthrough"] is True
    assert metadata["routing"][0]["event"] == "fallback_selected"
    assert metadata["extra"] == {"fast_path": "raw", "overlays": ["model"]}


def test_retention_prunes_oldest_transactions(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRANSACTION_LOG_RETENTION", "3")
    monkeypatch.setattr(
        "rotator_library.transaction_logger._get_transactions_dir",
        lambda: tmp_path / "transactions",
    )
    for index in range(5):
        logger = TransactionLogger("openai", "openai/gpt-test")
        logger.log_request({"model": "gpt-test", "messages": []})

    remaining = list((tmp_path / "transactions").iterdir())
    assert len(remaining) == 3


def test_streaming_chunks_flush_once_not_per_chunk(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TRANSACTION_LOG_LEVEL", raising=False)
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    logger.log_request({"model": "gpt-test", "stream": True, "messages": []})

    for _ in range(50):
        logger.log_stream_chunk({"choices": [{"delta": {"content": "x"}}]})

    # Buffered: nothing on disk until finalize.
    assert not _artifact(logger.log_dir, "streaming_chunks.jsonl").exists()

    logger.log_response({"model": "gpt-test", "choices": [], "usage": {}})
    entries = zstd_io.read_jsonl_any(logger.log_dir / "streaming_chunks.jsonl")
    assert len(entries) == 50
