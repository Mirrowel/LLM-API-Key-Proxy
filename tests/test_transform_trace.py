from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from rotator_library.transform_trace import (
    REDACTED,
    TransformTraceWriter,
    sanitize_filename,
    sanitize_for_trace,
    scrub_sensitive_text,
)


@dataclass
class ExamplePayload:
    when: datetime
    amount: Decimal


def test_sanitize_for_trace_redacts_sensitive_keys_recursively() -> None:
    payload = {
        "api_key": "secret-key",
        "headers": {"Authorization": "Bearer secret", "Cookie": "sid=secret", "normal": "token in normal text"},
        "items": [{"refresh_token": "refresh", "text": "token should remain in value"}],
    }

    sanitized = sanitize_for_trace(payload)

    assert sanitized["api_key"] == REDACTED
    assert sanitized["headers"]["Authorization"] == REDACTED
    assert sanitized["headers"]["Cookie"] == REDACTED
    assert sanitized["headers"]["normal"] == "token in normal text"
    assert sanitized["items"][0]["refresh_token"] == REDACTED
    assert sanitize_for_trace({"reasoning_content": "provider-state"})["reasoning_content"] == REDACTED
    assert sanitize_for_trace({"thoughtSignature": "provider-state"})["thoughtSignature"] == REDACTED
    assert sanitize_for_trace({"metadata": {"state": "provider-state"}})["metadata"]["state"] == REDACTED
    assert sanitize_for_trace({"metadata": {"prompt_cache_key": "cache-state"}})["metadata"]["prompt_cache_key"] == REDACTED
    assert sanitized["items"][0]["text"] == "token should remain in value"


def test_scrub_sensitive_text_targets_header_like_fragments_only() -> None:
    text = "normal token text remains\nAuthorization: Bearer abc123\nset-cookie: sid=secret\n{'Authorization': 'Bearer quoted'}"

    scrubbed = scrub_sensitive_text(text)

    assert "normal token text remains" in scrubbed
    assert "Authorization: [REDACTED]" in scrubbed
    assert "set-cookie: [REDACTED]" in scrubbed
    assert "quoted" not in scrubbed


def test_sanitize_for_trace_extracts_sdk_like_objects_before_repr() -> None:
    class SdkError:
        def __init__(self) -> None:
            self.status_code = 401
            self.headers = {"Authorization": "Bearer secret"}

        def __repr__(self) -> str:
            return "SdkError(Authorization: Bearer leaked)"

    sanitized = sanitize_for_trace(SdkError(), scrub_strings=True)

    assert sanitized["status_code"] == 401
    assert sanitized["headers"]["Authorization"] == REDACTED
    assert "leaked" not in json.dumps(sanitized)


def test_sanitize_for_trace_serializes_common_non_json_values() -> None:
    payload = ExamplePayload(when=datetime(2026, 1, 2, 3, 4, 5), amount=Decimal("1.5"))

    sanitized = sanitize_for_trace({"payload": payload, "bytes": b"hello"})

    assert sanitized["payload"]["when"] == "2026-01-02T03:04:05"
    assert sanitized["payload"]["amount"] == 1.5
    assert sanitized["bytes"] == "hello"
    json.dumps(sanitized)


def test_sanitize_filename_is_stable_and_filesystem_safe() -> None:
    assert sanitize_filename("Raw Client Request") == "raw_client_request"
    assert sanitize_filename("provider/request:payload") == "provider_request_payload"


def test_transform_trace_writer_records_jsonl_and_snapshots(tmp_path) -> None:
    writer = TransformTraceWriter(tmp_path, component="client", provider="openai", model="gpt-test", request_id="req_1")

    first = writer.record("raw_client_request", {"model": "gpt-test"}, direction="request", stage="client")
    second = writer.record("parsed_stream_chunk", {"delta": "hi"}, direction="stream", stage="client", snapshot=True)

    assert first is not None
    assert second is not None
    assert first.sequence == 1
    assert second.sequence == 2

    lines = (tmp_path / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["pass_name"] == "raw_client_request"
    assert json.loads(lines[0])["request_id"] == "req_1"
    assert json.loads(lines[1])["direction"] == "stream"
    assert (tmp_path / "transforms" / "0001_raw_client_request.json").exists()
    assert not (tmp_path / "transforms" / "0002_parsed_stream_chunk.json").exists()


def test_transform_trace_writer_disabled_writes_nothing(tmp_path) -> None:
    writer = TransformTraceWriter(tmp_path, component="client", enabled=False)

    assert writer.record("raw_client_request", {}, direction="request", stage="client") is None
    assert not (tmp_path / "transform_trace.jsonl").exists()


def test_transform_trace_writer_snapshot_namespace_prevents_collisions(tmp_path) -> None:
    first = TransformTraceWriter(tmp_path, component="provider", snapshot_namespace="provider_a")
    second = TransformTraceWriter(tmp_path, component="provider", snapshot_namespace="provider_b")

    first.record("provider_request_payload", {"a": 1}, direction="request", stage="provider")
    second.record("provider_request_payload", {"b": 2}, direction="request", stage="provider")

    assert (tmp_path / "transforms" / "0001_provider_a_provider_request_payload.json").exists()
    assert (tmp_path / "transforms" / "0001_provider_b_provider_request_payload.json").exists()
