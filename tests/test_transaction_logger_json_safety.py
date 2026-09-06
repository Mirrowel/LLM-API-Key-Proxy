from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rotator_library.transaction_logger import ProviderLogger, TransactionLogger
from rotator_library.utils import zstd_io
import pytest


def _artifact(log_dir, name):
    compressed = Path(str(log_dir / name) + ".zst")
    return compressed if compressed.exists() else log_dir / name



@dataclass
class NestedStats:
    count: int
    path: Path


@dataclass
class CircularNode:
    name: str
    child: object | None = None


class ProviderLeaf:
    def __str__(self) -> str:
        return "provider-leaf"


class ModelDumpResponse:
    """Small LiteLLM/Pydantic-like response object for logger regression tests."""

    def model_dump(self) -> dict:
        return {
            "id": "chatcmpl-test",
            "model": "gemini_cli/gemini-3-flash-preview",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                        "reasoning_content": "kept for metadata",
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "nested": NestedStats(count=2, path=Path("provider/state.json")),
            "created_at": datetime(2026, 1, 2, 3, 4, 5),
            "binary": b"hello",
            "set_values": {"a", "b"},
            "unknown": ProviderLeaf(),
        }


def test_transaction_logger_serializes_model_response_objects(tmp_path, caplog) -> None:
    logger = TransactionLogger("gemini_cli", "gemini_cli/gemini-3-flash-preview", parent_dir=tmp_path)

    logger.log_response(ModelDumpResponse())

    assert "Failed to write response.json" not in caplog.text
    response = zstd_io.read_json_any(logger.log_dir / "response.json")
    assert response["data"]["model"] == "gemini_cli/gemini-3-flash-preview"
    assert response["data"]["nested"] == {"count": 2, "path": str(Path("provider/state.json"))}
    assert response["data"]["created_at"] == "2026-01-02T03:04:05"
    assert response["data"]["binary"] == "hello"
    assert sorted(response["data"]["set_values"]) == ["a", "b"]
    assert response["data"]["unknown"] == "provider-leaf"

    metadata = zstd_io.read_json_any(logger.log_dir / "metadata.json")
    assert metadata["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    assert metadata["finish_reason"] == "stop"
    assert metadata["reasoning_found"] is True
    assert metadata["reasoning_content"] == "kept for metadata"


def test_transaction_logger_serializes_stream_chunks_with_provider_objects(tmp_path, caplog) -> None:
    logger = TransactionLogger("provider", "model", parent_dir=tmp_path)

    logger.log_stream_chunk({"chunk": ModelDumpResponse()})
    logger.log_response({"model": "model", "choices": []})  # finalize: flush buffered chunks

    assert "Failed to append to streaming_chunks.jsonl" not in caplog.text
    entries = zstd_io.read_jsonl_any(logger.log_dir / "streaming_chunks.jsonl")
    chunk_entry = next(entry for entry in entries if isinstance(entry.get("chunk"), dict) and isinstance(entry["chunk"].get("chunk"), dict) and entry["chunk"]["chunk"].get("id") == "chatcmpl-test")
    assert chunk_entry["chunk"]["chunk"]["id"] == "chatcmpl-test"


def test_provider_logger_serializes_final_response_objects(tmp_path, caplog) -> None:
    transaction_logger = TransactionLogger("provider", "model", parent_dir=tmp_path)
    provider_logger = ProviderLogger(transaction_logger.get_context())

    provider_logger.log_final_response(ModelDumpResponse())

    assert "ProviderLogger: Failed to write final_response.json" not in caplog.text
    payload = zstd_io.read_json_any(provider_logger.log_dir / "final_response.json")
    assert payload["id"] == "chatcmpl-test"
    assert payload["nested"]["path"] == str(Path("provider/state.json"))


def test_transaction_logger_handles_circular_provider_payloads(tmp_path) -> None:
    logger = TransactionLogger("provider", "model", parent_dir=tmp_path)
    node = CircularNode("root")
    node.child = node

    logger.log_response({"model": "model", "usage": {}, "node": node})

    response = zstd_io.read_json_any(logger.log_dir / "response.json")
    assert response["data"]["node"] == {"name": "root", "child": "[CIRCULAR]"}
