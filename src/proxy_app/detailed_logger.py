# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

# src/proxy_app/detailed_logger.py
"""
Raw I/O Logger for the Proxy Layer.

This logger captures the UNMODIFIED HTTP request and response at the proxy boundary.
It is disabled by default and should only be enabled for debugging the proxy itself.

Use this when you need to:
- Verify that requests/responses are not being corrupted
- Debug HTTP-level issues between the client and proxy
- Capture exact payloads as received/sent by the proxy

For normal request/response logging with provider correlation, use the
TransactionLogger in the rotator_library instead (enabled via --enable-request-logging).

Directory structure:
    logs/raw_io/{YYYYMMDD_HHMMSS}_{request_id}/
        request.json           # Unmodified incoming HTTP request
        streaming_chunks.jsonl # If streaming mode
        final_response.json    # Unmodified outgoing HTTP response
        metadata.json          # Summary metadata
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import logging

from rotator_library.utils.resilient_io import (
    safe_write_json,
    safe_log_write,
    safe_mkdir,
)
from rotator_library.utils.paths import get_logs_dir


def _get_raw_io_logs_dir() -> Path:
    """Get the raw I/O logs directory, creating it if needed."""
    logs_dir = get_logs_dir()
    raw_io_dir = logs_dir / "raw_io"
    raw_io_dir.mkdir(parents=True, exist_ok=True)
    return raw_io_dir


class RawIOLogger:
    """
    Logs raw HTTP request/response at the proxy boundary.

    This captures the EXACT data as received from and sent to the client,
    without any transformations. Useful for debugging the proxy itself.

    DISABLED by default. Enable with --enable-raw-logging flag.

    Uses fire-and-forget logging - if disk writes fail, logs are dropped (not buffered)
    to prevent memory issues, especially with streaming responses.
    """

    def __init__(self):
        """
        Initializes the logger for a single request, creating a unique directory
        to store all related log files.
        """
        self.start_time = time.time()
        self.request_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = _get_raw_io_logs_dir() / f"{timestamp}_{self.request_id}"
        self.streaming = False
        self._dir_available = safe_mkdir(self.log_dir, logging)

    def _write_json(self, filename: str, data: Dict[str, Any]):
        """Helper to write data to a JSON file in the log directory."""
        if not self._dir_available:
            # Try to create directory again in case it was recreated
            self._dir_available = safe_mkdir(self.log_dir, logging)
            if not self._dir_available:
                return

        safe_write_json(
            self.log_dir / filename,
            data,
            logging,
            atomic=False,
            indent=4,
            ensure_ascii=False,
        )

    def log_request(self, headers: Dict[str, Any], body: Dict[str, Any]):
        """Logs the raw incoming request details."""
        self.streaming = body.get("stream", False)
        request_data = {
            "request_id": self.request_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "headers": dict(headers),
            "body": body,
        }
        self._write_json("request.json", request_data)

    def log_stream_chunk(self, chunk: Dict[str, Any]):
        """Logs an individual chunk from a streaming response to a JSON Lines file."""
        if not self._dir_available:
            return

        log_entry = {"timestamp_utc": datetime.utcnow().isoformat(), "chunk": chunk}
        content = json.dumps(log_entry, ensure_ascii=False) + "\n"
        safe_log_write(self.log_dir / "streaming_chunks.jsonl", content, logging)

    def log_final_response(
        self, status_code: int, headers: Optional[Dict[str, Any]], body: Dict[str, Any]
    ):
        """Logs the raw outgoing response."""
        end_time = time.time()
        duration_ms = (end_time - self.start_time) * 1000

        response_data = {
            "request_id": self.request_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "status_code": status_code,
            "duration_ms": round(duration_ms),
            "headers": dict(headers) if headers else None,
            "body": body,
        }
        self._write_json("final_response.json", response_data)
        self._log_metadata(response_data)

    def _extract_reasoning(self, response_body: Dict[str, Any]) -> Optional[str]:
        """Recursively searches for and extracts 'reasoning' fields from the response body."""
        if not isinstance(response_body, dict):
            return None

        if "reasoning" in response_body:
            return response_body["reasoning"]

        if "choices" in response_body and response_body["choices"]:
            message = response_body["choices"][0].get("message", {})
            if "reasoning" in message:
                return message["reasoning"]
            if "reasoning_content" in message:
                return message["reasoning_content"]

        return None

    def _log_metadata(self, response_data: Dict[str, Any]):
        """Logs a summary of the transaction for quick analysis."""
        usage = response_data.get("body", {}).get("usage") or {}
        model = response_data.get("body", {}).get("model", "N/A")
        finish_reason = "N/A"
        if (
            "choices" in response_data.get("body", {})
            and response_data["body"]["choices"]
        ):
            finish_reason = response_data["body"]["choices"][0].get(
                "finish_reason", "N/A"
            )

        metadata = {
            "request_id": self.request_id,
            "timestamp_utc": response_data["timestamp_utc"],
            "duration_ms": response_data["duration_ms"],
            "status_code": response_data["status_code"],
            "model": model,
            "streaming": self.streaming,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "finish_reason": finish_reason,
            "reasoning_found": False,
            "reasoning_content": None,
        }

        reasoning = self._extract_reasoning(response_data.get("body", {}))
        if reasoning:
            metadata["reasoning_found"] = True
            metadata["reasoning_content"] = reasoning

        self._write_json("metadata.json", metadata)


# Backward compatibility alias
DetailedLogger = RawIOLogger
