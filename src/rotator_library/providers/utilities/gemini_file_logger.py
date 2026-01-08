# src/rotator_library/providers/utilities/gemini_file_logger.py
"""
Shared file logger for Gemini-based providers.

Provides transaction-level logging for debugging API requests and responses.
Each request gets its own directory with separate files for request, response,
and errors.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ...utils.paths import get_logs_dir

lib_logger = logging.getLogger("rotator_library")


class GeminiFileLogger:
    """
    Base transaction file logger for Gemini-based providers.

    Creates a unique directory for each request and logs:
    - Request payload (JSON)
    - Response stream chunks (text)
    - Final response (JSON)
    - Errors (text)

    Subclasses can extend with provider-specific logging methods.
    """

    __slots__ = ("enabled", "log_dir")

    def __init__(self, model_name: str, enabled: bool, log_subdir: str):
        """
        Initialize the file logger.

        Args:
            model_name: Name of the model (used in directory name)
            enabled: Whether logging is enabled
            log_subdir: Subdirectory under logs/ for this provider (e.g., "gemini_cli_logs")
        """
        self.enabled = enabled
        self.log_dir: Optional[Path] = None

        if not enabled:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Sanitize model name for directory
        safe_model_name = model_name.replace("/", "_").replace(":", "_")
        request_id = str(uuid.uuid4())

        logs_base = get_logs_dir() / log_subdir
        logs_base.mkdir(parents=True, exist_ok=True)

        self.log_dir = logs_base / f"{timestamp}_{safe_model_name}_{request_id}"

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            lib_logger.error(f"Failed to create log directory: {e}")
            self.enabled = False

    def log_request(self, payload: Dict[str, Any]) -> None:
        """Log the request payload sent to the API."""
        self._write_json("request_payload.json", payload)

    def log_response_chunk(self, chunk: str) -> None:
        """Log a raw chunk from the response stream."""
        self._append_text("response_stream.log", chunk)

    def log_error(self, error_message: str) -> None:
        """Log an error message with timestamp."""
        self._append_text(
            "error.log", f"[{datetime.utcnow().isoformat()}] {error_message}"
        )

    def log_final_response(self, response_data: Dict[str, Any]) -> None:
        """Log the final, reassembled response."""
        self._write_json("final_response.json", response_data)

    def _write_json(self, filename: str, data: Dict[str, Any]) -> None:
        """Write JSON data to a file."""
        if not self.enabled or not self.log_dir:
            return
        try:
            with open(self.log_dir / filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            lib_logger.error(f"GeminiFileLogger: Failed to write {filename}: {e}")

    def _append_text(self, filename: str, text: str) -> None:
        """Append text to a file."""
        if not self.enabled or not self.log_dir:
            return
        try:
            with open(self.log_dir / filename, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception as e:
            lib_logger.error(f"GeminiFileLogger: Failed to append to {filename}: {e}")


class GeminiCliFileLogger(GeminiFileLogger):
    """File logger for Gemini CLI provider."""

    def __init__(self, model_name: str, enabled: bool = True):
        super().__init__(model_name, enabled, "gemini_cli_logs")


class AntigravityFileLogger(GeminiFileLogger):
    """
    File logger for Antigravity provider.

    Extends base logger with malformed function call logging.
    """

    def __init__(self, model_name: str, enabled: bool = True):
        super().__init__(model_name, enabled, "antigravity_logs")

    def log_malformed_retry_request(
        self, retry_num: int, payload: Dict[str, Any]
    ) -> None:
        """Log a malformed call retry request payload."""
        self._write_json(f"malformed_retry_{retry_num}_request.json", payload)

    def log_malformed_retry_response(self, retry_num: int, chunk: str) -> None:
        """Append a chunk to the malformed retry response log."""
        self._append_text(f"malformed_retry_{retry_num}_response.log", chunk)

    def log_malformed_autofix(
        self, tool_name: str, raw_args: str, fixed_json: str
    ) -> None:
        """Log details of an auto-fixed malformed function call."""
        self._write_json(
            "malformed_autofix.json",
            {
                "tool_name": tool_name,
                "raw_args": raw_args,
                "fixed_json": fixed_json,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )
