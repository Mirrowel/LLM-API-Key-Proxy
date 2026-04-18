# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
In-memory ring buffer for tracking recent proxy errors.

Provides a lightweight, thread-safe store of the last N errors across all
providers and models. Used by the /v1/health and /v1/health/errors endpoints
to surface error diagnostics without parsing failures.log on every request.

Design decisions:
- Max 500 total records (deque evicts oldest automatically)
- No persistence — resets on restart (failures.log is the durable audit trail)
- Thread-safe via threading.Lock (errors are recorded in the failure path)
- Error messages are truncated to 500 chars to bound memory usage
"""

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Maximum number of error records to retain globally
MAX_ERROR_RECORDS: int = 500

# Maximum length of the error_message field per record
ERROR_MESSAGE_MAX_LEN: int = 500


@dataclass
class ErrorRecord:
    """A single captured error event."""

    timestamp: float          # Unix timestamp of the error
    provider: str             # Provider name (e.g., "modal", "antigravity")
    model: str                # Full model ID (e.g., "modal/qwen3-coder-480b")
    error_type: str           # Exception class name (e.g., "RateLimitError")
    status_code: Optional[int]  # HTTP status code if applicable
    error_message: str        # Truncated error message (max 500 chars)
    credential_masked: str    # Masked credential identifier
    attempt: int              # Attempt number (1-based)

    def to_dict(self) -> dict:
        """Serialize to a JSON-serializable dict for API responses."""
        return {
            "timestamp": datetime.fromtimestamp(
                self.timestamp, tz=timezone.utc
            ).isoformat(),
            "provider": self.provider,
            "model": self.model,
            "error_type": self.error_type,
            "status_code": self.status_code,
            "error_message": self.error_message,
            "credential": self.credential_masked,
            "attempt": self.attempt,
        }


class ErrorTracker:
    """
    Thread-safe in-memory ring buffer for recent proxy errors.

    Retains the last MAX_ERROR_RECORDS errors globally.
    Supports fast filtering by provider and/or model.
    """

    def __init__(self, max_records: int = MAX_ERROR_RECORDS):
        self._max_records = max_records
        self._records: deque = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record_error(
        self,
        provider: str,
        model: str,
        error_type: str,
        error_message: str,
        credential_masked: str,
        attempt: int,
        status_code: Optional[int] = None,
    ) -> None:
        """
        Record a new error event.

        Args:
            provider: Provider name (e.g., "modal")
            model: Full model ID (e.g., "modal/qwen3-coder-480b")
            error_type: Exception class name
            error_message: Error message (will be truncated)
            credential_masked: Already-masked credential string
            attempt: Attempt number (1-based)
            status_code: HTTP status code if available
        """
        import time

        record = ErrorRecord(
            timestamp=time.time(),
            provider=provider,
            model=model,
            error_type=error_type,
            status_code=status_code,
            error_message=error_message[:ERROR_MESSAGE_MAX_LEN],
            credential_masked=credential_masked,
            attempt=attempt,
        )
        with self._lock:
            self._records.append(record)

    def get_recent_errors(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        limit: int = 5,
    ) -> tuple:
        """
        Return the most recent errors, optionally filtered.

        Filters are applied in order: model (most specific) → provider.
        Returns the N most recent matching records (newest first).

        Args:
            provider: If set, only return errors for this provider
            model: If set, only return errors for this full model ID
            limit: Maximum number of records to return (capped at 50)

        Returns:
            Tuple of (matching_records_list, total_matching_count)
        """
        limit = min(max(1, limit), 50)

        with self._lock:
            # Snapshot to avoid holding lock during iteration
            records = list(self._records)

        # Filter (newest first — deque appends to right, so reversed = newest first)
        filtered = [
            r for r in reversed(records)
            if (model is None or r.model == model)
            and (provider is None or r.provider == provider)
        ]

        total = len(filtered)
        return filtered[:limit], total

    def get_error_summary(self) -> Dict:
        """
        Return an aggregated summary of all buffered errors.

        Groups counts by provider and model, with a breakdown of error types.

        Returns:
            Dict with total_errors, by_provider, by_model
        """
        with self._lock:
            records = list(self._records)

        # Aggregate
        by_provider: Dict[str, Dict] = {}
        by_model: Dict[str, Dict] = {}

        for r in records:
            # Per-provider
            if r.provider not in by_provider:
                by_provider[r.provider] = {"count": 0, "error_types": {}}
            by_provider[r.provider]["count"] += 1
            et = r.error_type
            by_provider[r.provider]["error_types"][et] = (
                by_provider[r.provider]["error_types"].get(et, 0) + 1
            )

            # Per-model
            if r.model not in by_model:
                by_model[r.model] = {"count": 0, "error_types": {}}
            by_model[r.model]["count"] += 1
            by_model[r.model]["error_types"][et] = (
                by_model[r.model]["error_types"].get(et, 0) + 1
            )

        return {
            "total_errors": len(records),
            "by_provider": by_provider,
            "by_model": by_model,
        }

    def clear(self) -> None:
        """Clear all buffered errors (for testing)."""
        with self._lock:
            self._records.clear()

    @property
    def record_count(self) -> int:
        """Current number of buffered records."""
        with self._lock:
            return len(self._records)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_error_tracker: Optional[ErrorTracker] = None
_tracker_lock = threading.Lock()


def get_error_tracker() -> ErrorTracker:
    """
    Get the global ErrorTracker singleton, initializing it if needed.

    Uses double-checked locking for thread-safe lazy initialization.
    """
    global _error_tracker
    if _error_tracker is None:
        with _tracker_lock:
            if _error_tracker is None:
                _error_tracker = ErrorTracker(max_records=MAX_ERROR_RECORDS)
    return _error_tracker
