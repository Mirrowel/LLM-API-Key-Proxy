# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Shared chat-wire streaming helpers.

The legacy ``StreamingHandler`` stream loop has been retired (W5/G4): the
neutral-event pipeline in ``client/stream_ops.py`` owns stream operations.
This module retains only the helpers the pipeline still imports — in-band
error detection, SSE data-payload extraction, the JSON reassembly buffer, and
SSE cost-frame parsing. Do not add new callers.
"""

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

from ..usage.accounting import UsageRecord, extract_usage_record


class StreamingHandler:
    """Legacy name retained for the shared chat-wire parsing helpers."""

    @classmethod
    def _in_band_error_payload(cls, chunk: Any) -> Optional[Dict[str, Any]]:
        """Return a structured provider error embedded in a stream frame."""

        event_name = None
        payload = chunk
        if isinstance(chunk, str):
            event_names = [line[6:].strip() for line in chunk.splitlines() if line.startswith("event:")]
            event_name = event_names[-1] if event_names else None
            data_payloads = cls._sse_data_payloads(chunk)
            if not data_payloads:
                return None
            try:
                payload = json.loads(data_payloads[-1])
            except json.JSONDecodeError:
                if event_name == "error":
                    return {"type": "upstream_error", "message": data_payloads[-1]}
                return None
        if not isinstance(payload, dict):
            return None
        error = payload.get("error")
        if isinstance(error, dict):
            return deepcopy(error)
        if error is not None:
            return {"type": "upstream_error", "message": str(error)}
        event_type = str(event_name or payload.get("event_type") or payload.get("type") or "")
        if event_type == "response.failed":
            response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
            nested = response.get("error")
            if isinstance(nested, dict):
                return deepcopy(nested)
        if event_type in {"error", "response.error"}:
            result = deepcopy(payload)
            result.pop("event_type", None)
            if result.get("type") in {None, "error", "response.error"}:
                result["type"] = str(result.get("error_type") or "upstream_error")
            result.setdefault("message", "Provider stream failed")
            return result
        return None

    @staticmethod
    def _sse_data_payloads(sse_string: str) -> List[str]:
        """Extract SSE data payloads from plain or event-prefixed frames."""

        payloads: List[str] = []
        for frame in re.split(r"\r?\n\r?\n", sse_string):
            data_lines = []
            for line in frame.splitlines():
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if data_lines:
                payloads.append("\n".join(data_lines).strip())
        return payloads


def _usage_record_from_sse_cost_chunk(chunk: Any, *, model: str) -> UsageRecord:
    """Extract provider-reported stream cost from SSE comments/events."""

    if not isinstance(chunk, str):
        return UsageRecord(source="stream_cost_event", model=model)
    cost_payload = _sse_cost_payload(chunk)
    if cost_payload is None:
        return UsageRecord(source="stream_cost_event", model=model)
    if isinstance(cost_payload, (int, float, str)):
        cost_payload = {"provider_reported_cost": cost_payload, "source": "sse_cost"}
    if not isinstance(cost_payload, dict):
        return UsageRecord(source="stream_cost_event", model=model)
    return extract_usage_record(
        {"usage": {"provider_reported_cost": cost_payload.get("provider_reported_cost", cost_payload.get("request_cost_usd", cost_payload.get("total_cost", cost_payload.get("cost", cost_payload.get("estimated_cost"))))), "currency": cost_payload.get("currency", "USD"), "cost_details": cost_payload}},
        model=model,
        source="stream_cost_event",
    )


def _sse_cost_payload(chunk: str) -> Any:
    """Parse `: cost ...` comments and `event: cost` frames."""

    event_type: Optional[str] = None
    data_lines: list[str] = []
    for line in chunk.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith(":"):
            comment = stripped[1:].strip()
            if comment.startswith("cost"):
                return _parse_cost_text(comment[4:].strip())
            continue
        if stripped.startswith("event:"):
            event_type = stripped[6:].strip()
            continue
        if stripped.startswith("data:"):
            data_lines.append(stripped[5:].strip())
    if event_type == "cost" and data_lines:
        return _parse_cost_text("\n".join(data_lines).strip())
    return None


def _parse_cost_text(text: str) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return float(text)
        except ValueError:
            return None


class StreamBuffer:
    """
    Buffer for reassembling fragmented JSON in streams.

    Some providers send JSON split across multiple chunks, especially
    for error responses. This class handles accumulation and parsing.
    """

    def __init__(self):
        self._buffer = ""
        self._complete = False

    def append(self, chunk: str) -> Optional[Dict]:
        """
        Append a chunk and try to parse.

        Args:
            chunk: Raw chunk string

        Returns:
            Parsed dict if complete, None if still accumulating
        """
        self._buffer += chunk

        try:
            result = json.loads(self._buffer)
            self._complete = True
            return result
        except json.JSONDecodeError:
            return None

    def reset(self) -> None:
        """Reset the buffer."""
        self._buffer = ""
        self._complete = False

    @property
    def content(self) -> str:
        """Get current buffer content."""
        return self._buffer

    @property
    def is_complete(self) -> bool:
        """Check if buffer contains complete JSON."""
        return self._complete
