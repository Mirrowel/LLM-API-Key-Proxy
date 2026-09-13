# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/transaction_logger.py
"""
Unified transaction logging for the rotator library (G10 record model).

Each API transaction accumulates in memory as one :class:`TransactionRecord`
(four wire boundaries, value-level change log, recipe, metadata) and is
sealed at request end into exactly ONE compressed archive file:

    logs/transactions/MMDD_HHMMSS_{protocol}[_{profile}]_{provider}_{model}_{request_id}.transaction.zst

All disk I/O funnels through the single background
:class:`~rotator_library.transaction.TransactionWriter` (group flush, atomic
tmp+rename, newest-N retention). The old multi-file directory layout, the
quadratic append-jsonl rewrites, and the capture/ side-directory are
retired: the record IS the buffer, so error capture is simply "seal with
everything in it".
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .transaction import TransactionRecord, TransactionWriter, archive as _archive
from .transform_trace import (
    sanitize_for_trace,
    scrub_sensitive_text,
)
from .utils.paths import get_logs_dir

lib_logger = logging.getLogger("rotator_library")

FRAMEWORK_KEYS = frozenset({
    "api_key",
    "api_base",
    "custom_llm_provider",
    "transaction_context",
    "credential_identifier",
})

_API_FORMAT_PROTOCOLS = {
    "oai": "openai_chat",
    "ant": "anthropic_messages",
    "gem": "gemini",
    "responses": "responses",
}


def _strip_framework_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if k not in FRAMEWORK_KEYS}
    return data


def _make_json_safe(value: Any, _seen: Optional[set[int]] = None) -> Any:
    """Recursively convert non-JSON-native values (dataclasses, Paths,
    datetimes, sets) into plain JSON structures; circular-safe."""

    if _seen is None:
        _seen = set()
    marker = id(value)
    if marker in _seen:
        return "<circular>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        _seen.add(marker)
        try:
            return _make_json_safe(model_dump(), _seen)
        except Exception:
            pass
        finally:
            _seen.discard(marker)
    as_dict = getattr(value, "to_dict", None)
    if callable(as_dict):
        _seen.add(marker)
        try:
            return _make_json_safe(as_dict(), _seen)
        except Exception:
            pass
        finally:
            _seen.discard(marker)
    if is_dataclass(value) and not isinstance(value, type):
        _seen.add(marker)
        try:
            return _make_json_safe(value.__dict__, _seen)
        finally:
            _seen.discard(marker)
    if isinstance(value, dict):
        _seen.add(marker)
        try:
            return {str(k): _make_json_safe(v, _seen) for k, v in value.items()}
        finally:
            _seen.discard(marker)
    if isinstance(value, (list, tuple, set, frozenset)):
        _seen.add(marker)
        try:
            return [_make_json_safe(item, _seen) for item in value]
        finally:
            _seen.discard(marker)
    return str(value)


def _get_transactions_dir() -> Path:
    base = Path(get_logs_dir()) / "transactions"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


# Rotation-class failures never trigger error capture (rate limit, quota,
# auth, transport) — the record still seals, just without the escalation.
_ROTATION_ERROR_TYPES = frozenset({
    "rate_limit_error",
    "rate_limit",
    "quota_exceeded",
    "authentication_error",
    "authentication",
    "permission_error",
    "forbidden",
    "timeout_error",
    "api_connection_error",
    "api_connection",
})


def _normalize_error_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9_]", "_", text)


def _error_qualifies_for_capture(error: BaseException) -> bool:
    error_type = _normalize_error_type(getattr(error, "error_type", None) or type(error).__name__)
    return error_type not in _ROTATION_ERROR_TYPES


@dataclass
class TransactionContext:
    """Lightweight correlation handle passed to providers."""

    log_dir: Optional[Path]
    request_id: str
    enabled: bool
    provider: str
    model: str
    trace_model: str
    session_id: Optional[str] = None
    scope_key: Optional[str] = None
    classifier: Optional[str] = None
    trace_enabled: bool = False
    trace_level: int = 1
    record: Optional[TransactionRecord] = None


class TransactionLogger:
    """Facade over the G10 transaction record for one client request.

    The public surface is unchanged from the directory-layout era; the
    internals accumulate into a :class:`TransactionRecord` and seal into
    one archive via the background writer at request end.
    """

    __slots__ = (
        "enabled",
        "log_dir",
        "start_time",
        "request_id",
        "provider",
        "model",
        "trace_model",
        "session_id",
        "scope_key",
        "classifier",
        "streaming",
        "api_format",
        "protocol_name",
        "profile_name",
        "_record",
        "_context",
        "_sealed",
        "_error_records",
        "_extra_metadata",
        "_capture_flushed",
        "sealed_envelope",
    )

    def __init__(
        self,
        provider: str,
        model: str,
        enabled: bool = True,
        api_format: str = "oai",
        parent_dir: Optional[Path] = None,
        *,
        protocol: Optional[str] = None,
        profile: Optional[str] = None,
        operation: str = "",
        execution_mode: str = "",
    ):
        self.enabled = enabled
        self.start_time = time.time()
        self.request_id = str(uuid.uuid4())[:8]
        self.provider = provider
        self.trace_model = model
        self.session_id: Optional[str] = None
        self.scope_key: Optional[str] = None
        self.classifier: Optional[str] = None
        self.api_format = api_format
        self.protocol_name = protocol or _API_FORMAT_PROTOCOLS.get(api_format, api_format)
        self.profile_name = profile
        self.streaming = False
        self.log_dir: Optional[Path] = None
        self._context: Optional[TransactionContext] = None
        self._sealed = False
        self._error_records: list[Dict[str, Any]] = []
        self._extra_metadata: Dict[str, Any] = {}
        self._capture_flushed = False
        self.sealed_envelope: Optional[Dict[str, Any]] = None

        model_name = model
        if "/" in model_name and model_name.split("/")[0] == provider:
            model_name = model_name.split("/", 1)[1]

        self.model = model_name

        self._record: Optional[TransactionRecord] = None
        if not enabled:
            return
        try:
            self._record = TransactionRecord(
                request_id=self.request_id,
                protocol=self.protocol_name,
                provider=provider,
                model=model_name,
                profile=profile,
                operation=operation,
                execution_mode=execution_mode,
            )
            self.log_dir = _get_transactions_dir()
        except Exception as exc:
            lib_logger.error("TransactionLogger: record init failed: %s", exc)
            self.enabled = False

    # -- context ---------------------------------------------------------

    def get_context(self) -> TransactionContext:
        if self._context is None:
            self._context = TransactionContext(
                log_dir=self.log_dir if self.log_dir else Path("."),
                request_id=self.request_id,
                enabled=self.enabled,
                provider=self.provider,
                model=self.model,
                trace_model=self.trace_model,
                session_id=self.session_id,
                scope_key=self.scope_key,
                classifier=self.classifier,
                trace_enabled=self.enabled,
                trace_level=1,
                record=self._record,
            )
        return self._context

    def set_trace_context(
        self,
        *,
        session_id: Optional[str] = None,
        scope_key: Optional[str] = None,
        classifier: Optional[str] = None,
    ) -> None:
        if session_id is not None:
            self.session_id = session_id
        if scope_key is not None:
            self.scope_key = scope_key
        if classifier is not None:
            self.classifier = classifier
        if self._record is not None:
            self._record.update_metadata(
                session_id=self.session_id,
                scope_key=self.scope_key,
                classifier=self.classifier,
            )
        if self._context:
            self._context.session_id = self.session_id
            self._context.scope_key = self.scope_key
            self._context.classifier = self.classifier

    # -- change log ------------------------------------------------------

    def log_transform_pass(
        self,
        pass_name: str,
        data: Any,
        *,
        direction: str,
        stage: str,
        protocol: Optional[str] = None,
        credential_id: Optional[str] = None,
        transport: Optional[str] = None,
        changed_from_previous: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scrub_strings: bool = False,
        snapshot: bool = True,
    ) -> None:
        """Record one transform-pipeline observation into the change log."""

        if not self.enabled or self._record is None:
            return
        payload = data
        if scrub_strings and isinstance(data, (str, bytes)):
            payload = scrub_sensitive_text(str(data))
        self._record.record_change(
            stage,
            "trace_pass",
            detail="/".join(str(part) for part in (pass_name, direction, stage) if part),
            value=sanitize_for_trace(_make_json_safe(payload)) if (snapshot or changed_from_previous) else None,
            code=pass_name,
        )

    def log_transform_error(
        self,
        failed_pass_name: str,
        error: BaseException,
        *,
        payload: Any = None,
        stage: str = "client",
        protocol: Optional[str] = None,
        transport: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._record is not None:
            self._record.record_error(
                _normalize_error_type(getattr(error, "error_type", None) or type(error).__name__),
                scrub_sensitive_text(str(error)),
                raw=payload,
            )
            self._error_records.append(
                {
                    "failed_pass_name": failed_pass_name,
                    "error_type": type(error).__name__,
                    "message": scrub_sensitive_text(str(error))[:2000],
                    "stage": stage,
                }
            )
        self.flush_capture_on_error(error)

    def flush_capture_on_error(self, error: Optional[BaseException] = None) -> bool:
        """Mark the record error-escalated (the sealed archive IS the capture)."""

        if not self.enabled or self._capture_flushed:
            return False
        if error is not None and not _error_qualifies_for_capture(error):
            return False
        if self._record is None:
            return False
        self._capture_flushed = True
        self._record.mark_escalation("error_capture")
        return True

    # -- metadata ----------------------------------------------------------

    def record_attempt(self, record: Dict[str, Any]) -> None:
        if self._record is not None and isinstance(record, dict):
            self._record.record_attempt(_make_json_safe(record))

    def record_routing(self, record: Dict[str, Any]) -> None:
        if self._record is not None and isinstance(record, dict):
            self._record.record_routing(_make_json_safe(record))

    def update_metadata(self, **fields: Any) -> None:
        self._extra_metadata.update(_make_json_safe(dict(fields)))
        if self._record is not None:
            self._record.update_metadata(**_make_json_safe(dict(fields)))

    def finalize_metadata(
        self,
        *,
        status_code: int = 200,
        error: Optional[BaseException] = None,
    ) -> None:
        """Seal the record (stream + responses routes; no response body)."""

        if error is not None and self._record is not None:
            record = {
                "failed_pass_name": "finalize",
                "error_type": type(error).__name__,
                "message": scrub_sensitive_text(str(error))[:2000],
                "stage": "final",
            }
            if not any(
                entry.get("error_type") == record["error_type"] and entry.get("message") == record["message"]
                for entry in self._error_records
            ):
                self._record.record_error(record["error_type"], record["message"])
        self._seal_and_submit(status_code)

    # -- boundaries --------------------------------------------------------

    def log_request(
        self, request_data: Dict[str, Any], filename: str = "request.json"
    ) -> None:
        if not self.enabled or self._record is None:
            return
        self.streaming = bool(request_data.get("stream", False))
        self._record.set_boundary("client_request", sanitize_for_trace(_make_json_safe(request_data)))

    def log_transformed_request(
        self,
        transformed_data: Dict[str, Any],
        original_data: Dict[str, Any],
        *,
        credential_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or self._record is None:
            return
        stripped_transformed = _strip_framework_keys(transformed_data)
        stripped_original = _strip_framework_keys(original_data)
        try:
            changed = json.dumps(stripped_transformed, sort_keys=True, default=str) != json.dumps(
                stripped_original, sort_keys=True, default=str
            )
        except (TypeError, ValueError):
            changed = True
        if changed is False:
            return
        self._record.set_boundary(
            "provider_request",
            _strip_framework_keys(sanitize_for_trace(_make_json_safe(transformed_data))),
        )

    def log_stream_chunk(self, chunk: Dict[str, Any]) -> None:
        if not self.enabled or self._record is None:
            return
        self._record.add_client_chunk(_make_json_safe(chunk))

    def log_response(
        self,
        response_data: Dict[str, Any],
        status_code: int = 200,
        headers: Optional[Dict[str, Any]] = None,
        filename: str = "response.json",
    ) -> None:
        if not self.enabled or self._record is None:
            return
        self._record.set_boundary(
            "client_egress",
            sanitize_for_trace(_make_json_safe(response_data)),
        )
        if headers:
            self._record.update_metadata(response_headers=sanitize_for_trace(_make_json_safe(dict(headers))))
        self._seal_and_submit(status_code)

    # -- sealing -----------------------------------------------------------

    def _seal_and_submit(self, status_code: int) -> None:
        if not self.enabled or self._sealed or self._record is None:
            return
        self._sealed = True
        duration_ms = (time.time() - self.start_time) * 1000
        self._record.update_metadata(
            duration_ms=round(duration_ms, 2),
            api_format=self.api_format,
            errors=list(self._error_records) or None,
        )
        envelope = self._record.seal(status_code=status_code)
        self.sealed_envelope = envelope
        filename = _archive.archive_filename(
            protocol=self.protocol_name,
            provider=self.provider,
            model=self.model,
            request_id=self.request_id,
            profile=self.profile_name,
            when=self.start_time,
        )
        TransactionWriter.instance().submit_sealed(envelope, filename=filename)

    @staticmethod
    def assemble_streaming_response(
        chunks: list, request_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assemble streaming chunks into a final chat.completion response.

        This mirrors the aggregation logic from main.py's streaming_response_wrapper.
        Takes a list of parsed chunk dicts and combines them into a complete response.

        Args:
            chunks: List of parsed streaming chunk dictionaries
            request_data: Optional original request data for context

        Returns:
            A complete chat.completion response dictionary
        """
        if not chunks:
            return {}

        choice_messages: Dict[int, Dict[str, Any]] = {}
        choice_tools: Dict[int, Dict[int, Dict[str, Any]]] = {}
        choice_finish: Dict[int, Optional[str]] = {}
        choice_order: list[int] = []
        usage_data = None

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            for choice in chunk.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                index = choice.get("index", 0)
                if index not in choice_messages:
                    choice_messages[index] = {"role": "assistant"}
                    choice_tools[index] = {}
                    choice_finish[index] = None
                    choice_order.append(index)

                final_message = choice_messages[index]
                aggregated_tool_calls = choice_tools[index]
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    delta = {}

                for key, value in delta.items():
                    if value is None:
                        continue

                    if key == "content":
                        if "content" not in final_message:
                            final_message["content"] = ""
                        if value:
                            final_message["content"] += value

                    elif key == "tool_calls":
                        for tc_chunk in value or []:
                            tc_index = tc_chunk.get("index", 0)
                            if tc_index not in aggregated_tool_calls:
                                aggregated_tool_calls[tc_index] = {
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            entry = aggregated_tool_calls[tc_index]
                            if "function" not in entry:
                                entry["function"] = {"name": "", "arguments": ""}
                            if tc_chunk.get("id"):
                                entry["id"] = tc_chunk["id"]
                            tc_function = tc_chunk.get("function")
                            if isinstance(tc_function, dict):
                                if tc_function.get("name") is not None:
                                    entry["function"]["name"] += tc_function["name"]
                                if tc_function.get("arguments") is not None:
                                    entry["function"]["arguments"] += tc_function[
                                        "arguments"
                                    ]

                    elif key == "function_call":
                        if "function_call" not in final_message:
                            final_message["function_call"] = {
                                "name": "",
                                "arguments": "",
                            }
                        if "name" in value and value["name"] is not None:
                            final_message["function_call"]["name"] += value["name"]
                        if "arguments" in value and value["arguments"] is not None:
                            final_message["function_call"]["arguments"] += value[
                                "arguments"
                            ]

                    else:
                        if key == "role":
                            final_message[key] = value
                        elif key not in final_message:
                            final_message[key] = value
                        elif isinstance(final_message.get(key), str) and isinstance(
                            value, str
                        ):
                            final_message[key] += value
                        elif isinstance(final_message.get(key), list) and isinstance(
                            value, list
                        ):
                            final_message[key].extend(value)
                        else:
                            final_message[key] = value

                if choice.get("finish_reason"):
                    choice_finish[index] = choice["finish_reason"]

            if isinstance(chunk.get("usage"), dict) and chunk["usage"]:
                usage_data = chunk["usage"]

        final_choices: list[Dict[str, Any]] = []
        for index in choice_order:
            final_message = choice_messages[index]
            aggregated_tool_calls = choice_tools[index]

            if aggregated_tool_calls:
                final_message["tool_calls"] = list(aggregated_tool_calls.values())

            for missing in ["content", "tool_calls", "function_call"]:
                if missing not in final_message:
                    final_message[missing] = None

            finish_reason = choice_finish[index]
            if not finish_reason:
                finish_reason = "tool_calls" if aggregated_tool_calls else "stop"

            final_choices.append(
                {
                    "index": index,
                    "message": final_message,
                    "finish_reason": finish_reason,
                }
            )

        first_chunk = chunks[0] if isinstance(chunks[0], dict) else {}

        return {
            "id": first_chunk.get("id"),
            "object": "chat.completion",
            "created": first_chunk.get("created"),
            "model": first_chunk.get("model"),
            "choices": final_choices,
            "usage": usage_data,
        }


class ProviderLogger:
    """Provider-side boundary logging onto the shared transaction record."""

    __slots__ = ("enabled", "log_dir", "_record")

    def __init__(self, context: Optional[TransactionContext]):
        self.enabled = False
        self.log_dir: Optional[Path] = None
        self._record: Optional[TransactionRecord] = None

        if context is None or not context.enabled:
            return
        record = getattr(context, "record", None)
        if record is None:
            return
        self.enabled = True
        self.log_dir = context.log_dir
        self._record = record

    def finalize(self) -> None:
        """No-op under the record model (the client logger seals)."""

    def log_request(self, payload: Dict[str, Any]) -> None:
        if self._record is not None:
            self._record.set_boundary("provider_request", sanitize_for_trace(_make_json_safe(payload)))

    def log_response_chunk(self, chunk: str) -> None:
        if self._record is not None:
            self._record.add_stream_chunk(chunk)

    def log_final_response(self, response_data: Dict[str, Any]) -> None:
        if self._record is not None:
            self._record.set_boundary("provider_response", sanitize_for_trace(_make_json_safe(response_data)))

    def log_error(self, error_message: str) -> None:
        if self._record is not None:
            self._record.record_error("provider_error", scrub_sensitive_text(str(error_message)))

    def log_extra(self, filename: str, data: Union[Dict[str, Any], str]) -> None:
        if self._record is not None:
            self._record.record_change(
                "provider",
                "provider_extra",
                detail=str(filename),
                value=_make_json_safe(data) if isinstance(data, dict) else str(data),
            )
