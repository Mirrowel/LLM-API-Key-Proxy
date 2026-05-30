# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/transaction_logger.py
"""
Unified transaction logging for the rotator library.

Provides correlated logging between the OpenAI-compatible client layer and
provider-specific implementations. Each API transaction gets a unique directory
containing both client-level I/O and provider-level details.

Directory structure:
    logs/transactions/MMDD_HHMMSS_{provider}_{model}_{request_id}/
        request.json              # OpenAI-compatible input to client
        response.json             # OpenAI-compatible output from client
        streaming_chunks.jsonl    # If streaming mode
        metadata.json             # Timing, usage, model, provider, etc.
        provider/                 # Provider-specific subdirectory (optional)
            request_payload.json  # Transformed request to provider API
            response_stream.log   # Raw streaming chunks from provider
            final_response.json   # Raw provider response
            error.log             # If any errors occurred
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .transform_trace import TransformTraceWriter
from .utils.paths import get_logs_dir

lib_logger = logging.getLogger("rotator_library")

FRAMEWORK_KEYS = frozenset({
    "api_key",
    "api_base",
    "custom_llm_provider",
    "transaction_context",
    "credential_identifier",
})


def _strip_framework_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if k not in FRAMEWORK_KEYS}


def _get_transactions_dir() -> Path:
    """Get the transactions log directory, creating it if needed."""
    logs_dir = get_logs_dir()
    transactions_dir = logs_dir / "transactions"
    transactions_dir.mkdir(parents=True, exist_ok=True)
    return transactions_dir


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for log records."""

    return datetime.now(UTC).isoformat()


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use in directory/file names."""
    # Replace problematic characters with underscores
    for char in '/\\:*?"<>|':
        name = name.replace(char, "_")
    return name


@dataclass
class TransactionContext:
    """
    Lightweight context passed to providers for correlated logging.

    Providers receive this context and can use it to create their own
    loggers that write to the transaction's directory structure.
    """

    log_dir: Path
    """Root directory for this transaction's logs."""

    request_id: str
    """Unique 8-character correlation ID for this transaction."""

    enabled: bool
    """Whether logging is enabled."""

    provider: str
    """Provider name (e.g., 'gemini_cli', 'openai')."""

    model: str
    """Model name (sanitized for filesystem use)."""

    trace_enabled: bool = False
    """Whether provider loggers should append transform trace entries."""


class TransactionLogger:
    """
    Logs complete API transactions at the client.py layer.

    Creates a unique directory for each transaction and logs:
    - OpenAI-compatible request (what client.py receives)
    - OpenAI-compatible response (what client.py returns)
    - Streaming chunks (if streaming mode)
    - Metadata (timing, usage, model info)

    Also provides a TransactionContext that can be passed to providers
    for correlated provider-level logging.
    """

    __slots__ = (
        "enabled",
        "log_dir",
        "start_time",
        "request_id",
        "provider",
        "model",
        "streaming",
        "api_format",
        "_dir_available",
        "_context",
        "_trace_writer",
    )

    def __init__(
        self,
        provider: str,
        model: str,
        enabled: bool = True,
        api_format: str = "oai",
        parent_dir: Optional[Path] = None,
    ):
        """
        Initialize transaction logger.

        Args:
            provider: Provider name (e.g., 'gemini_cli', 'openai')
            model: Model name (will be sanitized for filesystem)
            enabled: Whether logging is enabled
            api_format: API format prefix ('oai' for OpenAI, 'ant' for Anthropic)
            parent_dir: Optional parent directory for nested logging
        """
        self.enabled = enabled
        self.start_time = time.time()
        self.request_id = str(uuid.uuid4())[:8]  # 8-char short ID
        self.provider = provider
        self.api_format = api_format

        # Strip provider prefix from model if present
        # e.g., "gemini_cli/gemini-2.5-pro" -> "gemini-2.5-pro"
        model_name = model
        if "/" in model_name and model_name.split("/")[0] == provider:
            model_name = model_name.split("/", 1)[1]

        self.model = _sanitize_name(model_name)
        self.streaming = False
        self.log_dir: Optional[Path] = None
        self._dir_available = False
        self._context: Optional[TransactionContext] = None
        self._trace_writer: Optional[TransformTraceWriter] = None

        if not enabled:
            return

        # Create directory based on whether we have a parent directory
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        safe_provider = _sanitize_name(provider)

        if parent_dir:
            # Nested logging: create subdirectory inside parent
            # e.g., parent_dir/openai/ for OpenAI translation layer
            subdir_name = "openai" if api_format == "oai" else api_format
            self.log_dir = parent_dir / subdir_name
        else:
            # Root-level logging: MMDD_HHMMSS_{api_format}_{provider}_{model}_{request_id}
            dir_name = f"{timestamp}_{api_format}_{safe_provider}_{self.model}_{self.request_id}"
            self.log_dir = _get_transactions_dir() / dir_name

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._dir_available = True
            self._trace_writer = TransformTraceWriter(
                self.log_dir,
                component="client",
                provider=provider,
                model=self.model,
                enabled=True,
            )
        except Exception as e:
            lib_logger.error(f"TransactionLogger: Failed to create directory: {e}")
            self.enabled = False

    def get_context(self) -> TransactionContext:
        """
        Get the transaction context for passing to providers.

        Returns a lightweight dataclass that providers can use to create
        their own loggers with correlated directory structure.
        """
        if self._context is None:
            self._context = TransactionContext(
                log_dir=self.log_dir if self.log_dir else Path("."),
                request_id=self.request_id,
                enabled=self.enabled,
                provider=self.provider,
                model=self.model,
                trace_enabled=bool(self._trace_writer),
            )
        return self._context

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
        snapshot: bool = True,
    ) -> None:
        """Record an additive transform trace entry if tracing is available."""

        if not self.enabled or not self._dir_available or not self._trace_writer:
            return
        self._trace_writer.record(
            pass_name,
            data,
            direction=direction,
            stage=stage,
            protocol=protocol,
            credential_id=credential_id,
            transport=transport,
            changed_from_previous=changed_from_previous,
            metadata=metadata,
            snapshot=snapshot,
        )

    def log_request(
        self, request_data: Dict[str, Any], filename: str = "request.json"
    ) -> None:
        """
        Log the request received by client.py.

        Args:
            request_data: The request data dict (messages, model, etc.)
            filename: Custom filename for the log file (default: request.json)
        """
        if not self.enabled or not self._dir_available:
            return

        self.streaming = request_data.get("stream", False)

        data = {
            "request_id": self.request_id,
            "timestamp_utc": _utc_timestamp(),
            "data": request_data,
        }
        self.log_transform_pass(
            "raw_client_request",
            request_data,
            direction="request",
            stage="client",
            transport="sse" if self.streaming else "http",
        )
        self._write_json(filename, data)

    def log_transformed_request(
        self,
        transformed_data: Dict[str, Any],
        original_data: Dict[str, Any],
    ) -> None:
        """
        Log the transformed request if it differs from the original.

        Compares the two payloads after stripping framework infrastructure
        keys (api_key, api_base, custom_llm_provider, etc.).  If the
        remaining content is identical, nothing is written.

        Args:
            transformed_data: The kwargs after all transforms/sanitization.
            original_data: The original kwargs as received by the client.
        """
        if not self.enabled or not self._dir_available:
            return

        stripped_transformed = _strip_framework_keys(transformed_data)
        stripped_original = _strip_framework_keys(original_data)

        changed_from_previous: Optional[bool] = None
        try:
            changed_from_previous = json.dumps(stripped_transformed, sort_keys=True, default=str) != json.dumps(
                stripped_original, sort_keys=True, default=str
            )
        except (TypeError, ValueError):
            changed_from_previous = None

        self.log_transform_pass(
            "prepared_provider_request",
            transformed_data,
            direction="request",
            stage="client",
            transport="sse" if transformed_data.get("stream") else "http",
            changed_from_previous=changed_from_previous,
        )

        if changed_from_previous is False:
            return

        logged = _strip_framework_keys(transformed_data)
        data = {
            "request_id": self.request_id,
            "timestamp_utc": _utc_timestamp(),
            "data": logged,
        }
        self._write_json("request_transformed.json", data)

    def log_stream_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Log an individual chunk from a streaming response.

        Args:
            chunk: The streaming chunk data
        """
        if not self.enabled or not self._dir_available:
            return

        log_entry = {
            "timestamp_utc": _utc_timestamp(),
            "chunk": chunk,
        }
        self.log_transform_pass(
            "parsed_stream_chunk",
            chunk,
            direction="stream",
            stage="client",
            transport="sse",
            snapshot=False,
        )
        content = json.dumps(log_entry, ensure_ascii=False) + "\n"
        self._append_text("streaming_chunks.jsonl", content)

    def log_response(
        self,
        response_data: Dict[str, Any],
        status_code: int = 200,
        headers: Optional[Dict[str, Any]] = None,
        filename: str = "response.json",
    ) -> None:
        """
        Log the response returned by client.py.

        Args:
            response_data: The response data dict
            status_code: HTTP status code (default 200)
            headers: Optional response headers
            filename: Custom filename for the log file (default: response.json)
        """
        if not self.enabled or not self._dir_available:
            return

        end_time = time.time()
        duration_ms = (end_time - self.start_time) * 1000

        data = {
            "request_id": self.request_id,
            "timestamp_utc": _utc_timestamp(),
            "status_code": status_code,
            "duration_ms": round(duration_ms),
            "headers": dict(headers) if headers else None,
            "data": response_data,
        }
        self.log_transform_pass(
            "final_client_response",
            response_data,
            direction="response",
            stage="final",
            transport="sse" if self.streaming else "http",
            metadata={"status_code": status_code, "headers": dict(headers) if headers else None},
        )
        self._write_json(filename, data)

        # Also write metadata
        self._log_metadata(response_data, status_code, duration_ms)

    def _log_metadata(
        self, response_data: Dict[str, Any], status_code: int, duration_ms: float
    ) -> None:
        """Log transaction metadata summary."""
        usage = response_data.get("usage") or {}
        model = response_data.get("model", self.model)
        finish_reason = "N/A"

        if "choices" in response_data and response_data["choices"]:
            finish_reason = response_data["choices"][0].get("finish_reason", "N/A")

        # Check for provider subdirectory
        has_provider_logs = False
        if self.log_dir:
            provider_dir = self.log_dir / "provider"
            try:
                has_provider_logs = provider_dir.exists() and any(
                    provider_dir.iterdir()
                )
            except OSError:
                has_provider_logs = False

        metadata = {
            "request_id": self.request_id,
            "timestamp_utc": _utc_timestamp(),
            "duration_ms": round(duration_ms),
            "status_code": status_code,
            "provider": self.provider,
            "model": model,
            "streaming": self.streaming,
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "finish_reason": finish_reason,
            "has_provider_logs": has_provider_logs,
            "reasoning_found": False,
            "reasoning_content": None,
        }

        # Extract reasoning if present
        reasoning = self._extract_reasoning(response_data)
        if reasoning:
            metadata["reasoning_found"] = True
            metadata["reasoning_content"] = reasoning

        self._write_json("metadata.json", metadata)

    def _extract_reasoning(self, response_data: Dict[str, Any]) -> Optional[str]:
        """Recursively search for and extract 'reasoning' fields from response."""
        if not isinstance(response_data, dict):
            return None

        if "reasoning" in response_data:
            return response_data["reasoning"]

        if "choices" in response_data and response_data["choices"]:
            message = response_data["choices"][0].get("message", {})
            if "reasoning" in message:
                return message["reasoning"]
            if "reasoning_content" in message:
                return message["reasoning_content"]

        return None

    def _write_json(self, filename: str, data: Dict[str, Any]) -> None:
        """Write JSON data to a file in the log directory."""
        if not self.log_dir:
            return
        try:
            with open(self.log_dir / filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            lib_logger.error(f"TransactionLogger: Failed to write {filename}: {e}")

    def _append_text(self, filename: str, text: str) -> None:
        """Append text to a file in the log directory."""
        if not self.log_dir:
            return
        try:
            with open(self.log_dir / filename, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            lib_logger.error(f"TransactionLogger: Failed to append to {filename}: {e}")

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

        final_message: Dict[str, Any] = {"role": "assistant"}
        aggregated_tool_calls: Dict[int, Dict[str, Any]] = {}
        usage_data = None
        finish_reason = None

        for chunk in chunks:
            if "choices" in chunk and chunk["choices"]:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                # Dynamically aggregate all fields from the delta
                for key, value in delta.items():
                    if value is None:
                        continue

                    if key == "content":
                        if "content" not in final_message:
                            final_message["content"] = ""
                        if value:
                            final_message["content"] += value

                    elif key == "tool_calls":
                        for tc_chunk in value:
                            index = tc_chunk.get("index", 0)
                            if index not in aggregated_tool_calls:
                                aggregated_tool_calls[index] = {
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            if "function" not in aggregated_tool_calls[index]:
                                aggregated_tool_calls[index]["function"] = {
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc_chunk.get("id"):
                                aggregated_tool_calls[index]["id"] = tc_chunk["id"]
                            if "function" in tc_chunk:
                                if "name" in tc_chunk["function"]:
                                    if tc_chunk["function"]["name"] is not None:
                                        aggregated_tool_calls[index]["function"][
                                            "name"
                                        ] += tc_chunk["function"]["name"]
                                if "arguments" in tc_chunk["function"]:
                                    if tc_chunk["function"]["arguments"] is not None:
                                        aggregated_tool_calls[index]["function"][
                                            "arguments"
                                        ] += tc_chunk["function"]["arguments"]

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

                    else:  # Generic key handling for other data like 'reasoning'
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
                            # Provider-specific streamed fields can change type;
                            # keep logging robust by taking the latest value.
                            final_message[key] = value

                if "finish_reason" in choice and choice["finish_reason"]:
                    finish_reason = choice["finish_reason"]

            if "usage" in chunk and chunk["usage"]:
                usage_data = chunk["usage"]

        # Final Response Construction
        if aggregated_tool_calls:
            final_message["tool_calls"] = list(aggregated_tool_calls.values())
            finish_reason = "tool_calls"

        # Ensure standard fields are present
        for field in ["content", "tool_calls", "function_call"]:
            if field not in final_message:
                final_message[field] = None

        first_chunk = chunks[0]
        final_choice = {
            "index": 0,
            "message": final_message,
            "finish_reason": finish_reason,
        }

        full_response = {
            "id": first_chunk.get("id"),
            "object": "chat.completion",
            "created": first_chunk.get("created"),
            "model": first_chunk.get("model"),
            "choices": [final_choice],
            "usage": usage_data,
        }

        return full_response


class ProviderLogger:
    """
    Base class for provider-specific logging.

    Logs provider-level request/response data to a subdirectory of the
    transaction's log directory. Providers can extend this class to add
    custom logging methods.

    Default behavior:
    - Creates a 'provider/' subdirectory in the transaction log
    - Logs request payload, response chunks, final response, and errors

    Providers can override __init__ to use a different directory structure,
    or add custom methods for provider-specific logging needs.
    """

    __slots__ = ("enabled", "log_dir", "_trace_writer")

    def __init__(self, context: Optional[TransactionContext]):
        """
        Initialize provider logger from transaction context.

        Args:
            context: TransactionContext from TransactionLogger, or None to disable
        """
        self.enabled = False
        self.log_dir: Optional[Path] = None
        self._trace_writer: Optional[TransformTraceWriter] = None

        if context is None or not context.enabled:
            return

        self.enabled = True
        self.log_dir = context.log_dir / "provider"

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            if getattr(context, "trace_enabled", False):
                self._trace_writer = TransformTraceWriter(
                    context.log_dir,
                    component="provider",
                    provider=context.provider,
                    model=context.model,
                    enabled=True,
                )
        except Exception as e:
            lib_logger.error(f"ProviderLogger: Failed to create directory: {e}")
            self.enabled = False

    def _log_transform_pass(
        self,
        pass_name: str,
        data: Any,
        *,
        direction: str,
        stage: str = "provider",
        transport: Optional[str] = None,
        snapshot: bool = True,
    ) -> None:
        if not self.enabled or not self._trace_writer:
            return
        self._trace_writer.record(
            pass_name,
            data,
            direction=direction,
            stage=stage,
            transport=transport,
            snapshot=snapshot,
        )

    def log_request(self, payload: Dict[str, Any]) -> None:
        """
        Log the request payload sent to the provider API.

        Args:
            payload: The transformed request payload
        """
        self._log_transform_pass(
            "provider_request_payload",
            payload,
            direction="request",
            transport="http",
        )
        self._write_json("request_payload.json", payload)

    def log_response_chunk(self, chunk: str) -> None:
        """
        Log a raw chunk from the provider's response stream.

        Args:
            chunk: Raw chunk string from the stream
        """
        self._log_transform_pass(
            "provider_raw_stream_chunk",
            chunk,
            direction="stream",
            transport="sse",
            snapshot=False,
        )
        self._append_text("response_stream.log", chunk + "\n")

    def log_final_response(self, response_data: Dict[str, Any]) -> None:
        """
        Log the final, reassembled response from the provider.

        Args:
            response_data: The complete response data
        """
        self._log_transform_pass(
            "provider_final_response",
            response_data,
            direction="response",
        )
        self._write_json("final_response.json", response_data)

    def log_error(self, error_message: str) -> None:
        """
        Log an error message with timestamp.

        Args:
            error_message: The error message to log
        """
        timestamp = _utc_timestamp()
        self._log_transform_pass(
            "provider_error",
            {"timestamp_utc": timestamp, "message": error_message},
            direction="error",
            snapshot=False,
        )
        self._append_text("error.log", f"[{timestamp}] {error_message}\n")

    def log_extra(self, filename: str, data: Union[Dict[str, Any], str]) -> None:
        """
        Log arbitrary data to a custom file.

        Allows providers to log additional files without subclassing.

        Args:
            filename: Name of the file to write
            data: Either a dict (written as JSON) or string (written as text)
        """
        if isinstance(data, dict):
            self._write_json(filename, data)
        else:
            self._append_text(filename, data)

    def _write_json(self, filename: str, data: Dict[str, Any]) -> None:
        """Write JSON data to a file in the log directory."""
        if not self.enabled or not self.log_dir:
            return
        try:
            with open(self.log_dir / filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            lib_logger.error(f"ProviderLogger: Failed to write {filename}: {e}")

    def _append_text(self, filename: str, text: str) -> None:
        """Append text to a file in the log directory."""
        if not self.enabled or not self.log_dir:
            return
        try:
            with open(self.log_dir / filename, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            lib_logger.error(f"ProviderLogger: Failed to append to {filename}: {e}")

