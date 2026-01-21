# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import logging
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import Optional, Union

from .error_handler import mask_credential
from .utils.paths import get_logs_dir

# =============================================================================
# CONFIGURATION DEFAULTS
# =============================================================================

# Maximum size of the failure log file before rotation (in bytes)
# Default: 5 MB
FAILURE_LOG_MAX_SIZE: int = 5 * 1024 * 1024

# Number of backup log files to keep
FAILURE_LOG_BACKUP_COUNT: int = 2

# Maximum characters per individual error message in the chain
FAILURE_LOG_ERROR_MESSAGE_LIMIT: int = 2000

# Maximum error chain length to prevent excessive nesting
FAILURE_LOG_ERROR_CHAIN_LIMIT: int = 5

# Maximum length of full error message in detailed log
FAILURE_LOG_FULL_MESSAGE_LIMIT: int = 5000

# Maximum length of raw response in detailed log
FAILURE_LOG_RAW_RESPONSE_LIMIT: int = 10000


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logs."""

    def format(self, record):
        # The message is already a dict, so we just format it as a JSON string
        return json.dumps(record.msg)


# Module-level state for lazy initialization
_failure_logger: Optional[logging.Logger] = None
_configured_logs_dir: Optional[Path] = None


def configure_failure_logger(logs_dir: Optional[Union[Path, str]] = None) -> None:
    """
    Configure the failure logger to use a specific logs directory.

    Call this before first use if you want to override the default location.
    If not called, the logger will use get_logs_dir() on first use.

    Args:
        logs_dir: Path to the logs directory. If None, uses get_logs_dir().
    """
    global _configured_logs_dir, _failure_logger
    _configured_logs_dir = Path(logs_dir) if logs_dir else None
    # Reset logger so it gets reconfigured on next use
    _failure_logger = None


def _setup_failure_logger(logs_dir: Path) -> logging.Logger:
    """
    Sets up a dedicated JSON logger for writing detailed failure logs to a file.

    Args:
        logs_dir: Path to the logs directory.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("failure_logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers to prevent duplicates on re-setup
    logger.handlers.clear()

    try:
        logs_dir.mkdir(parents=True, exist_ok=True)

        handler = RotatingFileHandler(
            logs_dir / "failures.log",
            maxBytes=FAILURE_LOG_MAX_SIZE,
            backupCount=FAILURE_LOG_BACKUP_COUNT,
        )
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    except (OSError, PermissionError, IOError) as e:
        logging.warning(f"Cannot create failure log file handler: {e}")
        # Add NullHandler to prevent "no handlers" warning
        logger.addHandler(logging.NullHandler())

    return logger


def get_failure_logger() -> logging.Logger:
    """
    Get the failure logger, initializing it lazily if needed.

    Returns:
        The configured failure logger.
    """
    global _failure_logger, _configured_logs_dir

    if _failure_logger is None:
        logs_dir = _configured_logs_dir if _configured_logs_dir else get_logs_dir()
        _failure_logger = _setup_failure_logger(logs_dir)

    return _failure_logger


# Get the main library logger for concise, propagated messages
main_lib_logger = logging.getLogger("rotator_library")


def _extract_response_body(error: Exception) -> str:
    """
    Extract the full response body from various error types.

    Handles:
    - StreamedAPIError: wraps original exception in .data attribute
    - httpx.HTTPStatusError: response.text or response.content
    - litellm exceptions: various response attributes
    - Other exceptions: str(error)
    """
    # Handle StreamedAPIError which wraps the original exception in .data
    # This is used by our streaming wrapper when catching provider errors
    if hasattr(error, "data") and error.data is not None:
        inner = error.data
        # If data is a dict (parsed JSON error), return it as JSON
        if isinstance(inner, dict):
            try:
                return json.dumps(inner, indent=2)
            except Exception:
                return str(inner)
        # If data is an exception, recurse to extract from it
        if isinstance(inner, Exception):
            result = _extract_response_body(inner)
            if result:
                return result

    # Try to get response body from httpx errors
    if hasattr(error, "response") and error.response is not None:
        response = error.response
        # Try .text first (decoded)
        if hasattr(response, "text") and response.text:
            return response.text
        # Try .content (bytes)
        if hasattr(response, "content") and response.content:
            try:
                return response.content.decode("utf-8", errors="replace")
            except Exception:
                return str(response.content)

    # Check for litellm's body attribute
    if hasattr(error, "body") and error.body:
        return str(error.body)

    # Check for message attribute that might contain response
    if hasattr(error, "message") and error.message:
        return str(error.message)

    return None


def log_failure(
    api_key: str,
    model: str,
    attempt: int,
    error: Exception,
    request_headers: dict,
    raw_response_text: str = None,
):
    """
    Logs a detailed failure message to a file and a concise summary to the main logger.

    Args:
        api_key: The API key or credential path that was used
        model: The model that was requested
        attempt: The attempt number (1-based)
        error: The exception that occurred
        request_headers: Headers from the original request
        raw_response_text: Optional pre-extracted response body (e.g., from streaming)
    """
    # 1. Log the full, detailed error to the dedicated failures.log file
    # Prioritize the explicitly passed raw response text, as it may contain
    # reassembled data from a stream that is not available on the exception object.
    raw_response = raw_response_text
    if not raw_response:
        raw_response = _extract_response_body(error)

    # Get full error message (not truncated)
    full_error_message = str(error)

    # Also capture any nested/wrapped exception info
    error_chain = []
    visited = set()  # Track visited exceptions to detect circular references
    current_error = error
    while current_error:
        # Check for circular references
        error_id = id(current_error)
        if error_id in visited:
            break
        visited.add(error_id)

        error_chain.append(
            {
                "type": type(current_error).__name__,
                "message": str(current_error)[:FAILURE_LOG_ERROR_MESSAGE_LIMIT],
            }
        )
        current_error = getattr(current_error, "__cause__", None) or getattr(
            current_error, "__context__", None
        )
        if len(error_chain) > FAILURE_LOG_ERROR_CHAIN_LIMIT:
            break

    detailed_log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "api_key_ending": mask_credential(api_key),
        "model": model,
        "attempt_number": attempt,
        "error_type": type(error).__name__,
        "error_message": full_error_message[:FAILURE_LOG_FULL_MESSAGE_LIMIT],
        "raw_response": raw_response[:FAILURE_LOG_RAW_RESPONSE_LIMIT]
        if raw_response
        else None,
        "request_headers": request_headers,
        "error_chain": error_chain if len(error_chain) > 1 else None,
    }

    # 2. Log a concise summary to the main library logger, which will propagate
    summary_message = (
        f"API call failed for model {model} with key {mask_credential(api_key)}. "
        f"Error: {type(error).__name__}. See failures.log for details."
    )

    # Log to failure logger with resilience - if it fails, just continue
    try:
        get_failure_logger().error(detailed_log_data)
    except (OSError, IOError) as e:
        # Log file write failed - log to console instead
        logging.warning(f"Failed to write to failures.log: {e}")

    # Console log always succeeds
    main_lib_logger.error(summary_message)
