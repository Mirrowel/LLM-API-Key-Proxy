# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Error handling for the rotator library.

This module re-exports all exception classes and error handling utilities
from the main error_handler module, and adds any new error types needed
for the refactored architecture.

Note: The actual implementations remain in error_handler.py for backward
compatibility. This module provides a cleaner import path.
"""

# Re-export everything from error_handler
from ..error_handler import (
    # Exception classes
    NoAvailableKeysError,
    PreRequestCallbackError,
    CredentialNeedsReauthError,
    EmptyResponseError,
    TransientQuotaError,
    # Error classification
    ClassifiedError,
    RequestErrorAccumulator,
    classify_error,
    should_rotate_on_error,
    should_retry_same_key,
    is_abnormal_error,
    # Utilities
    mask_credential,
    get_retry_after,
    extract_retry_after_from_body,
    is_rate_limit_error,
    is_server_error,
    is_unrecoverable_error,
    # Constants
    ABNORMAL_ERROR_TYPES,
    NORMAL_ERROR_TYPES,
)


# =============================================================================
# NEW EXCEPTIONS FOR REFACTORED ARCHITECTURE
# =============================================================================


class StreamedAPIError(Exception):
    """
    Custom exception to signal an API error received over a stream.

    This is raised when an error is detected in streaming response data,
    allowing the retry logic to handle it appropriately.

    Attributes:
        message: Human-readable error message
        data: The parsed error data (dict or exception)
    """

    def __init__(self, message: str, data=None):
        super().__init__(message)
        self.data = data


class TerminalRequestError(Exception):
    """
    Sentinel exception that wraps a non-rotatable error (e.g. 404 NOT_FOUND).

    When the executor classifies an error as non-rotatable (invalid_request,
    context_window_exceeded, etc.) and raises the original exception, that raise
    happens inside an `async with acquire_credential()` block whose cleanup is
    wrapped by a broad `except Exception: pass`. That bare except swallows the
    re-raise, causing the rotator to silently move on to the next credential
    even though the error is model-level (not credential-level).

    Wrapping in TerminalRequestError lets executor.py catch it *before* the
    swallowing clause and propagate it correctly.
    """

    def __init__(self, original: Exception):
        super().__init__(str(original))
        self.original = original


class ProxyExhaustionError(Exception):
    """
    Raised by the executor when all credentials for a provider are exhausted
    (or a TerminalRequestError occurs that maps to a specific HTTP status).

    Carries the structured error response dict (ready for JSON serialization)
    and the dominant upstream error type so that main.py can pick the correct
    HTTP status code without duck-typing on the return value.

    HTTP status mapping (used in main.py):
      context_window_exceeded / invalid_request  -> 400
      authentication                             -> 401
      forbidden                                  -> 403
      rate_limit / quota_exceeded                -> 429
      timeout                                    -> 504
      server_error / api_connection / other      -> 502
    """

    # Maps dominant upstream error type to the HTTP status code the proxy should return.
    _CODE_TO_HTTP_STATUS: dict = {
        "context_window_exceeded": 400,
        "invalid_request": 400,
        "authentication": 401,
        "forbidden": 403,
        "rate_limit": 429,
        "quota_exceeded": 429,
        # server_error / api_connection / unknown -> 502 (default)
    }

    def __init__(self, error_response: dict, dominant_code: str | None = None):
        message = (
            error_response.get("error", {}).get("message", "Proxy exhaustion error")
        )
        super().__init__(message)
        self.error_response = error_response
        self.dominant_code = dominant_code

    @property
    def http_status(self) -> int:
        """Return the appropriate HTTP status code for this exhaustion error."""
        if self.dominant_code in self._CODE_TO_HTTP_STATUS:
            return self._CODE_TO_HTTP_STATUS[self.dominant_code]
        # Timeout: check the details flag when there is no dominant error code
        if self.error_response.get("error", {}).get("details", {}).get("timeout"):
            return 504
        return 502


__all__ = [
    # Exception classes
    "NoAvailableKeysError",
    "PreRequestCallbackError",
    "CredentialNeedsReauthError",
    "EmptyResponseError",
    "TransientQuotaError",
    "StreamedAPIError",
    "TerminalRequestError",
    "ProxyExhaustionError",
    # Error classification
    "ClassifiedError",
    "RequestErrorAccumulator",
    "classify_error",
    "should_rotate_on_error",
    "should_retry_same_key",
    "is_abnormal_error",
    # Utilities
    "mask_credential",
    "get_retry_after",
    "extract_retry_after_from_body",
    "is_rate_limit_error",
    "is_server_error",
    "is_unrecoverable_error",
    # Constants
    "ABNORMAL_ERROR_TYPES",
    "NORMAL_ERROR_TYPES",
]
