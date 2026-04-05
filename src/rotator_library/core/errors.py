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


__all__ = [
    # Exception classes
    "NoAvailableKeysError",
    "PreRequestCallbackError",
    "CredentialNeedsReauthError",
    "EmptyResponseError",
    "TransientQuotaError",
    "StreamedAPIError",
    "TerminalRequestError",
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
