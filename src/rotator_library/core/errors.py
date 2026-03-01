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

from typing import Any, Dict, Optional

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


class StreamBootstrapError(Exception):
    """
    Raised when a streaming request fails before the first chunk is emitted.

    This lets the HTTP layer return a proper non-200 response (e.g. HTTP 429)
    instead of committing an SSE stream and sending an in-band error payload.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_payload: Optional[Dict[str, Any]] = None,
        retry_after: Optional[int] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_payload = error_payload or {
            "error": {
                "message": message,
                "type": "proxy_error",
            }
        }
        self.retry_after = retry_after


__all__ = [
    # Exception classes
    "NoAvailableKeysError",
    "PreRequestCallbackError",
    "CredentialNeedsReauthError",
    "EmptyResponseError",
    "TransientQuotaError",
    "StreamedAPIError",
    "StreamBootstrapError",
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
