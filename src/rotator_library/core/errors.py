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


class StructuredAPIResponseError(Exception):
    """Raise a structured provider error before success-format conversion."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        status_code: int | None = None,
        response: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.response = response or {}

    @property
    def http_status(self) -> int:
        if self.status_code and 400 <= self.status_code <= 599:
            return self.status_code
        return {
            "authentication": 401,
            "forbidden": 403,
            "rate_limit": 429,
            "quota_exceeded": 429,
            "invalid_request": 400,
            "server_error": 502,
        }.get(self.error_type, 502)

    def to_protocol_payload(self, protocol: str) -> dict:
        """Format one terminal provider error in the selected client protocol."""

        message = str(self)
        if protocol == "anthropic_messages":
            error_type = {
                "authentication": "authentication_error",
                "forbidden": "permission_error",
                "rate_limit": "rate_limit_error",
                "quota_exceeded": "rate_limit_error",
                "invalid_request": "invalid_request_error",
            }.get(self.error_type, "api_error")
            return {"type": "error", "error": {"type": error_type, "message": message}}
        if protocol == "gemini":
            status = {
                "authentication": "UNAUTHENTICATED",
                "forbidden": "PERMISSION_DENIED",
                "rate_limit": "RESOURCE_EXHAUSTED",
                "quota_exceeded": "RESOURCE_EXHAUSTED",
                "invalid_request": "INVALID_ARGUMENT",
            }.get(self.error_type, "INTERNAL")
            return {"error": {"code": self.http_status, "message": message, "status": status}}
        return {
            "error": {
                "message": message,
                "type": self.error_type,
                "code": self.error_type,
            }
        }


def structured_api_response_error(response) -> StructuredAPIResponseError | None:
    """Normalize top-level provider error envelopes across execution modes."""

    if not isinstance(response, dict) or "error" not in response or response.get("error") in (None, "", False):
        return None
    value = response.get("error")
    details = value if isinstance(value, dict) else {"message": str(value)}
    raw_status = next(
        (
            candidate
            for candidate in (
                details.get("status_code"),
                details.get("code"),
                details.get("status"),
                response.get("status_code"),
                response.get("status"),
            )
            if candidate is not None
        ),
        None,
    )
    try:
        status_code = int(raw_status)
    except (TypeError, ValueError):
        status_code = None
    descriptor = " ".join(
        str(details.get(key) or "")
        for key in ("type", "status", "code", "message")
    ).lower()
    if status_code == 429 or any(token in descriptor for token in ("rate", "quota", "resource_exhausted")):
        error_type = "quota_exceeded" if "quota" in descriptor or "resource_exhausted" in descriptor else "rate_limit"
    elif status_code == 401 or "auth" in descriptor or "unauthorized" in descriptor or "unauthenticated" in descriptor:
        error_type = "authentication"
    elif status_code == 403 or "forbidden" in descriptor or "permission_denied" in descriptor:
        error_type = "forbidden"
    elif (status_code is not None and status_code >= 500) or any(token in descriptor for token in ("server", "unavailable", "internal")):
        error_type = "server_error"
    else:
        error_type = "invalid_request"
    message = str(details.get("message") or details.get("status") or value or "Provider returned a structured error response")
    return StructuredAPIResponseError(
        message,
        error_type=error_type,
        status_code=status_code,
        response=response,
    )


def is_structured_error_payload(response) -> bool:
    """Return whether a value is an explicit top-level API error envelope."""

    return isinstance(response, dict) and "error" in response and response.get("error") not in (None, "", False)


__all__ = [
    # Exception classes
    "NoAvailableKeysError",
    "PreRequestCallbackError",
    "CredentialNeedsReauthError",
    "EmptyResponseError",
    "TransientQuotaError",
    "StreamedAPIError",
    "StructuredAPIResponseError",
    "structured_api_response_error",
    "is_structured_error_payload",
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
