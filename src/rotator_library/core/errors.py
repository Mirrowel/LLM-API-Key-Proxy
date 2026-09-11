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
    is_context_window_error_text,
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
        headers: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.response = response or {}
        self.headers = headers or {}

    @property
    def http_status(self) -> int:
        if self.status_code and 400 <= self.status_code <= 599:
            return self.status_code
        return {
            "authentication": 401,
            "forbidden": 403,
            "not_found": 404,
            "conflict": 409,
            "request_too_large": 413,
            "rate_limit": 429,
            "quota_exceeded": 429,
            "invalid_request": 400,
            "context_window_exceeded": 400,
            "server_error": 502,
            "proxy_timeout": 504,
            "proxy_all_credentials_exhausted": 503,
        }.get(self.error_type, 502)

    def to_protocol_payload(self, protocol: str) -> dict:
        """Format one terminal provider error in the selected client protocol.

        Vocabulary per docs/experimental/error-reference.md 5.10 — official
        enums only. The anthropic branch is status-aware so 504 renders as
        timeout_error and 529 as overloaded_error even though both classify
        as the internal server_error family.
        """

        message = str(self)
        if protocol == "anthropic_messages":
            error_type = {
                "authentication": "authentication_error",
                "forbidden": "permission_error",
                "rate_limit": "rate_limit_error",
                "quota_exceeded": "rate_limit_error",
                "invalid_request": "invalid_request_error",
                "context_window_exceeded": "invalid_request_error",
                "not_found": "not_found_error",
                "conflict": "conflict_error",
                "request_too_large": "request_too_large",
            }.get(self.error_type, "api_error")
            if self.error_type in ("server_error", "proxy_timeout", "api_connection"):
                if self.http_status == 504:
                    error_type = "timeout_error"
                elif self.http_status == 529:
                    error_type = "overloaded_error"
                else:
                    error_type = "api_error"
            return {"type": "error", "error": {"type": error_type, "message": message}}
        if protocol == "gemini":
            status = {
                "authentication": "UNAUTHENTICATED",
                "forbidden": "PERMISSION_DENIED",
                "rate_limit": "RESOURCE_EXHAUSTED",
                "quota_exceeded": "RESOURCE_EXHAUSTED",
                "invalid_request": "INVALID_ARGUMENT",
                "context_window_exceeded": "INVALID_ARGUMENT",
                "not_found": "NOT_FOUND",
                "conflict": "ALREADY_EXISTS",
                "request_too_large": "INVALID_ARGUMENT",
                "proxy_timeout": "DEADLINE_EXCEEDED",
                "proxy_all_credentials_exhausted": "UNAVAILABLE",
            }.get(self.error_type, "INTERNAL")
            if self.error_type in ("server_error", "api_connection"):
                if self.http_status == 529 or self.http_status == 503:
                    status = "UNAVAILABLE"
                elif self.http_status == 504:
                    status = "DEADLINE_EXCEEDED"
            return {"error": {"code": self.http_status, "message": message, "status": status}}
        # openai_chat / responses default branch: official error-object
        # vocabulary (invalid_request_error, rate_limit_error, ...) with a
        # separate machine `code` and the spec's `param` key.
        type_map = {
            "invalid_request": "invalid_request_error",
            "context_window_exceeded": "invalid_request_error",
            "request_too_large": "invalid_request_error",
            "authentication": "authentication_error",
            "forbidden": "permission_error",
            "not_found": "not_found_error",
            "rate_limit": "rate_limit_error",
            "quota_exceeded": "insufficient_quota",
            "server_error": "server_error",
            "proxy_timeout": "server_error",
            "api_connection": "server_error",
            "proxy_all_credentials_exhausted": "server_error",
        }
        return {
            "error": {
                "message": message,
                "type": type_map.get(self.error_type, "server_error"),
                "param": None,
                "code": self.error_type,
            }
        }


def structured_api_response_error(
    response,
    *,
    headers: dict | None = None,
) -> StructuredAPIResponseError | None:
    """Normalize top-level provider error envelopes across execution modes.

    Classification shares the grounded evidence order with
    ``classify_error`` (docs/experimental/error-reference.md 5.9): quota
    markers at any status first, then structured wire vocabulary, then the
    status ladder. Free-text participates only in the two sanctioned narrow
    checks (context spellings; quota tokens at 400/429) — never bare
    substring tokens like "rate".
    """

    from ..error_handler import (
        _classify_structured_error_text,
        _structured_quota_signal,
    )

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
    message = str(details.get("message") or "")

    if _structured_quota_signal(details, {}, message, status_code):
        error_type = "quota_exceeded"
    else:
        structured_type = _classify_structured_error_text(details, {})
        if structured_type:
            error_type = structured_type
        elif status_code == 401:
            error_type = "authentication"
        elif status_code == 403:
            error_type = "forbidden"
        elif status_code == 404:
            error_type = "not_found"
        elif status_code == 409:
            error_type = "conflict"
        elif status_code == 413:
            error_type = "request_too_large"
        elif status_code == 429:
            error_type = "rate_limit"
        elif is_context_window_error_text(message):
            error_type = "context_window_exceeded"
        elif (status_code is not None and status_code >= 500) or str(
            details.get("status") or ""
        ).upper() in {"INTERNAL", "UNAVAILABLE", "DEADLINE_EXCEEDED"}:
            error_type = "server_error"
        else:
            error_type = "invalid_request"
    message = str(details.get("message") or details.get("status") or value or "Provider returned a structured error response")
    return StructuredAPIResponseError(
        message,
        error_type=error_type,
        status_code=status_code,
        response=response,
        headers=headers,
    )


def is_structured_error_payload(response) -> bool:
    """Return whether a value is an explicit top-level API error envelope."""

    return isinstance(response, dict) and "error" in response and response.get("error") not in (None, "", False)


def protocol_error_payload(
    error: BaseException | str,
    protocol: str,
    *,
    error_type: str,
    status_code: int,
) -> tuple[int, dict]:
    """Render one proxy-side terminal failure in the selected client protocol."""

    structured = StructuredAPIResponseError(
        str(error),
        error_type=error_type,
        status_code=status_code,
    )
    return structured.http_status, structured.to_protocol_payload(protocol)


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
    "protocol_error_payload",
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
