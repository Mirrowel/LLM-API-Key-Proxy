import re
import json
import os
import logging
from typing import Optional, Dict, Any
import httpx

from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    AuthenticationError,
    InvalidRequestError,
    BadRequestError,
    OpenAIError,
    InternalServerError,
    Timeout,
    ContextWindowExceededError,
)

lib_logger = logging.getLogger("rotator_library")


def _parse_duration_string(duration_str: str) -> Optional[int]:
    """
    Parse duration strings in various formats to total seconds.

    Handles:
    - Milliseconds: '290.979975ms' -> 1 second (rounds up for sub-second values)
    - Compound durations: '156h14m36.752463453s', '2h30m', '45m30s'
    - Simple durations: '562476.752463453s', '3600s', '60m', '2h'
    - Plain seconds (no unit): '562476'

    Args:
        duration_str: Duration string to parse

    Returns:
        Total seconds as integer, or None if parsing fails.
        For sub-second values, returns at least 1 to avoid retry floods.
    """
    if not duration_str:
        return None

    total_seconds = 0.0
    remaining = duration_str.strip().lower()

    # Try parsing as plain number first (no units)
    try:
        return int(float(remaining))
    except ValueError:
        pass

    # Handle pure milliseconds format: "290.979975ms"
    # MUST check this BEFORE checking 'm' for minutes to avoid misinterpreting 'ms'
    ms_match = re.match(r"^([\d.]+)ms$", remaining)
    if ms_match:
        ms_value = float(ms_match.group(1))
        seconds = ms_value / 1000.0
        # Round up to at least 1 second to avoid immediate retry floods
        return max(1, int(seconds)) if seconds > 0 else 0

    # Parse hours component
    hour_match = re.match(r"(\d+)h", remaining)
    if hour_match:
        total_seconds += int(hour_match.group(1)) * 3600
        remaining = remaining[hour_match.end() :]

    # Parse minutes component - use negative lookahead to avoid matching 'ms'
    min_match = re.match(r"(\d+)m(?!s)", remaining)
    if min_match:
        total_seconds += int(min_match.group(1)) * 60
        remaining = remaining[min_match.end() :]

    # Parse seconds component (including decimals like 36.752463453s)
    sec_match = re.match(r"([\d.]+)s", remaining)
    if sec_match:
        total_seconds += float(sec_match.group(1))

    # For sub-second values, round up to at least 1
    if total_seconds > 0:
        return max(1, int(total_seconds))
    return None


def extract_retry_after_from_body(error_body: Optional[str]) -> Optional[int]:
    """
    Extract the retry-after time from an API error response body.

    Handles various error formats including:
    - Gemini CLI: "Your quota will reset after 39s."
    - Antigravity: "quota will reset after 156h14m36s"
    - Generic: "quota will reset after 120s", "retry after 60s"

    Args:
        error_body: The raw error response body

    Returns:
        The retry time in seconds, or None if not found
    """
    if not error_body:
        return None

    # Pattern to match various "reset after" formats - capture the full duration string
    patterns = [
        r"quota will reset after\s*([\dhmso.]+)",  # Matches compound: 156h14m36s or 120s
        r"reset after\s*([\dhmso.]+)",
        r"retry after\s*([\dhmso.]+)",
        r"try again in\s*(\d+)\s*seconds?",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_body, re.IGNORECASE)
        if match:
            duration_str = match.group(1)
            result = _parse_duration_string(duration_str)
            if result is not None:
                return result

    return None


class NoAvailableKeysError(Exception):
    """Raised when no API keys are available for a request after waiting."""

    pass


class PreRequestCallbackError(Exception):
    """Raised when a pre-request callback fails."""

    pass


class CredentialNeedsReauthError(Exception):
    """
    Raised when a credential's refresh token is invalid and re-authentication is required.

    This is a rotatable error - the request should try the next credential while
    the broken credential is queued for re-authentication in the background.

    Unlike generic HTTPStatusError, this exception signals:
    - The credential is temporarily unavailable (needs user action)
    - Re-auth has already been queued
    - The request should rotate to the next credential without logging scary tracebacks

    Attributes:
        credential_path: Path to the credential file that needs re-auth
        message: Human-readable message about the error
    """

    def __init__(self, credential_path: str, message: str = ""):
        self.credential_path = credential_path
        self.message = (
            message or f"Credential '{credential_path}' requires re-authentication"
        )
        super().__init__(self.message)


class EmptyResponseError(Exception):
    """
    Raised when a provider returns an empty response after multiple retry attempts.

    This is a rotatable error - the request should try the next credential.
    Treated as a transient server-side issue (503 equivalent).

    Attributes:
        provider: The provider name (e.g., "antigravity")
        model: The model that was requested
        message: Human-readable message about the error
    """

    def __init__(self, provider: str, model: str, message: str = ""):
        self.provider = provider
        self.model = model
        self.message = (
            message
            or f"Empty response from {provider}/{model} after multiple retry attempts"
        )
        super().__init__(self.message)


class TransientQuotaError(Exception):
    """
    Raised when a provider returns a 429 without retry timing information.

    This indicates a transient rate limit rather than true quota exhaustion.
    The request has already been retried internally; this error signals
    that the credential should be rotated to try the next one.

    Treated as a transient server-side issue (503 equivalent), same as EmptyResponseError.

    Attributes:
        provider: The provider name (e.g., "antigravity")
        model: The model that was requested
        message: Human-readable message about the error
    """

    def __init__(self, provider: str, model: str, message: str = ""):
        self.provider = provider
        self.model = model
        self.message = (
            message
            or f"Transient 429 from {provider}/{model} after multiple retry attempts"
        )
        super().__init__(self.message)


# =============================================================================
# ERROR TRACKING FOR CLIENT REPORTING
# =============================================================================

# Abnormal errors that require attention and should always be reported to client
ABNORMAL_ERROR_TYPES = frozenset(
    {
        "forbidden",  # 403 - credential access issue
        "authentication",  # 401 - credential invalid/revoked
        "pre_request_callback_error",  # Internal proxy error
    }
)

# Normal/expected errors during operation - only report if ALL credentials fail
NORMAL_ERROR_TYPES = frozenset(
    {
        "rate_limit",  # 429 - expected during high load
        "quota_exceeded",  # Expected when quota runs out
        "server_error",  # 5xx - transient provider issues
        "api_connection",  # Network issues - transient
    }
)


def is_abnormal_error(classified_error: "ClassifiedError") -> bool:
    """
    Check if an error is abnormal and should be reported to the client.

    Abnormal errors indicate credential issues that need attention:
    - 403 Forbidden: Credential doesn't have access
    - 401 Unauthorized: Credential is invalid/revoked

    Normal errors are expected during operation:
    - 429 Rate limit: Expected during high load
    - 5xx Server errors: Transient provider issues
    """
    return classified_error.error_type in ABNORMAL_ERROR_TYPES


def mask_credential(credential: str) -> str:
    """
    Mask a credential for safe display in logs and error messages.

    - For API keys: shows last 6 characters (e.g., "...xyz123")
    - For OAuth file paths: shows just the filename (e.g., "antigravity_oauth_1.json")
    """
    if os.path.isfile(credential) or credential.endswith(".json"):
        return os.path.basename(credential)
    elif len(credential) > 6:
        return f"...{credential[-6:]}"
    else:
        return "***"


class RequestErrorAccumulator:
    """
    Tracks errors encountered during a request's credential rotation cycle.

    Used to build informative error messages for clients when all credentials
    are exhausted. Distinguishes between abnormal errors (that need attention)
    and normal errors (expected during operation).
    """

    def __init__(self):
        self.abnormal_errors: list = []  # 403, 401 - always report details
        self.normal_errors: list = []  # 429, 5xx - summarize only
        self._tried_credentials: set = set()  # Track unique credentials
        self.timeout_occurred: bool = False
        self.model: str = ""
        self.provider: str = ""

    def record_error(
        self, credential: str, classified_error: "ClassifiedError", error_message: str
    ):
        """Record an error for a credential."""
        self._tried_credentials.add(credential)
        masked_cred = mask_credential(credential)

        error_record = {
            "credential": masked_cred,
            "error_type": classified_error.error_type,
            "status_code": classified_error.status_code,
            "message": self._truncate_message(error_message, 150),
        }

        if is_abnormal_error(classified_error):
            self.abnormal_errors.append(error_record)
        else:
            self.normal_errors.append(error_record)

    @property
    def total_credentials_tried(self) -> int:
        """Return the number of unique credentials tried."""
        return len(self._tried_credentials)

    def _truncate_message(self, message: str, max_length: int = 150) -> str:
        """Truncate error message for readability."""
        # Take first line and truncate
        first_line = message.split("\n")[0]
        if len(first_line) > max_length:
            return first_line[:max_length] + "..."
        return first_line

    def has_errors(self) -> bool:
        """Check if any errors were recorded."""
        return bool(self.abnormal_errors or self.normal_errors)

    def has_abnormal_errors(self) -> bool:
        """Check if any abnormal errors were recorded."""
        return bool(self.abnormal_errors)

    def get_normal_error_summary(self) -> str:
        """Get a summary of normal errors (not individual details)."""
        if not self.normal_errors:
            return ""

        # Count by type
        counts = {}
        for err in self.normal_errors:
            err_type = err["error_type"]
            counts[err_type] = counts.get(err_type, 0) + 1

        # Build summary like "3 rate_limit, 1 server_error"
        parts = [f"{count} {err_type}" for err_type, count in counts.items()]
        return ", ".join(parts)

    def build_client_error_response(self) -> dict:
        """
        Build a structured error response for the client.

        Returns a dict suitable for JSON serialization in the error response.
        """
        # Determine the primary failure reason
        if self.timeout_occurred:
            error_type = "proxy_timeout"
            base_message = f"Request timed out after trying {self.total_credentials_tried} credential(s)"
        else:
            error_type = "proxy_all_credentials_exhausted"
            base_message = f"All {self.total_credentials_tried} credential(s) exhausted for {self.provider}"

        # Build human-readable message
        message_parts = [base_message]

        if self.abnormal_errors:
            message_parts.append("\n\nCredential issues (require attention):")
            for err in self.abnormal_errors:
                status = (
                    f"HTTP {err['status_code']}"
                    if err["status_code"] is not None
                    else err["error_type"]
                )
                message_parts.append(
                    f"\n  • {err['credential']}: {status} - {err['message']}"
                )

        normal_summary = self.get_normal_error_summary()
        if normal_summary:
            if self.abnormal_errors:
                message_parts.append(
                    f"\n\nAdditionally: {normal_summary} (expected during normal operation)"
                )
            else:
                message_parts.append(f"\n\nAll failures were: {normal_summary}")
                message_parts.append(
                    "\nThis is normal during high load - retry later or add more credentials."
                )

        response = {
            "error": {
                "message": "".join(message_parts),
                "type": error_type,
                "details": {
                    "model": self.model,
                    "provider": self.provider,
                    "credentials_tried": self.total_credentials_tried,
                    "timeout": self.timeout_occurred,
                },
            }
        }

        # Only include abnormal errors in details (they need attention)
        if self.abnormal_errors:
            response["error"]["details"]["abnormal_errors"] = self.abnormal_errors

        # Include summary of normal errors
        if normal_summary:
            response["error"]["details"]["normal_error_summary"] = normal_summary

        return response

    def build_log_message(self) -> str:
        """
        Build a concise log message for server-side logging.

        Shorter than client message, suitable for terminal display.
        """
        parts = []

        if self.timeout_occurred:
            parts.append(
                f"TIMEOUT: {self.total_credentials_tried} creds tried for {self.model}"
            )
        else:
            parts.append(
                f"ALL CREDS EXHAUSTED: {self.total_credentials_tried} tried for {self.model}"
            )

        if self.abnormal_errors:
            abnormal_summary = ", ".join(
                f"{e['credential']}={e['status_code'] or e['error_type']}"
                for e in self.abnormal_errors
            )
            parts.append(f"ISSUES: {abnormal_summary}")

        normal_summary = self.get_normal_error_summary()
        if normal_summary:
            parts.append(f"Normal: {normal_summary}")

        return " | ".join(parts)


class ClassifiedError:
    """A structured representation of a classified error."""

    def __init__(
        self,
        error_type: str,
        original_exception: Exception,
        status_code: Optional[int] = None,
        retry_after: Optional[int] = None,
        quota_reset_timestamp: Optional[float] = None,
    ):
        self.error_type = error_type
        self.original_exception = original_exception
        self.status_code = status_code
        self.retry_after = retry_after
        # Unix timestamp when quota resets (from quota_exhausted errors)
        # This is the authoritative reset time parsed from provider's error response
        self.quota_reset_timestamp = quota_reset_timestamp

    def __str__(self):
        parts = [
            f"type={self.error_type}",
            f"status={self.status_code}",
            f"retry_after={self.retry_after}",
        ]
        if self.quota_reset_timestamp:
            parts.append(f"quota_reset_ts={self.quota_reset_timestamp}")
        parts.append(f"original_exc={self.original_exception}")
        return f"ClassifiedError({', '.join(parts)})"


def _extract_retry_from_json_body(json_text: str) -> Optional[int]:
    """
    Extract retry delay from a JSON error response body.

    Handles Antigravity/Google API error formats with details array containing:
    - RetryInfo with retryDelay: "562476.752463453s"
    - ErrorInfo metadata with quotaResetDelay: "156h14m36.752463453s"

    Args:
        json_text: JSON string (original case, not lowercased)

    Returns:
        Retry delay in seconds, or None if not found
    """
    try:
        # Find JSON object in the text
        json_match = re.search(r"(\{.*\})", json_text, re.DOTALL)
        if not json_match:
            return None

        error_json = json.loads(json_match.group(1))
        details = error_json.get("error", {}).get("details", [])

        # Iterate through ALL details items (not just index 0)
        for detail in details:
            detail_type = detail.get("@type", "")

            # Check RetryInfo for retryDelay (most authoritative)
            # Note: Case-sensitive key names as returned by API
            if "google.rpc.RetryInfo" in detail_type:
                delay_str = detail.get("retryDelay")
                if delay_str:
                    # Handle both {"seconds": "123"} format and "123.456s" string format
                    if isinstance(delay_str, dict):
                        seconds = delay_str.get("seconds")
                        if seconds:
                            return int(float(seconds))
                    elif isinstance(delay_str, str):
                        result = _parse_duration_string(delay_str)
                        if result is not None:
                            return result

            # Check ErrorInfo metadata for quotaResetDelay (Antigravity-specific)
            if "google.rpc.ErrorInfo" in detail_type:
                metadata = detail.get("metadata", {})
                # Try both camelCase and lowercase variants
                quota_reset_delay = metadata.get("quotaResetDelay") or metadata.get(
                    "quotaresetdelay"
                )
                if quota_reset_delay:
                    result = _parse_duration_string(quota_reset_delay)
                    if result is not None:
                        return result

    except (json.JSONDecodeError, IndexError, KeyError, TypeError):
        pass

    return None


def get_retry_after(error: Exception) -> Optional[int]:
    """
    Extracts the 'retry-after' duration in seconds from an exception message.
    Handles both integer and string representations of the duration, as well as JSON bodies.
    Also checks HTTP response headers for httpx.HTTPStatusError instances.

    Supports Antigravity/Google API error formats:
    - RetryInfo with retryDelay: "562476.752463453s"
    - ErrorInfo metadata with quotaResetDelay: "156h14m36.752463453s"
    - Human-readable message: "quota will reset after 156h14m36s"
    """
    # 0. For httpx errors, check response body and headers
    if isinstance(error, httpx.HTTPStatusError):
        # First, try to parse the response body JSON (contains retryDelay/quotaResetDelay)
        # This is where Antigravity puts the retry information
        try:
            response_text = error.response.text
            if response_text:
                result = _extract_retry_from_json_body(response_text)
                if result is not None:
                    return result
        except Exception:
            pass  # Response body may not be available

        # Fallback to HTTP headers
        headers = error.response.headers
        # Check standard Retry-After header (case-insensitive)
        retry_header = headers.get("retry-after") or headers.get("Retry-After")
        if retry_header:
            try:
                return int(retry_header)  # Assumes seconds format
            except ValueError:
                pass  # Might be HTTP date format, skip for now

        # Check X-RateLimit-Reset header (Unix timestamp)
        reset_header = headers.get("x-ratelimit-reset") or headers.get(
            "X-RateLimit-Reset"
        )
        if reset_header:
            try:
                import time

                reset_timestamp = int(reset_header)
                current_time = int(time.time())
                wait_seconds = reset_timestamp - current_time
                if wait_seconds > 0:
                    return wait_seconds
            except (ValueError, TypeError):
                pass

    # 1. Try to parse JSON from the error string representation
    # Some exceptions embed JSON in their string representation
    error_str = str(error)
    result = _extract_retry_from_json_body(error_str)
    if result is not None:
        return result

    # 2. Common regex patterns for 'retry-after' (with compound duration support)
    # Use lowercase for pattern matching
    error_str_lower = error_str.lower()
    patterns = [
        r"retry[-_\s]after:?\s*(\d+)",  # Matches: retry-after, retry_after, retry after
        r"retry in\s*(\d+)\s*seconds?",
        r"wait for\s*(\d+)\s*seconds?",
        r'"retrydelay":\s*"([\d.]+)s?"',  # retryDelay in JSON (lowercased)
        r"x-ratelimit-reset:?\s*(\d+)",
        # Compound duration patterns (Antigravity format)
        r"quota will reset after\s*([\dhms.]+)",  # e.g., "156h14m36s" or "120s"
        r"reset after\s*([\dhms.]+)",
        r'"quotaresetdelay":\s*"([\dhms.]+)"',  # quotaResetDelay in JSON (lowercased)
    ]

    for pattern in patterns:
        match = re.search(pattern, error_str_lower)
        if match:
            duration_str = match.group(1)
            # Try parsing as compound duration first
            result = _parse_duration_string(duration_str)
            if result is not None:
                return result
            # Fallback to simple integer
            try:
                return int(duration_str)
            except (ValueError, IndexError):
                continue

    # 3. Handle cases where the error object itself has the attribute
    if hasattr(error, "retry_after"):
        value = getattr(error, "retry_after")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            result = _parse_duration_string(value)
            if result is not None:
                return result

    return None


def classify_error(e: Exception, provider: Optional[str] = None) -> ClassifiedError:
    """
    Classifies an exception into a structured ClassifiedError object.
    Now handles both litellm and httpx exceptions.

    If provider is specified and has a parse_quota_error() method,
    attempts provider-specific error parsing first before falling back
    to generic classification.

    Error types and their typical handling:
    - rate_limit (429): Rotate key, may retry with backoff
    - server_error (5xx): Retry with backoff, then rotate
    - forbidden (403): Rotate key immediately (access denied for this credential)
    - authentication (401): Rotate key, trigger re-auth if OAuth
    - quota_exceeded: Rotate key (credential quota exhausted)
    - invalid_request (400): Don't retry - client error in request
    - context_window_exceeded: Don't retry - request too large
    - api_connection: Retry with backoff, then rotate
    - unknown: Rotate key (safer to try another)

    Args:
        e: The exception to classify
        provider: Optional provider name for provider-specific error parsing

    Returns:
        ClassifiedError with error_type, status_code, retry_after, etc.
    """
    # Try provider-specific parsing first for 429/rate limit errors
    if provider:
        try:
            from .providers import PROVIDER_PLUGINS

            provider_class = PROVIDER_PLUGINS.get(provider)

            if provider_class and hasattr(provider_class, "parse_quota_error"):
                # Get error body if available
                error_body = None
                if hasattr(e, "response") and hasattr(e.response, "text"):
                    try:
                        error_body = e.response.text
                    except Exception:
                        pass
                elif hasattr(e, "body"):
                    error_body = str(e.body)

                quota_info = provider_class.parse_quota_error(e, error_body)

                if quota_info and quota_info.get("retry_after"):
                    retry_after = quota_info["retry_after"]
                    reason = quota_info.get("reason", "QUOTA_EXHAUSTED")
                    reset_ts = quota_info.get("reset_timestamp")
                    quota_reset_timestamp = quota_info.get("quota_reset_timestamp")

                    # Log the parsed result with human-readable duration
                    hours = retry_after / 3600
                    lib_logger.info(
                        f"Provider '{provider}' parsed quota error: "
                        f"retry_after={retry_after}s ({hours:.1f}h), reason={reason}"
                        + (f", resets at {reset_ts}" if reset_ts else "")
                    )

                    return ClassifiedError(
                        error_type="quota_exceeded",
                        original_exception=e,
                        status_code=429,
                        retry_after=retry_after,
                        quota_reset_timestamp=quota_reset_timestamp,
                    )
        except Exception as parse_error:
            lib_logger.debug(
                f"Provider-specific error parsing failed for '{provider}': {parse_error}"
            )
            # Fall through to generic classification

    # Generic classification logic
    status_code = getattr(e, "status_code", None)

    if isinstance(e, httpx.HTTPStatusError):  # [NEW] Handle httpx errors first
        status_code = e.response.status_code

        # Try to get error body for better classification
        try:
            error_body = e.response.text.lower() if hasattr(e.response, "text") else ""
        except Exception:
            error_body = ""

        if status_code == 401:
            return ClassifiedError(
                error_type="authentication",
                original_exception=e,
                status_code=status_code,
            )
        if status_code == 403:
            # 403 Forbidden - credential doesn't have access, should rotate
            # Could be: IP restriction, account disabled, permission denied, etc.
            return ClassifiedError(
                error_type="forbidden",
                original_exception=e,
                status_code=status_code,
            )
        if status_code == 429:
            retry_after = get_retry_after(e)
            # Check if this is a quota error vs rate limit
            if "quota" in error_body or "resource_exhausted" in error_body:
                return ClassifiedError(
                    error_type="quota_exceeded",
                    original_exception=e,
                    status_code=status_code,
                    retry_after=retry_after,
                )
            return ClassifiedError(
                error_type="rate_limit",
                original_exception=e,
                status_code=status_code,
                retry_after=retry_after,
            )
        if status_code == 400:
            # Check for context window / token limit errors with more specific patterns
            if any(
                pattern in error_body
                for pattern in [
                    "context_length",
                    "max_tokens",
                    "token limit",
                    "context window",
                    "too many tokens",
                    "too long",
                ]
            ):
                return ClassifiedError(
                    error_type="context_window_exceeded",
                    original_exception=e,
                    status_code=status_code,
                )
            return ClassifiedError(
                error_type="invalid_request",
                original_exception=e,
                status_code=status_code,
            )
            return ClassifiedError(
                error_type="invalid_request",
                original_exception=e,
                status_code=status_code,
            )
        if 400 <= status_code < 500:
            # Other 4xx errors - generally client errors
            return ClassifiedError(
                error_type="invalid_request",
                original_exception=e,
                status_code=status_code,
            )
        if 500 <= status_code:
            return ClassifiedError(
                error_type="server_error", original_exception=e, status_code=status_code
            )

    if isinstance(
        e, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)
    ):  # [NEW]
        return ClassifiedError(
            error_type="api_connection", original_exception=e, status_code=status_code
        )

    if isinstance(e, PreRequestCallbackError):
        return ClassifiedError(
            error_type="pre_request_callback_error",
            original_exception=e,
            status_code=400,  # Treat as a bad request
        )

    if isinstance(e, CredentialNeedsReauthError):
        # This is a rotatable error - credential is broken but re-auth is queued
        return ClassifiedError(
            error_type="credential_reauth_needed",
            original_exception=e,
            status_code=401,  # Treat as auth error for reporting purposes
        )

    if isinstance(e, EmptyResponseError):
        # Transient server-side issue - provider returned empty response
        # This is rotatable - try next credential
        return ClassifiedError(
            error_type="server_error",
            original_exception=e,
            status_code=503,
        )

    if isinstance(e, TransientQuotaError):
        # Transient 429 without retry info - provider returned bare rate limit
        # This is rotatable - try next credential
        return ClassifiedError(
            error_type="server_error",
            original_exception=e,
            status_code=503,
        )

    if isinstance(e, RateLimitError):
        retry_after = get_retry_after(e)
        # Check if this is a quota error vs rate limit
        error_msg = str(e).lower()
        if "quota" in error_msg or "resource_exhausted" in error_msg:
            return ClassifiedError(
                error_type="quota_exceeded",
                original_exception=e,
                status_code=status_code or 429,
                retry_after=retry_after,
            )
        return ClassifiedError(
            error_type="rate_limit",
            original_exception=e,
            status_code=status_code or 429,
            retry_after=retry_after,
        )

    if isinstance(e, (AuthenticationError,)):
        return ClassifiedError(
            error_type="authentication",
            original_exception=e,
            status_code=status_code or 401,
        )

    if isinstance(e, (InvalidRequestError, BadRequestError)):
        return ClassifiedError(
            error_type="invalid_request",
            original_exception=e,
            status_code=status_code or 400,
        )

    if isinstance(e, ContextWindowExceededError):
        return ClassifiedError(
            error_type="context_window_exceeded",
            original_exception=e,
            status_code=status_code or 400,
        )

    if isinstance(e, (APIConnectionError, Timeout)):
        return ClassifiedError(
            error_type="api_connection",
            original_exception=e,
            status_code=status_code or 503,  # Treat like a server error
        )

    if isinstance(e, (ServiceUnavailableError, InternalServerError)):
        # These are often temporary server-side issues
        # Note: OpenAIError removed - it's too broad and can catch client errors
        return ClassifiedError(
            error_type="server_error",
            original_exception=e,
            status_code=status_code or 503,
        )

    # Fallback for any other unclassified errors
    return ClassifiedError(
        error_type="unknown", original_exception=e, status_code=status_code
    )


def is_rate_limit_error(e: Exception) -> bool:
    """Checks if the exception is a rate limit error."""
    return isinstance(e, RateLimitError)


def is_server_error(e: Exception) -> bool:
    """Checks if the exception is a temporary server-side error."""
    return isinstance(
        e,
        (ServiceUnavailableError, APIConnectionError, InternalServerError, OpenAIError),
    )


def is_unrecoverable_error(e: Exception) -> bool:
    """
    Checks if the exception is a non-retriable client-side error.
    These are errors that will not resolve on their own.
    """
    return isinstance(e, (InvalidRequestError, AuthenticationError, BadRequestError))


def should_rotate_on_error(classified_error: ClassifiedError) -> bool:
    """
    Determines if an error should trigger key rotation.

    Errors that SHOULD rotate (try another key):
    - rate_limit: Current key is throttled
    - quota_exceeded: Current key/account exhausted
    - forbidden: Current credential denied access
    - authentication: Current credential invalid
    - credential_reauth_needed: Credential needs interactive re-auth (queued)
    - server_error: Provider having issues (might work with different endpoint/key)
    - api_connection: Network issues (might be transient)
    - unknown: Safer to try another key

    Errors that should NOT rotate (fail immediately):
    - invalid_request: Client error in request payload (won't help to retry)
    - context_window_exceeded: Request too large (won't help to retry)
    - pre_request_callback_error: Internal proxy error

    Returns:
        True if should rotate to next key, False if should fail immediately
    """
    non_rotatable_errors = {
        "invalid_request",
        "context_window_exceeded",
        "pre_request_callback_error",
    }
    return classified_error.error_type not in non_rotatable_errors


def should_retry_same_key(classified_error: ClassifiedError) -> bool:
    """
    Determines if an error should retry with the same key (with backoff).

    Only server errors and connection issues should retry the same key,
    as these are often transient.

    Returns:
        True if should retry same key, False if should rotate immediately
    """
    retryable_errors = {
        "server_error",
        "api_connection",
    }
    return classified_error.error_type in retryable_errors


class AllProviders:
    """
    A class to handle provider-specific settings, such as custom API bases.
    Supports custom OpenAI-compatible providers configured via environment variables.
    """

    def __init__(self):
        self.providers = {
            "chutes": {
                "api_base": "https://llm.chutes.ai/v1",
                "model_prefix": "openai/",
            }
        }
        # Load custom OpenAI-compatible providers from environment
        self._load_custom_providers()

    def _load_custom_providers(self):
        """
        Loads custom OpenAI-compatible providers from environment variables.
        Looks for environment variables in the format: PROVIDER_API_BASE
        where PROVIDER is the name of the custom provider.
        """
        import os

        # Get all environment variables that end with _API_BASE
        for env_var in os.environ:
            if env_var.endswith("_API_BASE"):
                provider_name = env_var.split("_API_BASE")[
                    0
                ].lower()  # Remove '_API_BASE' suffix and lowercase

                # Skip known providers that are already handled
                if provider_name in [
                    "openai",
                    "anthropic",
                    "google",
                    "gemini",
                    "nvidia",
                    "mistral",
                    "cohere",
                    "groq",
                    "openrouter",
                ]:
                    continue

                api_base = os.getenv(env_var)
                if api_base:
                    self.providers[provider_name] = {
                        "api_base": api_base.rstrip("/") if api_base else "",
                        "model_prefix": None,  # No prefix for custom providers
                    }

    def get_provider_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Returns provider-specific kwargs for a given model.
        """
        model = kwargs.get("model")
        if not model:
            return kwargs

        provider = self._get_provider_from_model(model)
        provider_settings = self.providers.get(provider, {})

        if "api_base" in provider_settings:
            kwargs["api_base"] = provider_settings["api_base"]

        if (
            "model_prefix" in provider_settings
            and provider_settings["model_prefix"] is not None
        ):
            kwargs["model"] = (
                f"{provider_settings['model_prefix']}{model.split('/', 1)[1]}"
            )

        return kwargs

    def _get_provider_from_model(self, model: str) -> str:
        """
        Determines the provider from the model name.
        """
        return model.split("/")[0]
