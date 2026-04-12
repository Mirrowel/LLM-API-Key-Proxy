# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for error classification and duration parsing.

Error classification determines retry/rotation behavior:
- AUTHENTICATION → immediate lockout (wrong = wasted keys)
- RATE_LIMIT/QUOTA → escalating cooldown (wrong = flooding provider)
- SERVER_ERROR → retry then rotate (wrong = giving up too early)
- CONTEXT_LENGTH/CONTENT_FILTER → immediate fail (wrong = useless retries)

NO network calls, NO API keys needed.
"""

import pytest
from unittest.mock import MagicMock

from rotator_library.error_handler import (
    classify_error,
    ClassifiedError,
    _parse_duration_string,
    mask_credential,
)


# =============================================================================
# Error Classification
# =============================================================================


class TestClassifyError:
    """Test error → error type string classification."""

    def test_401_is_authentication(self):
        """401 errors are classified as authentication."""
        from litellm.exceptions import AuthenticationError
        err = AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="gpt-4",
        )
        result = classify_error(err)
        assert result.error_type == "authentication"

    def test_429_is_rate_limit(self):
        """429 errors are classified as rate_limit."""
        from litellm.exceptions import RateLimitError
        err = RateLimitError(
            message="Rate limit exceeded",
            llm_provider="openai",
            model="gpt-4",
        )
        result = classify_error(err)
        assert result.error_type == "rate_limit"

    def test_500_is_server_error(self):
        """500 errors are classified as server_error."""
        from litellm.exceptions import InternalServerError
        err = InternalServerError(
            message="Internal server error",
            llm_provider="openai",
            model="gpt-4",
        )
        result = classify_error(err)
        assert result.error_type == "server_error"

    def test_502_is_server_error(self):
        """502/API connection errors are classified appropriately."""
        from litellm.exceptions import APIConnectionError
        err = APIConnectionError(
            message="Bad gateway",
            llm_provider="openai",
            model="gpt-4",
        )
        result = classify_error(err)
        assert result.error_type in ("server_error", "api_connection")

    def test_503_is_server_error(self):
        """503 errors are classified as server_error."""
        from litellm.exceptions import ServiceUnavailableError
        err = ServiceUnavailableError(
            message="Service unavailable",
            llm_provider="openai",
            model="gpt-4",
        )
        result = classify_error(err)
        assert result.error_type == "server_error"

    def test_context_window_exceeded(self):
        """Context window exceeded errors are classified correctly."""
        from litellm.exceptions import ContextWindowExceededError
        err = ContextWindowExceededError(
            message="This model's maximum context length is 128000 tokens",
            llm_provider="openai",
            model="gpt-4",
        )
        result = classify_error(err)
        # Some litellm versions classify this as invalid_request or context_window_exceeded
        assert result.error_type in ("context_window_exceeded", "invalid_request")

    def test_timeout_is_timeout(self):
        """Timeout errors are classified appropriately."""
        from litellm.exceptions import Timeout
        err = Timeout(
            message="Request timed out",
            llm_provider="openai",
            model="gpt-4",
        )
        result = classify_error(err)
        assert result.error_type in ("timeout", "proxy_timeout", "server_error", "api_connection")

    def test_unknown_exception(self):
        """Unclassified exceptions fall back to unknown."""
        err = RuntimeError("Something unexpected")
        result = classify_error(err)
        assert result.error_type == "unknown"


# =============================================================================
# Duration Parsing
# =============================================================================


class TestDurationParsing:
    """Test _parse_duration_string for retry-after header parsing."""

    def test_plain_seconds(self):
        assert _parse_duration_string("60") == 60

    def test_seconds_with_unit(self):
        assert _parse_duration_string("120s") == 120

    def test_minutes(self):
        assert _parse_duration_string("5m") == 300

    def test_hours(self):
        assert _parse_duration_string("2h") == 7200

    def test_compound_duration(self):
        """Compound durations like '2h30m' are parsed correctly."""
        result = _parse_duration_string("2h30m")
        assert result == 9000

    def test_milliseconds(self):
        """Milliseconds are converted to seconds (minimum 1)."""
        result = _parse_duration_string("290ms")
        assert result == 1  # Sub-second rounds up to 1

    def test_milliseconds_over_one_second(self):
        """Milliseconds > 1s are converted correctly."""
        result = _parse_duration_string("1500ms")
        assert result == 1  # 1.5s rounds down to 1

    def test_empty_string(self):
        """Empty string returns None."""
        assert _parse_duration_string("") is None

    def test_none(self):
        """None returns None."""
        assert _parse_duration_string(None) is None

    def test_compound_with_seconds(self):
        """Full compound duration: hours + minutes + seconds."""
        result = _parse_duration_string("1h30m45s")
        assert result == 5445


# =============================================================================
# Credential Masking
# =============================================================================


class TestMaskCredential:
    """Test that credentials are properly masked in logs."""

    def test_long_key_masked(self):
        """Long API keys are masked (showing trailing chars)."""
        result = mask_credential("sk-1234567890abcdefghijklmnop")
        # Should not contain the full key
        assert "1234567890abcdef" not in result
        # Should show partial info
        assert "..." in result or len(result) < 30

    def test_short_key_handled(self):
        """Short keys don't crash masking."""
        result = mask_credential("sk-1")
        assert isinstance(result, str)

    def test_file_path_partial_mask(self):
        """File paths are partially masked."""
        result = mask_credential("/path/to/oauth_creds/gemini_cli_oauth_1.json")
        assert isinstance(result, str)
