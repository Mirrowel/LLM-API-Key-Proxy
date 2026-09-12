"""
Tests for the error_handler module's error classification and rotation decision logic.
"""

import pytest
import httpx
from unittest.mock import Mock, MagicMock
from litellm.exceptions import (
    RateLimitError,
    AuthenticationError,
    InvalidRequestError,
    BadRequestError,
    ServiceUnavailableError,
    InternalServerError,
    ContextWindowExceededError,
)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rotator_library.error_handler import (
    classify_error,
    should_rotate_on_error,
    should_retry_same_key,
    ClassifiedError,
)


class TestClassifyError:
    """Test error classification logic."""
    
    def _make_http_error(self, status_code: int, text: str = "") -> httpx.HTTPStatusError:
        """Helper to create mock HTTP status errors."""
        response = Mock(spec=httpx.Response)
        response.status_code = status_code
        response.text = text
        response.headers = {}  # Add empty headers
        request = Mock(spec=httpx.Request)
        return httpx.HTTPStatusError(f"HTTP {status_code}", request=request, response=response)
    
    def test_classify_403_as_forbidden(self):
        """403 Forbidden should be classified as 'forbidden', not 'invalid_request'."""
        error = self._make_http_error(403, "Access denied")
        classified = classify_error(error)
        
        assert classified.error_type == "forbidden"
        assert classified.status_code == 403
    
    def test_classify_401_as_authentication(self):
        """401 Unauthorized should be classified as 'authentication'."""
        error = self._make_http_error(401, "Unauthorized")
        classified = classify_error(error)
        
        assert classified.error_type == "authentication"
        assert classified.status_code == 401
    
    def test_classify_429_as_rate_limit(self):
        """429 Too Many Requests should be classified as 'rate_limit'."""
        error = self._make_http_error(429, "Rate limited")
        classified = classify_error(error)
        
        assert classified.error_type == "rate_limit"
        assert classified.status_code == 429
    
    def test_classify_429_quota_as_quota_exceeded(self):
        """429 with quota message should be classified as 'quota_exceeded'."""
        error = self._make_http_error(429, "Quota exceeded")
        classified = classify_error(error)
        
        assert classified.error_type == "quota_exceeded"
        assert classified.status_code == 429
    
    def test_classify_400_as_invalid_request(self):
        """400 Bad Request should be classified as 'invalid_request'."""
        error = self._make_http_error(400, "Bad request")
        classified = classify_error(error)
        
        assert classified.error_type == "invalid_request"
        assert classified.status_code == 400
    
    def test_classify_400_context_as_context_exceeded(self):
        """400 with context/token message should be classified as 'context_window_exceeded'."""
        error = self._make_http_error(400, "Context length exceeded")
        classified = classify_error(error)
        
        assert classified.error_type == "context_window_exceeded"
        assert classified.status_code == 400
    
    def test_classify_500_as_server_error(self):
        """5xx errors should be classified as 'server_error'."""
        for status_code in [500, 502, 503, 504]:
            error = self._make_http_error(status_code, "Server error")
            classified = classify_error(error)
            
            assert classified.error_type == "server_error"
            assert classified.status_code == status_code


class TestShouldRotateOnError:
    """Test rotation decision logic."""
    
    def test_rotate_on_forbidden(self):
        """403 Forbidden should trigger rotation."""
        classified = ClassifiedError("forbidden", Exception(), 403)
        assert should_rotate_on_error(classified) is True
    
    def test_rotate_on_rate_limit(self):
        """Rate limit errors should trigger rotation."""
        classified = ClassifiedError("rate_limit", Exception(), 429)
        assert should_rotate_on_error(classified) is True
    
    def test_rotate_on_quota_exceeded(self):
        """Quota exceeded errors should trigger rotation."""
        classified = ClassifiedError("quota_exceeded", Exception(), 429)
        assert should_rotate_on_error(classified) is True
    
    def test_rotate_on_authentication(self):
        """Authentication errors should trigger rotation."""
        classified = ClassifiedError("authentication", Exception(), 401)
        assert should_rotate_on_error(classified) is True
    
    def test_rotate_on_server_error(self):
        """Server errors should trigger rotation."""
        classified = ClassifiedError("server_error", Exception(), 503)
        assert should_rotate_on_error(classified) is True
    
    def test_rotate_on_unknown(self):
        """Unknown errors should trigger rotation (safer)."""
        classified = ClassifiedError("unknown", Exception(), None)
        assert should_rotate_on_error(classified) is True
    
    def test_no_rotate_on_invalid_request(self):
        """Invalid request errors should NOT trigger rotation."""
        classified = ClassifiedError("invalid_request", Exception(), 400)
        assert should_rotate_on_error(classified) is False
    
    def test_no_rotate_on_context_exceeded(self):
        """Context window exceeded should NOT trigger rotation."""
        classified = ClassifiedError("context_window_exceeded", Exception(), 400)
        assert should_rotate_on_error(classified) is False


class TestShouldRetrySameKey:
    """Test same-key retry decision logic."""
    
    def test_retry_same_key_on_server_error(self):
        """Server errors should retry with same key."""
        classified = ClassifiedError("server_error", Exception(), 503)
        assert should_retry_same_key(classified) is True
    
    def test_retry_same_key_on_connection_error(self):
        """Connection errors should retry with same key."""
        classified = ClassifiedError("api_connection", Exception(), None)
        assert should_retry_same_key(classified) is True
    
    def test_no_retry_same_key_on_rate_limit(self):
        """Rate limit should NOT retry same key (rotate instead)."""
        classified = ClassifiedError("rate_limit", Exception(), 429)
        assert should_retry_same_key(classified) is False
    
    def test_no_retry_same_key_on_forbidden(self):
        """Forbidden should NOT retry same key (rotate instead)."""
        classified = ClassifiedError("forbidden", Exception(), 403)
        assert should_retry_same_key(classified) is False
    
    def test_no_retry_same_key_on_authentication(self):
        """Authentication errors should NOT retry same key."""
        classified = ClassifiedError("authentication", Exception(), 401)
        assert should_retry_same_key(classified) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
