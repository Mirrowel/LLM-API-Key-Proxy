# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for request sanitization.

Sanitization removes unsupported parameters from requests before
they reach providers. If this breaks:
- `dimensions` on non-OpenAI models → 400 Bad Request
- `thinking` on non-Gemini or non-Anthropic models → 400 Bad Request

NO network calls, NO API keys needed.
"""

import copy

from rotator_library.request_sanitizer import sanitize_request_payload


class TestSanitizeDimensions:
    """Test removal of `dimensions` parameter for non-embedding models."""

    def test_dimensions_removed_for_non_embedding_model(self):
        """dimensions is removed for any model without 'embedding' in its name."""
        payload = {"model": "openai/gpt-4o", "input": "test", "dimensions": 512}
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "dimensions" not in result

    def test_dimensions_kept_for_openai_embedding(self):
        """dimensions is preserved for OpenAI text-embedding-3 models."""
        for model in ["openai/text-embedding-3-small", "openai/text-embedding-3-large"]:
            payload = {"model": model, "input": "test", "dimensions": 512}
            result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
            assert result["dimensions"] == 512

    def test_dimensions_kept_for_gemini_embedding(self):
        """dimensions is preserved for Gemini embedding models."""
        payload = {"model": "google/gemini-embedding-2", "input": "test", "dimensions": 768}
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert result["dimensions"] == 768

    def test_no_dimensions_key(self):
        """Payload without dimensions is unchanged."""
        payload = {"model": "test-model", "input": "test"}
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert result == payload


class TestSanitizeThinking:
    """Test removal of `thinking` parameter for unsupported models."""

    def test_thinking_removed_for_non_whitelisted(self):
        """thinking is removed for models that aren't gemini or anthropic models."""
        payload = {
            "model": "openai/gpt-4o",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": -1},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "thinking" not in result

    def test_thinking_kept_for_anthropic(self):
        """thinking is preserved for anthropic models."""
        payload = {
            "model": "anthropic/claude-3-7-sonnet",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": -1},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "thinking" in result

    def test_thinking_kept_for_gemini(self):
        """thinking is preserved for gemini models."""
        payload = {
            "model": "gemini/gemini-2.0-flash",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": -1},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "thinking" in result

    def test_thinking_removed_if_different_value(self):
        """thinking is removed for non-whitelisted even if values are different."""
        payload = {
            "model": "some-model",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 5000},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "thinking" not in result

    def test_empty_payload(self):
        """Empty payload doesn't crash."""
        result = sanitize_request_payload({}, "any-model")
        assert result == {}

    def test_thinking_with_invalid_type(self):
        """thinking parameter with non-dict type doesn't crash and is removed for non-whitelisted models."""
        payload = {
            "model": "some-model",
            "messages": [],
            "thinking": "enabled",  # String instead of dict
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "thinking" not in result


class TestSanitizeCombined:
    """Test payloads containing multiple parameters that need sanitization."""

    def test_both_removed_for_unsupported_model(self):
        """Both dimensions and thinking are removed for unsupported model."""
        payload = {
            "model": "some/unsupported-model",
            "input": "test",
            "dimensions": 1024,
            "thinking": {"type": "enabled", "budget_tokens": -1},
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "dimensions" not in result
        assert "thinking" not in result

    def test_dimensions_removed_for_openai_non_embedding(self):
        """Dimensions removed for OpenAI chat models."""
        payload = {
            "model": "openai/gpt-4o",
            "messages": [],
            "dimensions": 512,
        }
        result = sanitize_request_payload(copy.deepcopy(payload), payload["model"])
        assert "dimensions" not in result
