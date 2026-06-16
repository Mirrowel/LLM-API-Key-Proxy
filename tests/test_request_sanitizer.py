# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Tests for request_sanitizer.sanitize_request_payload.

This module is a pure function that strips unsupported parameters
from LLM request payloads based on the target model name.
"""

import pytest

from request_sanitizer import sanitize_request_payload


class TestSanitizeDimensions:
    """Tests for the 'dimensions' parameter sanitization."""

    def test_dimensions_removed_for_non_openai_embedding_model(self):
        payload = {
            "model": "anthropic/claude-3-opus",
            "dimensions": 256,
        }
        result = sanitize_request_payload(payload, "anthropic/claude-3-opus")
        assert "dimensions" not in result

    def test_dimensions_kept_for_openai_text_embedding_3(self):
        payload = {
            "model": "openai/text-embedding-3-small",
            "dimensions": 256,
        }
        result = sanitize_request_payload(payload, "openai/text-embedding-3-small")
        assert result["dimensions"] == 256

    def test_dimensions_kept_for_openai_text_embedding_3_large(self):
        payload = {
            "dimensions": 1024,
        }
        result = sanitize_request_payload(payload, "openai/text-embedding-3-large")
        assert result["dimensions"] == 1024

    def test_dimensions_removed_for_gemini_model(self):
        payload = {
            "model": "gemini/gemini-2.5-pro",
            "dimensions": 512,
        }
        result = sanitize_request_payload(payload, "gemini/gemini-2.5-pro")
        assert "dimensions" not in result

    def test_no_dimensions_key_leaves_payload_unchanged(self):
        payload = {"model": "openai/gpt-4", "messages": []}
        result = sanitize_request_payload(payload, "openai/gpt-4")
        assert result == payload


class TestSanitizeThinking:
    """Tests for the 'thinking' parameter sanitization."""

    _ENABLED_BUDGET_NEG1 = {"type": "enabled", "budget_tokens": -1}

    def test_thinking_removed_for_non_gemini_model(self):
        payload = {
            "model": "openai/gpt-4",
            "thinking": self._ENABLED_BUDGET_NEG1,
        }
        result = sanitize_request_payload(payload, "openai/gpt-4")
        assert "thinking" not in result

    def test_thinking_kept_for_gemini_25_pro(self):
        payload = {
            "thinking": self._ENABLED_BUDGET_NEG1,
        }
        result = sanitize_request_payload(payload, "gemini/gemini-2.5-pro")
        assert result["thinking"] == self._ENABLED_BUDGET_NEG1

    def test_thinking_kept_for_gemini_25_flash(self):
        payload = {
            "thinking": self._ENABLED_BUDGET_NEG1,
        }
        result = sanitize_request_payload(payload, "gemini/gemini-2.5-flash")
        assert result["thinking"] == self._ENABLED_BUDGET_NEG1

    def test_thinking_with_other_value_is_kept(self):
        """Only the exact sentinel dict {type: enabled, budget_tokens: -1} is stripped."""
        thinking = {"type": "enabled", "budget_tokens": 1000}
        payload = {"thinking": thinking}
        result = sanitize_request_payload(payload, "openai/gpt-4")
        assert result["thinking"] == thinking

    def test_thinking_disabled_is_kept(self):
        thinking = {"type": "disabled"}
        payload = {"thinking": thinking}
        result = sanitize_request_payload(payload, "openai/gpt-4")
        assert result["thinking"] == thinking


class TestSanitizeGeneralBehavior:
    """General behavior tests."""

    def test_empty_payload(self):
        result = sanitize_request_payload({}, "openai/gpt-4")
        assert result == {}

    def test_unrelated_keys_preserved(self):
        payload = {
            "model": "openai/gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "max_tokens": 100,
        }
        result = sanitize_request_payload(payload, "openai/gpt-4")
        assert result == payload

    def test_returns_same_dict_object(self):
        """The function should modify and return the same dict object (in-place)."""
        payload = {"dimensions": 128}
        result = sanitize_request_payload(payload, "some/model")
        assert result is payload

    def test_both_dimensions_and_thinking_removed(self):
        payload = {
            "dimensions": 256,
            "thinking": {"type": "enabled", "budget_tokens": -1},
            "messages": [],
        }
        result = sanitize_request_payload(payload, "anthropic/claude-3-opus")
        assert "dimensions" not in result
        assert "thinking" not in result
        assert "messages" in result
