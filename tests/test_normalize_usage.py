# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Tests for rotator_library.core.utils.normalize_usage_for_response.

This function normalizes provider usage data so that reasoning_tokens
are inclusive of completion_tokens (the OpenAI convention), fixing
providers like Mistral that report them exclusively.
"""

import pytest

from rotator_library.core.utils import normalize_usage_for_response


# ============================================================================
# Dict-based usage tests
# ============================================================================


class TestNormalizeUsageDict:
    """Tests using dict-style usage objects (as from streamed chunks)."""

    def test_no_change_when_no_reasoning_tokens(self):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        normalize_usage_for_response(usage, "openai/gpt-4")
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_no_change_when_reasoning_is_inclusive(self):
        """When completion_tokens >= reasoning_tokens, the convention is already correct."""
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 80,
            "total_tokens": 180,
            "completion_tokens_details": {"reasoning_tokens": 30},
        }
        normalize_usage_for_response(usage, "openai/o3")
        assert usage["completion_tokens"] == 80
        assert usage["total_tokens"] == 180

    def test_normalizes_exclusive_reasoning(self):
        """When reasoning_tokens > completion_tokens, add them together."""
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "completion_tokens_details": {"reasoning_tokens": 50},
        }
        normalize_usage_for_response(usage, "mistral/mistral-large")
        assert usage["completion_tokens"] == 70  # 20 + 50
        assert usage["total_tokens"] == 170  # 100 + 70

    def test_handles_missing_completion_tokens(self):
        usage = {
            "prompt_tokens": 100,
            "total_tokens": 100,
            "completion_tokens_details": {"reasoning_tokens": 30},
        }
        normalize_usage_for_response(usage, "mistral/mistral-large")
        assert usage["completion_tokens"] == 30  # 0 + 30
        assert usage["total_tokens"] == 130  # 100 + 30

    def test_handles_missing_prompt_tokens(self):
        usage = {
            "completion_tokens": 10,
            "total_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 40},
        }
        normalize_usage_for_response(usage, "mistral/mistral-large")
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 50  # 0 + 50

    def test_zero_reasoning_tokens_no_change(self):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
        normalize_usage_for_response(usage, "openai/gpt-4")
        assert usage["completion_tokens"] == 50

    def test_none_usage_does_nothing(self):
        """Passing None should be a no-op."""
        normalize_usage_for_response(None, "openai/gpt-4")  # should not raise

    def test_empty_dict_does_nothing(self):
        normalize_usage_for_response({}, "openai/gpt-4")  # should not raise
        # No assertions needed - just verifying it doesn't raise

    def test_details_without_reasoning_key(self):
        """completion_tokens_details without reasoning_tokens should be a no-op."""
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "completion_tokens_details": {},
        }
        normalize_usage_for_response(usage, "openai/gpt-4")
        assert usage["completion_tokens"] == 50


# ============================================================================
# Object/attribute-based usage tests
# ============================================================================


class _MockUsageDetails:
    """Minimal mock for a pydantic-style usage details object."""

    def __init__(self, reasoning_tokens=0):
        self.reasoning_tokens = reasoning_tokens


class _MockUsage:
    """Minimal mock for a pydantic-style usage object."""

    def __init__(self, prompt=0, completion=0, reasoning=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.completion_tokens_details = _MockUsageDetails(reasoning)


class TestNormalizeUsageObject:
    """Tests using attribute-style usage objects (as from non-streamed responses)."""

    def test_no_change_when_no_reasoning(self):
        usage = _MockUsage(prompt=100, completion=50)
        normalize_usage_for_response(usage, "openai/gpt-4")
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_normalizes_exclusive_reasoning(self):
        usage = _MockUsage(prompt=100, completion=20, reasoning=50)
        normalize_usage_for_response(usage, "mistral/mistral-large")
        assert usage.completion_tokens == 70
        assert usage.total_tokens == 170

    def test_no_change_when_inclusive(self):
        usage = _MockUsage(prompt=100, completion=80, reasoning=30)
        normalize_usage_for_response(usage, "openai/o3")
        assert usage.completion_tokens == 80
        assert usage.total_tokens == 180

    def test_object_with_none_details(self):
        usage = _MockUsage(prompt=100, completion=50)
        usage.completion_tokens_details = None
        normalize_usage_for_response(usage, "openai/gpt-4")
        assert usage.completion_tokens == 50


# ============================================================================
# Edge cases
# ============================================================================


class TestNormalizeUsageEdgeCases:
    def test_details_as_dict_on_object(self):
        """An object with completion_tokens_details as a dict should also work."""
        usage = _MockUsage(prompt=100, completion=20)
        usage.completion_tokens_details = {"reasoning_tokens": 50}

        normalize_usage_for_response(usage, "mistral/mistral-large")

        assert usage.completion_tokens == 70
        assert usage.total_tokens == 170
