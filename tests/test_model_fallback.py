# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for MODEL_FALLBACK registry parsing.

Per-model provider fallback via MODEL_FALLBACK_<NAME> environment variables
is critical for automatic spillover when a primary provider has scaling issues.

Breakage here means:
- Provider fallback/spillover stops working
- Fallback targets are parsed incorrectly
- Retry mode isn't respected

NO network calls, NO API keys needed.
"""

import os
from unittest.mock import patch

from rotator_library.model_fallback_registry import (
    ModelFallbackRegistry,
    DEFAULT_FALLBACK_RETRY_MODE,
)
from rotator_library.model_alias_registry import AliasTarget


class TestModelFallbackRegistry:
    """Test MODEL_FALLBACK env var parsing and fallback resolution."""

    def test_parse_provider_only_targets(self):
        """Provider-only entries use the canonical model name."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_GEMMA_4_31B_IT": "nvidia_nim,ollama_cloud"
        }, clear=False):
            registry = ModelFallbackRegistry()
            targets = registry.resolve("gemma-4-31b-it")
            assert targets is not None
            assert len(targets) == 2
            assert targets[0].provider == "nvidia_nim"
            assert targets[0].model_name == "gemma-4-31b-it"
            assert targets[1].provider == "ollama_cloud"
            assert targets[1].model_name == "gemma-4-31b-it"

    def test_parse_explicit_model_targets(self):
        """Explicit provider:model entries use the specified model name."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_GEMMA_4_31B_IT": "nvidia_nim:google/gemma-4-31b-it,ollama_cloud:gemma-4-31b-it"
        }, clear=False):
            registry = ModelFallbackRegistry()
            targets = registry.resolve("gemma-4-31b-it")
            assert targets is not None
            assert len(targets) == 2
            assert targets[0].provider == "nvidia_nim"
            assert targets[0].model_name == "google/gemma-4-31b-it"
            assert targets[1].provider == "ollama_cloud"
            assert targets[1].model_name == "gemma-4-31b-it"

    def test_parse_mixed_targets(self):
        """Mix of provider-only and provider:model entries."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_DEEPSEEK_V3": "google,nvidia_nim:deepseek-chat"
        }, clear=False):
            registry = ModelFallbackRegistry()
            targets = registry.resolve("deepseek-v3")
            assert targets is not None
            assert len(targets) == 2
            assert targets[0].provider == "google"
            assert targets[0].model_name == "deepseek-v3"  # canonical name
            assert targets[1].provider == "nvidia_nim"
            assert targets[1].model_name == "deepseek-chat"  # explicit

    def test_default_retry_mode_exhaust(self):
        """Default retry mode for fallback is exhaust (not round_robin)."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_TEST_MODEL": "google,nvidia_nim"
        }, clear=False):
            registry = ModelFallbackRegistry()
            mode = registry.get_retry_mode("test-model")
            assert mode == "exhaust"
            assert DEFAULT_FALLBACK_RETRY_MODE == "exhaust"

    def test_parse_retry_mode_round_robin(self):
        """round_robin retry mode is parsed from pipe suffix."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_TEST_MODEL": "google,nvidia_nim|round_robin"
        }, clear=False):
            registry = ModelFallbackRegistry()
            mode = registry.get_retry_mode("test-model")
            assert mode == "round_robin"

    def test_parse_retry_mode_exhaust_explicit(self):
        """Explicit exhaust retry mode is parsed correctly."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_TEST_MODEL": "google,nvidia_nim|exhaust"
        }, clear=False):
            registry = ModelFallbackRegistry()
            mode = registry.get_retry_mode("test-model")
            assert mode == "exhaust"

    def test_no_matching_fallback(self):
        """Unknown model returns None."""
        with patch.dict(os.environ, {}, clear=False):
            registry = ModelFallbackRegistry()
            targets = registry.resolve("nonexistent-model")
            assert targets is None

    def test_has_fallback(self):
        """has_fallback returns True for registered models."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_MY_MODEL": "google"
        }, clear=False):
            registry = ModelFallbackRegistry()
            assert registry.has_fallback("my-model")

    def test_no_fallback(self):
        """has_fallback returns False for unknown models."""
        with patch.dict(os.environ, {}, clear=False):
            registry = ModelFallbackRegistry()
            assert not registry.has_fallback("nonexistent")

    def test_underscore_to_hyphen_normalization(self):
        """Env var underscores are normalized to hyphens in canonical names."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_GEMMA_4_31B_IT": "google"
        }, clear=False):
            registry = ModelFallbackRegistry()
            # GEMMA_4_31B_IT → canonical "gemma-4-31b-it"
            targets = registry.resolve("gemma-4-31b-it")
            assert targets is not None

    def test_period_hyphen_normalization(self):
        """Lookup normalizes periods to hyphens (gemma-4.31b-it → gemma-4-31b-it)."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_GEMMA_4_31B_IT": "google"
        }, clear=False):
            registry = ModelFallbackRegistry()
            # Lookup with periods should match
            targets = registry.resolve("gemma-4.31b-it")
            assert targets is not None

    def test_empty_value_returns_none(self):
        """Empty env var value is handled gracefully."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_EMPTY": ""
        }, clear=False):
            registry = ModelFallbackRegistry()
            assert not registry.has_fallback("empty")

    def test_whitespace_handling(self):
        """Whitespace around entries is trimmed."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_TEST_MODEL": " google , nvidia_nim "
        }, clear=False):
            registry = ModelFallbackRegistry()
            targets = registry.resolve("test-model")
            assert targets is not None
            assert len(targets) == 2
            assert targets[0].provider == "google"
            assert targets[1].provider == "nvidia_nim"

    def test_invalid_pipe_suffix_treated_as_value(self):
        """Invalid pipe suffix is treated as part of the value."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_TEST_MODEL": "google|invalid_mode"
        }, clear=False):
            registry = ModelFallbackRegistry()
            # "invalid_mode" is not a valid retry mode,
            # so the whole string is treated as value
            targets = registry.resolve("test-model")
            # "google|invalid_mode" → provider="google|invalid_mode" (no colon)
            # This is an odd edge case - the provider name will contain |
            assert targets is not None

    def test_get_all_fallbacks(self):
        """get_all_fallbacks returns all registered fallbacks."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_MODEL_A": "google",
            "MODEL_FALLBACK_MODEL_B": "nvidia_nim,ollama_cloud"
        }, clear=False):
            registry = ModelFallbackRegistry()
            all_fb = registry.get_all_fallbacks()
            assert "model-a" in all_fb
            assert "model-b" in all_fb
            assert len(all_fb["model-b"].targets) == 2

    def test_targets_are_alias_target_instances(self):
        """Fallback targets are AliasTarget instances (reused from alias registry)."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_TEST_MODEL": "google:gemma-4-31b-it"
        }, clear=False):
            registry = ModelFallbackRegistry()
            targets = registry.resolve("test-model")
            assert targets is not None
            assert isinstance(targets[0], AliasTarget)
            assert targets[0].full_model == "google/gemma-4-31b-it"

    def test_resolve_returns_copy(self):
        """resolve() returns a copy of the targets list, not the internal reference."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_TEST_MODEL": "google,nvidia_nim"
        }, clear=False):
            registry = ModelFallbackRegistry()
            targets1 = registry.resolve("test-model")
            targets2 = registry.resolve("test-model")
            assert targets1 is not targets2
            assert targets1 == targets2

    def test_single_provider_fallback(self):
        """Single provider as fallback works correctly."""
        with patch.dict(os.environ, {
            "MODEL_FALLBACK_GEMMA_4_31B_IT": "nvidia_nim"
        }, clear=False):
            registry = ModelFallbackRegistry()
            targets = registry.resolve("gemma-4-31b-it")
            assert targets is not None
            assert len(targets) == 1
            assert targets[0].provider == "nvidia_nim"
            assert targets[0].model_name == "gemma-4-31b-it"
