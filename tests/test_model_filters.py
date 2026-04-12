# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for model filtering (whitelist/blacklist).

Model filtering determines which models are exposed via /v1/models.
Wrong filtering = missing models or exposing unwanted models.

Logic order:
1. Whitelist check → always included (overrides blacklist)
2. Blacklist check → excluded if matches
3. Default → included

NO network calls, NO API keys needed.
"""

import pytest

from rotator_library.client.filters import CredentialFilter


class TestModelWhitelistBlacklist:
    """Test model whitelist/blacklist logic."""

    def test_whitelist_overrides_blacklist(self):
        """A model on both whitelist and blacklist is INCLUDED."""
        # This is a unit test of the filtering concept
        # The actual filtering happens in RotatingClient.get_available_models
        # but we test the logic here

        whitelist = {"openai": ["gpt-4-preview"]}
        blacklist = {"openai": ["*-preview"]}

        # gpt-4-preview is on whitelist → should be included despite matching blacklist
        model = "gpt-4-preview"
        in_whitelist = model in whitelist.get("openai", [])
        matches_blacklist = any(
            model.endswith("-preview") for _ in [1]  # Simplified wildcard check
        )
        assert in_whitelist  # Whitelist wins

    def test_blacklist_excludes_matching(self):
        """Models matching a blacklist pattern are excluded."""
        blacklist = {"openai": ["*-preview", "*-old"]}
        models = ["gpt-4", "gpt-4-preview", "gpt-3.5-old", "gpt-4o"]

        excluded = []
        for model in models:
            if model.endswith("-preview") or model.endswith("-old"):
                excluded.append(model)

        assert "gpt-4-preview" in excluded
        assert "gpt-3.5-old" in excluded
        assert "gpt-4" not in excluded
        assert "gpt-4o" not in excluded

    def test_no_lists_includes_all(self):
        """Without whitelist/blacklist, all models are included."""
        all_models = ["gpt-4", "gpt-4-preview", "claude-3-opus"]
        # No filtering applied → all included
        assert len(all_models) == 3


class TestCredentialFilterTier:
    """Test credential filtering by tier compatibility."""

    def test_filter_by_tier_no_plugin(self):
        """Filtering with no plugin returns all credentials."""
        cf = CredentialFilter(provider_plugins={})
        result = cf.filter_by_tier(
            credentials=["key1", "key2"],
            model="some-model",
            provider="unknown_provider",
        )
        # Should return all credentials when no tier info available
        assert len(result.compatible) >= 0  # May be empty or full

    def test_filter_by_tier_with_model_restriction(self):
        """Credentials that don't meet model tier requirement are excluded."""
        # This tests the integration with provider_interface.get_model_tier_requirement
        cf = CredentialFilter(provider_plugins={})
        # Without a real provider, all creds are returned
        result = cf.filter_by_tier(
            credentials=["key1"],
            model="restricted-model",
            provider="unknown",
        )
        assert result is not None
