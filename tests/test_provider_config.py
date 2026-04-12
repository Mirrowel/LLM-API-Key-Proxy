# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for provider configuration accessors.
"""

from unittest.mock import patch

import pytest

from rotator_library.provider_config import (
    get_full_provider_config,
    get_provider_ui_config,
)

@pytest.mark.parametrize("provider,expected_category", [
    ("openai", "popular"),
    ("anthropic", "popular"),
    ("gemini", "popular"),
])
def test_get_provider_ui_config_existing_parametrized(provider, expected_category):
    """Test getting an existing provider's UI config using hardcoded expectations."""
    result = get_provider_ui_config(provider)
    assert result["category"] == expected_category

def test_get_provider_ui_config_known_provider():
    """Test get_provider_ui_config with a known provider."""
    mock_litellm_providers = {
        "test_provider": {
            "category": "test_category",
            "note": "Test note",
        }
    }
    with patch("rotator_library.provider_config.LITELLM_PROVIDERS", mock_litellm_providers):
        config = get_provider_ui_config("test_provider")
        assert config == {"category": "test_category", "note": "Test note"}

def test_get_provider_ui_config_unknown_provider():
    """Test get_provider_ui_config with an unknown provider."""
    mock_litellm_providers = {}
    with patch("rotator_library.provider_config.LITELLM_PROVIDERS", mock_litellm_providers):
        config = get_provider_ui_config("unknown_provider")
        assert config == {"category": "other"}

def test_get_full_provider_config_known_provider():
    """Test get_full_provider_config with a known provider."""
    mock_scraped_providers = {
        "test_provider": {
            "api_base": "https://api.test.com",
            "models": ["model-a"],
        }
    }
    mock_litellm_providers = {
        "test_provider": {
            "category": "test_category",
            "note": "Test note",
        }
    }
    with patch("rotator_library.provider_config.SCRAPED_PROVIDERS", mock_scraped_providers):
        with patch("rotator_library.provider_config.LITELLM_PROVIDERS", mock_litellm_providers):
            config = get_full_provider_config("test_provider")

            # Should have properties from both
            assert config["api_base"] == "https://api.test.com"
            assert config["models"] == ["model-a"]
            assert config["category"] == "test_category"
            assert config["note"] == "Test note"

def test_get_full_provider_config_unknown_provider():
    """Test get_full_provider_config with an unknown provider."""
    mock_scraped_providers = {}
    mock_litellm_providers = {}

    with patch("rotator_library.provider_config.SCRAPED_PROVIDERS", mock_scraped_providers):
        with patch("rotator_library.provider_config.LITELLM_PROVIDERS", mock_litellm_providers):
            config = get_full_provider_config("unknown_provider")

            # Should fallback to default category and have no scraped properties
            assert config == {"category": "other"}

@pytest.mark.parametrize(
    "scraped_data, ui_data, provider_key, expected",
    [
        (
            {},
            {"test_provider": {"category": "ui_only_category"}},
            "test_provider",
            {"category": "ui_only_category"}
        ),
        (
            {"test_provider": {"api_base": "https://test.com"}},
            {},
            "test_provider",
            {"api_base": "https://test.com", "category": "other"}
        ),
        (
            {"test_provider": None},
            {"test_provider": None},
            "test_provider",
            {"category": "other"}
        ),
        (
            {"test_provider": {"category": "scraped_category", "api_base": "https://scraped.com"}},
            {"test_provider": {"category": "ui_category", "note": "ui note"}},
            "test_provider",
            {"category": "ui_category", "api_base": "https://scraped.com", "note": "ui note"}
        )
    ]
)
def test_get_full_provider_config_parametrized(scraped_data, ui_data, provider_key, expected):
    """Test get_full_provider_config with various partial and overlapping data scenarios."""

    with patch("rotator_library.provider_config.SCRAPED_PROVIDERS", scraped_data):
        with patch("rotator_library.provider_config.LITELLM_PROVIDERS", ui_data):
            config = get_full_provider_config(provider_key)
            assert config == expected

def test_get_provider_ui_config_malformed():
    """Test get_provider_ui_config when provider is malformed."""
    mock_litellm_providers = {"malformed_provider": None}
    with patch("rotator_library.provider_config.LITELLM_PROVIDERS", mock_litellm_providers):
        config = get_provider_ui_config("malformed_provider")
        assert config == {"category": "other"}
