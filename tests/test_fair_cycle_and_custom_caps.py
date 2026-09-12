"""Tests for Fair Cycle Rotation and Custom Caps features.

The fair-cycle / custom-cap / integration classes in this file historically
exercised the pre-decomposition UsageManager monolith (2028c272 "modularize
client and usage architecture"); they were superseded by the usage/ package
and its current coverage in tests/refactor/test_fair_cycle_*.py,
test_custom_cap_limits.py and test_custom_caps_advanced.py.

Retained below: TestEnvVarParsing, which tests live usage-config parsing.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestEnvVarParsing:
    """Tests for environment variable parsing."""

    def test_parse_single_tier(self):
        """Test parsing single tier env var."""
        print("\n[TEST] Parse single tier env var...")

        from rotator_library.usage.config import _parse_custom_cap_env_key

        tier_key, model_key = _parse_custom_cap_env_key("2_CLAUDE")

        assert tier_key == 2, f"Expected tier 2, got {tier_key}"
        assert model_key == "claude", f"Expected 'claude', got {model_key}"

        print("  PASS Single tier parsing works correctly")

    def test_parse_numeric_model_group(self):
        """Test parsing numeric model/group names after the tier."""
        print("\n[TEST] Parse numeric model/group env var...")

        from rotator_library.usage.config import _parse_custom_cap_env_key
        from rotator_library.core.config import ConfigLoader

        tier_key, model_key = _parse_custom_cap_env_key("2_25_FLASH")

        assert tier_key == 2, f"Expected tier 2, got {tier_key}"
        assert model_key == "25-flash", f"Expected '25-flash', got {model_key}"

        tier_key, model_key = _parse_custom_cap_env_key("2_3_FLASH")

        assert tier_key == 2, f"Expected tier 2, got {tier_key}"
        assert model_key == "3-flash", f"Expected '3-flash', got {model_key}"

        core_tier_key, core_model_key = ConfigLoader()._parse_tier_model_from_env(
            "2_25_FLASH"
        )
        assert core_tier_key == 2, f"Expected core tier 2, got {core_tier_key}"
        assert core_model_key == "25-flash", (
            f"Expected core '25-flash', got {core_model_key}"
        )

        print("  PASS Numeric model/group parsing works correctly")

    def test_parse_default_tier(self):
        """Test parsing default tier env var."""
        print("\n[TEST] Parse default tier env var...")

        from rotator_library.usage.config import _parse_custom_cap_env_key

        tier_key, model_key = _parse_custom_cap_env_key("DEFAULT_CLAUDE_SONNET")

        assert tier_key == "default", f"Expected 'default', got {tier_key}"
        assert model_key == "claude-sonnet", (
            f"Expected 'claude-sonnet', got {model_key}"
        )

        print("  PASS Default tier parsing works correctly")

    def test_parse_complex_model_name(self):
        """Test parsing complex model name with underscores."""
        print("\n[TEST] Parse complex model name...")

        from rotator_library.usage.config import _parse_custom_cap_env_key

        tier_key, model_key = _parse_custom_cap_env_key("2_CLAUDE_OPUS_4_5")

        assert tier_key == 2, f"Expected tier 2, got {tier_key}"
        # Underscores converted to dashes
        assert model_key == "claude-opus-4-5", (
            f"Expected 'claude-opus-4-5', got {model_key}"
        )

        print("  PASS Complex model name parsing works correctly")

    def test_provider_default_custom_caps_not_duplicated(self):
        """Test provider default custom caps are loaded exactly once."""
        from rotator_library.usage.config import load_provider_usage_config

        class ProviderWithDefaultCap:
            default_custom_caps = {
                2: {"25-flash": {"max_requests": 10, "cooldown_mode": "quota_reset"}}
            }

        with patch.dict(os.environ, {}, clear=True):
            config = load_provider_usage_config(
                "mock", {"mock": ProviderWithDefaultCap}
            )

        assert len(config.custom_caps) == 1
        assert config.custom_caps[0].model_or_group == "25-flash"

