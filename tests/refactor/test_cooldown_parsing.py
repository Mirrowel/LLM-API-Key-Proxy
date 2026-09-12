"""Comprehensive tests for cooldown value parsing in CustomCapConfig."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from rotator_library.usage.config import (
    CustomCapConfig,
    _parse_cooldown_config,
    _parse_duration_string,
    CapMode,
    CooldownMode,
)


# =============================================================================
# DURATION STRING PARSING TESTS
# =============================================================================


class TestParseDurationString:
    """Tests for _parse_duration_string() helper."""

    def test_plain_integer(self):
        """Plain integer seconds."""
        assert _parse_duration_string("300") == 300
        assert _parse_duration_string("0") == 0
        assert _parse_duration_string("86400") == 86400

    def test_simple_seconds(self):
        """Simple seconds with 's' suffix."""
        assert _parse_duration_string("60s") == 60
        assert _parse_duration_string("3600s") == 3600

    def test_simple_minutes(self):
        """Simple minutes with 'm' suffix."""
        assert _parse_duration_string("30m") == 1800
        assert _parse_duration_string("60m") == 3600

    def test_simple_hours(self):
        """Simple hours with 'h' suffix."""
        assert _parse_duration_string("1h") == 3600
        assert _parse_duration_string("24h") == 86400

    def test_simple_days(self):
        """Simple days with 'd' suffix."""
        assert _parse_duration_string("1d") == 86400
        assert _parse_duration_string("7d") == 604800

    def test_compound_hours_minutes(self):
        """Compound duration: hours + minutes."""
        assert _parse_duration_string("1h30m") == 5400  # 1.5 hours
        assert _parse_duration_string("2h45m") == 9900  # 2h 45m

    def test_compound_hours_minutes_seconds(self):
        """Compound duration: hours + minutes + seconds."""
        assert _parse_duration_string("1h30m45s") == 5445
        assert _parse_duration_string("0h5m30s") == 330

    def test_compound_days_hours_minutes(self):
        """Compound duration: days + hours + minutes."""
        assert _parse_duration_string("2d1h30m") == 178200  # 2*86400 + 1*3600 + 30*60
        assert _parse_duration_string("1d12h") == 129600  # 1.5 days

    def test_case_insensitive(self):
        """Duration parsing is case-insensitive."""
        assert _parse_duration_string("1H30M") == 5400
        assert _parse_duration_string("2D") == 172800

    def test_whitespace_handling(self):
        """Whitespace is stripped."""
        assert _parse_duration_string("  300  ") == 300
        assert _parse_duration_string(" 1h30m ") == 5400

    def test_empty_returns_none(self):
        """Empty string returns None."""
        assert _parse_duration_string("") is None
        assert _parse_duration_string("   ") is None

    def test_invalid_returns_none(self):
        """Invalid input returns None."""
        assert _parse_duration_string("abc") is None
        assert _parse_duration_string("xyz123") is None


# =============================================================================
# COOLDOWN CONFIG PARSING TESTS
# =============================================================================


class TestParseCooldownConfig:
    """Tests for _parse_cooldown_config() which auto-detects mode from value format."""

    # -------------------------------------------------------------------------
    # FLAT DURATION (FIXED mode)
    # -------------------------------------------------------------------------

    def test_flat_integer(self):
        """Plain integer -> FIXED mode."""
        mode, value = _parse_cooldown_config(None, 300)
        assert mode == CooldownMode.FIXED
        assert value == 300

    def test_flat_integer_string(self):
        """Plain integer as string -> FIXED mode."""
        mode, value = _parse_cooldown_config(None, "300")
        assert mode == CooldownMode.FIXED
        assert value == 300

    def test_flat_duration_string(self):
        """Duration string without sign -> FIXED mode."""
        mode, value = _parse_cooldown_config(None, "1h30m")
        assert mode == CooldownMode.FIXED
        assert value == 5400

    def test_zero_value(self):
        """Zero value -> QUOTA_RESET mode (default)."""
        mode, value = _parse_cooldown_config(None, 0)
        assert mode == CooldownMode.QUOTA_RESET
        assert value == 0

    # -------------------------------------------------------------------------
    # OFFSET (+ or - prefix)
    # -------------------------------------------------------------------------

    def test_positive_offset_integer(self):
        """+integer -> OFFSET mode with positive value."""
        mode, value = _parse_cooldown_config(None, "+300")
        assert mode == CooldownMode.OFFSET
        assert value == 300

    def test_negative_offset_integer(self):
        """-integer -> OFFSET mode with negative value."""
        mode, value = _parse_cooldown_config(None, "-300")
        assert mode == CooldownMode.OFFSET
        assert value == -300

    def test_positive_offset_duration(self):
        """+duration -> OFFSET mode with positive seconds."""
        mode, value = _parse_cooldown_config(None, "+1h30m")
        assert mode == CooldownMode.OFFSET
        assert value == 5400

    def test_negative_offset_duration(self):
        """-duration -> OFFSET mode with negative seconds."""
        mode, value = _parse_cooldown_config(None, "-5m")
        assert mode == CooldownMode.OFFSET
        assert value == -300

    # -------------------------------------------------------------------------
    # PERCENTAGE OFFSET
    # -------------------------------------------------------------------------

    def test_positive_percentage(self):
        """+50% -> OFFSET mode with encoded percentage."""
        mode, value = _parse_cooldown_config(None, "+50%")
        assert mode == CooldownMode.OFFSET
        # Encoded as -1000 - (sign * percentage) = -1000 - 50 = -1050
        assert value == -1050

    def test_negative_percentage(self):
        """-20% -> OFFSET mode with encoded negative percentage."""
        mode, value = _parse_cooldown_config(None, "-20%")
        assert mode == CooldownMode.OFFSET
        # Encoded as -1000 - (-1 * 20) = -1000 + 20 = -980
        assert value == -980

    def test_percentage_no_sign(self):
        """50% without sign -> treated as positive percentage."""
        mode, value = _parse_cooldown_config(None, "50%")
        assert mode == CooldownMode.OFFSET
        assert value == -1050  # Same as +50%

    # -------------------------------------------------------------------------
    # QUOTA_RESET (explicit string)
    # -------------------------------------------------------------------------

    def test_quota_reset_string(self):
        """'quota_reset' string -> QUOTA_RESET mode."""
        mode, value = _parse_cooldown_config(None, "quota_reset")
        assert mode == CooldownMode.QUOTA_RESET
        assert value == 0

    def test_quota_reset_variants(self):
        """Various quota_reset string formats."""
        for s in ["quota_reset", "quota-reset", "quotareset", "QUOTA_RESET"]:
            mode, value = _parse_cooldown_config(None, s)
            assert mode == CooldownMode.QUOTA_RESET, f"Failed for: {s}"

    # -------------------------------------------------------------------------
    # EXPLICIT MODE OVERRIDE
    # -------------------------------------------------------------------------

    def test_explicit_mode_fixed(self):
        """Explicit mode='fixed' with integer value."""
        mode, value = _parse_cooldown_config("fixed", 600)
        assert mode == CooldownMode.FIXED
        assert value == 600

    def test_explicit_mode_offset(self):
        """Explicit mode='offset' with integer value."""
        mode, value = _parse_cooldown_config("offset", 300)
        assert mode == CooldownMode.OFFSET
        assert value == 300

    def test_explicit_mode_quota_reset(self):
        """Explicit mode='quota_reset'."""
        mode, value = _parse_cooldown_config("quota_reset", 0)
        assert mode == CooldownMode.QUOTA_RESET
        assert value == 0

    def test_explicit_mode_with_duration_string(self):
        """Explicit mode with duration string value."""
        mode, value = _parse_cooldown_config("fixed", "2h")
        assert mode == CooldownMode.FIXED
        assert value == 7200


# =============================================================================
# CustomCapConfig.from_dict() INTEGRATION TESTS
# =============================================================================


class TestCustomCapConfigFromDict:
    """Tests for CustomCapConfig.from_dict() method."""

    def test_basic_config(self):
        """Basic config with integer values."""
        config = {"max_requests": 100, "cooldown_value": 300}
        cap = CustomCapConfig.from_dict("1", "model-x", config)

        assert cap.tier_key == "1"
        assert cap.model_or_group == "model-x"
        assert cap.max_requests == 100
        assert cap.cooldown_mode == CooldownMode.FIXED
        assert cap.cooldown_value == 300

    def test_percentage_max_requests(self):
        """max_requests as percentage string."""
        config = {"max_requests": "80%"}
        cap = CustomCapConfig.from_dict("1", "model-x", config)

        # Stored as a positive value with PERCENTAGE mode
        assert cap.max_requests == 80
        assert cap.max_requests_mode == CapMode.PERCENTAGE

    def test_duration_string_cooldown(self):
        """cooldown_value as duration string."""
        config = {"max_requests": 50, "cooldown_value": "1h30m"}
        cap = CustomCapConfig.from_dict("1", "model-x", config)

        assert cap.cooldown_mode == CooldownMode.FIXED
        assert cap.cooldown_value == 5400

    def test_offset_cooldown(self):
        """cooldown_value with offset format."""
        config = {"max_requests": 50, "cooldown_value": "+30m"}
        cap = CustomCapConfig.from_dict("1", "model-x", config)

        assert cap.cooldown_mode == CooldownMode.OFFSET
        assert cap.cooldown_value == 1800

    def test_explicit_mode_in_config(self):
        """Explicit cooldown_mode in config."""
        config = {
            "max_requests": 50,
            "cooldown_mode": "quota_reset",
        }
        cap = CustomCapConfig.from_dict("1", "model-x", config)

        assert cap.cooldown_mode == CooldownMode.QUOTA_RESET
        assert cap.cooldown_value == 0

    def test_default_tier(self):
        """Default tier key."""
        config = {"max_requests": 100}
        cap = CustomCapConfig.from_dict("default", "model-x", config)

        assert cap.tier_key == "default"

    # Superseded: empty/invalid max_requests configs are now rejected\n    # (CustomCapConfig.from_dict returns None) instead of defaulting to 0.\n
