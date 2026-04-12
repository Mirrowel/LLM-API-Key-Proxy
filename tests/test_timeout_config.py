# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for TimeoutConfig centralized timeout handling.
"""

import os
from unittest import mock
import pytest
from rotator_library.timeout_config import TimeoutConfig

class TestTimeoutConfig:
    """Test TimeoutConfig reading from env vars and default fallbacks."""

    @pytest.mark.parametrize(
        "method_name, env_key, default_value",
        [
            ("connect", "TIMEOUT_CONNECT", TimeoutConfig._CONNECT),
            ("write", "TIMEOUT_WRITE", TimeoutConfig._WRITE),
            ("pool", "TIMEOUT_POOL", TimeoutConfig._POOL),
            ("read_streaming", "TIMEOUT_READ_STREAMING", TimeoutConfig._READ_STREAMING),
            ("read_non_streaming", "TIMEOUT_READ_NON_STREAMING", TimeoutConfig._READ_NON_STREAMING),
        ],
    )
    def test_timeout_config_value_error(self, method_name, env_key, default_value):
        """Test that ValueError is caught and handled when env var is not a float."""
        with mock.patch.dict(os.environ, {env_key: "not-a-float"}):
            with mock.patch("rotator_library.timeout_config.lib_logger.warning") as mock_warning:
                method = getattr(TimeoutConfig, method_name)
                result = method()

                # Should return the default value
                assert result == default_value

                # Should log a warning about the invalid value
                mock_warning.assert_called_once_with(
                    f"Invalid value for {env_key}: not-a-float. Using default: {default_value}"
                )

    @pytest.mark.parametrize(
        "method_name, env_key",
        [
            ("connect", "TIMEOUT_CONNECT"),
            ("write", "TIMEOUT_WRITE"),
            ("pool", "TIMEOUT_POOL"),
            ("read_streaming", "TIMEOUT_READ_STREAMING"),
            ("read_non_streaming", "TIMEOUT_READ_NON_STREAMING"),
        ],
    )
    @pytest.mark.parametrize(
        "value_str, expected_float",
        [
            ("45.5", 45.5),
            ("0.0", 0.0),
            ("0.0001", 0.0001),
            ("-5.0", -5.0), # Assuming negative floats are valid floats for Python, specific implementation validation is tested elsewhere.
        ]
    )
    def test_timeout_config_custom_valid(self, method_name, env_key, value_str, expected_float):
        """Test that valid custom float values from env vars are returned."""
        with mock.patch.dict(os.environ, {env_key: value_str}):
            method = getattr(TimeoutConfig, method_name)
            result = method()
            assert result == expected_float

    def test_timeout_config_factory_streaming(self):
        """Test streaming factory method uses the correct overridden and default values."""
        with mock.patch.dict(os.environ, {"TIMEOUT_CONNECT": "1.5", "TIMEOUT_READ_STREAMING": "2.5"}):
            timeout = TimeoutConfig.streaming()

            # Replaced with custom values
            assert timeout.connect == 1.5
            assert timeout.read == 2.5

            # Using defaults
            assert timeout.write == TimeoutConfig._WRITE
            assert timeout.pool == TimeoutConfig._POOL

    def test_timeout_config_factory_non_streaming(self):
        """Test non-streaming factory method uses the correct overridden and default values."""
        with mock.patch.dict(os.environ, {"TIMEOUT_WRITE": "3.5", "TIMEOUT_READ_NON_STREAMING": "4.5"}):
            timeout = TimeoutConfig.non_streaming()

            # Replaced with custom values
            assert timeout.write == 3.5
            assert timeout.read == 4.5

            # Using defaults
            assert timeout.connect == TimeoutConfig._CONNECT
            assert timeout.pool == TimeoutConfig._POOL
