from __future__ import annotations

import pytest

from rotator_library.config.experimental import ExperimentalConfigError, get_stream_runtime_settings, load_config_from_mapping


def test_stream_settings_parse_json_values() -> None:
    config = load_config_from_mapping(
        {"streaming": {"ttfb_timeout_seconds": 5, "stall_timeout_seconds": 30, "heartbeat_interval_seconds": 10, "cancel_upstream_on_disconnect": False, "trace_metrics": False}}
    )

    settings = get_stream_runtime_settings(config=config, env={})

    assert settings.ttfb_timeout_seconds == 5
    assert settings.stall_timeout_seconds == 30
    assert settings.heartbeat_seconds == 10
    assert settings.cancel_upstream_on_disconnect is False
    assert settings.trace_metrics is False


def test_stream_settings_env_overrides_json_values() -> None:
    config = load_config_from_mapping({"streaming": {"trace_metrics": True, "heartbeat_seconds": 10}})

    settings = get_stream_runtime_settings(config=config, env={"STREAM_TRACE_METRICS": "false", "STREAM_HEARTBEAT_INTERVAL_SECONDS": "2"})

    assert settings.trace_metrics is False
    assert settings.heartbeat_seconds == 2


def test_stream_settings_accept_legacy_heartbeat_env_name() -> None:
    settings = get_stream_runtime_settings(env={"STREAM_HEARTBEAT_SECONDS": "3"})

    assert settings.heartbeat_seconds == 3


def test_stream_settings_invalid_boolean_fails_clearly() -> None:
    with pytest.raises(ExperimentalConfigError):
        get_stream_runtime_settings(env={"STREAM_TRACE_METRICS": "maybe"})
