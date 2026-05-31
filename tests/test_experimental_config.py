from __future__ import annotations

import pytest

from rotator_library.config.experimental import (
    ExperimentalConfigError,
    env_price_key,
    get_stream_runtime_settings,
    load_config_from_mapping,
    load_experimental_config,
    parse_field_cache_rules,
)


def test_missing_config_file_returns_empty(tmp_path) -> None:
    config = load_experimental_config(tmp_path / "missing.json", env={})

    assert config.is_empty


def test_loads_config_from_env_path(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"routing":{"model_routes":{"code":"group:chain"}},"extra":{}}', encoding="utf-8")

    config = load_experimental_config(env={"LLM_PROXY_CONFIG_FILE": str(path)})

    assert config.routing["model_routes"]["code"] == "group:chain"
    assert config.unknown_sections == {"extra": {}}
    assert config.warnings


def test_rejects_secret_like_json_keys() -> None:
    with pytest.raises(ExperimentalConfigError):
        load_config_from_mapping({"providers": {"openai": {"api_key": "hidden"}}})


def test_invalid_json_raises(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ExperimentalConfigError):
        load_experimental_config(path, env={})


def test_stream_runtime_settings_env_overrides_json() -> None:
    config = load_config_from_mapping({"streaming": {"trace_metrics": True, "stall_timeout_seconds": 30}})

    settings = get_stream_runtime_settings(config=config, env={"STREAM_TRACE_METRICS": "false"})

    assert settings.trace_metrics is False
    assert settings.stall_timeout_seconds == 30


def test_field_cache_rules_parse_wildcard_then_model_specific() -> None:
    config = load_config_from_mapping(
        {
            "field_cache": {
                "gemini_cli": {
                    "*": [{"name": "thought", "source": "response", "path": "$.thought", "target_path": "$.cached_thought"}],
                    "gemini-3": [{"name": "signature", "source": "response", "path": "$.sig", "scope": ["provider", "model"]}],
                }
            }
        }
    )

    rules = parse_field_cache_rules(config, "gemini_cli", "gemini-3")

    assert [rule.name for rule in rules] == ["thought", "signature"]
    assert rules[0].inject is not None


def test_env_price_key_sanitizes_provider_and_model() -> None:
    assert env_price_key("openai", "gpt-5.1-mini", "cache_read") == "MODEL_PRICE_OPENAI_GPT_5_1_MINI_CACHE_READ"
