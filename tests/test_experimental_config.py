from __future__ import annotations

import pytest

from rotator_library.config.experimental import (
    ExperimentalConfigError,
    as_int,
    env_price_key,
    get_responses_store_settings,
    get_retry_runtime_settings,
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


def test_retry_runtime_settings_env_overrides_json() -> None:
    config = load_config_from_mapping(
        {
            "retry": {
                "provider_cooldown": {"provider_cooldown_min_seconds": 20, "provider_cooldown_on_quota": True},
                "backoff": {"provider_backoff_threshold": 4, "failure_history_max_entries": 50},
            }
        }
    )

    settings = get_retry_runtime_settings(config=config, env={"PROVIDER_COOLDOWN_MIN_SECONDS": "5"})

    assert settings.provider_cooldown_min_seconds == 5
    assert settings.provider_cooldown_on_quota is True
    assert settings.provider_backoff_threshold == 4
    assert settings.failure_history_max_entries == 50


def test_responses_store_settings_env_overrides_json() -> None:
    config = load_config_from_mapping(
        {
            "responses": {
                "store": {
                    "ttl_seconds": 60,
                    "max_items": 10,
                    "store_failed": False,
                    "store_in_progress": True,
                }
            }
        }
    )

    settings = get_responses_store_settings(config=config, env={"RESPONSES_STORE_MAX_ITEMS": "5", "RESPONSES_STORE_FAILED": "true"})

    assert settings.ttl_seconds == 60
    assert settings.max_items == 5
    assert settings.store_failed is True
    assert settings.store_in_progress is True


def test_new_config_sections_still_reject_secret_like_keys() -> None:
    with pytest.raises(ExperimentalConfigError):
        load_config_from_mapping({"responses": {"store": {"authorization": "hidden"}}})


@pytest.mark.parametrize("secret_key", ["secret_key", "secret-key", "apiKey", "client-secret", "oauth_token", "oauthToken", "oauth-token", "id_token", "oauth_token_secret"])
def test_secret_key_variants_are_rejected(secret_key: str) -> None:
    with pytest.raises(ExperimentalConfigError):
        load_config_from_mapping({"retry": {secret_key: "hidden"}})


def test_retry_runtime_settings_malformed_env_preserves_defaults() -> None:
    config = load_config_from_mapping({"retry": {"provider_cooldown_default_seconds": 45}})

    settings = get_retry_runtime_settings(
        config=config,
        env={
            "PROVIDER_COOLDOWN_DEFAULT_SECONDS": "not-an-int",
            "PROVIDER_COOLDOWN_ON_QUOTA": "not-a-bool",
            "PROVIDER_BACKOFF_BASE_SECONDS": "not-an-int",
        },
    )

    assert settings.provider_cooldown_default_seconds == 30
    assert settings.provider_cooldown_on_quota is False
    assert settings.provider_backoff_base_seconds is None


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


def test_field_cache_rules_match_unprefixed_model_alias() -> None:
    config = load_config_from_mapping(
        {
            "field_cache": {
                "gemini_cli": {
                    "gemini-3": [{"name": "signature", "source": "response", "path": "sig"}],
                }
            }
        }
    )

    rules = parse_field_cache_rules(config, "gemini_cli", "gemini_cli/gemini-3")

    assert [rule.name for rule in rules] == ["signature"]


def test_field_cache_rule_parses_ttl_metadata_and_insert_injection() -> None:
    config = load_config_from_mapping(
        {
            "field_cache": {
                "provider": {
                    "*": [
                        {
                            "name": "tool_state",
                            "source": "stream_event",
                            "path": "raw.tool.state",
                            "mode": "per_tool_call",
                            "ttl_seconds": 120,
                            "metadata": {"tool_container_path": "tools"},
                            "inject": {"target": "request", "path": "metadata.tool_state", "insert": True},
                        }
                    ]
                }
            }
        }
    )

    rule = parse_field_cache_rules(config, "provider", "model")[0]

    assert rule.ttl_seconds == 120
    assert rule.metadata == {"tool_container_path": "tools"}
    assert rule.inject is not None
    assert rule.inject.insert is True


def test_field_cache_rule_rejects_invalid_config_values() -> None:
    config = load_config_from_mapping(
        {
            "field_cache": {
                "provider": {
                    "*": [
                        {"name": "bad_source", "source": "not_a_source", "path": "x"},
                    ]
                }
            }
        }
    )

    with pytest.raises(ValueError, match="Unsupported field-cache source"):
        parse_field_cache_rules(config, "provider", "model")


def test_field_cache_rules_reject_malformed_shapes() -> None:
    config = load_config_from_mapping({"field_cache": {"provider": {"*": ["not-a-rule"]}}})

    with pytest.raises(ExperimentalConfigError, match="rule entries"):
        parse_field_cache_rules(config, "provider", "model")


def test_field_cache_rules_reject_non_list_model_rules() -> None:
    config = load_config_from_mapping({"field_cache": {"provider": {"*": {"name": "bad"}}}})

    with pytest.raises(ExperimentalConfigError, match="model rules"):
        parse_field_cache_rules(config, "provider", "model")


def test_field_cache_rules_reject_malformed_nested_shapes() -> None:
    bad_inject = load_config_from_mapping({"field_cache": {"provider": {"*": [{"name": "bad", "source": "response", "path": "x", "inject": "not-object"}]}}})

    with pytest.raises(ExperimentalConfigError, match="inject"):
        parse_field_cache_rules(bad_inject, "provider", "model")

    bad_metadata = load_config_from_mapping({"field_cache": {"provider": {"*": [{"name": "bad", "source": "response", "path": "x", "metadata": "not-object"}]}}})

    with pytest.raises(ExperimentalConfigError, match="metadata"):
        parse_field_cache_rules(bad_metadata, "provider", "model")


def test_env_price_key_sanitizes_provider_and_model() -> None:
    assert env_price_key("openai", "gpt-5.1-mini", "cache_read") == "MODEL_PRICE_OPENAI_GPT_5_1_MINI_CACHE_READ"


def test_as_int_parses_integers_with_redacted_errors() -> None:
    assert as_int("5", name="TEST_INT") == 5
    with pytest.raises(ExperimentalConfigError, match="TEST_INT"):
        as_int("not-secret-value", name="TEST_INT")
