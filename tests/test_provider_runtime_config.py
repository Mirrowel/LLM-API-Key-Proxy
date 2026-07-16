from __future__ import annotations

import json
import pytest

from rotator_library.providers.provider_interface import ProviderInterface
from rotator_library.providers import PROVIDER_PLUGINS, _create_dynamic_plugin_class, _register_providers
from rotator_library.client.rotating_client import _add_configured_no_auth_credentials
from rotator_library.client.scopes import NO_AUTH_CREDENTIAL, ScopeManager
from rotator_library.config.experimental import load_config_from_mapping
from rotator_library.providers.claude_code_provider import ClaudeCodeProvider
from rotator_library.providers.codex_provider import CodexProvider
from rotator_library.providers.copilot_provider import CopilotProvider


class ConfiguredProvider(ProviderInterface):
    provider_env_name = "configured"
    protocol_name = "litellm_fallback"
    adapter_names = ("noop",)
    model_quota_groups = {"base": ["base-model"]}

    async def get_models(self, api_key, client):
        return []


def _write_config(tmp_path, payload: dict) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_provider_json_protocol_adapters_field_cache_and_quota_groups_are_wired(tmp_path, monkeypatch) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "providers": {
                "configured": {
                    "protocol_name": "openai_chat",
                    "default_output_protocol": "anthropic_messages",
                    "adapter_names": ["model_override"],
                    "adapter_config": {"model_override": {"model": "upstream-model"}},
                    "native_streaming_supported": True,
                    "field_cache": [
                        {"name": "state", "source": "response", "path": "metadata.state", "target_path": "metadata.cached_state"}
                    ],
                    "model_quota_groups": {"json_group": ["json-model"]},
                }
            }
        },
    )
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", config_path)
    provider = ConfiguredProvider()

    assert provider.get_protocol_name("configured/gpt") == "openai_chat"
    assert provider.get_default_output_protocol("configured/gpt") == "anthropic_messages"
    assert provider.get_adapter_names("configured/gpt") == ("model_override",)
    assert provider.get_adapter_config("configured/gpt") == {"model_override": {"model": "upstream-model"}}
    assert provider.supports_native_streaming("configured/gpt", "chat") is True
    assert [rule.name for rule in provider.get_field_cache_rules("configured/gpt")] == ["state"]
    assert provider.get_model_quota_group("json-model") == "json_group"
    assert provider.get_model_quota_group("base-model") == "base"


def test_provider_runtime_config_can_be_bound_to_startup_snapshot(tmp_path, monkeypatch) -> None:
    class SnapshotProvider(ProviderInterface):
        provider_env_name = "snapshot_provider"

        async def get_models(self, api_key, client):
            return []

    startup = load_config_from_mapping({
        "providers": {"snapshot_provider": {"protocol_name": "gemini"}}
    })
    changed_path = _write_config(
        tmp_path,
        {"providers": {"snapshot_provider": {"protocol_name": "responses"}}},
    )
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", changed_path)
    provider = SnapshotProvider()
    provider.bind_runtime_config(startup)

    assert provider.get_protocol_name("model") == "gemini"


def test_provider_singleton_rejects_incompatible_snapshot_rebinding() -> None:
    class BoundProvider(ProviderInterface):
        provider_env_name = "bound_provider"

        async def get_models(self, api_key, client):
            return []

    first = load_config_from_mapping({"providers": {"bound_provider": {"protocol_name": "gemini"}}})
    equivalent = load_config_from_mapping({"providers": {"bound_provider": {"protocol_name": "gemini"}}})
    incompatible = load_config_from_mapping({"providers": {"bound_provider": {"protocol_name": "responses"}}})
    provider = BoundProvider()

    provider.bind_runtime_config(first)
    provider.bind_runtime_config(equivalent)
    with pytest.raises(RuntimeError, match="already bound"):
        provider.bind_runtime_config(incompatible)


def test_provider_json_quota_groups_still_allow_env_override(tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path, {"providers": {"configured": {"model_quota_groups": {"json_group": ["json-model"]}}}})
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", config_path)
    monkeypatch.setenv("QUOTA_GROUPS_CONFIGURED_JSON_GROUP", "env-model")

    assert ConfiguredProvider().get_model_quota_group("env-model") == "json_group"
    assert ConfiguredProvider().get_model_quota_group("json-model") is None


def test_priority_provider_overrides_respect_json_adapter_config_and_streaming(tmp_path, monkeypatch) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "providers": {
                "claude_code": {
                    "native_streaming_supported": True,
                    "adapter_config": {"suppress_developer_role": {"mode": "assistant"}},
                },
                "copilot": {
                    "native_streaming_supported": True,
                    "adapter_config": {"suppress_developer_role": {"mode": "user"}},
                },
                "codex": {"native_streaming_supported": True},
            }
        },
    )
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", config_path)

    assert ClaudeCodeProvider().supports_native_streaming("claude_code/claude", "messages") is True
    assert ClaudeCodeProvider().get_adapter_config("claude_code/claude")["suppress_developer_role"]["mode"] == "assistant"
    assert CopilotProvider().supports_native_streaming("copilot/gpt", "chat") is True
    assert CopilotProvider().get_adapter_config("copilot/gpt")["suppress_developer_role"]["mode"] == "user"
    assert CodexProvider().supports_native_streaming("codex/gpt", "responses") is True


@pytest.mark.asyncio
async def test_config_defined_provider_can_use_any_native_protocol(tmp_path, monkeypatch) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "providers": {
                "configured_native": {
                    "api_base": "https://native.example/v1beta",
                    "protocol_name": "gemini",
                    "default_output_protocol": "anthropic_messages",
                    "auth_mode": "x-goog-api-key",
                    "models": ["gemini-custom"],
                    "native_streaming_supported": True,
                    "endpoint_paths": {
                        "generate": "/models/{model}:generateContent",
                        "stream_generate": "/models/{model}:streamGenerateContent?alt=sse",
                    },
                }
            }
        },
    )
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", config_path)
    provider = _create_dynamic_plugin_class("configured_native")()

    assert provider.get_protocol_name("configured_native/gemini-custom") == "gemini"
    assert provider.get_default_output_protocol("configured_native/gemini-custom") == "anthropic_messages"
    assert provider.get_native_operation("gemini-custom", stream=True) == "stream_generate"
    assert provider.get_native_endpoint("configured_native/gemini-custom", "generate") == (
        "https://native.example/v1beta/models/gemini-custom:generateContent"
    )
    assert provider.get_native_headers("secret", operation="generate")["x-goog-api-key"] == "secret"
    assert provider.supports_native_streaming("configured_native/gemini-custom", "stream_generate") is True
    assert await provider.get_models("secret", object()) == ["configured_native/gemini-custom"]


def test_config_defined_provider_is_registered_without_api_base_env(tmp_path, monkeypatch) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "providers": {
                "config_only_native": {
                    "api_base": "https://native.example/v1",
                    "protocol_name": "anthropic_messages",
                    "models": ["claude-custom"],
                }
            }
        },
    )
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", config_path)
    monkeypatch.delenv("CONFIG_ONLY_NATIVE_API_BASE", raising=False)

    try:
        _register_providers()
        plugin = PROVIDER_PLUGINS["config_only_native"]()
        assert plugin.get_protocol_name("config_only_native/claude-custom") == "anthropic_messages"
        assert plugin.get_api_base() == "https://native.example/v1"
    finally:
        PROVIDER_PLUGINS.pop("config_only_native", None)


def test_file_provider_rejects_silently_ignored_custom_transport_keys(tmp_path, monkeypatch) -> None:
    config_path = _write_config(
        tmp_path,
        {"providers": {"openai": {"api_base": "https://ignored.example/v1"}}},
    )
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", config_path)

    with pytest.raises(ValueError, match="implemented in code"):
        _register_providers()


def test_custom_provider_rejects_non_generative_protocol(tmp_path, monkeypatch) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "providers": {
                "embedding_only": {
                    "api_base": "https://embedding.example/v1",
                    "protocol_name": "embeddings",
                }
            }
        },
    )
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", config_path)

    with pytest.raises(ValueError, match="supported generative protocol"):
        _register_providers()


def test_dynamic_provider_transport_config_is_snapshotted_at_construction(tmp_path, monkeypatch) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "providers": {
                "snapshot_native": {
                    "api_base": "https://first.example/v1",
                    "protocol_name": "responses",
                }
            }
        },
    )
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", config_path)
    provider = _create_dynamic_plugin_class("snapshot_native")()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "providers": {
                    "snapshot_native": {
                        "api_base": "https://second.example/v1",
                        "protocol_name": "gemini",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert provider.get_api_base() == "https://first.example/v1"
    assert provider.get_protocol_name("snapshot_native/model") == "responses"


def test_no_auth_custom_provider_gets_internal_selection_slot_without_dummy_secret() -> None:
    credentials = {}
    config = load_config_from_mapping(
        {
            "providers": {
                "local_native": {
                    "api_base": "http://localhost:9000/v1",
                    "protocol_name": "openai_chat",
                    "auth_mode": "none",
                }
            }
        }
    )

    _add_configured_no_auth_credentials(
        credentials,
        {"local_native"},
        config=config,
    )

    assert credentials == {"local_native": ["__proxy_no_auth__"]}


def test_registered_dynamic_provider_binds_config_before_lazy_instantiation(tmp_path, monkeypatch) -> None:
    config_path = _write_config(
        tmp_path,
        {
            "providers": {
                "registered_snapshot": {
                    "api_base": "https://first.example/v1",
                    "protocol_name": "responses",
                }
            }
        },
    )
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", config_path)

    try:
        _register_providers()
        plugin_class = PROVIDER_PLUGINS["registered_snapshot"]
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "providers": {
                        "registered_snapshot": {
                            "api_base": "https://second.example/v1",
                            "protocol_name": "gemini",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        provider = plugin_class()

        assert provider.get_api_base() == "https://first.example/v1"
        assert provider.get_protocol_name("registered_snapshot/model") == "responses"
    finally:
        PROVIDER_PLUGINS.pop("registered_snapshot", None)


@pytest.mark.asyncio
async def test_no_auth_slot_is_preserved_in_classifier_scope(tmp_path) -> None:
    created = []

    async def ensure(provider, classifier, credentials):
        created.append((provider, classifier, credentials))
        return f"classifier:{classifier}:{provider}"

    manager = ScopeManager(
        all_credentials={"local_native": [NO_AUTH_CREDENTIAL]},
        usage_base_path=tmp_path,
        fingerprint_key=b"test",
        model_list_cache={},
        ensure_scoped_usage_manager=ensure,
    )

    scope = await manager.resolve_scope_for_provider(
        "local_native",
        "tenant-a",
        None,
        None,
        True,
    )

    assert scope["credentials"] == [NO_AUTH_CREDENTIAL]
    assert scope["credential_secrets"] == {}
    assert created == [("local_native", "tenant-a", [NO_AUTH_CREDENTIAL])]
