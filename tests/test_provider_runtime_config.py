from __future__ import annotations

import json

from rotator_library.providers.provider_interface import ProviderInterface


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
    assert provider.get_adapter_names("configured/gpt") == ("model_override",)
    assert provider.get_adapter_config("configured/gpt") == {"model_override": {"model": "upstream-model"}}
    assert provider.supports_native_streaming("configured/gpt", "chat") is True
    assert [rule.name for rule in provider.get_field_cache_rules("configured/gpt")] == ["state"]
    assert provider.get_model_quota_group("json-model") == "json_group"


def test_provider_json_quota_groups_still_allow_env_override(tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path, {"providers": {"configured": {"model_quota_groups": {"json_group": ["json-model"]}}}})
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", config_path)
    monkeypatch.setenv("QUOTA_GROUPS_CONFIGURED_JSON_GROUP", "env-model")

    assert ConfiguredProvider().get_model_quota_group("env-model") == "json_group"
    assert ConfiguredProvider().get_model_quota_group("json-model") is None
