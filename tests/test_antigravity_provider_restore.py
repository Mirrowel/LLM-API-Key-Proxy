from __future__ import annotations

import pytest

from rotator_library.providers import PROVIDER_PLUGINS
from rotator_library.providers.antigravity_provider import AntigravityProvider


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def post(self, url, *, headers, json, timeout):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse(self.payload)


def test_antigravity_provider_is_discovered() -> None:
    assert "antigravity" in PROVIDER_PLUGINS


def test_antigravity_provider_restores_safe_declarations() -> None:
    provider = AntigravityProvider()

    assert provider.get_protocol_name("gemini-3-flash") == "gemini"
    assert provider.get_adapter_names("gemini-3-flash") == ()
    assert provider.get_adapter_config("gemini-3-flash") == {}
    assert provider.get_model_tier_requirement("antigravity/gemini-3-flash") is None
    rules = provider.get_field_cache_rules("gemini-3-flash")
    assert rules[0].name == "antigravity_thought_signature"
    assert rules[0].scope == ("provider", "model", "credential", "session")


def test_antigravity_provider_builds_static_headers_without_device_profile(monkeypatch) -> None:
    monkeypatch.setenv("ANTIGRAVITY_API_BASE", "https://antigravity.test/v1internal")
    provider = AntigravityProvider()

    headers = provider.get_native_headers("token")

    assert provider.get_native_endpoint(operation="generate") == "https://antigravity.test/v1internal:streamGenerateContent?alt=sse"
    assert provider.get_native_endpoint(operation="models") == "https://antigravity.test/v1internal:fetchAvailableModels"
    assert headers["Authorization"] == "Bearer token"
    assert headers["X-Goog-Api-Client"] == "google-cloud-sdk vscode_cloudshelleditor/0.1"
    assert "X-Client-Device-Id" not in headers


def test_antigravity_model_aliases_and_tracking_normalization() -> None:
    provider = AntigravityProvider()

    assert provider._alias_to_internal("claude-sonnet-4.5") == "claude-sonnet-4-5"
    assert provider.normalize_native_model("antigravity/claude-sonnet-4.5") == "claude-sonnet-4-5"
    assert provider.normalize_model_for_tracking("antigravity/claude-sonnet-4-5") == "antigravity/claude-sonnet-4.5"


def test_antigravity_native_operation_model_and_stream_support() -> None:
    provider = AntigravityProvider()

    assert provider.get_native_operation("gemini-3-flash", {}, stream=False) == "generate"
    assert provider.get_native_operation("gemini-3-flash", {}, stream=True) == "stream_generate"
    assert provider.supports_native_streaming("gemini-3-flash", operation="stream_generate") is True
    assert provider.supports_native_streaming("gemini-3-flash", operation="generate") is False
    assert provider.prepare_native_request({"model": "antigravity/gemini-3-pro-low"}, model="gemini-3-pro-preview", operation="generate")["model"] == "gemini-3-pro-preview"


@pytest.mark.asyncio
async def test_antigravity_get_models_filters_and_aliases_mocked_response(monkeypatch) -> None:
    monkeypatch.setenv("ANTIGRAVITY_API_BASE", "https://antigravity.test/v1internal")
    client = FakeClient({"models": {"gemini-3-pro-low": {}, "chat_20706": {}, "claude-opus-4-6": {}}})
    provider = AntigravityProvider()

    models = await provider.get_models("token", client)

    assert models == ["antigravity/gemini-3-pro-preview", "antigravity/claude-opus-4.6"]
    assert client.calls[0]["url"] == "https://antigravity.test/v1internal:fetchAvailableModels"
    assert client.calls[0]["headers"]["Authorization"] == "Bearer token"


@pytest.mark.asyncio
async def test_antigravity_get_models_falls_back_on_errors() -> None:
    class BrokenClient:
        async def post(self, *args, **kwargs):
            raise RuntimeError("offline")

    models = await AntigravityProvider().get_models("token", BrokenClient())

    assert "antigravity/gemini-3-flash" in models
    assert "antigravity/claude-opus-4.6" in models
