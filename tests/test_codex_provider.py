from __future__ import annotations

import pytest

from rotator_library.providers import PROVIDER_PLUGINS
from rotator_library.providers.codex_provider import CodexProvider


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

    async def get(self, url, *, headers, timeout):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        return FakeResponse(self.payload)


def test_codex_provider_is_discovered() -> None:
    assert "codex" in PROVIDER_PLUGINS


def test_codex_provider_declares_responses_protocol_and_cache_rule() -> None:
    provider = CodexProvider()

    assert provider.get_protocol_name("codex-mini-latest") == "responses"
    assert provider.get_adapter_names("codex-mini-latest") == ()
    rules = provider.get_field_cache_rules("codex-mini-latest")
    assert rules[0].name == "codex_previous_response_id"
    assert rules[0].inject.path == "previous_response_id"


def test_codex_provider_builds_native_headers_and_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("CODEX_API_BASE", "https://codex.test")
    provider = CodexProvider()

    assert provider.get_native_endpoint(operation="responses") == "https://codex.test/v1/responses"
    assert provider.get_native_endpoint(operation="models") == "https://codex.test/v1/models"
    assert provider.get_native_headers("token") == {"Authorization": "Bearer token", "content-type": "application/json"}


@pytest.mark.asyncio
async def test_codex_provider_get_models_filters_codex_models(monkeypatch) -> None:
    monkeypatch.setenv("CODEX_API_BASE", "https://codex.test")
    client = FakeClient({"data": [{"id": "gpt-5.1-codex"}, {"id": "gpt-5.1"}]})
    provider = CodexProvider()

    models = await provider.get_models("token", client)

    assert models == ["codex/gpt-5.1-codex"]
    assert client.calls[0]["url"] == "https://codex.test/v1/models"


@pytest.mark.asyncio
async def test_codex_provider_get_models_falls_back_on_errors() -> None:
    class BrokenClient:
        async def get(self, *args, **kwargs):
            raise RuntimeError("offline")

    assert await CodexProvider().get_models("token", BrokenClient()) == ["codex/codex-mini-latest", "codex/gpt-5.1-codex"]
