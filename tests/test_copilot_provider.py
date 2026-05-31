from __future__ import annotations

import pytest

from rotator_library.adapters import AdapterContext, get_adapter, run_adapter_chain
from rotator_library.providers import PROVIDER_PLUGINS
from rotator_library.providers.copilot_provider import CopilotProvider


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


def test_copilot_provider_is_discovered() -> None:
    assert "copilot" in PROVIDER_PLUGINS


def test_copilot_provider_declares_openai_chat_protocol_and_adapter() -> None:
    provider = CopilotProvider()

    assert provider.get_protocol_name("gpt-4.1") == "openai_chat"
    assert provider.get_adapter_names("gpt-4.1") == ("suppress_developer_role",)
    assert provider.get_adapter_config("gpt-4.1") == {"suppress_developer_role": {"mode": "system"}}
    assert provider.get_field_cache_rules("gpt-4.1") == ()


@pytest.mark.asyncio
async def test_copilot_adapter_config_converts_developer_role_to_system() -> None:
    provider = CopilotProvider()

    result = await run_adapter_chain(
        [get_adapter("suppress_developer_role")],
        {"messages": [{"role": "developer", "content": "rules"}]},
        AdapterContext(adapter_config=provider.get_adapter_config("gpt-4.1")),
        stage="request",
    )

    assert result["messages"][0]["role"] == "system"


def test_copilot_provider_builds_native_headers_and_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("COPILOT_API_BASE", "https://copilot.test")
    monkeypatch.setenv("COPILOT_INTEGRATION_ID", "proxy-test")
    provider = CopilotProvider()

    assert provider.get_native_endpoint(operation="chat") == "https://copilot.test/chat/completions"
    assert provider.get_native_endpoint(operation="models") == "https://copilot.test/models"
    assert provider.get_native_headers("token") == {
        "Authorization": "Bearer token",
        "content-type": "application/json",
        "Copilot-Integration-Id": "proxy-test",
    }


def test_copilot_native_operation_model_and_stream_support() -> None:
    provider = CopilotProvider()

    assert provider.get_native_operation("gpt-4.1", {}, stream=False) == "chat"
    assert provider.normalize_native_model("copilot/gpt-4.1") == "gpt-4.1"
    assert provider.supports_native_streaming("gpt-4.1", operation="chat") is False
    assert provider.supports_native_streaming("gpt-4.1", operation="responses") is False


@pytest.mark.asyncio
async def test_copilot_provider_get_models_uses_mocked_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("COPILOT_API_BASE", "https://copilot.test")
    client = FakeClient({"data": [{"id": "gpt-4.1"}]})
    provider = CopilotProvider()

    models = await provider.get_models("token", client)

    assert models == ["copilot/gpt-4.1"]
    assert client.calls[0]["url"] == "https://copilot.test/models"


@pytest.mark.asyncio
async def test_copilot_provider_get_models_falls_back_on_errors() -> None:
    class BrokenClient:
        async def get(self, *args, **kwargs):
            raise RuntimeError("offline")

    assert await CopilotProvider().get_models("token", BrokenClient()) == ["copilot/gpt-4.1", "copilot/claude-sonnet-4-5"]
