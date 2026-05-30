from __future__ import annotations

import pytest

from rotator_library.providers import PROVIDER_PLUGINS
from rotator_library.providers.claude_code_provider import ClaudeCodeProvider


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


def test_claude_code_provider_is_discovered() -> None:
    assert "claude_code" in PROVIDER_PLUGINS


def test_claude_code_provider_declares_native_protocol_adapters_and_cache_rules() -> None:
    provider = ClaudeCodeProvider()

    assert provider.get_protocol_name("claude-sonnet-4-5") == "anthropic_messages"
    assert provider.get_adapter_names("claude-sonnet-4-5") == ("suppress_developer_role",)
    assert provider.get_adapter_config("claude-sonnet-4-5") == {"suppress_developer_role": {"replacement_role": "user"}}
    rules = provider.get_field_cache_rules("claude-sonnet-4-5")
    assert rules[0].name == "claude_code_thinking_signature"
    assert rules[0].scope == ("provider", "model", "credential", "session")


def test_claude_code_provider_builds_native_headers_and_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_API_BASE", "https://claude-code.test")
    monkeypatch.setenv("CLAUDE_CODE_ANTHROPIC_VERSION", "2099-01-01")
    provider = ClaudeCodeProvider()

    assert provider.get_native_endpoint(operation="messages") == "https://claude-code.test/v1/messages"
    assert provider.get_native_headers("token") == {
        "Authorization": "Bearer token",
        "anthropic-version": "2099-01-01",
        "content-type": "application/json",
    }


@pytest.mark.asyncio
async def test_claude_code_provider_get_models_uses_mocked_models_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_API_BASE", "https://claude-code.test")
    client = FakeClient({"data": [{"id": "claude-sonnet-test"}]})
    provider = ClaudeCodeProvider()

    models = await provider.get_models("token", client)

    assert models == ["claude_code/claude-sonnet-test"]
    assert client.calls[0]["url"] == "https://claude-code.test/v1/models"
    assert client.calls[0]["headers"]["Authorization"] == "Bearer token"


@pytest.mark.asyncio
async def test_claude_code_provider_get_models_falls_back_on_errors() -> None:
    class BrokenClient:
        async def get(self, *args, **kwargs):
            raise RuntimeError("offline")

    assert await ClaudeCodeProvider().get_models("token", BrokenClient()) == [
        "claude_code/claude-sonnet-4-5",
        "claude_code/claude-opus-4-5",
    ]
