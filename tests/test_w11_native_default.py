"""W11 acceptance: native-by-default execution for declared providers.

LiteLLM is an explicit, logged fallback — never a silent one. Every built-in
generative provider (except custom-logic providers like deepseek, which keep
their verified custom path until manually migrated) must resolve to native
execution in auto mode, for both streaming and non-streaming.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotator_library.client.executor import (
    _should_use_native_protocol,
    _should_use_native_streaming,
)
from rotator_library.providers import PROVIDER_PLUGINS


#: Built-in generative providers that flipped native-by-default in W11.
NATIVE_DEFAULT_PROVIDERS = [
    "openai",
    "openrouter",
    "groq",
    "mistral",
    "cohere",
    "chutes",
    "nanogpt",
    "firmware",
    "nvidia_nim",
    "gemini",
]

#: Custom-logic providers keep their verified custom path (custom-first).
CUSTOM_FIRST_PROVIDERS = ["deepseek"]


def _plugin(name: str):
    plugin = PROVIDER_PLUGINS.get(name)
    assert plugin is not None, f"provider {name} missing from registry"
    return plugin()


@pytest.mark.parametrize("provider", NATIVE_DEFAULT_PROVIDERS)
def test_generative_providers_resolve_native_by_default(provider: str) -> None:
    plugin = _plugin(provider)
    model = f"{provider}/test-model"
    kwargs = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
    assert _should_use_native_protocol(plugin, model, None, kwargs, stream=False, execution="auto") is True


@pytest.mark.parametrize("provider", NATIVE_DEFAULT_PROVIDERS)
def test_generative_providers_stream_native_by_default(provider: str) -> None:
    plugin = _plugin(provider)
    model = f"{provider}/test-model"
    assert _should_use_native_streaming(plugin, model, None, "auto", provider) is True


@pytest.mark.parametrize("provider", CUSTOM_FIRST_PROVIDERS)
def test_custom_logic_providers_keep_custom_path(provider: str) -> None:
    plugin = _plugin(provider)
    assert plugin.has_custom_logic() is True


@pytest.mark.parametrize("provider", NATIVE_DEFAULT_PROVIDERS)
def test_declared_endpoints_are_absolute_and_protocol_shaped(provider: str) -> None:
    plugin = _plugin(provider)
    protocol = plugin.get_protocol_name("m")
    endpoint = plugin.get_native_endpoint(model="test-model", operation="chat")
    assert endpoint.startswith("https://")
    if protocol == "openai_chat":
        assert endpoint.endswith("/chat/completions")
    elif protocol == "gemini":
        assert ":generateContent" in endpoint
        stream_endpoint = plugin.get_native_endpoint(model="test-model", operation=plugin.get_native_operation("test-model", None, stream=True))
        assert "streamGenerateContent" in stream_endpoint and "alt=sse" in stream_endpoint


def test_env_api_base_override_wins_over_class_default(monkeypatch) -> None:
    plugin = _plugin("openai")
    monkeypatch.setenv("OPENAI_API_BASE", "https://override.example.internal/v1")
    assert plugin.get_provider_api_base() == "https://override.example.internal/v1"
    assert plugin.get_native_endpoint(model="gpt-test") == "https://override.example.internal/v1/chat/completions"


def test_bearer_auth_header_default() -> None:
    plugin = _plugin("openai")
    assert plugin.get_native_headers("sk-secret") == {"Authorization": "Bearer sk-secret"}


def test_gemini_auth_header_shape() -> None:
    plugin = _plugin("gemini")
    assert plugin.get_native_headers("AIza-key") == {"x-goog-api-key": "AIza-key"}


def test_litellm_fallback_identity_recorded_when_protocol_available() -> None:
    """The no-accidental-fallback guard: when LiteLLM executes a request for a
    provider that declared a native protocol, the fallback identity must be
    visible (metadata attempt record + warning log), never silent."""

    from rotator_library.client.executor import RequestExecutor
    from rotator_library.core.types import RequestContext

    executor = RequestExecutor.__new__(RequestExecutor)
    recorder = []

    class _Logger:
        def record_attempt(self, record):
            recorder.append(("attempt", record))

        def update_metadata(self, **fields):
            recorder.append(("metadata", fields))

    class _Context:
        transaction_logger = _Logger()

    plugin = _plugin("openai")
    executor._record_litellm_fallback_identity(_Context(), "openai", plugin, "openai/gpt-test", stream=False)
    assert any(entry[0] == "attempt" and entry[1]["execution"] == "litellm_fallback" for entry in recorder)
    assert any(entry[0] == "metadata" and entry[1].get("execution_mode") == "litellm_fallback" for entry in recorder)


def test_litellm_identity_silent_for_undeclared_providers() -> None:
    """Providers without a native declaration running LiteLLM are the normal
    path — no fallback warning is recorded for them."""

    from rotator_library.client.executor import RequestExecutor

    executor = RequestExecutor.__new__(RequestExecutor)
    recorder = []

    class _Logger:
        def record_attempt(self, record):
            recorder.append(record)

        def update_metadata(self, **fields):
            recorder.append(fields)

    class _Context:
        transaction_logger = _Logger()

    plugin = _plugin("deepseek")
    executor._record_litellm_fallback_identity(_Context(), "deepseek", plugin, "deepseek/chat", stream=False)
    assert recorder == []
