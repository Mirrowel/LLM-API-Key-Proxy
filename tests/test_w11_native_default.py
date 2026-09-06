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


def test_litellm_identity_fires_on_explicit_fallback_branch() -> None:
    """Explicit @litellm routing on a declared provider records the fallback
    identity (both execution modes stay visible, never silent)."""

    import asyncio

    from rotator_library.client.executor import RequestExecutor, _current_route_target

    executor = RequestExecutor.__new__(RequestExecutor)
    recorder = []

    class _Logger:
        def record_attempt(self, record):
            recorder.append(record)

        def update_metadata(self, **fields):
            recorder.append(fields)

    class _Context:
        transaction_logger = _Logger()
        input_protocol_name = "openai_chat"
        input_provider = "openai"
        protocol_request = None
        unified_request = None
        usage_manager_key = "test"
        classifier = None
        session_id = None
        disable_provider_continuation = False

    class _Target:
        execution = "litellm_fallback"
        provider = "openai"
        protocol = "openai_chat"
        name = "openai"
        prefixed_model = "openai/gpt-test"

    import rotator_library.client.executor as executor_module

    original = executor_module._current_route_target
    executor_module._current_route_target = lambda ctx: _Target()
    try:
        plugin = _plugin("openai")

        async def _fake_litellm(*args, **kwargs):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

        executor._execute_litellm_request = _fake_litellm  # type: ignore[method-assign]
        executor._log_routing_trace = lambda *a, **k: None  # type: ignore[method-assign]
        executor._log_executor_trace = lambda *a, **k: None  # type: ignore[method-assign]
        executor._format_execution_response = lambda response, protocol, ctx: response  # type: ignore[method-assign]

        asyncio.run(
            executor._execute_provider_request(  # type: ignore[arg-type]
                "openai",
                "openai/gpt-test",
                plugin,
                "sk-test",
                "cred-1",
                {"model": "openai/gpt-test", "messages": []},
                _Context(),
            )
        )
    finally:
        executor_module._current_route_target = original
    assert any(
        isinstance(entry, dict) and entry.get("execution") == "litellm_fallback"
        for entry in recorder
    )


def test_native_chat_stream_requests_terminal_usage() -> None:
    """OpenAI-compatible native streams must ask for include_usage: without
    it the provider omits usage from the final chunk and accounting zeros."""

    import asyncio

    from rotator_library.native_provider.context import NativeProviderContext
    from rotator_library.native_provider.executor import NativeProviderExecutor

    sent = {}

    class _Transport:
        async def stream_json_lines(self, endpoint, *, headers, payload):
            sent.update(payload)

            async def _frames():
                yield {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"content": "hi"}}]}
                yield {
                    "id": "chatcmpl-1",
                    "choices": [],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }

            yield_frame = _frames()
            while True:
                try:
                    yield await yield_frame.__anext__()
                except StopAsyncIteration:
                    return

    context = NativeProviderContext(
        provider="openai",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://api.openai.test/v1/chat/completions",
        operation="chat",
        input_protocol_name="openai_chat",
        client_protocol_name="openai_chat",
        raw_client_request={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers={},
        credential_id="cred-1",
        transport="sse",
    )
    context.unified_request = None
    executor = NativeProviderExecutor()

    async def _consume():
        async for _event in executor.stream({"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True}, context, _Transport()):
            pass

    # The unified request must exist for the native stream path; build one.
    from rotator_library.protocols import get_protocol

    unified = get_protocol("openai_chat").parse_request(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        None,
    )

    async def _consume_with_unified():
        context.unified_request = unified
        async for _event in executor.stream(
            {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            context,
            _Transport(),
        ):
            pass

    asyncio.run(_consume_with_unified())
    assert sent.get("stream_options", {}).get("include_usage") is True
    assert any(
        overlay.get("field") == "stream_options.include_usage"
        for overlay in (context.request_transport_overlays or [])
    )


def test_native_gemini_stream_injects_no_chat_stream_options() -> None:
    """Gemini streams always carry usageMetadata — no chat-wire field is
    injected into the gemini payload."""

    from rotator_library.native_provider.context import NativeProviderContext
    from rotator_library.native_provider.executor import NativeProviderExecutor

    sent = {}

    class _Transport:
        async def stream_json_lines(self, endpoint, *, headers, payload):
            sent.update(payload)

            async def _frames():
                yield {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}}

            frame_iter = _frames()
            while True:
                try:
                    yield await frame_iter.__anext__()
                except StopAsyncIteration:
                    return

    from rotator_library.protocols import get_protocol

    unified = get_protocol("gemini").parse_request(
        {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        None,
    )
    unified.stream = True
    context = NativeProviderContext(
        provider="gemini",
        model="gemini-test",
        protocol_name="gemini",
        endpoint="https://gemini.test/v1beta/models/gemini-test:streamGenerateContent?alt=sse",
        operation="stream_generate",
        input_protocol_name="openai_chat",
        client_protocol_name="openai_chat",
        headers={},
        credential_id="cred-1",
        transport="sse",
    )
    context.unified_request = unified
    executor = NativeProviderExecutor()

    import asyncio

    async def _consume():
        async for _event in executor.stream({"contents": []}, context, _Transport()):
            pass

    asyncio.run(_consume())
    assert "stream_options" not in sent
