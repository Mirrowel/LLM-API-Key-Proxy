from __future__ import annotations

import json

import pytest

from rotator_library.client import executor as executor_module
from rotator_library.client.executor import RequestExecutor, RoutingExecutionError
from rotator_library.core.types import RequestContext
from rotator_library.field_cache import FieldCacheInjection, FieldCacheRule
from rotator_library.providers.antigravity_provider import AntigravityProvider
from rotator_library.providers.claude_code_provider import ClaudeCodeProvider
from rotator_library.providers.codex_provider import CodexProvider
from rotator_library.providers.copilot_provider import CopilotProvider
from rotator_library.routing import parse_route_target


class FakeNativeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeHTTPClient:
    def __init__(self):
        self.calls = []

    async def post(self, endpoint, *, headers, json):
        self.calls.append({"endpoint": endpoint, "headers": headers, "json": json})
        return FakeNativeResponse({"id": "chat_native", "choices": [{"message": {"role": "assistant", "content": "ok"}}]})


class SequencedHTTPClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post(self, endpoint, *, headers, json):
        self.calls.append({"endpoint": endpoint, "headers": headers, "json": json})
        return FakeNativeResponse(self.responses.pop(0))


class NativePlugin:
    def has_custom_logic(self):
        return False

    def get_protocol_name(self, model=""):
        return "openai_chat"

    def get_native_endpoint(self, model="", operation="chat"):
        return "https://native.test/chat"

    def get_native_headers(self, credential_identifier, model="", operation="chat"):
        return {"Authorization": f"Bearer {credential_identifier}", "X-Operation": operation, "X-Model": model}

    def get_native_operation(self, model="", request=None, stream=False):
        return "messages" if stream else "chat"

    def normalize_native_model(self, model=""):
        return model.split("/", 1)[1] if "/" in model else model

    def get_adapter_names(self, model=""):
        return ()

    def get_adapter_config(self, model=""):
        return {}

    def get_field_cache_rules(self, model=""):
        return ()


class NativePluginWithRule(NativePlugin):
    def get_field_cache_rules(self, model=""):
        return (
            FieldCacheRule(
                name="state",
                source="response",
                path="choices.0.message.reasoning_content",
                inject=FieldCacheInjection(target="request", path="metadata.state"),
                allow_missing_session=True,
            ),
        )


class NativePluginWithVendorRule(NativePlugin):
    def get_field_cache_rules(self, model=""):
        return (
            FieldCacheRule(
                name="vendor_state",
                source="response",
                path="choices.0.message.vendor_state",
                inject=FieldCacheInjection(target="request", path="metadata.vendor_state"),
                allow_missing_session=True,
            ),
        )


class NativePluginWithStreamVendorRule(NativePlugin):
    def get_field_cache_rules(self, model=""):
        return (
            FieldCacheRule(
                name="vendor_state",
                source="stream_event",
                path="raw.choices.0.message.vendor_state",
                allow_missing_session=True,
            ),
        )


class CustomPlugin:
    def __init__(self):
        self.calls = []

    def has_custom_logic(self):
        return True

    async def acompletion(self, client, **kwargs):
        self.calls.append(kwargs)
        return {"id": "custom"}

    def get_protocol_name(self, model=""):
        return "gemini"

    def get_native_endpoint(self, model="", operation="chat"):
        return "https://native.test/should-not-run"

    def get_native_headers(self, credential_identifier, model="", operation="chat"):
        return {"Authorization": f"Bearer {credential_identifier}"}


def _context(target=None) -> RequestContext:
    return RequestContext(
        model="provider/gpt-test",
        provider="provider",
        kwargs={"model": "provider/gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        streaming=False,
        credentials=["cred"],
        deadline=9999999999.0,
        routing_targets=(target,) if target else None,
    )


def _provider_context(provider: str, model: str, kwargs: dict, target=None) -> RequestContext:
    return RequestContext(
        model=model,
        provider=provider,
        kwargs=kwargs,
        streaming=False,
        credentials=["cred"],
        deadline=9999999999.0,
        routing_targets=(target,) if target else None,
    )


def _executor(http_client=None) -> RequestExecutor:
    executor = RequestExecutor.__new__(RequestExecutor)
    executor._http_client = http_client or FakeHTTPClient()
    executor._apply_litellm_logger = lambda kwargs: None
    return executor


@pytest.mark.asyncio
async def test_native_declared_provider_uses_native_executor_in_auto_mode() -> None:
    http_client = FakeHTTPClient()
    target = parse_route_target("provider/gpt-test")
    context = _context(target)
    context.routing_target_index = 0

    response = await _executor(http_client)._execute_provider_request(
        "provider", "provider/gpt-test", NativePlugin(), "secret", "stable", dict(context.kwargs), context
    )

    assert response["id"] == "chat_native"
    assert http_client.calls[0]["endpoint"] == "https://native.test/chat"
    assert http_client.calls[0]["headers"]["Authorization"] == "Bearer secret"
    assert http_client.calls[0]["headers"]["X-Operation"] == "chat"
    assert http_client.calls[0]["headers"]["X-Model"] == "gpt-test"
    assert http_client.calls[0]["json"]["model"] == "gpt-test"


@pytest.mark.asyncio
async def test_auto_mode_prefers_custom_logic_over_native_declaration() -> None:
    http_client = FakeHTTPClient()
    plugin = CustomPlugin()
    context = _context(parse_route_target("provider/gpt-test"))
    context.routing_target_index = 0

    response = await _executor(http_client)._execute_provider_request(
        "provider", "provider/gpt-test", plugin, "secret", "stable", dict(context.kwargs), context
    )

    assert response == {"id": "custom"}
    assert plugin.calls[0]["credential_identifier"] == "secret"
    assert http_client.calls == []


@pytest.mark.asyncio
async def test_request_executor_reuses_native_field_cache_store() -> None:
    http_client = SequencedHTTPClient(
        [
            {"id": "chat_1", "choices": [{"message": {"role": "assistant", "content": "ok", "reasoning_content": "cached"}}]},
            {"id": "chat_2", "choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        ]
    )
    executor = _executor(http_client)
    context = _context(parse_route_target("provider/gpt-test"))
    context.routing_target_index = 0

    await executor._execute_provider_request("provider", "provider/gpt-test", NativePluginWithRule(), "secret", "stable", dict(context.kwargs), context)
    await executor._execute_provider_request("provider", "provider/gpt-test", NativePluginWithRule(), "secret", "stable", dict(context.kwargs), context)

    assert http_client.calls[1]["json"]["metadata"]["state"] == "cached"


def test_native_context_merges_json_field_cache_rules(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "field_cache": {
                    "provider": {
                        "*": [
                            {
                                "name": "state",
                                "source": "response",
                                "path": "json.path",
                                "inject": {"target": "request", "path": "metadata.state"},
                            },
                            {
                                "name": "extra",
                                "source": "response",
                                "path": "json.extra",
                                "inject": {"target": "request", "path": "metadata.extra"},
                            },
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", str(config_path))

    native_context = _executor()._build_native_provider_context(
        "provider",
        "provider/gpt-test",
        NativePluginWithRule(),
        "secret",
        "stable",
        _context(),
        None,
    )

    assert [rule.name for rule in native_context.field_cache_rules] == ["state", "extra"]
    assert native_context.field_cache_rules[0].path == "json.path"


@pytest.mark.asyncio
async def test_claude_code_provider_runs_mock_live_native_request(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_CODE_API_BASE", "https://claude-code.test")
    http_client = SequencedHTTPClient([
        {"id": "msg_1", "type": "message", "role": "assistant", "content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 1}}
    ])
    provider = ClaudeCodeProvider()
    target = parse_route_target("claude_code/claude-sonnet-4-5")
    context = _provider_context(
        "claude_code",
        "claude_code/claude-sonnet-4-5",
        {"model": "claude_code/claude-sonnet-4-5", "messages": [{"role": "developer", "content": "rules"}, {"role": "user", "content": "hi"}]},
        target,
    )
    context.routing_target_index = 0

    response = await _executor(http_client)._execute_provider_request("claude_code", context.model, provider, "secret", "stable", dict(context.kwargs), context)

    assert response["id"] == "msg_1"
    assert http_client.calls[0]["endpoint"] == "https://claude-code.test/v1/messages"
    assert http_client.calls[0]["json"]["model"] == "claude-sonnet-4-5"
    assert http_client.calls[0]["json"]["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_codex_provider_runs_mock_live_native_request(monkeypatch) -> None:
    monkeypatch.setenv("CODEX_API_BASE", "https://codex.test")
    http_client = SequencedHTTPClient([
        {"id": "resp_1", "object": "response", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]}
    ])
    provider = CodexProvider()
    target = parse_route_target("codex/gpt-5.1-codex")
    context = _provider_context("codex", "codex/gpt-5.1-codex", {"model": "codex/gpt-5.1-codex", "messages": [{"role": "user", "content": "hi"}]}, target)
    context.routing_target_index = 0

    response = await _executor(http_client)._execute_provider_request("codex", context.model, provider, "secret", "stable", dict(context.kwargs), context)

    assert response["id"] == "resp_1"
    assert http_client.calls[0]["endpoint"] == "https://codex.test/v1/responses"
    assert http_client.calls[0]["json"]["model"] == "gpt-5.1-codex"
    assert http_client.calls[0]["json"]["input"][0]["content"] == [{"type": "text", "text": "hi"}]
    assert "messages" not in http_client.calls[0]["json"]


@pytest.mark.asyncio
async def test_copilot_provider_runs_mock_live_native_request(monkeypatch) -> None:
    monkeypatch.setenv("COPILOT_API_BASE", "https://copilot.test")
    http_client = SequencedHTTPClient([
        {"id": "chat_1", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    ])
    provider = CopilotProvider()
    target = parse_route_target("copilot/gpt-4.1")
    context = _provider_context("copilot", "copilot/gpt-4.1", {"model": "copilot/gpt-4.1", "messages": [{"role": "developer", "content": "rules"}, {"role": "user", "content": "hi"}]}, target)
    context.routing_target_index = 0

    response = await _executor(http_client)._execute_provider_request("copilot", context.model, provider, "secret", "stable", dict(context.kwargs), context)

    assert response["id"] == "chat_1"
    assert http_client.calls[0]["endpoint"] == "https://copilot.test/chat/completions"
    assert http_client.calls[0]["json"]["model"] == "gpt-4.1"
    assert http_client.calls[0]["json"]["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_antigravity_provider_runs_mock_live_native_request(monkeypatch) -> None:
    monkeypatch.setenv("ANTIGRAVITY_API_BASE", "https://antigravity.test/v1internal")
    http_client = SequencedHTTPClient([
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}], "usageMetadata": {"totalTokenCount": 2}}
    ])
    provider = AntigravityProvider()
    target = parse_route_target("antigravity/claude-sonnet-4.5")
    context = _provider_context("antigravity", "antigravity/claude-sonnet-4.5", {"model": "antigravity/claude-sonnet-4.5", "messages": [{"role": "user", "content": "hi"}]}, target)
    context.routing_target_index = 0

    response = await _executor(http_client)._execute_provider_request("antigravity", context.model, provider, "secret", "stable", dict(context.kwargs), context)

    assert response["candidates"][0]["content"]["parts"][0]["text"] == "ok"
    assert http_client.calls[0]["endpoint"] == "https://antigravity.test/v1internal:generateContent"
    assert http_client.calls[0]["json"]["model"] == "claude-sonnet-4-5"
    assert http_client.calls[0]["json"]["request"]["contents"][0]["parts"][0]["text"] == "hi"
    assert http_client.calls[0]["json"]["requestType"] == "CHAT_COMPLETION"
    assert "requestId" in http_client.calls[0]["json"]
    assert "messages" not in http_client.calls[0]["json"]["request"]


def test_native_request_payload_drops_litellm_only_fields() -> None:
    payload = executor_module._native_request_payload(
        {
            "model": "provider/gpt-test",
            "messages": [{"role": "user", "content": "hi"}],
            "custom_llm_provider": "openai",
            "api_base": "https://litellm-only.test",
            "transaction_context": {"id": "trace"},
            "litellm_call_id": "call",
        }
    )

    assert payload == {"model": "provider/gpt-test", "messages": [{"role": "user", "content": "hi"}]}


def test_route_error_type_from_response_hard_stop_wins_over_retry_summary() -> None:
    response = {
        "error": {
            "type": "authentication",
            "message": "provider said quota secret-token",
            "details": {"normal_error_summary": "rate_limit quota capacity", "status_code": 401},
        }
    }

    assert executor_module._route_error_type_from_response(response) == "authentication"


def test_route_error_type_from_response_uses_structured_status_codes() -> None:
    assert executor_module._route_error_type_from_response({"error": {"code": 403}}) == "forbidden"
    assert executor_module._route_error_type_from_response({"error": {"details": {"status_code": 503}}}) == "server_error"


def test_target_failure_summary_is_structural_and_sanitized() -> None:
    summary = executor_module._target_failure_summary(parse_route_target("openai/gpt"), "rate-limit", status_code=429)

    assert summary["error_type"] == "rate_limit"
    assert summary["status_code"] == 429
    assert summary["message"] == ""


def test_explicit_native_streaming_fails_when_provider_does_not_support_it() -> None:
    target = parse_route_target("provider/gpt-test@native")

    with pytest.raises(RoutingExecutionError) as exc:
        executor_module._should_use_native_streaming(NativePlugin(), "provider/gpt-test", target, "native", "provider")

    assert exc.value.error_type == "configuration_error"


def test_antigravity_cache_injection_targets_safe_envelope() -> None:
    provider = AntigravityProvider()

    rules = provider.get_field_cache_rules("gemini-3-flash")

    assert rules[0].inject.path == "request.metadata.thoughtSignatures"


def test_native_context_raises_on_invalid_field_cache_config(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"field_cache": {"provider": {"*": [{"name": "bad", "source": "response"}]}}}), encoding="utf-8")
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", str(config_path))

    with pytest.raises(RoutingExecutionError) as exc:
        _executor()._build_native_provider_context("provider", "provider/gpt-test", NativePlugin(), "secret", "stable", _context(), None)

    assert exc.value.error_type == "configuration_error"


def test_executor_trace_redaction_uses_native_field_cache_response_paths() -> None:
    context = _context(parse_route_target("provider/gpt-test"))
    payload = {"choices": [{"message": {"role": "assistant", "content": "ok", "vendor_state": "opaque-vendor-state"}}]}

    redacted = executor_module._redact_context_field_cache_paths(payload, context, "response", NativePluginWithVendorRule())

    assert redacted["choices"][0]["message"]["vendor_state"] == "[REDACTED]"
    assert payload["choices"][0]["message"]["vendor_state"] == "opaque-vendor-state"


def test_executor_stream_trace_redaction_uses_native_field_cache_paths() -> None:
    context = _context(parse_route_target("provider/gpt-test"))
    sse_line = 'data: {"choices":[{"delta":{"content":"ok"},"message":{"vendor_state":"opaque-vendor-state"}}]}\n\n'

    redacted = executor_module._redact_stream_sse_for_trace(sse_line, context, NativePluginWithStreamVendorRule())

    parsed = json.loads(redacted[6:].strip())

    assert "opaque-vendor-state" not in redacted
    assert parsed["choices"][0]["message"]["vendor_state"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_litellm_fallback_execution_is_explicit(monkeypatch) -> None:
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return {"id": "litellm"}

    monkeypatch.setattr(executor_module.litellm, "acompletion", fake_acompletion)
    target = parse_route_target("provider/gpt-test@litellm_fallback")
    context = _context(target)
    context.routing_target_index = 0

    response = await _executor()._execute_provider_request("provider", "provider/gpt-test", NativePlugin(), "secret", "stable", dict(context.kwargs), context)

    assert response == {"id": "litellm"}
    assert calls[0]["api_key"] == "secret"


@pytest.mark.asyncio
async def test_custom_execution_mode_requires_custom_plugin() -> None:
    target = parse_route_target("provider/gpt-test@custom")
    context = _context(target)
    context.routing_target_index = 0

    plugin = CustomPlugin()

    response = await _executor()._execute_provider_request("provider", "provider/gpt-test", plugin, "secret", "stable", dict(context.kwargs), context)

    assert response == {"id": "custom"}
    assert plugin.calls[0]["credential_identifier"] == "secret"


@pytest.mark.asyncio
async def test_native_execution_mode_fails_when_provider_has_no_native_declaration() -> None:
    target = parse_route_target("provider/gpt-test@native")
    context = _context(target)
    context.routing_target_index = 0

    with pytest.raises(RoutingExecutionError) as exc:
        await _executor()._execute_provider_request("provider", "provider/gpt-test", None, "secret", "stable", dict(context.kwargs), context)

    assert exc.value.error_type == "unsupported_operation"
