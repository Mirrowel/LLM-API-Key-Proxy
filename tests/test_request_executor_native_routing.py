from __future__ import annotations

import json

import pytest

from rotator_library.client import executor as executor_module
from rotator_library.client.executor import RequestExecutor, RoutingExecutionError
from rotator_library.core.types import RequestContext
from rotator_library.field_cache import FieldCacheInjection, FieldCacheRule
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
        return {"Authorization": f"Bearer {credential_identifier}"}

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


class CustomPlugin:
    def __init__(self):
        self.calls = []

    def has_custom_logic(self):
        return True

    async def acompletion(self, client, **kwargs):
        self.calls.append(kwargs)
        return {"id": "custom"}


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


def test_native_context_raises_on_invalid_field_cache_config(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"field_cache": {"provider": {"*": [{"name": "bad", "source": "response"}]}}}), encoding="utf-8")
    monkeypatch.setenv("LLM_PROXY_CONFIG_FILE", str(config_path))

    with pytest.raises(RoutingExecutionError) as exc:
        _executor()._build_native_provider_context("provider", "provider/gpt-test", NativePlugin(), "secret", "stable", _context(), None)

    assert exc.value.error_type == "configuration_error"


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
