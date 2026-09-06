from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from proxy_app import main as proxy_main
from rotator_library.client.gemini import GeminiHandler


class SurfaceClient:
    """Minimal route client that records client-protocol handoff details."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def agenerate(self, payload, *, input_protocol, request=None, **kwargs):
        self.calls.append(
            {
                "payload": payload,
                "kwargs": dict(kwargs),
                "input_protocol": input_protocol,
            }
        )
        if payload.get("stream"):

            async def stream():
                yield 'data: {"id":"chat_surface_stream","object":"chat.completion.chunk","choices":[{"delta":{"content":"streamed"}}]}\n\n'

            return stream()
        return {
            "id": "chat_surface",
            "object": "chat.completion",
            "model": payload["model"],
            "choices": [{"message": {"role": "assistant", "content": "chat"}, "finish_reason": "stop"}],
        }

    async def gemini_generate(self, payload, *, model, raw_request=None):
        self.calls.append(
            {
                "payload": payload,
                "model": model,
                "input_protocol": "gemini",
            }
        )
        return {
            "responseId": "gemini_surface",
            "modelVersion": model,
            "candidates": [{"content": {"role": "model", "parts": [{"text": "gemini"}]}, "finishReason": "STOP"}],
        }

    async def gemini_stream_generate(self, payload, *, model, raw_request=None):
        self.calls.append({"payload": payload, "model": model, "operation": "stream_generate"})

        async def stream():
            yield 'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"gemini-stream"}]}}]}\n\n'

        return stream()

    def gemini_count_tokens(self, payload, *, model):
        self.calls.append({"payload": payload, "model": model, "operation": "count_tokens"})
        return {"totalTokens": 7}


def _surface_client() -> tuple[TestClient, SurfaceClient]:
    proxy_main.PROXY_API_KEY = None
    proxy_main.ENABLE_RAW_LOGGING = False
    rotating = SurfaceClient()
    proxy_main.app.state.rotating_client = rotating
    return TestClient(proxy_main.app), rotating


def test_chat_route_uses_client_request_protocol_without_override_machinery() -> None:
    client, rotating = _surface_client()

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Proxy-Output-Protocol": "anthropic_messages"},
        json={"model": "openai/gpt-test", "messages": [{"role": "user", "content": "hello"}]},
    )

    # D1: the response protocol equals the request protocol regardless of any
    # client-supplied header; unknown headers are plain data and never honored.
    assert response.status_code == 200
    assert response.json()["object"] == "chat.completion"
    assert rotating.calls[0]["input_protocol"] == "openai_chat"


def test_spoofed_output_headers_are_ignored_for_streams_and_unknown_values() -> None:
    client, _ = _surface_client()

    unknown = client.post(
        "/v1/chat/completions",
        headers={"X-Proxy-Output-Protocol": "garbage"},
        json={"model": "openai/gpt-test", "messages": [{"role": "user", "content": "hello"}]},
    )
    streamed = client.post(
        "/v1/chat/completions",
        headers={"X-Proxy-Output-Protocol": "gemini"},
        json={"model": "openai/gpt-test", "messages": [{"role": "user", "content": "hello"}], "stream": True},
    )

    assert unknown.status_code == 200
    assert unknown.json()["object"] == "chat.completion"
    assert streamed.status_code == 200
    assert "chat.completion.chunk" in streamed.text


def test_no_proxy_internal_kwargs_leak_into_client_payload() -> None:
    client, rotating = _surface_client()

    client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-test", "messages": [{"role": "user", "content": "hello"}]},
    )

    kwargs = rotating.calls[0]["kwargs"]
    leaked = [key for key in (*kwargs.keys(), *rotating.calls[0]["payload"].keys()) if key.startswith("_")]
    assert leaked == []


def test_gemini_generate_and_count_routes_preserve_native_client_shape() -> None:
    client, rotating = _surface_client()
    payload = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}

    generated = client.post("/v1beta/models/gemini-2.5-pro:generateContent", json=payload)
    counted = client.post("/v1beta/models/gemini-2.5-pro:countTokens", json=payload)

    assert generated.status_code == 200
    assert generated.json()["candidates"][0]["content"]["parts"][0]["text"] == "gemini"
    assert counted.json() == {"totalTokens": 7}
    assert rotating.calls[0]["input_protocol"] == "gemini"
    assert rotating.calls[0]["payload"] == payload
    assert rotating.calls[1]["operation"] == "count_tokens"


def test_gemini_stream_generate_route_preserves_native_stream_shape() -> None:
    client, rotating = _surface_client()

    response = client.post(
        "/v1beta/models/gemini-2.5-pro:streamGenerateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
    )

    assert response.status_code == 200
    assert "gemini-stream" in response.text
    assert rotating.calls[0]["operation"] == "stream_generate"


def test_gemini_generate_rejects_stream_flag_until_stream_route_exists() -> None:
    client, _ = _surface_client()

    response = client.post(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}], "stream": True},
    )

    assert response.status_code == 400
    assert response.json()["error"]["status"] == "INVALID_ARGUMENT"


class FailingSurfaceClient(SurfaceClient):
    async def agenerate(self, payload, **kwargs):
        raise ValueError("local validation failed")

    async def anthropic_messages(self, body, **kwargs):
        raise ValueError("local validation failed")

    async def gemini_generate(self, payload, **kwargs):
        raise ValueError("local validation failed")


@pytest.mark.parametrize(
    ("path", "body", "assertion"),
    (
        (
            "/v1/chat/completions",
            {"model": "openai/gpt-test", "messages": [{"role": "user", "content": "hello"}]},
            lambda payload: payload["error"]["type"] == "invalid_request",
        ),
        (
            "/v1/messages",
            {"model": "claude-test", "max_tokens": 8, "messages": [{"role": "user", "content": "hello"}]},
            lambda payload: payload["type"] == "error" and payload["error"]["type"] == "invalid_request_error",
        ),
        (
            "/v1beta/models/gemini-2.5-pro:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
            lambda payload: payload["error"]["status"] == "INVALID_ARGUMENT",
        ),
    ),
)
def test_proxy_side_errors_use_each_routes_own_protocol(path, body, assertion) -> None:
    proxy_main.PROXY_API_KEY = None
    proxy_main.ENABLE_RAW_LOGGING = False
    proxy_main.app.state.rotating_client = FailingSurfaceClient()

    response = TestClient(proxy_main.app).post(path, json=body)

    assert response.status_code == 400
    assert assertion(response.json())
    assert "detail" not in response.json()


def test_responses_malformed_json_uses_responses_error_shape() -> None:
    client, _ = _surface_client()

    response = client.post(
        "/v1/responses",
        headers={"Content-Type": "application/json"},
        content="{",
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request"
    assert "detail" not in response.json()


def test_cost_estimate_degrades_gracefully_without_pricing_service() -> None:
    """The route answers even when no model-info service is wired (or its
    pricing lookup fails) — never a 500."""

    class UnknownPricingService:
        def estimate_cost(self, *args, **kwargs):
            raise RuntimeError("registry exploded")

    proxy_main.PROXY_API_KEY = None
    proxy_main.app.state.rotating_client = FailingSurfaceClient()
    proxy_main.app.state.model_info_service = UnknownPricingService()
    try:
        response = TestClient(proxy_main.app).post(
            "/v1/cost-estimate",
            json={"model": "unknown/model-x", "prompt_tokens": 10, "completion_tokens": 5},
        )
        assert response.status_code == 200
        assert response.json()["model"] == "unknown/model-x"
    finally:
        # The exploding service must not poison later tests.
        proxy_main.app.state.model_info_service = _GracefulPricingService()


class _GracefulPricingService:
    def estimate_cost(self, *args, **kwargs):
        return {"cost": None, "currency": "USD", "pricing": {}, "source": "unknown", "error": "Pricing data not available for this model"}


class GeminiRuntimeClient:
    def __init__(self) -> None:
        self.calls = []

    async def agenerate(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        return {"candidates": []}

    def token_count(self, *, model, messages=None, text=None):
        return 5 if messages is not None else 2


def test_gemini_handler_defaults_bare_models_but_preserves_model_routes(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ROUTE_ALIAS", "openai/gpt-test")
    client = GeminiRuntimeClient()
    handler = GeminiHandler(client)

    assert handler._routable_model("gemini-2.5-pro") == "gemini/gemini-2.5-pro"
    assert handler._routable_model("alias") == "alias"
    assert handler._routable_model("configured/model") == "configured/model"


@pytest.mark.asyncio
async def test_gemini_handler_forwards_raw_request_to_generic_runtime() -> None:
    client = GeminiRuntimeClient()
    handler = GeminiHandler(client)
    request = SimpleNamespace(headers={})

    await handler.generate(
        {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        model="gemini-2.5-pro",
        raw_request=request,
    )

    assert client.calls[0][1]["input_protocol"] == "gemini"
    assert client.calls[0][1]["request"] is request
