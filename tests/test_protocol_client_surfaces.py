from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from proxy_app import main as proxy_main
from rotator_library.client.gemini import GeminiHandler
from rotator_library.client.rotating_client import RotatingClient
from rotator_library.client.protocol_selection import request_output_protocol


class SurfaceClient:
    """Minimal route client that records client-protocol handoff details."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def resolve_output_protocol(self, payload, *, input_protocol, request=None, explicit=None):
        return explicit or request_output_protocol(request) or input_protocol

    async def agenerate(self, payload, *, input_protocol, request=None, **kwargs):
        self.resolve_output_protocol(
            payload,
            input_protocol=input_protocol,
            request=request,
            explicit=kwargs.get("output_protocol"),
        )
        self.calls.append(
            {
                "payload": payload,
                "input_protocol": input_protocol,
                "output_header": request.headers.get("X-Proxy-Output-Protocol"),
            }
        )
        if payload.get("stream"):
            async def stream():
                yield 'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"streamed"}]}}]}\n\n'
            return stream()
        if request.headers.get("X-Proxy-Output-Protocol") == "anthropic_messages":
            return {
                "id": "msg_surface",
                "type": "message",
                "role": "assistant",
                "model": payload["model"],
                "content": [{"type": "text", "text": "selected"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
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
                "output_header": raw_request.headers.get("X-Proxy-Output-Protocol"),
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


def test_chat_route_forwards_independent_output_protocol_selector() -> None:
    client, rotating = _surface_client()

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Proxy-Output-Protocol": "anthropic_messages"},
        json={"model": "openai/gpt-test", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert response.json()["type"] == "message"
    assert rotating.calls[0]["input_protocol"] == "openai_chat"
    assert rotating.calls[0]["output_header"] == "anthropic_messages"


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


def test_cross_protocol_stream_selection_reaches_selected_wire_format() -> None:
    client, _ = _surface_client()

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Proxy-Output-Protocol": "gemini"},
        json={
            "model": "openai/gpt-test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert '"candidates"' in response.text
    assert "streamed" in response.text


def test_gemini_generate_rejects_stream_flag_until_stream_route_exists() -> None:
    client, _ = _surface_client()

    response = client.post(
        "/v1beta/models/gemini-2.5-pro:generateContent",
        json={"contents": [{"role": "user", "parts": [{"text": "hello"}]}], "stream": True},
    )

    assert response.status_code == 400
    assert response.json()["error"]["status"] == "INVALID_ARGUMENT"


def test_output_protocol_precedence_is_explicit_then_header_then_provider_then_input() -> None:
    client = RotatingClient.__new__(RotatingClient)
    client._get_provider_instance = lambda provider: SimpleNamespace(
        get_default_output_protocol=lambda model: "responses"
    )
    request = SimpleNamespace(headers={"X-Proxy-Output-Protocol": "anthropic"})
    payload = {"model": "configured/model"}

    assert client.resolve_output_protocol(payload, input_protocol="gemini", request=request, explicit="chat") == "openai_chat"
    assert client.resolve_output_protocol(payload, input_protocol="gemini", request=request) == "anthropic_messages"
    assert client.resolve_output_protocol(payload, input_protocol="gemini") == "responses"
    assert client.resolve_output_protocol({"model": "bare"}, input_protocol="gemini") == "gemini"


def test_output_protocol_provider_default_resolves_through_model_alias(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ROUTE_ALIAS", "configured/model")
    client = RotatingClient.__new__(RotatingClient)
    client._get_provider_instance = lambda provider: SimpleNamespace(
        get_default_output_protocol=lambda model: "anthropic_messages"
    )

    assert client.resolve_output_protocol({"model": "alias"}, input_protocol="gemini") == "anthropic_messages"


def test_output_header_is_case_insensitive_and_unknown_values_are_client_errors() -> None:
    assert request_output_protocol(SimpleNamespace(headers={"X-Proxy-Output-Protocol": "Anthropic"})) == "anthropic_messages"
    try:
        request_output_protocol(SimpleNamespace(headers={"X-Proxy-Output-Protocol": "unknown"}))
    except ValueError as error:
        assert "Unsupported output protocol" in str(error)
    else:
        raise AssertionError("unknown output protocol must fail")


class FailingSurfaceClient(SurfaceClient):
    async def agenerate(self, payload, **kwargs):
        raise ValueError("local validation failed")

    async def anthropic_messages(self, body, **kwargs):
        raise ValueError("local validation failed")

    async def gemini_generate(self, payload, **kwargs):
        raise ValueError("local validation failed")


@pytest.mark.parametrize(
    ("path", "headers", "body", "assertion"),
    (
        (
            "/v1/chat/completions",
            {"X-Proxy-Output-Protocol": "anthropic_messages"},
            {"model": "openai/gpt-test", "messages": [{"role": "user", "content": "hello"}]},
            lambda payload: payload["type"] == "error" and payload["error"]["type"] == "invalid_request_error",
        ),
        (
            "/v1/messages",
            {"X-Proxy-Output-Protocol": "gemini"},
            {"model": "claude-test", "max_tokens": 8, "messages": [{"role": "user", "content": "hello"}]},
            lambda payload: payload["error"]["status"] == "INVALID_ARGUMENT",
        ),
        (
            "/v1beta/models/gemini-2.5-pro:generateContent",
            {"X-Proxy-Output-Protocol": "openai_chat"},
            {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
            lambda payload: payload["error"]["type"] == "invalid_request",
        ),
    ),
)
def test_proxy_side_errors_use_selected_output_protocol(path, headers, body, assertion) -> None:
    proxy_main.PROXY_API_KEY = None
    proxy_main.ENABLE_RAW_LOGGING = False
    proxy_main.app.state.rotating_client = FailingSurfaceClient()

    response = TestClient(proxy_main.app).post(path, headers=headers, json=body)

    assert response.status_code == 400
    assert assertion(response.json())
    assert "detail" not in response.json()


def test_unknown_output_header_is_a_400_in_input_protocol_shape() -> None:
    client, _ = _surface_client()

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Proxy-Output-Protocol": "unknown"},
        json={"model": "openai/gpt-test", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request"
    assert "Unsupported output protocol" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("output_protocol", "assertion"),
    (
        (
            "anthropic_messages",
            lambda payload: payload["type"] == "error" and payload["error"]["type"] == "invalid_request_error",
        ),
        (
            "gemini",
            lambda payload: payload["error"]["status"] == "INVALID_ARGUMENT",
        ),
    ),
)
def test_responses_malformed_json_uses_selected_output_protocol(output_protocol, assertion) -> None:
    client, _ = _surface_client()

    response = client.post(
        "/v1/responses",
        headers={
            "Content-Type": "application/json",
            "X-Proxy-Output-Protocol": output_protocol,
        },
        content="{",
    )

    assert response.status_code == 400
    assert assertion(response.json())
    assert "detail" not in response.json()


def test_library_accepts_all_generative_cross_protocol_stream_pairs() -> None:
    from rotator_library.client.protocol_selection import require_same_protocol_stream

    for source in ("openai_chat", "anthropic_messages", "responses", "gemini"):
        for output in ("openai_chat", "anthropic_messages", "responses", "gemini"):
            require_same_protocol_stream(source, output)


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
async def test_gemini_handler_forwards_raw_request_to_generic_runtime_selector() -> None:
    client = GeminiRuntimeClient()
    handler = GeminiHandler(client)
    request = SimpleNamespace(headers={"X-Proxy-Output-Protocol": "anthropic_messages"})

    await handler.generate(
        {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
        model="gemini-2.5-pro",
        raw_request=request,
    )

    assert client.calls[0][1]["input_protocol"] == "gemini"
    assert client.calls[0][1]["request"] is request
