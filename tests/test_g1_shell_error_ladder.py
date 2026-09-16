"""G1 proxy-shell error-ladder pins.

The shell only declares routes; classification (``classify_error``) and
rendering (``protocol_error_payload``) live in the library and are bridged by
one shared route-side helper. These pins cover the four required families:
non-dict body → dialect 400, ladder parity (429 stays 429), per-dialect auth
401 shapes, and string-``stream`` rejection.
"""

from __future__ import annotations

import litellm
import pytest
from fastapi.testclient import TestClient

from proxy_app import main as proxy_main


class RouteClient:
    """Minimal route client whose methods raise a configured exception."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, object]] = []

    async def agenerate(
        self, payload, *, input_protocol="openai_chat", request=None, **kwargs
    ):
        self.calls.append(("agenerate", input_protocol))
        if self.error is not None:
            raise self.error
        if payload.get("stream"):

            async def stream():
                yield (
                    'data: {"id":"c","object":"chat.completion.chunk",'
                    '"choices":[{"delta":{"content":"x"}}]}\n\n'
                )

            return stream()
        return {
            "id": "c",
            "object": "chat.completion",
            "model": payload.get("model"),
            "choices": [
                {
                    "message": {"role": "assistant", "content": "x"},
                    "finish_reason": "stop",
                }
            ],
        }

    async def anthropic_messages(self, body, **kwargs):
        self.calls.append(("anthropic_messages", None))
        if self.error is not None:
            raise self.error
        return {"type": "message", "content": [{"type": "text", "text": "x"}]}

    async def anthropic_count_tokens(self, body):
        self.calls.append(("anthropic_count_tokens", None))
        if self.error is not None:
            raise self.error
        return {"input_tokens": 1}

    async def gemini_generate(self, payload, *, model, raw_request=None):
        self.calls.append(("gemini_generate", None))
        if self.error is not None:
            raise self.error
        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "x"}]},
                    "finishReason": "STOP",
                }
            ]
        }

    async def gemini_stream_generate(self, payload, *, model, raw_request=None):
        self.calls.append(("gemini_stream_generate", None))
        if self.error is not None:
            raise self.error

        async def stream():
            yield 'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"x"}]}}]}\n\n'

        return stream()

    def gemini_count_tokens(self, payload, *, model):
        self.calls.append(("gemini_count_tokens", None))
        if self.error is not None:
            raise self.error
        return {"totalTokens": 3}


def _install(monkeypatch, error: BaseException | None = None) -> TestClient:
    monkeypatch.setattr(proxy_main, "PROXY_API_KEY", None)
    monkeypatch.setattr(proxy_main, "ENABLE_RAW_LOGGING", False)
    proxy_main.app.state.rotating_client = RouteClient(error)
    return TestClient(proxy_main.app)


def _assert_dialect_400(payload: dict, protocol: str) -> None:
    assert "detail" not in payload
    if protocol == "gemini":
        assert payload["error"]["status"] == "INVALID_ARGUMENT"
        assert payload["error"]["code"] == 400
    elif protocol == "anthropic_messages":
        assert payload["type"] == "error"
        assert payload["error"]["type"] == "invalid_request_error"
    else:
        assert payload["error"]["type"] == "invalid_request_error"


NON_DICT_CASES = (
    ("/v1/chat/completions", "openai_chat"),
    ("/v1/messages", "anthropic_messages"),
    ("/v1/messages/count_tokens", "anthropic_messages"),
    ("/v1beta/models/gemini-2.5-pro:generateContent", "gemini"),
    ("/v1beta/models/gemini-2.5-pro:streamGenerateContent", "gemini"),
    ("/v1beta/models/gemini-2.5-pro:countTokens", "gemini"),
    ("/v1/cost-estimate", "openai_chat"),
    ("/v1/token-count", "openai_chat"),
    ("/v1/quota-stats", "openai_chat"),
    ("/v1/embeddings", "openai_chat"),
    ("/v1/responses", "responses"),
)


@pytest.mark.parametrize(("path", "protocol"), NON_DICT_CASES)
def test_non_dict_body_is_route_dialect_400(monkeypatch, path, protocol) -> None:
    client = _install(monkeypatch)

    response = client.post(path, json=[])

    assert response.status_code == 400, response.text
    _assert_dialect_400(response.json(), protocol)


AUTH_CASES = (
    ("/v1/chat/completions", "openai_chat"),
    ("/v1/models", "openai_chat"),
    ("/v1/messages", "anthropic_messages"),
    ("/v1/messages/count_tokens", "anthropic_messages"),
    ("/v1beta/models/gemini-2.5-pro:generateContent", "gemini"),
    ("/v1beta/models/gemini-2.5-pro:countTokens", "gemini"),
    ("/v1/responses", "responses"),
    ("/v1/embeddings", "openai_chat"),
)


def _assert_auth_dialect(payload: dict, protocol: str) -> None:
    assert "detail" not in payload
    if protocol == "gemini":
        assert payload["error"]["status"] == "UNAUTHENTICATED"
        assert payload["error"]["code"] == 401
    elif protocol == "anthropic_messages":
        assert payload["type"] == "error"
        assert payload["error"]["type"] == "authentication_error"
    else:
        assert payload["error"]["type"] == "authentication_error"


@pytest.mark.parametrize(("path", "protocol"), AUTH_CASES)
def test_auth_401_uses_route_dialect(monkeypatch, path, protocol) -> None:
    monkeypatch.setattr(proxy_main, "PROXY_API_KEY", "secret-key")
    monkeypatch.setattr(proxy_main, "ENABLE_RAW_LOGGING", False)
    proxy_main.app.state.rotating_client = RouteClient()
    client = TestClient(proxy_main.app)

    if path == "/v1/models":
        response = client.get(path)
    else:
        response = client.post(path, json={"model": "m", "messages": []})
    assert response.status_code == 401, response.text
    _assert_auth_dialect(response.json(), protocol)


def test_count_tokens_classified_429_is_not_500(monkeypatch) -> None:
    client = _install(
        monkeypatch, litellm.RateLimitError("slow down", "openai", "claude-test")
    )

    response = client.post(
        "/v1/messages/count_tokens",
        json={"model": "claude-test", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 429, response.text
    assert response.json()["error"]["type"] == "rate_limit_error"


def test_gemini_count_tokens_classified_429_maps_resource_exhausted(monkeypatch) -> None:
    client = _install(
        monkeypatch, litellm.RateLimitError("slow down", "gemini", "gemini-test")
    )

    response = client.post(
        "/v1beta/models/gemini-2.5-pro:countTokens",
        json={"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    )

    assert response.status_code == 429, response.text
    assert response.json()["error"]["status"] == "RESOURCE_EXHAUSTED"


def test_anthropic_timeout_renders_timeout_error(monkeypatch) -> None:
    client = _install(monkeypatch, litellm.Timeout("slow", "anthropic", "claude-test"))

    response = client.post(
        "/v1/messages",
        json={
            "model": "claude-test",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 504, response.text
    assert response.json()["error"]["type"] == "timeout_error"


@pytest.mark.parametrize("value", ["false", "true", "0", "1", 1, 0])
def test_chat_string_stream_is_protocol_400(monkeypatch, value) -> None:
    client = _install(monkeypatch)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [], "stream": value},
    )

    assert response.status_code == 400, response.text
    body = response.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"] == "stream must be a boolean"


def test_chat_null_stream_treated_as_non_streaming(monkeypatch) -> None:
    client = _install(monkeypatch)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [], "stream": None},
    )

    assert response.status_code == 200, response.text


def test_chat_true_stream_still_streams(monkeypatch) -> None:
    client = _install(monkeypatch)

    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [], "stream": True},
    )

    assert response.status_code == 200, response.text
    assert "chat.completion.chunk" in response.text


def test_quota_stats_failure_uses_stable_message(monkeypatch) -> None:
    monkeypatch.setattr(proxy_main, "PROXY_API_KEY", None)
    monkeypatch.setattr(proxy_main, "ENABLE_RAW_LOGGING", False)

    class ExplodingQuotaClient(RouteClient):
        async def get_quota_stats(self, provider_filter=None):
            raise RuntimeError("internal path C:/secrets/usage.json exploded")

    proxy_main.app.state.rotating_client = ExplodingQuotaClient()
    client = TestClient(proxy_main.app)

    response = client.get("/v1/quota-stats")

    assert response.status_code == 502, response.text
    body = response.json()
    assert body["error"]["message"] == "Failed to retrieve quota statistics."
    assert "secrets" not in response.text
    assert "detail" not in body
