from __future__ import annotations

import json

from fastapi.testclient import TestClient

from proxy_app import main as proxy_main
from proxy_app.detailed_logger import RawIOLogger
from rotator_library.responses import InMemoryResponsesStore, ResponsesService


class FakeClient:
    async def acompletion(self, **kwargs):
        if kwargs.get("stream"):
            async def chunks():
                yield 'data: {"choices":[{"delta":{"content":"route"}}]}\n\n'
                yield 'data: {"choices":[{"delta":{"content":" stream"}}]}\n\n'
                yield "data: [DONE]\n\n"

            return chunks()
        return {
            "id": "chat_route_1",
            "model": kwargs["model"],
            "choices": [{"message": {"role": "assistant", "content": "route ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }


def _client() -> TestClient:
    proxy_main.PROXY_API_KEY = None
    proxy_main.ENABLE_RAW_LOGGING = False
    proxy_main.app.state.rotating_client = FakeClient()
    proxy_main.app.state.responses_service = ResponsesService(store=InMemoryResponsesStore())
    return TestClient(proxy_main.app)


def test_raw_logger_never_persists_authentication_headers() -> None:
    logger = RawIOLogger.__new__(RawIOLogger)
    logger.request_id = "request"
    logger.streaming = False
    written = {}
    logger._write_json = lambda filename, data: written.update(
        {"filename": filename, "data": data}
    )

    logger.log_request(
        {
            "Authorization": "Bearer proxy-secret",
            "X-Api-Key": "provider-secret",
            "X-Proxy-Session-Domain": "bundle:private.capability",
            "Content-Type": "application/json",
        },
        {"model": "gpt-test"},
    )

    assert written["data"]["headers"] == {
        "Authorization": "<redacted>",
        "X-Api-Key": "<redacted>",
        "X-Proxy-Session-Domain": "<redacted>",
        "Content-Type": "application/json",
    }


def test_post_responses_non_stream_success() -> None:
    client = _client()

    response = client.post("/v1/responses", json={"model": "gpt-test", "input": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "chat_route_1"
    assert body["object"] == "response"
    assert body["output"][0]["content"][0]["text"] == "route ok"
    assert response.headers["x-proxy-session-domain"] == "public"


def test_post_responses_missing_model_returns_400() -> None:
    client = _client()

    response = client.post("/v1/responses", json={"input": "hello"})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_post_responses_stream_missing_model_returns_400_before_sse() -> None:
    client = _client()

    response = client.post("/v1/responses", json={"input": "hello", "stream": True})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_post_responses_stream_missing_previous_response_returns_404_before_sse() -> None:
    client = _client()

    response = client.post("/v1/responses", json={"model": "gpt-test", "input": "hello", "stream": True, "previous_response_id": "missing"})

    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"


def test_get_delete_and_input_items_routes() -> None:
    client = _client()
    created = client.post("/v1/responses", json={"model": "gpt-test", "input": ["hello"]}).json()

    get_response = client.get(f"/v1/responses/{created['id']}")
    input_items = client.get(f"/v1/responses/{created['id']}/input_items")
    deleted = client.delete(f"/v1/responses/{created['id']}")
    missing = client.get(f"/v1/responses/{created['id']}")

    assert get_response.status_code == 200
    assert get_response.json()["id"] == created["id"]
    assert input_items.status_code == 200
    assert input_items.json() == {"object": "list", "data": ["hello"]}
    assert deleted.status_code == 200
    assert deleted.json() == {"id": created["id"], "object": "response.deleted", "deleted": True}
    assert missing.status_code == 404
    assert missing.json()["error"]["type"] == "not_found_error"


def test_scoped_response_retrieval_requires_creation_domain_header() -> None:
    client = _client()
    created_response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-test",
            "input": "private",
            "api_keys": {"openai": ["private-key"]},
            "private": True,
        },
    )
    response_id = created_response.json()["id"]
    domain = created_response.headers["x-proxy-session-domain"]
    raw_domain = domain.rsplit(".", 1)[0]

    assert domain.startswith("bundle:")
    assert raw_domain.startswith("bundle:")
    assert client.get(f"/v1/responses/{response_id}").status_code == 404
    assert (
        client.get(
            f"/v1/responses/{response_id}",
            headers={"X-Proxy-Session-Domain": raw_domain},
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/v1/responses/{response_id}",
            headers={"X-Proxy-Session-Domain": domain},
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/v1/responses/{response_id}",
            headers={"X-Proxy-Session-Domain": "bundle:wrong"},
        ).status_code
        == 404
    )


def test_scoped_response_access_capabilities_are_unique_per_response() -> None:
    client = _client()
    payload = {
        "model": "gpt-test",
        "input": "private",
        "api_keys": {"openai": ["private-key"]},
        "private": True,
    }

    first = client.post("/v1/responses", json=payload)
    second = client.post("/v1/responses", json=payload)

    assert first.headers["x-proxy-session-domain"] != second.headers["x-proxy-session-domain"]


def test_scoped_continuation_requires_parent_access_capability() -> None:
    client = _client()
    payload = {
        "model": "gpt-test",
        "input": "private",
        "api_keys": {"openai": ["private-key"]},
        "private": True,
    }
    parent = client.post("/v1/responses", json=payload)
    continuation = {**payload, "previous_response_id": parent.json()["id"]}

    denied = client.post("/v1/responses", json=continuation)
    allowed = client.post(
        "/v1/responses",
        json=continuation,
        headers={
            "X-Proxy-Session-Domain": parent.headers["x-proxy-session-domain"]
        },
    )

    assert denied.status_code == 404
    assert allowed.status_code == 200


def test_post_responses_stream_returns_sse_events() -> None:
    client = _client()

    response = client.post("/v1/responses", json={"model": "gpt-test", "input": "hello", "stream": True})

    assert response.status_code == 200
    assert response.headers["x-proxy-session-domain"] == "public"
    assert "event: response.created" in response.text
    assert "event: response.completed" in response.text


def test_scoped_stream_uses_same_access_capability_for_stored_response() -> None:
    client = _client()

    response = client.post(
        "/v1/responses",
        json={
            "model": "gpt-test",
            "input": "private stream",
            "stream": True,
            "api_keys": {"openai": ["private-key"]},
            "private": True,
        },
    )
    capability = response.headers["x-proxy-session-domain"]
    created_payload = next(
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    )
    response_id = created_payload["id"]

    assert response.status_code == 200
    assert capability.startswith("bundle:")
    assert (
        client.get(
            f"/v1/responses/{response_id}",
            headers={"X-Proxy-Session-Domain": capability},
        ).status_code
        == 200
    )
