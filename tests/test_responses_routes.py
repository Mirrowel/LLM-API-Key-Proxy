from __future__ import annotations

from fastapi.testclient import TestClient

from proxy_app import main as proxy_main
from rotator_library.responses import InMemoryResponsesStore, ResponsesService


class FakeClient:
    async def acompletion(self, **kwargs):
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


def test_post_responses_non_stream_success() -> None:
    client = _client()

    response = client.post("/v1/responses", json={"model": "gpt-test", "input": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "chat_route_1"
    assert body["object"] == "response"
    assert body["output"][0]["content"][0]["text"] == "route ok"


def test_post_responses_missing_model_returns_400() -> None:
    client = _client()

    response = client.post("/v1/responses", json={"input": "hello"})

    assert response.status_code == 400
    assert response.json()["detail"]["error"]["type"] == "invalid_request_error"


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


def test_post_responses_stream_checkpoint_returns_501_until_streaming_slice() -> None:
    client = _client()

    response = client.post("/v1/responses", json={"model": "gpt-test", "input": "hello", "stream": True})

    assert response.status_code == 501
    assert response.json()["detail"]["error"]["type"] == "not_implemented_error"
