import os
import sys
import json
import types
import asyncio
from pathlib import Path

# Ensure the src directory is on the path before importing the app
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

# Set required env vars before importing the app module
os.environ.setdefault("PROXY_API_KEY", "test-proxy-key")
# Provide at least one provider key to satisfy RotatingClient init during app lifespan
os.environ.setdefault("OPENAI_API_KEY", "sk-test-123")
os.environ.setdefault("SKIP_OAUTH_INIT_CHECK", "true")

# Ensure FastAPI is installed; otherwise skip these smoke tests gracefully
try:
    import fastapi  # noqa: F401
    from fastapi.testclient import TestClient  # type: ignore
except Exception:
    import unittest
    raise unittest.SkipTest("FastAPI not installed; skipping smoke tests. Run 'pip install -r requirements.txt' to enable.")

from proxy_app.main import app, get_rotating_client


class SimpleModelResponse:
    def __init__(self, payload: dict, status_code: int = 200, headers: dict | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def model_dump(self):
        return self._payload


class MockRotatingClient:
    def acompletion(self, request=None, **kwargs):
        if kwargs.get("stream"):
            async def agen():
                base = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": kwargs.get("model", "openai/gpt-4o-mini"),
                }
                # Two content chunks
                yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'content': 'Hello'}, 'finish_reason': None}]})}\n\n"
                yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'content': ' world!'}, 'finish_reason': None}], 'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}})}\n\n"
                # Final stop
                yield "data: [DONE]\n\n"
            return agen()
        else:
            async def coro():
                payload = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "created": 1,
                    "model": kwargs.get("model", "openai/gpt-4o-mini"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello world!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                }
                return SimpleModelResponse(payload)
            return coro()

    def aembedding(self, request=None, **kwargs):
        async def coro():
            inputs = kwargs.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            data = [
                {"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]}
                for i, _ in enumerate(inputs)
            ]
            return {
                "object": "list",
                "model": kwargs.get("model", "openai/text-embedding-3-small"),
                "data": data,
                "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
            }
        return coro()

    async def get_all_available_models(self, grouped: bool = False):
        models = ["openai/gpt-4o-mini", "anthropic/claude-3-haiku"]
        return models if not grouped else {"openai": [models[0]], "anthropic": [models[1]]}

    def token_count(self, **kwargs) -> int:
        # Simple deterministic count for smoke testing
        messages = kwargs.get("messages") or []
        return sum(len((m.get("content") or "").split()) for m in messages)


# Override the dependency to avoid real network calls
app.dependency_overrides[get_rotating_client] = lambda request: MockRotatingClient()


def auth_headers():
    return {"Authorization": f"Bearer {os.environ['PROXY_API_KEY']}"}


def test_root_healthcheck():
    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json().get("Status")


def test_chat_completions_non_stream():
    with TestClient(app) as client:
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Say hi"}],
            "stream": False,
        }
        resp = client.post("/v1/chat/completions", json=payload, headers=auth_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Hello world!"


def test_chat_completions_streaming_sse():
    with TestClient(app) as client:
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Say hi"}],
            "stream": True,
        }
        with client.stream("POST", "/v1/chat/completions", json=payload, headers=auth_headers()) as resp:
            assert resp.status_code == 200
            text = b"".join(resp.iter_raw()).decode()
            assert "data: [DONE]" in text
            # Ensure chunks were sent
            assert "\"delta\": {\"content\": \"Hello\"}" in text


def test_embeddings_endpoint():
    with TestClient(app) as client:
        payload = {
            "model": "openai/text-embedding-3-small",
            "input": ["a", "b", "c"],
        }
        resp = client.post("/v1/embeddings", json=payload, headers=auth_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 3


def test_models_list_endpoint():
    with TestClient(app) as client:
        resp = client.get("/v1/models", headers=auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert any(item["id"] == "openai/gpt-4o-mini" for item in data["data"])


def test_token_count_endpoint():
    with TestClient(app) as client:
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "one two three"}],
        }
        resp = client.post("/v1/token-count", json=payload, headers=auth_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert body["token_count"] == 3
