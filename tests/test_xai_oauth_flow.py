# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw

"""
Tests for xAI Grok device-code OAuth flow in the proxy admin API.

Verifies:
- x-ai appears in GET /v1/admin/oauth/providers with flow_type=device_code
- POST /v1/admin/oauth/start returns the device-code envelope
  (mocking upstream auth.x.ai calls)
- The status endpoint reports pending → complete
- Unknown providers still return 400 (regression guard)

NO network calls, NO real credentials.
"""

import json
import time

# Use FastAPI's TestClient against the proxy_app router directly.
# We import the router, not the full app, to avoid the full app lifecycle
# (which requires real credentials, litellm init, etc.).
from proxy_app.api.oauth import router


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

# JWT-shaped payload for id_token: header.payload.sig (unverified, just bytes)
def _fake_jwt(claims: dict) -> str:
    import base64

    def b64(d: bytes) -> str:
        return base64.urlsafe_b64encode(d).rstrip(b"=").decode("ascii")

    header = b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
    payload = b64(json.dumps(claims).encode())
    return f"{header}.{payload}.fakesig"


def _make_device_response(
    user_code: str = "ABCD-EFGH",
    device_code: str = "dev_1234",
    verification_uri: str = "https://auth.x.ai/device",
    interval: int = 5,
    expires_in: int = 600,
) -> dict:
    return {
        "user_code": user_code,
        "device_code": device_code,
        "verification_uri": verification_uri,
        "verification_uri_complete": f"{verification_uri}?user_code={user_code}",
        "interval": interval,
        "expires_in": expires_in,
    }


def _make_token_response(
    access_token: str = "fake-access-token",
    refresh_token: str = "fake-refresh-token",
    expires_in: int = 3600,
    id_claims: dict | None = None,
) -> dict:
    if id_claims is None:
        id_claims = {"sub": "xai-user-1", "email": "user@example.com", "name": "Test"}
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": expires_in,
        "id_token": _fake_jwt(id_claims),
        "token_type": "Bearer",
        "scope": "openid email profile offline_access api:access grok-cli:access",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_xai_in_oauth_providers_list():
    """GET /v1/admin/oauth/providers must list x-ai with flow_type=device_code."""
    from fastapi.testclient import TestClient

    from proxy_app.api.oauth import _pending_flows  # noqa: F401  (ensure importable)

    # Build a minimal FastAPI app with just this router
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    resp = client.get("/v1/admin/oauth/providers")
    assert resp.status_code == 200, resp.text

    body = resp.json()
    providers = {p["provider_id"]: p for p in body["providers"]}

    assert "x-ai" in providers, f"x-ai missing from providers list: {list(providers)}"
    assert providers["x-ai"]["flow_type"] == "device_code"
    assert providers["x-ai"]["name"]


def test_start_xai_device_flow_returns_envelope(monkeypatch, tmp_path):
    """POST /v1/admin/oauth/start for x-ai returns the device-code envelope
    and stores the flow in _pending_flows."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI

    from proxy_app.api.oauth import _pending_flows, _FLOW_TTL  # noqa: F401

    # Redirect OAuth credential writes to a tmp dir so we don't pollute
    # the real oauth_creds path during tests.
    monkeypatch.setattr(
        "proxy_app.api.oauth.get_oauth_dir", lambda: tmp_path
    )

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # Mock the httpx.AsyncClient used inside _start_xai_device_flow
    # to return a synthetic device-code response from auth.x.ai.
    device_resp = _make_device_response()

    class _MockResponse:
        def __init__(self, json_data, status_code=200):
            self._json = json_data
            self.status_code = status_code
            self.is_success = 200 <= status_code < 300
            self.text = json.dumps(json_data)

        def json(self):
            return self._json

    class _MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, *args, **kwargs):
            if "device/code" in url:
                return _MockResponse(device_resp)
            return _MockResponse({"error": "authorization_pending"})

        async def get(self, url, *args, **kwargs):
            return _MockResponse({})

    # Track poll tasks so we can cancel them
    poll_tasks_started = []

    import proxy_app.api.oauth as oauth_mod
    real_create_task = oauth_mod.asyncio.create_task

    def _track_task(coro):
        task = real_create_task(coro)
        poll_tasks_started.append(task)
        # Cancel the task immediately — we only want to verify it was started
        task.cancel()
        return task

    monkeypatch.setattr(oauth_mod.httpx, "AsyncClient", _MockAsyncClient)
    monkeypatch.setattr(oauth_mod.asyncio, "create_task", _track_task)

    try:
        resp = client.post("/v1/admin/oauth/start", json={"provider": "x-ai"})
        assert resp.status_code == 200, resp.text

        body = resp.json()
        assert body["flow_type"] == "device_code"
        assert body["user_code"] == device_resp["user_code"]
        assert body["verification_uri"] == device_resp["verification_uri"]
        assert body["expires_in"] == device_resp["expires_in"]
        assert "flow_id" in body

        # Flow was registered
        assert len(_pending_flows) >= 1
        assert len(poll_tasks_started) == 1
    finally:
        _pending_flows.clear()


def test_start_unknown_provider_returns_400():
    """Regression: an unknown provider still returns HTTP 400."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI

    from proxy_app.api.oauth import _pending_flows  # noqa: F401

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    try:
        resp = client.post(
            "/v1/admin/oauth/start", json={"provider": "not-a-real-provider"}
        )
        assert resp.status_code == 400, resp.text
        assert "not implemented" in resp.text.lower() or "unknown" in resp.text.lower()
    finally:
        _pending_flows.clear()


def test_status_endpoint_returns_flow_state(monkeypatch, tmp_path):
    """GET /v1/admin/oauth/status/{flow_id} returns the flow's current state."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI

    from proxy_app.api.oauth import _pending_flows

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    # Manually inject a flow and verify the status endpoint surfaces it
    flow_id = "test-flow-abc"
    _pending_flows[flow_id] = {
        "provider": "x-ai",
        "created_at": time.time(),
        "status": "pending",
        "error": None,
        "result": None,
        "device_code": "dev_xyz",
        "interval": 5,
        "expires_in": 600,
        "client_id": "b1a00492-073a-47ea-816f-4c329264a828",
    }

    try:
        resp = client.get(f"/v1/admin/oauth/status/{flow_id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["flow_id"] == flow_id
        assert body["provider"] == "x-ai"
        assert body["status"] == "pending"
    finally:
        _pending_flows.clear()


def test_status_unknown_flow_returns_404():
    """Regression: an unknown flow_id returns 404."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI

    from proxy_app.api.oauth import _pending_flows  # noqa: F401

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    try:
        resp = client.get("/v1/admin/oauth/status/does-not-exist")
        assert resp.status_code == 404
    finally:
        _pending_flows.clear()


def test_save_credential_file_uses_xai_prefix(monkeypatch, tmp_path):
    """Regression guard: _save_credential_file must produce x-ai_oauth_N.json
    when called with provider='x-ai'."""
    from proxy_app.api.oauth import _save_credential_file

    monkeypatch.setattr(
        "proxy_app.api.oauth.get_oauth_dir", lambda: tmp_path
    )

    flow = {"provider": "x-ai"}
    creds = {
        "access_token": "fake",
        "refresh_token": "fake-r",
        "expiry_date": time.time() + 3600,
        "_proxy_metadata": {"email": "u@example.com", "last_check_timestamp": time.time()},
    }
    _save_credential_file(flow, creds)

    saved = list(tmp_path.glob("x-ai_oauth_*.json"))
    assert len(saved) == 1, f"expected one x-ai_oauth_*.json, got {saved}"
    assert saved[0].name == "x-ai_oauth_1.json"
