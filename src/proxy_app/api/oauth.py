"""Admin API for OAuth provider information and web-driven credential setup."""

import asyncio
import hashlib
import base64
import json as _json_mod
import re as _re_mod
import secrets
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rotator_library.provider_factory import (
    get_available_providers,
    get_provider_auth_class,
)
from rotator_library.utils.paths import get_oauth_dir

lib_logger = logging.getLogger("rotator_library")

_app_ref: Optional[Any] = None


def set_app_ref(app) -> None:
    """Called once at startup so background OAuth tasks can hot-load credentials."""
    global _app_ref
    _app_ref = app

router = APIRouter(prefix="/v1/admin", tags=["admin-oauth"])

# ---------------------------------------------------------------------------
# In-memory store for pending OAuth flows
# ---------------------------------------------------------------------------
_pending_flows: Dict[str, Dict[str, Any]] = {}
_FLOW_TTL = 600  # seconds

PROVIDER_META = {
    "gemini_cli": {
        "name": "Gemini CLI",
        "flow_type": "authorization_code_paste",
        "description": "Google Gemini via OAuth. Paste the redirect URL after sign-in.",
    },
    "codex": {
        "name": "Codex (OpenAI)",
        "flow_type": "authorization_code_paste",
        "description": "OpenAI Codex via OAuth. Paste the redirect URL after sign-in.",
    },
    "anthropic": {
        "name": "Anthropic",
        "flow_type": "authorization_code_paste",
        "description": "Anthropic via OAuth. Paste the redirect URL after sign-in.",
    },
    "copilot": {
        "name": "GitHub Copilot",
        "flow_type": "device_code",
        "description": "GitHub Copilot via device flow. Enter code at GitHub.",
    },
    "x-ai": {
        "name": "xAI Grok",
        "flow_type": "device_code",
        "description": "xAI Grok via device flow. Enter code at auth.x.ai.",
    },
}


def _cleanup_expired():
    now = time.time()
    expired = [k for k, v in _pending_flows.items() if now - v["created_at"] > _FLOW_TTL]
    for k in expired:
        _pending_flows.pop(k, None)


def _generate_pkce():
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


# ---------------------------------------------------------------------------
# GET /v1/admin/oauth/providers  — list providers
# ---------------------------------------------------------------------------
@router.get("/oauth/providers")
async def list_oauth_providers():
    providers = get_available_providers()
    result = []
    for p in providers:
        info = PROVIDER_META.get(p, {
            "name": p,
            "flow_type": "unknown",
            "description": f"OAuth provider: {p}",
        })
        info["provider_id"] = p
        result.append(info)
    return {"providers": result}


# ---------------------------------------------------------------------------
# POST /v1/admin/oauth/start  — initiate a flow
# ---------------------------------------------------------------------------
class OAuthStartRequest(BaseModel):
    provider: str


@router.post("/oauth/start")
async def start_oauth_flow(req: OAuthStartRequest):
    _cleanup_expired()
    provider = req.provider.lower()
    if provider not in get_available_providers():
        raise HTTPException(400, f"Unknown OAuth provider: {provider}")

    flow_id = secrets.token_urlsafe(16)
    flow: Dict[str, Any] = {
        "provider": provider,
        "created_at": time.time(),
        "status": "pending",
        "error": None,
        "result": None,
    }

    if provider == "copilot":
        return await _start_copilot_device_flow(flow_id, flow)
    elif provider == "x-ai":
        return await _start_xai_device_flow(flow_id, flow)
    elif provider in ("codex", "gemini_cli", "anthropic"):
        return _start_paste_flow(flow_id, flow, provider)
    else:
        raise HTTPException(400, f"OAuth flow not implemented for: {provider}")


# ---------------------------------------------------------------------------
# Copilot: device flow
# ---------------------------------------------------------------------------
async def _start_copilot_device_flow(flow_id: str, flow: dict):
    client_id = base64.b64decode(
        "SXYxLmI1MDdhMDhjODdlY2ZlOTg="
    ).decode()

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://github.com/login/device/code",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "GitHubCopilotChat/0.35.0",
            },
            data={"client_id": client_id, "scope": "read:user"},
            timeout=30.0,
        )
        if not resp.is_success:
            raise HTTPException(502, f"GitHub device code request failed: {resp.text}")

        data = resp.json()

    flow["device_code"] = data["device_code"]
    flow["interval"] = data.get("interval", 5)
    flow["expires_in"] = data.get("expires_in", 900)
    flow["client_id"] = client_id
    _pending_flows[flow_id] = flow

    # Start background polling
    asyncio.create_task(_poll_copilot_device(flow_id))

    return {
        "flow_id": flow_id,
        "flow_type": "device_code",
        "verification_uri": data.get("verification_uri", "https://github.com/login/device"),
        "user_code": data.get("user_code", ""),
        "expires_in": data.get("expires_in", 900),
    }


async def _poll_copilot_device(flow_id: str):
    flow = _pending_flows.get(flow_id)
    if not flow:
        return

    client_id = flow["client_id"]
    device_code = flow["device_code"]
    interval = flow["interval"]
    max_polls = flow["expires_in"] // interval

    async with httpx.AsyncClient() as client:
        for _ in range(max_polls):
            await asyncio.sleep(interval)
            if flow_id not in _pending_flows:
                return

            try:
                resp = await client.post(
                    "https://github.com/login/oauth/access_token",
                    headers={
                        "Accept": "application/json",
                        "Content-Type": "application/x-www-form-urlencoded",
                        "User-Agent": "GitHubCopilotChat/0.35.0",
                    },
                    data={
                        "client_id": client_id,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    timeout=30.0,
                )
                if not resp.is_success:
                    continue

                token_data = resp.json()
                if "access_token" in token_data:
                    github_token = token_data["access_token"]
                    await _finalize_copilot(flow_id, flow, github_token, client)
                    return

                error = token_data.get("error", "")
                if error == "expired_token":
                    flow["status"] = "error"
                    flow["error"] = "Device code expired. Please try again."
                    return
            except Exception as e:
                lib_logger.debug(f"Copilot poll error: {e}")
                continue

    flow["status"] = "error"
    flow["error"] = "Device flow timed out."


async def _finalize_copilot(flow_id: str, flow: dict, github_token: str, client: httpx.AsyncClient):
    new_creds: Dict[str, Any] = {
        "refresh_token": github_token,
        "access_token": "",
        "expiry_date": 0,
        "_proxy_metadata": {"last_check_timestamp": time.time()},
    }

    try:
        user_resp = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {github_token}"},
            timeout=10.0,
        )
        if user_resp.is_success:
            login = user_resp.json().get("login", "unknown")
            new_creds["_proxy_metadata"]["login"] = login
    except Exception:
        new_creds["_proxy_metadata"]["login"] = "unknown"

    # Fetch Copilot API token
    try:
        copilot_resp = await client.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers={
                "Authorization": f"Bearer {github_token}",
                "User-Agent": "GitHubCopilotChat/0.35.0",
            },
            timeout=10.0,
        )
        if copilot_resp.is_success:
            copilot_data = copilot_resp.json()
            token_str = copilot_data.get("token", "")
            new_creds["access_token"] = token_str
            exp = copilot_data.get("expires_at")
            if exp:
                new_creds["expiry_date"] = exp * 1000

            from rotator_library.providers.copilot_auth_base import _get_base_url_from_token
            base_url = _get_base_url_from_token(token_str)
            if base_url:
                new_creds["copilot_base_url"] = base_url

            sku = copilot_data.get("sku", "")
            if sku:
                new_creds["_proxy_metadata"]["sku"] = sku
    except Exception as e:
        lib_logger.warning(f"Failed to fetch Copilot token: {e}")

    _save_credential_file(flow, new_creds)
    flow["status"] = "complete"
    flow["result"] = {
        "login": new_creds["_proxy_metadata"].get("login", "unknown"),
        "provider": "copilot",
    }


# ---------------------------------------------------------------------------
# xAI Grok: device flow
# ---------------------------------------------------------------------------
async def _start_xai_device_flow(flow_id: str, flow: dict):
    # Reuse xAI provider constants — public client, no client_secret.
    from rotator_library.providers.x_ai_auth_base import (
        XAI_CLIENT_ID,
        XAI_OAUTH_SCOPES,
        XAI_DEVICE_CODE_URL,
    )

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            XAI_DEVICE_CODE_URL,
            data={
                "client_id": XAI_CLIENT_ID,
                "scope": " ".join(XAI_OAUTH_SCOPES),
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            timeout=30.0,
        )
        if not resp.is_success:
            raise HTTPException(
                502, f"xAI device code request failed: {resp.text}"
            )

        data = resp.json()

    flow["device_code"] = data["device_code"]
    flow["interval"] = data.get("interval", 5)
    flow["expires_in"] = data.get("expires_in", 600)
    flow["client_id"] = XAI_CLIENT_ID
    _pending_flows[flow_id] = flow

    # Start background polling
    asyncio.create_task(_poll_xai_device(flow_id))

    return {
        "flow_id": flow_id,
        "flow_type": "device_code",
        "verification_uri": data.get("verification_uri", "https://auth.x.ai/device"),
        "user_code": data.get("user_code", ""),
        "expires_in": data.get("expires_in", 600),
    }


async def _poll_xai_device(flow_id: str):
    flow = _pending_flows.get(flow_id)
    if not flow:
        return

    from rotator_library.providers.x_ai_auth_base import XAI_TOKEN_URL

    client_id = flow["client_id"]
    device_code = flow["device_code"]
    interval = flow["interval"]
    max_polls = flow["expires_in"] // interval

    async with httpx.AsyncClient() as client:
        for _ in range(max_polls):
            await asyncio.sleep(interval)
            if flow_id not in _pending_flows:
                return

            try:
                resp = await client.post(
                    XAI_TOKEN_URL,
                    data={
                        "client_id": client_id,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                    },
                    timeout=30.0,
                )
                if not resp.is_success:
                    continue

                token_data = resp.json()
                if "access_token" in token_data:
                    await _finalize_xai(flow_id, flow, token_data, client)
                    return

                error = token_data.get("error", "")
                if error == "expired_token":
                    flow["status"] = "error"
                    flow["error"] = "Device code expired. Please try again."
                    return
            except Exception as e:
                lib_logger.debug(f"xAI poll error: {e}")
                continue

    flow["status"] = "error"
    flow["error"] = "Device flow timed out."


async def _finalize_xai(
    flow_id: str, flow: dict, token_data: dict, client: httpx.AsyncClient
):
    """Persist xAI credentials and mark flow complete.

    Credential shape mirrors XAiAuthBase._build_credentials_from_token_data
    so the existing provider loader picks them up unchanged.
    """
    from rotator_library.providers.x_ai_auth_base import XAI_USERINFO_URL

    access_token = token_data.get("access_token", "")
    refresh_token = token_data.get("refresh_token", "")

    # Email discovery: id_token JWT → userinfo → sub fallback
    id_claims = _decode_jwt_payload(token_data.get("id_token", "")) or {}
    email = id_claims.get("email", "")
    sub = id_claims.get("sub", "")

    if not email and access_token:
        try:
            userinfo_resp = await client.get(
                XAI_USERINFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0,
            )
            if userinfo_resp.is_success:
                userinfo = userinfo_resp.json()
                email = userinfo.get("email", "") or email
                sub = sub or userinfo.get("sub", "")
        except Exception as e:
            lib_logger.debug(f"xAI userinfo fetch failed: {e}")

    if not email:
        email = sub or f"xai-user-{int(time.time())}"

    expires_in = token_data.get("expires_in", 3600)

    new_creds: Dict[str, Any] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expiry_date": time.time() + expires_in,
        "account_id": sub or email,
        "_proxy_metadata": {
            "email": email,
            "account_id": sub or email,
            "last_check_timestamp": time.time(),
        },
    }

    _save_credential_file(flow, new_creds)
    flow["status"] = "complete"
    flow["result"] = {"login": email, "provider": "x-ai"}


# ---------------------------------------------------------------------------
# Codex / Gemini CLI / Anthropic: PKCE auth code + paste redirect URL
# ---------------------------------------------------------------------------
def _start_paste_flow(flow_id: str, flow: dict, provider: str):
    from urllib.parse import urlencode

    verifier, challenge = _generate_pkce()
    state = secrets.token_urlsafe(16)
    flow["code_verifier"] = verifier
    flow["state"] = state

    auth_class = get_provider_auth_class(provider)
    auth_inst = auth_class()

    if provider == "codex":
        port = auth_inst.callback_port if hasattr(auth_inst, 'callback_port') else getattr(auth_inst, 'CALLBACK_PORT', 1455)
        redirect_uri = f"http://localhost:{port}{getattr(auth_inst, 'CALLBACK_PATH', '/auth/callback')}"
        flow["redirect_uri"] = redirect_uri
        flow["token_url"] = auth_inst.TOKEN_URL
        flow["client_id"] = auth_inst.CLIENT_ID
        auth_url = f"{auth_inst.AUTH_URL}?" + urlencode({
            "response_type": "code",
            "client_id": auth_inst.CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": " ".join(auth_inst.OAUTH_SCOPES),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "codex_cli_simplified_flow": "true",
        })
        paste_hint = "After signing in, your browser will redirect to a localhost URL that may not load. Copy the full URL from the address bar and paste it below."

    elif provider == "gemini_cli":
        port = auth_inst.callback_port if hasattr(auth_inst, 'callback_port') else getattr(auth_inst, 'CALLBACK_PORT', 8085)
        redirect_uri = f"http://localhost:{port}/oauth2callback"
        flow["redirect_uri"] = redirect_uri
        flow["token_url"] = "https://oauth2.googleapis.com/token"
        flow["client_id"] = auth_inst.CLIENT_ID
        flow["client_secret"] = auth_inst.CLIENT_SECRET
        auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
            "response_type": "code",
            "client_id": auth_inst.CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": " ".join(auth_inst.OAUTH_SCOPES),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        })
        paste_hint = "After signing in, your browser will redirect to a localhost URL that may not load. Copy the full URL from the address bar and paste it below."

    elif provider == "anthropic":
        redirect_uri = "https://console.anthropic.com/oauth/code/callback"
        flow["redirect_uri"] = redirect_uri
        flow["token_url"] = "https://console.anthropic.com/v1/oauth/token"
        flow["client_id"] = auth_inst.CLIENT_ID
        scopes = getattr(auth_inst, "OAUTH_SCOPES", ["org:create_api_key", "user:profile", "user:inference"])
        auth_url = "https://claude.ai/oauth/authorize?" + urlencode({
            "response_type": "code",
            "client_id": auth_inst.CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes),
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        })
        paste_hint = "After authorizing, copy the redirect URL from your browser and paste it below."
    else:
        raise HTTPException(400, f"Paste flow not configured for: {provider}")

    flow["auth_url"] = auth_url
    _pending_flows[flow_id] = flow

    return {
        "flow_id": flow_id,
        "flow_type": "authorization_code_paste",
        "auth_url": auth_url,
        "paste_hint": paste_hint,
    }


# ---------------------------------------------------------------------------
# POST /v1/admin/oauth/callback  — submit auth code
# ---------------------------------------------------------------------------
class OAuthCallbackRequest(BaseModel):
    flow_id: str
    code: str


@router.post("/oauth/callback")
async def submit_oauth_code(req: OAuthCallbackRequest):
    flow = _pending_flows.get(req.flow_id)
    if not flow:
        raise HTTPException(404, "Flow not found or expired")
    if flow["status"] != "pending":
        raise HTTPException(400, f"Flow already {flow['status']}")

    code = req.code.strip()
    # If user pasted a full URL, extract the code param
    if "code=" in code and ("http://" in code or "https://" in code):
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(code)
        params = parse_qs(parsed.query)
        code = params.get("code", [code])[0]

    await _exchange_code(req.flow_id, flow, code)

    return {"status": flow["status"], "result": flow.get("result"), "error": flow.get("error")}


# ---------------------------------------------------------------------------
# GET /v1/admin/oauth/status/{flow_id}  — poll status
# ---------------------------------------------------------------------------
@router.get("/oauth/status/{flow_id}")
async def get_oauth_status(flow_id: str):
    flow = _pending_flows.get(flow_id)
    if not flow:
        raise HTTPException(404, "Flow not found or expired")

    return {
        "flow_id": flow_id,
        "provider": flow["provider"],
        "status": flow["status"],
        "result": flow.get("result"),
        "error": flow.get("error"),
    }


# ---------------------------------------------------------------------------
# Code exchange (shared by PKCE flows)
# ---------------------------------------------------------------------------
async def _exchange_code(flow_id: str, flow: dict, code: str):
    provider = flow["provider"]
    token_url = flow["token_url"]
    client_id = flow["client_id"]
    redirect_uri = flow["redirect_uri"]
    verifier = flow["code_verifier"]

    try:
        async with httpx.AsyncClient() as client:
            if provider == "anthropic":
                resp = await client.post(
                    token_url,
                    json={
                        "grant_type": "authorization_code",
                        "code": code,
                        "client_id": client_id,
                        "redirect_uri": redirect_uri,
                        "code_verifier": verifier,
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=30.0,
                )
            else:
                data = {
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": client_id,
                    "redirect_uri": redirect_uri,
                    "code_verifier": verifier,
                }
                if provider == "gemini_cli" and "client_secret" in flow:
                    data["client_secret"] = flow["client_secret"]

                resp = await client.post(
                    token_url,
                    data=data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=30.0,
                )

            if not resp.is_success:
                flow["status"] = "error"
                flow["error"] = f"Token exchange failed (HTTP {resp.status_code}): {resp.text[:200]}"
                return

            token_data = resp.json()

            new_creds: Dict[str, Any] = {
                "access_token": token_data.get("access_token", ""),
                "refresh_token": token_data.get("refresh_token", ""),
                "expiry_date": _compute_expiry(token_data),
                "_proxy_metadata": {"last_check_timestamp": time.time()},
            }

            if provider == "codex":
                _enrich_codex_creds(new_creds, token_data)
            elif provider == "gemini_cli":
                _enrich_gemini_creds(new_creds, token_data)
            elif provider == "anthropic":
                _enrich_anthropic_creds(new_creds, token_data)

            _save_credential_file(flow, new_creds)
            login = (
                new_creds.get("_proxy_metadata", {}).get("email")
                or new_creds.get("_proxy_metadata", {}).get("login")
                or "unknown"
            )
            flow["status"] = "complete"
            flow["result"] = {"login": login, "provider": provider}

    except Exception as e:
        flow["status"] = "error"
        flow["error"] = f"Token exchange error: {e}"


def _compute_expiry(token_data: dict) -> float:
    if "expires_in" in token_data:
        return time.time() + token_data["expires_in"]
    if "expiry_date" in token_data:
        return token_data["expiry_date"]
    return time.time() + 3600


def _decode_jwt_payload(token: str) -> dict:
    """Decode a JWT payload without verification."""
    import json as _json
    try:
        payload = token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)
        return _json.loads(base64.b64decode(payload))
    except Exception:
        return {}


def _enrich_codex_creds(creds: dict, token_data: dict):
    id_token = token_data.get("id_token", "")
    access_token_str = token_data.get("access_token", "")
    if id_token:
        creds["id_token"] = id_token

    id_claims = _decode_jwt_payload(id_token) if id_token else {}
    access_claims = _decode_jwt_payload(access_token_str) if access_token_str else {}

    auth_claims = id_claims.get("https://api.openai.com/auth", {})
    account_id = auth_claims.get("user_id", id_claims.get("sub", ""))
    org_id = id_claims.get("org_id")
    project_id = id_claims.get("project_id")
    email = id_claims.get("email", "")
    plan_type = (
        auth_claims.get("chatgpt_plan_type")
        or access_claims.get("chatgpt_plan_type", "")
    )

    organizations = auth_claims.get("organizations", [])
    workspace_title = ""
    if organizations and isinstance(organizations, list):
        for org in organizations:
            if isinstance(org, dict) and org.get("is_default"):
                workspace_title = org.get("title", "")
                break
        if not workspace_title and isinstance(organizations[0], dict):
            workspace_title = organizations[0].get("title", "")

    if account_id:
        creds["account_id"] = account_id
    creds["_proxy_metadata"].update({
        "email": email,
        "account_id": account_id,
        "org_id": org_id,
        "project_id": project_id,
        "plan_type": plan_type,
        "workspace_title": workspace_title,
    })


def _enrich_gemini_creds(creds: dict, token_data: dict):
    creds["scope"] = token_data.get("scope", "")
    creds["token_type"] = token_data.get("token_type", "Bearer")
    if "id_token" in token_data:
        creds["id_token"] = token_data["id_token"]
        try:
            import json as _json
            payload = token_data["id_token"].split(".")[1]
            payload += "=" * (4 - len(payload) % 4)
            claims = _json.loads(base64.b64decode(payload))
            creds["_proxy_metadata"]["email"] = claims.get("email", "")
        except Exception:
            pass
    creds["client_id"] = creds.get("client_id", "")
    creds["token_uri"] = "https://oauth2.googleapis.com/token"
    creds["type"] = "authorized_user"

    auth_class = get_provider_auth_class("gemini_cli")
    auth_inst = auth_class()
    creds["client_id"] = auth_inst.CLIENT_ID
    creds["client_secret"] = auth_inst.CLIENT_SECRET


def _enrich_anthropic_creds(creds: dict, token_data: dict):
    pass


# ---------------------------------------------------------------------------
# Save credential file (with replacement and hot-load)
# ---------------------------------------------------------------------------
def _extract_identity(creds: dict) -> str:
    """Return a stable identity string for matching existing credentials."""
    meta = creds.get("_proxy_metadata", {})
    return (
        meta.get("email")
        or meta.get("login")
        or creds.get("account_id")
        or ""
    )


def _find_existing_credential(
    oauth_dir: Path, prefix: str, new_identity: str,
) -> Optional[Path]:
    """Find an existing credential file for the same account."""
    if not new_identity:
        return None

    for f in sorted(oauth_dir.glob(f"{prefix}_oauth_*.json")):
        m = _re_mod.search(r"_oauth_(\d+)\.json$", f.name)
        if not m:
            continue
        try:
            data = _json_mod.loads(f.read_text())
            existing_identity = _extract_identity(data)
            if existing_identity and existing_identity == new_identity:
                return f
        except Exception:
            continue
    return None


def _save_credential_file(flow: dict, creds: dict):
    provider = flow["provider"]
    oauth_dir = get_oauth_dir()
    oauth_dir.mkdir(parents=True, exist_ok=True)

    prefix_map = {
        "gemini_cli": "gemini_cli",
        "codex": "codex",
        "anthropic": "anthropic",
        "copilot": "copilot",
    }
    prefix = prefix_map.get(provider, provider)
    new_identity = _extract_identity(creds)

    replaced_path = _find_existing_credential(oauth_dir, prefix, new_identity)

    if replaced_path:
        # Back up the old file and overwrite in-place
        bak = replaced_path.with_suffix(
            f".json.bak.{time.strftime('%Y%m%d_%H%M%S')}"
        )
        replaced_path.rename(bak)
        filepath = replaced_path
        lib_logger.info(
            f"Replacing existing credential {replaced_path.name} for "
            f"'{new_identity}' (backup: {bak.name})"
        )
    else:
        existing = sorted(oauth_dir.glob(f"{prefix}_oauth_*.json"))
        numbers = []
        for f in existing:
            m = _re_mod.search(r"_oauth_(\d+)\.json$", f.name)
            if m:
                numbers.append(int(m.group(1)))
        next_num = (max(numbers) + 1) if numbers else 1
        filepath = oauth_dir / f"{prefix}_oauth_{next_num}.json"

    with open(filepath, "w") as f:
        _json_mod.dump(creds, f, indent=2)

    resolved = str(filepath.resolve())
    lib_logger.info(f"Saved OAuth credential: {filepath}")
    flow["saved_path"] = resolved

    _hot_load_credential(provider, resolved, replaced_path if replaced_path else None)


def _hot_load_credential(
    provider: str,
    new_path: str,
    replaced_path: Optional[Path],
) -> None:
    """Register a newly saved credential in the running RotatingClient."""
    if _app_ref is None:
        return
    try:
        client = _app_ref.state.rotating_client
    except Exception:
        return

    old_resolved = str(replaced_path.resolve()) if replaced_path else None

    # Update oauth_credentials
    if provider not in client.oauth_credentials:
        client.oauth_credentials[provider] = []
    if old_resolved and old_resolved in client.oauth_credentials[provider]:
        idx = client.oauth_credentials[provider].index(old_resolved)
        client.oauth_credentials[provider][idx] = new_path
    elif new_path not in client.oauth_credentials[provider]:
        client.oauth_credentials[provider].append(new_path)

    # Update all_credentials
    if provider not in client.all_credentials:
        client.all_credentials[provider] = []
    if old_resolved and old_resolved in client.all_credentials[provider]:
        idx = client.all_credentials[provider].index(old_resolved)
        client.all_credentials[provider][idx] = new_path
    elif new_path not in client.all_credentials[provider]:
        client.all_credentials[provider].append(new_path)

    # Ensure provider is tracked as an OAuth provider
    if hasattr(client, "oauth_providers"):
        client.oauth_providers.add(provider)

    # Update usage manager: swap or register
    usage_manager = client.get_usage_manager(provider)
    if usage_manager:
        asyncio.ensure_future(_update_usage_manager(
            usage_manager, provider, client, new_path, old_resolved,
        ))

    lib_logger.info(
        f"Hot-loaded credential into running proxy for {provider}: "
        f"{Path(new_path).name}"
        + (f" (replaced {replaced_path.name})" if replaced_path else " (new)")
    )


async def _update_usage_manager(
    usage_manager, provider: str, client, new_path: str,
    old_path: Optional[str],
) -> None:
    """Async helper to update usage manager after a hot-load."""
    try:
        if old_path:
            await usage_manager.remove_credential(old_path)

        credentials = client.all_credentials.get(provider, [])
        priorities, tiers = {}, {}
        plugin = client._get_provider_instance(provider)
        if plugin:
            if hasattr(plugin, "get_credential_priority"):
                p = plugin.get_credential_priority(new_path)
                if p is not None:
                    priorities[new_path] = p
            if hasattr(plugin, "get_credential_tier_name"):
                t = plugin.get_credential_tier_name(new_path)
                if t:
                    tiers[new_path] = t

        await usage_manager.initialize(credentials, priorities=priorities, tiers=tiers)
    except Exception as e:
        lib_logger.warning(f"Failed to update usage manager for {provider}: {e}")
