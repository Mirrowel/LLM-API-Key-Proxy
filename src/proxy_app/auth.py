import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import AsyncGenerator

from fastapi import Depends, Form, HTTPException, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from proxy_app.db_models import User
from proxy_app.security_config import get_session_secret

SESSION_COOKIE_NAME = "proxy_session"
SESSION_COOKIE_PATH = "/"
SESSION_COOKIE_SAMESITE = "lax"
CSRF_COOKIE_NAME = "proxy_csrf"
CSRF_COOKIE_PATH = "/ui"
CSRF_FORM_FIELD = "csrf_token"
CSRF_TOKEN_BYTES = 32
DEFAULT_SESSION_TTL_SECONDS = 60 * 60 * 24


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)

def _get_session_ttl_seconds() -> int:
    raw = os.getenv("SESSION_TTL_SECONDS", str(DEFAULT_SESSION_TTL_SECONDS))
    try:
        ttl = int(raw)
    except ValueError:
        ttl = DEFAULT_SESSION_TTL_SECONDS
    return max(60, ttl)


def _get_cookie_secure() -> bool:
    return os.getenv("SESSION_COOKIE_SECURE", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _get_cookie_samesite() -> str:
    configured = os.getenv("SESSION_COOKIE_SAMESITE", SESSION_COOKIE_SAMESITE).strip().lower()
    if configured not in {"lax", "strict", "none"}:
        configured = SESSION_COOKIE_SAMESITE
    return configured


def _resolve_cookie_secure(samesite: str) -> bool:
    configured_secure = _get_cookie_secure()
    if samesite == "none":
        return True
    return configured_secure


def _is_valid_csrf_token(token: str | None) -> bool:
    if not token:
        return False
    if len(token) < 20 or len(token) > 256:
        return False
    return True


def _generate_csrf_token() -> str:
    return secrets.token_urlsafe(CSRF_TOKEN_BYTES)


def get_csrf_token_for_request(request: Request) -> str:
    token = request.cookies.get(CSRF_COOKIE_NAME)
    if _is_valid_csrf_token(token):
        return token
    return _generate_csrf_token()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, iterations_raw, salt_b64, digest_b64 = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
    except Exception:
        return False

    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(candidate, expected)


def create_session_token(*, user_id: int, username: str, role: str) -> str:
    now = int(time.time())
    payload = {
        "uid": user_id,
        "usr": username,
        "rol": role,
        "iat": now,
        "exp": now + _get_session_ttl_seconds(),
    }
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    payload_b64 = _b64url_encode(payload_json)
    signature = hmac.new(
        get_session_secret().encode("utf-8"),
        payload_b64.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{payload_b64}.{_b64url_encode(signature)}"


def decode_session_token(token: str) -> dict | None:
    try:
        payload_b64, sig_b64 = token.split(".", 1)
        expected_sig = hmac.new(
            get_session_secret().encode("utf-8"),
            payload_b64.encode("ascii"),
            hashlib.sha256,
        ).digest()
        actual_sig = _b64url_decode(sig_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
        exp = int(payload["exp"])
        if exp < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def set_session_cookie(response: Response, token: str) -> None:
    same_site = _get_cookie_samesite()
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=_resolve_cookie_secure(same_site),
        samesite=same_site,
        max_age=_get_session_ttl_seconds(),
        path=SESSION_COOKIE_PATH,
    )


def clear_session_cookie(response: Response) -> None:
    same_site = _get_cookie_samesite()
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path=SESSION_COOKIE_PATH,
        secure=_resolve_cookie_secure(same_site),
        httponly=True,
        samesite=same_site,
    )


def attach_csrf_cookie(request: Request, response: Response, *, token: str | None = None) -> str:
    token_value = token if _is_valid_csrf_token(token) else get_csrf_token_for_request(request)

    same_site = _get_cookie_samesite()
    current_cookie = request.cookies.get(CSRF_COOKIE_NAME)
    if current_cookie != token_value:
        response.set_cookie(
            key=CSRF_COOKIE_NAME,
            value=token_value,
            httponly=True,
            secure=_resolve_cookie_secure(same_site),
            samesite=same_site,
            max_age=_get_session_ttl_seconds(),
            path=CSRF_COOKIE_PATH,
        )
    return token_value


def clear_csrf_cookie(response: Response) -> None:
    same_site = _get_cookie_samesite()
    response.delete_cookie(
        key=CSRF_COOKIE_NAME,
        path=CSRF_COOKIE_PATH,
        secure=_resolve_cookie_secure(same_site),
        httponly=True,
        samesite=same_site,
    )


async def require_csrf(
    request: Request,
    csrf_token: str = Form(default="", alias=CSRF_FORM_FIELD),
) -> None:
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    if not (_is_valid_csrf_token(cookie_token) and _is_valid_csrf_token(csrf_token)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF validation failed")
    if not hmac.compare_digest(cookie_token, csrf_token):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF validation failed")


@dataclass
class SessionUser:
    id: int
    username: str
    role: str


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    session_maker: async_sessionmaker[AsyncSession] = request.app.state.db_session_maker
    async with session_maker() as session:
        yield session


async def require_user(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> SessionUser:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    payload = decode_session_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session",
        )

    user = await session.get(User, int(payload.get("uid", 0)))
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session",
        )

    return SessionUser(id=user.id, username=user.username, role=user.role)


async def require_admin(current_user: SessionUser = Depends(require_user)) -> SessionUser:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user
