import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import datetime

from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from proxy_app.auth import get_db_session
from proxy_app.db_models import ApiKey, User
from proxy_app.security_config import get_api_token_pepper

AUTH_MODE_USERS = "users"
AUTH_MODE_LEGACY = "legacy"
AUTH_MODE_BOTH = "both"
DEFAULT_AUTH_MODE = AUTH_MODE_BOTH

AUTH_SOURCE_USER_API_KEY = "user_api_key"
AUTH_SOURCE_LEGACY_MASTER = "legacy_master"

HASH_SCHEME_HMAC_SHA256 = "hmac_sha256_v1"
HASH_SCHEME_LEGACY_SHA256_PREFIX = "sha256_prefix_v1"

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
anthropic_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


@dataclass
class ApiActor:
    user_id: int | None
    api_key_id: int | None
    role: str
    auth_source: str


def get_auth_mode() -> str:
    mode = (os.getenv("AUTH_MODE") or DEFAULT_AUTH_MODE).strip().lower()
    return normalize_auth_mode(mode)


def normalize_auth_mode(mode: str) -> str:
    if mode in {AUTH_MODE_USERS, AUTH_MODE_LEGACY, AUTH_MODE_BOTH}:
        return mode
    return DEFAULT_AUTH_MODE


def get_legacy_master_key() -> str:
    return os.getenv("PROXY_API_KEY") or ""


def parse_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    value = authorization.strip()
    if not value:
        return None
    parts = value.split(" ", 1)
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def extract_api_token_from_headers(
    *, x_api_key: str | None, authorization: str | None
) -> str | None:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    return parse_bearer_token(authorization)


def hash_api_token(token: str, *, pepper: str | None = None) -> str:
    active_pepper = pepper if pepper is not None else get_api_token_pepper()
    digest = hmac.new(
        active_pepper.encode("utf-8"),
        token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest


def hash_api_token_legacy(token: str, *, pepper: str | None = None) -> str:
    active_pepper = pepper if pepper is not None else get_api_token_pepper()
    payload = f"{active_pepper}{token}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _active_user_api_key_query(token_hashes: list[str]):
    now = datetime.utcnow()
    return (
        select(ApiKey, User)
        .join(User, ApiKey.user_id == User.id)
        .where(ApiKey.token_hash.in_(token_hashes))
        .where(ApiKey.revoked_at.is_(None))
        .where((ApiKey.expires_at.is_(None)) | (ApiKey.expires_at > now))
        .where(User.is_active.is_(True))
    )


async def _lookup_user_actor(
    *,
    session: AsyncSession,
    token: str,
    token_pepper: str | None,
) -> ApiActor | None:
    new_hash = hash_api_token(token, pepper=token_pepper)
    legacy_hash = hash_api_token_legacy(token, pepper=token_pepper)
    lookup_hashes = [new_hash] if legacy_hash == new_hash else [new_hash, legacy_hash]

    rows = (await session.execute(_active_user_api_key_query(lookup_hashes))).all()
    if not rows:
        return None

    api_key: ApiKey
    user: User
    api_key, user = rows[0]
    for row in rows:
        candidate_key, candidate_user = row
        if candidate_key.token_hash == new_hash:
            api_key, user = candidate_key, candidate_user
            break

    if api_key.token_hash == legacy_hash and api_key.token_hash != new_hash:
        try:
            api_key.token_hash = new_hash
            await session.commit()
        except Exception:
            await session.rollback()

    return ApiActor(
        user_id=user.id,
        api_key_id=api_key.id,
        role=user.role,
        auth_source=AUTH_SOURCE_USER_API_KEY,
    )


async def resolve_api_actor_from_token(
    *,
    session: AsyncSession,
    token: str,
    auth_mode: str | None = None,
    legacy_master_key: str | None = None,
    token_pepper: str | None = None,
) -> ApiActor | None:
    mode = normalize_auth_mode(auth_mode) if auth_mode else get_auth_mode()

    if mode in {AUTH_MODE_USERS, AUTH_MODE_BOTH}:
        actor = await _lookup_user_actor(
            session=session,
            token=token,
            token_pepper=token_pepper,
        )
        if actor:
            return actor

    if mode in {AUTH_MODE_LEGACY, AUTH_MODE_BOTH}:
        legacy_key = (
            get_legacy_master_key() if legacy_master_key is None else legacy_master_key
        )
        if legacy_key and token == legacy_key:
            return ApiActor(
                user_id=None,
                api_key_id=None,
                role="admin",
                auth_source=AUTH_SOURCE_LEGACY_MASTER,
            )

    return None


async def get_api_actor(
    session: AsyncSession = Depends(get_db_session),
    x_api_key: str | None = Depends(anthropic_api_key_header),
    authorization: str | None = Depends(api_key_header),
) -> ApiActor:
    token = extract_api_token_from_headers(x_api_key=x_api_key, authorization=authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

    actor = await resolve_api_actor_from_token(session=session, token=token)
    if actor:
        return actor

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )


async def require_admin_api_actor(actor: ApiActor = Depends(get_api_actor)) -> ApiActor:
    if actor.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return actor
