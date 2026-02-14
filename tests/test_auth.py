from datetime import datetime, timedelta

import pytest

from proxy_app.api_token_auth import (
    AUTH_MODE_BOTH,
    AUTH_MODE_LEGACY,
    AUTH_MODE_USERS,
    AUTH_SOURCE_LEGACY_MASTER,
    AUTH_SOURCE_USER_API_KEY,
    extract_api_token_from_headers,
    hash_api_token,
    hash_api_token_legacy,
    resolve_api_actor_from_token,
)
from proxy_app.auth import create_session_token, decode_session_token, verify_password
from proxy_app.db import hash_password
from proxy_app.db_models import ApiKey


def test_verify_password_basics() -> None:
    password_hash = hash_password("strong-pass")
    assert verify_password("strong-pass", password_hash) is True
    assert verify_password("wrong-pass", password_hash) is False


def test_session_token_round_trip_and_tamper_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SESSION_SECRET", "unit-test-secret")
    monkeypatch.setattr("proxy_app.auth.time.time", lambda: 1_700_000_000)

    token = create_session_token(user_id=42, username="alice", role="admin")
    payload = decode_session_token(token)

    assert payload is not None
    assert payload["uid"] == 42
    assert payload["usr"] == "alice"
    assert payload["rol"] == "admin"

    tampered = token[:-1] + ("A" if token[-1] != "A" else "B")
    assert decode_session_token(tampered) is None


def test_session_token_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SESSION_SECRET", "unit-test-secret")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "60")
    monkeypatch.setattr("proxy_app.auth.time.time", lambda: 1_700_000_000)

    token = create_session_token(user_id=7, username="bob", role="user")

    monkeypatch.setattr("proxy_app.auth.time.time", lambda: 1_700_000_061)
    assert decode_session_token(token) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "token", "expected_source"),
    [
        (AUTH_MODE_USERS, "user-token", AUTH_SOURCE_USER_API_KEY),
        (AUTH_MODE_USERS, "legacy-master", None),
        (AUTH_MODE_LEGACY, "user-token", None),
        (AUTH_MODE_LEGACY, "legacy-master", AUTH_SOURCE_LEGACY_MASTER),
        (AUTH_MODE_BOTH, "user-token", AUTH_SOURCE_USER_API_KEY),
        (AUTH_MODE_BOTH, "legacy-master", AUTH_SOURCE_LEGACY_MASTER),
    ],
)
async def test_auth_mode_matrix(
    session_maker,
    seeded_user,
    mode: str,
    token: str,
    expected_source: str | None,
) -> None:
    async with session_maker() as session:
        key = ApiKey(
            user_id=seeded_user.id,
            name="primary",
            token_prefix="pk_user",
            token_hash=hash_api_token("user-token", pepper="pepper-1"),
            expires_at=datetime.utcnow() + timedelta(days=1),
        )
        session.add(key)
        await session.commit()

    async with session_maker() as session:
        actor = await resolve_api_actor_from_token(
            session=session,
            token=token,
            auth_mode=mode,
            legacy_master_key="legacy-master",
            token_pepper="pepper-1",
        )

    if expected_source is None:
        assert actor is None
        return

    assert actor is not None
    assert actor.auth_source == expected_source


def test_x_api_key_header_precedence_over_authorization() -> None:
    token = extract_api_token_from_headers(
        x_api_key="  x-api-token  ",
        authorization="Bearer bearer-token",
    )
    assert token == "x-api-token"

    fallback = extract_api_token_from_headers(
        x_api_key=" ",
        authorization="Bearer bearer-token",
    )
    assert fallback == "bearer-token"


@pytest.mark.asyncio
async def test_legacy_token_hash_opportunistic_migration(
    session_maker,
    seeded_user,
) -> None:
    async with session_maker() as session:
        key = ApiKey(
            user_id=seeded_user.id,
            name="legacy-key",
            token_prefix="pk_legacy",
            token_hash=hash_api_token_legacy("legacy-user-token", pepper="pepper-1"),
            expires_at=datetime.utcnow() + timedelta(days=1),
        )
        session.add(key)
        await session.commit()
        await session.refresh(key)
        key_id = key.id

    async with session_maker() as session:
        actor = await resolve_api_actor_from_token(
            session=session,
            token="legacy-user-token",
            auth_mode=AUTH_MODE_USERS,
            token_pepper="pepper-1",
        )
        assert actor is not None
        assert actor.api_key_id == key_id

    async with session_maker() as session:
        migrated_key = await session.get(ApiKey, key_id)

    assert migrated_key is not None
    assert migrated_key.token_hash == hash_api_token("legacy-user-token", pepper="pepper-1")
