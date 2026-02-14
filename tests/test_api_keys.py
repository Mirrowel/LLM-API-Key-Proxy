import pytest
from sqlalchemy import select

from proxy_app.api_token_auth import hash_api_token
from proxy_app.auth import SessionUser
from proxy_app.db_models import ApiKey, UsageEvent
from proxy_app.routers.user_api import (
    CreateApiKeyRequest,
    create_my_api_key,
    list_my_api_keys,
    revoke_my_api_key,
)


@pytest.mark.asyncio
async def test_create_list_revoke_api_key_hides_plaintext_at_rest(
    session_maker,
    seeded_user,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_TOKEN_PEPPER", "pepper-for-tests")
    monkeypatch.setattr(
        "proxy_app.routers.user_api.generate_api_token",
        lambda: "pk_plaintext_value_for_tests_1234567890",
    )

    session_user = SessionUser(id=seeded_user.id, username=seeded_user.username, role="user")

    async with session_maker() as session:
        created = await create_my_api_key(
            payload=CreateApiKeyRequest(name="  personal key  "),
            current_user=session_user,
            session=session,
        )

    assert created.token == "pk_plaintext_value_for_tests_1234567890"
    assert created.name == "personal key"

    async with session_maker() as session:
        stored = await session.scalar(select(ApiKey).where(ApiKey.id == created.id))

    assert stored is not None
    assert stored.token_hash == hash_api_token(created.token)
    assert created.token not in stored.token_hash
    assert stored.token_prefix == created.token[:20]

    async with session_maker() as session:
        listed = await list_my_api_keys(current_user=session_user, session=session)

    assert len(listed.api_keys) == 1
    listed_item = listed.api_keys[0]
    dumped = listed_item.model_dump()
    assert "token" not in dumped
    assert "token_hash" not in dumped
    assert dumped["token_prefix"] == created.token[:20]

    async with session_maker() as session:
        revoked = await revoke_my_api_key(
            id=created.id,
            current_user=session_user,
            session=session,
        )
        reloaded = await session.scalar(select(ApiKey).where(ApiKey.id == created.id))

    assert revoked == {"ok": True}
    assert reloaded is not None
    assert reloaded.revoked_at is not None


@pytest.mark.asyncio
async def test_list_api_keys_uses_derived_last_used_timestamp(
    session_maker,
    seeded_user,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_TOKEN_PEPPER", "pepper-for-tests")
    monkeypatch.setattr(
        "proxy_app.routers.user_api.generate_api_token",
        lambda: "pk_plaintext_for_last_used_case",
    )

    session_user = SessionUser(id=seeded_user.id, username=seeded_user.username, role="user")

    async with session_maker() as session:
        created = await create_my_api_key(
            payload=CreateApiKeyRequest(name="usage key"),
            current_user=session_user,
            session=session,
        )

    async with session_maker() as session:
        session.add(
            UsageEvent(
                user_id=seeded_user.id,
                api_key_id=created.id,
                endpoint="/v1/chat/completions",
                provider="openai",
                model="openai/gpt-4o-mini",
                request_id="req-derived-last-used",
                status_code=200,
                total_tokens=12,
            )
        )
        await session.commit()

    async with session_maker() as session:
        listed = await list_my_api_keys(current_user=session_user, session=session)

    assert listed.api_keys
    listed_item = listed.api_keys[0]
    assert listed_item.last_used_at is not None
