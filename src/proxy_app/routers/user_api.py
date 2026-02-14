import secrets
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from proxy_app.auth import SessionUser, get_db_session, require_user
from proxy_app.api_token_auth import hash_api_token
from proxy_app.db_models import ApiKey, User
from proxy_app.usage_queries import (
    fetch_usage_by_day,
    fetch_usage_by_model,
    fetch_usage_summary,
)

router = APIRouter(prefix="/api", tags=["user"])

def generate_api_token() -> str:
    return f"pk_{secrets.token_urlsafe(32)}"


class MeResponse(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool


class ApiKeyItem(BaseModel):
    id: int
    name: str
    token_prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None
    expires_at: datetime | None


class ApiKeyListResponse(BaseModel):
    api_keys: list[ApiKeyItem]


class CreateApiKeyRequest(BaseModel):
    name: str


class CreateApiKeyResponse(BaseModel):
    id: int
    name: str
    token: str
    token_prefix: str
    created_at: datetime


class UsageTotals(BaseModel):
    request_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None


class UsageByDayItem(UsageTotals):
    day: str


class UsageByDayResponse(BaseModel):
    days: int
    rows: list[UsageByDayItem]


class UsageByModelItem(UsageTotals):
    model: str | None


class UsageByModelResponse(BaseModel):
    days: int
    rows: list[UsageByModelItem]


@router.get("/me", response_model=MeResponse)
async def get_me(
    current_user: SessionUser = Depends(require_user),
    session: AsyncSession = Depends(get_db_session),
) -> MeResponse:
    user = await session.get(User, current_user.id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid session",
        )

    return MeResponse(
        id=user.id,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
    )


@router.get("/me/api-keys", response_model=ApiKeyListResponse)
async def list_my_api_keys(
    current_user: SessionUser = Depends(require_user),
    session: AsyncSession = Depends(get_db_session),
) -> ApiKeyListResponse:
    rows = await session.scalars(
        select(ApiKey)
        .where(ApiKey.user_id == current_user.id)
        .order_by(ApiKey.created_at.desc())
    )
    return ApiKeyListResponse(
        api_keys=[
            ApiKeyItem(
                id=row.id,
                name=row.name,
                token_prefix=row.token_prefix,
                created_at=row.created_at,
                last_used_at=row.last_used_at,
                revoked_at=row.revoked_at,
                expires_at=row.expires_at,
            )
            for row in rows
        ]
    )


@router.post("/me/api-keys", response_model=CreateApiKeyResponse)
async def create_my_api_key(
    payload: CreateApiKeyRequest,
    current_user: SessionUser = Depends(require_user),
    session: AsyncSession = Depends(get_db_session),
) -> CreateApiKeyResponse:
    name = payload.name.strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API key name cannot be empty",
        )

    token = generate_api_token()
    key = ApiKey(
        user_id=current_user.id,
        name=name,
        token_prefix=token[:20],
        token_hash=hash_api_token(token),
    )
    session.add(key)
    await session.commit()
    await session.refresh(key)

    return CreateApiKeyResponse(
        id=key.id,
        name=key.name,
        token=token,
        token_prefix=key.token_prefix,
        created_at=key.created_at,
    )


@router.delete("/me/api-keys/{id}")
async def revoke_my_api_key(
    id: int,
    current_user: SessionUser = Depends(require_user),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, bool]:
    key = await session.scalar(
        select(ApiKey).where(ApiKey.id == id, ApiKey.user_id == current_user.id)
    )
    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    if not key.revoked_at:
        key.revoked_at = datetime.utcnow()
        await session.commit()

    return {"ok": True}


@router.get("/me/usage/summary", response_model=UsageTotals)
async def get_my_usage_summary(
    current_user: SessionUser = Depends(require_user),
    session: AsyncSession = Depends(get_db_session),
) -> UsageTotals:
    return UsageTotals(**(await fetch_usage_summary(session, user_id=current_user.id)))


@router.get("/me/usage/by-day", response_model=UsageByDayResponse)
async def get_my_usage_by_day(
    days: int = Query(default=30, ge=1, le=365),
    current_user: SessionUser = Depends(require_user),
    session: AsyncSession = Depends(get_db_session),
) -> UsageByDayResponse:
    rows = await fetch_usage_by_day(session, user_id=current_user.id, days=days)
    return UsageByDayResponse(days=days, rows=[UsageByDayItem(**row) for row in rows])


@router.get("/me/usage/by-model", response_model=UsageByModelResponse)
async def get_my_usage_by_model(
    days: int = Query(default=30, ge=1, le=365),
    current_user: SessionUser = Depends(require_user),
    session: AsyncSession = Depends(get_db_session),
) -> UsageByModelResponse:
    rows = await fetch_usage_by_model(session, user_id=current_user.id, days=days)
    return UsageByModelResponse(
        days=days,
        rows=[UsageByModelItem(**row) for row in rows],
    )
