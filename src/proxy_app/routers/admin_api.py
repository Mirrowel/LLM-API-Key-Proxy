import secrets
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from proxy_app.auth import SessionUser, get_db_session, require_admin
from proxy_app.db import hash_password
from proxy_app.db_models import User
from proxy_app.usage_queries import (
    fetch_usage_by_day,
    fetch_usage_by_model,
    fetch_usage_summary,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])


class AdminUserItem(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None


class AdminUserListResponse(BaseModel):
    users: list[AdminUserItem]


class CreateAdminUserRequest(BaseModel):
    username: str
    password: str
    role: Literal["admin", "user"] = "user"
    is_active: bool = True


class ResetPasswordRequest(BaseModel):
    password: str | None = None


class ResetPasswordResponse(BaseModel):
    id: int
    username: str
    password: str


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


def _serialize_user(user: User) -> AdminUserItem:
    return AdminUserItem(
        id=user.id,
        username=user.username,
        role=user.role,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )


async def _require_target_user(session: AsyncSession, user_id: int) -> User:
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    return user


@router.get("/users", response_model=AdminUserListResponse)
async def admin_list_users(
    _: SessionUser = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> AdminUserListResponse:
    rows = await session.scalars(select(User).order_by(User.created_at.asc()))
    return AdminUserListResponse(users=[_serialize_user(row) for row in rows])


@router.post("/users", response_model=AdminUserItem)
async def admin_create_user(
    payload: CreateAdminUserRequest,
    _: SessionUser = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> AdminUserItem:
    username = payload.username.strip()
    if not username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username cannot be empty",
        )

    if not payload.password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password cannot be empty",
        )

    existing = await session.scalar(select(User).where(User.username == username))
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists",
        )

    user = User(
        username=username,
        password_hash=hash_password(payload.password),
        role=payload.role,
        is_active=payload.is_active,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return _serialize_user(user)


@router.post("/users/{id}/disable")
async def admin_disable_user(
    id: int,
    current_admin: SessionUser = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, bool]:
    if current_admin.id == id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot disable your own account",
        )

    user = await session.get(User, id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if user.is_active:
        user.is_active = False
        await session.commit()
    return {"ok": True}


@router.post("/users/{id}/enable")
async def admin_enable_user(
    id: int,
    _: SessionUser = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, bool]:
    user = await session.get(User, id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if not user.is_active:
        user.is_active = True
        await session.commit()
    return {"ok": True}


@router.post("/users/{id}/reset-password", response_model=ResetPasswordResponse)
async def admin_reset_password(
    id: int,
    payload: ResetPasswordRequest,
    _: SessionUser = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> ResetPasswordResponse:
    user = await session.get(User, id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    new_password = payload.password or secrets.token_urlsafe(12)
    user.password_hash = hash_password(new_password)
    await session.commit()

    return ResetPasswordResponse(id=user.id, username=user.username, password=new_password)


@router.get("/users/{id}/usage/summary", response_model=UsageTotals)
async def admin_user_usage_summary(
    id: int,
    _: SessionUser = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> UsageTotals:
    await _require_target_user(session, id)
    return UsageTotals(**(await fetch_usage_summary(session, user_id=id)))


@router.get("/users/{id}/usage/by-day", response_model=UsageByDayResponse)
async def admin_user_usage_by_day(
    id: int,
    days: int = Query(default=30, ge=1, le=365),
    _: SessionUser = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> UsageByDayResponse:
    await _require_target_user(session, id)
    rows = await fetch_usage_by_day(session, user_id=id, days=days)
    return UsageByDayResponse(days=days, rows=[UsageByDayItem(**row) for row in rows])


@router.get("/users/{id}/usage/by-model", response_model=UsageByModelResponse)
async def admin_user_usage_by_model(
    id: int,
    days: int = Query(default=30, ge=1, le=365),
    _: SessionUser = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> UsageByModelResponse:
    await _require_target_user(session, id)
    rows = await fetch_usage_by_model(session, user_id=id, days=days)
    return UsageByModelResponse(
        days=days,
        rows=[UsageByModelItem(**row) for row in rows],
    )
