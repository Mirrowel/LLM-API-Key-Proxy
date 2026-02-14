from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from proxy_app.auth import (
    SessionUser,
    clear_session_cookie,
    create_session_token,
    get_db_session,
    require_user,
    set_session_cookie,
    verify_password,
)
from proxy_app.db_models import User

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class IdentityResponse(BaseModel):
    id: int
    username: str
    role: str


@router.post("/login", response_model=IdentityResponse)
async def login(
    payload: LoginRequest,
    response: Response,
    session: AsyncSession = Depends(get_db_session),
) -> IdentityResponse:
    username = payload.username.strip()
    user = await session.scalar(select(User).where(User.username == username))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User is inactive",
        )

    user.last_login_at = datetime.utcnow()
    await session.commit()

    token = create_session_token(user_id=user.id, username=user.username, role=user.role)
    set_session_cookie(response, token)

    return IdentityResponse(id=user.id, username=user.username, role=user.role)


@router.post("/logout")
async def logout(response: Response) -> dict[str, bool]:
    clear_session_cookie(response)
    return {"ok": True}


@router.get("/me", response_model=IdentityResponse)
async def me(current_user: SessionUser = Depends(require_user)) -> IdentityResponse:
    return IdentityResponse(
        id=current_user.id,
        username=current_user.username,
        role=current_user.role,
    )
