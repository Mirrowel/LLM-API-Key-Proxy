import secrets
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from proxy_app.api_token_auth import hash_api_token
from proxy_app.auth import (
    CSRF_FORM_FIELD,
    SESSION_COOKIE_NAME,
    SessionUser,
    attach_csrf_cookie,
    clear_csrf_cookie,
    clear_session_cookie,
    create_session_token,
    decode_session_token,
    get_db_session,
    get_csrf_token_for_request,
    require_csrf,
    require_admin,
    require_user,
    set_session_cookie,
    verify_password,
)
from proxy_app.db import hash_password
from proxy_app.db_models import ApiKey, User
from proxy_app.usage_queries import (
    fetch_usage_by_day,
    fetch_usage_by_model,
    fetch_usage_summary,
)

router = APIRouter(prefix="/ui", tags=["ui"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


def _redirect(path: str) -> RedirectResponse:
    return RedirectResponse(url=path, status_code=status.HTTP_303_SEE_OTHER)


def _template_response(
    *,
    request: Request,
    name: str,
    context: dict,
    status_code: int = status.HTTP_200_OK,
) -> HTMLResponse:
    merged_context = dict(context)
    merged_context["request"] = request
    csrf_token = get_csrf_token_for_request(request)
    merged_context[CSRF_FORM_FIELD] = csrf_token
    response = templates.TemplateResponse(
        request=request,
        name=name,
        context=merged_context,
        status_code=status_code,
    )
    attach_csrf_cookie(request, response, token=csrf_token)
    return response


def _generate_api_token() -> str:
    return f"pk_{secrets.token_urlsafe(32)}"


async def _optional_session_user(
    request: Request,
    session: AsyncSession,
) -> SessionUser | None:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None

    payload = decode_session_token(token)
    if not payload:
        return None

    user = await session.get(User, int(payload.get("uid", 0)))
    if not user or not user.is_active:
        return None

    return SessionUser(id=user.id, username=user.username, role=user.role)


async def _load_me_context(
    session: AsyncSession,
    user_id: int,
    *,
    days: int,
) -> dict:
    rows = await session.scalars(
        select(ApiKey).where(ApiKey.user_id == user_id).order_by(ApiKey.created_at.desc())
    )
    api_keys = list(rows)
    usage_summary = await fetch_usage_summary(session, user_id=user_id)
    usage_by_day = await fetch_usage_by_day(session, user_id=user_id, days=days)
    return {
        "api_keys": api_keys,
        "usage_summary": usage_summary,
        "usage_by_day": usage_by_day,
        "days": days,
    }


async def _load_admin_user_detail_context(
    session: AsyncSession,
    user_id: int,
    *,
    days: int,
) -> dict:
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    usage_summary = await fetch_usage_summary(session, user_id=user_id)
    usage_by_day = await fetch_usage_by_day(session, user_id=user_id, days=days)
    usage_by_model = await fetch_usage_by_model(session, user_id=user_id, days=days)
    return {
        "target_user": user,
        "usage_summary": usage_summary,
        "usage_by_day": usage_by_day,
        "usage_by_model": usage_by_model,
        "days": days,
    }


@router.get("/login", response_class=HTMLResponse)
async def ui_login_page(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
    session_user = await _optional_session_user(request, session)
    if session_user:
        return _redirect("/ui/me")

    return _template_response(request=request, name="login.html", context={"error": None})


@router.post("/login", response_class=HTMLResponse)
async def ui_login_submit(
    request: Request,
    _: None = Depends(require_csrf),
    username: str = Form(default=""),
    password: str = Form(default=""),
    session: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
    username = username.strip()
    user = await session.scalar(select(User).where(User.username == username))
    if not user or not verify_password(password, user.password_hash):
        return _template_response(
            request=request,
            name="login.html",
            status_code=status.HTTP_401_UNAUTHORIZED,
            context={"error": "Invalid username or password"},
        )

    if not user.is_active:
        return _template_response(
            request=request,
            name="login.html",
            status_code=status.HTTP_403_FORBIDDEN,
            context={"error": "User is inactive"},
        )

    user.last_login_at = datetime.utcnow()
    await session.commit()

    response = _redirect("/ui/me")
    token = create_session_token(user_id=user.id, username=user.username, role=user.role)
    set_session_cookie(response, token)
    return response


@router.post("/logout")
async def ui_logout(_: None = Depends(require_csrf)) -> RedirectResponse:
    response = _redirect("/ui/login")
    clear_session_cookie(response)
    clear_csrf_cookie(response)
    return response


@router.get("/me", response_class=HTMLResponse)
async def ui_me_page(
    request: Request,
    days: int = Query(default=30, ge=1, le=365),
    current_user: SessionUser = Depends(require_user),
    session: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
    context = await _load_me_context(session, current_user.id, days=days)
    context.update(
        {
            "current_user": current_user,
            "new_api_key_token": None,
            "error": None,
        }
    )
    return _template_response(request=request, name="me.html", context=context)


@router.post("/me/api-keys", response_class=HTMLResponse)
async def ui_create_api_key(
    request: Request,
    _: None = Depends(require_csrf),
    name: str = Form(default=""),
    days: int = Query(default=30, ge=1, le=365),
    current_user: SessionUser = Depends(require_user),
    session: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
    key_name = name.strip()
    if not key_name:
        context = await _load_me_context(session, current_user.id, days=days)
        context.update(
            {
                "current_user": current_user,
                "new_api_key_token": None,
                "error": "API key name cannot be empty",
            }
        )
        return _template_response(
            request=request,
            name="me.html",
            context=context,
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    token = _generate_api_token()
    key = ApiKey(
        user_id=current_user.id,
        name=key_name,
        token_prefix=token[:20],
        token_hash=hash_api_token(token),
    )
    session.add(key)
    await session.commit()

    context = await _load_me_context(session, current_user.id, days=days)
    context.update(
        {
            "current_user": current_user,
            "new_api_key_token": token,
            "error": None,
        }
    )
    return _template_response(request=request, name="me.html", context=context)


@router.post("/me/api-keys/{id}/revoke")
async def ui_revoke_api_key(
    request: Request,
    id: int,
    _: None = Depends(require_csrf),
    current_user: SessionUser = Depends(require_user),
    session: AsyncSession = Depends(get_db_session),
) -> RedirectResponse:
    key = await session.scalar(
        select(ApiKey).where(ApiKey.id == id, ApiKey.user_id == current_user.id)
    )
    if not key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")

    if not key.revoked_at:
        key.revoked_at = datetime.utcnow()
        await session.commit()

    return _redirect("/ui/me")


@router.get("/admin", response_class=HTMLResponse)
async def ui_admin_page(
    request: Request,
    current_admin: SessionUser = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
    rows = await session.scalars(select(User).order_by(User.created_at.asc()))
    return _template_response(
        request=request,
        name="admin.html",
        context={
            "current_admin": current_admin,
            "users": list(rows),
            "error": None,
        },
    )


@router.post("/admin/users", response_class=HTMLResponse)
async def ui_admin_create_user(
    request: Request,
    _: None = Depends(require_csrf),
    username: str = Form(default=""),
    password: str = Form(default=""),
    role: str = Form(default="user"),
    is_active: bool = Form(default=False),
    current_admin: SessionUser = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
    username = username.strip()
    error = None

    if not username:
        error = "Username cannot be empty"
    elif not password:
        error = "Password cannot be empty"
    elif role not in {"admin", "user"}:
        error = "Role must be admin or user"

    if not error:
        existing = await session.scalar(select(User).where(User.username == username))
        if existing:
            error = "Username already exists"

    if error:
        rows = await session.scalars(select(User).order_by(User.created_at.asc()))
        return _template_response(
            request=request,
            name="admin.html",
            status_code=status.HTTP_400_BAD_REQUEST,
            context={
                "current_admin": current_admin,
                "users": list(rows),
                "error": error,
            },
        )

    user = User(
        username=username,
        password_hash=hash_password(password),
        role=role,
        is_active=is_active,
    )
    session.add(user)
    await session.commit()
    return _redirect("/ui/admin")


@router.get("/admin/users/{id}", response_class=HTMLResponse)
async def ui_admin_user_detail(
    id: int,
    request: Request,
    days: int = Query(default=30, ge=1, le=365),
    current_admin: SessionUser = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
    context = await _load_admin_user_detail_context(session, id, days=days)
    context.update(
        {
            "current_admin": current_admin,
            "error": None,
            "new_password": None,
        }
    )
    return _template_response(request=request, name="admin_user_detail.html", context=context)


@router.post("/admin/users/{id}/disable")
async def ui_admin_disable_user(
    request: Request,
    id: int,
    _: None = Depends(require_csrf),
    current_admin: SessionUser = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> RedirectResponse:
    if current_admin.id == id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot disable your own account",
        )

    user = await session.get(User, id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if user.is_active:
        user.is_active = False
        await session.commit()

    return _redirect(f"/ui/admin/users/{id}")


@router.post("/admin/users/{id}/enable")
async def ui_admin_enable_user(
    request: Request,
    id: int,
    csrf_ok: None = Depends(require_csrf),
    session: AsyncSession = Depends(get_db_session),
    _: SessionUser = Depends(require_admin),
) -> RedirectResponse:
    user = await session.get(User, id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if not user.is_active:
        user.is_active = True
        await session.commit()

    return _redirect(f"/ui/admin/users/{id}")


@router.post("/admin/users/{id}/reset-password", response_class=HTMLResponse)
async def ui_admin_reset_password(
    id: int,
    request: Request,
    _: None = Depends(require_csrf),
    password: str = Form(default=""),
    days: int = Query(default=30, ge=1, le=365),
    current_admin: SessionUser = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
) -> HTMLResponse:
    user = await session.get(User, id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    new_password = password or secrets.token_urlsafe(12)
    user.password_hash = hash_password(new_password)
    await session.commit()

    context = await _load_admin_user_detail_context(session, id, days=days)
    context.update(
        {
            "current_admin": current_admin,
            "error": None,
            "new_password": new_password,
        }
    )
    return _template_response(request=request, name="admin_user_detail.html", context=context)
