import base64
import hashlib
import logging
import os
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from proxy_app.db_models import Base, User

PBKDF2_ITERATIONS = 200000


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def get_database_url(root_dir: Path) -> str:
    configured = os.getenv("DATABASE_URL")
    if configured:
        return configured
    db_dir = root_dir / "data"
    db_dir.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{db_dir / 'proxy.db'}"


async def _bootstrap_initial_admin(session: AsyncSession) -> bool:
    username = (os.getenv("INITIAL_ADMIN_USERNAME") or "").strip()
    password = os.getenv("INITIAL_ADMIN_PASSWORD") or ""
    if not username or not password:
        logging.info("INITIAL_ADMIN_USERNAME/PASSWORD not set, skipping bootstrap")
        return False

    existing = await session.scalar(select(User).where(User.username == username))
    if existing:
        return False

    admin = User(
        username=username,
        password_hash=hash_password(password),
        role="admin",
        is_active=True,
    )
    session.add(admin)
    await session.commit()
    logging.info("Bootstrapped initial admin user '%s'", username)
    return True


async def init_db(root_dir: Path) -> async_sessionmaker[AsyncSession]:
    database_url = get_database_url(root_dir)
    engine = create_async_engine(database_url, future=True)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_maker() as session:
        await _bootstrap_initial_admin(session)

    return session_maker
