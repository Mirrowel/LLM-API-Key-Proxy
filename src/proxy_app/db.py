import base64
import hashlib
import logging
import os
from pathlib import Path

from sqlalchemy import event, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

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


def _is_sqlite_url(database_url: str) -> bool:
    driver = make_url(database_url).get_backend_name()
    return driver == "sqlite"


def _get_sqlite_busy_timeout_ms() -> int:
    raw = os.getenv("SQLITE_BUSY_TIMEOUT_MS", "5000")
    try:
        timeout = int(raw)
    except ValueError:
        timeout = 5000
    return max(1000, timeout)


def _configure_sqlite_engine(engine: AsyncEngine) -> None:
    busy_timeout_ms = _get_sqlite_busy_timeout_ms()

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        cursor.close()


def create_db_engine(database_url: str) -> AsyncEngine:
    connect_args = {}
    if _is_sqlite_url(database_url):
        connect_args["timeout"] = _get_sqlite_busy_timeout_ms() / 1000

    engine = create_async_engine(database_url, future=True, connect_args=connect_args)
    if _is_sqlite_url(database_url):
        _configure_sqlite_engine(engine)
    return engine


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
    engine = create_db_engine(database_url)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_maker() as session:
        await _bootstrap_initial_admin(session)

    return session_maker


async def init_db_runtime(
    root_dir: Path,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    database_url = get_database_url(root_dir)
    engine = create_db_engine(database_url)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with session_maker() as session:
        await _bootstrap_initial_admin(session)

    return engine, session_maker
