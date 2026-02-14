import pytest
from sqlalchemy import text

from proxy_app.db import create_db_engine


@pytest.mark.asyncio
async def test_sqlite_pragmas_are_applied(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("SQLITE_BUSY_TIMEOUT_MS", "7000")
    db_file = tmp_path / "pragmas.db"
    engine = create_db_engine(f"sqlite+aiosqlite:///{db_file}")

    try:
        async with engine.connect() as conn:
            journal_mode = (
                await conn.execute(text("PRAGMA journal_mode"))
            ).scalar_one_or_none()
            synchronous = (
                await conn.execute(text("PRAGMA synchronous"))
            ).scalar_one_or_none()
            foreign_keys = (
                await conn.execute(text("PRAGMA foreign_keys"))
            ).scalar_one_or_none()
            busy_timeout = (
                await conn.execute(text("PRAGMA busy_timeout"))
            ).scalar_one_or_none()

        assert str(journal_mode).lower() == "wal"
        assert int(synchronous) == 1  # NORMAL
        assert int(foreign_keys) == 1
        assert int(busy_timeout) == 7000
    finally:
        await engine.dispose()
