from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select

from proxy_app.db_models import UsageEvent
from proxy_app.usage_recorder import prune_usage_events


@pytest.mark.asyncio
async def test_prune_usage_events_removes_old_rows(session_maker) -> None:
    now = datetime.utcnow()
    old_time = now - timedelta(days=60)
    recent_time = now - timedelta(days=5)

    async with session_maker() as session:
        session.add_all(
            [
                UsageEvent(
                    timestamp=old_time,
                    user_id=1,
                    api_key_id=1,
                    endpoint="/v1/chat/completions",
                    provider="openai",
                    model="openai/gpt-4o-mini",
                    request_id="req-old",
                    status_code=200,
                ),
                UsageEvent(
                    timestamp=recent_time,
                    user_id=1,
                    api_key_id=1,
                    endpoint="/v1/chat/completions",
                    provider="openai",
                    model="openai/gpt-4o-mini",
                    request_id="req-recent",
                    status_code=200,
                ),
            ]
        )
        await session.commit()

    deleted = await prune_usage_events(session_maker, retention_days=30)
    assert deleted == 1

    async with session_maker() as session:
        count = await session.scalar(select(func.count(UsageEvent.id)))

    assert count == 1
