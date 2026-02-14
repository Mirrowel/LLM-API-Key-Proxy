from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from proxy_app.db_models import UsageEvent


def _sum_int(column: Any) -> Any:
    return func.coalesce(func.sum(column), 0)


def _sum_cost(column: Any) -> Any:
    return func.sum(column)


def _window_start(days: int) -> datetime:
    return datetime.utcnow() - timedelta(days=days)


async def fetch_usage_summary(
    session: AsyncSession,
    *,
    user_id: int,
) -> dict[str, int | float | None]:
    row = (
        await session.execute(
            select(
                func.count(UsageEvent.id),
                _sum_int(UsageEvent.prompt_tokens),
                _sum_int(UsageEvent.completion_tokens),
                _sum_int(UsageEvent.total_tokens),
                _sum_cost(UsageEvent.cost_usd),
            ).where(UsageEvent.user_id == user_id)
        )
    ).one()

    return {
        "request_count": int(row[0]),
        "prompt_tokens": int(row[1]),
        "completion_tokens": int(row[2]),
        "total_tokens": int(row[3]),
        "cost_usd": float(row[4]) if row[4] is not None else None,
    }


async def fetch_usage_by_day(
    session: AsyncSession,
    *,
    user_id: int,
    days: int,
) -> list[dict[str, int | float | str | None]]:
    day_bucket = func.date(UsageEvent.timestamp)
    rows = await session.execute(
        select(
            day_bucket,
            func.count(UsageEvent.id),
            _sum_int(UsageEvent.prompt_tokens),
            _sum_int(UsageEvent.completion_tokens),
            _sum_int(UsageEvent.total_tokens),
            _sum_cost(UsageEvent.cost_usd),
        )
        .where(
            UsageEvent.user_id == user_id,
            UsageEvent.timestamp >= _window_start(days),
        )
        .group_by(day_bucket)
        .order_by(day_bucket.asc())
    )

    return [
        {
            "day": str(row[0]),
            "request_count": int(row[1]),
            "prompt_tokens": int(row[2]),
            "completion_tokens": int(row[3]),
            "total_tokens": int(row[4]),
            "cost_usd": float(row[5]) if row[5] is not None else None,
        }
        for row in rows
    ]


async def fetch_usage_by_model(
    session: AsyncSession,
    *,
    user_id: int,
    days: int,
) -> list[dict[str, int | float | str | None]]:
    rows = await session.execute(
        select(
            UsageEvent.model,
            func.count(UsageEvent.id),
            _sum_int(UsageEvent.prompt_tokens),
            _sum_int(UsageEvent.completion_tokens),
            _sum_int(UsageEvent.total_tokens),
            _sum_cost(UsageEvent.cost_usd),
        )
        .where(
            UsageEvent.user_id == user_id,
            UsageEvent.timestamp >= _window_start(days),
        )
        .group_by(UsageEvent.model)
        .order_by(func.count(UsageEvent.id).desc(), UsageEvent.model.asc())
    )

    return [
        {
            "model": row[0],
            "request_count": int(row[1]),
            "prompt_tokens": int(row[2]),
            "completion_tokens": int(row[3]),
            "total_tokens": int(row[4]),
            "cost_usd": float(row[5]) if row[5] is not None else None,
        }
        for row in rows
    ]


async def fetch_api_key_last_used_map(
    session: AsyncSession,
    *,
    user_id: int,
    api_key_ids: list[int],
) -> dict[int, datetime]:
    if not api_key_ids:
        return {}

    rows = await session.execute(
        select(UsageEvent.api_key_id, func.max(UsageEvent.timestamp))
        .where(UsageEvent.user_id == user_id)
        .where(UsageEvent.api_key_id.in_(api_key_ids))
        .group_by(UsageEvent.api_key_id)
    )

    result: dict[int, datetime] = {}
    for api_key_id, last_used in rows:
        if api_key_id is not None and last_used is not None:
            result[int(api_key_id)] = last_used
    return result
