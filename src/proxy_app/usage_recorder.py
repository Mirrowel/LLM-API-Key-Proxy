import asyncio
import logging
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from proxy_app.db_models import UsageEvent

logger = logging.getLogger(__name__)


def get_usage_retention_days() -> int:
    raw = os.getenv("USAGE_RETENTION_DAYS", "30")
    try:
        days = int(raw)
    except ValueError:
        days = 30
    return max(1, days)


async def prune_usage_events(
    session_maker: async_sessionmaker[AsyncSession],
    *,
    retention_days: int | None = None,
) -> int:
    active_retention_days = retention_days or get_usage_retention_days()
    cutoff = datetime.utcnow() - timedelta(days=active_retention_days)
    async with session_maker() as session:
        result = await session.execute(
            delete(UsageEvent).where(UsageEvent.timestamp < cutoff)
        )
        await session.commit()
        deleted = result.rowcount if result.rowcount is not None else 0

    if deleted > 0:
        logger.info(
            "Pruned %d usage events older than %d days",
            deleted,
            active_retention_days,
        )
    return deleted


@dataclass(slots=True)
class UsageEventPayload:
    user_id: int | None
    api_key_id: int | None
    endpoint: str
    provider: str | None
    model: str | None
    request_id: str
    status_code: int
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    error_type: str | None
    error_message: str | None


_SENTINEL = object()


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _actor_field(actor: Any, key: str) -> Any:
    if actor is None:
        return None
    if isinstance(actor, dict):
        return actor.get(key)
    return getattr(actor, key, None)


class UsageRecorder:
    def __init__(
        self,
        session_maker: async_sessionmaker[AsyncSession],
        *,
        queue_maxsize: int = 2000,
        batch_size: int = 100,
        flush_interval_seconds: float = 1.0,
    ):
        self._session_maker = session_maker
        self._queue: asyncio.Queue[UsageEventPayload | object] = asyncio.Queue(
            maxsize=queue_maxsize
        )
        self._batch_size = batch_size
        self._flush_interval_seconds = flush_interval_seconds
        self._worker_task: asyncio.Task[None] | None = None
        self._accepting = False

    async def start(self) -> None:
        if self._worker_task:
            return
        self._accepting = True
        self._worker_task = asyncio.create_task(self._run_worker(), name="usage-recorder")

    async def stop(self) -> None:
        if not self._worker_task:
            return
        self._accepting = False
        try:
            self._queue.put_nowait(_SENTINEL)
        except asyncio.QueueFull:
            await self._queue.put(_SENTINEL)
        await self._worker_task
        self._worker_task = None

    async def record_usage_event(
        self,
        *,
        actor: Any,
        endpoint: str,
        model: str | None,
        status_code: int,
        usage: dict[str, Any] | None,
        request_id: str,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        if not self._accepting:
            return

        usage = usage or {}
        payload = UsageEventPayload(
            user_id=_maybe_int(_actor_field(actor, "user_id")),
            api_key_id=_maybe_int(_actor_field(actor, "api_key_id")),
            endpoint=endpoint,
            provider=model.split("/", 1)[0] if model else None,
            model=model,
            request_id=request_id,
            status_code=int(status_code),
            prompt_tokens=_maybe_int(usage.get("prompt_tokens")),
            completion_tokens=_maybe_int(usage.get("completion_tokens")),
            total_tokens=_maybe_int(usage.get("total_tokens")),
            cost_usd=_maybe_float(usage.get("cost_usd")),
            error_type=error_type,
            error_message=error_message,
        )

        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:
            logger.warning("Usage recorder queue full; dropping event request_id=%s", request_id)

    async def _run_worker(self) -> None:
        batch: list[UsageEventPayload] = []

        while True:
            item = await self._queue.get()

            if item is _SENTINEL:
                if batch:
                    await self._flush_batch(batch)
                    batch = []
                await self._drain_queue(batch)
                if batch:
                    await self._flush_batch(batch)
                return

            batch.append(item)

            if len(batch) >= self._batch_size:
                await self._flush_batch(batch)
                batch = []
                continue

            await self._collect_with_timeout(batch)
            if batch:
                await self._flush_batch(batch)
                batch = []

    async def _collect_with_timeout(self, batch: list[UsageEventPayload]) -> None:
        while len(batch) < self._batch_size:
            try:
                item = await asyncio.wait_for(
                    self._queue.get(), timeout=self._flush_interval_seconds
                )
            except asyncio.TimeoutError:
                return

            if item is _SENTINEL:
                try:
                    self._queue.put_nowait(_SENTINEL)
                except asyncio.QueueFull:
                    await self._queue.put(_SENTINEL)
                return

            batch.append(item)

    async def _drain_queue(self, batch: list[UsageEventPayload]) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            if item is _SENTINEL:
                continue
            batch.append(item)

    async def _flush_batch(self, batch: list[UsageEventPayload]) -> None:
        if not batch:
            return

        rows = [
            UsageEvent(
                user_id=item.user_id,
                api_key_id=item.api_key_id,
                endpoint=item.endpoint,
                provider=item.provider,
                model=item.model,
                request_id=item.request_id,
                status_code=item.status_code,
                prompt_tokens=item.prompt_tokens,
                completion_tokens=item.completion_tokens,
                total_tokens=item.total_tokens,
                cost_usd=item.cost_usd,
                error_type=item.error_type,
                error_message=item.error_message,
            )
            for item in batch
        ]

        try:
            async with self._session_maker() as session:
                session.add_all(rows)
                await session.commit()
        except Exception:
            logger.exception("Failed to flush %d usage events", len(batch))


_usage_recorder: UsageRecorder | None = None


async def start_usage_recorder(
    session_maker: async_sessionmaker[AsyncSession],
) -> UsageRecorder:
    global _usage_recorder
    recorder = UsageRecorder(session_maker)
    await recorder.start()
    _usage_recorder = recorder
    return recorder


async def stop_usage_recorder() -> None:
    global _usage_recorder
    if _usage_recorder is None:
        return
    await _usage_recorder.stop()
    _usage_recorder = None


def get_usage_recorder() -> UsageRecorder | None:
    return _usage_recorder


async def record_usage_event(
    *,
    actor: Any,
    endpoint: str,
    model: str | None,
    status_code: int,
    usage: dict[str, Any] | None,
    request_id: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    recorder = get_usage_recorder()
    if recorder is None:
        return

    await recorder.record_usage_event(
        actor=actor,
        endpoint=endpoint,
        model=model,
        status_code=status_code,
        usage=usage,
        request_id=request_id,
        error_type=error_type,
        error_message=error_message,
    )
