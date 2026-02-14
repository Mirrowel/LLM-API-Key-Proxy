import pytest
from sqlalchemy import select

from proxy_app.api_token_auth import ApiActor
from proxy_app.db_models import UsageEvent
from proxy_app.usage_recorder import UsageRecorder


@pytest.mark.asyncio
async def test_usage_recorder_non_stream_persists_usage_fields(session_maker) -> None:
    recorder = UsageRecorder(session_maker, batch_size=1, flush_interval_seconds=0.01)
    await recorder.start()

    await recorder.record_usage_event(
        actor={"user_id": 11, "api_key_id": 22},
        endpoint="/v1/chat/completions",
        model="openai/gpt-4o-mini",
        status_code=200,
        usage={
            "prompt_tokens": "10",
            "completion_tokens": 5,
            "total_tokens": "15",
            "cost_usd": "0.0025",
        },
        request_id="req-non-stream-1",
    )

    await recorder.stop()

    async with session_maker() as session:
        row = (await session.execute(select(UsageEvent))).scalar_one()

    assert row.user_id == 11
    assert row.api_key_id == 22
    assert row.provider == "openai"
    assert row.model == "openai/gpt-4o-mini"
    assert row.prompt_tokens == 10
    assert row.completion_tokens == 5
    assert row.total_tokens == 15
    assert row.cost_usd == pytest.approx(0.0025)
    assert row.error_type is None


@pytest.mark.asyncio
async def test_usage_recorder_streaming_fallback_persists_without_usage(session_maker) -> None:
    recorder = UsageRecorder(session_maker, batch_size=1, flush_interval_seconds=0.01)
    await recorder.start()

    await recorder.record_usage_event(
        actor=ApiActor(user_id=33, api_key_id=44, role="user", auth_source="user_api_key"),
        endpoint="/v1/chat/completions",
        model="anthropic/claude-3.5-sonnet",
        status_code=500,
        usage=None,
        request_id="req-stream-fallback-1",
        error_type="RuntimeError",
        error_message="stream interrupted",
    )

    await recorder.stop()

    async with session_maker() as session:
        row = (await session.execute(select(UsageEvent))).scalar_one()

    assert row.user_id == 33
    assert row.api_key_id == 44
    assert row.provider == "anthropic"
    assert row.prompt_tokens is None
    assert row.completion_tokens is None
    assert row.total_tokens is None
    assert row.cost_usd is None
    assert row.error_type == "RuntimeError"
    assert row.error_message == "stream interrupted"
