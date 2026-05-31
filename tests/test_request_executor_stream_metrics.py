from __future__ import annotations

import json

import pytest

from rotator_library.client.streaming import StreamingHandler
from rotator_library.transaction_logger import TransactionLogger


async def _chunks():
    yield {"id": "chunk_1", "choices": [{"delta": {"content": "hi"}}]}
    yield {"id": "chunk_2", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _trace_passes(log_dir):
    return [json.loads(line)["pass_name"] for line in (log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.asyncio
async def test_streaming_handler_emits_lifecycle_metrics_without_changing_output(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)

    chunks = [chunk async for chunk in StreamingHandler().wrap_stream(_chunks(), "cred", "openai/gpt-test", transaction_logger=logger)]

    assert chunks[0].startswith("data: ")
    assert chunks[-1] == "data: [DONE]\n\n"
    pass_names = _trace_passes(logger.log_dir)
    assert "stream_started" in pass_names
    assert "stream_first_byte" in pass_names
    assert "stream_first_visible_output" in pass_names
    assert "stream_completed" in pass_names
    assert "stream_metrics_final" in pass_names
