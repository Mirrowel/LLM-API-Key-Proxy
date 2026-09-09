from __future__ import annotations

from pathlib import Path

import json

import pytest

from rotator_library.native_provider import NativeHTTPTransport, NativeProviderContext, NativeProviderExecutor
from rotator_library.transaction_logger import TransactionLogger
from rotator_library.utils import zstd_io


@pytest.fixture(autouse=True)
def _trace_level_2(monkeypatch):
    """Trace mechanics live at L2 (D15 tiers)."""
    monkeypatch.setenv("TRANSACTION_LOG_LEVEL", "2")


def _trace_text(log_dir):
    entries = zstd_io.read_jsonl_any(Path(log_dir) / "transform_trace.jsonl")
    return "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries)


class FakeNativeTransport(NativeHTTPTransport):
    def __init__(self):
        pass

    async def post_json(self, endpoint, *, headers, payload):
        return {
            "id": "chatcmpl_1",
            "model": "gpt-test",
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "completion_tokens_details": {"reasoning_tokens": 2}},
        }


@pytest.mark.asyncio
async def test_native_executor_traces_normalized_usage(tmp_path) -> None:
    logger = TransactionLogger("openai", "gpt-test", parent_dir=tmp_path)
    context = NativeProviderContext(
        provider="openai",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://example.test/chat",
        headers={},
        transaction_logger=logger,
    )

    await NativeProviderExecutor().execute({"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}, context, FakeNativeTransport())

    entries = [json.loads(line) for line in _trace_text(logger.log_dir).splitlines()]
    usage_entries = [entry for entry in entries if entry["pass_name"] == "usage_accounting_summary"]
    assert usage_entries[-1]["data"]["usage"]["completion_tokens"] == 4
    assert usage_entries[-1]["data"]["usage"]["reasoning_tokens"] == 2
