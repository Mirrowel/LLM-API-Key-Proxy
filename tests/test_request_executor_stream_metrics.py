from __future__ import annotations

from pathlib import Path

import json

import pytest

from rotator_library.client.stream_ops import (
    ChatWireStreamAdapter,
    NeutralStreamPipeline,
)
from rotator_library.protocols.types import ProtocolContext
from rotator_library.transaction_logger import TransactionLogger
from rotator_library.utils import zstd_io


@pytest.fixture(autouse=True)
def _trace_level_2(monkeypatch):
    """Trace mechanics live at L2 (D15 tiers)."""
    monkeypatch.setenv("TRANSACTION_LOG_LEVEL", "2")


def _protocol_context(model: str) -> ProtocolContext:
    provider = model.split("/", 1)[0] if "/" in model else "openai"
    return ProtocolContext(
        provider=provider,
        model=model,
        source_protocol="openai_chat",
        target_protocol="openai_chat",
        input_protocol="openai_chat",
        provider_protocol="openai_chat",
        client_protocol="openai_chat",
        transport="sse",
    )


def _pipeline(model: str, **kwargs) -> NeutralStreamPipeline:
    return NeutralStreamPipeline(
        client_protocol_name="openai_chat",
        protocol_context=_protocol_context(model),
        model=model,
        **kwargs,
    )


async def _run_chat(raw_stream, model: str, **kwargs):
    pipeline = _pipeline(model, **kwargs)
    adapter = ChatWireStreamAdapter(model, repair_state=pipeline.repair_state)
    return [frame async for frame in pipeline.run(adapter.events(raw_stream, pipeline.usage))]


async def _chunks():
    yield {"id": "chunk_1", "choices": [{"delta": {"content": "hi"}}]}
    yield {"id": "chunk_2", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _trace_entries(log_dir):
    from rotator_library.utils import zstd_io

    return zstd_io.read_jsonl_any(Path(log_dir) / "transform_trace.jsonl")


def _trace_text_exists(log_dir):
    from rotator_library.utils import zstd_io

    return (Path(log_dir) / "transform_trace.jsonl").exists() or (Path(log_dir) / "transform_trace.jsonl.zst").exists()


def _trace_passes(log_dir):
    return [entry["pass_name"] for entry in zstd_io.read_jsonl_any(Path(log_dir) / "transform_trace.jsonl")]


@pytest.mark.asyncio
async def test_streaming_handler_emits_lifecycle_metrics_without_changing_output(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)

    chunks = await _run_chat(_chunks(), "openai/gpt-test", transaction_logger=logger)

    assert chunks[0].startswith("data: ")
    assert chunks[-1] == "data: [DONE]\n\n"
    pass_names = _trace_passes(logger.log_dir)
    assert "stream_started" in pass_names
    assert "stream_first_byte" in pass_names
    assert "stream_first_visible_output" in pass_names
    assert "stream_completed" in pass_names
    assert "stream_metrics_final" in pass_names


@pytest.mark.asyncio
async def test_stream_trace_metrics_can_be_disabled_without_changing_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("STREAM_TRACE_METRICS", "false")
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)

    chunks = await _run_chat(_chunks(), "openai/gpt-test", transaction_logger=logger)

    assert chunks[0].startswith("data: ")
    assert chunks[-1] == "data: [DONE]\n\n"
    if _trace_text_exists(logger.log_dir):
        pass_names = _trace_passes(logger.log_dir)
        assert "stream_started" in pass_names
        assert "stream_metrics_final" in pass_names


@pytest.mark.asyncio
async def test_pipeline_parses_formatted_sse_chunks(monkeypatch) -> None:
    async def formatted_stream():
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'

    chunks = await _run_chat(formatted_stream(), "openai/gpt-test")

    assert '"content": "hi"' in chunks[0]
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_pipeline_does_not_duplicate_direct_done_sentinel(monkeypatch) -> None:
    async def done_stream():
        yield "data: [DONE]\n\n"

    chunks = await _run_chat(done_stream(), "openai/gpt-test")

    assert chunks.count("data: [DONE]\n\n") == 1


@pytest.mark.asyncio
async def test_pipeline_splits_mixed_reasoning_and_content_delta() -> None:
    """Clients receive the reasoning-to-answer transition as distinct events."""

    async def mixed_stream():
        yield {
            "id": "chatcmpl-diffusion",
            "model": "google/diffusiongemma-26b-a4b-it",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "reasoning_content": "end of thought",
                        "content": "beginning of answer",
                    },
                }
            ],
        }
        yield {
            "id": "chatcmpl-diffusion",
            "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        }

    completed_responses = []
    chunks = await _run_chat(
        mixed_stream(),
        "nvidia_nim/google/diffusiongemma-26b-a4b-it",
        response_callback=completed_responses.append,
    )

    reasoning_chunk = json.loads(chunks[0][len("data: ") :])
    content_chunk = json.loads(chunks[1][len("data: ") :])
    reasoning_delta = reasoning_chunk["choices"][0]["delta"]
    content_delta = content_chunk["choices"][0]["delta"]

    assert reasoning_delta["reasoning_content"] == "end of thought"
    assert "content" not in reasoning_delta
    assert reasoning_chunk["choices"][0]["finish_reason"] is None
    assert reasoning_chunk.get("usage") is None
    assert content_delta["content"] == "beginning of answer"
    assert "reasoning_content" not in content_delta
    assert completed_responses[0]["messages"][0]["content"] == "beginning of answer"
    assert chunks[-1] == "data: [DONE]\n\n"
