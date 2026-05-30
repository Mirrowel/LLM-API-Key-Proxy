from __future__ import annotations

import json

import pytest

from rotator_library.client.executor import RequestExecutor
from rotator_library.transaction_logger import ProviderLogger, TransactionLogger
from rotator_library.transform_trace import REDACTED


def _trace_entries(log_dir):
    trace_file = log_dir / "transform_trace.jsonl"
    return [json.loads(line) for line in trace_file.read_text(encoding="utf-8").splitlines()]


def test_log_request_writes_legacy_file_and_raw_trace(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)

    logger.log_request({"model": "gpt-test", "api_key": "secret", "messages": [{"role": "user", "content": "hi"}]})

    assert (logger.log_dir / "request.json").exists()
    entries = _trace_entries(logger.log_dir)
    assert entries[0]["pass_name"] == "raw_client_request"
    assert entries[0]["direction"] == "request"
    assert entries[0]["data"]["api_key"] == REDACTED
    assert (logger.log_dir / "transforms" / "0001_raw_client_request.json").exists()


def test_log_transformed_request_records_trace_even_when_legacy_file_skips(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    request = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}

    logger.log_transformed_request(request, dict(request))

    entries = _trace_entries(logger.log_dir)
    assert entries[0]["pass_name"] == "prepared_provider_request"
    assert entries[0]["changed_from_previous"] is False
    assert not (logger.log_dir / "request_transformed.json").exists()


def test_log_response_and_stream_chunk_write_trace_entries(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    logger.log_request({"model": "gpt-test", "stream": True})

    logger.log_stream_chunk({"choices": [{"delta": {"content": "Hi"}}]})
    logger.log_response({"model": "gpt-test", "choices": [], "usage": {"total_tokens": 2}})

    entries = _trace_entries(logger.log_dir)
    pass_names = [entry["pass_name"] for entry in entries]
    assert "parsed_stream_chunk" in pass_names
    assert "final_client_response" in pass_names
    stream_entry = next(entry for entry in entries if entry["pass_name"] == "parsed_stream_chunk")
    assert stream_entry["direction"] == "stream"
    assert not (logger.log_dir / "transforms" / "0002_parsed_stream_chunk.json").exists()


def test_provider_logger_writes_provider_trace_entries(tmp_path) -> None:
    logger = TransactionLogger("gemini_cli", "gemini_cli/gemini-test", parent_dir=tmp_path)
    provider_logger = ProviderLogger(logger.get_context())

    provider_logger.log_request({"credential_identifier": "secret", "body": {"text": "hi"}})
    provider_logger.log_response_chunk("data: chunk")
    provider_logger.log_final_response({"ok": True})
    provider_logger.log_error("provider failed")

    entries = _trace_entries(logger.log_dir)
    pass_names = [entry["pass_name"] for entry in entries]
    assert pass_names == [
        "provider_request_payload",
        "provider_raw_stream_chunk",
        "provider_final_response",
        "provider_error",
    ]
    assert entries[0]["component"] == "provider"
    assert entries[0]["data"]["credential_identifier"] == REDACTED
    assert (logger.log_dir / "provider" / "request_payload.json").exists()


@pytest.mark.asyncio
async def test_stream_wrapper_records_raw_parsed_and_assembled_trace(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    executor = RequestExecutor.__new__(RequestExecutor)

    async def stream():
        yield 'data: {"id":"chunk_1","choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    chunks = [chunk async for chunk in executor._transaction_logging_stream_wrapper(stream(), logger, {})]

    assert chunks[-1] == "data: [DONE]\n\n"
    entries = _trace_entries(logger.log_dir)
    pass_names = [entry["pass_name"] for entry in entries]
    assert pass_names.count("raw_stream_chunk") == 3
    assert pass_names.count("parsed_stream_chunk") == 2
    assert "assembled_stream_response" in pass_names
    assert "final_client_response" in pass_names


def test_transaction_logger_disabled_writes_no_trace(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", enabled=False, parent_dir=tmp_path)

    logger.log_request({"model": "gpt-test"})
    logger.log_response({"model": "gpt-test"})

    assert logger.log_dir is None
    assert not list(tmp_path.iterdir())
