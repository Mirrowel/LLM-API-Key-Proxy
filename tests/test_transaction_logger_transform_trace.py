from __future__ import annotations

import json

import pytest

from rotator_library.client.executor import RequestExecutor
from rotator_library.client.transforms import ProviderTransforms
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
    assert entries[0]["request_id"] == logger.request_id
    assert entries[0]["direction"] == "request"
    assert entries[0]["data"]["api_key"] == REDACTED
    assert (logger.log_dir / "transforms" / "0001_raw_client_request.json").exists()


def test_log_transformed_request_records_trace_even_when_legacy_file_skips(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    logger.set_trace_context(session_id="session_1", scope_key="scope_1", classifier="class_a")
    request = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}

    logger.log_transformed_request(request, dict(request), credential_id="cred_1")

    entries = _trace_entries(logger.log_dir)
    assert entries[0]["pass_name"] == "prepared_provider_request"
    assert entries[0]["changed_from_previous"] is False
    assert entries[0]["credential_id"] == "cred_1"
    assert entries[0]["session_id"] == "session_1"
    assert entries[0]["scope_key"] == "scope_1"
    assert entries[0]["classifier"] == "class_a"
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
    assert entries[0]["request_id"] == logger.request_id
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
    assert pass_names.count("stream_done_event") == 1
    assert "assembled_stream_response" in pass_names
    assert "final_client_response" in pass_names


@pytest.mark.asyncio
async def test_stream_wrapper_records_error_events(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    executor = RequestExecutor.__new__(RequestExecutor)

    async def stream():
        yield 'data: {"error":{"type":"rate_limit","message":"slow down"}}\n\n'
        yield "data: [DONE]\n\n"

    chunks = [chunk async for chunk in executor._transaction_logging_stream_wrapper(stream(), logger, {})]

    assert chunks[-1] == "data: [DONE]\n\n"
    pass_names = [entry["pass_name"] for entry in _trace_entries(logger.log_dir)]
    assert "stream_error_event" in pass_names
    assert "stream_done_event" in pass_names


def test_executor_terminal_stream_errors_are_traced(tmp_path) -> None:
    class Context:
        pass

    context = Context()
    context.transaction_logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    context.streaming = True
    context.provider = "openai"
    context.model = "openai/gpt-test"
    context.session_id = None
    context.usage_manager_key = "openai"
    context.classifier = None

    executor = RequestExecutor.__new__(RequestExecutor)

    lines = executor._terminal_stream_error_lines(context, {"error": {"type": "proxy_error"}})

    assert lines[-1] == "data: [DONE]\n\n"
    pass_names = [entry["pass_name"] for entry in _trace_entries(context.transaction_logger.log_dir)]
    assert pass_names == ["stream_error_event", "stream_done_event"]


def test_transaction_logger_disabled_writes_no_trace(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", enabled=False, parent_dir=tmp_path)

    logger.log_request({"model": "gpt-test"})
    logger.log_response({"model": "gpt-test"})

    assert logger.log_dir is None
    assert not list(tmp_path.iterdir())


def test_provider_error_trace_scrubs_header_like_secret_text(tmp_path) -> None:
    logger = TransactionLogger("gemini_cli", "gemini_cli/gemini-test", parent_dir=tmp_path)
    provider_logger = ProviderLogger(logger.get_context())

    provider_logger.log_error("upstream failed Authorization: Bearer secret-token")

    entries = _trace_entries(logger.log_dir)
    assert entries[0]["pass_name"] == "provider_error"
    assert "secret-token" not in entries[0]["data"]["message"]
    assert "[REDACTED]" in entries[0]["data"]["message"]


def test_log_transform_error_uses_standard_shape_and_scrubs_text(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)

    logger.log_transform_error(
        "after_field_cache_injection",
        RuntimeError("bad Authorization: Bearer secret"),
        payload={"cookie": "sid=secret"},
    )

    entries = _trace_entries(logger.log_dir)
    assert entries[0]["pass_name"] == "transform_log_error"
    assert entries[0]["data"]["failed_pass_name"] == "after_field_cache_injection"
    assert "secret" not in json.dumps(entries[0]["data"])


def test_trace_redacts_camel_case_secret_keys(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)

    logger.log_transform_pass(
        "camel_secret_payload",
        {"apiKey": "a", "accessToken": "b", "refreshToken": "c", "clientSecret": "d", "idToken": "e"},
        direction="request",
        stage="client",
    )

    data = _trace_entries(logger.log_dir)[0]["data"]
    assert set(data.values()) == {REDACTED}


@pytest.mark.asyncio
async def test_provider_transforms_trace_each_live_boundary(tmp_path) -> None:
    class HookPlugin:
        async def transform_request(self, kwargs, model, credential):
            kwargs["hooked"] = credential
            return ["hooked request"]

        def get_model_options(self, model):
            return {"reasoning_effort": "low", "temperature": 0.2}

    class Config:
        def convert_for_litellm(self, provider_override=None, **kwargs):
            converted = dict(kwargs)
            converted["converted_for_litellm"] = True
            return converted

    logger = TransactionLogger("dedaluslabs", "dedaluslabs/test", parent_dir=tmp_path)
    transforms = ProviderTransforms(
        {"dedaluslabs": HookPlugin()},
        provider_config=Config(),
    )

    result = await transforms.apply(
        "dedaluslabs",
        "dedaluslabs/test",
        "secret-credential",
        {"model": "dedaluslabs/test", "tool_choice": "auto"},
        transaction_logger=logger,
        credential_id="stable_cred",
        transport="http",
        trace_metadata={"scope_key": "scope"},
    )

    assert "tool_choice" not in result
    assert result["hooked"] == "secret-credential"
    assert result["reasoning_effort"] == "low"
    assert result["converted_for_litellm"] is True
    entries = _trace_entries(logger.log_dir)
    pass_names = [entry["pass_name"] for entry in entries]
    assert pass_names == [
        "pre_provider_transform_request",
        "after_builtin_provider_transform",
        "after_provider_hook_transform",
        "after_provider_model_options",
        "before_litellm_conversion",
        "after_litellm_conversion",
    ]
    assert all(entry["credential_id"] == "stable_cred" for entry in entries)
    assert entries[1]["metadata"]["transform_provider"] == "dedaluslabs"
    assert entries[-1]["changed_from_previous"] is True


@pytest.mark.asyncio
async def test_provider_builtin_transform_errors_are_traced(tmp_path) -> None:
    def broken_transform(kwargs, model, provider):
        raise RuntimeError("bad apiKey: secret")

    logger = TransactionLogger("broken", "broken/test", parent_dir=tmp_path)
    transforms = ProviderTransforms({})
    transforms._transforms["broken"] = [broken_transform]

    with pytest.raises(RuntimeError):
        await transforms.apply(
            "broken",
            "broken/test",
            "cred",
            {"model": "broken/test", "apiKey": "secret"},
            transaction_logger=logger,
            credential_id="cred_1",
        )

    entries = _trace_entries(logger.log_dir)
    error_entry = [entry for entry in entries if entry["pass_name"] == "transform_log_error"][-1]
    assert error_entry["data"]["failed_pass_name"] == "builtin_provider_transform"
    assert "secret" not in json.dumps(error_entry["data"])


@pytest.mark.asyncio
async def test_provider_transforms_do_not_deepcopy_for_trace_when_disabled() -> None:
    class NoCopyValue:
        def __deepcopy__(self, memo):
            raise AssertionError("trace comparison should not copy when tracing is disabled")

    class HookPlugin:
        async def transform_request(self, kwargs, model, credential):
            kwargs["hooked"] = True
            return ["hooked request"]

    transforms = ProviderTransforms({"dedaluslabs": HookPlugin()})

    result = await transforms.apply(
        "dedaluslabs",
        "dedaluslabs/test",
        "secret-credential",
        {"model": "dedaluslabs/test", "tool_choice": "auto", "opaque": NoCopyValue()},
    )

    assert "tool_choice" not in result
    assert result["hooked"] is True
