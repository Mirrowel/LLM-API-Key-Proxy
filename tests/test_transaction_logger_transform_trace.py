from __future__ import annotations

import json

import pytest

from rotator_library.client.executor import RequestExecutor
from rotator_library.client.transforms import ProviderTransforms
from rotator_library.transaction_logger import ProviderLogger, TransactionLogger
from rotator_library.transform_trace import REDACTED
from tests.txn_helpers import (
    boundaries,
    by_pass,
    change_text,
    changes,
    client_chunks,
    error_records,
    pass_names,
    record_errors,
    sealed,
    stream_chunks,
)


def test_log_request_records_sanitized_client_boundary(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)

    logger.log_request({"model": "gpt-test", "api_key": "secret", "messages": [{"role": "user", "content": "hi"}]})

    payload = boundaries(logger)["client_request"]
    assert payload["api_key"] == REDACTED
    assert payload["model"] == "gpt-test"
    assert logger._record.boundary_order == ["client_request"]


def test_log_transformed_request_skips_when_identical_but_keeps_context(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    logger.set_trace_context(session_id="session_1", scope_key="scope_1", classifier="class_a")
    request = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}

    logger.log_transformed_request(request, dict(request), credential_id="cred_1")

    # Identical after framework-key strip: no provider_request boundary, but the
    # correlation context is still carried on the record.
    assert "provider_request" not in boundaries(logger)
    assert logger._record.metadata["session_id"] == "session_1"
    assert logger._record.metadata["scope_key"] == "scope_1"
    assert logger._record.metadata["classifier"] == "class_a"


def test_log_response_and_stream_chunk_record_client_egress_and_chunks(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    logger.log_request({"model": "gpt-test", "stream": True})

    logger.log_stream_chunk({"choices": [{"delta": {"content": "Hi"}}]})
    logger.log_response({"model": "gpt-test", "choices": [], "usage": {"total_tokens": 2}})

    assert client_chunks(logger) == [{"choices": [{"delta": {"content": "Hi"}}]}]
    assert "client_egress" in boundaries(logger)
    envelope = sealed(logger)
    assert envelope["format"] == "proxy-transaction/2"
    assert envelope["boundaries"]["client_egress"]["usage"] == {"total_tokens": 2}


def test_log_response_redacts_headers_channel(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)

    logger.log_response(
        {"model": "gpt-test", "choices": []},
        headers={
            "Authorization": "Bearer response-secret",
            "X-Goog-Api-Key": "AIzaRESPONSESECRET",
            "Content-Type": "application/json",
        },
    )

    headers = sealed(logger)["metadata"]["response_headers"]
    assert headers["Authorization"] == REDACTED
    assert headers["X-Goog-Api-Key"] == REDACTED
    assert headers["Content-Type"] == "application/json"


def test_provider_logger_records_provider_boundaries_and_errors(tmp_path) -> None:
    logger = TransactionLogger("gemini_cli", "gemini_cli/gemini-test", parent_dir=tmp_path)
    provider_logger = ProviderLogger(logger.get_context())

    provider_logger.log_request({"credential_identifier": "secret", "body": {"text": "hi"}})
    provider_logger.log_response_chunk("data: chunk")
    provider_logger.log_final_response({"ok": True})
    provider_logger.log_error("provider failed")

    assert logger._record.boundary_order == ["provider_request", "provider_response"]
    assert boundaries(logger)["provider_request"] == {"credential_identifier": REDACTED, "body": {"text": "hi"}}
    assert boundaries(logger)["provider_response"] == {"ok": True}
    assert stream_chunks(logger) == ["data: chunk"]
    assert record_errors(logger)[0]["type"] == "provider_error"


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
    names = pass_names(logger)
    assert names.count("raw_stream_chunk") == 3
    assert names.count("stream_done_event") == 1
    assert "assembled_stream_response" in names
    assert len(client_chunks(logger)) == 2
    assert "client_egress" in boundaries(logger)


@pytest.mark.asyncio
async def test_stream_wrapper_records_error_events(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)
    executor = RequestExecutor.__new__(RequestExecutor)

    async def stream():
        yield 'data: {"error":{"type":"rate_limit","message":"slow down"}}\n\n'
        yield "data: [DONE]\n\n"

    chunks = [chunk async for chunk in executor._transaction_logging_stream_wrapper(stream(), logger, {})]

    assert chunks[-1] == "data: [DONE]\n\n"
    names = pass_names(logger)
    assert "stream_error_event" in names
    assert "stream_done_event" in names


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
    assert pass_names(context.transaction_logger) == ["stream_error_event", "stream_done_event"]


def test_transaction_logger_disabled_writes_no_trace(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", enabled=False, parent_dir=tmp_path)

    logger.log_request({"model": "gpt-test"})
    logger.log_response({"model": "gpt-test"})

    assert logger.log_dir is None
    assert logger._record is None
    assert not list(tmp_path.iterdir())


def test_provider_error_trace_scrubs_header_like_secret_text(tmp_path) -> None:
    logger = TransactionLogger("gemini_cli", "gemini_cli/gemini-test", parent_dir=tmp_path)
    provider_logger = ProviderLogger(logger.get_context())

    provider_logger.log_error("upstream failed Authorization: Bearer secret-token")

    message = record_errors(logger)[0]["message"]
    assert "secret-token" not in message
    assert "[REDACTED]" in message


def test_log_transform_error_uses_standard_shape_and_scrubs_text(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)

    logger.log_transform_error(
        "after_field_cache_injection",
        RuntimeError("bad Authorization: Bearer secret"),
        payload={"cookie": "sid=secret"},
    )

    records = error_records(logger)
    assert records[0]["failed_pass_name"] == "after_field_cache_injection"
    assert "secret" not in change_text(logger)
    assert "secret" not in json.dumps(records)


def test_trace_redacts_camel_case_secret_keys(tmp_path) -> None:
    logger = TransactionLogger("openai", "openai/gpt-test", parent_dir=tmp_path)

    logger.log_transform_pass(
        "camel_secret_payload",
        {"apiKey": "a", "accessToken": "b", "refreshToken": "c", "clientSecret": "d", "idToken": "e"},
        direction="request",
        stage="client",
    )

    data = by_pass(logger, "camel_secret_payload")[0]["data"]
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
    names = pass_names(logger)
    assert names == [
        "pre_provider_transform_request",
        "after_builtin_provider_transform",
        "after_provider_hook_transform",
        "after_provider_model_options",
        "before_litellm_conversion",
        "after_litellm_conversion",
    ]
    hook_entry = by_pass(logger, "after_provider_hook_transform")[-1]
    assert hook_entry["data"]["hooked"] == "secret-credential"
    conversion_entry = by_pass(logger, "after_litellm_conversion")[-1]
    assert conversion_entry["data"]["converted_for_litellm"] is True


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

    error_entry = [entry for entry in error_records(logger) if entry["failed_pass_name"] == "builtin_provider_transform"][-1]
    assert "secret" not in json.dumps(error_entry)


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
