from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from rotator_library.client.executor import RequestExecutor
from rotator_library.client.filters import CredentialFilter
from rotator_library.client.transforms import ProviderTransforms
from rotator_library.core.types import RequestContext
from rotator_library.native_provider import NativeProviderContext, NativeProviderExecutor
from rotator_library.providers.provider_interface import ProviderInterface
from rotator_library.protocols import ProtocolContext, get_protocol
from rotator_library.protocols.streaming import ProtocolStreamConverter


PROTOCOLS = ("openai_chat", "anthropic_messages", "responses", "gemini")


def _source_frames(protocol: str) -> list[object]:
    if protocol == "openai_chat":
        return [
            {"id": "chat_1", "model": "model-a", "choices": [{"delta": {"reasoning_content": "think"}}]},
            {"id": "chat_1", "model": "model-a", "choices": [{"delta": {"content": "hi"}}]},
            {"id": "chat_1", "model": "model-a", "choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "weather", "arguments": '{"city":'}}]}}]},
            {"id": "chat_1", "model": "model-a", "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}},
            "[DONE]",
        ]
    if protocol == "anthropic_messages":
        return [
            {"type": "message_start", "message": {"id": "msg_1", "role": "assistant", "content": [], "model": "model-a", "usage": {"input_tokens": 2, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "think"}},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "call_1", "name": "weather", "input": {}}},
            {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"city":"Paris"}'}},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 3}},
            {"type": "message_stop"},
        ]
    if protocol == "responses":
        return [
            {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "model": "model-a", "output": []}},
            {"type": "response.output_item.added", "output_index": 0, "item": {"id": "rs_0", "type": "reasoning", "summary": [{"type": "summary_text", "text": "think"}]}},
            {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 1, "content_index": 0, "delta": "hi"},
            {"type": "response.output_item.added", "output_index": 2, "item": {"id": "fc_2", "type": "function_call", "call_id": "call_1", "name": "weather", "arguments": ""}},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_2", "call_id": "call_1", "output_index": 2, "delta": '{"city":"Paris"}'},
            {"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "model": "model-a", "output": [], "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}}},
        ]
    return [
        {"responseId": "gem_1", "modelVersion": "model-a", "candidates": [{"content": {"role": "model", "parts": [{"text": "think", "thought": True}, {"text": "hi"}, {"functionCall": {"id": "call_1", "name": "weather", "args": {"city": "Paris"}}}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5}},
        "[DONE]",
    ]


@pytest.mark.parametrize("source_protocol", PROTOCOLS)
@pytest.mark.parametrize("output_protocol", PROTOCOLS)
def test_streaming_matrix_preserves_text_reasoning_tools_usage_and_terminal_lifecycle(
    source_protocol: str,
    output_protocol: str,
) -> None:
    context = ProtocolContext(
        model="model-a",
        source_protocol=source_protocol,
        target_protocol=output_protocol,
        input_protocol=source_protocol,
        provider_protocol=source_protocol,
        output_protocol=output_protocol,
        transport="sse",
    )
    converter = ProtocolStreamConverter(
        get_protocol(source_protocol),
        get_protocol(output_protocol),
        context,
    )

    frames = [formatted for raw in _source_frames(source_protocol) for formatted in converter.convert(raw)]
    output = "".join(str(frame) for frame in frames)

    assert "hi" in output
    assert "think" in output
    assert "weather" in output
    assert "Paris" in output
    if output_protocol == "openai_chat":
        assert "chat.completion.chunk" in output
        assert "reasoning_content" in output
        assert 'data: [DONE]' in output
    elif output_protocol == "anthropic_messages":
        assert "event: message_start" in output
        assert "thinking_delta" in output
        assert "input_json_delta" in output
        assert "event: message_stop" in output
    elif output_protocol == "responses":
        assert "event: response.created" in output
        assert "response.reasoning_summary_text.delta" in output
        assert "response.function_call_arguments.delta" in output
        assert "event: response.completed" in output
    else:
        assert "modelVersion" in output
        assert "functionCall" in output
        assert '"thought": true' in output
        assert "finishReason" in output


@pytest.mark.parametrize("output_protocol", PROTOCOLS)
def test_streaming_errors_are_formatted_in_selected_protocol(output_protocol: str) -> None:
    context = ProtocolContext(model="model-a", source_protocol="openai_chat", target_protocol=output_protocol, transport="sse")
    converter = ProtocolStreamConverter(get_protocol("openai_chat"), get_protocol(output_protocol), context)

    output = "".join(converter.convert({"error": {"type": "rate_limit", "message": "slow down"}}))

    assert "slow down" in output
    if output_protocol == "anthropic_messages":
        assert "event: error" in output
    elif output_protocol == "responses":
        assert "event: response.failed" in output
    else:
        assert '"error"' in output


@pytest.mark.parametrize("output_protocol", ("openai_chat", "anthropic_messages", "gemini"))
def test_responses_failed_stream_is_an_error_in_every_alternate_output(output_protocol: str) -> None:
    context = ProtocolContext(model="model-a", source_protocol="responses", target_protocol=output_protocol, transport="sse")
    converter = ProtocolStreamConverter(get_protocol("responses"), get_protocol(output_protocol), context)

    output = "".join(converter.convert({
        "type": "response.failed",
        "response": {
            "id": "resp_failed",
            "status": "failed",
            "model": "model-a",
            "output": [],
            "error": {"type": "rate_limit", "message": "try later"},
        },
    }))

    assert "try later" in output
    assert "response.completed" not in output


def test_gemini_rejects_incomplete_foreign_tool_arguments_at_terminal() -> None:
    context = ProtocolContext(model="model-a", source_protocol="openai_chat", target_protocol="gemini", transport="sse")
    converter = ProtocolStreamConverter(get_protocol("openai_chat"), get_protocol("gemini"), context)
    converter.convert({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "weather", "arguments": '{"city":'}}]}}]})

    with pytest.raises(ValueError, match="incomplete streamed tool-call"):
        converter.convert("[DONE]")


def test_gemini_emits_fragmented_tool_call_exactly_once() -> None:
    context = ProtocolContext(model="model-a", source_protocol="responses", target_protocol="gemini", transport="sse")
    converter = ProtocolStreamConverter(get_protocol("responses"), get_protocol("gemini"), context)
    source = [
        {"type": "response.output_item.added", "output_index": 0, "item": {"id": "fc_0", "type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": ""}},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": "fc_0", "call_id": "call_1", "delta": '{"city":'},
        {"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": "fc_0", "call_id": "call_1", "delta": '"Paris"}'},
        {"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "model": "model-a", "output": []}},
    ]

    output = "".join(frame for event in source for frame in converter.convert(event))

    assert output.count("functionCall") == 1
    assert '"city": "Paris"' in output


def test_gemini_flushes_genuine_zero_argument_tool_call_once_at_terminal() -> None:
    context = ProtocolContext(model="model-a", source_protocol="responses", target_protocol="gemini", transport="sse")
    converter = ProtocolStreamConverter(get_protocol("responses"), get_protocol("gemini"), context)
    converter.convert({"type": "response.output_item.added", "output_index": 0, "item": {"id": "fc_0", "type": "function_call", "call_id": "call_1", "name": "ping", "arguments": ""}})

    output = "".join(converter.convert({"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "model": "model-a", "output": []}}))

    assert output.count("functionCall") == 1
    assert '"args": {}' in output


def test_stream_formatter_never_exposes_foreign_thought_signatures() -> None:
    context = ProtocolContext(model="model-a", source_protocol="gemini", target_protocol="gemini", transport="sse")
    converter = ProtocolStreamConverter(get_protocol("gemini"), get_protocol("gemini"), context)

    output = "".join(converter.convert({"candidates": [{"content": {"role": "model", "parts": [{"text": "hidden", "thought": True, "thoughtSignature": "opaque-secret"}]}}]}))

    assert "hidden" in output
    assert "opaque-secret" not in output


@pytest.mark.parametrize("output_protocol", PROTOCOLS)
def test_bare_done_never_fabricates_success(output_protocol: str) -> None:
    context = ProtocolContext(model="model-a", source_protocol="openai_chat", target_protocol=output_protocol, transport="sse")
    converter = ProtocolStreamConverter(get_protocol("openai_chat"), get_protocol(output_protocol), context)

    output = "".join(converter.convert("[DONE]"))

    assert '"finish_reason": "stop"' not in output
    assert '"stop_reason": "end_turn"' not in output
    assert '"finishReason": "STOP"' not in output
    if output_protocol == "responses":
        assert "response.incomplete" in output


def test_terminal_responses_error_reuses_active_stream_lifecycle() -> None:
    protocol_context = ProtocolContext(model="model-a", source_protocol="openai_chat", target_protocol="responses", transport="sse")
    converter = ProtocolStreamConverter(get_protocol("openai_chat"), get_protocol("responses"), protocol_context)
    initial = "".join(converter.convert({"choices": [{"delta": {"content": "partial"}}]}))
    response_id = json.loads(initial.split("data: ", 1)[1].split("\n", 1)[0])["response"]["id"]
    request_context = RequestContext(
        model="provider/model-a",
        provider="provider",
        kwargs={},
        streaming=True,
        credentials=[],
        deadline=9999999999.0,
        input_protocol_name="openai_chat",
        output_protocol_name="responses",
    )

    terminal = "".join(RequestExecutor.__new__(RequestExecutor)._terminal_stream_error_lines(
        request_context,
        {"error": {"type": "server_error", "message": "failed after partial"}},
        protocol_context=protocol_context,
    ))

    assert "response.created" not in terminal
    assert response_id in terminal
    assert "partial" in terminal
    assert "failed after partial" in terminal


class _NativeStreamTransport:
    def __init__(self, chunks: list[object]) -> None:
        self.chunks = chunks
        self.requests = []

    async def stream_json_lines(self, endpoint, *, headers, payload):
        self.requests.append(payload)
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_protocol", PROTOCOLS)
@pytest.mark.parametrize("output_protocol", PROTOCOLS)
async def test_native_executor_stream_matrix_uses_provider_reader_and_selected_writer(
    provider_protocol: str,
    output_protocol: str,
) -> None:
    operations = {
        "openai_chat": "chat",
        "anthropic_messages": "messages",
        "responses": "responses",
        "gemini": "generate",
    }
    context = NativeProviderContext(
        provider="provider",
        model="model-a",
        protocol_name=provider_protocol,
        input_protocol_name="openai_chat",
        output_protocol_name=output_protocol,
        endpoint="https://provider.test/stream",
        operation=operations[provider_protocol],
    )
    transport = _NativeStreamTransport(_source_frames(provider_protocol))

    frames = [
        frame
        async for frame in NativeProviderExecutor().stream(
            {"model": "model-a", "messages": [{"role": "user", "content": "hello"}], "stream": True},
            context,
            transport,
        )
    ]
    output = "".join(frames)

    assert "hi" in output
    assert "weather" in output
    assert transport.requests


class _RuntimeProvider(ProviderInterface):
    provider_env_name = "stream_runtime"

    async def get_models(self, api_key, client):
        return []


class _CredentialContext:
    def __init__(self, credential: str = "credential-1") -> None:
        self.credential = credential
        self.stable_id = credential

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def mark_success(self, **kwargs):
        self.success = kwargs

    def mark_failure(self, error):
        self.failure = error


class _UsageManager:
    def __init__(self, credentials: tuple[str, ...] = ("credential-1",)) -> None:
        self.initialized = False
        self.states = {
            credential: SimpleNamespace(
                tier="standard",
                priority=1,
                totals=SimpleNamespace(request_count=0),
                usage=SimpleNamespace(),
                model_usage={},
                group_usage={},
                get_usage_for_scope=lambda *args, **kwargs: None,
            )
            for credential in credentials
        }
        self.window_manager = SimpleNamespace(get_primary_definition=lambda: None)

    async def initialize(self, credentials, priorities=None, tiers=None):
        self.initialized = True

    async def acquire_credential(self, **kwargs):
        candidates = kwargs.get("candidates") or ["credential-1"]
        return _CredentialContext(candidates[0])

    async def get_availability_stats(self, model, quota_group=None):
        return {
            "total": 1,
            "available": 1,
            "blocked": 0,
            "blocked_by": {"cooldowns": 0, "window_limits": 0, "custom_caps": 0, "fair_cycle": 0, "concurrent": 0},
            "rotation_mode": "sequential",
        }

    def get_model_quota_group(self, model):
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("output_protocol", PROTOCOLS)
async def test_request_executor_formats_litellm_stream_in_selected_protocol(
    monkeypatch,
    output_protocol: str,
) -> None:
    usage_manager = _UsageManager()
    executor = RequestExecutor(
        usage_managers={"stream_runtime": usage_manager},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"stream_runtime": _RuntimeProvider}),
        provider_transforms=ProviderTransforms({"stream_runtime": _RuntimeProvider}, None),
        provider_plugins={"stream_runtime": _RuntimeProvider},
        http_client=httpx.AsyncClient(),
        max_retries=1,
        global_timeout=5,
        litellm_logger_fn=lambda payload: None,
    )

    async def upstream():
        yield {"id": "chat_runtime", "model": "model-a", "choices": [{"delta": {"content": "runtime"}}]}
        yield {"id": "chat_runtime", "model": "model-a", "choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    async def fake_acompletion(**kwargs):
        return upstream()

    monkeypatch.setattr("rotator_library.client.executor.litellm.acompletion", fake_acompletion)
    context = RequestContext(
        model="stream_runtime/model-a",
        provider="stream_runtime",
        kwargs={"model": "stream_runtime/model-a", "messages": [], "stream": True},
        streaming=True,
        credentials=["credential-1"],
        credential_secrets={"credential-1": "secret"},
        deadline=9999999999.0,
        input_protocol_name="gemini",
        output_protocol_name=output_protocol,
    )

    chunks = [chunk async for chunk in executor._execute_streaming(context)]
    output = "".join(chunks)

    assert "runtime" in output
    markers = {
        "openai_chat": "chat.completion.chunk",
        "anthropic_messages": "event: message_start",
        "responses": "event: response.created",
        "gemini": '"candidates"',
    }
    assert markers[output_protocol] in output


class _NativeErrorRuntimeProvider(_RuntimeProvider):
    provider_env_name = "native_stream_runtime"
    protocol_name = "openai_chat"
    native_streaming_supported = True

    def get_native_headers(self, credential_identifier, model="", operation="chat"):
        return {"Authorization": f"Bearer {credential_identifier}"}

    def get_native_endpoint(self, model="", operation="chat"):
        return "https://provider.test/chat"

    def normalize_native_model(self, model=""):
        return model.split("/", 1)[-1]


class _RotatingNativeStreamClient:
    def __init__(self) -> None:
        self.credentials = []

    async def stream_json_lines(self, endpoint, *, headers, json):
        credential = headers["Authorization"].removeprefix("Bearer ")
        self.credentials.append(credential)
        if credential == "secret-1":
            yield {"error": {"type": "rate_limit", "message": "rotate me", "status_code": 429}}
            return
        yield {"choices": [{"delta": {"content": "after rotation"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        yield "[DONE]"


@pytest.mark.asyncio
async def test_native_stream_error_rotates_credentials_before_selected_output(monkeypatch) -> None:
    monkeypatch.setenv("TRANSIENT_RETRY_DELAY", "0")
    monkeypatch.setenv("TRANSIENT_RETRY_JITTER", "0")
    credentials = ("credential-1", "credential-2")
    usage_manager = _UsageManager(credentials)
    http_client = _RotatingNativeStreamClient()
    executor = RequestExecutor(
        usage_managers={"native_stream_runtime": usage_manager},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"native_stream_runtime": _NativeErrorRuntimeProvider}),
        provider_transforms=ProviderTransforms({"native_stream_runtime": _NativeErrorRuntimeProvider}, None),
        provider_plugins={"native_stream_runtime": _NativeErrorRuntimeProvider},
        http_client=http_client,
        max_retries=1,
        global_timeout=5,
        litellm_logger_fn=lambda payload: None,
    )
    kwargs = {
        "model": "native_stream_runtime/model-a",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    context = RequestContext(
        model=kwargs["model"],
        provider="native_stream_runtime",
        kwargs=kwargs,
        streaming=True,
        credentials=list(credentials),
        credential_secrets={"credential-1": "secret-1", "credential-2": "secret-2"},
        deadline=9999999999.0,
        input_protocol_name="openai_chat",
        output_protocol_name="responses",
        protocol_request=dict(kwargs),
        unified_request=get_protocol("openai_chat").parse_request(kwargs),
        input_provider="native_stream_runtime",
    )

    output = "".join([chunk async for chunk in executor._execute_streaming(context)])

    assert http_client.credentials == ["secret-1", "secret-2"]
    assert "after rotation" in output
    assert "event: response.completed" in output
    assert "rotate me" not in output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("output_protocol", "start_marker"),
    (("anthropic_messages", "event: message_start"), ("responses", "event: response.created")),
)
async def test_litellm_in_band_error_rotates_without_duplicate_destination_start(
    monkeypatch,
    output_protocol: str,
    start_marker: str,
) -> None:
    monkeypatch.setenv("TRANSIENT_RETRY_DELAY", "0")
    monkeypatch.setenv("TRANSIENT_RETRY_JITTER", "0")
    credentials = ("credential-1", "credential-2")
    usage_manager = _UsageManager(credentials)
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs["api_key"])

        async def failed_stream():
            yield {"choices": [{"delta": {"role": "assistant"}}]}
            yield 'event: error\ndata: {"error":{"type":"rate_limit","message":"retry this credential"}}\n\n'

        async def successful_stream():
            yield {"choices": [{"delta": {"content": "one lifecycle"}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        return failed_stream() if kwargs["api_key"] == "secret-1" else successful_stream()

    monkeypatch.setattr("rotator_library.client.executor.litellm.acompletion", fake_acompletion)
    executor = RequestExecutor(
        usage_managers={"stream_runtime": usage_manager},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"stream_runtime": _RuntimeProvider}),
        provider_transforms=ProviderTransforms({"stream_runtime": _RuntimeProvider}, None),
        provider_plugins={"stream_runtime": _RuntimeProvider},
        http_client=httpx.AsyncClient(),
        max_retries=1,
        global_timeout=5,
        litellm_logger_fn=lambda payload: None,
    )
    context = RequestContext(
        model="stream_runtime/model-a",
        provider="stream_runtime",
        kwargs={"model": "stream_runtime/model-a", "messages": [], "stream": True},
        streaming=True,
        credentials=list(credentials),
        credential_secrets={"credential-1": "secret-1", "credential-2": "secret-2"},
        deadline=9999999999.0,
        input_protocol_name="openai_chat",
        output_protocol_name=output_protocol,
    )

    output = "".join([chunk async for chunk in executor._execute_streaming(context)])

    assert calls == ["secret-1", "secret-2"]
    assert output.count(start_marker) == 1
    assert "one lifecycle" in output
    assert "retry this credential" not in output


class _CustomErrorRuntimeProvider(_RuntimeProvider):
    provider_env_name = "custom_stream_runtime"
    credentials_seen: list[str] = []

    def has_custom_logic(self):
        return True

    async def acompletion(self, client, **kwargs):
        credential = kwargs["credential_identifier"]
        type(self).credentials_seen.append(credential)

        async def stream():
            if credential == "secret-1":
                yield {"error": {"type": "rate_limit", "message": "custom retry"}}
                return
            yield {"choices": [{"delta": {"content": "custom success"}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        return stream()


@pytest.mark.asyncio
async def test_custom_in_band_mapping_error_rotates_before_output(monkeypatch) -> None:
    monkeypatch.setenv("TRANSIENT_RETRY_DELAY", "0")
    monkeypatch.setenv("TRANSIENT_RETRY_JITTER", "0")
    _CustomErrorRuntimeProvider.credentials_seen = []
    credentials = ("credential-1", "credential-2")
    usage_manager = _UsageManager(credentials)
    executor = RequestExecutor(
        usage_managers={"custom_stream_runtime": usage_manager},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"custom_stream_runtime": _CustomErrorRuntimeProvider}),
        provider_transforms=ProviderTransforms({"custom_stream_runtime": _CustomErrorRuntimeProvider}, None),
        provider_plugins={"custom_stream_runtime": _CustomErrorRuntimeProvider},
        http_client=httpx.AsyncClient(),
        max_retries=1,
        global_timeout=5,
        litellm_logger_fn=lambda payload: None,
    )
    context = RequestContext(
        model="custom_stream_runtime/model-a",
        provider="custom_stream_runtime",
        kwargs={"model": "custom_stream_runtime/model-a", "messages": [], "stream": True},
        streaming=True,
        credentials=list(credentials),
        credential_secrets={"credential-1": "secret-1", "credential-2": "secret-2"},
        deadline=9999999999.0,
        input_protocol_name="openai_chat",
        output_protocol_name="gemini",
    )

    output = "".join([chunk async for chunk in executor._execute_streaming(context)])

    assert _CustomErrorRuntimeProvider.credentials_seen == ["secret-1", "secret-2"]
    assert "custom success" in output
    assert "custom retry" not in output


@pytest.mark.asyncio
async def test_in_band_error_after_visible_output_closes_same_responses_lifecycle(monkeypatch) -> None:
    credentials = ("credential-1", "credential-2")
    usage_manager = _UsageManager(credentials)
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs["api_key"])

        async def stream():
            yield {"choices": [{"delta": {"content": "already visible"}}]}
            yield 'data: {"error":{"type":"rate_limit","message":"failed after visible"}}\n\n'

        return stream()

    monkeypatch.setattr("rotator_library.client.executor.litellm.acompletion", fake_acompletion)
    executor = RequestExecutor(
        usage_managers={"stream_runtime": usage_manager},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"stream_runtime": _RuntimeProvider}),
        provider_transforms=ProviderTransforms({"stream_runtime": _RuntimeProvider}, None),
        provider_plugins={"stream_runtime": _RuntimeProvider},
        http_client=httpx.AsyncClient(),
        max_retries=1,
        global_timeout=5,
        litellm_logger_fn=lambda payload: None,
    )
    context = RequestContext(
        model="stream_runtime/model-a",
        provider="stream_runtime",
        kwargs={"model": "stream_runtime/model-a", "messages": [], "stream": True},
        streaming=True,
        credentials=list(credentials),
        credential_secrets={"credential-1": "secret-1", "credential-2": "secret-2"},
        deadline=9999999999.0,
        input_protocol_name="openai_chat",
        output_protocol_name="responses",
    )

    output = "".join([chunk async for chunk in executor._execute_streaming(context)])

    assert calls == ["secret-1"]
    assert output.count("event: response.created") == 1
    assert "already visible" in output
    assert "event: response.failed" in output
    assert "failed after visible" in output
