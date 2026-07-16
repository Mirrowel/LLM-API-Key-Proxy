from __future__ import annotations

import json
from copy import deepcopy
from types import MethodType, SimpleNamespace
from typing import Any

import pytest

from rotator_library.native_provider import NativeProviderContext, NativeProviderExecutor
from rotator_library.client.executor import RequestExecutor
from rotator_library.client.request_builder import RequestContextBuilder
from rotator_library.client.rotating_client import RotatingClient
from rotator_library.core.errors import StructuredAPIResponseError, is_structured_error_payload
from rotator_library.core.types import ErrorAction, RequestContext
from rotator_library.providers.codex_provider import CodexProvider
from rotator_library.providers.antigravity_provider import AntigravityProvider
from rotator_library.protocols import ProtocolContext, get_protocol
from rotator_library.protocols.types import first_text
from rotator_library.routing import parse_route_target


PROTOCOLS = ("openai_chat", "anthropic_messages", "responses", "gemini")

REQUESTS: dict[str, dict[str, Any]] = {
    "openai_chat": {
        "model": "provider/model-test",
        "messages": [
            {"role": "system", "content": "follow the rule"},
            {"role": "user", "content": "hello"},
        ],
    },
    "anthropic_messages": {
        "model": "provider/model-test",
        "system": "follow the rule",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 256,
    },
    "responses": {
        "model": "provider/model-test",
        "instructions": "follow the rule",
        "input": "hello",
    },
    "gemini": {
        "model": "provider/model-test",
        "systemInstruction": {"parts": [{"text": "follow the rule"}]},
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
    },
}


def test_success_dictionaries_are_not_misclassified_as_api_errors() -> None:
    assert is_structured_error_payload(RESPONSES["openai_chat"]) is False
    assert is_structured_error_payload({"error": {"message": "failed"}}) is True

RESPONSES: dict[str, dict[str, Any]] = {
    "openai_chat": {
        "id": "chat_1",
        "model": "model-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    },
    "anthropic_messages": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "model-test",
        "content": [{"type": "text", "text": "answer"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "output_tokens": 1},
    },
    "responses": {
        "id": "resp_1",
        "object": "response",
        "model": "model-test",
        "status": "completed",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "answer"}],
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    },
    "gemini": {
        "responseId": "gemini_1",
        "modelVersion": "model-test",
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "answer"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
    },
}

OPERATIONS = {
    "openai_chat": "chat",
    "anthropic_messages": "messages",
    "responses": "responses",
    "gemini": "generate",
}


class RecordingTransport:
    """Return one provider response while retaining the native request."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = deepcopy(response)
        self.payload: dict[str, Any] | None = None

    async def post_json(self, endpoint: str, *, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        self.payload = deepcopy(payload)
        return deepcopy(self.response)


class RecordingHTTPClient:
    """Expose the HTTPX-style post seam used by RequestExecutor."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def post(self, endpoint: str, *, headers: dict[str, str], json: dict[str, Any]):
        self.calls.append({"endpoint": endpoint, "headers": headers, "json": deepcopy(json)})
        response = deepcopy(self.response)

        class Result:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, Any]:
                return response

        return Result()


def _assert_provider_shape(protocol_name: str, payload: dict[str, Any]) -> None:
    """Assert that a provider hook receives only its declared wire format."""

    if protocol_name == "openai_chat":
        assert "messages" in payload and "contents" not in payload and "input" not in payload
    elif protocol_name == "anthropic_messages":
        assert "messages" in payload and "system" in payload and "contents" not in payload
        payload.setdefault("max_tokens", 256)
    elif protocol_name == "responses":
        assert "input" in payload and "messages" not in payload and "contents" not in payload
    else:
        assert "contents" in payload and "messages" not in payload and "input" not in payload


def _output_text(protocol_name: str, payload: dict[str, Any]) -> str | None:
    """Parse a client payload again to prove output-language validity."""

    response = get_protocol(protocol_name).parse_response(
        payload,
        ProtocolContext(source_protocol=protocol_name, target_protocol=protocol_name),
    )
    for message in response.messages:
        text = first_text(message.content)
        if text:
            return text
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize("input_protocol", PROTOCOLS)
@pytest.mark.parametrize("provider_protocol", PROTOCOLS)
@pytest.mark.parametrize("output_protocol", PROTOCOLS)
async def test_runtime_keeps_input_provider_and_output_protocols_independent(
    input_protocol: str,
    provider_protocol: str,
    output_protocol: str,
) -> None:
    preparer_calls: list[dict[str, Any]] = []

    def prepare(payload: dict[str, Any], *, model: str, operation: str) -> dict[str, Any]:
        _assert_provider_shape(provider_protocol, payload)
        preparer_calls.append(deepcopy(payload))
        return payload

    context = NativeProviderContext(
        provider="provider",
        model="model-test",
        protocol_name=provider_protocol,
        endpoint="https://provider.test/generate",
        operation=OPERATIONS[provider_protocol],
        input_protocol_name=input_protocol,
        output_protocol_name=output_protocol,
        credential_id="credential-1",
        session_id="session-1",
        metadata={"public_model": "provider/model-test", "input_provider": "provider"},
        request_preparer=prepare,
    )
    transport = RecordingTransport(RESPONSES[provider_protocol])

    result = await NativeProviderExecutor().execute(
        deepcopy(REQUESTS[input_protocol]),
        context,
        transport,
    )

    assert len(preparer_calls) == 1
    prepared_payload = deepcopy(preparer_calls[0])
    assert prepared_payload.pop("_proxy_model") == "provider/model-test"
    assert "_proxy_model" not in transport.payload
    assert transport.payload == prepared_payload
    assert _output_text(output_protocol, result) == "answer"


@pytest.mark.parametrize("output_protocol", PROTOCOLS)
def test_litellm_and_custom_chat_results_use_the_selected_output_protocol(output_protocol: str) -> None:
    context = RequestContext(
        model="provider/model-test",
        provider="provider",
        kwargs=deepcopy(REQUESTS["openai_chat"]),
        streaming=False,
        credentials=["credential-1"],
        deadline=9999999999.0,
        input_protocol_name="gemini",
        output_protocol_name=output_protocol,
    )

    result = RequestExecutor._format_execution_response(
        deepcopy(RESPONSES["openai_chat"]),
        "openai_chat",
        context,
    )

    assert _output_text(output_protocol, result) == "answer"


@pytest.mark.asyncio
async def test_structured_errors_raise_before_cross_protocol_success_formatting() -> None:
    class ErrorPlugin:
        def has_custom_logic(self) -> bool:
            return True

        async def acompletion(self, client, **kwargs):
            return {"error": {"type": "rate_limit_error", "message": "slow down", "status_code": 429}}

    target = parse_route_target("provider/model-test@custom")
    context = RequestContext(
        model="provider/model-test",
        provider="provider",
        kwargs=deepcopy(REQUESTS["openai_chat"]),
        streaming=False,
        credentials=["credential-1"],
        deadline=9999999999.0,
        output_protocol_name="anthropic_messages",
        routing_targets=(target,),
    )
    context.routing_target_index = 0
    executor = RequestExecutor.__new__(RequestExecutor)
    executor._http_client = object()

    with pytest.raises(StructuredAPIResponseError) as raised:
        await executor._execute_provider_request(
            "provider",
            context.model,
            ErrorPlugin(),
            "secret",
            "credential-1",
            deepcopy(context.kwargs),
            context,
        )

    assert raised.value.error_type == "rate_limit"


@pytest.mark.asyncio
async def test_structured_error_survives_real_credential_exhaustion_loop() -> None:
    class CredentialContext:
        def __init__(self, credential: str) -> None:
            self.credential = credential
            self.stable_id = f"stable-{credential}"

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class UsageManager:
        states = {}

        async def get_availability_stats(self, model, quota_group=None):
            return {"available": 2, "total": 2, "blocked": 0, "blocked_by": {}, "rotation_mode": "sequential"}

        async def acquire_credential(self, *, candidates, **kwargs):
            return CredentialContext(candidates[0])

    class ErrorPlugin:
        def has_custom_logic(self):
            return True

        async def acompletion(self, client, **kwargs):
            return {"error": {"status": 429, "message": "busy"}}

    executor = RequestExecutor.__new__(RequestExecutor)
    executor._max_retries = 1
    executor._http_client = object()
    executor._cooldown = None
    manager = UsageManager()

    async def prepare_execution(self, context):
        filter_result = SimpleNamespace(priorities={}, tier_names={})
        return manager, filter_result, list(context.credentials), None, {}

    async def prepare_kwargs(self, provider, model, credential, context, **kwargs):
        return deepcopy(context.kwargs)

    async def handle_error(
        self,
        error,
        credential_context,
        model,
        provider,
        attempt,
        accumulator,
        retry_state,
        request_headers,
        context,
    ):
        accumulator.normal_errors.append({"error_type": error.error_type})
        return ErrorAction.ROTATE

    executor._prepare_execution = MethodType(prepare_execution, executor)
    executor._prepare_request_kwargs = MethodType(prepare_kwargs, executor)
    executor._handle_error_with_context = MethodType(handle_error, executor)
    executor._get_plugin_instance = MethodType(lambda self, provider: ErrorPlugin(), executor)
    context = RequestContext(
        model="provider/model-test",
        provider="provider",
        kwargs=deepcopy(REQUESTS["openai_chat"]),
        streaming=False,
        credentials=["credential-1", "credential-2"],
        deadline=9999999999.0,
    )

    with pytest.raises(StructuredAPIResponseError) as raised:
        await executor._execute_non_streaming(context)

    assert raised.value.error_type == "rate_limit"


@pytest.mark.asyncio
async def test_native_structured_errors_raise_before_provider_response_parsing() -> None:
    context = NativeProviderContext(
        provider="provider",
        model="model-test",
        protocol_name="gemini",
        endpoint="https://provider.test/generate",
        operation="generate",
        input_protocol_name="openai_chat",
        output_protocol_name="responses",
    )
    transport = RecordingTransport(
        {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota exhausted"}}
    )

    with pytest.raises(StructuredAPIResponseError) as raised:
        await NativeProviderExecutor().execute(
            deepcopy(REQUESTS["openai_chat"]),
            context,
            transport,
        )

    assert raised.value.error_type == "quota_exceeded"


@pytest.mark.parametrize(
    ("error_payload", "expected_type", "expected_status"),
    [
        ({"error": "upstream failed"}, "invalid_request", 400),
        ({"error": {"status": 429, "message": "busy"}}, "rate_limit", 429),
        ({"error": {"code": 403, "status": "PERMISSION_DENIED"}}, "forbidden", 403),
    ],
)
@pytest.mark.asyncio
async def test_structured_error_variants_are_normalized_before_success_parsing(
    error_payload: dict[str, Any],
    expected_type: str,
    expected_status: int,
) -> None:
    context = NativeProviderContext(
        provider="provider",
        model="model-test",
        protocol_name="gemini",
        endpoint="https://provider.test/generate",
        operation="generate",
    )

    with pytest.raises(StructuredAPIResponseError) as raised:
        await NativeProviderExecutor().execute({}, context, RecordingTransport(error_payload))

    assert raised.value.error_type == expected_type
    assert raised.value.http_status == expected_status


@pytest.mark.asyncio
async def test_response_adapters_run_on_selected_client_protocol_payload() -> None:
    context = NativeProviderContext(
        provider="provider",
        model="model-test",
        protocol_name="anthropic_messages",
        endpoint="https://provider.test/messages",
        operation="messages",
        input_protocol_name="openai_chat",
        output_protocol_name="openai_chat",
        adapter_names=("reasoning_content",),
        adapter_config={
            "reasoning_content": {
                "source_fields": ["reasoning_content"],
                "output_field": "analysis",
            }
        },
    )
    response = deepcopy(RESPONSES["anthropic_messages"])
    response["content"].insert(0, {"type": "thinking", "thinking": "provider reasoning"})

    result = await NativeProviderExecutor().execute(
        deepcopy(REQUESTS["openai_chat"]),
        context,
        RecordingTransport(response),
    )

    assert result["choices"][0]["message"]["analysis"] == "provider reasoning"


@pytest.mark.asyncio
async def test_opaque_thought_signatures_are_cached_but_never_returned_to_clients() -> None:
    provider = AntigravityProvider()
    rules = provider.get_field_cache_rules("antigravity/gemini-3-pro")
    adapter_names = provider.get_adapter_names("antigravity/gemini-3-pro")
    adapter_config = provider.get_adapter_config("antigravity/gemini-3-pro")
    context = NativeProviderContext(
        provider="antigravity",
        model="gemini-3-pro",
        protocol_name="gemini",
        endpoint="https://provider.test/generate",
        operation="generate",
        input_protocol_name="gemini",
        output_protocol_name="gemini",
        credential_id="credential-1",
        session_id="session-1",
        field_cache_rules=rules,
        adapter_names=adapter_names,
        adapter_config=adapter_config,
        metadata={"public_model": "antigravity/gemini-3-pro", "input_provider": "antigravity"},
    )
    signed_response = deepcopy(RESPONSES["gemini"])
    signed_response["candidates"][0]["content"]["parts"].insert(
        0,
        {"text": "hidden reasoning", "thought": True, "thoughtSignature": "provider-secret-signature"},
    )
    executor = NativeProviderExecutor()

    first_result = await executor.execute(
        deepcopy(REQUESTS["gemini"]),
        context,
        RecordingTransport(signed_response),
    )

    assert "provider-secret-signature" not in json.dumps(first_result)

    second_transport = RecordingTransport(deepcopy(RESPONSES["gemini"]))
    await executor.execute(deepcopy(REQUESTS["gemini"]), context, second_transport)

    assert second_transport.payload["request"]["metadata"]["thoughtSignatures"] == [
        "provider-secret-signature"
    ]


@pytest.mark.asyncio
async def test_real_callback_nested_edits_preserve_source_native_metadata() -> None:
    source_payload = deepcopy(REQUESTS["anthropic_messages"])
    source_payload["system"] = [
        {
            "type": "text",
            "text": "follow the rule",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    source_payload["tools"] = [
        {
            "name": "lookup",
            "description": "original description",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]
    anthropic = get_protocol("anthropic_messages")
    chat = get_protocol("openai_chat")
    unified = anthropic.parse_request(source_payload)
    chat_view = chat.build_request(
        unified,
        ProtocolContext(source_protocol="anthropic_messages", target_protocol="openai_chat"),
    )
    context = RequestContext(
        model="provider/model-test",
        provider="provider",
        kwargs=deepcopy(chat_view),
        streaming=False,
        credentials=["credential-1"],
        deadline=9999999999.0,
        input_protocol_name="anthropic_messages",
        output_protocol_name="anthropic_messages",
        protocol_request=source_payload,
        unified_request=unified,
        input_provider="provider",
    )
    async def callback(request, kwargs):
        kwargs["messages"][-1]["content"] = "changed by callback"
        kwargs["tools"][0]["function"]["description"] = "changed tool description"

    context.pre_request_callback = callback
    executor = RequestExecutor.__new__(RequestExecutor)
    attempted = await executor._prepare_request_kwargs(
        "provider",
        context.model,
        "credential-1",
        context,
        native_execution=True,
    )
    await executor._run_pre_request_callback(context, attempted)
    assert context.kwargs["messages"][-1]["content"] == "hello"
    assert context.kwargs["tools"][0]["function"]["description"] == "original description"

    merged = RequestExecutor._canonical_request_for_native(context, attempted)
    provider_payload = anthropic.build_request(
        merged,
        ProtocolContext(
            source_protocol="anthropic_messages",
            target_protocol="anthropic_messages",
            source_provider="provider",
            target_provider="provider",
            provider_state_compatible=True,
        ),
    )

    assert provider_payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert provider_payload["messages"][-1]["content"][0]["text"] == "changed by callback"
    assert provider_payload["tools"][0]["description"] == "changed tool description"


def test_internal_attempt_fields_do_not_become_same_protocol_extensions() -> None:
    source_payload = deepcopy(REQUESTS["openai_chat"])
    source_payload["vendor_setting"] = {"enabled": True}
    chat = get_protocol("openai_chat")
    unified = chat.parse_request(source_payload)
    context = RequestContext(
        model="provider/model-test",
        provider="provider",
        kwargs=deepcopy(source_payload),
        streaming=False,
        credentials=["credential-1"],
        deadline=9999999999.0,
        input_protocol_name="openai_chat",
        output_protocol_name="openai_chat",
        protocol_request=source_payload,
        unified_request=unified,
        input_provider="provider",
    )
    attempted = deepcopy(source_payload)
    attempted["transaction_context"] = {"request_id": "internal"}
    attempted["litellm_params"] = {"api_base": "internal"}

    merged = RequestExecutor._canonical_request_for_native(context, attempted)
    provider_payload = chat.build_request(
        merged,
        ProtocolContext(
            source_protocol="openai_chat",
            target_protocol="openai_chat",
            source_provider="provider",
            target_provider="provider",
        ),
    )

    assert provider_payload["vendor_setting"] == {"enabled": True}
    assert "transaction_context" not in provider_payload
    assert "litellm_params" not in provider_payload


@pytest.mark.asyncio
async def test_proxy_expanded_responses_history_suppresses_cached_provider_continuation() -> None:
    rules = CodexProvider().get_field_cache_rules("codex/gpt-5.1-codex")
    executor = NativeProviderExecutor()
    base_context = NativeProviderContext(
        provider="codex",
        model="gpt-5.1-codex",
        protocol_name="responses",
        endpoint="https://provider.test/responses",
        operation="responses",
        input_protocol_name="responses",
        output_protocol_name="responses",
        credential_id="credential-1",
        session_id="session-1",
        field_cache_rules=rules,
        metadata={"public_model": "codex/gpt-5.1-codex", "input_provider": "codex"},
    )
    first_transport = RecordingTransport(deepcopy(RESPONSES["responses"]))
    await executor.execute(deepcopy(REQUESTS["responses"]), base_context, first_transport)

    expanded_request = deepcopy(REQUESTS["responses"])
    expanded_request["input"] = [
        "earlier input",
        RESPONSES["responses"]["output"][0],
        "current input",
    ]
    second_context = NativeProviderContext(
        **{
            **base_context.__dict__,
            "metadata": {**base_context.metadata, "disable_provider_continuation": True},
        }
    )
    second_transport = RecordingTransport({**deepcopy(RESPONSES["responses"]), "id": "resp_2"})
    await executor.execute(expanded_request, second_context, second_transport)

    assert "previous_response_id" not in second_transport.payload
    assert len(second_transport.payload["input"]) == 3
    assert "earlier input" in str(second_transport.payload["input"])
    assert "current input" in str(second_transport.payload["input"])
    assert "answer" in str(second_transport.payload["input"])


@pytest.mark.asyncio
async def test_agenerate_runs_real_builder_to_native_executor_handoff() -> None:
    validated_provider_requests: list[dict[str, Any]] = []

    class Resolver:
        def resolve_model_id(self, model: str, provider: str) -> str:
            return model

    class Session:
        session_id = "session-1"
        affinity_key = "affinity-1"
        tracking_namespace = "namespace"

    class SessionTracker:
        def infer_session(self, *args, **kwargs):
            return Session()

    async def scope(provider, classifier, request_api_keys, request_providers, private):
        return {
            "credentials": ["credential-1"],
            "usage_manager_key": provider,
            "provider_config": {},
            "credential_secrets": {"credential-1": "secret"},
            "classifier": classifier or "global",
        }

    class GeminiProvider:
        def has_custom_logic(self) -> bool:
            return False

        def get_protocol_name(self, model="") -> str:
            return "gemini"

        def get_native_operation(self, model="", request=None, stream=False) -> str:
            return "generate"

        def get_native_endpoint(self, model="", operation="generate") -> str:
            return "https://provider.test/generate"

        def get_native_headers(self, credential_identifier, model="", operation="generate"):
            return {"Authorization": f"Bearer {credential_identifier}"}

        def normalize_native_model(self, model="") -> str:
            return model.split("/", 1)[-1]

        def get_adapter_names(self, model=""):
            return ()

        def get_adapter_config(self, model=""):
            return {}

        def get_field_cache_rules(self, model=""):
            return ()

        def validate_request(self, request, model=""):
            validated_provider_requests.append(deepcopy(request))
            return True

    builder = RequestContextBuilder(
        resolve_scope_for_provider=scope,
        model_resolver=Resolver(),
        session_tracker=SessionTracker(),
        get_global_timeout=lambda: 30,
        get_enable_request_logging=lambda: False,
    )
    http_client = RecordingHTTPClient(RESPONSES["gemini"])
    real_executor = RequestExecutor.__new__(RequestExecutor)
    real_executor._http_client = http_client

    class HandoffExecutor:
        async def execute(self, context: RequestContext):
            target = parse_route_target("provider/model-test@native")
            context.routing_targets = (target,)
            context.routing_target_index = 0
            attempt_kwargs = deepcopy(context.kwargs)
            attempt_kwargs["messages"][-1]["content"] = "changed by callback"
            attempt_kwargs["transaction_context"] = {"request_id": "internal"}
            attempt_kwargs["litellm_params"] = {"api_base": "internal"}
            return await real_executor._execute_provider_request(
                "provider",
                context.model,
                GeminiProvider(),
                "secret",
                "credential-1",
                attempt_kwargs,
                context,
            )

    client = RotatingClient.__new__(RotatingClient)
    client._request_builder = builder
    client._executor = HandoffExecutor()

    result = await client.agenerate(
        deepcopy(REQUESTS["anthropic_messages"]),
        input_protocol="anthropic_messages",
        output_protocol="responses",
    )

    assert "contents" in http_client.calls[0]["json"]
    assert "messages" not in http_client.calls[0]["json"]
    assert "transaction_context" not in http_client.calls[0]["json"]
    assert "litellm_params" not in http_client.calls[0]["json"]
    assert "contents" in validated_provider_requests[0]
    assert "messages" not in validated_provider_requests[0]
    assert http_client.calls[0]["json"]["systemInstruction"]["parts"][0]["text"] == "follow the rule"
    assert http_client.calls[0]["json"]["contents"][0]["parts"][0]["text"] == "changed by callback"
    assert result["object"] == "response"
    assert _output_text("responses", result) == "answer"
