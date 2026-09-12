"""G3 regression pins: foreign opaque-state stripping on the raw path.

Slice B of G3 freezes the Phase-1 behavior:

1. the per-protocol strip matrix (exact field descriptors, detect-only,
   unknown/non-dict safety, portable chat reasoning never stripped);
2. the basis-seam provider-switch strip (wire has no foreign signatures,
   the strip is disclosed as an overlay, same-provider stays byte-verbatim);
3. the failover no-leak guarantee (cache key provider floor);
4. the reactive strip-and-retry (one bounded retry, same credential);
5. the reserved-metadata-key guard;
6. the build_cache_key provider+model floor for narrow scopes;
7. the Gemini skip-signature sentinel's exact wire shape;
8. the D4 raw/rebuild gate's operation/logical_operation/input divergence.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from rotator_library.client.executor import RequestExecutor
from rotator_library.client.filters import CredentialFilter
from rotator_library.client.transforms import ProviderTransforms
from rotator_library.core.types import RequestContext
from rotator_library.field_cache import (
    FieldCacheEngine,
    FieldCacheInjection,
    FieldCacheRule,
    InMemoryFieldCacheStore,
    build_cache_key,
)
from rotator_library.field_cache.types import FieldCacheContext
from rotator_library.native_provider import (
    NativeHTTPTransport,
    NativeProviderContext,
    NativeProviderExecutor,
)
from rotator_library.native_provider.executor import _wire_view_matches_unified
from rotator_library.protocols import ProtocolContext, get_protocol
from rotator_library.protocols.opaque_strip import (
    payload_carries_opaque_state,
    strip_foreign_opaque_state,
)
from rotator_library.protocols.streaming import _gemini_skip_signature_sentinel
from rotator_library.providers.provider_interface import ProviderInterface


# --------------------------------------------------------------------------
# fakes (mirroring the native_provider / g2 executor test patterns)
# --------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, body: dict, status: int = 200):
        self._body = body
        self.status_code = status

    def json(self):
        return self._body


class _FakeClient:
    def __init__(self, response_body: dict):
        self._response_body = response_body
        self.calls: list[dict] = []

    async def post(self, endpoint, headers=None, json=None, **kwargs):
        self.calls.append({"endpoint": endpoint, "headers": dict(headers or {}), "json": json})
        return _FakeResponse(self._response_body)


def _chat_response(reasoning: str | None = None) -> dict:
    message: dict = {"role": "assistant", "content": "hi"}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-test",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _anthropic_response() -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }


# --------------------------------------------------------------------------
# 1. strip module unit matrix
# --------------------------------------------------------------------------


def test_anthropic_strip_matrix_exact_descriptors_and_detect_only() -> None:
    payload = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "t", "signature": "sig"},
                    {"type": "text", "text": "visible"},
                    {"type": "redacted_thinking", "data": "x"},
                ],
            },
        ]
    }
    detected = strip_foreign_opaque_state(payload, "anthropic_messages", mutate=False)
    assert detected == ["messages[1].thinking", "messages[1].redacted_thinking"]
    # detect-only leaves the payload byte-identical
    assert [block["type"] for block in payload["messages"][1]["content"]] == [
        "thinking",
        "text",
        "redacted_thinking",
    ]
    assert payload_carries_opaque_state(payload, "anthropic_messages") is True

    stripped = strip_foreign_opaque_state(payload, "anthropic_messages")
    assert stripped == ["messages[1].thinking", "messages[1].redacted_thinking"]
    assert payload["messages"][1]["content"] == [{"type": "text", "text": "visible"}]


def test_gemini_strip_matrix_any_part_signature_keys() -> None:
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "a", "thoughtSignature": "sig-a"},
                    {"functionCall": {"name": "f", "args": {}}, "thought_signature": "sig-b"},
                ],
            }
        ]
    }
    detected = strip_foreign_opaque_state(payload, "gemini", mutate=False)
    assert detected == [
        "contents[0].parts[0].thoughtSignature",
        "contents[0].parts[1].thought_signature",
    ]
    assert payload["contents"][0]["parts"][0]["thoughtSignature"] == "sig-a"

    assert strip_foreign_opaque_state(payload, "gemini") == [
        "contents[0].parts[0].thoughtSignature",
        "contents[0].parts[1].thought_signature",
    ]
    assert payload["contents"][0]["parts"] == [
        {"text": "a"},
        {"functionCall": {"name": "f", "args": {}}},
    ]


def test_chat_strip_matrix_vendor_keys_and_portable_reasoning_survives() -> None:
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": "a",
                "extra_content": {"google": {"thought_signature": "sig-a"}},
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "extra_content": {"google": {"thought_signature": "sig-b"}},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "portable",
                "reasoning_content": "portable-reasoning",
            },
        ]
    }
    expected = [
        "messages[0].extra_content.google.thought_signature",
        "messages[1].tool_calls[call_1].extra_content.google.thought_signature",
    ]
    assert strip_foreign_opaque_state(payload, "openai_chat", mutate=False) == expected
    assert strip_foreign_opaque_state(payload, "openai_chat") == expected
    assert "extra_content" not in payload["messages"][0]
    assert "extra_content" not in payload["messages"][1]["tool_calls"][0]
    # plaintext reasoning is PORTABLE: never a strip carrier.
    assert payload["messages"][2]["reasoning_content"] == "portable-reasoning"


def test_chat_reasoning_content_alone_is_not_opaque() -> None:
    chat = {
        "messages": [
            {"role": "assistant", "content": "x", "reasoning_content": "portable-reasoning"}
        ]
    }
    assert strip_foreign_opaque_state(chat, "openai_chat") is None
    assert payload_carries_opaque_state(chat, "openai_chat") is False
    assert chat["messages"][0]["reasoning_content"] == "portable-reasoning"


def test_responses_strip_matrix_encrypted_content_only() -> None:
    payload = {
        "input": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "s"}],
                "encrypted_content": "enc",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            },
        ]
    }
    assert strip_foreign_opaque_state(payload, "responses", mutate=False) == [
        "input[0].reasoning.encrypted_content"
    ]
    assert strip_foreign_opaque_state(payload, "responses") == [
        "input[0].reasoning.encrypted_content"
    ]
    assert "encrypted_content" not in payload["input"][0]
    assert payload["input"][0]["summary"] == [{"type": "summary_text", "text": "s"}]


def test_strip_matrix_non_dict_and_unknown_protocol_are_none() -> None:
    for payload in ([], "x", None, 7):
        assert strip_foreign_opaque_state(payload, "anthropic_messages") is None
        assert payload_carries_opaque_state(payload, "gemini") is False
    unknown = {"messages": [{"content": [{"type": "thinking"}]}]}
    assert strip_foreign_opaque_state(unknown, "not_a_protocol") is None
    assert payload_carries_opaque_state(unknown, "not_a_protocol") is False


# --------------------------------------------------------------------------
# 2. basis-seam provider-switch strip
# --------------------------------------------------------------------------

_ANTHROPIC_THINKING_REQUEST = {
    "model": "claude-test",
    "max_tokens": 256,
    "messages": [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "provider-a reasoning", "signature": "sig-a"},
                {"type": "text", "text": "answer"},
            ],
        },
        {"role": "user", "content": "more"},
    ],
}


def _anthropic_context(input_provider: str) -> NativeProviderContext:
    return NativeProviderContext(
        provider="provider_b",
        model="claude-test",
        protocol_name="anthropic_messages",
        input_protocol_name="anthropic_messages",
        client_protocol_name="anthropic_messages",
        operation="messages",
        endpoint="https://provider.example/v1/messages",
        headers={"x-api-key": "k"},
        raw_client_request=deepcopy(_ANTHROPIC_THINKING_REQUEST),
        metadata={"input_provider": input_provider},
    )


@pytest.mark.asyncio
async def test_basis_seam_strips_foreign_thinking_and_discloses_overlay() -> None:
    context = _anthropic_context(input_provider="provider_a")
    client = _FakeClient(_anthropic_response())

    await NativeProviderExecutor().execute(
        deepcopy(_ANTHROPIC_THINKING_REQUEST),
        context,
        NativeHTTPTransport(client),
    )

    wire = client.calls[0]["json"]
    blocks = wire["messages"][1]["content"]
    assert [block["type"] for block in blocks] == ["text"]
    assert "signature" not in json.dumps(wire)
    overlays = context.request_transport_overlays or []
    stripped = [o for o in overlays if o.get("kind") == "foreign_bound_state_stripped"]
    assert len(stripped) == 1
    assert stripped[0]["from_provider"] == "provider_a"
    assert stripped[0]["fields"] == ["messages[1].thinking"]


@pytest.mark.asyncio
async def test_basis_seam_same_provider_ships_blocks_verbatim() -> None:
    context = _anthropic_context(input_provider="provider_b")
    client = _FakeClient(_anthropic_response())

    await NativeProviderExecutor().execute(
        deepcopy(_ANTHROPIC_THINKING_REQUEST),
        context,
        NativeHTTPTransport(client),
    )

    wire = client.calls[0]["json"]
    blocks = wire["messages"][1]["content"]
    assert [block["type"] for block in blocks] == ["thinking", "text"]
    assert blocks[0]["signature"] == "sig-a"
    assert blocks[0]["thinking"] == "provider-a reasoning"
    overlays = context.request_transport_overlays or []
    assert not any(o.get("kind") == "foreign_bound_state_stripped" for o in overlays)


# --------------------------------------------------------------------------
# 3. failover no-leak (cache keying pin)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failover_never_injects_other_providers_cached_state() -> None:
    store = InMemoryFieldCacheStore()
    rule = FieldCacheRule(
        name="bound_state",
        source="response",
        path="choices.0.message.reasoning_content",
        inject=FieldCacheInjection(target="request", path="metadata.bound_state"),
        scope=("provider", "model"),
    )
    executor = NativeProviderExecutor(field_cache_store=store)

    def context_for(provider: str) -> NativeProviderContext:
        return NativeProviderContext(
            provider=provider,
            model="gpt-test",
            protocol_name="openai_chat",
            endpoint="https://provider.example/v1/chat/completions",
            field_cache_rules=(rule,),
        )

    provider_a = context_for("provider_a")
    await executor.execute(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "seed"}]},
        provider_a,
        NativeHTTPTransport(_FakeClient(_chat_response(reasoning="A-bound-secret"))),
    )
    seeded_key = build_cache_key(rule, provider_a.field_cache_context())
    assert await store.get(seeded_key) is not None

    provider_b = context_for("provider_b")
    client_b = _FakeClient(_chat_response())
    await executor.execute(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "other"}]},
        provider_b,
        NativeHTTPTransport(client_b),
    )
    wire = client_b.calls[0]["json"]
    assert "A-bound-secret" not in json.dumps(wire)
    assert "bound_state" not in wire.get("metadata", {})


# --------------------------------------------------------------------------
# 4. reactive strip-and-retry
# --------------------------------------------------------------------------


class _FakeCredentialContext:
    def __init__(self, credential: str):
        self.credential = credential
        self.stable_id = "stable-id"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def mark_success(self, **kwargs) -> None:
        pass

    def mark_failure(self, classified) -> None:
        pass


class _FakeUsageManager:
    def __init__(self, credential: str = "cred-1"):
        self.credential = credential
        self.initialized = True
        self.window_manager = SimpleNamespace(get_primary_definition=lambda: None)

    async def initialize(self, credentials=None, priorities=None, tiers=None):
        self.initialized = True

    async def acquire_credential(self, *args, **kwargs):
        return _FakeCredentialContext(self.credential)

    def get_model_quota_group(self, model):
        return None

    async def get_availability_stats(self, model, quota_group=None):
        return {"available": 1, "total": 1}


def _provider_error(status: int, message: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.example/v1/chat")
    response = httpx.Response(
        status,
        json={"error": {"message": message, "type": "invalid_request_error"}},
        request=request,
    )
    return httpx.HTTPStatusError(message, request=request, response=response)


def _request_context(kwargs: dict, protocol_request: dict) -> RequestContext:
    return RequestContext(
        model="dummy/test",
        provider="dummy",
        kwargs=deepcopy(kwargs),
        streaming=False,
        credentials=["cred-1"],
        deadline=9999999999.0,
        input_protocol_name="openai_chat",
        protocol_request=deepcopy(protocol_request),
    )


def _executor(plugin_cls: type) -> RequestExecutor:
    class DummyProvider(plugin_cls):  # type: ignore[misc, valid-type]
        provider_env_name = "dummy"

        async def get_models(self, api_key, client):
            return []

    return RequestExecutor(
        usage_managers={"dummy": _FakeUsageManager("cred-1")},
        cooldown_manager=None,
        credential_filter=CredentialFilter({"dummy": DummyProvider}),
        provider_transforms=ProviderTransforms({"dummy": DummyProvider}, None),
        provider_plugins={"dummy": DummyProvider},
        http_client=httpx.AsyncClient(),
        max_retries=3,
        global_timeout=5,
    )


def _opaque_chat_payload() -> dict:
    return {
        "model": "dummy/test",
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "answer",
                "extra_content": {"google": {"thought_signature": "sig-a"}},
            },
        ],
    }


@pytest.mark.asyncio
async def test_reactive_retry_strips_once_and_retries_same_credential() -> None:
    class SignatureRejectingPlugin(ProviderInterface):  # type: ignore[misc]
        def has_custom_logic(self) -> bool:
            return True

        def __init__(self):
            self.seen: list[dict] = []

        async def acompletion(self, client, **kwargs):
            self.seen.append(deepcopy(kwargs))
            if len(self.seen) == 1:
                raise _provider_error(400, "invalid signature in thinking block")
            return _chat_response()

    plugin_cls = SignatureRejectingPlugin
    executor = _executor(plugin_cls)
    payload = _opaque_chat_payload()
    context = _request_context(payload, payload)

    with patch(
        "rotator_library.client.executor.litellm.acompletion",
        side_effect=AssertionError("litellm must not be used"),
    ):
        result = await executor._execute_non_streaming(context)

    plugin = executor._get_plugin_instance("dummy")
    assert len(plugin.seen) == 2
    first_call, second_call = plugin.seen
    assert first_call["messages"][1]["extra_content"]["google"]["thought_signature"] == "sig-a"
    assert "extra_content" not in second_call["messages"][1]
    assert result["choices"][0]["message"]["content"] == "hi"


@pytest.mark.asyncio
async def test_reactive_retry_does_not_fire_for_non_signature_error() -> None:
    class SchemaRejectingPlugin(ProviderInterface):  # type: ignore[misc]
        def has_custom_logic(self) -> bool:
            return True

        def __init__(self):
            self.calls = 0

        async def acompletion(self, client, **kwargs):
            self.calls += 1
            raise _provider_error(400, "tool schema invalid")

    executor = _executor(SchemaRejectingPlugin)
    payload = _opaque_chat_payload()
    context = _request_context(payload, payload)

    with pytest.raises(httpx.HTTPStatusError):
        await executor._execute_non_streaming(context)

    plugin = executor._get_plugin_instance("dummy")
    assert plugin.calls == 1


# --------------------------------------------------------------------------
# 5. reserved-key guard
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reserved_metadata_key_cannot_be_overwritten_by_cache() -> None:
    rule = FieldCacheRule(
        name="smuggle_input_provider",
        source="response",
        path="choices.0.message.reasoning_content",
        inject=FieldCacheInjection(target="metadata", path="input_provider"),
        scope=("provider", "model"),
    )
    executor = NativeProviderExecutor()
    cache_engine = FieldCacheEngine((rule,), store=executor.field_cache_store)
    context = NativeProviderContext(
        provider="provider_b",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://provider.example/v1/chat/completions",
        field_cache_rules=(rule,),
        metadata={"input_provider": "provider_a"},
    )
    # seed the row the rule will try to inject
    await cache_engine.extract(
        "response",
        {"choices": [{"message": {"reasoning_content": "attacker-provider"}}]},
        context.field_cache_context(),
    )

    result = await executor._inject_metadata(context, cache_engine)

    assert result.metadata["input_provider"] == "provider_a"


# --------------------------------------------------------------------------
# 6. build_cache_key provider+model floor
# --------------------------------------------------------------------------


def test_build_cache_key_provider_floor_survives_narrow_scope() -> None:
    rule = FieldCacheRule(
        name="session_only",
        source="response",
        path="choices.0.message.reasoning_content",
        scope=("session",),
    )
    key_a = build_cache_key(
        rule, FieldCacheContext(provider="provider_a", model="m", session_id="s")
    )
    key_b = build_cache_key(
        rule, FieldCacheContext(provider="provider_b", model="m", session_id="s")
    )
    assert key_a is not None and key_b is not None
    assert key_a != key_b
    assert "provider=" in key_a
    assert "session=" in key_a


# --------------------------------------------------------------------------
# 7. sentinel wire shape
# --------------------------------------------------------------------------


def test_gemini_skip_signature_sentinel_serializes_to_legal_shape() -> None:
    assert (
        json.dumps(_gemini_skip_signature_sentinel())
        == '{"thoughtSignature": "skip_thought_signature_validator"}'
    )


# --------------------------------------------------------------------------
# 8. D4 raw/rebuild gate divergence
# --------------------------------------------------------------------------


def test_wire_view_gate_diverges_on_operation_logical_operation_and_input() -> None:
    protocol = get_protocol("openai_chat")
    protocol_context = ProtocolContext(
        source_protocol="openai_chat",
        target_protocol="openai_chat",
    )
    base = protocol.parse_request(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        protocol_context,
    )
    assert _wire_view_matches_unified(deepcopy(base), base) is True
    assert _wire_view_matches_unified(replace(base, operation="embeddings"), base) is False
    assert _wire_view_matches_unified(replace(base, logical_operation="count_tokens"), base) is False
    assert _wire_view_matches_unified(replace(base, input=["diverged"]), base) is False
