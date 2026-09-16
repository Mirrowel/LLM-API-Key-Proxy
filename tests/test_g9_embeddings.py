# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G9 first-class embeddings acceptance tests.

Coverage: the openai_embeddings wire adapter (build/format/usage), gemini
single + batch conversion both directions, request validation before any
credential work, the executor-level conversion matrix (openai -> openai raw
passthrough, openai -> gemini single and array->batch, ollama passthrough,
face-aware operation gating, unsupported-provider rejection), prompt-only
usage accounting, the litellm aembedding dispatch, the builder's mixed-group
skip-and-record, and the route helper's verbatim input contract.
"""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from types import SimpleNamespace

import pytest

from proxy_app.route_helpers import classify_route_error, execute_embeddings, route_error_response
from rotator_library.client.executor import RequestExecutor, RoutingExecutionError, _should_use_native_protocol
from rotator_library.client.filters import CredentialFilter
from rotator_library.client.request_builder import RequestContextBuilder
from rotator_library.client.transforms import ProviderTransforms
from rotator_library.core.types import RequestContext
from rotator_library.protocols import (
    OPERATION_EMBEDDINGS,
    ProtocolContext,
    ProtocolError,
    get_protocol,
)
from rotator_library.protocols.validation import validate_embeddings_request
from rotator_library.providers.provider_interface import ProviderInterface
from rotator_library.routing.types import RouteTarget
from rotator_library.usage.accounting import extract_embeddings_usage_record
import litellm


def _openai():
    return get_protocol("openai_embeddings")


def _gemini():
    return get_protocol("gemini")


# ---------------------------------------------------------------------------
# openai_embeddings adapter: openai wire build / envelope normalization
# ---------------------------------------------------------------------------


def test_openai_embeddings_build_request_emits_forwarding_wire() -> None:
    """Cross-protocol INTO an openai-family target emits the /embeddings
    body: model + verbatim input + the canonical controls in openai
    spelling, gemini's output_dimensionality mapped onto dimensions, and
    gemini-only retrieval controls disclosed as dropped."""

    adapter = _openai()
    request = adapter.parse_request(
        {"model": "text-embed", "input": ["one", "two"], "dimensions": 128, "user": "u1", "custom_flag": True}
    )
    built = adapter.build_request(request, ProtocolContext(target_protocol="openai_embeddings"))
    assert built == {
        "model": "text-embed",
        "input": ["one", "two"],
        "dimensions": 128,
        "user": "u1",
        "custom_flag": True,
    }

    # A gemini-shaped canonical request converts: dimensions naming + drop
    # disclosure for the retrieval controls with no openai representation.
    gemini_unified = _gemini().parse_request(
        {
            "model": "models/text-embed",
            "content": {"parts": [{"text": "doc one"}]},
            "taskType": "RETRIEVAL_DOCUMENT",
            "title": "Doc",
            "outputDimensionality": 256,
        },
        ProtocolContext(metadata={"operation": "embeddings"}),
    )
    converted = adapter.build_request(gemini_unified, ProtocolContext(source_protocol="gemini", target_protocol="openai_embeddings"))
    assert converted["dimensions"] == 256
    assert "task_type" not in converted and "title" not in converted
    # The conversion output only follows the openai surface.
    assert set(converted) == {"model", "input", "dimensions"}
    dropped = {warning.field for warning in gemini_unified.warnings if warning.code == "unsupported_optional_control"}
    assert {"task_type", "title"} <= dropped


def test_openai_embeddings_format_response_normalizes_entries_and_prompt_usage() -> None:
    """Any source's data entries land in the openai list envelope with
    embedding+index (unknown entry fields preserved), and usage is emitted
    in the openai spelling even when the source reported gemini buckets."""

    adapter = _openai()
    gemini_parsed = _gemini().parse_response(
        {
            "embedding": {"values": [0.1, 0.2]},
            "usageMetadata": {"promptTokenCount": 7, "totalTokenCount": 7},
        },
        ProtocolContext(metadata={"operation": "embeddings"}),
    )

    formatted = adapter.format_response(gemini_parsed, ProtocolContext(source_protocol="gemini", target_protocol="openai_embeddings"))
    assert formatted["object"] == "list"
    assert formatted["data"] == [{"embedding": [0.1, 0.2], "object": "embedding", "index": 0}]
    assert formatted["usage"] == {"prompt_tokens": 7, "total_tokens": 7}

    # Same-protocol openai entries keep their provider extras verbatim.
    same = adapter.parse_response(
        {
            "data": [{"object": "embedding", "embedding": [0.5], "index": 0, "sparse": {"indices": [1]}}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }
    )
    echoed = adapter.format_response(same, ProtocolContext(source_protocol="openai_embeddings", target_protocol="openai_embeddings"))
    assert echoed["data"][0]["sparse"] == {"indices": [1]}
    assert echoed["usage"] == {"prompt_tokens": 2, "total_tokens": 2}

    # A bare vector list (ollama source) is wrapped into entry objects.
    bare = adapter.format_response(
        type(same)(operation=OPERATION_EMBEDDINGS, data=[[0.9]], usage=None),
        ProtocolContext(source_protocol="ollama", target_protocol="openai_embeddings"),
    )
    assert bare["data"] == [{"embedding": [0.9], "object": "embedding", "index": 0}]


def test_openai_embeddings_extract_usage_is_prompt_only() -> None:
    adapter = _openai()
    usage = adapter.extract_usage({"usage": {"prompt_tokens": 4, "total_tokens": 4}})
    assert usage is not None
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (4, 0, 4)


# ---------------------------------------------------------------------------
# gemini adapter: single / batch request + response conversion
# ---------------------------------------------------------------------------


def test_gemini_single_embeddings_round_trip() -> None:
    adapter = _gemini()
    raw = {
        "model": "models/embed-1",
        "content": {"parts": [{"text": "hello"}]},
        "taskType": "RETRIEVAL_QUERY",
        "outputDimensionality": 64,
    }
    unified = adapter.parse_request(raw, ProtocolContext(metadata={"operation": "embeddings"}))
    assert unified.operation == "embeddings"
    assert unified.input == "hello"
    assert unified.model == "models/embed-1"
    assert unified.generation_params == {"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 64}
    # Same-protocol rebuild is faithful (body model preserved, wire names back).
    assert adapter.build_request(unified, ProtocolContext(metadata={"operation": "embeddings"})) == raw


def test_gemini_batch_embeddings_round_trip_preserves_items_and_controls() -> None:
    adapter = _gemini()
    raw = {
        "requests": [
            {"model": "models/embed-1", "content": {"parts": [{"text": "a"}]}, "taskType": "RETRIEVAL_DOCUMENT", "title": "Doc"},
            {"model": "models/embed-1", "content": {"parts": [{"text": "b"}]}, "taskType": "RETRIEVAL_DOCUMENT", "title": "Doc"},
        ]
    }
    unified = adapter.parse_request(raw)
    # The wire shape is authoritative even without a context operation.
    assert unified.operation == "embeddings_batch"
    assert unified.input == ["a", "b"]
    assert unified.model == "models/embed-1"
    assert adapter.build_request(unified, ProtocolContext(metadata={"operation": "embeddings_batch"})) == raw


def test_gemini_embeddings_response_parsing_single_and_batch() -> None:
    adapter = _gemini()
    single = adapter.parse_response(
        {"embedding": {"values": [0.1, 0.2]}, "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 3}},
        ProtocolContext(metadata={"operation": "embeddings"}),
    )
    assert single.operation == "embeddings"
    assert single.data == [{"embedding": [0.1, 0.2], "object": "embedding", "index": 0}]
    assert single.usage is not None and single.usage.input_tokens == 3 and single.usage.output_tokens == 0

    batch = adapter.parse_response(
        {
            "embeddings": [
                {"values": [0.1]},
                {"values": [0.2], "statistics": {"truncated": False}},
            ]
        }
    )
    assert batch.operation == "embeddings_batch"
    assert [entry["index"] for entry in batch.data] == [0, 1]
    assert batch.data[0]["embedding"] == [0.1]
    assert batch.data[1]["statistics"] == {"truncated": False}

    # Gemini client formatting restores the native envelopes.
    assert adapter.format_response(single) == {
        "embedding": {"values": [0.1, 0.2]},
        "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 3},
    }
    assert adapter.format_response(batch) == {"embeddings": [{"values": [0.1]}, {"values": [0.2]}]}


def test_gemini_cross_build_from_openai_single_and_batch() -> None:
    """Cross-protocol (openai canonical -> gemini target): a string input
    builds embedContent (no body model when the source had none); an array
    builds batchEmbedContents with the resource model on every item and the
    openai dimensions mapped onto outputDimensionality (uniform by
    construction — per-item taskType cannot differ)."""

    adapter = _gemini()
    openai_adapter = _openai()

    single_unified = openai_adapter.parse_request({"model": "embed-1", "input": "hello"})
    single = adapter.build_request(single_unified, ProtocolContext(metadata={"operation": "embeddings"}))
    assert single == {"content": {"parts": [{"text": "hello"}]}}

    array_unified = openai_adapter.parse_request({"model": "embed-1", "input": ["a", "b"], "dimensions": 32})
    batch = adapter.build_request(array_unified, ProtocolContext(metadata={"operation": "embeddings_batch"}))
    assert batch == {
        "requests": [
            {"model": "models/embed-1", "content": {"parts": [{"text": "a"}]}, "outputDimensionality": 32},
            {"model": "models/embed-1", "content": {"parts": [{"text": "b"}]}, "outputDimensionality": 32},
        ]
    }

    # A list input forces batch even when a stale context says single: an
    # array cannot ride the single endpoint.
    stale_context_batch = adapter.build_request(array_unified, ProtocolContext(metadata={"operation": "embeddings"}))
    assert "requests" in stale_context_batch


def test_gemini_cross_build_discloses_unrepresentable_openai_controls() -> None:
    adapter = _gemini()
    unified = _openai().parse_request(
        {"model": "embed-1", "input": "x", "encoding_format": "base64", "user": "u1"}
    )
    built = adapter.build_request(unified, ProtocolContext(metadata={"operation": "embeddings"}))
    assert built == {"content": {"parts": [{"text": "x"}]}}
    assert "encoding_format" not in built and "user" not in built
    dropped = {warning.field for warning in unified.warnings if warning.code == "unsupported_optional_control"}
    assert {"encoding_format", "user"} <= dropped


def test_gemini_cross_build_rejects_token_arrays() -> None:
    """Gemini embeds text parts only: a tokenized input is a named rejection,
    never a silent mangling into a string."""

    unified = _openai().parse_request({"model": "embed-1", "input": [[1, 2, 3]]})
    with pytest.raises(ProtocolError, match="text inputs only"):
        _gemini().build_request(unified, ProtocolContext(metadata={"operation": "embeddings_batch"}))


# ---------------------------------------------------------------------------
# Validation: shape contract + the 400 ladder before rotation
# ---------------------------------------------------------------------------


def test_validate_embeddings_request_shape_matrix() -> None:
    adapter = _openai()
    context = ProtocolContext()

    def _unified(**overrides):
        payload = {"model": "m", "input": "x"}
        payload.update(overrides)
        return adapter.parse_request(payload)

    # Valid shapes.
    validate_embeddings_request(_unified(), "openai_embeddings", context)
    validate_embeddings_request(_unified(input=["a", "b"]), "openai_embeddings", context)
    validate_embeddings_request(_unified(input=[1, 2, 3]), "openai_embeddings", context)
    validate_embeddings_request(_unified(input=[[1, 2], [3]]), "openai_embeddings", context)
    validate_embeddings_request(_unified(dimensions=64, encoding_format="base64"), "openai_embeddings", context)

    for bad in (
        _unified(input=""),
        _unified(input="   "),
        _unified(input=[]),
        _unified(input=[""]),
        _unified(input=[[1, -2]]),
        _unified(input=["a", 3]),
    ):
        with pytest.raises(ProtocolError):
            validate_embeddings_request(bad, "openai_embeddings", context)
    with pytest.raises(ProtocolError, match="dimensions"):
        validate_embeddings_request(_unified(dimensions=0), "openai_embeddings", context)
    with pytest.raises(ProtocolError, match="dimensions"):
        validate_embeddings_request(_unified(dimensions=True), "openai_embeddings", context)
    with pytest.raises(ProtocolError, match="encoding_format"):
        validate_embeddings_request(_unified(encoding_format="int8"), "openai_embeddings", context)


@pytest.mark.asyncio
async def test_embedding_builder_validation_fails_before_credentials(monkeypatch) -> None:
    """Malformed embeddings payloads raise the protocol-shaped 400 from the
    builder — before routing, provider selection, or any credential attempt
    (rotation never burns keys on a client error)."""

    monkeypatch.delenv("FALLBACK_GROUPS", raising=False)
    builder = RequestContextBuilder(
        resolve_scope_for_provider=_scope,
        model_resolver=_FakeModelResolver(),
        session_tracker=_FakeSessionTracker(),
        get_global_timeout=lambda: 30,
        get_enable_request_logging=lambda: False,
    )

    with pytest.raises(ProtocolError, match="non-empty"):
        await builder.build_embedding_context(None, None, {"model": "openai/text-embed", "input": ""})
    with pytest.raises(ProtocolError, match="input array must not be empty"):
        await builder.build_embedding_context(None, None, {"model": "openai/text-embed", "input": []})
    with pytest.raises(ProtocolError, match="encoding_format"):
        await builder.build_embedding_context(
            None, None, {"model": "openai/text-embed", "input": "x", "encoding_format": "int8"}
        )

    # The route ladder renders the ProtocolError as the openai-shaped 400.
    error = ProtocolError("Embeddings input must be a non-empty string", protocol="openai_embeddings", pass_name="validate_request")
    assert classify_route_error(error) == ("invalid_request", 400)
    status, body = route_error_response(error, protocol="openai_chat")
    assert status == 400
    assert body["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Executor matrix: conversion, raw passthrough, usage, dispatch
# ---------------------------------------------------------------------------


class _FakeCredentialContext:
    def __init__(self) -> None:
        self.credential = "cred-1"
        self.stable_id = "stable-id"
        self.recorded: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def mark_success(self, **kwargs) -> None:
        self.recorded.update(kwargs)

    def mark_failure(self, classified) -> None:
        pass


class _FakeUsageManager:
    def __init__(self) -> None:
        self.initialized = True
        self.window_manager = SimpleNamespace(get_primary_definition=lambda: None)
        self.credential_context = _FakeCredentialContext()

    async def initialize(self, credentials=None, priorities=None, tiers=None):
        self.initialized = True

    async def acquire_credential(self, *args, **kwargs):
        return self.credential_context

    def get_model_quota_group(self, model):
        return None

    async def get_availability_stats(self, model, quota_group=None):
        return {"available": 1, "total": 1}


class _OpenAICompatFace(ProviderInterface):
    """Single-face openai_chat provider (dummy base URL, no network)."""

    provider_env_name = "dummy"
    speaks = ("openai_chat",)
    default_api_base = "https://dummy.example/v1"


class _GeminiFace(ProviderInterface):
    provider_env_name = "dummygem"
    speaks = ("gemini",)
    default_api_base = "https://gemini.example"


class _OllamaFace(ProviderInterface):
    provider_env_name = "dummyollama"
    speaks = ("ollama",)
    default_api_base = "https://ollama.example"


class _AnthropicFace(ProviderInterface):
    provider_env_name = "dummyanth"
    speaks = ("anthropic_messages",)
    default_api_base = "https://anth.example"


class _MultiFaceProvider(ProviderInterface):
    """Responses-first multi-face provider (the real openai shape): the
    default face has no embeddings route, the chat face does."""

    provider_env_name = "dummymulti"
    speaks = (("responses", {}), ("chat", "openai_chat", {}))
    default_api_base = "https://multi.example/v1"


class _FakeHTTPResponse:
    def __init__(self, payload, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers: dict = {}

    def json(self):
        return self._payload


class _FakeHTTPClient:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    async def post(self, endpoint, *, headers, json, **kwargs):
        self.calls.append({"endpoint": endpoint, "headers": headers, "json": json})
        return _FakeHTTPResponse(self.payload)


def _executor_for(plugin_cls: type, provider: str):
    usage_manager = _FakeUsageManager()
    executor = RequestExecutor(
        usage_managers={provider: usage_manager},
        cooldown_manager=None,
        credential_filter=CredentialFilter({provider: plugin_cls}),
        provider_transforms=ProviderTransforms({provider: plugin_cls}, None),
        provider_plugins={provider: plugin_cls},
        http_client=None,
        max_retries=1,
        global_timeout=5,
    )
    return executor, usage_manager


def _embedding_context(
    provider: str,
    model: str,
    payload: dict,
    *,
    input_protocol: str = "openai_embeddings",
) -> RequestContext:
    protocol = get_protocol(input_protocol)
    unified = protocol.parse_request(deepcopy(payload), ProtocolContext(metadata={"operation": "embeddings"}))
    return RequestContext(
        model=model,
        provider=provider,
        kwargs=deepcopy(payload),
        streaming=False,
        credentials=["cred-1"],
        deadline=9999999999.0,
        input_protocol_name=input_protocol,
        protocol_request=deepcopy(payload),
        requested_operation="embeddings",
        unified_request=unified,
        credential_secrets={"cred-1": "secret"},
    )


@pytest.mark.asyncio
async def test_executor_openai_to_openai_raw_passthrough_prompt_usage() -> None:
    """openai client -> openai-family provider: the pristine client payload
    IS the transport basis (raw fast path) with only the upstream model
    overlay, and usage books prompt tokens with zero completion."""

    provider_response = {
        "object": "list",
        "model": "text-embed",
        "data": [
            {"object": "embedding", "embedding": [0.1], "index": 0},
            {"object": "embedding", "embedding": [0.2], "index": 1},
        ],
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }
    executor, usage_manager = _executor_for(_OpenAICompatFace, "dummy")
    fake_http = _FakeHTTPClient(provider_response)
    executor._http_client = fake_http
    payload = {"model": "dummy/text-embed", "input": ["a", "b"], "dimensions": 64}
    context = _embedding_context("dummy", "dummy/text-embed", payload)

    result = await executor._execute_non_streaming(context)

    assert fake_http.calls[0]["endpoint"] == "https://dummy.example/v1/embeddings"
    # Raw basis: input/dimensions verbatim; only the model is overlaid to
    # the upstream id.
    assert fake_http.calls[0]["json"] == {"model": "text-embed", "input": ["a", "b"], "dimensions": 64}
    assert result["data"][1]["embedding"] == [0.2]
    recorded = usage_manager.credential_context.recorded
    assert recorded["prompt_tokens"] == 4
    assert recorded["completion_tokens"] == 0
    assert recorded["thinking_tokens"] == 0


@pytest.mark.asyncio
async def test_executor_openai_to_gemini_single_conversion() -> None:
    executor, usage_manager = _executor_for(_GeminiFace, "dummygem")
    fake_http = _FakeHTTPClient(
        {"embedding": {"values": [0.1, 0.2]}, "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 3}}
    )
    executor._http_client = fake_http
    context = _embedding_context("dummygem", "dummygem/embed-1", {"model": "dummygem/embed-1", "input": "hello"})

    result = await executor._execute_non_streaming(context)

    assert fake_http.calls[0]["endpoint"] == "https://gemini.example/v1beta/models/embed-1:embedContent"
    # Cross-protocol single: no body model (the resource rides the URL).
    assert fake_http.calls[0]["json"] == {"content": {"parts": [{"text": "hello"}]}}
    assert result["data"] == [{"embedding": [0.1, 0.2], "object": "embedding", "index": 0}]
    assert result["usage"] == {"prompt_tokens": 3, "total_tokens": 3}
    recorded = usage_manager.credential_context.recorded
    assert recorded["prompt_tokens"] == 3 and recorded["completion_tokens"] == 0


@pytest.mark.asyncio
async def test_executor_openai_to_gemini_array_converts_to_batch() -> None:
    executor, _ = _executor_for(_GeminiFace, "dummygem")
    fake_http = _FakeHTTPClient({"embeddings": [{"values": [0.1]}, {"values": [0.2]}]})
    executor._http_client = fake_http
    context = _embedding_context("dummygem", "dummygem/embed-1", {"model": "dummygem/embed-1", "input": ["a", "b"]})

    result = await executor._execute_non_streaming(context)

    assert fake_http.calls[0]["endpoint"] == "https://gemini.example/v1beta/models/embed-1:batchEmbedContents"
    assert fake_http.calls[0]["json"] == {
        "requests": [
            {"model": "models/embed-1", "content": {"parts": [{"text": "a"}]}},
            {"model": "models/embed-1", "content": {"parts": [{"text": "b"}]}},
        ]
    }
    assert [entry["index"] for entry in result["data"]] == [0, 1]


@pytest.mark.asyncio
async def test_executor_ollama_to_ollama_raw_passthrough_with_prompt_usage() -> None:
    provider_response = {"model": "embed", "embeddings": [[0.1, 0.2]], "prompt_eval_count": 2}
    executor, usage_manager = _executor_for(_OllamaFace, "dummyollama")
    fake_http = _FakeHTTPClient(provider_response)
    executor._http_client = fake_http
    payload = {"model": "dummyollama/embed", "input": "embed me"}
    context = _embedding_context("dummyollama", "dummyollama/embed", payload, input_protocol="ollama")

    result = await executor._execute_non_streaming(context)

    assert fake_http.calls[0]["endpoint"] == "https://ollama.example/api/embed"
    assert result == provider_response
    recorded = usage_manager.credential_context.recorded
    assert recorded["prompt_tokens"] == 2 and recorded["completion_tokens"] == 0


@pytest.mark.asyncio
async def test_executor_multi_face_provider_routes_embeddings_to_chat_face() -> None:
    """The default responses face has no embeddings surface; the operation
    gate's declared-face scan keeps the request native and the context
    builder resolves the chat face — the request still rides the raw
    openai embeddings path."""

    plugin = _MultiFaceProvider()
    assert plugin.supports_native_operation("dummymulti/m", "embeddings") is False
    assert (
        _should_use_native_protocol(
            plugin,
            "dummymulti/m",
            None,
            {"model": "dummymulti/m", "input": "x"},
            stream=False,
            execution="auto",
            requested_operation="embeddings",
        )
        is True
    )

    executor, _ = _executor_for(_MultiFaceProvider, "dummymulti")
    fake_http = _FakeHTTPClient({"object": "list", "data": [{"object": "embedding", "embedding": [0.1], "index": 0}]})
    executor._http_client = fake_http
    context = _embedding_context("dummymulti", "dummymulti/text-embed", {"model": "dummymulti/text-embed", "input": "x"})

    result = await executor._execute_non_streaming(context)

    assert fake_http.calls[0]["endpoint"] == "https://multi.example/v1/embeddings"
    assert fake_http.calls[0]["json"] == {"model": "text-embed", "input": "x"}
    assert result["object"] == "list"


@pytest.mark.asyncio
async def test_executor_unsupported_provider_raises_operation_unsupported() -> None:
    """A provider with no embeddings surface is an honest, failover-eligible
    operation gap (never a chat call to /messages, never a silent empty)."""

    executor, _ = _executor_for(_AnthropicFace, "dummyanth")
    executor._http_client = _FakeHTTPClient({})
    context = _embedding_context("dummyanth", "dummyanth/claude", {"model": "dummyanth/claude", "input": "x"})
    context.routing_targets = (RouteTarget(provider="dummyanth", model="dummyanth/claude", execution="native"),)
    context.routing_target_index = 0

    with pytest.raises(RoutingExecutionError) as raised:
        await executor._execute_provider_request(
            "dummyanth",
            "dummyanth/claude",
            executor._get_plugin_instance("dummyanth"),
            "secret",
            "stable-id",
            dict(context.kwargs),
            context,
        )

    assert raised.value.error_type == "operation_unsupported"


@pytest.mark.asyncio
async def test_executor_litellm_branch_dispatches_aembedding(monkeypatch) -> None:
    """The declared LiteLLM fallback for embeddings must call aembedding —
    never acompletion (an embeddings payload must never land on chat)."""

    calls: dict = {}

    async def fake_aembedding(**kwargs):
        calls.update(kwargs)
        return {"object": "list", "data": []}

    async def fake_acompletion(**kwargs):  # pragma: no cover — the assertion
        raise AssertionError("acompletion must not be used for embeddings")

    monkeypatch.setattr(litellm, "aembedding", fake_aembedding)
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    executor = RequestExecutor.__new__(RequestExecutor)
    executor._litellm_logger_fn = None
    context = SimpleNamespace(
        requested_operation="embeddings",
        transaction_logger=None,
        provider="openai",
        model="openai/text-embed",
        streaming=False,
        session_id=None,
        usage_manager_key=None,
        classifier=None,
    )

    result = await executor._execute_litellm_request(
        {"model": "openai/text-embed", "input": "x", "transaction_context": {"drop": True}},
        "secret",
        context=context,
    )

    assert result["object"] == "list"
    assert calls["model"] == "openai/text-embed"
    assert calls["input"] == "x"
    assert calls["api_key"] == "secret"
    assert "transaction_context" not in calls


def test_embeddings_usage_record_is_prompt_only_across_client_shapes() -> None:
    openai_record = extract_embeddings_usage_record({"usage": {"prompt_tokens": 5, "total_tokens": 5}})
    assert (openai_record.input_tokens, openai_record.completion_tokens, openai_record.total_tokens) == (5, 0, 5)

    gemini_record = extract_embeddings_usage_record(
        {"usageMetadata": {"promptTokenCount": 6, "totalTokenCount": 6}}
    )
    assert (gemini_record.input_tokens, gemini_record.completion_tokens) == (6, 0)

    ollama_record = extract_embeddings_usage_record({"prompt_eval_count": 7, "eval_count": 0})
    assert (ollama_record.input_tokens, ollama_record.completion_tokens) == (7, 0)


# ---------------------------------------------------------------------------
# Builder: mixed fallback group skip-and-record
# ---------------------------------------------------------------------------


class _FakeModelResolver:
    def resolve_model_id(self, model, provider):
        return model


class _FakeSession:
    session_id = "session"
    affinity_key = "affinity"
    tracking_namespace = "namespace"


class _FakeSessionTracker:
    def infer_session(self, *args, **kwargs):
        return _FakeSession()


async def _scope(provider, classifier, request_api_keys, request_providers, private):
    return {
        "credentials": [f"{provider}-cred"],
        "usage_manager_key": provider,
        "provider_config": {"provider": provider},
        "credential_secrets": {f"{provider}-cred": f"{provider}-secret"},
        "classifier": classifier or "global",
    }


@pytest.mark.asyncio
async def test_builder_skips_embeddings_incapable_target_in_mixed_group(monkeypatch, caplog) -> None:
    monkeypatch.setenv("FALLBACK_GROUPS", "embed_chain")
    monkeypatch.setenv("FALLBACK_GROUP_EMBED_CHAIN", "gemini/embed-1,anthropic/claude-1")
    monkeypatch.setenv("MODEL_ROUTE_EMBED", "group:embed_chain")

    instances = {"gemini": _GeminiFace(), "anthropic": _AnthropicFace()}
    builder = RequestContextBuilder(
        resolve_scope_for_provider=_scope,
        model_resolver=_FakeModelResolver(),
        session_tracker=_FakeSessionTracker(),
        get_global_timeout=lambda: 30,
        get_enable_request_logging=lambda: False,
        get_provider_instance=lambda provider: instances.get(provider),
    )

    logger = logging.getLogger("rotator_library")
    original_propagate = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="rotator_library"):
            context = await builder.build_embedding_context(None, None, {"model": "embed", "input": "hi"})
    finally:
        logger.propagate = original_propagate

    assert context.requested_operation == "embeddings"
    # The anthropic target (no embeddings endpoint) was skipped with a
    # record; the gemini target survives the chain.
    assert [target.provider for target in context.routing_targets] == ["gemini"]
    assert any("no embeddings surface" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Route helpers: verbatim input + positional raw_request
# ---------------------------------------------------------------------------


def test_execute_embeddings_input_rides_verbatim_and_request_is_positional() -> None:
    class _Client:
        """Mirrors RotatingClient.aembedding: the transport context kwarg is
        deliberately NOT named "request" so a payload field can't collide."""

        def __init__(self) -> None:
            self.calls: list[tuple] = []

        async def aembedding(self, http_request=None, **kwargs):
            self.calls.append((http_request, kwargs))
            return {"ok": True}

    client = _Client()
    payload = {"model": "m", "input": "hello", "request": "client-supplied"}
    result = asyncio.run(execute_embeddings(client, payload, raw_request="RAW-REQUEST"))

    assert result == {"ok": True}
    raw_request, kwargs = client.calls[0]
    assert raw_request == "RAW-REQUEST"
    # The string input is NOT rewritten to [str]; a client field named
    # "request" survives as payload.
    assert kwargs["input"] == "hello"
    assert kwargs["request"] == "client-supplied"
    assert payload["input"] == "hello"
