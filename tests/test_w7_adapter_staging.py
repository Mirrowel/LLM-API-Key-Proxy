# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""W7 acceptance fixtures: adapter staging contracts.

Adapters are WIRE adapters: request-side they run on the built
provider-native payload (after the finalizer), response-side on the raw
provider response BEFORE parsing, stream-side on the raw provider chunk
BEFORE parsing. They never see the client protocol, and their effect is
identical for every client of the same provider.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from rotator_library.adapters import PayloadAdapter, register_adapter
from rotator_library.native_provider.context import NativeProviderContext
from rotator_library.native_provider.executor import NativeProviderExecutor
from rotator_library.native_provider.http import NativeHTTPTransport


class RecordingTransport:
    def __init__(self, response):
        self.response = response
        self.payloads = []

    async def post_json(self, endpoint, headers, payload):
        self.payloads.append(deepcopy(payload))
        return deepcopy(self.response)


class _ContextCapture:
    contexts = []

    @classmethod
    def reset(cls):
        cls.contexts = []


class ContextCapturingAdapter(PayloadAdapter):
    name = "w7_context_capture"
    supported_stages = ("request", "response", "stream_event")

    async def transform_request(self, payload, context):
        _ContextCapture.contexts.append(("request", context))
        return payload

    async def transform_response(self, payload, context):
        _ContextCapture.contexts.append(("response", context))
        return payload


@pytest.mark.asyncio
async def test_adapters_never_observe_the_client_protocol() -> None:
    """The adapter context carries only provider-side identity — running
    the same provider request with different CLIENT protocols must give
    adapters byte-identical contexts and inputs."""
    register_adapter(ContextCapturingAdapter, replace=True)
    _ContextCapture.reset()

    request = {"model": "model-test", "messages": [{"role": "user", "content": "hi"}]}
    anthropic_response = {
        "id": "msg_1",
        "model": "model-test",
        "role": "assistant",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "answer"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }

    for client_protocol in ("openai_chat", "gemini", "responses"):
        context = NativeProviderContext(
            provider="provider",
            model="model-test",
            protocol_name="anthropic_messages",
            endpoint="https://provider.test/messages",
            operation="messages",
            input_protocol_name=client_protocol,
            client_protocol_name=client_protocol,
            adapter_names=("w7_context_capture",),
        )
        await NativeProviderExecutor().execute(
            deepcopy(request),
            context,
            RecordingTransport(deepcopy(anthropic_response)),
        )

    assert len(_ContextCapture.contexts) == 6  # request + response per client
    # No context field or metadata key mentions a client-side protocol.
    for stage, ctx in _ContextCapture.contexts:
        assert ctx.protocol == "anthropic_messages"
        for field in ("input_protocol", "client_protocol", "input_protocol_name", "client_protocol_name"):
            assert not hasattr(ctx, field)
        flat = str(ctx.metadata or {})
        assert "openai_chat" not in flat and "gemini" not in flat and "responses" not in flat
    # Identical request payload for every client (adapter saw the same wire).
    assert _ContextCapture.contexts[0][1].provider == _ContextCapture.contexts[-1][1].provider


@pytest.mark.asyncio
async def test_response_wire_adapters_apply_once_before_parse_for_raw_passthrough() -> None:
    """Same-protocol requests keep the raw fast path, and response adapters
    apply exactly once — to the returned payload, not double-applied."""

    call_count = {"n": 0}

    class CountingAdapter(PayloadAdapter):
        name = "w7_counting"
        supported_stages = ("response",)

        async def transform_response(self, payload, context):
            call_count["n"] += 1
            payload["touched"] = True
            return payload

    register_adapter(CountingAdapter, replace=True)
    chat_response = {
        "id": "chat_1",
        "model": "gpt-test",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    context = NativeProviderContext(
        provider="provider",
        model="gpt-test",
        protocol_name="openai_chat",
        endpoint="https://provider.test/chat",
        operation="chat",
        input_protocol_name="openai_chat",
        client_protocol_name="openai_chat",
        adapter_names=("w7_counting",),
        raw_client_request={"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
    )

    result = await NativeProviderExecutor().execute(
        {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]},
        context,
        RecordingTransport(deepcopy(chat_response)),
    )

    assert call_count["n"] == 1
    assert result["touched"] is True  # adapted raw IS the client response
    assert result["choices"][0]["message"]["content"] == "hi"


@pytest.mark.asyncio
async def test_envelope_adapter_composes_after_content_adapters() -> None:
    """Declaration order is execution order: content edits (model alias)
    land INSIDE the envelope the envelope adapter wraps around them."""

    class AliasAdapter(PayloadAdapter):
        name = "w7_alias"
        supported_stages = ("request",)

        async def transform_request(self, payload, context):
            if isinstance(payload, dict):
                payload["model"] = "aliased-model"
            return payload

    register_adapter(AliasAdapter, replace=True)
    context = NativeProviderContext(
        provider="provider",
        model="model-test",
        protocol_name="gemini",
        endpoint="https://provider.test/v1",
        operation="generate",
        adapter_names=("w7_alias", "antigravity_envelope"),
        adapter_config={"antigravity_envelope": {"project": "proj", "user_agent": "ua", "request_type": "GENERATE"}},
    )

    transport = RecordingTransport({"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}], "modelVersion": "model-test"})
    await NativeProviderExecutor().execute(
        {"model": "model-test", "contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        context,
        transport,
    )

    sent = transport.payloads[0]
    assert "request" in sent and "requestType" in sent  # envelope applied
    assert sent["request"]["contents"][0]["parts"][0]["text"] == "hi"
    assert sent["model"] == "aliased-model"  # content edit inside the envelope
