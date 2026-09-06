# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""W7 acceptance fixtures: adapter staging contracts.

Adapters are WIRE adapters on the request and response sides: request-side
they run on the built provider-native payload (after the finalizer),
response-side on the raw provider response BEFORE parsing. Stream-side
they run on the NEUTRAL parsed event (plan §2.5) — provider frames are
SSE-wrapped transport, and neutral is the protocol-free seam. Adapters
never see the client protocol, and their effect is identical for every
client of the same provider.
"""

from __future__ import annotations

from pathlib import Path

from copy import deepcopy

import pytest

from rotator_library.adapters import PayloadAdapter, register_adapter
from rotator_library.native_provider.context import NativeProviderContext
from rotator_library.native_provider.executor import NativeProviderExecutor
from rotator_library.native_provider.http import NativeHTTPTransport




@pytest.fixture(autouse=True)


def _trace_text(log_dir):
    from rotator_library.utils import zstd_io

    entries = zstd_io.read_jsonl_any(Path(log_dir) / "transform_trace.jsonl")
    return "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries)
def _trace_level_2(monkeypatch):
    """Trace mechanics live at L2 (D15 tiers)."""
    monkeypatch.setenv("TRANSACTION_LOG_LEVEL", "2")
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

    async def transform_stream_event(self, payload, context):
        # Streaming neutral events also carry no client-protocol view.
        _ContextCapture.contexts.append(("stream_event", context))
        return payload


def _context_signature(context) -> tuple:
    """Everything an adapter can observe about a context."""
    import dataclasses

    return dataclasses.asdict(context)


@pytest.mark.asyncio
async def test_adapters_never_observe_the_client_protocol() -> None:
    """The adapter context carries only provider-side identity — running
    the same provider request with different CLIENT protocols must give
    adapters identical contexts (production-like metadata included)."""
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

    signatures = []
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
            metadata={"public_model": "provider/model-test", "input_provider": "provider"},
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
        for protocol_name in ("openai_chat", "gemini", "responses"):
            assert f"'{protocol_name}'" not in flat and f'"{protocol_name}"' not in flat
    # Contexts are byte-identical across the three clients (equality, not
    # substring luck): production-like metadata included.
    signatures = [_context_signature(ctx) for _, ctx in _ContextCapture.contexts]
    assert signatures[0] == signatures[2] == signatures[4]
    assert signatures[1] == signatures[3] == signatures[5]

    # Streaming neutral events carry the same client-protocol blind spot.
    from .test_native_provider_streaming import FakeStreamingClient

    stream_context = NativeProviderContext(
        provider="provider",
        model="model-test",
        protocol_name="openai_chat",
        endpoint="https://provider.test/chat",
        operation="chat",
        input_protocol_name="anthropic_messages",
        client_protocol_name="anthropic_messages",
        adapter_names=("w7_context_capture",),
        metadata={"public_model": "provider/model-test", "input_provider": "provider"},
    )
    _ContextCapture.reset()
    [
        event
        async for event in NativeProviderExecutor().stream(
            {"model": "model-test", "messages": [{"role": "user", "content": "hi"}]},
            stream_context,
            NativeHTTPTransport(FakeStreamingClient([{"choices": [{"delta": {"content": "x"}}]}, "[DONE]"])),
        )
    ]
    stream_stages = {stage for stage, _ in _ContextCapture.contexts}
    assert "stream_event" in stream_stages
    _, stream_ctx = _ContextCapture.contexts[-1]
    assert stream_ctx.protocol == "openai_chat"
    assert not hasattr(stream_ctx, "client_protocol")


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
            payload = deepcopy(payload)
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
                payload = deepcopy(payload)
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


@pytest.mark.asyncio
async def test_adapted_wire_feeds_cache_extract_usage_and_traces(tmp_path) -> None:
    """Field-cache extraction and usage accounting both observe the
    ADAPTED provider wire (adapters fix the dialect first), and the trace
    order pins adapter-before-extract."""
    import json

    from rotator_library.transaction_logger import TransactionLogger

    class ReasoningFixer(PayloadAdapter):
        name = "w7_reasoning_fixer"
        supported_stages = ("response",)

        async def transform_response(self, payload, context):
            if isinstance(payload, dict):
                payload = deepcopy(payload)
                for block in payload.get("content", []):
                    if block.get("type") == "text" and block.get("text") == "hidden":
                        block["text"] = "fixed"
                # Rewrite usage so the accounting pin cannot pass vacuously.
                if isinstance(payload.get("usage"), dict):
                    payload["usage"]["input_tokens"] = 99
            return payload

    register_adapter(ReasoningFixer, replace=True)
    logger = TransactionLogger("provider", "model-test", parent_dir=tmp_path)
    context = NativeProviderContext(
        provider="provider",
        model="model-test",
        protocol_name="anthropic_messages",
        endpoint="https://provider.test/messages",
        operation="messages",
        input_protocol_name="anthropic_messages",
        client_protocol_name="anthropic_messages",
        adapter_names=("w7_reasoning_fixer",),
        raw_client_request={"model": "model-test", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]},
        transaction_logger=logger,
    )
    response = {
        "id": "msg_1",
        "model": "model-test",
        "role": "assistant",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "hidden"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }

    result = await NativeProviderExecutor().execute(
        {"model": "model-test", "messages": [{"role": "user", "content": "hi"}]},
        context,
        RecordingTransport(response),
    )

    # The adapted value is what the client receives AND what downstream
    # stages (cache extract, usage) saw — traced in the right order.
    assert result["content"][0]["text"] == "fixed"
    trace = [json.loads(line) for line in _trace_text(logger.log_dir).splitlines()]
    pass_names = [entry["pass_name"] for entry in trace]
    assert pass_names.index("after_response_adapter_chain") < pass_names.index("after_response_field_cache_extraction")
    assert pass_names.index("after_response_adapter_chain") < pass_names.index("parsed_native_unified_response")
    usage_entry = next(entry for entry in trace if entry["pass_name"] == "usage_accounting_summary")
    assert usage_entry["data"]["usage"]["input_tokens"] == 99  # adapted wire usage won
