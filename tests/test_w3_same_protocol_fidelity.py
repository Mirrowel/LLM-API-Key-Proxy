# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""W3 acceptance fixtures: same-protocol fidelity (D4 raw fast path).

- Same-protocol requests transport the ORIGINAL client payload byte-for-byte
  (unknown extensions, encrypted reasoning, filenames all survive).
- Same-protocol responses pass through raw when no response adapters run.
- Every deviation from the raw basis is a traced overlay
  (context.request_transport_overlays + transaction trace).
- Encrypted reasoning survives Responses continuation round-trips.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from rotator_library.native_provider import NativeProviderContext, NativeProviderExecutor
from rotator_library.protocols import get_protocol
from rotator_library.protocols.types import ProtocolContext


class RecordingTransport:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.payload: dict | None = None

    async def post_json(self, endpoint: str, *, headers: dict, payload: dict) -> dict:
        self.payload = payload
        return self.response

    async def stream_json_lines(self, endpoint: str, *, headers: dict, payload: dict):
        yield {}


def _context(protocol: str, *, raw_client_request=None, model="provider/model-a", field_cache_rules=()) -> NativeProviderContext:
    operation = {
        "openai_chat": "chat",
        "anthropic_messages": "messages",
        "responses": "responses",
        "gemini": "generate",
    }[protocol]
    return NativeProviderContext(
        provider="provider",
        model=model,
        protocol_name=protocol,
        input_protocol_name=protocol,
        client_protocol_name=protocol,
        endpoint="https://example.test/api",
        operation=operation,
        raw_client_request=dict(raw_client_request) if raw_client_request is not None else None,
        field_cache_rules=tuple(field_cache_rules),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocol",
    ["openai_chat", "anthropic_messages", "responses", "gemini"],
)
async def test_same_protocol_request_transports_original_payload(protocol: str) -> None:
    payload = _exotic_payload(protocol)
    transport = RecordingTransport({"ok": True})
    context = _context(protocol, raw_client_request=payload)

    await NativeProviderExecutor().execute(payload, context, transport)

    assert transport.payload == payload
    assert context.request_transport_overlays == []
    assert transport.payload is not payload  # deep-copied, never mutated in place


@pytest.mark.asyncio
async def test_same_protocol_response_passes_through_raw() -> None:
    protocol = "responses"
    payload = _exotic_payload(protocol)
    response = {
        "id": "resp_9",
        "object": "response",
        "status": "completed",
        "model": "provider/model-a",
        "output": [
            {
                "id": "rs_0",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "hidden"}],
                "encrypted_content": "ENC[opaque-bytes]",
                "unknown_extension": {"future": True},
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "vendor_extension": {"kept": True},
    }
    transport = RecordingTransport(response)
    context = _context(protocol, raw_client_request=payload)

    result = await NativeProviderExecutor().execute(payload, context, transport)

    # Byte-for-byte: unknown extensions and encrypted reasoning survive the
    # round trip without canonical rebuild.
    assert result == response
    assert result["vendor_extension"] == {"kept": True}
    assert result["output"][0]["encrypted_content"] == "ENC[opaque-bytes]"


@pytest.mark.asyncio
async def test_model_overlay_is_traced_on_raw_path() -> None:
    payload = {
        "model": "client-visible-name",
        "messages": [{"role": "user", "content": "hi"}],
    }
    transport = RecordingTransport({"choices": [], "usage": {"prompt_tokens": 1}})
    context = _context("openai_chat", raw_client_request=payload, model="native-alias")

    await NativeProviderExecutor().execute(payload, context, transport)

    assert transport.payload["model"] == "native-alias"
    assert context.request_transport_overlays == [
        {"field": "model", "from": "client-visible-name", "to": "native-alias"}
    ]


@pytest.mark.asyncio
async def test_unified_state_injection_forces_traced_rebuild() -> None:
    from rotator_library.field_cache import FieldCacheInjection, FieldCacheRule

    payload = {"model": "provider/model-a", "input": "hello"}
    transport = RecordingTransport(_responses_ok("rebuilt-id"))
    context = _context(
        "responses",
        raw_client_request=payload,
        field_cache_rules=(
            FieldCacheRule(
                name="continuation",
                source="response",
                path="id",
                mode="last",
                inject=FieldCacheInjection(target="unified_request", path="previous_response_id"),
                allow_missing_session=True,
                scope=("provider", "model", "credential", "session"),
                metadata={"provider_continuation": True},
            ),
        ),
    )
    context.session_id = "session-1"
    context.credential_id = "credential-1"

    executor = NativeProviderExecutor()
    first = await executor.execute(payload, context, RecordingTransport(_responses_ok("resp_parent")))
    assert first["id"] == "resp_parent"

    await executor.execute(payload, context, transport)

    assert context.request_transport_overlays == [
        {"kind": "canonical_rebuild", "reason": "unified_state_injection"}
    ]
    assert transport.payload["previous_response_id"] == "resp_parent"


@pytest.mark.asyncio
async def test_attempt_mutations_force_traced_rebuild() -> None:
    payload = {"model": "provider/model-a", "messages": [{"role": "user", "content": "original"}]}
    mutated = {"model": "provider/model-a", "messages": [{"role": "user", "content": "mutated"}]}
    transport = RecordingTransport({"choices": [], "usage": {"prompt_tokens": 1}})
    context = _context("openai_chat", raw_client_request=payload)

    # The attempt payload (first argument) diverges from the pristine wire
    # payload: the canonical rebuild carries the mutation, raw cannot.
    await NativeProviderExecutor().execute(mutated, context, transport)

    assert context.request_transport_overlays == [{"kind": "canonical_rebuild", "reason": "payload_divergence"}]
    assert transport.payload["messages"][0]["content"] == "mutated"


@pytest.mark.asyncio
async def test_generation_param_mutation_is_never_silently_dropped() -> None:
    payload = {
        "model": "provider/model-a",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.5,
    }
    mutated = {
        "model": "provider/model-a",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.9,
    }
    transport = RecordingTransport({"choices": [], "usage": {"prompt_tokens": 1}})
    context = _context("openai_chat", raw_client_request=payload)

    await NativeProviderExecutor().execute(mutated, context, transport)

    # The gate must see generation-param divergence: rebuild carries 0.9 and
    # the deviation is traced — never silently dropped on the raw basis.
    assert context.request_transport_overlays == [{"kind": "canonical_rebuild", "reason": "payload_divergence"}]
    assert transport.payload["temperature"] == 0.9


@pytest.mark.asyncio
async def test_interleaved_system_messages_keep_raw_path() -> None:
    payload = {
        "model": "provider/model-a",
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "system", "content": "mid-conversation instruction"},
            {"role": "user", "content": "q2"},
        ],
    }
    transport = RecordingTransport({"choices": [], "usage": {"prompt_tokens": 1}})
    context = _context("openai_chat", raw_client_request=payload)

    await NativeProviderExecutor().execute(deepcopy(payload), context, transport)

    # Legal interleaved system messages are source-native: the raw basis
    # preserves the exact order (no instruction hoisting on the diagonal) and
    # no overlay is recorded.
    assert transport.payload == payload
    assert context.request_transport_overlays == []


@pytest.mark.asyncio
async def test_executor_wires_protocol_request_as_raw_basis() -> None:
    from rotator_library.client.executor import RequestExecutor
    from rotator_library.core.types import RequestContext

    class _Plugin:
        def get_protocol_name(self, model):
            return "openai_chat"

        def get_native_endpoint(self, *, model, operation):
            return "https://example.test/v1/chat/completions"

        def get_native_headers(self, credential, *, model, operation):
            return {}

    executor = RequestExecutor.__new__(RequestExecutor)
    context = RequestContext(
        model="provider/model-a",
        provider="provider",
        kwargs={"model": "provider/model-a", "messages": []},
        streaming=False,
        credentials=["cred-1"],
        deadline=9999999999.0,
        protocol_request={"model": "provider/model-a", "messages": [{"role": "user", "content": "hi"}], "future": True},
        input_protocol_name="openai_chat",
    )
    native_context = executor._build_native_provider_context(
        "provider", "provider/model-a", _Plugin(), "secret", "cred-1", context, None
    )
    assert native_context.raw_client_request == context.protocol_request

    context.input_protocol_name = "anthropic_messages"
    cross_context = executor._build_native_provider_context(
        "provider", "provider/model-a", _Plugin(), "secret", "cred-1", context, None
    )
    assert cross_context.raw_client_request is None


@pytest.mark.asyncio
async def test_cross_protocol_never_uses_raw_basis() -> None:
    payload = {"model": "provider/model-a", "messages": [{"role": "user", "content": "hi"}]}
    transport = RecordingTransport({"choices": [], "usage": {"prompt_tokens": 1}})
    context = _context("openai_chat", raw_client_request=payload)
    context.protocol_name = "anthropic_messages"
    context.operation = "messages"

    await NativeProviderExecutor().execute(payload, context, transport)

    assert context.request_transport_overlays == [{"kind": "canonical_rebuild", "reason": "cross_protocol"}]
    assert "messages" in transport.payload or "system" in transport.payload


def test_custom_tool_call_replays_verbatim_and_keeps_native_spelling() -> None:
    responses = get_protocol("responses")
    request = {
        "model": "model-a",
        "input": [
            {"role": "user", "content": "hi"},
            {"type": "custom_tool_call", "id": "ctc_0", "call_id": "call_9", "name": "extract", "input": "RAW-INPUT", "status": "completed"},
            {"type": "custom_tool_output", "call_id": "call_9", "output": "done"},
        ],
    }
    parsed = responses.parse_request(request)
    ctx = ProtocolContext(
        source_protocol="responses",
        target_protocol="responses",
        input_protocol="responses",
        client_protocol="responses",
    )
    built = responses.build_request(parsed, ctx)
    calls = [item for item in built["input"] if str(item.get("type", "")).endswith("tool_call")]
    assert calls, "custom tool call must survive the round trip"
    call = calls[0]
    # Verbatim identity: native spelling, native input member, and NEVER a
    # hybrid function_call+input shape (strict-param 400 upstream).
    assert call["type"] == "custom_tool_call"
    assert call["input"] == "RAW-INPUT"
    assert "arguments" not in call


def test_encrypted_reasoning_survives_responses_round_trip() -> None:
    responses = get_protocol("responses")
    request = {
        "model": "model-a",
        "input": [
            {"role": "user", "content": "think"},
            {
                "type": "reasoning",
                "id": "rs_0",
                "summary": [{"type": "summary_text", "text": "because"}],
                "encrypted_content": "ENC[reasoning-1]",
            },
            {"type": "function_call", "id": "fc_0", "call_id": "call_1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": "42"},
        ],
    }
    parsed = responses.parse_request(request)
    reasoning_blocks = [block for m in parsed.messages if m.role == "assistant" for block in m.content if block.reasoning]
    assert reasoning_blocks, "reasoning item must parse into the canonical model"
    assert reasoning_blocks[0].reasoning.encrypted_content == "ENC[reasoning-1]"

    ctx = ProtocolContext(
        source_protocol="responses",
        target_protocol="responses",
        input_protocol="responses",
        client_protocol="responses",
    )
    built = responses.build_request(parsed, ctx)
    reasoning_items = [item for item in built["input"] if item.get("type") == "reasoning"]
    assert reasoning_items
    assert reasoning_items[0]["encrypted_content"] == "ENC[reasoning-1]"


def _exotic_payload(protocol: str) -> dict:
    if protocol == "openai_chat":
        return {
            "model": "provider/model-a",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "file", "file": {"filename": "report.pdf", "file_data": "data:application/pdf;base64,AA=="}, "unknown_flag": True}]}],
            "vendor_extension": {"keep": "me"},
            "future_field": [1, 2, 3],
        }
    if protocol == "anthropic_messages":
        return {
            "model": "provider/model-a",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "AA==", "name": "report.pdf"}, "citations": {"enabled": True}, "future_key": 7}]}],
            "anthropic_beta": ["future-feature-v2"],
        }
    if protocol == "responses":
        return {
            "model": "provider/model-a",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}, {"type": "input_file", "file_id": "file_abc", "filename": "report.pdf"}]},
                {"type": "reasoning", "id": "rs_prev", "summary": [{"type": "summary_text", "text": "prior"}], "encrypted_content": "ENC[prior]"},
            ],
            "reasoning": {"effort": "high"},
            "text": {"verbosity": "low", "future": {"nested": True}},
        }
    return {
        "contents": [{"role": "user", "parts": [{"text": "hi"}, {"fileData": {"mimeType": "application/pdf", "fileUri": "https://files.test/report.pdf"}}]}],
        "generationConfig": {"temperature": 0.2, "futureField": {"kept": True}},
    }


def _responses_ok(response_id: str) -> dict:
    return {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "model": "provider/model-a",
        "output": [{"id": "msg_0", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "ok"}]}],
    }
