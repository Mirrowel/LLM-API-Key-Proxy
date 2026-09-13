# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G10 Phase B disclosure pins: conversion notes are record-only.

The client-visible ``x-proxy-conversion`` body key is retired. Every
deliberate conversion (drop, merge, approximation, substitution, fallback,
stream repair, overlay) lands in the transaction record's change log —
never on the wire, never on the console.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, AsyncIterator, List

from rotator_library.client.anthropic import AnthropicHandler
from rotator_library.client.gemini import GeminiHandler
from rotator_library.client.stream_ops import NeutralStreamPipeline
from rotator_library.native_provider.context import NativeProviderContext
from rotator_library.native_provider.executor import NativeProviderExecutor
from rotator_library.native_provider.http import NativeHTTPTransport
from rotator_library.protocols import get_protocol
from rotator_library.protocols.canonical import resolve_tool_result_names
from rotator_library.protocols.operation import OPERATION_COUNT_TOKENS
from rotator_library.protocols.types import (
    ContentBlock,
    ConversionWarning,
    ProtocolContext,
    ToolCall,
    ToolResult,
    UnifiedMessage,
    UnifiedResponse,
    UnifiedStreamEvent,
)
from rotator_library.streaming.relay import RelayStreamItem, StreamRepairState
from rotator_library.transaction import TransactionWriter
from rotator_library.transaction_logger import TransactionLogger


def _ctx(source: str, target: str) -> ProtocolContext:
    return ProtocolContext(
        source_protocol=source,
        target_protocol=target,
        input_protocol=source,
        provider_protocol=target,
        client_protocol=source,
    )


def _logger(tmp_path, monkeypatch, protocol: str = "openai_chat") -> TransactionLogger:
    writer = TransactionWriter.instance()
    monkeypatch.setattr(writer, "submit_sealed", lambda envelope, *, filename: None)
    return TransactionLogger("testprov", "testprov/model-x", parent_dir=tmp_path, protocol=protocol)


def _codes(logger: TransactionLogger) -> list[str]:
    return [event.code for event in logger._record.change_log]


def _stages(logger: TransactionLogger) -> list[str]:
    return [event.stage for event in logger._record.change_log]


def _kinds(logger: TransactionLogger) -> list[str]:
    return [event.kind for event in logger._record.change_log]


async def _agen(items: List[Any]) -> AsyncIterator[Any]:
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Negative: no protocol's format_response ever carries the retired body key
# ---------------------------------------------------------------------------

_PROTOCOLS = ("openai_chat", "responses", "anthropic_messages", "gemini")

_RESPONSE_FIXTURES = {
    "openai_chat": {
        "id": "c1",
        "object": "chat.completion",
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "responses": {
        "id": "r1",
        "model": "m",
        "status": "completed",
        "output": [
            {"type": "message", "id": "msg_1", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}
        ],
    },
    "anthropic_messages": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "hi"}],
    },
    "gemini": {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    },
}


def test_no_response_payload_ever_carries_the_retired_summary_key() -> None:
    for source in _PROTOCOLS:
        unified = get_protocol(source).parse_response(_RESPONSE_FIXTURES[source], _ctx(source, source))
        # Seed a warning so any residual summary writer would have something
        # to render — absence must be structural, not merely data-dependent.
        unified.warnings.append(
            ConversionWarning(
                code="media_dropped",
                message="seeded",
                field="content",
                source_protocol=source,
                target_protocol="openai_chat",
            )
        )
        for target in _PROTOCOLS:
            try:
                payload = get_protocol(target).format_response(unified, _ctx(source, target))
            except Exception:
                # Some cross-protocol shapes are honestly rejected; the
                # negative still holds for every payload that IS produced.
                continue
            assert "x-proxy-conversion" not in payload, (source, target)
            assert "_proxy_warnings" not in payload, (source, target)


def test_conversion_warnings_never_hit_the_console(tmp_path, monkeypatch, caplog) -> None:
    logger = _logger(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO)
    logger.log_conversion_warnings(
        [
            ConversionWarning(
                code="media_dropped",
                message="audio output has no Anthropic representation; dropped",
                field="content[audio]",
                source_protocol="gemini",
                target_protocol="anthropic_messages",
            )
        ]
    )
    assert _codes(logger) == ["media_dropped"]
    assert "media_dropped" not in caplog.text
    assert "conversion warning" not in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.INFO and "conversion" in record.getMessage().lower()]


# ---------------------------------------------------------------------------
# Matrix: every warning/mutation family lands in the change log
# ---------------------------------------------------------------------------


def test_media_dropped_family_lands_in_change_log(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path, monkeypatch)
    gemini = get_protocol("gemini")
    response = gemini.parse_response(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"inlineData": {"mimeType": "audio/wav", "data": "UklGRg=="}}]},
                    "finishReason": "STOP",
                }
            ]
        },
        _ctx("gemini", "anthropic_messages"),
    )
    get_protocol("anthropic_messages").format_response(response, _ctx("gemini", "anthropic_messages"))
    logger.log_conversion_warnings(response.warnings)
    assert "media_dropped" in _codes(logger)
    assert all(kind == "conversion_warning" for kind in _kinds(logger))


def test_unsupported_optional_control_family_lands_in_change_log(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path, monkeypatch)
    chat = get_protocol("openai_chat")
    unified = chat.parse_request(
        {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], "seed": 42},
        _ctx("openai_chat", "anthropic_messages"),
    )
    get_protocol("anthropic_messages").build_request(unified, _ctx("openai_chat", "anthropic_messages"))
    logger.log_conversion_warnings(unified.warnings)
    assert "unsupported_optional_control" in _codes(logger)


def test_reasoning_approximation_family_lands_in_change_log(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path, monkeypatch)
    chat = get_protocol("openai_chat")
    unified = chat.parse_request(
        {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high"},
        _ctx("openai_chat", "anthropic_messages"),
    )
    get_protocol("anthropic_messages").build_request(unified, _ctx("openai_chat", "anthropic_messages"))
    logger.log_conversion_warnings(unified.warnings)
    assert "reasoning_effort_approximated" in _codes(logger)


def test_stop_reason_approximated_family_lands_in_change_log(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path, monkeypatch)
    response = UnifiedResponse(
        operation="chat",
        stop_reason="pause",
        source_protocol="anthropic_messages",
        messages=[UnifiedMessage(role="assistant", content=[ContentBlock(type="text", text="x")])],
    )
    get_protocol("openai_chat").format_response(response, _ctx("anthropic_messages", "openai_chat"))
    logger.log_conversion_warnings(response.warnings)
    assert "stop_reason_approximated" in _codes(logger)


def test_instructions_merged_family_lands_in_change_log(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path, monkeypatch)
    chat = get_protocol("openai_chat")
    # Chat targets merge multiple instruction text blocks into one content
    # field (a single block stays silent).
    unified = chat.parse_request(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                {"role": "user", "content": "hi"},
            ],
        },
        _ctx("openai_chat", "openai_chat"),
    )
    get_protocol("openai_chat").build_request(unified, _ctx("responses", "openai_chat"))
    logger.log_conversion_warnings(unified.warnings)
    assert "instructions_merged" in _codes(logger)
    # A single instruction block stays silent (no fabricated merge warning).
    single = chat.parse_request(
        {"model": "m", "messages": [{"role": "system", "content": "one"}, {"role": "user", "content": "hi"}]},
        _ctx("openai_chat", "openai_chat"),
    )
    get_protocol("openai_chat").build_request(single, _ctx("responses", "openai_chat"))
    assert "instructions_merged" not in [w.code for w in single.warnings]


def test_tool_result_id_synthesized_family_lands_in_change_log(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path, monkeypatch)
    messages = [
        UnifiedMessage(role="assistant", tool_calls=[ToolCall(id=None, name="lookup", arguments={"q": "x"})]),
        UnifiedMessage(
            role="tool",
            content=[ContentBlock(type="tool_result", tool_result=ToolResult(tool_call_id=None, name="lookup", content="ok"))],
        ),
    ]
    warnings: list = []
    resolve_tool_result_names(messages, warnings)
    logger.log_conversion_warnings(warnings)
    assert "tool_result_id_synthesized" in _codes(logger)


def test_embeddings_option_warning_family_lands_in_change_log(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path, monkeypatch, protocol="openai_embeddings")
    adapter = get_protocol("openai_embeddings")
    unified = adapter.parse_request({"model": "m", "input": "one", "custom": True})
    assert any(w.code == "unsupported_optional_control" for w in unified.warnings)
    response = adapter.parse_response({"model": "m", "data": [], "usage": {"prompt_tokens": 1, "total_tokens": 1}})
    response.warnings = list(unified.warnings)
    payload = adapter.format_response(response)
    assert "_proxy_warnings" in payload
    logger.log_conversion_warnings(payload.pop("_proxy_warnings"))
    assert "unsupported_optional_control" in _codes(logger)


def test_count_tokens_private_channel_drains_to_change_log(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path, monkeypatch, protocol="anthropic_messages")
    anthropic = get_protocol("anthropic_messages")
    response = anthropic.parse_response(
        {"input_tokens": 12},
        ProtocolContext(
            source_protocol="anthropic_messages",
            target_protocol="anthropic_messages",
            client_protocol="anthropic_messages",
            provider_options={"operation": OPERATION_COUNT_TOKENS},
        ),
    )
    response.warnings.append(
        ConversionWarning(
            code="stop_reason_approximated",
            message="count-token response drop",
            field="stop_reason",
            source_protocol="anthropic_messages",
            target_protocol=None,
        )
    )
    payload = anthropic.format_response(response)
    assert "_proxy_warnings" in payload
    logger.log_conversion_warnings(payload.pop("_proxy_warnings"), stage="count_tokens")
    assert "stop_reason_approximated" in _codes(logger)
    assert "count_tokens" in _stages(logger)


def test_stream_repair_family_lands_in_change_log(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path, monkeypatch)
    ctx = _ctx("openai_chat", "openai_chat")
    pipeline = NeutralStreamPipeline(
        client_protocol_name="openai_chat",
        protocol_context=ctx,
        model="m",
        transaction_logger=logger,
    )
    events = [
        UnifiedStreamEvent(
            type="message.delta",
            source_protocol="openai_chat",
            native_type="message.delta",
            delta=UnifiedMessage(role="assistant", content=[ContentBlock(type="text", text="hi")]),
        )
    ]

    async def _drive() -> None:
        async for _ in pipeline.run(_agen(events)):
            pass

    import asyncio

    asyncio.run(_drive())
    assert "repair" in _codes(logger)
    assert "synthesized finish reason" in [event.detail for event in logger._record.change_log]


def test_stream_relay_disengage_family_lands_in_change_log(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path, monkeypatch)
    ctx = _ctx("openai_chat", "openai_chat")
    repair_state = StreamRepairState()
    repair_state.edited_by_hook = True
    pipeline = NeutralStreamPipeline(
        client_protocol_name="openai_chat",
        protocol_context=ctx,
        model="m",
        transaction_logger=logger,
        relay_eligible=True,
        repair_state=repair_state,
    )
    event = UnifiedStreamEvent(
        type="message.delta",
        source_protocol="openai_chat",
        native_type="message.delta",
        delta=UnifiedMessage(role="assistant", content=[ContentBlock(type="text", text="hi")]),
    )
    item = RelayStreamItem(events=[event], raw='data: {"x": 1}')

    async def _drive() -> None:
        async for _ in pipeline.run(_agen([item])):
            pass

    import asyncio

    asyncio.run(_drive())
    assert "relay_disengage" in _codes(logger)
    assert "byte-relay disengaged" in [event.detail for event in logger._record.change_log]


def test_overlay_family_lands_in_change_log(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path, monkeypatch)
    context = NativeProviderContext(provider="p", model="m", protocol_name="openai_chat", endpoint="https://x")
    context.request_transport_overlays = [
        {"kind": "hook_edit", "stage": "transport_basis_selected"},
        {"kind": "canonical_rebuild", "reason": "cross_protocol"},
    ]
    NativeProviderExecutor._record_transport_overlays(context, logger)
    assert _codes(logger).count("overlay") == 2
    assert _kinds(logger) == ["overlay", "overlay"]


def test_fallback_family_lands_in_change_log(tmp_path, monkeypatch) -> None:
    from rotator_library.client.executor import RequestExecutor

    logger = _logger(tmp_path, monkeypatch)
    context = SimpleNamespace(transaction_logger=logger, _litellm_fallback_warned=False)
    plugin = SimpleNamespace(get_protocol_name=lambda _model: "gemini")
    RequestExecutor._record_litellm_fallback_identity(
        None, context, "gemini_test", plugin, "gemini-test", stream=False
    )
    assert "fallback" in _codes(logger)


def test_routing_substitution_family_lands_in_change_log(tmp_path, monkeypatch) -> None:
    logger = _logger(tmp_path, monkeypatch, protocol="gemini")
    logger.log_runtime_event(
        "routing",
        "routing_substitution",
        "single-protocol conversion per default protocol priority",
        {"provider": "p", "client_protocol": "openai_chat", "protocol": "gemini"},
    )
    assert "routing_substitution" in _codes(logger)


# ---------------------------------------------------------------------------
# count_tokens facades drain the private channel before responding
# ---------------------------------------------------------------------------


class _FakeCountClient:
    def __init__(self, logger: TransactionLogger) -> None:
        self.enable_request_logging = True
        self._logger = logger
        self.calls: list[dict] = []

    async def agenerate(self, payload, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs.get("_request_context_callback")
        if callback is not None:
            callback(SimpleNamespace(transaction_logger=self._logger))
        return {
            "input_tokens": 3,
            "_proxy_warnings": [
                ConversionWarning(
                    code="usage_detail_dropped",
                    message="count-token usage detail dropped",
                    field="usage",
                    source_protocol="anthropic_messages",
                    target_protocol=None,
                )
            ],
        }


def test_anthropic_count_tokens_facade_drains_private_channel(tmp_path, monkeypatch) -> None:
    import asyncio

    logger = _logger(tmp_path, monkeypatch, protocol="anthropic_messages")
    handler = AnthropicHandler(_FakeCountClient(logger))
    result = asyncio.run(handler.count_tokens({"model": "testprov/model-x", "messages": []}))
    assert "_proxy_warnings" not in result
    assert "usage_detail_dropped" in _codes(logger)


def test_gemini_count_tokens_facade_drains_private_channel(tmp_path, monkeypatch) -> None:
    import asyncio

    logger = _logger(tmp_path, monkeypatch, protocol="gemini")
    handler = GeminiHandler(_FakeCountClient(logger))
    result = asyncio.run(handler.count_tokens({"contents": []}, model="testprov/model-x"))
    assert "_proxy_warnings" not in result
    assert "usage_detail_dropped" in _codes(logger)


# ---------------------------------------------------------------------------
# Native stream: request-side warnings are recorded (stream_request stage)
# ---------------------------------------------------------------------------


class _FakeStreamClient:
    def __init__(self, frames: list[str]) -> None:
        self._frames = frames

    def stream(self, method, endpoint, headers=None, json=None, **kwargs):
        lines: list[str] = []
        for frame in self._frames:
            lines.extend(frame.splitlines())

        class _Resp:
            status_code = 200

            def aiter_lines(self):
                async def _gen():
                    for line in lines:
                        yield line

                return _gen()

        class _Ctx:
            async def __aenter__(self):
                return _Resp()

            async def __aexit__(self, *args):
                return False

        return _Ctx()


def test_native_stream_request_warnings_are_recorded(tmp_path, monkeypatch) -> None:
    import asyncio

    logger = _logger(tmp_path, monkeypatch, protocol="gemini")
    context = NativeProviderContext(
        provider="gemini_test",
        model="gemini-test",
        protocol_name="gemini",
        endpoint="https://provider.example/v1beta/models/gemini-test:streamGenerateContent",
        input_protocol_name="openai_chat",
        client_protocol_name="gemini",
        operation="chat",
        headers={"Authorization": "Bearer test"},
        transaction_logger=logger,
    )
    raw_request = {
        "model": "gemini-test",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "ultra",
    }

    async def _drive() -> None:
        stream = NativeProviderExecutor().stream(
            raw_request, context, NativeHTTPTransport(_FakeStreamClient([]))
        )
        async for _ in stream:
            pass

    asyncio.run(_drive())
    assert "stream_request" in _stages(logger)
    assert "reasoning_effort_unknown" in _codes(logger)


def test_native_stream_strips_foreign_opaque_state_and_records_overlay(tmp_path, monkeypatch) -> None:
    import asyncio

    import rotator_library.protocols.opaque_strip as opaque_strip

    logger = _logger(tmp_path, monkeypatch)
    context = NativeProviderContext(
        provider="provider_b",
        model="m",
        protocol_name="openai_chat",
        endpoint="https://provider.example/v1/chat/completions",
        input_protocol_name="openai_chat",
        client_protocol_name="openai_chat",
        operation="chat",
        headers={"Authorization": "Bearer test"},
        metadata={"input_provider": "provider_a"},
        transaction_logger=logger,
    )
    calls: dict = {}

    def fake_strip(payload, protocol_name, *, mutate=True):
        calls["protocol"] = protocol_name
        return ["messages[0].extra_content.google.thought_signature"]

    monkeypatch.setattr(opaque_strip, "strip_foreign_opaque_state", fake_strip)
    raw_request = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}

    async def _drive() -> None:
        stream = NativeProviderExecutor().stream(
            raw_request, context, NativeHTTPTransport(_FakeStreamClient([]))
        )
        async for _ in stream:
            pass

    asyncio.run(_drive())
    assert calls.get("protocol") == "openai_chat"
    overlays = [event.value for event in logger._record.change_log if event.code == "overlay"]
    assert any(value and value.get("kind") == "foreign_bound_state_stripped" for value in overlays)
