from __future__ import annotations

import json
from typing import Any

import pytest

from rotator_library.native_provider import NativeProviderContext, NativeProviderExecutor
from rotator_library.providers.antigravity_provider import AntigravityProvider
from rotator_library.providers.claude_code_provider import ClaudeCodeProvider
from rotator_library.providers.codex_provider import CodexProvider
from rotator_library.providers.copilot_provider import CopilotProvider


OUTPUT_PROTOCOLS = ("openai_chat", "anthropic_messages", "responses", "gemini")
STATEFUL_PROVIDERS = (AntigravityProvider(), ClaudeCodeProvider(), CodexProvider())


class RecordingResponseTransport:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def post_json(self, endpoint, *, headers, payload):
        self.requests.append(payload)
        return self.responses.pop(0)


class RecordingStreamTransport:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = chunks
        self.requests: list[dict[str, Any]] = []

    async def stream_json_lines(self, endpoint, *, headers, payload):
        self.requests.append(payload)
        for chunk in self.chunks:
            yield chunk


def _context(
    provider,
    *,
    output_protocol: str = "openai_chat",
    credential_id: str = "credential-a",
    session_id: str = "session-a",
    model: str = "model-a",
    stream: bool = False,
    disable_provider_continuation: bool = False,
) -> NativeProviderContext:
    operation = provider.get_native_operation(model, None, stream=stream)
    return NativeProviderContext(
        provider=provider.provider_env_name,
        model=model,
        protocol_name=provider.get_protocol_name(model),
        input_protocol_name=provider.get_protocol_name(model),
        output_protocol_name=output_protocol,
        endpoint="https://provider.test/generate",
        operation=operation,
        credential_id=credential_id,
        session_id=session_id,
        scope_key="scope-a",
        adapter_names=provider.get_adapter_names(model),
        adapter_config=provider.get_adapter_config(model),
        field_cache_rules=provider.get_field_cache_rules(model),
        request_preparer=provider.prepare_native_request,
        request_validator=getattr(provider, "validate_request", None),
        metadata={"disable_provider_continuation": disable_provider_continuation},
    )


def _request(provider) -> dict[str, Any]:
    if isinstance(provider, AntigravityProvider):
        return {"model": "model-a", "contents": [{"role": "user", "parts": [{"text": "hello"}]}]}
    if isinstance(provider, ClaudeCodeProvider):
        return {"model": "model-a", "max_tokens": 32, "messages": [{"role": "user", "content": "hello"}]}
    if isinstance(provider, CodexProvider):
        return {"model": "model-a", "input": "hello"}
    return {"model": "model-a", "messages": [{"role": "user", "content": "hello"}]}


def _response(provider, state: str) -> dict[str, Any]:
    if isinstance(provider, AntigravityProvider):
        return {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": "private thought", "thought": True, "thoughtSignature": state}, {"text": "answer"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        }
    if isinstance(provider, ClaudeCodeProvider):
        return {
            "id": "msg_provider",
            "type": "message",
            "role": "assistant",
            "model": "model-a",
            "content": [{"type": "thinking", "thinking": "private thought", "signature": state}, {"type": "text", "text": "answer"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    if isinstance(provider, CodexProvider):
        return {
            "id": state,
            "object": "response",
            "status": "completed",
            "model": "model-a",
            "output": [{"id": "msg_0", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "answer"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }
    return {"id": "chat_1", "model": "model-a", "choices": [{"message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}]}


def _stream_chunks(provider, state: str) -> list[Any]:
    if isinstance(provider, AntigravityProvider):
        return [
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "private stream thought", "thought": True, "thoughtSignature": state}, {"text": "stream answer"}]}, "finishReason": "STOP"}]},
            "[DONE]",
        ]
    if isinstance(provider, ClaudeCodeProvider):
        return [
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": state}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "stream answer"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
    return [
        {"type": "response.output_text.delta", "item_id": "msg_0", "output_index": 0, "content_index": 0, "delta": "stream answer"},
        {"type": "response.completed", "response": _response(provider, state)},
        "[DONE]",
    ]


def _injected_state(provider, payload: dict[str, Any]) -> Any:
    if isinstance(provider, AntigravityProvider):
        return payload["request"]["metadata"]["thoughtSignatures"]
    if isinstance(provider, ClaudeCodeProvider):
        return payload["metadata"]["thinking_signatures"]
    return payload["previous_response_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", STATEFUL_PROVIDERS, ids=lambda provider: provider.provider_env_name)
@pytest.mark.parametrize("output_protocol", OUTPUT_PROTOCOLS)
async def test_real_provider_non_stream_state_is_output_protocol_independent(provider, output_protocol: str) -> None:
    executor = NativeProviderExecutor()
    transport = RecordingResponseTransport([_response(provider, "state-one"), _response(provider, "state-two")])
    context = _context(provider, output_protocol=output_protocol)

    first = await executor.execute(_request(provider), context, transport)
    await executor.execute(_request(provider), context, transport)

    assert _injected_state(provider, transport.requests[1]) == (["state-one"] if not isinstance(provider, CodexProvider) else "state-one")
    serialized = json.dumps(first)
    if not isinstance(provider, CodexProvider):
        assert "state-one" not in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", STATEFUL_PROVIDERS, ids=lambda provider: provider.provider_env_name)
async def test_real_provider_mixed_stream_and_non_stream_share_latest_logical_state(provider) -> None:
    executor = NativeProviderExecutor()
    first_transport = RecordingResponseTransport([_response(provider, "non-stream-state")])
    await executor.execute(_request(provider), _context(provider), first_transport)

    stream_transport = RecordingStreamTransport(_stream_chunks(provider, "stream-state"))
    stream_output = [frame async for frame in executor.stream(_request(provider), _context(provider, stream=True), stream_transport)]

    final_transport = RecordingResponseTransport([_response(provider, "final-state")])
    await executor.execute(_request(provider), _context(provider), final_transport)

    expected = "stream-state" if isinstance(provider, CodexProvider) else ["non-stream-state", "stream-state"]
    assert _injected_state(provider, final_transport.requests[0]) == expected
    assert "stream-state" not in "".join(stream_output)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", STATEFUL_PROVIDERS, ids=lambda provider: provider.provider_env_name)
@pytest.mark.parametrize(
    "isolated_context",
    (
        {"credential_id": "credential-b"},
        {"session_id": "session-b"},
        {"model": "model-b"},
    ),
)
async def test_real_provider_state_isolated_by_model_credential_and_session(provider, isolated_context) -> None:
    executor = NativeProviderExecutor()
    await executor.execute(
        _request(provider),
        _context(provider),
        RecordingResponseTransport([_response(provider, "private-state")]),
    )
    isolated_transport = RecordingResponseTransport([_response(provider, "other-state")])

    await executor.execute(
        _request(provider),
        _context(provider, **isolated_context),
        isolated_transport,
    )

    with pytest.raises((KeyError, TypeError)):
        _injected_state(provider, isolated_transport.requests[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", STATEFUL_PROVIDERS, ids=lambda provider: provider.provider_env_name)
@pytest.mark.parametrize("missing_scope", ({"credential_id": None}, {"session_id": None}))
async def test_real_provider_state_skips_when_required_scope_is_missing(provider, missing_scope) -> None:
    executor = NativeProviderExecutor()
    context = _context(provider, **missing_scope)
    transport = RecordingResponseTransport([
        _response(provider, "private-state"),
        _response(provider, "second-state"),
    ])

    await executor.execute(_request(provider), context, transport)
    await executor.execute(_request(provider), context, transport)

    with pytest.raises((KeyError, TypeError)):
        _injected_state(provider, transport.requests[1])


@pytest.mark.asyncio
async def test_codex_local_lineage_disables_both_continuation_sources() -> None:
    provider = CodexProvider()
    executor = NativeProviderExecutor()
    await executor.execute(
        _request(provider),
        _context(provider),
        RecordingResponseTransport([_response(provider, "resp_non_stream")]),
    )
    _ = [
        frame
        async for frame in executor.stream(
            _request(provider),
            _context(provider, stream=True),
            RecordingStreamTransport(_stream_chunks(provider, "resp_stream")),
        )
    ]
    suppressed = RecordingResponseTransport([_response(provider, "resp_final")])

    await executor.execute(
        _request(provider),
        _context(provider, disable_provider_continuation=True),
        suppressed,
    )

    assert "previous_response_id" not in suppressed.requests[0]


@pytest.mark.asyncio
async def test_copilot_native_contract_has_no_invented_provider_state() -> None:
    provider = CopilotProvider()
    executor = NativeProviderExecutor()
    transport = RecordingResponseTransport([_response(provider, "unused"), _response(provider, "unused")])
    context = _context(provider)

    await executor.execute(_request(provider), context, transport)
    await executor.execute(_request(provider), context, transport)

    assert transport.requests[1] == transport.requests[0]
