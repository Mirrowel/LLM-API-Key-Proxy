from __future__ import annotations

import json

import pytest

import rotator_library.client.anthropic as anthropic_client_module
from rotator_library.client.anthropic import AnthropicHandler
from rotator_library.transaction_logger import TransactionLogger


def _trace_entries(log_dir):
    return [json.loads(line) for line in (log_dir / "transform_trace.jsonl").read_text(encoding="utf-8").splitlines()]


class FakeNativeProtocolClient:
    enable_request_logging = False

    def __init__(self) -> None:
        self.call = None

    async def agenerate(self, payload, *, input_protocol, **kwargs):
        self.call = {
            "payload": payload,
            "input_protocol": input_protocol,
        }
        if payload.get("stream"):
            async def stream():
                yield 'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_stream","type":"message","role":"assistant","content":[],"model":"claude-test","usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
                yield 'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
                yield 'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"native stream"}}\n\n'
                yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            return stream()
        return {
            "id": "msg_provider",
            "type": "message",
            "role": "assistant",
            "model": payload["model"],
            "content": [{"type": "text", "text": "native answer"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }


class FakeCountingClient:
    enable_request_logging = False

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def token_count(self, *, model, messages=None, text=None):
        self.calls.append({"model": model, "messages": messages, "text": text})
        return 3


@pytest.mark.asyncio
async def test_anthropic_handler_transports_raw_body_verbatim() -> None:
    client = FakeNativeProtocolClient()
    payload = {
        "model": "claude_code/claude-test",
        "max_tokens": 16,
        "system": "rule",
        "messages": [{"role": "user", "content": "hi"}],
        "unknown_extension_field": {"nested": True},
        "explicit_null": None,
    }

    response = await AnthropicHandler(client).messages(payload)

    assert client.call["input_protocol"] == "anthropic_messages"
    # The pristine payload is transported (D4): unknown fields and explicit
    # nulls survive to the runtime untouched.
    assert client.call["payload"] == payload
    assert response["content"][0]["text"] == "native answer"
    assert response["id"].startswith("msg_")


@pytest.mark.asyncio
async def test_anthropic_handler_uses_protocol_native_runtime_for_streams() -> None:
    client = FakeNativeProtocolClient()
    payload = {
        "model": "claude_code/claude-test",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    stream = await AnthropicHandler(client).messages(payload)
    output = "".join([chunk async for chunk in stream])

    assert client.call["input_protocol"] == "anthropic_messages"
    assert "native stream" in output


@pytest.mark.asyncio
async def test_anthropic_handler_traces_boundary_when_logging_enabled(tmp_path, monkeypatch) -> None:
    created: list[TransactionLogger] = []

    def logger_factory(provider, model, enabled=True, api_format="ant", parent_dir=None):
        logger = TransactionLogger(provider, model, enabled=enabled, api_format=api_format, parent_dir=tmp_path)
        created.append(logger)
        return logger

    monkeypatch.setattr(anthropic_client_module, "TransactionLogger", logger_factory)
    payload = {"model": "openai/gpt-test", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}

    client = FakeNativeProtocolClient()
    client.enable_request_logging = True
    response = await AnthropicHandler(client).messages(payload)

    assert response["content"][0]["text"] == "native answer"
    pass_names = [entry["pass_name"] for entry in _trace_entries(created[0].log_dir)]
    assert "anthropic_raw_request" in pass_names
    assert "anthropic_native_protocol_response" in pass_names
    assert "final_client_response" in pass_names


@pytest.mark.asyncio
async def test_anthropic_count_tokens_projects_through_chat_view() -> None:
    client = FakeCountingClient()
    payload = {
        "model": "openai/gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"name": "lookup", "description": "d", "input_schema": {"type": "object"}}],
    }

    result = await AnthropicHandler(client).count_tokens(payload)

    assert result == {"input_tokens": 6}  # messages + serialized tools
    assert len(client.calls) == 2
    assert client.calls[0]["messages"][0]["role"] == "user"
    assert "lookup" in client.calls[1]["text"]
