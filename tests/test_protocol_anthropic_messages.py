from __future__ import annotations

import json

from rotator_library.protocols import get_protocol, list_protocols


def test_anthropic_build_uses_mutated_unified_thinking_signature() -> None:
    adapter = get_protocol("anthropic_messages")
    raw = {
        "model": "anthropic/claude-test",
        "messages": [{"role": "assistant", "content": [{"type": "thinking", "thinking": "before", "signature": "sig_1"}]}],
    }

    unified = adapter.parse_request(raw)
    unified.messages[0].content[0].reasoning.text = "after"
    rebuilt = adapter.build_request(unified)

    assert rebuilt["messages"][0]["content"][0]["thinking"] == "after"
    assert rebuilt["messages"][0]["content"][0]["signature"] == "sig_1"


def test_anthropic_messages_protocol_is_discovered_with_aliases() -> None:
    assert "anthropic_messages" in list_protocols()
    assert get_protocol("anthropic") is get_protocol("anthropic_messages")
    assert get_protocol("messages") is get_protocol("anthropic_messages")


def test_anthropic_request_round_trip_preserves_thinking_tools_and_cache_metadata() -> None:
    adapter = get_protocol("anthropic_messages")
    raw = {
        "model": "anthropic/claude-test",
        "system": [{"type": "text", "text": "system"}],
        "max_tokens": 100,
        "stream": True,
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "worked", "signature": "sig_1"},
                    {"type": "text", "text": "answer"},
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}},
                ],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "result"}]},
        ],
        "tools": [{"name": "lookup", "description": "Lookup", "input_schema": {"type": "object"}}],
        "vendor_extension": {"kept": True},
    }

    unified = adapter.parse_request(raw)
    rebuilt = adapter.build_request(unified)

    assert unified.model == "anthropic/claude-test"
    assert unified.system[0].text == "system"
    assert unified.messages[1].reasoning[0].signature == "sig_1"
    assert unified.messages[1].tool_calls[0].name == "lookup"
    assert unified.messages[2].content[0].tool_result.tool_call_id == "toolu_1"
    assert unified.tools[0].input_schema == {"type": "object"}
    assert rebuilt["vendor_extension"] == {"kept": True}
    assert isinstance(rebuilt["system"], list)
    assert rebuilt["system"][0]["text"] == "system"
    assert rebuilt["messages"][1]["content"][0]["signature"] == "sig_1"


def test_anthropic_system_preserves_block_metadata() -> None:
    adapter = get_protocol("anthropic_messages")
    raw = {
        "model": "anthropic/claude-test",
        "system": [{"type": "text", "text": "system", "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": "hello"}],
    }

    unified = adapter.parse_request(raw)
    unified.system[0].text = "updated"
    rebuilt = adapter.build_request(unified)

    assert rebuilt["system"] == [{"type": "text", "text": "updated", "cache_control": {"type": "ephemeral"}}]


def test_anthropic_response_extracts_content_and_usage() -> None:
    adapter = get_protocol("anthropic_messages")
    raw = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "anthropic/claude-test",
        "content": [
            {"type": "redacted_thinking", "signature": "sig_2"},
            {"type": "text", "text": "answer"},
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 3,
        },
    }

    unified = adapter.parse_response(raw)

    assert unified.id == "msg_1"
    assert unified.messages[0].content[1].text == "answer"
    assert unified.messages[0].reasoning[0].signature == "sig_2"
    assert unified.messages[0].reasoning[0].redacted is True
    assert unified.usage is not None
    assert unified.usage.input_tokens == 10
    assert unified.usage.output_tokens == 5
    assert unified.usage.cache_write_tokens == 2
    assert unified.usage.cache_read_tokens == 3


def test_anthropic_stream_event_parses_text_delta_and_usage() -> None:
    adapter = get_protocol("anthropic_messages")
    text_event = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}
    usage_event = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}}

    parsed_text = adapter.parse_stream_event(f"data: {json.dumps(text_event)}\n\n")
    parsed_usage = adapter.parse_stream_event(usage_event)

    assert parsed_text.type == "content_block_delta"
    assert parsed_text.delta is not None
    assert parsed_text.delta.content[0].text == "Hi"
    assert parsed_usage.usage is not None
    assert parsed_usage.usage.output_tokens == 7
