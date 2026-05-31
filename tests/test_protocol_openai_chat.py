from __future__ import annotations

import json

from rotator_library.protocols import get_protocol, list_protocols


def test_openai_chat_protocol_is_discovered_with_aliases() -> None:
    assert "openai_chat" in list_protocols()
    assert get_protocol("openai") is get_protocol("openai_chat")
    assert get_protocol("chat_completions") is get_protocol("openai_chat")


def test_litellm_fallback_protocol_preserves_raw_payload() -> None:
    adapter = get_protocol("litellm_fallback")
    raw = {"model": "custom/model", "messages": [{"role": "user", "content": "hi"}], "vendor_flag": True}

    unified = adapter.parse_request(raw)
    rebuilt = adapter.build_request(unified)

    assert unified.extra["messages"] == raw["messages"]
    assert rebuilt == raw


def test_openai_chat_request_round_trip_preserves_tools_and_reasoning() -> None:
    adapter = get_protocol("openai_chat")
    raw = {
        "model": "openai/gpt-test",
        "stream": True,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "developer", "content": "dev"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {
                "role": "assistant",
                "content": "thinking done",
                "reasoning_content": "internal chain",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup a value",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ],
        "vendor_extension": {"kept": True},
    }

    unified = adapter.parse_request(raw)
    rebuilt = adapter.build_request(unified)

    assert unified.model == "openai/gpt-test"
    assert unified.stream is True
    assert unified.generation_params["temperature"] == 0.2
    assert unified.messages[3].reasoning[0].text == "internal chain"
    assert unified.messages[3].tool_calls[0].name == "lookup"
    assert unified.tools[0].input_schema["properties"]["q"]["type"] == "string"
    assert rebuilt["vendor_extension"] == {"kept": True}
    assert rebuilt["messages"][3]["reasoning_content"] == "internal chain"


def test_openai_chat_build_uses_mutated_unified_fields_and_preserves_block_extras() -> None:
    adapter = get_protocol("openai_chat")
    raw = {
        "model": "openai/gpt-test",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "before", "cache_control": {"type": "ephemeral"}}],
            }
        ],
    }

    unified = adapter.parse_request(raw)
    unified.messages[0].content[0].text = "after"
    rebuilt = adapter.build_request(unified)

    assert isinstance(rebuilt["messages"][0]["content"], list)
    assert rebuilt["messages"][0]["content"][0]["text"] == "after"
    assert rebuilt["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_openai_chat_response_extracts_usage_cost_and_reasoning() -> None:
    adapter = get_protocol("openai_chat")
    raw = {
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "created": 123,
        "model": "openai/gpt-test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "reasoned",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 18,
            "prompt_tokens_details": {"cached_tokens": 3, "cache_creation_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 3},
            "cost_details": {"total_cost": 0.01, "currency": "USD"},
        },
    }

    unified = adapter.parse_response(raw)

    assert unified.id == "chatcmpl_1"
    assert unified.messages[0].content[0].text == "answer"
    assert unified.messages[0].reasoning[0].text == "reasoned"
    assert unified.usage is not None
    assert unified.usage.input_tokens == 10
    assert unified.usage.output_tokens == 5
    assert unified.usage.reasoning_tokens == 3
    assert unified.usage.cache_read_tokens == 3
    assert unified.usage.cache_write_tokens == 2
    assert unified.usage.cost is not None
    assert unified.usage.cost.provider_reported_cost == 0.01

    formatted = adapter.format_response(unified)
    assert formatted["usage"]["prompt_tokens"] == 10
    assert formatted["usage"]["completion_tokens"] == 5
    assert formatted["usage"]["total_tokens"] == 18
    assert formatted["usage"]["prompt_tokens_details"] == {"cached_tokens": 3, "cache_creation_tokens": 2}
    assert formatted["usage"]["completion_tokens_details"] == {"reasoning_tokens": 3}
    assert formatted["usage"]["cost_details"]["total_cost"] == 0.01
    assert "input_tokens" not in formatted["usage"]
    assert "output_tokens" not in formatted["usage"]


def test_openai_legacy_function_call_is_unified_and_round_trips() -> None:
    adapter = get_protocol("openai_chat")
    raw = {
        "model": "openai/gpt-test",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "lookup", "arguments": "{\"q\":\"x\"}"},
            }
        ],
    }

    unified = adapter.parse_request(raw)
    rebuilt = adapter.build_request(unified)

    assert unified.messages[0].tool_calls[0].name == "lookup"
    assert unified.messages[0].tool_calls[0].extra["legacy_function_call"] is True
    assert rebuilt["messages"][0]["function_call"] == {"name": "lookup", "arguments": "{\"q\":\"x\"}"}
    assert "tool_calls" not in rebuilt["messages"][0]


def test_openai_chat_stream_event_parses_sse_delta_and_done() -> None:
    adapter = get_protocol("openai_chat")
    event = {
        "id": "chunk_1",
        "model": "openai/gpt-test",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}],
    }

    parsed = adapter.parse_stream_event(f"data: {json.dumps(event)}\n\n")
    done = adapter.parse_stream_event("data: [DONE]\n\n")

    assert parsed.type == "message_delta"
    assert parsed.delta is not None
    assert parsed.delta.content[0].text == "Hel"
    assert done.type == "done"
