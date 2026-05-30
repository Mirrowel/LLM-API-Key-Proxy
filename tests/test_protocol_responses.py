from __future__ import annotations

import json

from rotator_library.protocols import get_protocol, list_protocols


def test_responses_protocol_is_discovered_with_aliases_and_websocket_support() -> None:
    adapter = get_protocol("responses")

    assert "responses" in list_protocols()
    assert get_protocol("openai_responses") is adapter
    assert adapter.supports_transport("websocket") is False
    assert adapter.is_future_transport("websocket") is True


def test_responses_request_round_trip_preserves_previous_response_and_tools() -> None:
    adapter = get_protocol("responses")
    raw = {
        "model": "openai/gpt-test",
        "instructions": "system",
        "previous_response_id": "resp_prev",
        "stream": True,
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]},
            {"type": "function_call_output", "call_id": "call_1", "output": "result"},
        ],
        "tools": [{"type": "function", "name": "lookup", "description": "Lookup", "parameters": {"type": "object"}}],
        "reasoning": {"effort": "medium"},
        "vendor_extension": {"kept": True},
    }

    unified = adapter.parse_request(raw)
    rebuilt = adapter.build_request(unified)

    assert unified.previous_response_id == "resp_prev"
    assert unified.system[0].text == "system"
    assert unified.messages[0].content[0].text == "hello"
    assert unified.messages[1].tool_call_id == "call_1"
    assert unified.tools[0].name == "lookup"
    assert rebuilt["previous_response_id"] == "resp_prev"
    assert rebuilt["reasoning"] == {"effort": "medium"}
    assert rebuilt["vendor_extension"] == {"kept": True}


def test_responses_response_extracts_output_items_reasoning_calls_and_usage() -> None:
    adapter = get_protocol("responses")
    raw = {
        "id": "resp_1",
        "object": "response",
        "created_at": 123,
        "model": "openai/gpt-test",
        "status": "completed",
        "output": [
            {"id": "msg_1", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
            {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "reasoned"}]},
            {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{\"q\":\"x\"}"},
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 18,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 3},
            "cost_details": {"total_cost": 0.02, "currency": "USD"},
        },
    }

    unified = adapter.parse_response(raw)

    assert unified.id == "resp_1"
    assert len(unified.output) == 3
    assert unified.messages[0].content[0].text == "answer"
    assert unified.messages[1].reasoning[0].text == "reasoned"
    assert unified.messages[2].tool_calls[0].name == "lookup"
    assert unified.usage is not None
    assert unified.usage.input_tokens == 10
    assert unified.usage.reasoning_tokens == 3
    assert unified.usage.cache_read_tokens == 3
    assert unified.usage.cost is not None
    assert unified.usage.cost.provider_reported_cost == 0.02


def test_responses_format_preserves_unknown_output_items() -> None:
    adapter = get_protocol("responses")
    raw = {
        "id": "resp_1",
        "model": "openai/gpt-test",
        "output": [
            {"id": "msg_1", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "before"}]},
            {"id": "future_1", "type": "future_item", "custom": {"kept": True}},
        ],
    }

    unified = adapter.parse_response(raw)
    unified.messages[0].content[0].text = "after"
    rebuilt = adapter.format_response(unified)

    assert rebuilt["output"][0]["content"][0]["text"] == "after"
    assert rebuilt["output"][1] == {"id": "future_1", "type": "future_item", "custom": {"kept": True}}


def test_responses_stream_event_parses_text_delta_and_completed_response() -> None:
    adapter = get_protocol("responses")
    delta = {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "Hi"}
    completed = {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "model": "openai/gpt-test",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi"}]}],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        },
    }

    parsed_delta = adapter.parse_stream_event(f"data: {json.dumps(delta)}\n\n")
    parsed_completed = adapter.parse_stream_event(completed)

    assert parsed_delta.type == "message_delta"
    assert parsed_delta.delta is not None
    assert parsed_delta.delta.content[0].text == "Hi"
    assert parsed_completed.type == "response.completed"
    assert parsed_completed.usage is not None
    assert parsed_completed.usage.total_tokens == 2
