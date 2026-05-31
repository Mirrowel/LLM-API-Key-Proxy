from __future__ import annotations

from rotator_library.protocols.responses import ResponsesProtocol
from rotator_library.responses import ResponsesBridge


def test_bridge_converts_responses_request_to_chat_kwargs() -> None:
    protocol = ResponsesProtocol()
    bridge = ResponsesBridge(protocol)
    unified = protocol.parse_request(
        {
            "model": "gpt-test",
            "instructions": "Follow rules.",
            "input": "Hello",
            "max_output_tokens": 20,
            "metadata": {"trace": "yes"},
        }
    )

    kwargs = bridge.to_chat_kwargs(unified)

    assert kwargs["model"] == "gpt-test"
    assert kwargs["messages"] == [
        {"role": "system", "content": "Follow rules."},
        {"role": "user", "content": "Hello"},
    ]
    assert kwargs["max_tokens"] == 20
    assert kwargs["metadata"] == {"trace": "yes"}


def test_bridge_adds_parent_response_messages_for_previous_response_id() -> None:
    protocol = ResponsesProtocol()
    bridge = ResponsesBridge(protocol)
    unified = protocol.parse_request({"model": "gpt-test", "input": "Continue", "previous_response_id": "resp_parent"})
    parent = {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Earlier"}]}]}

    kwargs = bridge.to_chat_kwargs(unified, parent_response=parent)

    assert kwargs["messages"] == [
        {"role": "assistant", "content": "Earlier"},
        {"role": "user", "content": "Continue"},
    ]
    assert kwargs["_responses_bridge"]["previous_response_id"] == "resp_parent"
    assert kwargs["_session_tracking_hints"]["strong_anchors"] == ["responses_previous_response_id:resp_parent"]
    assert kwargs["_session_tracking_hints"]["affinity_key"] == "responses_previous_response_id:resp_parent"
    assert "session_scope" not in kwargs["_session_tracking_hints"]


def test_bridge_replays_parent_input_and_output_lineage() -> None:
    protocol = ResponsesProtocol()
    bridge = ResponsesBridge(protocol)
    unified = protocol.parse_request({"model": "gpt-test", "input": "Now"})
    lineage = [
        {
            "request": {"model": "gpt-test", "input": "First user"},
            "response": {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "First assistant"}]}]},
        },
        {
            "request": {"model": "gpt-test", "input": "Second user", "previous_response_id": "resp_1"},
            "response": {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Second assistant"}]}]},
        },
    ]

    kwargs = bridge.to_chat_kwargs(unified, parent_responses=lineage)

    assert kwargs["messages"] == [
        {"role": "user", "content": "First user"},
        {"role": "assistant", "content": "First assistant"},
        {"role": "user", "content": "Second user"},
        {"role": "assistant", "content": "Second assistant"},
        {"role": "user", "content": "Now"},
    ]


def test_bridge_replays_parent_tool_call_outputs() -> None:
    protocol = ResponsesProtocol()
    bridge = ResponsesBridge(protocol)
    unified = protocol.parse_request({"model": "gpt-test", "input": "Continue"})
    lineage = [
        {
            "request": {"model": "gpt-test", "input": "Use tool"},
            "response": {"output": [{"id": "call_1", "type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"}]},
        }
    ]

    kwargs = bridge.to_chat_kwargs(unified, parent_responses=lineage)

    assert kwargs["messages"] == [
        {"role": "user", "content": "Use tool"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "user", "content": "Continue"},
    ]


def test_bridge_replays_parent_tool_result_inputs_as_tool_messages() -> None:
    protocol = ResponsesProtocol()
    bridge = ResponsesBridge(protocol)
    unified = protocol.parse_request({"model": "gpt-test", "input": "Continue"})
    lineage = [
        {
            "request": {
                "model": "gpt-test",
                "input": [
                    {"type": "message", "role": "user", "content": "Use tool"},
                    {"type": "function_call_output", "call_id": "call_1", "output": "tool result"},
                ],
            },
            "response": {"output": []},
        }
    ]

    kwargs = bridge.to_chat_kwargs(unified, parent_responses=lineage)

    assert kwargs["messages"] == [
        {"role": "user", "content": "Use tool"},
        {"role": "tool", "content": "tool result", "tool_call_id": "call_1"},
        {"role": "user", "content": "Continue"},
    ]


def test_bridge_preserves_tool_definitions() -> None:
    protocol = ResponsesProtocol()
    bridge = ResponsesBridge(protocol)
    unified = protocol.parse_request(
        {
            "model": "gpt-test",
            "input": "Use tool",
            "tools": [{"type": "function", "name": "lookup", "description": "Lookup", "parameters": {"type": "object"}}],
        }
    )

    kwargs = bridge.to_chat_kwargs(unified)

    assert kwargs["tools"] == [
        {"type": "function", "function": {"name": "lookup", "description": "Lookup", "parameters": {"type": "object"}}}
    ]


def test_bridge_preserves_unsupported_fields_for_trace_metadata() -> None:
    protocol = ResponsesProtocol()
    bridge = ResponsesBridge(protocol)
    unified = protocol.parse_request({"model": "gpt-test", "input": "Hi", "custom_unsupported": 42})

    kwargs = bridge.to_chat_kwargs(unified)

    assert kwargs["_responses_bridge"]["extra"]["custom_unsupported"] == 42


def test_bridge_converts_chat_response_to_responses_payload() -> None:
    protocol = ResponsesProtocol()
    bridge = ResponsesBridge(protocol)
    unified = protocol.parse_request({"model": "gpt-test", "input": "Hello"})
    chat_response = {
        "id": "chat_1",
        "model": "gpt-test",
        "created": 123,
        "choices": [{"message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }

    response = bridge.from_chat_response(chat_response, unified)

    assert response["id"] == "chat_1"
    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert response["output"][0]["content"] == [{"type": "output_text", "text": "Hi"}]
    assert response["usage"]["input_tokens"] == 1
    assert response["usage"]["output_tokens"] == 2
    assert response["usage"]["total_tokens"] == 3


def test_bridge_converts_chat_tool_calls_to_responses_output_items() -> None:
    protocol = ResponsesProtocol()
    bridge = ResponsesBridge(protocol)
    unified = protocol.parse_request({"model": "gpt-test", "input": "Hello"})
    chat_response = {
        "model": "gpt-test",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}],
                }
            }
        ],
    }

    response = bridge.from_chat_response(chat_response, unified, response_id="resp_tool")

    assert response["id"] == "resp_tool"
    assert response["output"] == [
        {"id": "call_1", "type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{}"}
    ]
