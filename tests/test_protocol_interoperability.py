from __future__ import annotations

from typing import Any

import pytest

from rotator_library.protocols import ProtocolContext, ProtocolError, UnifiedMessage, get_protocol
from rotator_library.protocols.canonical import (
    instruction_blocks,
    message_reasoning,
    message_tool_calls,
    message_tool_results,
    ordered_message_blocks,
)


PROTOCOLS = ("openai_chat", "anthropic_messages", "responses", "gemini")


REQUEST_FIXTURES: dict[str, dict[str, Any]] = {
    "openai_chat": {
        "model": "model-test",
        "messages": [
            {"role": "system", "content": "system rule"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}},
                ],
            },
            {
                "role": "assistant",
                "content": "calling",
                "reasoning_content": "reasoned",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "lookup", "content": '{"value":1}'},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ],
        "max_completion_tokens": 123,
        "temperature": 0.2,
        "stop": ["END"],
        "tool_choice": "auto",
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {"type": "object", "properties": {"value": {"type": "integer"}}},
            },
        },
        "vendor_extension": {"source_only": True},
    },
    "anthropic_messages": {
        "model": "model-test",
        "system": [{"type": "text", "text": "system rule"}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "reasoned", "signature": "sig_anthropic"},
                    {"type": "text", "text": "calling"},
                    {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "x"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": {"value": 1}}
                ],
            },
        ],
        "tools": [
            {
                "name": "lookup",
                "description": "Lookup",
                "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ],
        "max_tokens": 123,
        "temperature": 0.2,
        "stop_sequences": ["END"],
        "tool_choice": {"type": "auto"},
        "output_config": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "strict": True,
                "schema": {"type": "object", "properties": {"value": {"type": "integer"}}},
            }
        },
        "vendor_extension": {"source_only": True},
    },
    "responses": {
        "model": "model-test",
        "instructions": "system rule",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look"},
                    {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="},
                ],
            },
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "reasoned"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"q":"x"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {"value": 1},
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Lookup",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        ],
        "max_output_tokens": 123,
        "temperature": 0.2,
        "tool_choice": "auto",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "strict": True,
                "schema": {"type": "object", "properties": {"value": {"type": "integer"}}},
            }
        },
        "vendor_extension": {"source_only": True},
    },
    "gemini": {
        "model": "model-test",
        "systemInstruction": {"parts": [{"text": "system rule"}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "look"},
                    {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}},
                ],
            },
            {
                "role": "model",
                "parts": [
                    {"text": "reasoned", "thought": True, "thoughtSignature": "sig_gemini"},
                    {"text": "calling"},
                    {"functionCall": {"id": "call_1", "name": "lookup", "args": {"q": "x"}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"id": "call_1", "name": "lookup", "response": {"value": 1}}}
                ],
            },
        ],
        "tools": [
            {
                "functionDeclarations": [
                    {
                        "name": "lookup",
                        "description": "Lookup",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    }
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": 123,
            "temperature": 0.2,
            "stopSequences": ["END"],
            "responseMimeType": "application/json",
            "responseJsonSchema": {"type": "object", "properties": {"value": {"type": "integer"}}},
        },
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        "vendor_extension": {"source_only": True},
    },
}


RESPONSE_FIXTURES: dict[str, dict[str, Any]] = {
    "openai_chat": {
        "id": "source_response",
        "model": "model-test",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "reasoned",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"q":"x"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
        "vendor_extension": {"source_only": True},
    },
    "anthropic_messages": {
        "id": "source_response",
        "type": "message",
        "role": "assistant",
        "model": "model-test",
        "content": [
            {"type": "thinking", "thinking": "reasoned", "signature": "sig_anthropic"},
            {"type": "text", "text": "answer"},
            {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "x"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 4, "output_tokens": 3},
        "vendor_extension": {"source_only": True},
    },
    "responses": {
        "id": "source_response",
        "object": "response",
        "model": "model-test",
        "status": "completed",
        "output": [
            {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "reasoned"}]},
            {"id": "msg_1", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
            {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": '{"q":"x"}'},
        ],
        "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
        "vendor_extension": {"source_only": True},
    },
    "gemini": {
        "responseId": "source_response",
        "modelVersion": "model-test",
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "reasoned", "thought": True, "thoughtSignature": "sig_gemini"},
                        {"text": "answer"},
                        {"functionCall": {"id": "call_1", "name": "lookup", "args": {"q": "x"}}},
                    ],
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 3, "totalTokenCount": 7},
        "vendor_extension": {"source_only": True},
    },
}


def _context(source: str, target: str) -> ProtocolContext:
    return ProtocolContext(source_protocol=source, target_protocol=target)


def _request_semantics(request: Any) -> dict[str, Any]:
    instruction_text = "".join(block.text or "" for block in instruction_blocks(request))
    visible_text: list[str] = []
    reasoning_text: list[str] = []
    calls: list[tuple[str | None, str | None, Any]] = []
    results: list[tuple[str | None, str | None, Any]] = []
    media: list[tuple[str, str | None]] = []
    sequence: list[tuple[str, str, Any]] = []
    for message in request.messages:
        if message.role in {"system", "developer"}:
            continue
        for block in message.content:
            if block.text and not block.reasoning:
                visible_text.append(block.text)
            if block.type in {"image", "file", "document", "audio", "video"}:
                media_type = getattr(block.source, "media_type", None)
                media.append((block.type, media_type))
        reasoning_text.extend(block.text or "" for block in message_reasoning(message))
        calls.extend((call.id, call.name, call.arguments) for call in message_tool_calls(message))
        results.extend((result.tool_call_id, result.name, result.content) for result in message_tool_results(message))
        for block in ordered_message_blocks(message):
            if block.reasoning:
                sequence.append((message.role, "reasoning", block.reasoning.text))
            elif block.tool_call:
                sequence.append((message.role, "tool_call", (block.tool_call.name, block.tool_call.arguments)))
            elif block.tool_result:
                sequence.append((message.role, "tool_result", (block.tool_result.name, block.tool_result.content)))
            elif block.text:
                sequence.append((message.role, "text", block.text))
            elif block.type in {"image", "file", "document", "audio", "video"}:
                sequence.append((message.role, block.type, getattr(block.source, "media_type", None)))
    return {
        "instructions": instruction_text,
        "visible_text": visible_text,
        "reasoning_text": reasoning_text,
        "calls": calls,
        "results": results,
        "media": media,
        "tools": [(tool.name, tool.input_schema) for tool in request.tools],
        "max_output_tokens": request.generation_params.get("max_output_tokens"),
        "temperature": request.generation_params.get("temperature"),
        "stop_sequences": request.generation_params.get("stop_sequences"),
        "tool_choice": request.generation_params.get("tool_choice"),
        "structured_output": request.generation_params.get("structured_output"),
        "sequence": sequence,
    }


def _response_semantics(messages: list[UnifiedMessage]) -> dict[str, Any]:
    text: list[str] = []
    reasoning: list[str] = []
    calls: list[tuple[str | None, str | None, Any]] = []
    sequence: list[tuple[str, Any]] = []
    for message in messages:
        text.extend(block.text or "" for block in message.content if block.text and not block.reasoning)
        reasoning.extend(block.text or "" for block in message_reasoning(message))
        calls.extend((call.id, call.name, call.arguments) for call in message_tool_calls(message))
        for block in ordered_message_blocks(message):
            if block.reasoning:
                sequence.append(("reasoning", block.reasoning.text))
            elif block.tool_call:
                sequence.append(("tool_call", (block.tool_call.name, block.tool_call.arguments)))
            elif block.text:
                sequence.append(("text", block.text))
    return {"text": text, "reasoning": reasoning, "calls": calls, "sequence": sequence}


@pytest.mark.parametrize("source_protocol", PROTOCOLS)
@pytest.mark.parametrize("target_protocol", PROTOCOLS)
def test_request_semantics_survive_every_protocol_pair(source_protocol: str, target_protocol: str) -> None:
    source = get_protocol(source_protocol)
    target = get_protocol(target_protocol)
    unified = source.parse_request(REQUEST_FIXTURES[source_protocol], _context(source_protocol, target_protocol))
    before = _request_semantics(unified)

    provider_payload = target.build_request(unified, _context(source_protocol, target_protocol))
    reparsed = target.parse_request(provider_payload, _context(target_protocol, target_protocol))
    after = _request_semantics(reparsed)

    assert after["instructions"] == before["instructions"]
    assert after["visible_text"] == before["visible_text"]
    assert after["reasoning_text"] == before["reasoning_text"]
    assert after["calls"] == before["calls"]
    assert after["results"] == before["results"]
    assert after["media"] == before["media"]
    assert after["tools"] == before["tools"]
    assert after["max_output_tokens"] == before["max_output_tokens"]
    assert after["temperature"] == before["temperature"]
    assert after["tool_choice"] == before["tool_choice"]
    assert after["structured_output"]["type"] == before["structured_output"]["type"]
    assert after["structured_output"].get("schema") == before["structured_output"].get("schema")
    assert after["sequence"] == before["sequence"]
    if target_protocol != "responses":
        assert after["stop_sequences"] == before["stop_sequences"]
    if source_protocol != target_protocol:
        assert "vendor_extension" not in provider_payload


@pytest.mark.parametrize("source_protocol", PROTOCOLS)
@pytest.mark.parametrize("target_protocol", PROTOCOLS)
def test_response_semantics_survive_every_output_protocol(source_protocol: str, target_protocol: str) -> None:
    source = get_protocol(source_protocol)
    target = get_protocol(target_protocol)
    unified = source.parse_response(RESPONSE_FIXTURES[source_protocol], _context(source_protocol, target_protocol))
    before = _response_semantics(unified.messages)

    client_payload = target.format_response(unified, _context(source_protocol, target_protocol))
    reparsed = target.parse_response(client_payload, _context(target_protocol, target_protocol))
    after = _response_semantics(reparsed.messages)

    assert after["text"] == before["text"]
    assert after["reasoning"] == before["reasoning"]
    assert after["calls"] == before["calls"]
    assert after["sequence"] == before["sequence"]
    assert reparsed.stop_reason == unified.stop_reason
    if source_protocol != target_protocol:
        assert "vendor_extension" not in client_payload


def test_required_unknown_content_is_rejected_before_cross_protocol_transport() -> None:
    source = get_protocol("openai_chat")
    target = get_protocol("anthropic_messages")
    request = source.parse_request(
        {
            "model": "model-test",
            "messages": [{"role": "user", "content": [{"type": "future_required_media", "value": "x"}]}],
        }
    )

    with pytest.raises(ProtocolError, match="future_required_media"):
        target.build_request(request, _context("openai_chat", "anthropic_messages"))


def test_same_protocol_unknown_content_is_preserved() -> None:
    adapter = get_protocol("openai_chat")
    raw = {
        "model": "model-test",
        "messages": [{"role": "user", "content": [{"type": "future_required_media", "value": "x"}]}],
    }

    request = adapter.parse_request(raw)
    assert adapter.build_request(request) == raw


def test_optional_unsupported_controls_are_dropped_with_conversion_warnings() -> None:
    source = get_protocol("anthropic_messages")
    target = get_protocol("openai_chat")
    request = source.parse_request(
        {
            "model": "model-test",
            "messages": [{"role": "user", "content": "hello"}],
            "top_k": 12,
        }
    )

    payload = target.build_request(request, _context("anthropic_messages", "openai_chat"))

    assert "top_k" not in payload
    assert [(warning.field, warning.target_protocol) for warning in request.warnings] == [("top_k", "openai_chat")]


def test_responses_stop_sequence_loss_is_explicit() -> None:
    source = get_protocol("openai_chat")
    target = get_protocol("responses")
    request = source.parse_request(
        {
            "model": "model-test",
            "messages": [{"role": "user", "content": "hello"}],
            "stop": ["END"],
        }
    )

    payload = target.build_request(request, _context("openai_chat", "responses"))

    assert "stop" not in payload
    assert any(warning.field == "stop_sequences" for warning in request.warnings)


def test_source_identity_prevents_foreign_passthrough_without_context() -> None:
    source = get_protocol("openai_chat")
    target = get_protocol("gemini")
    request = source.parse_request(
        {
            "model": "model-test",
            "messages": [{"role": "user", "content": "hello", "chat_only": True}],
            "vendor_extension": {"source_only": True},
        }
    )

    payload = target.build_request(request)

    assert "vendor_extension" not in payload
    assert "chat_only" not in payload["contents"][0]


def test_opaque_reasoning_signatures_do_not_cross_protocol_families() -> None:
    anthropic = get_protocol("anthropic_messages")
    gemini = get_protocol("gemini")
    anthropic_request = anthropic.parse_request(REQUEST_FIXTURES["anthropic_messages"])
    gemini_request = gemini.parse_request(REQUEST_FIXTURES["gemini"])

    gemini_payload = gemini.build_request(anthropic_request, _context("anthropic_messages", "gemini"))
    anthropic_payload = anthropic.build_request(gemini_request, _context("gemini", "anthropic_messages"))

    assert "thoughtSignature" not in str(gemini_payload)
    assert "sig_anthropic" not in str(gemini_payload)
    assert "signature" not in str(anthropic_payload)
    assert "sig_gemini" not in str(anthropic_payload)
    assert anthropic_request.messages[1].reasoning[0].signature == "sig_anthropic"
    assert gemini_request.messages[1].reasoning[0].signature == "sig_gemini"


def test_unknown_system_content_is_validated() -> None:
    source = get_protocol("anthropic_messages")
    target = get_protocol("openai_chat")
    request = source.parse_request(
        {
            "model": "model-test",
            "system": [{"type": "future_required_instruction", "value": "x"}],
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    with pytest.raises(ProtocolError, match="future_required_instruction"):
        target.build_request(request, _context("anthropic_messages", "openai_chat"))


def test_provider_bound_responses_continuation_is_not_silently_dropped() -> None:
    source = get_protocol("responses")
    target = get_protocol("anthropic_messages")
    request = source.parse_request(
        {
            "model": "model-test",
            "previous_response_id": "resp_provider_bound",
            "input": "continue",
        }
    )

    with pytest.raises(ProtocolError, match="previous_response_id"):
        target.build_request(request, _context("responses", "anthropic_messages"))


def test_gemini_safety_policy_is_not_silently_dropped() -> None:
    source = get_protocol("gemini")
    target = get_protocol("openai_chat")
    request = source.parse_request(
        {
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "safetySettings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_LOW_AND_ABOVE"}],
        }
    )

    with pytest.raises(ProtocolError, match="safety settings"):
        target.build_request(request, _context("gemini", "openai_chat"))


def test_missing_media_identity_is_rejected() -> None:
    source = get_protocol("openai_chat")
    target = get_protocol("gemini")
    request = source.parse_request(
        {
            "model": "model-test",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
        }
    )

    with pytest.raises(ProtocolError, match="without URL, data, or file identity"):
        target.build_request(request, _context("openai_chat", "gemini"))


def test_tool_argument_and_result_wire_types_are_destination_valid() -> None:
    source = get_protocol("openai_chat")
    request = source.parse_request(REQUEST_FIXTURES["openai_chat"])

    anthropic_payload = get_protocol("anthropic_messages").build_request(
        request,
        _context("openai_chat", "anthropic_messages"),
    )
    gemini_payload = get_protocol("gemini").build_request(
        request,
        _context("openai_chat", "gemini"),
    )
    responses_payload = get_protocol("responses").build_request(
        request,
        _context("openai_chat", "responses"),
    )

    anthropic_call = next(block for message in anthropic_payload["messages"] for block in message["content"] if block["type"] == "tool_use")
    anthropic_result = next(block for message in anthropic_payload["messages"] for block in message["content"] if block["type"] == "tool_result")
    gemini_call = next(part for content in gemini_payload["contents"] for part in content["parts"] if "functionCall" in part)
    gemini_result = next(part for content in gemini_payload["contents"] for part in content["parts"] if "functionResponse" in part)
    responses_call = next(item for item in responses_payload["input"] if item["type"] == "function_call")
    responses_result = next(item for item in responses_payload["input"] if item["type"] == "function_call_output")

    assert anthropic_call["input"] == {"q": "x"}
    assert isinstance(anthropic_result["content"], str)
    assert gemini_call["functionCall"]["args"] == {"q": "x"}
    assert gemini_result["functionResponse"]["response"] == {"value": 1}
    assert responses_call["arguments"] == '{"q":"x"}'
    assert responses_result["output"] == '{"value":1}'


def test_structured_output_has_one_canonical_request_field_for_every_source() -> None:
    for source_name in PROTOCOLS:
        request = get_protocol(source_name).parse_request(REQUEST_FIXTURES[source_name])
        assert request.response_format == request.generation_params["structured_output"]


def test_unknown_native_stop_reason_never_leaks_as_invalid_chat_value() -> None:
    response = get_protocol("gemini").parse_response(
        {
            "responseId": "resp_unknown",
            "candidates": [{"finishReason": "OTHER", "content": {"role": "model", "parts": [{"text": "answer"}]}}],
        }
    )

    payload = get_protocol("openai_chat").format_response(response, _context("gemini", "openai_chat"))

    assert payload["choices"][0]["finish_reason"] is None
    assert response.metadata["native_stop_reason"] == "OTHER"


@pytest.mark.parametrize(
    ("source_choice", "expected_mode"),
    [
        ({"type": "function", "function": {"name": "lookup"}}, "named"),
        ("required", "required"),
        ("none", "none"),
    ],
)
def test_tool_choice_modes_survive_all_destinations(source_choice: Any, expected_mode: str) -> None:
    source = get_protocol("openai_chat")
    request = source.parse_request(
        {
            "model": "model-test",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": REQUEST_FIXTURES["openai_chat"]["tools"],
            "tool_choice": source_choice,
        }
    )

    for target_name in PROTOCOLS:
        target = get_protocol(target_name)
        payload = target.build_request(request, _context("openai_chat", target_name))
        reparsed = target.parse_request(payload)
        assert reparsed.generation_params["tool_choice"]["mode"] == expected_mode


def test_anthropic_system_preserves_instruction_boundaries_and_order() -> None:
    request = get_protocol("openai_chat").parse_request(
        {
            "model": "model-test",
            "messages": [
                {"role": "system", "content": "system A"},
                {"role": "developer", "content": "developer B"},
                {"role": "user", "content": "hello"},
            ],
        }
    )

    payload = get_protocol("anthropic_messages").build_request(
        request,
        _context("openai_chat", "anthropic_messages"),
    )

    assert payload["system"] == [
        {"type": "text", "text": "system A"},
        {"type": "text", "text": "developer B"},
    ]


def test_opaque_state_requires_compatible_provider_domain_even_for_same_protocol() -> None:
    adapter = get_protocol("anthropic_messages")
    request = adapter.parse_request(REQUEST_FIXTURES["anthropic_messages"])
    incompatible = ProtocolContext(
        source_protocol="anthropic_messages",
        target_protocol="anthropic_messages",
        source_provider="provider-a",
        target_provider="provider-b",
    )
    compatible = ProtocolContext(
        source_protocol="anthropic_messages",
        target_protocol="anthropic_messages",
        source_provider="provider-a",
        target_provider="provider-a",
    )

    incompatible_payload = adapter.build_request(request, incompatible)
    compatible_payload = adapter.build_request(request, compatible)
    unscoped_payload = adapter.build_request(request)

    assert "sig_anthropic" not in str(incompatible_payload)
    assert "sig_anthropic" not in str(unscoped_payload)
    assert "sig_anthropic" in str(compatible_payload)


def test_foreign_reasoning_extensions_never_enter_target_payload() -> None:
    request = get_protocol("gemini").parse_request(
        {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "text": "thinking",
                            "thought": True,
                            "thoughtSignature": "sig_private",
                            "vendor_reasoning_extension": "must-not-leak",
                        }
                    ],
                }
            ]
        }
    )

    payload = get_protocol("anthropic_messages").build_request(
        request,
        _context("gemini", "anthropic_messages"),
    )

    assert "vendor_reasoning_extension" not in str(payload)
    assert "sig_private" not in str(payload)


def test_repeated_name_only_gemini_calls_get_distinct_cross_protocol_ids_without_same_protocol_leak() -> None:
    adapter = get_protocol("gemini")
    raw = {
        "contents": [
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "lookup", "args": {"q": "a"}}},
                    {"functionCall": {"name": "lookup", "args": {"q": "b"}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "lookup", "response": {"value": "a"}}},
                    {"functionResponse": {"name": "lookup", "response": {"value": "b"}}},
                ],
            },
        ]
    }
    request = adapter.parse_request(raw)

    same_protocol = adapter.build_request(request)
    chat_payload = get_protocol("openai_chat").build_request(
        request,
        _context("gemini", "openai_chat"),
    )

    assert "id" not in same_protocol["contents"][0]["parts"][0]["functionCall"]
    assert "id" not in same_protocol["contents"][0]["parts"][1]["functionCall"]
    assert [call["id"] for call in chat_payload["messages"][0]["tool_calls"]] == ["call_0", "call_1"]
    assert [message["tool_call_id"] for message in chat_payload["messages"][1:]] == ["call_0", "call_1"]


def test_tool_result_error_state_survives_supported_protocol_conversions() -> None:
    source = get_protocol("anthropic_messages")
    request = source.parse_request(
        {
            "model": "model-test",
            "messages": [
                {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": [{"type": "text", "text": "failed"}],
                            "is_error": True,
                        }
                    ],
                },
            ],
        }
    )

    same_payload = source.build_request(request)
    assert same_payload["messages"][1]["content"][0]["content"] == [{"type": "text", "text": "failed"}]
    assert same_payload["messages"][1]["content"][0]["is_error"] is True

    chat_payload = get_protocol("openai_chat").build_request(
        request,
        _context("anthropic_messages", "openai_chat"),
    )
    responses_payload = get_protocol("responses").build_request(
        request,
        _context("anthropic_messages", "responses"),
    )
    gemini_payload = get_protocol("gemini").build_request(
        request,
        _context("anthropic_messages", "gemini"),
    )

    assert chat_payload["messages"][1]["content"] == '{"error":[{"type":"text","text":"failed"}]}'
    assert responses_payload["input"][1]["output"] == '{"error":[{"type":"text","text":"failed"}]}'
    assert gemini_payload["contents"][1]["parts"][0]["functionResponse"]["response"] == {
        "error": [{"type": "text", "text": "failed"}]
    }


@pytest.mark.parametrize("protocol_name", PROTOCOLS)
@pytest.mark.parametrize(
    "result_content",
    [
        {"error": None},
        {"error": "domain-status"},
        {"error": None, "data": "ok"},
    ],
)
def test_successful_tool_result_with_error_field_is_not_corrupted(
    protocol_name: str,
    result_content: dict[str, Any],
) -> None:
    request = get_protocol("openai_chat").parse_request(
        {
            "model": "model-test",
            "messages": [
                {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": result_content},
            ],
        }
    )
    target = get_protocol(protocol_name)
    payload = target.build_request(request, _context("openai_chat", protocol_name))
    reparsed = target.parse_request(payload)
    results = [result for message in reparsed.messages for result in message_tool_results(message)]

    assert results[0].is_error is not True
    assert results[0].content == result_content


def test_responses_same_protocol_function_arguments_remain_json_text() -> None:
    adapter = get_protocol("responses")
    response = adapter.parse_response(RESPONSE_FIXTURES["responses"])

    payload = adapter.format_response(response)
    call = next(item for item in payload["output"] if item["type"] == "function_call")

    assert call["arguments"] == '{"q":"x"}'


def test_required_response_modalities_map_or_reject() -> None:
    source = get_protocol("openai_chat")
    request = source.parse_request(
        {
            "model": "model-test",
            "messages": [{"role": "user", "content": "speak"}],
            "modalities": ["text", "audio"],
            "audio": {"voice": "alloy", "format": "wav"},
        }
    )

    gemini_payload = get_protocol("gemini").build_request(request, _context("openai_chat", "gemini"))
    assert gemini_payload["generationConfig"]["responseModalities"] == ["TEXT", "AUDIO"]

    with pytest.raises(ProtocolError, match="response modalities"):
        get_protocol("anthropic_messages").build_request(
            request,
            _context("openai_chat", "anthropic_messages"),
        )


def test_gemini_schema_enforcement_round_trips_as_strict() -> None:
    request = get_protocol("openai_chat").parse_request(REQUEST_FIXTURES["openai_chat"])
    payload = get_protocol("gemini").build_request(request, _context("openai_chat", "gemini"))
    reparsed = get_protocol("gemini").parse_request(payload)

    assert payload["generationConfig"]["responseJsonSchema"] == request.response_format["schema"]
    assert reparsed.response_format["strict"] is True


def test_explicit_non_strict_schema_records_gemini_strengthening() -> None:
    request = get_protocol("openai_chat").parse_request(REQUEST_FIXTURES["openai_chat"])
    request.response_format["strict"] = False
    request.generation_params["structured_output"]["strict"] = False

    get_protocol("gemini").build_request(request, _context("openai_chat", "gemini"))

    assert any(warning.code == "structured_output_strictness_strengthened" for warning in request.warnings)


def test_failed_response_is_not_relabelled_as_success() -> None:
    response = get_protocol("responses").parse_response(
        {"id": "resp_failed", "object": "response", "status": "failed", "output": []}
    )

    for target_name in ("openai_chat", "anthropic_messages", "gemini"):
        with pytest.raises(ProtocolError, match="failed provider response"):
            get_protocol(target_name).format_response(
                response,
                _context("responses", target_name),
            )
    assert get_protocol("responses").format_response(response)["status"] == "failed"


def test_named_tool_choice_requires_a_declared_name() -> None:
    request = get_protocol("openai_chat").parse_request(
        {
            "model": "model-test",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {}},
        }
    )

    with pytest.raises(ProtocolError, match="requires a tool name"):
        get_protocol("anthropic_messages").build_request(
            request,
            _context("openai_chat", "anthropic_messages"),
        )


def test_provider_bound_responses_controls_reject_cross_protocol_conversion() -> None:
    request = get_protocol("responses").parse_request(
        {"model": "model-test", "input": "hello", "background": True}
    )

    with pytest.raises(ProtocolError, match="Provider-bound Responses controls"):
        get_protocol("openai_chat").build_request(
            request,
            _context("responses", "openai_chat"),
        )


def test_optional_responses_controls_warn_when_target_cannot_represent_them() -> None:
    request = get_protocol("responses").parse_request(
        {"model": "model-test", "input": "hello", "prompt_cache_key": "cache-key"}
    )

    payload = get_protocol("anthropic_messages").build_request(
        request,
        _context("responses", "anthropic_messages"),
    )

    assert "prompt_cache_key" not in payload
    assert any(warning.field == "prompt_cache_key" for warning in request.warnings)
