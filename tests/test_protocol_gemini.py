from __future__ import annotations

import json

from src.rotator_library.protocols import get_protocol, list_protocols


def test_gemini_protocol_is_discovered_with_aliases() -> None:
    assert "gemini" in list_protocols()
    assert get_protocol("google_gemini") is get_protocol("gemini")
    assert get_protocol("generate_content") is get_protocol("gemini")


def test_gemini_request_round_trip_preserves_parts_tools_and_settings() -> None:
    adapter = get_protocol("gemini")
    raw = {
        "model": "gemini/gemini-test",
        "systemInstruction": {"parts": [{"text": "system"}]},
        "contents": [
            {"role": "user", "parts": [{"text": "hello"}, {"inlineData": {"mimeType": "image/png", "data": "abc"}}]},
            {
                "role": "model",
                "parts": [
                    {"text": "thought", "thought": True, "thoughtSignature": "sig_1"},
                    {"functionCall": {"name": "lookup", "args": {"q": "x"}}},
                ],
            },
            {"role": "user", "parts": [{"functionResponse": {"name": "lookup", "response": {"value": 1}}}]},
        ],
        "tools": [{"functionDeclarations": [{"name": "lookup", "description": "Lookup", "parameters": {"type": "object"}}]}],
        "generationConfig": {"temperature": 0.3},
        "safetySettings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}],
        "vendor_extension": {"kept": True},
    }

    unified = adapter.parse_request(raw)
    rebuilt = adapter.build_request(unified)

    assert unified.model == "gemini/gemini-test"
    assert unified.system[0].text == "system"
    assert unified.messages[1].role == "assistant"
    assert unified.messages[1].reasoning[0].signature == "sig_1"
    assert unified.messages[1].tool_calls[0].name == "lookup"
    assert unified.messages[2].content[0].tool_result.content == {"value": 1}
    assert unified.tools[0].name == "lookup"
    assert rebuilt["generationConfig"] == {"temperature": 0.3}
    assert rebuilt["safetySettings"][0]["threshold"] == "BLOCK_NONE"
    assert rebuilt["vendor_extension"] == {"kept": True}


def test_gemini_response_extracts_usage_and_thought_signature() -> None:
    adapter = get_protocol("gemini")
    raw = {
        "responseId": "resp_1",
        "modelVersion": "gemini-test-001",
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"role": "model", "parts": [{"text": "answer", "thought": True, "thoughtSignature": "sig_2"}]},
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "thoughtsTokenCount": 2,
            "cachedContentTokenCount": 3,
            "totalTokenCount": 17,
        },
    }

    unified = adapter.parse_response(raw)

    assert unified.id == "resp_1"
    assert unified.stop_reason == "STOP"
    assert unified.messages[0].content[0].text == "answer"
    assert unified.messages[0].reasoning[0].signature == "sig_2"
    assert unified.usage is not None
    assert unified.usage.input_tokens == 10
    assert unified.usage.output_tokens == 5
    assert unified.usage.reasoning_tokens == 2
    assert unified.usage.cache_read_tokens == 3


def test_gemini_stream_event_parses_candidate_delta() -> None:
    adapter = get_protocol("gemini")
    event = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Hi"}]}, "finishReason": None}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }

    parsed = adapter.parse_stream_event(f"data: {json.dumps(event)}\n\n")

    assert parsed.type == "message_delta"
    assert parsed.delta is not None
    assert parsed.delta.content[0].text == "Hi"
    assert parsed.usage is not None
    assert parsed.usage.total_tokens == 2
