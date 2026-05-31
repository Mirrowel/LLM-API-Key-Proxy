from __future__ import annotations

from rotator_library.protocols import (
    OPERATION_CHAT,
    OPERATION_COUNT_TOKENS,
    OPERATION_EMBEDDINGS,
    OPERATION_MESSAGES,
    OPERATION_RESPONSES,
    OPERATION_UNKNOWN,
    ProtocolAdapter,
    ProtocolContext,
    UnifiedRequest,
    UnifiedResponse,
    get_protocol,
    normalize_operation,
)


def test_operation_names_are_extensible_strings() -> None:
    assert normalize_operation(" Chat ") == OPERATION_CHAT
    assert normalize_operation(None) == OPERATION_UNKNOWN
    assert normalize_operation("custom_gateway_op") == "custom_gateway_op"


def test_unified_request_response_carry_non_chat_operation_fields() -> None:
    request = UnifiedRequest(
        operation=OPERATION_EMBEDDINGS,
        model="text-embedding-test",
        input=["a", "b"],
        modalities=["text"],
        files=[{"name": "audio.wav", "content_type": "audio/wav"}],
    )
    response = UnifiedResponse(
        operation=OPERATION_EMBEDDINGS,
        model="text-embedding-test",
        data=[{"embedding": [0.1, 0.2]}],
        content_type="application/json",
    )

    assert request.to_dict()["operation"] == OPERATION_EMBEDDINGS
    assert request.to_dict()["input"] == ["a", "b"]
    assert response.to_dict()["data"][0]["embedding"] == [0.1, 0.2]
    assert response.to_dict()["content_type"] == "application/json"


def test_base_adapter_preserves_operation_fields() -> None:
    adapter = ProtocolAdapter()
    raw = {
        "operation": "custom_op",
        "model": "provider/model",
        "input": "payload",
        "modalities": ["text"],
        "files": [{"name": "file.bin"}],
        "custom": True,
    }

    unified = adapter.parse_request(raw)
    rebuilt = adapter.build_request(unified)

    assert unified.operation == "custom_op"
    assert unified.input == "payload"
    assert rebuilt == raw


def test_protocols_advertise_supported_operations() -> None:
    assert get_protocol("openai_chat").supports_operation(OPERATION_CHAT)
    assert not get_protocol("openai_chat").supports_operation(OPERATION_EMBEDDINGS)
    assert get_protocol("litellm_fallback").supports_operation(OPERATION_UNKNOWN)


def test_core_protocols_stamp_parsed_operation_fields() -> None:
    chat = get_protocol("openai_chat").parse_request({"model": "m", "messages": []})
    messages = get_protocol("anthropic_messages").parse_request({"model": "m", "messages": []})
    responses = get_protocol("responses").parse_request({"model": "m", "input": "hi"})
    gemini = get_protocol("gemini").parse_request({"contents": []})

    assert chat.operation == OPERATION_CHAT
    assert messages.operation == OPERATION_MESSAGES
    assert responses.operation == OPERATION_RESPONSES
    assert gemini.operation == OPERATION_CHAT


def test_count_tokens_operation_can_be_context_selected() -> None:
    context = ProtocolContext(provider_options={"operation": OPERATION_COUNT_TOKENS})
    anthropic = get_protocol("anthropic_messages").parse_response({"input_tokens": 12}, context)
    gemini = get_protocol("gemini").parse_response({"totalTokens": 14}, context)

    assert anthropic.operation == OPERATION_COUNT_TOKENS
    assert anthropic.usage is not None
    assert anthropic.usage.input_tokens == 12
    assert get_protocol("anthropic_messages").format_response(anthropic) == {"input_tokens": 12}
    assert gemini.operation == OPERATION_COUNT_TOKENS
    assert gemini.usage is not None
    assert gemini.usage.total_tokens == 14
    assert get_protocol("gemini").format_response(gemini) == {"totalTokens": 14}
