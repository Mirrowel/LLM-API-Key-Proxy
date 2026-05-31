from __future__ import annotations

from rotator_library.protocols import (
    OPERATION_CHAT,
    OPERATION_EMBEDDINGS,
    OPERATION_UNKNOWN,
    ProtocolAdapter,
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
