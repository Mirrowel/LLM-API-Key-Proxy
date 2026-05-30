from __future__ import annotations

import json

import pytest

from src.rotator_library.protocols import (
    ContentBlock,
    ProtocolAdapter,
    ProtocolContext,
    ProtocolError,
    ToolCall,
    UnifiedMessage,
    UnifiedRequest,
    get_protocol,
    get_protocol_class,
    list_protocols,
    register_protocol,
    serialize_value,
)


class ExampleProtocol(ProtocolAdapter):
    name = "example_test_protocol"
    aliases = ("example_test_alias",)


def test_protocol_serialization_preserves_nested_values() -> None:
    request = UnifiedRequest(
        model="provider/model",
        messages=[
            UnifiedMessage(
                role="assistant",
                content=[ContentBlock(type="text", text="hello")],
                tool_calls=[ToolCall(id="call_1", name="lookup", arguments={"q": "x"})],
                extra={"provider_field": {"kept": True}},
            )
        ],
        metadata={"request_id": "req_1"},
    )

    serialized = request.to_dict()

    assert serialized["messages"][0]["content"][0]["text"] == "hello"
    assert serialized["messages"][0]["tool_calls"][0]["arguments"] == {"q": "x"}
    assert serialized["messages"][0]["extra"]["provider_field"] == {"kept": True}
    json.dumps(serialize_value(request))


def test_base_protocol_preserves_unknown_request_fields() -> None:
    adapter = ProtocolAdapter()
    raw = {"model": "provider/model", "stream": True, "custom": {"value": 1}}

    unified = adapter.parse_request(raw, ProtocolContext(provider="provider"))
    rebuilt = adapter.build_request(unified)

    assert unified.model == "provider/model"
    assert unified.stream is True
    assert unified.extra == {"custom": {"value": 1}}
    assert rebuilt == raw


def test_register_protocol_resolves_alias_and_reuses_instance() -> None:
    register_protocol(ExampleProtocol, replace=True)

    assert get_protocol_class("example_test_protocol") is ExampleProtocol
    assert get_protocol_class("example_test_alias") is ExampleProtocol
    assert get_protocol("example_test_protocol") is get_protocol("example_test_alias")
    assert "example_test_protocol" in list_protocols()


def test_register_protocol_rejects_duplicate_alias() -> None:
    class FirstProtocol(ProtocolAdapter):
        name = "duplicate_alias_first"
        aliases = ("duplicate_alias",)

    class SecondProtocol(ProtocolAdapter):
        name = "duplicate_alias_second"
        aliases = ("duplicate_alias",)

    register_protocol(FirstProtocol, replace=True)

    with pytest.raises(ValueError, match="alias already registered"):
        register_protocol(SecondProtocol)


def test_protocol_error_includes_pass_and_payload_preview() -> None:
    error = ProtocolError(
        "failed",
        protocol="example",
        pass_name="parse_request",
        payload={"secret": "not-redacted-here", "value": 1},
    )

    assert "example.parse_request" in str(error)
    assert "value" in str(error)
