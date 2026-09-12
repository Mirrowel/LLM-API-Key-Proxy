"""G14 pins: countTokens passthrough, ?alt= wire behavior, finish reasons,
capability softening, and gemini error-envelope details (fix-pass plan G14)."""

import pytest

from rotator_library.client.gemini import GeminiHandler
from rotator_library.client.executor import RoutingExecutionError
from rotator_library.core.errors import StructuredAPIResponseError
from rotator_library.error_handler import _structured_quota_signal
from rotator_library.protocols import get_protocol
from rotator_library.protocols.canonical import canonical_stop_reason, format_stop_reason
from rotator_library.protocols.types import ProtocolContext


# ---------------------------------------------------------------- count


class _UnsupportedNativeClient:
    """Fake runtime whose provider cannot serve count_tokens natively."""

    async def agenerate(self, payload, **kwargs):
        raise RoutingExecutionError(
            "Provider openai does not support native operation count_tokens",
            error_type="operation_unsupported",
        )


class _CountingNativeClient(_UnsupportedNativeClient):
    def token_count(self, *, model, messages=None, text=None):
        return 4


@pytest.mark.asyncio
async def test_gemini_count_tokens_errors_without_native_support(monkeypatch) -> None:
    monkeypatch.delenv("COUNT_TOKENS_LOCAL_ESTIMATE", raising=False)
    handler = GeminiHandler(_UnsupportedNativeClient())

    with pytest.raises(RoutingExecutionError) as raised:
        await handler.count_tokens(
            {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            model="gemini/gemini-2.5-pro",
        )

    assert raised.value.error_type == "operation_unsupported"


@pytest.mark.asyncio
async def test_gemini_count_tokens_local_estimate_is_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("COUNT_TOKENS_LOCAL_ESTIMATE", "1")
    handler = GeminiHandler(_CountingNativeClient())

    result = await handler.count_tokens(
        {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        model="gemini/gemini-2.5-pro",
    )

    assert result["totalTokens"] == 4
    assert result["x-proxy-estimate"] == "local-projection"


@pytest.mark.asyncio
async def test_gemini_count_tokens_native_passthrough() -> None:
    class _NativeCountClient:
        async def agenerate(self, payload, **kwargs):
            return {"totalTokens": 42, "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 42}]}

    handler = GeminiHandler(_NativeCountClient())
    result = await handler.count_tokens(
        {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        model="gemini/gemini-2.5-pro",
    )

    assert result["totalTokens"] == 42
    assert "x-proxy-estimate" not in result


# ------------------------------------------------- count request shapes

def _count_context():
    return ProtocolContext(
        source_protocol="gemini",
        target_protocol="gemini",
        model="gemini-2.5-pro",
        metadata={"operation": "count_tokens"},
    )


def test_count_tokens_build_nested_envelope_xor() -> None:
    gemini = get_protocol("gemini")
    unified = gemini.parse_request(
        {
            "model": "gemini-2.5-pro",
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "tools": [{"functionDeclarations": [{"name": "search"}]}],
        },
        _count_context(),
    )

    payload = gemini.build_request(unified, _count_context())

    # XOR: contents lives INSIDE the nested envelope, never at both levels.
    assert "contents" not in payload
    nested = payload["generateContentRequest"]
    assert nested["contents"]
    assert nested["tools"]


def test_count_tokens_build_flat_without_generate_members() -> None:
    gemini = get_protocol("gemini")
    unified = gemini.parse_request(
        {"model": "gemini-2.5-pro", "contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
        _count_context(),
    )

    payload = gemini.build_request(unified, _count_context())

    assert payload == {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}


def test_count_tokens_ingress_parses_sdk_nested_form() -> None:
    gemini = get_protocol("gemini")
    unified = gemini.parse_request(
        {
            "generateContentRequest": {
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "systemInstruction": {"parts": [{"text": "be brief"}]},
            }
        },
        _count_context(),
    )

    assert unified.messages and unified.messages[0].content[0].text == "hi"
    assert unified.system


# ------------------------------------------------------- finish reasons

def test_finish_reason_drift_batch_maps_honestly() -> None:
    assert canonical_stop_reason("MISSING_THOUGHT_SIGNATURE") == "error"
    assert canonical_stop_reason("NO_IMAGE") == "content_filter"
    assert canonical_stop_reason("IMAGE_RECITATION") == "content_filter"
    assert canonical_stop_reason("UNEXPECTED_TOOL_CALL") == "error"
    assert canonical_stop_reason("ESCALATION") == "content_filter"
    # unknown values never leak raw spellings onto foreign wires
    assert format_stop_reason("missing_thought_signature", "gemini") == "OTHER"
    assert format_stop_reason("no_image", "openai_chat") == "content_filter"


# ---------------------------------------------- capability softening

def test_video_input_is_chat_protocol_capability() -> None:
    chat = get_protocol("openai_chat")
    unified = chat.parse_request(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "video_url", "video_url": {"url": "https://x/v.mp4"}},
                    ],
                }
            ],
        }
    )

    # video survives as a canonical block; an anthropic target drops it
    # with disclosure instead of killing the request (#28.8).
    assert any(b.type == "video" for m in unified.messages for b in m.content)

    anthropic = get_protocol("anthropic_messages")
    payload = anthropic.build_request(
        unified,
        ProtocolContext(source_protocol="openai_chat", target_protocol="anthropic_messages"),
    )
    assert any(w.code == "unsupported_content_dropped" for w in unified.warnings)


# ------------------------------------------------- error envelope details

def test_gemini_error_envelope_keeps_rpc_details() -> None:
    error = StructuredAPIResponseError(
        "Resource has been exhausted",
        error_type="quota_exceeded",
        status_code=429,
        response={
            "error": {
                "code": 429,
                "message": "Resource has been exhausted",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": "GenerateContentPerMinutePerProject"}]},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "45s"},
                ],
            }
        },
    )

    payload = error.to_protocol_payload("gemini")
    envelope = payload["error"]
    assert envelope["status"] == "RESOURCE_EXHAUSTED"
    assert envelope["details"][0]["violations"][0]["quotaId"] == "GenerateContentPerMinutePerProject"
    assert envelope["details"][1]["retryDelay"] == "45s"


def test_retryinfo_alone_is_not_quota() -> None:
    # RetryInfo is generic retry advice (rides UNAVAILABLE too) — only
    # QuotaFailure (or explicit quota vocabulary) signals quota.
    unavailable = {"code": 503, "status": "UNAVAILABLE", "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "13s"}]}
    assert _structured_quota_signal(unavailable, unavailable.get("details"), "try again", 503) is False

    quota = {"code": 429, "status": "RESOURCE_EXHAUSTED"}
    assert _structured_quota_signal(quota, quota.get("details"), "", 429) is True
