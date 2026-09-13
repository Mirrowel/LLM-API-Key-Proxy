"""G14 pins: Gemini depth completion (hosted tools, responseFormat/speech/
image config, displayName, urlContextMetadata, usage aliases, structured
output dialect, stream candidates, empty candidates, VALIDATED allowlist,
provider version paths, discovery ingress, stream residuals, raw sentinel,
finishMessage)."""

from __future__ import annotations

import json

import pytest

from rotator_library.protocols import get_protocol
from rotator_library.protocols.streaming import format_canonical_stream_event, stream_format_state
from rotator_library.protocols.types import (
    ContentBlock,
    MediaSource,
    ProtocolContext,
    UnifiedMessage,
    UnifiedRequest,
)
from rotator_library.protocols.validation import ProtocolError
from rotator_library.providers.gemini_provider import GeminiProvider, _strip_gemini_api_version
from rotator_library.protocols.opaque_strip import (
    payload_carries_opaque_state,
    strip_foreign_opaque_state,
)


def _ctx(source: str = "gemini", target: str = "gemini") -> ProtocolContext:
    return ProtocolContext(
        source_protocol=source,
        target_protocol=target,
        input_protocol=source,
        provider_protocol=target,
        client_protocol=source,
    )


def _gemini():
    return get_protocol("gemini")


# --------------------------------------------------------------- hosted tools


def test_hosted_tool_vocabulary_parses_without_fabricated_name() -> None:
    gemini = _gemini()
    raw = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "tools": [
            {"computerUse": {}},
            {"mcpServers": [{"serverName": "x"}]},
            {"enterpriseWebSearch": {}},
            {"exaAiSearch": {}},
            {"parallelAiSearch": {}},
        ],
    }

    unified = gemini.parse_request(raw, _ctx())

    names = [tool.name for tool in unified.tools]
    assert names == ["computerUse", "mcpServers", "enterpriseWebSearch", "exaAiSearch", "parallelAiSearch"]
    assert "gemini_tool" not in names
    for tool in unified.tools:
        assert tool.type == "server"
        assert tool.extra["gemini_hosted_tool"] == tool.name

    rebuilt = gemini.build_request(unified, _ctx())
    assert rebuilt["tools"] == raw["tools"]


def test_unmodeled_tool_never_fabricates_gemini_tool_name() -> None:
    gemini = _gemini()
    unified = gemini.parse_request(
        {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "tools": [{"name": "weird", "custom": {"x": 1}}]},
        _ctx(),
    )

    assert unified.tools[0].name == "weird"
    assert unified.tools[0].extra["gemini_unmodeled_tool"] == "weird"


def test_hosted_tool_cross_protocol_rejects_with_tool_name() -> None:
    gemini = _gemini()
    chat = get_protocol("openai_chat")
    unified = gemini.parse_request(
        {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "tools": [{"computerUse": {}}]},
        _ctx(),
    )

    with pytest.raises(ProtocolError, match="computerUse"):
        chat.build_request(unified, _ctx("gemini", "openai_chat"))


# ---------------------------------------------------- responseFormat / config


def test_response_format_root_preserved_same_protocol_and_siblings_disclosed() -> None:
    gemini = _gemini()
    raw = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {
            "responseFormat": {
                "text": {"mimeType": "application/json", "schema": {"type": "OBJECT"}},
                "audio": {"mimeType": "audio/mp3"},
                "image": {"mimeType": "image/png"},
            }
        },
    }

    unified = gemini.parse_request(raw, _ctx())
    rebuilt = gemini.build_request(unified, _ctx())
    assert rebuilt["generationConfig"]["responseFormat"] == raw["generationConfig"]["responseFormat"]
    # responseFormat's audio/image sub-configs are parsed into extensions.
    stored = unified.extensions["gemini"]["generationConfig"]["responseFormat"]
    assert stored["audio"] == {"mimeType": "audio/mp3"}
    assert stored["image"] == {"mimeType": "image/png"}

    chat = get_protocol("openai_chat")
    cross = gemini.parse_request(raw, _ctx())
    chat.build_request(cross, _ctx("gemini", "openai_chat"))
    fields = {warning.field for warning in cross.warnings}
    # The text sub-config maps onto the canonical structured output: no
    # whole-envelope drop warning, one disclosure per audio/image fact.
    assert "generationConfig.responseFormat" not in fields
    assert "generationConfig.responseFormat.audio" in fields
    assert "generationConfig.responseFormat.image" in fields


def test_speech_and_image_config_same_protocol_and_cross_drop() -> None:
    gemini = _gemini()
    raw = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
            "imageConfig": {"aspectRatio": "1:1"},
        },
    }

    unified = gemini.parse_request(raw, _ctx())
    rebuilt = gemini.build_request(unified, _ctx())
    assert rebuilt["generationConfig"] == raw["generationConfig"]

    chat = get_protocol("openai_chat")
    cross = gemini.parse_request(raw, _ctx())
    chat.build_request(cross, _ctx("gemini", "openai_chat"))
    fields = {warning.field for warning in cross.warnings}
    assert "generationConfig.speechConfig" in fields
    assert "generationConfig.imageConfig" in fields


# ------------------------------------------------------------------ displayName


def test_media_display_name_survives_same_and_cross_protocol() -> None:
    gemini = _gemini()
    raw = {
        "contents": [
            {"role": "user", "parts": [{"inlineData": {"mimeType": "image/png", "data": "abc", "displayName": "pic.png"}}]}
        ]
    }
    unified = gemini.parse_request(raw, _ctx())
    rebuilt = gemini.build_request(unified, _ctx())
    assert rebuilt["contents"][0]["parts"][0]["inlineData"]["displayName"] == "pic.png"

    cross_request = UnifiedRequest(
        model="m",
        source_protocol="openai_chat",
        messages=[
            UnifiedMessage(
                role="user",
                content=[
                    ContentBlock(
                        type="image",
                        source=MediaSource(kind="base64", media_type="image/png", data="abc", extra={"displayName": "pic.png"}),
                    )
                ],
            )
        ],
    )
    cross = gemini.build_request(cross_request, _ctx("openai_chat", "gemini"))
    assert cross["contents"][0]["parts"][0]["inlineData"]["displayName"] == "pic.png"


# --------------------------------------------------------------- urlContext


def test_url_context_metadata_lifts_to_annotations_and_replays() -> None:
    gemini = _gemini()
    response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "see"}]},
                "urlContextMetadata": {
                    "urlMetadata": [
                        {"retrievedUrl": "https://example.com/a", "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"}
                    ]
                },
            }
        ]
    }

    unified = gemini.parse_response(response, _ctx())

    annotations = unified.messages[0].content[0].annotations
    assert [a.type for a in annotations] == ["url_context"]
    assert annotations[0].url == "https://example.com/a"

    rebuilt = gemini.format_response(unified, _ctx())
    assert rebuilt["candidates"][0]["urlContextMetadata"] == response["candidates"][0]["urlContextMetadata"]


# ----------------------------------------------------------------- usage


def test_usage_response_token_alias_and_tool_bucket_total() -> None:
    gemini = _gemini()
    usage = gemini.extract_usage(
        {"usageMetadata": {"promptTokenCount": 3, "responseTokenCount": 4, "toolUsePromptTokenCount": 5}}
    )

    assert usage.output_tokens == 4
    assert usage.total_tokens == 12
    assert usage.extra["tool_use_prompt_tokens"] == 5


def test_usage_detail_arrays_replay_same_protocol() -> None:
    gemini = _gemini()
    raw = {
        "responseId": "r",
        "candidates": [{"content": {"role": "model", "parts": [{"text": "x"}]}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 2,
            "totalTokenCount": 3,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1}],
            "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 2}],
            "trafficType": "ON_DEMAND",
            "serviceTier": "standard",
        },
    }

    unified = gemini.parse_response(raw, _ctx())
    rebuilt = gemini.format_response(unified, _ctx())["usageMetadata"]

    assert rebuilt["promptTokensDetails"] == [{"modality": "TEXT", "tokenCount": 1}]
    assert rebuilt["candidatesTokensDetails"] == [{"modality": "TEXT", "tokenCount": 2}]
    assert rebuilt["trafficType"] == "ON_DEMAND"
    assert rebuilt["serviceTier"] == "standard"


# --------------------------------------------------------- structured output


def test_gemini_source_never_fabricates_strict() -> None:
    gemini = _gemini()
    unified = gemini.parse_request(
        {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": {"type": "OBJECT"}},
        },
        _ctx(),
    )

    assert "strict" not in unified.response_format


def test_schema_types_lowercased_for_chat_and_responses_not_gemini() -> None:
    gemini = _gemini()
    raw = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {"type": "OBJECT", "properties": {"a": {"type": "STRING"}}},
        },
    }

    chat_request = gemini.parse_request(raw, _ctx())
    chat_payload = get_protocol("openai_chat").build_request(chat_request, _ctx("gemini", "openai_chat"))
    assert chat_payload["response_format"]["json_schema"]["schema"]["type"] == "object"
    assert chat_payload["response_format"]["json_schema"]["schema"]["properties"]["a"]["type"] == "string"

    responses_request = gemini.parse_request(raw, _ctx())
    responses_payload = get_protocol("responses").build_request(responses_request, _ctx("gemini", "responses"))
    assert responses_payload["text"]["format"]["schema"]["type"] == "object"

    # Gemini itself keeps its OpenAPI dialect (uppercase) — never lowercased.
    foreign = UnifiedRequest(
        model="m",
        source_protocol="openai_chat",
        response_format={"type": "json_schema", "schema": {"type": "OBJECT", "properties": {"a": {"type": "STRING"}}}},
        generation_params={"structured_output": {"type": "json_schema", "schema": {"type": "OBJECT"}}},
    )
    gemini_payload = gemini.build_request(foreign, _ctx("openai_chat", "gemini"))
    assert gemini_payload["generationConfig"]["responseJsonSchema"]["type"] == "OBJECT"


def test_text_enum_mode_is_not_folded_into_json_schema() -> None:
    gemini = _gemini()
    raw = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"responseMimeType": "text/x.enum", "responseSchema": {"type": "STRING", "enum": ["A", "B"]}},
    }

    unified = gemini.parse_request(raw, _ctx())
    assert "structured_output" not in unified.generation_params
    assert unified.generation_params["response_mime_type"] == "text/x.enum"

    # Same-protocol: no false drop warning; the mode replays via extensions.
    rebuilt = gemini.build_request(unified, _ctx())
    assert rebuilt["generationConfig"]["responseMimeType"] == "text/x.enum"

    # Cross-protocol: the enum mode has no representation — disclosed once.
    chat = get_protocol("openai_chat")
    cross = gemini.parse_request(raw, _ctx())
    chat.build_request(cross, _ctx("gemini", "openai_chat"))
    assert [w.field for w in cross.warnings if w.field == "response_mime_type"]


# ---------------------------------------------------------- candidate count


def test_candidate_count_clamped_on_stream_only() -> None:
    gemini = _gemini()
    stream_unified = gemini.parse_request(
        {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "generationConfig": {"candidateCount": 3},
            "stream": True,
        },
        _ctx(),
    )
    streamed = gemini.build_request(stream_unified, _ctx())
    assert streamed["generationConfig"]["candidateCount"] == 1

    nonstream_unified = gemini.parse_request(
        {"contents": [{"role": "user", "parts": [{"text": "hi"}]}], "generationConfig": {"candidateCount": 3}},
        _ctx(),
    )
    nonstream = gemini.build_request(nonstream_unified, _ctx())
    assert nonstream["generationConfig"]["candidateCount"] == 3


# --------------------------------------------------------- empty candidates


def test_empty_candidates_is_honest_success_with_warning() -> None:
    gemini = _gemini()
    unified = gemini.parse_response({"responseId": "r", "candidates": []}, _ctx())

    assert any(w.code == "empty_candidates" for w in unified.warnings)
    assert unified.messages == []
    payload = gemini.format_response(unified, _ctx())
    assert payload["candidates"] == []
    assert "x-proxy-conversion" not in payload


# ------------------------------------------------------- VALIDATED tool choice


def test_validated_tool_choice_preserves_allowlist() -> None:
    gemini = _gemini()
    raw = {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "toolConfig": {"functionCallingConfig": {"mode": "VALIDATED", "allowedFunctionNames": ["lookup"]}},
    }
    unified = gemini.parse_request(raw, _ctx())
    assert unified.generation_params["tool_choice"] == {"mode": "validated", "allowed_names": ["lookup"]}

    rebuilt = gemini.build_request(unified, _ctx())
    assert rebuilt["toolConfig"] == raw["toolConfig"]


# ------------------------------------------------------------ finishMessage


def test_finish_message_preserved_and_disclosed_cross_protocol() -> None:
    gemini = _gemini()
    unified = gemini.parse_response(
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "x"}]}, "finishReason": "STOP", "finishMessage": "done"}
            ]
        },
        _ctx(),
    )

    assert unified.metadata["native_stop_message"] == "done"
    # Same-protocol replay keeps the candidate verbatim.
    assert gemini.format_response(unified, _ctx())["candidates"][0]["finishMessage"] == "done"

    chat_payload = get_protocol("openai_chat").format_response(unified, _ctx("gemini", "openai_chat"))
    assert "x-proxy-conversion" not in chat_payload
    assert any(w.code == "stop_message_dropped" for w in unified.warnings)


# ------------------------------------------------------------ stream residuals


def test_stream_prompt_feedback_and_model_status_survive() -> None:
    gemini = _gemini()
    events = gemini.parse_stream_events(
        {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}, "modelStatus": {"message": "busy"}},
        _ctx(),
    )
    state = stream_format_state(_ctx(), "gemini")
    frames = format_canonical_stream_event(events[0], "gemini", _ctx(), state=state)

    payload = json.loads(frames[0][len("data: ") :].strip())
    assert payload["promptFeedback"] == {"blockReason": "SAFETY"}
    assert payload["modelStatus"] == {"message": "busy"}


def test_stream_usage_details_and_tool_bucket_same_protocol() -> None:
    gemini = _gemini()
    events = gemini.parse_stream_events(
        {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 1,
                "candidatesTokenCount": 2,
                "totalTokenCount": 3,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 1}],
                "trafficType": "ON_DEMAND",
                "toolUsePromptTokenCount": 5,
            },
        },
        _ctx(),
    )
    state = stream_format_state(_ctx(), "gemini")
    frames = format_canonical_stream_event(events[0], "gemini", _ctx(), state=state)

    usage = json.loads(frames[0][len("data: ") :].strip())["usageMetadata"]
    assert usage["promptTokensDetails"] == [{"modality": "TEXT", "tokenCount": 1}]
    assert usage["trafficType"] == "ON_DEMAND"
    assert usage["toolUsePromptTokenCount"] == 5


def test_stream_part_residuals_replay_same_protocol() -> None:
    gemini = _gemini()
    events = gemini.parse_stream_events(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"text": "hi", "partMetadata": {"k": 1}, "videoMetadata": {"fps": 24}}],
                    },
                    "finishReason": "STOP",
                }
            ]
        },
        _ctx(),
    )
    state = stream_format_state(_ctx(), "gemini")
    frames = format_canonical_stream_event(events[0], "gemini", _ctx(), state=state)

    part = json.loads(frames[0][len("data: ") :].strip())["candidates"][0]["content"]["parts"][0]
    assert part["partMetadata"] == {"k": 1}
    assert part["videoMetadata"] == {"fps": 24}


# ------------------------------------------------------------ raw sentinel


def test_raw_strip_injects_sentinel_only_with_signed_sibling() -> None:
    signed = {
        "contents": [
            {
                "parts": [
                    {"functionCall": {"name": "a"}, "thoughtSignature": "sigA"},
                    {"functionCall": {"name": "b"}, "thoughtSignature": "sigB"},
                ]
            }
        ]
    }
    assert strip_foreign_opaque_state(signed, "gemini")
    assert signed["contents"][0]["parts"][0]["thoughtSignature"] == "skip_thought_signature_validator"
    assert signed["contents"][0]["parts"][1]["thoughtSignature"] == "skip_thought_signature_validator"

    single = {"contents": [{"parts": [{"functionCall": {"name": "a"}, "thoughtSignature": "sigA"}]}]}
    assert strip_foreign_opaque_state(single, "gemini")
    # No signed sibling: conservative rule never fabricates the sentinel.
    assert "thoughtSignature" not in single["contents"][0]["parts"][0]

    unsigned = {"contents": [{"parts": [{"functionCall": {"name": "a"}}]}]}
    assert strip_foreign_opaque_state(unsigned, "gemini") is None
    assert payload_carries_opaque_state(unsigned, "gemini") is False


# ------------------------------------------------------------- provider paths


def test_provider_base_strips_version_and_profile_endpoints(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_BASE", "https://mirror.example/v1beta")
    provider = GeminiProvider()

    assert provider.get_native_endpoint("gemini-2.5-pro", "generate") == (
        "https://mirror.example/v1beta/models/gemini-2.5-pro:generateContent"
    )
    assert provider.get_native_endpoint("gemini-2.5-pro", "count_tokens") == (
        "https://mirror.example/v1beta/models/gemini-2.5-pro:countTokens"
    )

    provider.transport_profiles = {
        "custom": {"protocol": "gemini", "endpoint_paths": {"generate": "/custom/{model}:generate"}}
    }
    assert provider.get_native_endpoint("m", "generate", profile="custom") == "https://mirror.example/custom/m:generate"
    assert _strip_gemini_api_version("https://x.example/v1") == "https://x.example"


@pytest.mark.asyncio
async def test_provider_get_models_uses_configured_base_and_paginates(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_BASE", "https://mirror.example/v1beta")
    provider = GeminiProvider()
    calls: list[dict] = []

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _Client:
        async def get(self, url, headers=None, params=None):
            calls.append({"url": url, "params": dict(params or {})})
            if params and params.get("pageToken"):
                return _Response({"models": [{"name": "models/gemini-2.5-flash"}]})
            return _Response({"models": [{"name": "models/gemini-2.5-pro"}], "nextPageToken": "tok"})

    models = await provider.get_models("key", _Client())

    assert models == ["gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash"]
    assert all(call["url"] == "https://mirror.example/v1beta/models" for call in calls)
    assert calls[1]["params"].get("pageToken") == "tok"


# --------------------------------------------------------- discovery ingress


def test_gemini_models_discovery_route_serves_official_shape() -> None:
    from fastapi.testclient import TestClient

    from proxy_app import main as proxy_main

    class _DiscoveryClient:
        async def get_all_available_models(self, grouped=True, **kwargs):
            return ["gemini/gemini-2.5-pro", "openai/gpt-4o", "gemini/gemini-2.5-flash"]

    proxy_main.PROXY_API_KEY = None
    proxy_main.app.state.rotating_client = _DiscoveryClient()
    client = TestClient(proxy_main.app)

    response = client.get("/v1beta/models")

    assert response.status_code == 200
    names = [model["name"] for model in response.json()["models"]]
    assert names == ["models/gemini-2.5-pro", "models/gemini-2.5-flash"]

    paged = client.get("/v1beta/models?pageSize=1")
    assert len(paged.json()["models"]) == 1
    assert paged.json().get("nextPageToken") == "1"


def test_gemini_models_discovery_uses_gemini_error_ladder() -> None:
    from fastapi.testclient import TestClient

    from proxy_app import main as proxy_main

    class _BrokenClient:
        async def get_all_available_models(self, grouped=True, **kwargs):
            raise RuntimeError("discovery exploded")

    proxy_main.PROXY_API_KEY = None
    proxy_main.app.state.rotating_client = _BrokenClient()
    client = TestClient(proxy_main.app)

    response = client.get("/v1beta/models")

    assert response.status_code >= 400
    assert "error" in response.json()
