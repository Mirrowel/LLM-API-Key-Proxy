# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G14 Ollama-native protocol tests: adapter, stream, transport, routes."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from proxy_app import main as proxy_main
from rotator_library.native_provider.http import _NDJSONFrameDecoder
from rotator_library.protocols import (
    OPERATION_EMBEDDINGS,
    OPERATION_OLLAMA_CHAT,
    OPERATION_OLLAMA_GENERATE,
    ContentBlock,
    MediaSource,
    ProtocolContext,
    ToolCall,
    UnifiedMessage,
    UnifiedRequest,
    UnifiedStreamEvent,
    Usage,
    get_protocol,
)
from rotator_library.protocols import streaming
from rotator_library.protocols.validation import validate_generative_request
from rotator_library.providers import PROVIDER_PLUGINS


def _ollama():
    return get_protocol("ollama")


# ---------------------------------------------------------------------------
# Adapter round-trips
# ---------------------------------------------------------------------------


def test_chat_round_trip_preserves_tools_images_thinking_and_options() -> None:
    adapter = _ollama()
    raw = {
        "model": "llama3",
        "stream": True,
        "messages": [
            {"role": "user", "content": "look at this", "images": ["aGVsbG8="]},
            {
                "role": "assistant",
                "content": "",
                "thinking": "let me look",
                "tool_calls": [{"function": {"name": "lookup", "arguments": {"q": "x"}}}],
            },
            {"role": "tool", "content": "found", "tool_name": "lookup"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "look things up",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ],
        "think": "high",
        "options": {
            "num_predict": 128,
            "temperature": 0.2,
            "top_p": 0.9,
            "stop": ["END"],
            "num_ctx": 4096,
        },
        "keep_alive": "5m",
    }

    unified = adapter.parse_request(raw, ProtocolContext())
    assert unified.operation == OPERATION_OLLAMA_CHAT
    assert unified.tools[0].name == "lookup"
    assert unified.tools[0].input_schema["properties"]["q"] == {"type": "string"}
    # options mapping into canonical homes; Modelfile-only key stays verbatim
    assert unified.generation_params["max_output_tokens"] == 128
    assert unified.generation_params["temperature"] == 0.2
    assert unified.generation_params["top_p"] == 0.9
    assert unified.generation_params["stop_sequences"] == ["END"]
    assert unified.generation_params["options"] == {"num_ctx": 4096}
    # think -> canonical reasoning
    assert unified.generation_params["reasoning"] == {"enabled": True, "effort": "high"}
    # images -> canonical media blocks (both directions)
    image = next(block for block in unified.messages[0].content if block.type == "image")
    assert isinstance(image.source, MediaSource)
    assert image.source.data == "aGVsbG8="
    # assistant tool_calls carry object arguments and synthetic ids
    call = unified.messages[1].tool_calls[0]
    assert call.name == "lookup" and call.arguments == {"q": "x"}
    assert call.extra.get("synthetic_id") is True
    # tool result turn
    result = unified.messages[2].content[0]
    assert result.type == "tool_result" and result.tool_result.name == "lookup"
    assert result.tool_result.content == "found"

    built = adapter.build_request(unified)
    assert built["messages"][0]["images"] == ["aGVsbG8="]
    assert built["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"q": "x"}
    assert built["messages"][1]["thinking"] == "let me look"
    assert built["messages"][2] == {"role": "tool", "content": "found", "tool_name": "lookup"}
    assert built["tools"][0]["function"]["parameters"]["properties"]["q"] == {"type": "string"}
    assert built["think"] == "high"
    assert built["options"]["num_predict"] == 128
    assert built["options"]["num_ctx"] == 4096
    assert built["keep_alive"] == "5m"
    # stable same-protocol round-trip
    assert adapter.build_request(adapter.parse_request(built)) == built


def test_think_bool_and_effort_mapping_both_directions() -> None:
    adapter = _ollama()
    assert adapter.parse_request({"model": "m", "prompt": "x", "think": True}).generation_params["reasoning"] == {"enabled": True}
    assert adapter.parse_request({"model": "m", "prompt": "x", "think": False}).generation_params["reasoning"] == {"enabled": False}
    assert adapter.parse_request({"model": "m", "prompt": "x", "think": "max"}).generation_params["reasoning"] == {
        "enabled": True,
        "effort": "max",
    }

    request = UnifiedRequest(operation=OPERATION_OLLAMA_GENERATE, model="m", generation_params={"reasoning": {"enabled": True, "effort": "medium"}})
    assert adapter.build_request(request)["think"] == "medium"
    disabled = UnifiedRequest(operation=OPERATION_OLLAMA_GENERATE, model="m", generation_params={"reasoning": {"enabled": False}})
    assert adapter.build_request(disabled)["think"] is False


def test_generate_and_embedding_shapes() -> None:
    adapter = _ollama()
    generate = adapter.parse_request({"model": "llama3", "prompt": "write", "options": {"temperature": 0.1}})
    assert generate.operation == OPERATION_OLLAMA_GENERATE
    assert adapter.build_request(generate)["prompt"] == "write"

    embeddings = adapter.parse_request({"model": "llama3", "input": "embed me"})
    assert embeddings.operation == OPERATION_EMBEDDINGS
    assert adapter.build_request(embeddings)["input"] == "embed me"

    # newer response shape -> embeddings list
    response = adapter.parse_response({"model": "llama3", "embeddings": [[0.1, 0.2]]})
    assert response.data == [[0.1, 0.2]]
    assert adapter.format_response(response)["embeddings"] == [[0.1, 0.2]]


def test_done_reason_mapping_and_native_preservation() -> None:
    adapter = _ollama()
    stopped = adapter.parse_response({"model": "m", "message": {"role": "assistant", "content": "x"}, "done": True, "done_reason": "stop"})
    assert stopped.stop_reason == "stop"
    assert adapter.format_response(stopped)["done_reason"] == "stop"

    length = adapter.parse_response({"model": "m", "response": "x", "done": True, "done_reason": "length"})
    assert length.stop_reason == "max_tokens"
    assert adapter.format_response(length)["done_reason"] == "length"

    # load/unload are not clean completions -> canonical error, native spelling kept
    unloaded = adapter.parse_response({"model": "m", "response": "", "done": True, "done_reason": "unload"})
    assert unloaded.stop_reason == "error"
    assert unloaded.extra["done_reason"] == "unload"
    assert adapter.format_response(unloaded)["done_reason"] == "unload"


# ---------------------------------------------------------------------------
# Stream parsing (fragment accumulation)
# ---------------------------------------------------------------------------


def test_stream_accumulates_tool_calls_until_done() -> None:
    adapter = _ollama()
    context = ProtocolContext()
    first = adapter.parse_stream_event({"model": "m", "message": {"role": "assistant", "content": "Hel"}, "done": False}, context)
    fragment_one = adapter.parse_stream_event(
        {"model": "m", "message": {"role": "assistant", "content": "lo ", "tool_calls": [{"function": {"name": "lookup", "arguments": '{"q":'}}]}, "done": False},
        context,
    )
    fragment_two = adapter.parse_stream_event(
        {"model": "m", "message": {"role": "assistant", "content": "world", "tool_calls": [{"function": {"name": "lookup", "arguments": '"x"}'}}]}, "done": False},
        context,
    )
    final = adapter.parse_stream_event(
        {
            "model": "m",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "length",
            "prompt_eval_count": 5,
            "eval_count": 7,
            "total_duration": 1000,
        },
        context,
    )

    assert first.type == "message_delta"
    assert fragment_one.delta.tool_calls == [] and fragment_two.delta.tool_calls == []
    assert final.type == "done"
    assert final.stop_reason == "max_tokens"
    assert [call.name for call in final.delta.tool_calls] == ["lookup"]
    assert final.delta.tool_calls[0].arguments == {"q": "x"}
    assert final.usage is not None
    assert final.usage.input_tokens == 5 and final.usage.output_tokens == 7
    assert final.usage.raw["total_duration"] == 1000
    # buffer is drained so a sibling stream on the same context starts clean
    assert "_ollama_stream_tool_calls" not in context.metadata


def test_stream_response_and_thinking_fields() -> None:
    adapter = _ollama()
    thinking = adapter.parse_stream_event({"model": "m", "response": "", "thinking": "hmm", "done": False})
    assert thinking.delta.reasoning[0].text == "hmm"
    assert any(block.type == "reasoning" for block in thinking.delta.content)
    delta = adapter.parse_stream_event({"model": "m", "response": "he", "done": False})
    assert delta.delta.content[0].text == "he"


# ---------------------------------------------------------------------------
# NDJSON decoder
# ---------------------------------------------------------------------------


def test_ndjson_decoder_preserves_raw_and_treats_done_as_frame() -> None:
    decoder = _NDJSONFrameDecoder()
    line = '  {"model":"llama3","done": false}  '
    frames = decoder.feed(line)
    assert len(frames) == 1
    assert frames[0].raw == line
    assert frames[0].parsed == {"model": "llama3", "done": False}

    done = decoder.feed('{"model":"llama3","done":true,"done_reason":"stop"}')
    assert done[0].parsed["done"] is True
    assert done[0].parsed["done_reason"] == "stop"

    assert decoder.feed("") == []
    assert decoder.feed(b'{"a":1}')[0].parsed == {"a": 1}
    assert decoder.flush() == []

    undecodable = decoder.feed("not json")[0]
    assert undecodable.parsed == "not json" and undecodable.raw == "not json"


# ---------------------------------------------------------------------------
# Stream formatter
# ---------------------------------------------------------------------------


def test_ollama_stream_formatter_content_tool_done_error_frames() -> None:
    state = streaming.stream_format_state(None, "ollama")
    content = streaming.format_canonical_stream_event(
        UnifiedStreamEvent(
            type="message_delta",
            source_protocol="ollama",
            delta=UnifiedMessage(role="assistant", content=[ContentBlock(type="text", text="hi")]),
        ),
        "ollama",
        state=state,
    )
    assert json.loads(content[0])["message"]["content"] == "hi"
    assert json.loads(content[0])["done"] is False

    tool = streaming.format_canonical_stream_event(
        UnifiedStreamEvent(
            type="message_delta",
            source_protocol="ollama",
            delta=UnifiedMessage(
                role="assistant",
                content=[ContentBlock(type="tool_call", tool_call=ToolCall(id="call_0", name="lookup", arguments={"q": "x"}, index=0))],
            ),
        ),
        "ollama",
        state=state,
    )
    parsed_tool = json.loads(tool[0])
    assert parsed_tool["message"]["tool_calls"][0]["function"]["arguments"] == {"q": "x"}

    done = streaming.format_canonical_stream_event(
        UnifiedStreamEvent(type="done", source_protocol="ollama", stop_reason="stop", usage=Usage(input_tokens=3, output_tokens=5, raw={"total_duration": 42})),
        "ollama",
        state=state,
    )
    parsed_done = json.loads(done[0])
    assert parsed_done["done"] is True
    assert parsed_done["done_reason"] == "stop"
    assert parsed_done["prompt_eval_count"] == 3 and parsed_done["eval_count"] == 5
    assert parsed_done["total_duration"] == 42
    assert state.terminal is True


def test_ollama_stream_formatter_error_frame() -> None:
    state = streaming.stream_format_state(None, "ollama")
    frames = streaming.format_canonical_stream_event(
        UnifiedStreamEvent(type="error", error={"message": "boom"}), "ollama", state=state
    )
    assert "error" in json.loads(frames[0])
    assert state.terminal is True


def test_ollama_parse_and_format_converter_round_trip() -> None:
    converter = streaming.ProtocolStreamConverter(_ollama(), _ollama(), ProtocolContext(model="llama3"))
    frames: list[str] = []
    frames += converter.convert({"model": "llama3", "message": {"role": "assistant", "content": "He"}, "done": False})
    frames += converter.convert({"model": "llama3", "message": {"role": "assistant", "content": "llo"}, "done": False})
    frames += converter.convert(
        {
            "model": "llama3",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 2,
        }
    )

    parsed = [json.loads(frame) for frame in frames]
    text = "".join(item["message"]["content"] for item in parsed if not item["done"])
    assert text == "Hello"
    assert parsed[-1]["done"] is True
    assert parsed[-1]["done_reason"] == "stop"


# ---------------------------------------------------------------------------
# Validation capabilities
# ---------------------------------------------------------------------------


def test_validation_keeps_image_and_discloses_audio_drop() -> None:
    request = UnifiedRequest(
        operation="chat",
        model="llama3",
        messages=[
            UnifiedMessage(
                role="user",
                content=[
                    ContentBlock(type="image", source=MediaSource(kind="data", data="aGk=")),
                    ContentBlock(type="audio", source=MediaSource(kind="data", data="YXVkaW8=", media_type="audio/wav")),
                ],
            )
        ],
        source_protocol="openai_chat",
    )
    validate_generative_request(request, "ollama", ProtocolContext(source_protocol="openai_chat"))
    types = [block.type for block in request.messages[0].content]
    assert "image" in types and "audio" not in types
    assert any(warning.code == "unsupported_content_dropped" and warning.field == "message:0.content[1]" for warning in request.warnings)


# ---------------------------------------------------------------------------
# Provider declaration
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHTTP:
    def __init__(self, payload):
        self.payload = payload
        self.calls: list[str] = []

    async def get(self, url, **kwargs):
        self.calls.append(url)
        return _FakeResponse(self.payload)


def test_ollama_provider_endpoints_models_and_optional_auth() -> None:
    provider = PROVIDER_PLUGINS["ollama"]()
    assert provider.protocol_name == "ollama"
    assert provider.native_streaming_supported is True

    assert provider.get_native_endpoint(operation="ollama_chat").endswith("/api/chat")
    assert provider.get_native_endpoint(operation="ollama_generate").endswith("/api/generate")
    assert provider.get_native_endpoint(operation="embeddings").endswith("/api/embed")

    real = provider.get_native_headers("sk-secret")
    assert real["Authorization"] == "Bearer sk-secret"
    assert "Authorization" not in provider.get_native_headers("__proxy_no_auth__")
    assert "Authorization" not in provider.get_native_headers("")

    assert provider.get_native_operation(request={"messages": []}, stream=True) == "ollama_chat"
    assert provider.get_native_operation(request={"prompt": "x"}) == "ollama_generate"


async def test_ollama_provider_models_from_tags_shape() -> None:
    provider = PROVIDER_PLUGINS["ollama"]()
    fake = _FakeHTTP({"models": [{"name": "llama3:latest"}, {"model": "qwen2"}]})

    models = await provider.get_models("", fake)

    assert models == ["ollama/llama3:latest", "ollama/qwen2"]
    assert fake.calls[-1].endswith("/api/tags")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class OllamaRuntimeClient:
    """Fake runtime capturing native-client handoff and emitting NDJSON."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def agenerate(self, payload, *, input_protocol, request=None, **kwargs):
        self.calls.append({"payload": payload, "input_protocol": input_protocol, "kwargs": dict(kwargs)})
        if payload.get("stream"):
            async def stream():
                yield json.dumps({"model": payload["model"], "message": {"role": "assistant", "content": "he"}, "done": False}) + "\n"
                yield json.dumps({"model": payload["model"], "message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop", "eval_count": 2}) + "\n"

            return stream()
        if input_protocol == "ollama" and "input" in payload:
            return {"model": payload["model"], "embeddings": [[0.1, 0.2]], "total_duration": 7}
        return {"model": payload["model"], "response": "generated", "done": True, "done_reason": "stop"}

    async def get_all_available_models(self, grouped=False):
        return ["ollama/llama3:latest", "openai/gpt-test"]


def _ollama_client() -> tuple[TestClient, OllamaRuntimeClient]:
    proxy_main.PROXY_API_KEY = None
    proxy_main.ENABLE_RAW_LOGGING = False
    rotating = OllamaRuntimeClient()
    proxy_main.app.state.rotating_client = rotating
    return TestClient(proxy_main.app), rotating


def test_api_chat_streams_ndjson_by_default() -> None:
    client, rotating = _ollama_client()

    response = client.post("/api/chat", json={"model": "ollama/llama3", "messages": [{"role": "user", "content": "hi"}]})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert lines[0]["message"]["content"] == "he" and lines[0]["done"] is False
    assert lines[1]["done"] is True and lines[1]["done_reason"] == "stop"
    assert rotating.calls[0]["input_protocol"] == "ollama"
    assert rotating.calls[0]["kwargs"]["_requested_operation"] == "ollama_chat"


def test_api_generate_non_stream_returns_json_object() -> None:
    client, rotating = _ollama_client()

    response = client.post(
        "/api/generate",
        json={"model": "ollama/llama3", "prompt": "write", "stream": False},
    )

    assert response.status_code == 200
    assert response.json()["response"] == "generated"
    assert rotating.calls[0]["input_protocol"] == "ollama"
    assert rotating.calls[0]["kwargs"]["_requested_operation"] == "ollama_generate"


def test_api_embed_returns_embedding_object() -> None:
    client, rotating = _ollama_client()

    response = client.post("/api/embed", json={"model": "ollama/llama3", "input": "embed me"})

    assert response.status_code == 200
    assert response.json()["embeddings"] == [[0.1, 0.2]]
    assert rotating.calls[0]["kwargs"]["_requested_operation"] == "embeddings"


def test_api_tags_lists_ollama_models() -> None:
    client, _ = _ollama_client()

    response = client.get("/api/tags")

    assert response.status_code == 200
    assert response.json() == {"models": [{"name": "llama3:latest", "model": "llama3:latest"}]}


def test_api_errors_render_ollama_plain_string_shape() -> None:
    client, _ = _ollama_client()

    response = client.post(
        "/api/chat",
        content="{",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert isinstance(response.json()["error"], str)
    assert "detail" not in response.json()


def test_api_auth_error_uses_ollama_shape() -> None:
    proxy_main.PROXY_API_KEY = "secret-key"
    proxy_main.app.state.rotating_client = OllamaRuntimeClient()
    try:
        response = TestClient(proxy_main.app).post(
            "/api/chat", json={"model": "ollama/llama3", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 401
        assert isinstance(response.json()["error"], str)
    finally:
        proxy_main.PROXY_API_KEY = None
