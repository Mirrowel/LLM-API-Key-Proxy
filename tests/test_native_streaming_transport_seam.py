from __future__ import annotations

import pytest

from rotator_library.native_provider.http import NativeHTTPTransport
from rotator_library.native_provider.streaming import native_stream_event_from_formatted, provider_supports_native_streaming
from rotator_library.providers.provider_interface import ProviderInterface


class DefaultProvider(ProviderInterface):
    async def get_models(self, api_key, client):
        return []


class StreamingProvider(DefaultProvider):
    native_streaming_supported = True


def test_provider_native_streaming_support_defaults_false() -> None:
    assert provider_supports_native_streaming(DefaultProvider(), model="gpt-test") is False
    assert provider_supports_native_streaming(StreamingProvider(), model="gpt-test") is True


def test_native_formatted_sse_chunk_uses_common_stream_event_seam() -> None:
    event = native_stream_event_from_formatted('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')

    assert event.visible_output is True
    assert event.event_type == "delta"


class FakeStreamResponse:
    def __init__(self, lines):
        self.lines = lines
        self.raised = False

    def raise_for_status(self):
        self.raised = True

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class FakeStreamContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeHttpxClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def stream(self, method, endpoint, *, headers, json):
        self.calls.append((method, endpoint, headers, json))
        return FakeStreamContext(self.response)


@pytest.mark.asyncio
async def test_native_http_transport_streams_httpx_lines() -> None:
    response = FakeStreamResponse(["", ": heartbeat", "event: content_block_delta", 'data: {"delta":', 'data: "hi"}', "", "data: [DONE]", ""])
    client = FakeHttpxClient(response)

    chunks = [chunk async for chunk in NativeHTTPTransport(client).stream_json_lines("https://example.test/stream", headers={"h": "v"}, payload={"p": True})]

    assert client.calls == [("POST", "https://example.test/stream", {"h": "v"}, {"p": True})]
    assert response.raised is True
    assert chunks == [{"delta": "hi", "type": "content_block_delta"}, "[DONE]"]


class FakeByteResponse:
    def raise_for_status(self):
        pass

    async def aiter_bytes(self):
        yield b'event: response.output_text.delta\ndata: {"a":'
        yield b' 1}\n\ndata: [DO'
        yield b'NE]\n\n'


@pytest.mark.asyncio
async def test_native_http_transport_streams_httpx_bytes() -> None:
    chunks = [chunk async for chunk in NativeHTTPTransport(FakeHttpxClient(FakeByteResponse())).stream_json_lines("url", headers={}, payload={})]

    assert chunks == [{"a": 1, "type": "response.output_text.delta"}, "[DONE]"]
