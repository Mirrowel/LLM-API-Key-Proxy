from __future__ import annotations

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
