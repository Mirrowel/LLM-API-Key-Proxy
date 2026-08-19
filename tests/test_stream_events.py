from __future__ import annotations

from rotator_library.streaming import stream_event_from_sse_chunk


def test_stream_event_from_sse_chunk_detects_visible_chat_delta() -> None:
    event = stream_event_from_sse_chunk('data: {"choices":[{"delta":{"content":"hi"}}]}\n\n')

    assert event.event_type == "delta"
    assert event.visible_output is True


def test_stream_event_from_sse_chunk_treats_error_and_done_as_not_visible() -> None:
    error_event = stream_event_from_sse_chunk('data: {"error":{"type":"rate_limit"}}\n\n')
    done_event = stream_event_from_sse_chunk("data: [DONE]\n\n")

    assert error_event.event_type == "error"
    assert error_event.visible_output is False
    assert done_event.event_type == "completed"
    assert done_event.visible_output is False


def test_stream_event_from_sse_chunk_malformed_fails_closed() -> None:
    event = stream_event_from_sse_chunk("data: not-json\n\n")

    assert event.event_type == "metadata"
    assert event.visible_output is False
