from __future__ import annotations

from rotator_library.client.stream_retry_policy import can_retry_stream_after_error as compat_retry_policy
from rotator_library.streaming.policy import can_retry_stream_after_error, is_visible_stream_output


def test_reasoning_only_retry_policy_is_preserved() -> None:
    reasoning_chunk = 'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'
    text_chunk = 'data: {"choices":[{"delta":{"content":"visible"}}]}\n\n'

    assert can_retry_stream_after_error(None, False) is True
    assert can_retry_stream_after_error(reasoning_chunk, True) is True
    assert can_retry_stream_after_error(reasoning_chunk, False) is False
    assert can_retry_stream_after_error(text_chunk, True) is False
    assert compat_retry_policy(reasoning_chunk, True) is True


def test_heartbeat_comments_do_not_block_stream_retry() -> None:
    assert can_retry_stream_after_error(': heartbeat\n\n', False) is True


def test_visible_output_detection_for_chat_chunks() -> None:
    assert is_visible_stream_output('data: {"choices":[{"delta":{"content":"hello"}}]}\n\n') is True
    assert is_visible_stream_output('data: {"choices":[{"delta":{"tool_calls":[{"id":"call_1"}]}}]}\n\n') is True
    assert is_visible_stream_output('data: {"error":{"type":"rate_limit"}}\n\n') is False
    assert is_visible_stream_output('event: error\ndata: {"type":"error","error":{"message":"x"}}\n\n') is False
    assert is_visible_stream_output('data: {"type":"error","error":{"message":"x"}}\n\n') is False
    assert is_visible_stream_output(': heartbeat\n\n') is False
    assert is_visible_stream_output("data: [DONE]\n\n") is False
    assert is_visible_stream_output("not-sse") is True


def test_visible_output_detection_for_responses_events() -> None:
    assert is_visible_stream_output('data: {"event_type":"response.output_text.delta","delta":"hi"}\n\n', protocol="responses") is True
    assert is_visible_stream_output('data: {"event_type":"response.output_text.delta","delta":"hi"}\n\n') is True
    assert is_visible_stream_output('data: {"event_type":"response.function_call_arguments.delta","delta":"{\\"x\\":"}\n\n') is True
    assert is_visible_stream_output('data: {"event_type":"response.failed","error":{"message":"x"}}\n\n', protocol="responses") is False
    assert is_visible_stream_output('event: response.failed\ndata: {"error":{"message":"x"}}\n\n', protocol="responses") is False
