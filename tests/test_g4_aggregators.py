"""G4 aggregator pins: n>1 siblings, provider-finish-wins, event: frames.

Both chat aggregators share one ruling: every choice is aggregated by its
provider index, the provider's own finish reason wins, and tool_calls is
inferred only when the provider never stated a reason for that choice.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from proxy_app.route_helpers import _aggregate_chat_chunks
from rotator_library.client.executor import RequestExecutor
from rotator_library.transaction_logger import TransactionLogger

AGGREGATORS: list[tuple[str, Callable[[list[dict[str, Any]]], dict[str, Any]]]] = [
    ("transaction_logger", TransactionLogger.assemble_streaming_response),
    ("route_helpers", _aggregate_chat_chunks),
]


def _choice(
    index: int,
    *,
    content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"index": index, "delta": delta, "finish_reason": finish_reason}


def _chunk(*choices: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-test",
        "choices": list(choices),
        **extra,
    }


def _tool_call(index: int = 0, *, name: str = "f", arguments: str = "{}") -> dict[str, Any]:
    return {
        "index": index,
        "id": "call-1",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


@pytest.mark.parametrize("name,aggregate", AGGREGATORS)
def test_n_gt_1_stream_keeps_both_choices_with_finishes(name, aggregate):
    # Two sibling naive choices: content "alpha"/"beta", finishes stop/length.
    chunks = [
        _chunk(
            _choice(0, content="alpha"),
            _choice(1, content="beta"),
        ),
        _chunk(
            _choice(0, finish_reason="stop"),
            _choice(1, finish_reason="length"),
        ),
    ]

    result = aggregate(chunks)

    assert [c["index"] for c in result["choices"]] == [0, 1]
    assert result["choices"][0]["message"]["content"] == "alpha"
    assert result["choices"][1]["message"]["content"] == "beta"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["choices"][1]["finish_reason"] == "length"


@pytest.mark.parametrize("name,aggregate", AGGREGATORS)
def test_provider_finish_stop_after_tool_calls_stays_stop(name, aggregate):
    # Tool calls arrive, then the provider explicitly states "stop".
    chunks = [
        _chunk(_choice(0, tool_calls=[_tool_call()])),
        _chunk(_choice(0, finish_reason="stop")),
    ]

    result = aggregate(chunks)

    choice = result["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["tool_calls"][0]["id"] == "call-1"


@pytest.mark.parametrize("name,aggregate", AGGREGATORS)
def test_missing_finish_with_tool_calls_infers_tool_calls(name, aggregate):
    # No provider-stated finish anywhere: infer from the tool calls seen.
    chunks = [
        _chunk(_choice(0, tool_calls=[_tool_call()])),
    ]

    result = aggregate(chunks)

    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "f"


@pytest.mark.parametrize("name,aggregate", AGGREGATORS)
def test_missing_finish_without_tool_calls_infers_stop(name, aggregate):
    chunks = [_chunk(_choice(0, content="hello"))]

    result = aggregate(chunks)

    assert result["choices"][0]["finish_reason"] == "stop"


def test_aggregators_agree_on_shared_scenario():
    chunks = [
        _chunk(
            _choice(0, content="a"),
            _choice(1, content="b"),
            _choice(2, tool_calls=[_tool_call()]),
        ),
        _chunk(
            _choice(0, finish_reason="stop"),
            _choice(1, finish_reason="tool_calls"),
        ),
    ]

    assert _aggregate_chat_chunks(chunks) == TransactionLogger.assemble_streaming_response(
        chunks
    )


# ---------------------------------------------------------------------------
# Wrapper: event:-framed (anthropic/responses) frames must reach the L1.


class _CapturingLogger:
    def __init__(self) -> None:
        self.stream_chunks: list[Any] = []
        self.passes: list[str] = []
        self.responses: list[Any] = []

    def log_transform_pass(self, name, data=None, **kwargs):  # noqa: ANN001
        self.passes.append(name)

    def log_stream_chunk(self, chunk):  # noqa: ANN001
        self.stream_chunks.append(chunk)

    def log_response(self, response, **kwargs):  # noqa: ANN001
        self.responses.append(response)

    def finalize_metadata(self, **kwargs):  # noqa: ANN001
        return None

    def log_transform_error(self, *args, **kwargs):  # noqa: ANN001
        return None


async def test_stream_wrapper_captures_event_framed_anthropic_chunks():
    logger = _CapturingLogger()
    executor = RequestExecutor.__new__(RequestExecutor)

    async def stream():
        yield ": keep-alive comment\n"
        yield "event: message_start\n"
        yield 'data: {"type":"message_start","message":{"id":"msg_1"}}\n'
        yield "\n"
        yield "event: content_block_delta\n"
        yield 'data: {"type":"content_block_delta","index":0,\n'
        yield 'data: "delta":{"type":"text_delta","text":"Hi"}}\n'
        yield "\n"
        yield "event: message_stop\n"
        yield 'data: {"type":"message_stop"}\n'
        yield "\n"
        yield "data: [DONE]\n"
        yield "\n"

    collected = [
        line
        async for line in executor._transaction_logging_stream_wrapper(
            stream(), logger, {}
        )
    ]

    assert collected[-1] == "\n"
    assert any(line == "data: [DONE]\n" for line in collected)

    events = [c.get("event") for c in logger.stream_chunks if isinstance(c, dict)]
    assert "message_start" in events
    assert "content_block_delta" in events
    assert "message_stop" in events

    # Multi-field (multiple data lines) frame is joined before parsing.
    delta = next(c for c in logger.stream_chunks if c.get("event") == "content_block_delta")
    assert delta["delta"]["text"] == "Hi"

    # The terminating [DONE] sentinel is an event, never a parsed chunk.
    assert all(c.get("event") != "[DONE]" for c in logger.stream_chunks if isinstance(c, dict))
    assert any(p == "stream_done_event" for p in logger.passes)
    # The assembled L1 envelope is still emitted, chat-shaped best effort.
    assert logger.responses
    assert logger.responses[-1]["object"] == "chat.completion"
