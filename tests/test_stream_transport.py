from __future__ import annotations

from rotator_library.streaming import JSONLineStreamFormatter, SSEStreamFormatter, StreamEvent, WebSocketStreamFormatter


def test_sse_formatter_outputs_named_event() -> None:
    formatted = SSEStreamFormatter().format_event(StreamEvent("delta", data={"text": "hi"}, visible_output=True))

    assert formatted.startswith("event: delta\n")
    assert "data:" in formatted


def test_websocket_formatter_exposes_future_transport_shape() -> None:
    formatted = WebSocketStreamFormatter().format_event(StreamEvent("delta", data={"text": "hi"}))

    assert formatted["type"] == "delta"
    assert formatted["payload"]["transport"] == "sse"


def test_jsonl_formatter_outputs_one_line_json() -> None:
    formatted = JSONLineStreamFormatter().format_event(StreamEvent("metadata", data={"ok": True}))

    assert formatted.endswith("\n")
    assert "metadata" in formatted
