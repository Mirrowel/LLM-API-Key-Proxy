from __future__ import annotations

from rotator_library.streaming import StreamEvent, StreamMonitor


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_stream_metrics_records_ttfb_ttft_and_counts() -> None:
    clock = FakeClock()
    monitor = StreamMonitor(clock=clock)
    clock.advance(0.5)
    monitor.record_event(StreamEvent("parsed_chunk"))
    clock.advance(0.7)
    monitor.record_event(StreamEvent("delta", visible_output=True))
    clock.advance(0.3)
    monitor.complete()

    assert monitor.metrics.ttfb_seconds == 0.5
    assert monitor.metrics.ttft_seconds == 1.2
    assert monitor.metrics.duration_seconds == 1.5
    assert monitor.metrics.chunk_count == 2
    assert monitor.metrics.visible_chunk_count == 1


def test_stream_monitor_records_cancel_and_stall() -> None:
    clock = FakeClock()
    monitor = StreamMonitor(clock=clock)
    monitor.record_event(StreamEvent("parsed_chunk"))
    clock.advance(5)

    assert monitor.is_stalled(2) is True
    monitor.cancel()
    assert monitor.metrics.cancelled is True
