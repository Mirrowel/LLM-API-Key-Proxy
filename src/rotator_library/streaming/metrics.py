# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Stream lifecycle metrics and stall monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .events import StreamEvent


Clock = Callable[[], float]


@dataclass
class StreamMetrics:
    """Timing and lifecycle counters for one stream."""

    started_at: float
    first_byte_at: Optional[float] = None
    first_visible_output_at: Optional[float] = None
    last_chunk_at: Optional[float] = None
    completed_at: Optional[float] = None
    chunk_count: int = 0
    visible_chunk_count: int = 0
    error_count: int = 0
    cancelled: bool = False

    @property
    def ttfb_seconds(self) -> Optional[float]:
        return None if self.first_byte_at is None else self.first_byte_at - self.started_at

    @property
    def ttft_seconds(self) -> Optional[float]:
        return None if self.first_visible_output_at is None else self.first_visible_output_at - self.started_at

    @property
    def duration_seconds(self) -> Optional[float]:
        return None if self.completed_at is None else self.completed_at - self.started_at

    @property
    def idle_seconds(self) -> Optional[float]:
        if self.last_chunk_at is None or self.completed_at is None:
            return None
        return self.completed_at - self.last_chunk_at

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "first_byte_at": self.first_byte_at,
            "first_visible_output_at": self.first_visible_output_at,
            "last_chunk_at": self.last_chunk_at,
            "completed_at": self.completed_at,
            "chunk_count": self.chunk_count,
            "visible_chunk_count": self.visible_chunk_count,
            "error_count": self.error_count,
            "cancelled": self.cancelled,
            "ttfb_seconds": self.ttfb_seconds,
            "ttft_seconds": self.ttft_seconds,
            "duration_seconds": self.duration_seconds,
            "idle_seconds": self.idle_seconds,
        }


class StreamMonitor:
    """Record stream lifecycle events without changing stream behavior."""

    def __init__(self, *, clock: Clock) -> None:
        self._clock = clock
        self.metrics = StreamMetrics(started_at=clock())

    def record_event(self, event: StreamEvent) -> None:
        now = self._clock()
        if self.metrics.first_byte_at is None:
            self.metrics.first_byte_at = now
        self.metrics.last_chunk_at = now
        self.metrics.chunk_count += 1
        if event.visible_output:
            self.metrics.visible_chunk_count += 1
            if self.metrics.first_visible_output_at is None:
                self.metrics.first_visible_output_at = now
        if event.event_type == "error":
            self.metrics.error_count += 1

    def complete(self) -> None:
        self.metrics.completed_at = self._clock()

    def cancel(self) -> None:
        self.metrics.cancelled = True
        self.metrics.completed_at = self._clock()

    def is_stalled(self, timeout_seconds: float) -> bool:
        if timeout_seconds <= 0 or self.metrics.last_chunk_at is None:
            return False
        return self._clock() - self.metrics.last_chunk_at > timeout_seconds
