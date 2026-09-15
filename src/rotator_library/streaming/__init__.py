# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Reusable streaming primitives shared by protocol and provider layers."""

from .events import StreamEvent, stream_event_from_sse_chunk
from .errors import StreamingErrorDecision, decide_streaming_error_action
from .metrics import StreamMetrics, StreamMonitor
from .policy import can_retry_stream_after_error, is_visible_stream_output
from .transport import JSONLineStreamFormatter, SSEStreamFormatter, WebSocketStreamFormatter

__all__ = [
    "JSONLineStreamFormatter",
    "SSEStreamFormatter",
    "StreamEvent",
    "StreamingErrorDecision",
    "StreamMetrics",
    "StreamMonitor",
    "WebSocketStreamFormatter",
    "can_retry_stream_after_error",
    "decide_streaming_error_action",
    "is_visible_stream_output",
    "stream_event_from_sse_chunk",
]
