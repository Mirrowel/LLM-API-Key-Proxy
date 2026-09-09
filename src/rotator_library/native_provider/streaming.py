# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Streaming helpers for provider-native execution."""

from __future__ import annotations

from typing import Any

from ..streaming import StreamEvent, stream_event_from_sse_chunk


def stream_event_payload(event: Any) -> Any:
    """Return a JSON-safe payload for stream field-cache and trace passes."""

    if hasattr(event, "to_dict"):
        return event.to_dict()
    return event


def provider_supports_native_streaming(provider: Any, *, model: str = "", operation: str = "chat") -> bool:
    """Return explicit provider native-streaming support.

    Providers must opt in. Missing methods and exceptions fail closed so routed
    streaming can fallback before output rather than accidentally claiming native
    streaming support.
    """

    method = getattr(provider, "supports_native_streaming", None)
    if not method:
        return False
    try:
        return bool(method(model=model, operation=operation))
    except TypeError:
        return bool(method(model, operation))
    except Exception:
        return False


def native_stream_event_from_formatted(formatted: Any, *, protocol: str = "openai_chat") -> StreamEvent:
    """Convert a formatted native stream chunk into the common event seam."""

    if isinstance(formatted, str):
        return stream_event_from_sse_chunk(formatted, protocol=protocol)
    return StreamEvent("parsed_chunk", protocol=protocol, data=formatted, raw=formatted)
