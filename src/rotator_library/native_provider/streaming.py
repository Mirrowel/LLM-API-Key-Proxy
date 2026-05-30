# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Streaming helpers for provider-native execution."""

from __future__ import annotations

from typing import Any


def stream_event_payload(event: Any) -> Any:
    """Return a JSON-safe payload for stream field-cache and trace passes."""

    if hasattr(event, "to_dict"):
        return event.to_dict()
    return event
