# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Fallback eligibility policy for ordered route chains."""

from __future__ import annotations

from .types import DEFAULT_FAILOVER_ON, DEFAULT_STOP_ON, FallbackGroup


class FallbackPolicy:
    """Decide whether a failed target may advance to the next target."""

    def should_fallback(
        self,
        error_type: str,
        *,
        group: FallbackGroup | None = None,
        emitted_output: bool = False,
        stream: bool = False,
    ) -> bool:
        """Return whether fallback is allowed for a classified failure."""

        normalized = error_type.lower()
        active_stop = group.stop_on if group else DEFAULT_STOP_ON
        active_failover = group.failover_on if group else DEFAULT_FAILOVER_ON
        if stream and emitted_output:
            return False
        if normalized in active_stop:
            return False
        return normalized in active_failover
