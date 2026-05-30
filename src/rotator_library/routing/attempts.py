# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Helpers for creating per-target request contexts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from ..core.types import RequestContext
from .types import RouteTarget


def clone_context_for_target(
    context: RequestContext,
    target: RouteTarget,
    *,
    credentials: Sequence[str] | None = None,
    usage_manager_key: str | None = None,
    target_index: int = 0,
) -> RequestContext:
    """Return a target-specific request context without mutating the original.

    Fallback routing must preserve the original request context for traceability
    and for safe retry decisions. This helper copies the request kwargs, updates
    the selected model/provider, and carries routing metadata for executor traces.
    """

    kwargs: dict[str, Any] = dict(context.kwargs)
    kwargs["model"] = target.prefixed_model
    return replace(
        context,
        model=target.prefixed_model,
        provider=target.provider,
        kwargs=kwargs,
        credentials=list(credentials) if credentials is not None else list(context.credentials),
        usage_manager_key=usage_manager_key if usage_manager_key is not None else target.provider,
        routing_target_index=target_index,
    )
