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
    provider_config: dict[str, Any] | None = None,
    credential_secrets: dict[str, str] | None = None,
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
        provider_config=provider_config if provider_config is not None else context.provider_config,
        credential_secrets=dict(credential_secrets) if credential_secrets is not None else dict(context.credential_secrets),
        routing_target_index=target_index,
        session_tracking_namespace=_namespace_for_target(context.session_tracking_namespace, target),
    )


def _namespace_for_target(namespace: str | None, target: RouteTarget) -> str | None:
    """Rewrite standard session namespaces for the fallback target provider/model."""

    if not namespace or ":provider:" not in namespace or ":model:" not in namespace:
        return namespace
    prefix, _, rest = namespace.partition(":provider:")
    _provider, sep, _model = rest.partition(":model:")
    if not sep:
        return namespace
    return f"{prefix}:provider:{target.provider}:model:{target.prefixed_model}"
