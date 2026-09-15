# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Fallback eligibility policy for ordered route chains."""

from __future__ import annotations

from .types import DEFAULT_FAILOVER_ON, DEFAULT_STOP_ON, HARD_STOP_ON, FallbackGroup


_ALIASES = {
    "auth": "authentication",
    "invalid_api_key": "authentication",
    "invalid_key": "authentication",
    "unauthorized": "authentication",
    "permission": "forbidden",
    "permission_denied": "forbidden",
    "access_denied": "forbidden",
    "bad_request": "invalid_request",
    "invalid_argument": "invalid_request",
    "bad_argument": "invalid_request",
    "validation": "invalid_request",
    "permanent": "invalid_request",
    "context_length": "context_window_exceeded",
    "context_length_exceeded": "context_window_exceeded",
    "context_window": "context_window_exceeded",
    "too_many_tokens": "context_window_exceeded",
    "max_tokens_exceeded": "context_window_exceeded",
    "token_limit_exceeded": "context_window_exceeded",
    "pre_request_callback": "pre_request_callback_error",
    "configuration": "configuration_error",
    "config": "configuration_error",
    "quota": "quota_exceeded",
    "resource_exhausted": "quota_exceeded",
    "capacity": "rate_limit",
    "rate_limited": "rate_limit",
    "too_many_requests": "rate_limit",
    "transient": "server_error",
    "unavailable": "server_error",
    "network": "api_connection",
    "connection": "api_connection",
    "deadline_exceeded": "api_connection",
    "timeout": "api_connection",
    "400": "invalid_request",
    "401": "authentication",
    "403": "forbidden",
    "429": "rate_limit",
    "500": "server_error",
    "502": "server_error",
    "503": "server_error",
    "504": "server_error",
}


def normalize_route_error_type(error_type: str | None) -> str:
    """Return the policy vocabulary for a raw classifier or config value.

    Routing config is user-facing, while the executor consumes classifier output.
    Keeping normalization in one place prevents aliases such as `auth` from
    bypassing non-overridable hard-stop categories.
    """

    normalized = str(error_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    return _ALIASES.get(normalized, normalized)


def normalize_route_error_set(values: frozenset[str]) -> frozenset[str]:
    """Normalize a policy set while preserving unknown custom categories."""

    return frozenset(normalize_route_error_type(value) for value in values)


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

        normalized = normalize_route_error_type(error_type)
        if stream and emitted_output:
            return False
        if normalized in HARD_STOP_ON:
            return False
        active_stop = normalize_route_error_set(group.stop_on if group else DEFAULT_STOP_ON)
        active_failover = normalize_route_error_set(group.failover_on if group else DEFAULT_FAILOVER_ON)
        if normalized in active_stop:
            return False
        return normalized in active_failover
