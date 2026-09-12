# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Typed route targets and fallback groups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ExecutionMode = Literal["auto", "native", "custom", "litellm_fallback"]
StreamingFallbackPolicy = Literal["pre_output_only", "never"]

DEFAULT_FAILOVER_ON = frozenset(
    {
        "rate_limit",
        "quota_exceeded",
        "server_error",
        "api_connection",
        "unsupported_operation",
        # Credential-scoped failures advance the chain too (operator-approved
        # matrix, error-reference 5.9): a dead key on provider A must not
        # stop provider B from being tried, and a missing model is a
        # provider-level fact another target may resolve.
        "authentication",
        "forbidden",
        "not_found",
        "conflict",
        # Human-friendly aliases for config files; classifier output uses the
        # names above, but config authors should not need to know every internal
        # error string.
        "quota",
        "capacity",
        "transient",
    }
)
DEFAULT_STOP_ON = frozenset(
    {
        "invalid_request",
        "context_window_exceeded",
        "request_too_large",
        "credential_reauth_needed",
        "pre_request_callback_error",
        "cancelled",
        # Config aliases retained for readability.
        "validation",
        "permanent",
        "pre_request_callback",
    }
)
HARD_STOP_ON = frozenset(
    {
        "invalid_request",
        "context_window_exceeded",
        "request_too_large",
        "credential_reauth_needed",
        "pre_request_callback_error",
        "cancelled",
        "configuration_error",
    }
)


class RoutingConfigError(ValueError):
    """Routing configuration is malformed; never silently degraded around."""


@dataclass(frozen=True)
class RouteTarget:
    """One concrete provider/model execution target in a fallback chain."""

    provider: str
    model: str
    name: str = ""
    protocol: str | None = None
    # D13: transport profile addressed as provider:profile/model. Identity
    # stays provider-level; the profile only steers transport.
    profile: str | None = None
    execution: ExecutionMode = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider or not self.model:
            raise RoutingConfigError("route targets require provider and model")
        if self.execution not in {"auto", "custom", "native", "litellm_fallback"}:
            raise RoutingConfigError(f"unsupported execution mode: {self.execution}")
        if not self.name:
            object.__setattr__(
                self,
                "name",
                f"{self.provider}:{self.profile}/{self.model}" if self.profile else f"{self.provider}/{self.model}",
            )

    @property
    def prefixed_model(self) -> str:
        """Return `provider/model` without double-prefixing an already-prefixed model."""

        return self.model if self.model.startswith(f"{self.provider}/") else f"{self.provider}/{self.model}"

    def identity_key(self) -> tuple[str, str, str]:
        """Profile-aware comparison identity (provider, profile, model)."""

        return (self.provider.lower(), (self.profile or "").lower(), self.model.lower())


@dataclass(frozen=True)
class FallbackGroup:
    """Deterministic ordered chain of route targets."""

    name: str
    targets: tuple[RouteTarget, ...]
    failover_on: frozenset[str] = DEFAULT_FAILOVER_ON
    stop_on: frozenset[str] = DEFAULT_STOP_ON
    streaming_policy: StreamingFallbackPolicy = "pre_output_only"
    max_targets: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise RoutingConfigError("fallback group name is required")
        if not self.targets:
            raise RoutingConfigError("fallback groups require at least one target")
        if self.max_targets is not None and self.max_targets <= 0:
            raise RoutingConfigError("max_targets must be positive")
        if self.max_targets is not None and len(self.targets) > self.max_targets:
            raise RoutingConfigError("fallback group target count exceeds max_targets")

    def effective_targets(self) -> tuple[RouteTarget, ...]:
        """Runtime attempt chain: declaration order, capped by max_targets."""

        if self.max_targets is not None:
            return self.targets[: self.max_targets]
        return self.targets


@dataclass(frozen=True)
class RoutingConfig:
    """Routing configuration loaded from env or tests."""

    fallback_groups: dict[str, FallbackGroup] = field(default_factory=dict)
    model_routes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingDecision:
    """Resolved routing plan for a requested model."""

    requested_model: str
    targets: tuple[RouteTarget, ...]
    group_name: str | None = None
    group: FallbackGroup | None = None
    reason: str = "direct"
