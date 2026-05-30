# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Typed route targets and fallback groups."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ExecutionMode = Literal["auto", "native", "custom", "litellm_fallback"]
StreamingFallbackPolicy = Literal["pre_output_only", "never"]

DEFAULT_FAILOVER_ON = frozenset({"rate_limit", "quota", "capacity", "server_error", "api_connection", "transient", "unsupported_operation"})
DEFAULT_STOP_ON = frozenset({"auth", "authentication", "validation", "permanent", "pre_request_callback", "cancelled"})


@dataclass(frozen=True)
class RouteTarget:
    """One concrete provider/model execution target in a fallback chain."""

    provider: str
    model: str
    name: str = ""
    protocol: str | None = None
    execution: ExecutionMode = "auto"
    priority: int | None = None
    weight: float | None = None
    conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider or not self.model:
            raise ValueError("route targets require provider and model")
        if self.execution not in {"auto", "native", "custom", "litellm_fallback"}:
            raise ValueError(f"unsupported execution mode: {self.execution}")
        if not self.name:
            object.__setattr__(self, "name", f"{self.provider}/{self.model}")

    @property
    def prefixed_model(self) -> str:
        """Return `provider/model` without double-prefixing an already-prefixed model."""

        return self.model if self.model.startswith(f"{self.provider}/") else f"{self.provider}/{self.model}"


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
            raise ValueError("fallback group name is required")
        if not self.targets:
            raise ValueError("fallback groups require at least one target")
        if self.max_targets is not None and self.max_targets <= 0:
            raise ValueError("max_targets must be positive")
        if self.max_targets is not None and len(self.targets) > self.max_targets:
            raise ValueError("fallback group target count exceeds max_targets")


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
    selected_target_index: int = 0
    reason: str = "direct"


@dataclass(frozen=True)
class RouteAttemptResult:
    """Result summary for one attempted route target."""

    target: RouteTarget
    success: bool
    error_type: str | None = None
    emitted_output: bool = False
    usage: dict[str, Any] = field(default_factory=dict)


class TargetSelector(Protocol):
    """Future target-group selector seam; Phase 6 keeps ordered fallback only."""

    def select(self, targets: tuple[RouteTarget, ...]) -> RouteTarget:
        """Return one target from a richer target group."""


@dataclass(frozen=True)
class TargetGroup:
    """Future richer target group; not used for Phase 6 ordered fallback."""

    name: str
    targets: tuple[RouteTarget, ...]
    selector: str = "ordered"
