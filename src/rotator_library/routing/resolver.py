# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Resolve requested models to direct targets or fallback groups."""

from __future__ import annotations

from .config import RoutingConfigError, parse_route_target
from .types import FallbackGroup, RouteTarget, RoutingConfig, RoutingDecision


class FallbackResolver:
    """Resolve a model name using deterministic fallback group rules."""

    def __init__(self, config: RoutingConfig) -> None:
        self.config = config

    def resolve(self, requested_model: str) -> RoutingDecision:
        """Return the ordered targets for a requested model."""

        route = self.config.model_routes.get(requested_model.lower())
        if route and route.startswith("group:"):
            group_name = route[len("group:") :]
            group = self.config.fallback_groups.get(group_name)
            if not group:
                raise RoutingConfigError(f"unknown fallback group {group_name}")
            targets = _promote_requested_target(group, requested_model)
            reason = "model_route_group_promoted" if targets != group.targets else "model_route_group"
            return RoutingDecision(requested_model=requested_model, group_name=group.name, group=group, targets=targets, reason=reason)
        if route:
            return RoutingDecision(requested_model=requested_model, targets=(parse_route_target(route),), reason="model_route_target")
        for group in self.config.fallback_groups.values():
            targets = _promote_requested_target(group, requested_model)
            if targets != group.targets or any(_same_target(target, requested_model) for target in group.targets):
                return RoutingDecision(requested_model=requested_model, group_name=group.name, group=group, targets=targets, reason="provider_model_group_promoted")
        if "/" in requested_model:
            return RoutingDecision(requested_model=requested_model, targets=(parse_route_target(requested_model),), reason="direct_provider_model")
        raise RoutingConfigError(f"model {requested_model!r} is not provider-prefixed and has no route")


def _promote_requested_target(group: FallbackGroup, requested_model: str) -> tuple[RouteTarget, ...]:
    """Return group targets with the requested provider/model attempted first."""

    matching = [target for target in group.targets if _same_target(target, requested_model)]
    if not matching:
        return group.targets
    selected = matching[0]
    return (selected, *(target for target in group.targets if target is not selected))


def _same_target(target: RouteTarget, requested_model: str) -> bool:
    return target.prefixed_model.lower() == requested_model.lower()
