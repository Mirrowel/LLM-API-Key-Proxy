# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Resolve requested models to direct targets or fallback groups."""

from __future__ import annotations

from .config import RoutingConfigError, parse_route_target
from .types import RoutingConfig, RoutingDecision


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
            return RoutingDecision(requested_model=requested_model, group_name=group.name, group=group, targets=group.targets, reason="model_route_group")
        if route:
            return RoutingDecision(requested_model=requested_model, targets=(parse_route_target(route),), reason="model_route_target")
        if "/" in requested_model:
            return RoutingDecision(requested_model=requested_model, targets=(parse_route_target(requested_model),), reason="direct_provider_model")
        raise RoutingConfigError(f"model {requested_model!r} is not provider-prefixed and has no route")
