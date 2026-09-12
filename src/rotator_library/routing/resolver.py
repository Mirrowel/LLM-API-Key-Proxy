# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Resolve requested models to direct targets or fallback groups."""

from __future__ import annotations

from .config import RoutingConfigError, parse_route_target
from .profiles import parse_model_reference
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
            reason = "model_route_group_promoted" if targets != group.effective_targets() else "model_route_group"
            return RoutingDecision(requested_model=requested_model, group_name=group.name, group=group, targets=targets, reason=reason)
        if route:
            return RoutingDecision(requested_model=requested_model, targets=(parse_route_target(route),), reason="model_route_target")
        for group in self.config.fallback_groups.values():
            targets = _promote_requested_target(group, requested_model)
            if targets != group.effective_targets() or any(_same_target(target, requested_model) for target in group.effective_targets()):
                return RoutingDecision(requested_model=requested_model, group_name=group.name, group=group, targets=targets, reason="provider_model_group_promoted")
        if "/" in requested_model:
            return RoutingDecision(requested_model=requested_model, targets=(parse_route_target(requested_model),), reason="direct_provider_model")
        raise RoutingConfigError(f"model {requested_model!r} is not provider-prefixed and has no route")


def _promote_requested_target(group: FallbackGroup, requested_model: str) -> tuple[RouteTarget, ...]:
    """Return capped, deduplicated targets with the requested one first.

    Promotion is profile-aware (D13): an explicit ``provider:profile``
    request matches only that profile variant, and duplicate attempts on
    the same (provider, profile, model) identity are collapsed.
    """

    targets = group.effective_targets()
    matching = [target for target in targets if _same_target(target, requested_model)]
    ordered = targets
    if matching:
        selected = matching[0]
        ordered = (selected, *(target for target in targets if target is not selected))
    deduplicated: list[RouteTarget] = []
    seen: set[tuple[str, str, str]] = set()
    for target in ordered:
        identity = target.identity_key()
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(target)
    return tuple(deduplicated)


def _same_target(target: RouteTarget, requested_model: str) -> bool:
    """Profile-aware match: provider, profile, and model must all agree."""

    try:
        reference = parse_model_reference(requested_model)
    except Exception:
        return False
    return (
        target.provider.lower() == reference.provider.lower()
        and (target.profile or "").lower() == (reference.profile or "").lower()
        and target.model.lower() == reference.model.lower()
    )
