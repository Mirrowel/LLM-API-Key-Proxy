# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Environment parser for Phase 6 fallback routing configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .types import FallbackGroup, RouteTarget, RoutingConfig


class RoutingConfigError(ValueError):
    """Raised when fallback routing configuration is invalid."""


def parse_route_target(spec: str) -> RouteTarget:
    """Parse `provider/model` with optional `@execution` suffix.

    The suffix is intentionally small because Phase 6 prioritizes ordered
    fallback groups. Rich selector syntax belongs to the config polish phase.
    """

    text = spec.strip()
    if not text:
        raise RoutingConfigError("route target cannot be empty")
    target_text, _, execution = text.partition("@")
    if "/" not in target_text:
        raise RoutingConfigError(f"route target requires provider/model: {spec}")
    provider, model = target_text.split("/", 1)
    if not provider or not model:
        raise RoutingConfigError(f"route target requires provider/model: {spec}")
    return RouteTarget(provider=provider.strip(), model=model.strip(), execution=(execution.strip() or "auto"))


def load_routing_config_from_env(env: Mapping[str, str] | None = None) -> RoutingConfig:
    """Load fallback groups and model-route aliases from environment variables."""

    source = env if env is not None else os.environ
    group_names = _csv(source.get("FALLBACK_GROUPS", ""))
    if len(group_names) != len(set(group_names)):
        raise RoutingConfigError("fallback group names must be unique")
    groups: dict[str, FallbackGroup] = {}
    for name in group_names:
        key = f"FALLBACK_GROUP_{_env_key(name)}"
        target_specs = _csv(source.get(key, ""))
        if not target_specs:
            raise RoutingConfigError(f"fallback group {name} has no targets")
        groups[name] = FallbackGroup(name=name, targets=tuple(parse_route_target(spec) for spec in target_specs))

    model_routes: dict[str, str] = {}
    for key, value in source.items():
        if not key.startswith("MODEL_ROUTE_"):
            continue
        model_alias = key[len("MODEL_ROUTE_") :].lower()
        route = value.strip()
        if route.startswith("group:") and route[len("group:") :] not in groups:
            raise RoutingConfigError(f"model route {key} references unknown fallback group {route}")
        model_routes[model_alias] = route
    return RoutingConfig(fallback_groups=groups, model_routes=model_routes)


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _env_key(value: str) -> str:
    return value.upper().replace("-", "_")
