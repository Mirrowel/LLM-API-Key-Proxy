# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Environment parser for Phase 6 fallback routing configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .types import DEFAULT_FAILOVER_ON, DEFAULT_STOP_ON, FallbackGroup, RouteTarget, RoutingConfig


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


def load_routing_config_from_env(env: Mapping[str, str] | None = None, config: object | None = None) -> RoutingConfig:
    """Load fallback groups and model-route aliases from JSON then environment.

    Environment variables intentionally remain the final override layer so the
    existing `.env` deployment model keeps working exactly as before. The
    optional JSON object is a convenience for structured routing plans.
    """

    source = env if env is not None else os.environ
    if config is None:
        from ..config.experimental import load_experimental_config

        config = load_experimental_config(env=source)
    groups: dict[str, FallbackGroup] = _groups_from_json_config(config)

    group_names = _csv(source.get("FALLBACK_GROUPS", ""))
    if len(group_names) != len(set(group_names)):
        raise RoutingConfigError("fallback group names must be unique")
    for name in group_names:
        key = f"FALLBACK_GROUP_{_env_key(name)}"
        target_specs = _csv(source.get(key, ""))
        if not target_specs:
            raise RoutingConfigError(f"fallback group {name} has no targets")
        groups[name] = FallbackGroup(name=name, targets=tuple(parse_route_target(spec) for spec in target_specs))

    model_routes: dict[str, str] = _model_routes_from_json_config(config)
    for key, value in source.items():
        if not key.startswith("MODEL_ROUTE_"):
            continue
        model_alias = key[len("MODEL_ROUTE_") :].lower()
        route = value.strip()
        model_routes[model_alias] = route
    _validate_model_routes(model_routes, groups)
    return RoutingConfig(fallback_groups=groups, model_routes=model_routes)


def _groups_from_json_config(config: object) -> dict[str, FallbackGroup]:
    routing = getattr(config, "routing", {})
    if not isinstance(routing, Mapping):
        return {}
    raw_groups = routing.get("fallback_groups", {})
    if not isinstance(raw_groups, Mapping):
        raise RoutingConfigError("routing.fallback_groups must be an object")
    groups: dict[str, FallbackGroup] = {}
    for name, raw_group in raw_groups.items():
        if not isinstance(raw_group, Mapping):
            raise RoutingConfigError(f"fallback group {name} must be an object")
        raw_targets = raw_group.get("targets", [])
        if not isinstance(raw_targets, list) or not raw_targets:
            raise RoutingConfigError(f"fallback group {name} has no targets")
        groups[str(name)] = FallbackGroup(
            name=str(name),
            targets=tuple(parse_route_target(str(spec)) for spec in raw_targets),
            failover_on=_string_set(raw_group.get("failover_on"), DEFAULT_FAILOVER_ON),
            stop_on=_string_set(raw_group.get("stop_on"), DEFAULT_STOP_ON),
            max_targets=int(raw_group["max_targets"]) if raw_group.get("max_targets") is not None else None,
            metadata=dict(raw_group.get("metadata", {})) if isinstance(raw_group.get("metadata", {}), Mapping) else {},
        )
    return groups


def _model_routes_from_json_config(config: object) -> dict[str, str]:
    routing = getattr(config, "routing", {})
    if not isinstance(routing, Mapping):
        return {}
    raw_routes = routing.get("model_routes", {})
    if not isinstance(raw_routes, Mapping):
        raise RoutingConfigError("routing.model_routes must be an object")
    return {str(alias).lower(): str(route).strip() for alias, route in raw_routes.items()}


def _validate_model_routes(model_routes: Mapping[str, str], groups: Mapping[str, FallbackGroup]) -> None:
    for key, route in model_routes.items():
        if route.startswith("group:") and route[len("group:") :] not in groups:
            raise RoutingConfigError(f"model route {key} references unknown fallback group {route}")


def _string_set(value: object, default: frozenset[str]) -> frozenset[str]:
    if value is None:
        return default
    if isinstance(value, str):
        return frozenset(_csv(value))
    if isinstance(value, list):
        return frozenset(str(item) for item in value)
    raise RoutingConfigError("routing policy lists must be strings or arrays")


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _env_key(value: str) -> str:
    return value.upper().replace("-", "_")
