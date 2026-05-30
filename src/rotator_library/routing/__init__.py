# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Routing and fallback group primitives."""

from .config import RoutingConfigError, load_routing_config_from_env, parse_route_target
from .policy import FallbackPolicy
from .resolver import FallbackResolver
from .types import FallbackGroup, RouteTarget, RoutingConfig, RoutingDecision, TargetGroup, TargetSelector

__all__ = [
    "FallbackGroup",
    "FallbackPolicy",
    "FallbackResolver",
    "RouteTarget",
    "RoutingConfig",
    "RoutingConfigError",
    "RoutingDecision",
    "TargetGroup",
    "TargetSelector",
    "load_routing_config_from_env",
    "parse_route_target",
]
