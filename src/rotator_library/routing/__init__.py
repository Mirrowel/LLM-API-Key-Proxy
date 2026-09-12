# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Routing and fallback group primitives."""

from .config import RoutingConfigError, load_routing_config_from_env, parse_route_target
from .attempts import clone_context_for_target
from .policy import FallbackPolicy
from .profiles import (
    DEFAULT_PROTOCOL_PRIORITY,
    ModelReference,
    ModelReferenceError,
    parse_model_reference,
    resolve_profile,
    valid_profile_name,
)
from .resolver import FallbackResolver
from .types import FallbackGroup, RouteTarget, RoutingConfig, RoutingDecision

__all__ = [
    "DEFAULT_PROTOCOL_PRIORITY",
    "FallbackGroup",
    "FallbackPolicy",
    "FallbackResolver",
    "RouteTarget",
    "RoutingConfig",
    "RoutingConfigError",
    "RoutingDecision",
    "clone_context_for_target",
    "load_routing_config_from_env",
    "parse_route_target",
]
