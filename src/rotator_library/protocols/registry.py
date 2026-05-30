# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Auto-discovery registry for native protocol adapters."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Type

from .base import ProtocolAdapter

lib_logger = logging.getLogger("rotator_library")

PROTOCOL_PLUGINS: dict[str, Type[ProtocolAdapter]] = {}
PROTOCOL_ALIASES: dict[str, str] = {}
_PROTOCOL_INSTANCES: dict[str, ProtocolAdapter] = {}

_INFRASTRUCTURE_MODULES = {"base", "registry", "types"}


def register_protocol(protocol_class: Type[ProtocolAdapter], *, replace: bool = False) -> Type[ProtocolAdapter]:
    """Register a protocol adapter class and its aliases.

    The registry mirrors the provider plugin system while staying stricter about
    duplicate names. This matters for custom protocol modules: accidental alias
    collisions should fail early instead of silently changing conversion logic.
    """

    if not inspect.isclass(protocol_class) or not issubclass(protocol_class, ProtocolAdapter):
        raise TypeError("protocol_class must inherit ProtocolAdapter")
    if protocol_class is ProtocolAdapter:
        raise TypeError("cannot register ProtocolAdapter itself")

    name = protocol_class.name
    if not name:
        raise ValueError(f"Protocol {protocol_class.__name__} must define a name")

    existing = PROTOCOL_PLUGINS.get(name)
    if existing and existing is not protocol_class and not replace:
        raise ValueError(f"Protocol name already registered: {name}")

    PROTOCOL_PLUGINS[name] = protocol_class
    _PROTOCOL_INSTANCES.pop(name, None)

    for alias in protocol_class.aliases:
        existing_name = PROTOCOL_ALIASES.get(alias)
        if existing_name and existing_name != name and not replace:
            raise ValueError(f"Protocol alias already registered: {alias}")
        if alias in PROTOCOL_PLUGINS and alias != name and not replace:
            raise ValueError(f"Protocol alias conflicts with registered name: {alias}")
        PROTOCOL_ALIASES[alias] = name

    lib_logger.debug("Registered protocol: %s", name)
    return protocol_class


def resolve_protocol_name(name: str) -> str:
    """Resolve aliases to canonical protocol names."""

    if name in PROTOCOL_PLUGINS:
        return name
    if name in PROTOCOL_ALIASES:
        return PROTOCOL_ALIASES[name]
    raise KeyError(f"Unknown protocol: {name}")


def get_protocol_class(name: str) -> Type[ProtocolAdapter]:
    """Return a registered protocol adapter class by name or alias."""

    return PROTOCOL_PLUGINS[resolve_protocol_name(name)]


def get_protocol(name: str) -> ProtocolAdapter:
    """Return a shared stateless protocol adapter instance by name or alias."""

    canonical = resolve_protocol_name(name)
    if canonical not in _PROTOCOL_INSTANCES:
        _PROTOCOL_INSTANCES[canonical] = PROTOCOL_PLUGINS[canonical]()
    return _PROTOCOL_INSTANCES[canonical]


def list_protocols() -> list[str]:
    """Return canonical protocol names in deterministic order."""

    return sorted(PROTOCOL_PLUGINS)


def _register_protocols() -> None:
    """Discover protocol modules in this package and register adapter classes.

    Private modules and infrastructure modules are skipped so local experiments
    can live next to production protocols without being imported accidentally.
    """

    package = importlib.import_module(__package__ or "rotator_library.protocols")
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        if module_name.startswith("_") or module_name in _INFRASTRUCTURE_MODULES:
            continue
        module = importlib.import_module(f"{package.__name__}.{module_name}")
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if (
                inspect.isclass(attribute)
                and issubclass(attribute, ProtocolAdapter)
                and attribute is not ProtocolAdapter
                and attribute.__module__ == module.__name__
            ):
                register_protocol(attribute)


_register_protocols()
