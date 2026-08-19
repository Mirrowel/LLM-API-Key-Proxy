# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Auto-discovery registry for payload adapters."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Type

from .base import PayloadAdapter

lib_logger = logging.getLogger("rotator_library")

ADAPTER_PLUGINS: dict[str, Type[PayloadAdapter]] = {}
ADAPTER_ALIASES: dict[str, str] = {}
_ADAPTER_INSTANCES: dict[str, PayloadAdapter] = {}

_INFRASTRUCTURE_MODULES = {"base", "registry"}


def register_adapter(adapter_class: Type[PayloadAdapter], *, replace: bool = False) -> Type[PayloadAdapter]:
    """Register an adapter class and its aliases with collision checks."""

    if not inspect.isclass(adapter_class) or not issubclass(adapter_class, PayloadAdapter):
        raise TypeError("adapter_class must inherit PayloadAdapter")
    if adapter_class is PayloadAdapter:
        raise TypeError("cannot register PayloadAdapter itself")
    name = adapter_class.name
    if not name:
        raise ValueError(f"Adapter {adapter_class.__name__} must define a name")
    alias_owner = ADAPTER_ALIASES.get(name)
    if alias_owner and alias_owner != name and not replace:
        raise ValueError(f"Adapter name conflicts with registered alias: {name}")
    existing = ADAPTER_PLUGINS.get(name)
    if existing and existing is not adapter_class and not replace:
        raise ValueError(f"Adapter name already registered: {name}")
    if replace and existing and existing is not adapter_class:
        for alias, owner in list(ADAPTER_ALIASES.items()):
            if owner == name:
                ADAPTER_ALIASES.pop(alias, None)
    ADAPTER_PLUGINS[name] = adapter_class
    _ADAPTER_INSTANCES.pop(name, None)
    for alias in adapter_class.aliases:
        existing_name = ADAPTER_ALIASES.get(alias)
        if existing_name and existing_name != name and not replace:
            raise ValueError(f"Adapter alias already registered: {alias}")
        if alias in ADAPTER_PLUGINS and alias != name and not replace:
            raise ValueError(f"Adapter alias conflicts with registered name: {alias}")
        ADAPTER_ALIASES[alias] = name
    lib_logger.debug("Registered adapter: %s", name)
    return adapter_class


def resolve_adapter_name(name: str) -> str:
    if name in ADAPTER_PLUGINS:
        return name
    if name in ADAPTER_ALIASES:
        return ADAPTER_ALIASES[name]
    raise KeyError(f"Unknown adapter: {name}")


def get_adapter_class(name: str) -> Type[PayloadAdapter]:
    return ADAPTER_PLUGINS[resolve_adapter_name(name)]


def get_adapter(name: str) -> PayloadAdapter:
    canonical = resolve_adapter_name(name)
    if canonical not in _ADAPTER_INSTANCES:
        _ADAPTER_INSTANCES[canonical] = ADAPTER_PLUGINS[canonical]()
    return _ADAPTER_INSTANCES[canonical]


def list_adapters() -> list[str]:
    return sorted(ADAPTER_PLUGINS)


def _register_adapters() -> None:
    package = importlib.import_module(__package__ or "rotator_library.adapters")
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        if module_name.startswith("_") or module_name in _INFRASTRUCTURE_MODULES:
            continue
        module = importlib.import_module(f"{package.__name__}.{module_name}")
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if (
                inspect.isclass(attribute)
                and issubclass(attribute, PayloadAdapter)
                and attribute is not PayloadAdapter
                and attribute.__module__ == module.__name__
            ):
                register_adapter(attribute)


_register_adapters()
