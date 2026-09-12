"""Reference demo hooks for the G2 hookable pipeline (demo-scoped).

Nothing in this package is consumed by the production engine. The modules
prove expressiveness of the hook stages; see each module's docstring for
scope and danger notes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .proxy_tools import (
        ToolCallInterceptorHook,
        ToolInjectorHook,
        ToolStripperHook,
        proxy_tools_registry_snapshot,
        reset_proxy_tools_registry,
    )

__lazy = {
    "ToolStripperHook": ("proxy_tools", "ToolStripperHook"),
    "ToolInjectorHook": ("proxy_tools", "ToolInjectorHook"),
    "ToolCallInterceptorHook": ("proxy_tools", "ToolCallInterceptorHook"),
    "proxy_tools_registry_snapshot": ("proxy_tools", "proxy_tools_registry_snapshot"),
    "reset_proxy_tools_registry": ("proxy_tools", "reset_proxy_tools_registry"),
}

__all__ = list(__lazy)


def __getattr__(name: str):
    try:
        module_name, attr = __lazy[name]
    except KeyError:
        raise AttributeError(name) from None
    import importlib

    module = importlib.import_module(f".{module_name}", __package__)
    return getattr(module, attr)
