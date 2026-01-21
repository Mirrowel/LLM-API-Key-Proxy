# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Provider hook dispatcher for usage manager.

Wraps optional provider hooks for request lifecycle events.
"""

import asyncio
from typing import Any, Dict, Optional

from ...core.types import RequestCompleteResult


class HookDispatcher:
    """Dispatch optional provider hooks."""

    def __init__(self, provider_plugins: Optional[Dict[str, Any]] = None):
        self._plugins = provider_plugins or {}
        self._instances: Dict[str, Any] = {}

    def _get_instance(self, provider: str) -> Optional[Any]:
        if provider not in self._instances:
            plugin_class = self._plugins.get(provider)
            if not plugin_class:
                return None
            if isinstance(plugin_class, type):
                self._instances[provider] = plugin_class()
            else:
                self._instances[provider] = plugin_class
        return self._instances[provider]

    async def dispatch_request_complete(
        self,
        provider: str,
        credential: str,
        model: str,
        success: bool,
        response: Optional[Any],
        error: Optional[Any],
    ) -> Optional[RequestCompleteResult]:
        plugin = self._get_instance(provider)
        if not plugin or not hasattr(plugin, "on_request_complete"):
            return None

        result = plugin.on_request_complete(credential, model, success, response, error)
        if asyncio.iscoroutine(result):
            result = await result

        return result
