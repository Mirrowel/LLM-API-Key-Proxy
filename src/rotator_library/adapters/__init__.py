# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Composable payload adapters used between protocols and providers."""

from .base import AdapterContext, PayloadAdapter, run_adapter_chain
from .registry import (
    ADAPTER_ALIASES,
    ADAPTER_PLUGINS,
    get_adapter,
    get_adapter_class,
    list_adapters,
    register_adapter,
    resolve_adapter_name,
)

__all__ = [
    "ADAPTER_ALIASES",
    "ADAPTER_PLUGINS",
    "AdapterContext",
    "PayloadAdapter",
    "get_adapter",
    "get_adapter_class",
    "list_adapters",
    "register_adapter",
    "resolve_adapter_name",
    "run_adapter_chain",
]
