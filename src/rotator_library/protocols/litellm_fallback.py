# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Explicit fallback protocol for the existing LiteLLM-shaped path.

This adapter intentionally does very little beyond preserving raw payloads. It
gives later execution and transform-logging code a named protocol path whenever
native adapters do not yet cover a provider or request shape.
"""

from __future__ import annotations

from typing import ClassVar

from .base import ProtocolAdapter


class LiteLLMFallbackProtocol(ProtocolAdapter):
    """Protocol adapter that keeps the current LiteLLM-compatible payload shape.

    Providers should prefer native protocol adapters when available. This class
    remains as a safe compatibility base for unsupported provider shapes and as
    a clear marker in future transaction transform traces.
    """

    name: ClassVar[str] = "litellm_fallback"
    aliases: ClassVar[tuple[str, ...]] = ("litellm", "fallback")
    supported_transports: ClassVar[tuple[str, ...]] = ("http", "sse")
