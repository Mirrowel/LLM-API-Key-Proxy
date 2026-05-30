# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Native protocol adapters for provider-independent request handling.

Importing this package auto-discovers built-in protocol adapters, similar to the
provider plugin system. Runtime execution is not changed by Phase 1; the package
only exposes reusable protocol primitives for later phases.
"""

from .base import ProtocolAdapter
from .registry import (
    PROTOCOL_ALIASES,
    PROTOCOL_PLUGINS,
    get_protocol,
    get_protocol_class,
    list_protocols,
    register_protocol,
    resolve_protocol_name,
)
from .types import (
    ContentBlock,
    CostDetails,
    ProtocolContext,
    ProtocolError,
    ReasoningBlock,
    ToolCall,
    ToolDefinition,
    ToolResult,
    UnifiedMessage,
    UnifiedRequest,
    UnifiedResponse,
    UnifiedStreamEvent,
    Usage,
    first_text,
    serialize_value,
    text_blocks,
)

__all__ = [
    "PROTOCOL_ALIASES",
    "PROTOCOL_PLUGINS",
    "ContentBlock",
    "CostDetails",
    "ProtocolAdapter",
    "ProtocolContext",
    "ProtocolError",
    "ReasoningBlock",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "UnifiedMessage",
    "UnifiedRequest",
    "UnifiedResponse",
    "UnifiedStreamEvent",
    "Usage",
    "first_text",
    "get_protocol",
    "get_protocol_class",
    "list_protocols",
    "register_protocol",
    "resolve_protocol_name",
    "serialize_value",
    "text_blocks",
]
