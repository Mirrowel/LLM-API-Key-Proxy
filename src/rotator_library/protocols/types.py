# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Protocol-neutral data structures used by native protocol adapters.

These types intentionally model common LLM API concepts without pretending the
set is complete. Providers and future protocol implementations can preserve
non-standard fields in ``extra`` or ``raw`` instead of dropping them. That
preservation is important for transform-pass logging, field-cache rules, and
provider-specific overrides added in later experimental phases.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, ClassVar, Iterable, Mapping, Optional


JsonObject = dict[str, Any]


def serialize_value(value: Any) -> Any:
    """Return a JSON-friendly copy of a protocol value.

    The transaction logging phase needs reliable snapshots after every protocol
    and adapter pass. This helper keeps that concern centralized and avoids
    mutating live request/response objects while preparing logs or fixtures.
    """

    if isinstance(value, ProtocolSerializable):
        return value.to_dict()
    if is_dataclass(value):
        return serialize_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [serialize_value(v) for v in value]
    return deepcopy(value)


def copy_mapping(value: Optional[Mapping[str, Any]]) -> JsonObject:
    """Return a deep-copied dict for extension fields."""

    return serialize_value(dict(value or {}))


class ProtocolSerializable:
    """Mixin for dataclasses that need stable dict serialization.

    Concrete protocol types define ``_fields`` so ``to_dict`` stays explicit and
    future additions do not accidentally disappear from transform logs.
    """

    _fields: ClassVar[tuple[str, ...]] = ()

    def to_dict(self) -> JsonObject:
        return {field_name: serialize_value(getattr(self, field_name)) for field_name in self._fields}


@dataclass
class CostDetails(ProtocolSerializable):
    """Normalized cost metadata from provider-reported or estimated sources."""

    provider_reported_cost: Optional[float] = None
    estimated_cost: Optional[float] = None
    currency: str = "USD"
    source: Optional[str] = None
    metadata: JsonObject = field(default_factory=dict)

    _fields: ClassVar[tuple[str, ...]] = (
        "provider_reported_cost",
        "estimated_cost",
        "currency",
        "source",
        "metadata",
    )


@dataclass
class Usage(ProtocolSerializable):
    """Protocol-neutral token and cost usage values.

    Existing usage tracking has provider-specific extraction paths. Native
    protocols use this shape first, then later phases can normalize it into the
    current usage manager without replacing that engine.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost: Optional[CostDetails] = None
    raw: Any = None
    extra: JsonObject = field(default_factory=dict)

    _fields: ClassVar[tuple[str, ...]] = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "cost",
        "raw",
        "extra",
    )

    def __post_init__(self) -> None:
        if self.total_tokens <= 0:
            self.total_tokens = self.input_tokens + self.output_tokens + self.reasoning_tokens


@dataclass
class ReasoningBlock(ProtocolSerializable):
    """Reasoning/thinking content and signatures preserved across protocols."""

    type: str = "reasoning"
    text: Optional[str] = None
    signature: Optional[str] = None
    redacted: bool = False
    extra: JsonObject = field(default_factory=dict)

    _fields: ClassVar[tuple[str, ...]] = ("type", "text", "signature", "redacted", "extra")


@dataclass
class ToolCall(ProtocolSerializable):
    """Protocol-neutral tool/function call emitted by an assistant."""

    id: Optional[str] = None
    name: Optional[str] = None
    arguments: Any = None
    type: str = "function"
    index: Optional[int] = None
    extra: JsonObject = field(default_factory=dict)

    _fields: ClassVar[tuple[str, ...]] = ("id", "name", "arguments", "type", "index", "extra")


@dataclass
class ToolResult(ProtocolSerializable):
    """Protocol-neutral result associated with a prior tool call."""

    tool_call_id: Optional[str] = None
    content: Any = None
    is_error: Optional[bool] = None
    extra: JsonObject = field(default_factory=dict)

    _fields: ClassVar[tuple[str, ...]] = ("tool_call_id", "content", "is_error", "extra")


@dataclass
class ToolDefinition(ProtocolSerializable):
    """Protocol-neutral tool schema exposed to a model."""

    name: str = ""
    description: Optional[str] = None
    input_schema: JsonObject = field(default_factory=dict)
    type: str = "function"
    extra: JsonObject = field(default_factory=dict)

    _fields: ClassVar[tuple[str, ...]] = ("name", "description", "input_schema", "type", "extra")


@dataclass
class ContentBlock(ProtocolSerializable):
    """A single message content block.

    ``type`` follows the source protocol when practical. The dedicated fields
    cover common text, image, document, tool, and reasoning cases, while ``extra``
    keeps provider-specific payloads for later field-cache extraction.
    """

    type: str = "text"
    text: Optional[str] = None
    source: Any = None
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    reasoning: Optional[ReasoningBlock] = None
    raw: Any = None
    extra: JsonObject = field(default_factory=dict)

    _fields: ClassVar[tuple[str, ...]] = (
        "type",
        "text",
        "source",
        "tool_call",
        "tool_result",
        "reasoning",
        "raw",
        "extra",
    )


@dataclass
class UnifiedMessage(ProtocolSerializable):
    """A protocol-neutral chat/message turn."""

    role: str
    content: list[ContentBlock] = field(default_factory=list)
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: list[ReasoningBlock] = field(default_factory=list)
    raw: Any = None
    extra: JsonObject = field(default_factory=dict)

    _fields: ClassVar[tuple[str, ...]] = (
        "role",
        "content",
        "name",
        "tool_call_id",
        "tool_calls",
        "reasoning",
        "raw",
        "extra",
    )


@dataclass
class UnifiedRequest(ProtocolSerializable):
    """A request after parsing from a client or provider protocol."""

    model: str = ""
    messages: list[UnifiedMessage] = field(default_factory=list)
    system: list[ContentBlock] = field(default_factory=list)
    tools: list[ToolDefinition] = field(default_factory=list)
    stream: bool = False
    generation_params: JsonObject = field(default_factory=dict)
    response_format: Any = None
    previous_response_id: Optional[str] = None
    metadata: JsonObject = field(default_factory=dict)
    raw: Any = None
    extra: JsonObject = field(default_factory=dict)

    _fields: ClassVar[tuple[str, ...]] = (
        "model",
        "messages",
        "system",
        "tools",
        "stream",
        "generation_params",
        "response_format",
        "previous_response_id",
        "metadata",
        "raw",
        "extra",
    )


@dataclass
class UnifiedResponse(ProtocolSerializable):
    """A complete provider/client response in protocol-neutral form."""

    id: Optional[str] = None
    model: Optional[str] = None
    messages: list[UnifiedMessage] = field(default_factory=list)
    output: list[Any] = field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: Optional[Usage] = None
    metadata: JsonObject = field(default_factory=dict)
    raw: Any = None
    extra: JsonObject = field(default_factory=dict)

    _fields: ClassVar[tuple[str, ...]] = (
        "id",
        "model",
        "messages",
        "output",
        "stop_reason",
        "usage",
        "metadata",
        "raw",
        "extra",
    )


@dataclass
class UnifiedStreamEvent(ProtocolSerializable):
    """A single protocol-neutral stream event.

    Future SSE and WebSocket transports should consume this type instead of raw
    provider chunks so transport code can stay independent from protocol parsing.
    """

    type: str
    delta: Optional[UnifiedMessage] = None
    message: Optional[UnifiedMessage] = None
    tool_call: Optional[ToolCall] = None
    usage: Optional[Usage] = None
    error: Any = None
    raw: Any = None
    extra: JsonObject = field(default_factory=dict)

    _fields: ClassVar[tuple[str, ...]] = (
        "type",
        "delta",
        "message",
        "tool_call",
        "usage",
        "error",
        "raw",
        "extra",
    )


@dataclass
class ProtocolContext(ProtocolSerializable):
    """Execution context passed through protocol methods.

    Only a small subset is needed in Phase 1, but the fields anticipate later
    provider overrides, transaction tracing, field-cache scoping, and transport
    selection without forcing those systems to exist yet.
    """

    provider: Optional[str] = None
    model: Optional[str] = None
    source_protocol: Optional[str] = None
    target_protocol: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    credential_stable_id: Optional[str] = None
    transport: str = "http"
    provider_options: JsonObject = field(default_factory=dict)
    metadata: JsonObject = field(default_factory=dict)

    _fields: ClassVar[tuple[str, ...]] = (
        "provider",
        "model",
        "source_protocol",
        "target_protocol",
        "request_id",
        "session_id",
        "credential_stable_id",
        "transport",
        "provider_options",
        "metadata",
    )


class ProtocolError(ValueError):
    """Error raised by protocol parsing/building passes."""

    def __init__(
        self,
        message: str,
        *,
        protocol: str,
        pass_name: str,
        payload: Any = None,
    ):
        self.protocol = protocol
        self.pass_name = pass_name
        self.payload_preview = _payload_preview(payload)
        details = f"{protocol}.{pass_name}: {message}"
        if self.payload_preview is not None:
            details = f"{details} | payload={self.payload_preview}"
        super().__init__(details)


def _payload_preview(payload: Any, limit: int = 500) -> Optional[str]:
    if payload is None:
        return None
    text = repr(serialize_value(payload))
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def text_blocks(text: Optional[str]) -> list[ContentBlock]:
    """Return a single text block list for simple string content."""

    if text is None:
        return []
    return [ContentBlock(type="text", text=str(text))]


def first_text(blocks: Iterable[ContentBlock]) -> Optional[str]:
    """Return concatenated text from content blocks, or ``None`` if absent."""

    parts = [block.text for block in blocks if block.type == "text" and block.text]
    return "".join(parts) if parts else None
