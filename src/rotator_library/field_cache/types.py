# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Data types for provider field-cache rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

FieldCacheSource = Literal[
    "request",
    "response",
    "stream_event",
    "unified_request",
    "unified_response",
    "unified_stream_event",
]
FieldCacheTarget = Literal["request", "unified_request", "metadata"]
FieldCacheMode = Literal["last", "all", "last_user_turn", "last_assistant_turn", "per_tool_call"]
FieldCacheScope = Literal["provider", "model", "credential", "session", "conversation", "classifier"]

DEFAULT_SCOPE: tuple[FieldCacheScope, ...] = ("provider", "model", "classifier", "session")
_VALID_SOURCES = {"request", "response", "stream_event", "unified_request", "unified_response", "unified_stream_event"}
_VALID_TARGETS = {"request", "unified_request", "metadata"}
_VALID_SCOPES = {"provider", "model", "credential", "session", "conversation", "classifier"}


@dataclass(frozen=True)
class FieldCacheInjection:
    """Where and how a cached value should be injected into a later payload."""

    target: FieldCacheTarget
    path: str
    when_missing_only: bool = False
    insert: bool = False
    as_list: bool = False


@dataclass(frozen=True)
class FieldCacheRule:
    """Declarative rule for extracting and re-injecting provider state.

    Rules are protocol/provider extensions, not session-affinity logic. Session
    tracking decides continuity; field-cache rules preserve protocol state such
    as reasoning content, thought signatures, prompt cache keys, and response IDs.
    """

    name: str
    source: FieldCacheSource
    path: str
    mode: FieldCacheMode = "last"
    scope: tuple[FieldCacheScope, ...] = DEFAULT_SCOPE
    inject: Optional[FieldCacheInjection] = None
    enabled: bool = True
    ttl_seconds: Optional[int] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    allow_missing_session: bool = False

    def __post_init__(self) -> None:
        if not self.name or any(char in self.name for char in "/\\:"):
            raise ValueError("FieldCacheRule.name must be non-empty and filesystem-safe")
        if self.mode not in {"last", "all", "last_user_turn", "last_assistant_turn", "per_tool_call"}:
            raise ValueError(f"Unsupported field-cache mode: {self.mode}")
        if self.source not in _VALID_SOURCES:
            raise ValueError(f"Unsupported field-cache source: {self.source}")
        if self.inject and self.inject.target not in _VALID_TARGETS:
            raise ValueError(f"Unsupported field-cache injection target: {self.inject.target}")
        if not self.scope:
            raise ValueError("FieldCacheRule.scope must contain at least one dimension")
        invalid_scopes = [scope for scope in self.scope if scope not in _VALID_SCOPES]
        if invalid_scopes:
            raise ValueError(f"Unsupported field-cache scope: {invalid_scopes[0]}")


@dataclass(frozen=True)
class FieldCacheContext:
    """Scope values used to isolate cached provider fields."""

    provider: Optional[str] = None
    model: Optional[str] = None
    credential_id: Optional[str] = None
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    classifier: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def value_for_scope(self, scope: FieldCacheScope) -> Optional[str]:
        if scope == "provider":
            return self.provider
        if scope == "model":
            return self.model
        if scope == "credential":
            return self.credential_id
        if scope == "session":
            return self.session_id
        if scope == "conversation":
            return self.conversation_id
        if scope == "classifier":
            return self.classifier
        raise ValueError(f"Unsupported field-cache scope: {scope}")
