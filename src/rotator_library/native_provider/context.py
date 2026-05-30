# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Context objects for native provider execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..adapters import AdapterContext
from ..field_cache import FieldCacheContext, FieldCacheRule
from ..protocols import ProtocolContext


@dataclass
class NativeProviderContext:
    """Metadata needed to execute a provider through native protocol helpers.

    This context intentionally mirrors the trace, adapter, protocol, and
    field-cache contexts so provider-native execution can remain opt-in and
    testable without changing the existing LiteLLM-backed path.
    """

    provider: str
    model: str
    protocol_name: str
    endpoint: str
    headers: dict[str, str] = field(default_factory=dict)
    credential_id: Optional[str] = None
    session_id: Optional[str] = None
    scope_key: Optional[str] = None
    classifier: Optional[str] = None
    transport: str = "http"
    adapter_names: tuple[str, ...] = ()
    adapter_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    field_cache_rules: tuple[FieldCacheRule, ...] = ()
    transaction_logger: Optional[Any] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def protocol_context(self, *, target_protocol: Optional[str] = None) -> ProtocolContext:
        """Build a protocol context for parse/build/format passes."""

        return ProtocolContext(
            provider=self.provider,
            model=self.model,
            source_protocol=self.protocol_name,
            target_protocol=target_protocol or self.protocol_name,
            session_id=self.session_id,
            credential_stable_id=self.credential_id,
            transport=self.transport,
            metadata=dict(self.metadata),
        )

    def adapter_context(self) -> AdapterContext:
        """Build an adapter context for provider payload adapters."""

        return AdapterContext(
            provider=self.provider,
            model=self.model,
            protocol=self.protocol_name,
            credential_id=self.credential_id,
            session_id=self.session_id,
            scope_key=self.scope_key,
            classifier=self.classifier,
            transport=self.transport,
            metadata=dict(self.metadata),
            adapter_config=dict(self.adapter_config),
            transaction_logger=self.transaction_logger,
        )

    def field_cache_context(self) -> FieldCacheContext:
        """Build a field-cache context with provider isolation metadata."""

        return FieldCacheContext(
            provider=self.provider,
            model=self.model,
            credential_id=self.credential_id,
            session_id=self.session_id,
            conversation_id=self.scope_key,
            classifier=self.classifier,
            metadata=dict(self.metadata),
        )
