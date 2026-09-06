# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Context objects for native provider execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

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
    operation: str = "chat"
    input_protocol_name: Optional[str] = None
    client_protocol_name: Optional[str] = None
    # Pristine same-protocol client wire payload (D4 raw fast path). When the
    # client protocol equals the provider protocol and no semantic edits are
    # required, this payload is the transport basis instead of a rebuild.
    raw_client_request: Optional[Dict[str, Any]] = None
    # Traceable overlay record for the chosen transport basis (W3; the
    # transaction-log reconstruction consumes this).
    request_transport_overlays: Optional[List[Dict[str, Any]]] = None
    # Authoritative stream usage assembled by the native executor from events
    # AND raw provider chunks (including provider-reported cost frames). The
    # operational stream layer adopts this record on completion.
    stream_usage_record: Any = None
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
    request_preparer: Optional[Callable[..., dict[str, Any]]] = None
    request_validator: Optional[Callable[..., Any]] = None

    def protocol_context(
        self,
        *,
        source_protocol: Optional[str] = None,
        target_protocol: Optional[str] = None,
        source_provider: Optional[str] = None,
        target_provider: Optional[str] = None,
        provider_state_compatible: bool = False,
    ) -> ProtocolContext:
        """Build a protocol context for parse/build/format passes."""

        input_protocol = self.input_protocol_name or self.protocol_name
        client_protocol = self.client_protocol_name or input_protocol
        return ProtocolContext(
            provider=self.provider,
            model=self.model,
            source_protocol=source_protocol or input_protocol,
            target_protocol=target_protocol or self.protocol_name,
            input_protocol=input_protocol,
            provider_protocol=self.protocol_name,
            client_protocol=client_protocol,
            source_provider=source_provider,
            target_provider=target_provider,
            provider_state_compatible=provider_state_compatible,
            session_id=self.session_id,
            credential_stable_id=self.credential_id,
            transport=self.transport,
            provider_options={"operation": self.operation},
            metadata={"operation": self.operation, **dict(self.metadata)},
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
            metadata={"operation": self.operation, **dict(self.metadata)},
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
            metadata={"operation": self.operation, **dict(self.metadata)},
        )
