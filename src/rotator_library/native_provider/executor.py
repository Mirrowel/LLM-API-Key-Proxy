# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Opt-in executor for provider-native protocol calls."""

from __future__ import annotations

from typing import Any, AsyncGenerator

from ..adapters import get_adapter, run_adapter_chain
from ..field_cache import FieldCacheEngine
from ..protocols import get_protocol
from ..usage.accounting import extract_usage_record
from .context import NativeProviderContext
from .http import NativeHTTPTransport
from .streaming import stream_event_payload


class NativeProviderExecutor:
    """Run one native provider request through protocol/adapter/cache passes.

    This executor is intentionally not wired into the live client path yet. Phase
    5 providers can test native behavior here first, then later phases can route
    declared providers into it without disturbing undeclared providers.
    """

    def __init__(self, *, field_cache_store: Any = None) -> None:
        self.field_cache_store = field_cache_store

    async def execute(self, raw_request: dict[str, Any], context: NativeProviderContext, transport: NativeHTTPTransport) -> dict[str, Any]:
        """Execute a non-streaming native provider request."""

        logger = context.transaction_logger
        protocol = get_protocol(context.protocol_name)
        self._trace(context, "native_protocol_selected", {"protocol": protocol.name}, direction="metadata", stage="protocol")
        try:
            protocol_context = context.protocol_context()
            unified_request = protocol.parse_request(raw_request, protocol_context)
            provider_request = protocol.build_request(unified_request, protocol_context)
            adapters = [get_adapter(name) for name in context.adapter_names]
            provider_request = await run_adapter_chain(adapters, provider_request, context.adapter_context(), stage="request")
            cache_engine = FieldCacheEngine(context.field_cache_rules, store=self.field_cache_store)
            provider_request, _ = await cache_engine.inject(
                "request",
                provider_request,
                context.field_cache_context(),
                transaction_logger=logger,
            )
            self._trace(context, "after_field_cache_injection", provider_request, direction="request", stage="adapter")
            self._trace(context, "native_provider_request", provider_request, direction="request", stage="provider")
            raw_response = await transport.post_json(context.endpoint, headers=context.headers, payload=provider_request)
            self._trace(context, "raw_native_provider_response", raw_response, direction="response", stage="provider")
            unified_response = protocol.parse_response(raw_response, protocol_context)
            provider_response = protocol.format_response(unified_response, protocol_context)
            self._trace(context, "parsed_native_provider_response", provider_response, direction="response", stage="protocol")
            provider_response = await run_adapter_chain(adapters, provider_response, context.adapter_context(), stage="response")
            await cache_engine.extract("response", provider_response, context.field_cache_context(), transaction_logger=logger)
            usage_record = extract_usage_record(
                provider_response,
                provider=context.provider,
                model=context.model,
                source="native_provider_response",
            )
            self._trace(
                context,
                "usage_accounting_summary",
                {"usage": usage_record.to_dict()},
                direction="metadata",
                stage="final",
                snapshot=False,
            )
            self._trace(context, "final_client_response", provider_response, direction="response", stage="final")
            return provider_response
        except Exception as exc:
            if logger:
                logger.log_transform_error(
                    "native_provider_execute",
                    exc,
                    payload=raw_request,
                    stage="provider",
                    protocol=context.protocol_name,
                    metadata={"provider": context.provider, "model": context.model},
                )
            raise

    async def stream(self, raw_request: dict[str, Any], context: NativeProviderContext, transport: NativeHTTPTransport) -> AsyncGenerator[Any, None]:
        """Execute a streaming native provider request and yield client events."""

        logger = context.transaction_logger
        protocol = get_protocol(context.protocol_name)
        self._trace(context, "native_protocol_selected", {"protocol": protocol.name}, direction="metadata", stage="protocol")
        try:
            protocol_context = context.protocol_context()
            request_payload = dict(raw_request)
            request_payload["stream"] = True
            unified_request = protocol.parse_request(request_payload, protocol_context)
            provider_request = protocol.build_request(unified_request, protocol_context)
            adapters = [get_adapter(name) for name in context.adapter_names]
            provider_request = await run_adapter_chain(adapters, provider_request, context.adapter_context(), stage="request")
            cache_engine = FieldCacheEngine(context.field_cache_rules, store=self.field_cache_store)
            provider_request, _ = await cache_engine.inject(
                "request",
                provider_request,
                context.field_cache_context(),
                transaction_logger=logger,
            )
            self._trace(context, "after_field_cache_injection", provider_request, direction="request", stage="adapter")
            self._trace(context, "native_provider_stream_request", provider_request, direction="request", stage="provider")
            async for raw_chunk in transport.stream_json_lines(context.endpoint, headers=context.headers, payload=provider_request):
                self._trace(context, "raw_native_provider_stream_chunk", raw_chunk, direction="stream", stage="provider")
                event = protocol.parse_stream_event(raw_chunk, protocol_context)
                event_payload = stream_event_payload(event)
                self._trace(context, "parsed_native_stream_event", event_payload, direction="stream", stage="protocol")
                await cache_engine.extract("stream_event", event_payload, context.field_cache_context(), transaction_logger=logger)
                self._trace(
                    context,
                    "after_field_cache_stream_extraction",
                    {"source": "stream_event"},
                    direction="stream",
                    stage="adapter",
                    snapshot=False,
                )
                formatted = protocol.format_stream_event(event, protocol_context)
                self._trace(context, "formatted_client_stream_event", formatted, direction="stream", stage="final", snapshot=False)
                yield formatted
        except Exception as exc:
            if logger:
                logger.log_transform_error(
                    "native_provider_stream",
                    exc,
                    payload=raw_request,
                    stage="provider",
                    protocol=context.protocol_name,
                    transport=context.transport,
                    metadata={"provider": context.provider, "model": context.model},
                )
            raise

    @staticmethod
    def _trace(
        context: NativeProviderContext,
        pass_name: str,
        data: Any,
        *,
        direction: str,
        stage: str,
        metadata: dict[str, Any] | None = None,
        snapshot: bool = True,
    ) -> None:
        if not context.transaction_logger:
            return
        context.transaction_logger.log_transform_pass(
            pass_name,
            data,
            direction=direction,
            stage=stage,
            protocol=context.protocol_name,
            credential_id=context.credential_id,
            transport=context.transport,
            metadata={
                "provider": context.provider,
                "model": context.model,
                "session_id": context.session_id,
                "scope_key": context.scope_key,
                "classifier": context.classifier,
                **(metadata or {}),
            },
            snapshot=snapshot,
        )
