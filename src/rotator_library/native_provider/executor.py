# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Opt-in executor for provider-native protocol calls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, AsyncGenerator

from ..adapters import get_adapter, run_adapter_chain
from ..field_cache import FieldCacheEngine, InMemoryFieldCacheStore
from ..field_cache.paths import FieldCachePathError, PathToken, parse_path
from ..protocols import get_protocol, serialize_value
from ..transform_trace import REDACTED
from ..usage.accounting import extract_usage_record
from ..usage.costs import CostCalculator
from .context import NativeProviderContext
from .http import NativeHTTPTransport
from .streaming import stream_event_payload


class NativeProviderExecutor:
    """Run one native provider request through protocol/adapter/cache passes.

    The default field-cache store is process-local per executor. That preserves
    provider protocol state across native requests without adding a database;
    production callers can still inject a persistent store when needed.
    """

    def __init__(self, *, field_cache_store: Any = None) -> None:
        self.field_cache_store = field_cache_store or InMemoryFieldCacheStore()

    async def execute(self, raw_request: dict[str, Any], context: NativeProviderContext, transport: NativeHTTPTransport) -> dict[str, Any]:
        """Execute a non-streaming native provider request."""

        logger = context.transaction_logger
        protocol = get_protocol(context.protocol_name)
        self._trace(context, "native_protocol_selected", {"protocol": protocol.name}, direction="metadata", stage="protocol")
        try:
            self._trace(context, "raw_native_client_request", raw_request, direction="request", stage="client")
            protocol_context = context.protocol_context()
            unified_request = protocol.parse_request(raw_request, protocol_context)
            self._trace(context, "parsed_native_unified_request", unified_request, direction="request", stage="protocol")
            provider_request = protocol.build_request(unified_request, protocol_context)
            self._trace(context, "built_native_provider_request", provider_request, direction="request", stage="protocol")
            adapters = [get_adapter(name) for name in context.adapter_names]
            provider_request = await run_adapter_chain(adapters, provider_request, context.adapter_context(), stage="request")
            self._trace(context, "after_request_adapter_chain", provider_request, direction="request", stage="adapter")
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
            self._trace(context, "parsed_native_unified_response", unified_response, direction="response", stage="protocol")
            provider_response = protocol.format_response(unified_response, protocol_context)
            self._trace(context, "formatted_native_response", provider_response, direction="response", stage="protocol")
            provider_response = await run_adapter_chain(adapters, provider_response, context.adapter_context(), stage="response")
            self._trace(context, "after_response_adapter_chain", provider_response, direction="response", stage="adapter")
            await cache_engine.extract("response", provider_response, context.field_cache_context(), transaction_logger=logger)
            usage_record = extract_usage_record(
                provider_response,
                provider=context.provider,
                model=context.model,
                source="native_provider_response",
            )
            cost_breakdown = CostCalculator().calculate(usage_record, model=context.model, provider=context.provider)
            self._trace(
                context,
                "usage_accounting_summary",
                {"usage": usage_record.to_dict(), "cost": cost_breakdown.to_dict()},
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
            self._trace(context, "raw_native_client_request", raw_request, direction="request", stage="client")
            protocol_context = context.protocol_context()
            request_payload = dict(raw_request)
            request_payload["stream"] = True
            unified_request = protocol.parse_request(request_payload, protocol_context)
            self._trace(context, "parsed_native_unified_request", unified_request, direction="request", stage="protocol")
            provider_request = protocol.build_request(unified_request, protocol_context)
            self._trace(context, "built_native_provider_request", provider_request, direction="request", stage="protocol")
            adapters = [get_adapter(name) for name in context.adapter_names]
            provider_request = await run_adapter_chain(adapters, provider_request, context.adapter_context(), stage="request")
            self._trace(context, "after_request_adapter_chain", provider_request, direction="request", stage="adapter")
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
                self._trace(context, "parsed_native_unified_stream_event", event, direction="stream", stage="protocol", snapshot=False)
                event_payload = stream_event_payload(event)
                self._trace(context, "parsed_native_stream_event", event_payload, direction="stream", stage="protocol")
                if event.type == "done":
                    break
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
            _redact_field_cache_paths(data, context, direction),
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


def _redact_field_cache_paths(data: Any, context: NativeProviderContext, direction: str) -> Any:
    """Redact configured cache paths before broad native payload traces.

    Field-cache rules can inject opaque state under arbitrary configured keys,
    so key-based trace redaction is not enough. Native traces apply the active
    rules' source and injection paths to a copy before handing data to the normal
    transaction trace sanitizer.
    """

    if not context.field_cache_rules:
        return data
    redacted = serialize_value(deepcopy(data))
    for rule in context.field_cache_rules:
        paths: list[str] = []
        if direction == "request" and rule.inject:
            paths.append(rule.inject.path)
        if direction in {"response", "stream"}:
            paths.append(rule.path)
        for path in _trace_redaction_paths(paths, direction=direction):
            try:
                tokens = parse_path(path)
                _redact_path(redacted, tokens)
                _redact_leaf_key(redacted, tokens)
            except (FieldCachePathError, TypeError, ValueError):
                continue
    return redacted


def _trace_redaction_paths(paths: list[str], *, direction: str) -> list[str]:
    """Return configured paths plus raw-stream envelope fallbacks for traces."""

    expanded: list[str] = []
    for path in paths:
        expanded.append(path)
        if direction == "stream" and path.startswith("raw."):
            expanded.append(path[4:])
    return expanded


def _redact_path(value: Any, tokens: tuple[PathToken, ...]) -> None:
    if not tokens:
        return
    token = tokens[0]
    rest = tokens[1:]
    if token.kind == "key":
        if isinstance(value, dict) and token.value in value:
            if rest:
                _redact_path(value[token.value], rest)
            else:
                value[token.value] = REDACTED
        return
    if token.kind == "index":
        if isinstance(value, list) and value:
            index = int(token.value)
            if -len(value) <= index < len(value):
                if rest:
                    _redact_path(value[index], rest)
                else:
                    value[index] = REDACTED
        return
    if token.kind == "wildcard":
        if isinstance(value, dict):
            for key in list(value.keys()):
                if rest:
                    _redact_path(value[key], rest)
                else:
                    value[key] = REDACTED
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if rest:
                    _redact_path(item, rest)
                else:
                    value[index] = REDACTED


def _redact_leaf_key(value: Any, tokens: tuple[PathToken, ...]) -> None:
    """Redact the configured terminal key wherever stream traces duplicate it."""

    leaf = next((token.value for token in reversed(tokens) if token.kind == "key"), None)
    if not leaf:
        return
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key == leaf:
                value[key] = REDACTED
            else:
                _redact_leaf_key(item, tokens)
    elif isinstance(value, list):
        for item in value:
            _redact_leaf_key(item, tokens)
