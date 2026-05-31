# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Opt-in executor for provider-native protocol calls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, AsyncGenerator

from ..adapters import get_adapter, run_adapter_chain
from ..field_cache import FieldCacheEngine, InMemoryFieldCacheStore
from ..field_cache.paths import FieldCachePathError, PathToken, parse_path
from ..protocols import ProtocolError, get_protocol, serialize_value
from ..protocols.types import UnifiedRequest
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
        self._ensure_supported_operation(protocol, context)
        self._trace(context, "native_protocol_selected", {"protocol": protocol.name}, direction="metadata", stage="protocol")
        try:
            self._trace(context, "raw_native_client_request", raw_request, direction="request", stage="client")
            cache_engine = FieldCacheEngine(context.field_cache_rules, store=self.field_cache_store)
            context = await self._inject_metadata(context, cache_engine)
            protocol_context = context.protocol_context()
            unified_request = protocol.parse_request(raw_request, protocol_context)
            self._trace(context, "parsed_native_unified_request", unified_request, direction="request", stage="protocol")
            await cache_engine.extract("unified_request", serialize_value(unified_request), context.field_cache_context(), transaction_logger=logger)
            self._trace(context, "after_unified_request_field_cache_extraction", {"source": "unified_request"}, direction="request", stage="adapter", snapshot=False)
            unified_request = await self._inject_unified_request(unified_request, context, cache_engine)
            provider_request = protocol.build_request(unified_request, protocol_context)
            self._trace(context, "built_native_provider_request", provider_request, direction="request", stage="protocol")
            adapters = [get_adapter(name) for name in context.adapter_names]
            adapter_context = context.adapter_context()
            adapter_context.transaction_logger = None
            provider_request = await run_adapter_chain(adapters, provider_request, adapter_context, stage="request")
            self._trace(context, "after_request_adapter_chain", provider_request, direction="request", stage="adapter")
            provider_request, _ = await cache_engine.inject(
                "request",
                provider_request,
                context.field_cache_context(),
                transaction_logger=logger,
            )
            self._trace(context, "after_field_cache_injection", provider_request, direction="request", stage="adapter")
            await cache_engine.extract("request", provider_request, context.field_cache_context(), transaction_logger=logger)
            self._trace(context, "after_request_field_cache_extraction", {"source": "request"}, direction="request", stage="adapter", snapshot=False)
            self._trace(context, "native_provider_request", provider_request, direction="request", stage="provider")
            raw_response = await transport.post_json(context.endpoint, headers=context.headers, payload=provider_request)
            self._trace(context, "raw_native_provider_response", raw_response, direction="response", stage="provider")
            await cache_engine.extract("response", raw_response, context.field_cache_context(), transaction_logger=logger)
            self._trace(context, "after_response_field_cache_extraction", {"source": "response", "payload": "raw_provider_response"}, direction="response", stage="adapter", snapshot=False)
            unified_response = protocol.parse_response(raw_response, protocol_context)
            self._trace(context, "parsed_native_unified_response", unified_response, direction="response", stage="protocol")
            await cache_engine.extract("unified_response", serialize_value(unified_response), context.field_cache_context(), transaction_logger=logger)
            self._trace(context, "after_unified_response_field_cache_extraction", {"source": "unified_response"}, direction="response", stage="adapter", snapshot=False)
            response_protocol = get_protocol(context.client_protocol_name) if context.client_protocol_name else protocol
            response_context = context.protocol_context(target_protocol=response_protocol.name)
            self._trace(context, "native_response_protocol_selected", {"protocol": response_protocol.name}, direction="metadata", stage="protocol", snapshot=False)
            provider_response = response_protocol.format_response(unified_response, response_context)
            self._trace(context, "formatted_native_response", provider_response, direction="response", stage="protocol")
            adapter_context = context.adapter_context()
            adapter_context.transaction_logger = None
            provider_response = await run_adapter_chain(adapters, provider_response, adapter_context, stage="response")
            self._trace(context, "after_response_adapter_chain", provider_response, direction="response", stage="adapter")
            usage_record = extract_usage_record(
                provider_response,
                provider=context.provider,
                model=context.model,
                source="native_provider_response",
            )
            raw_usage_record = extract_usage_record(
                raw_response,
                provider=context.provider,
                model=context.model,
                source="native_provider_raw_response",
            )
            if usage_record.provider_reported_cost is None and raw_usage_record.provider_reported_cost is not None:
                usage_record = replace(
                    usage_record,
                    provider_reported_cost=raw_usage_record.provider_reported_cost,
                    cost_currency=raw_usage_record.cost_currency,
                    cost_source=raw_usage_record.cost_source,
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
        self._ensure_supported_operation(protocol, context)
        self._trace(context, "native_protocol_selected", {"protocol": protocol.name}, direction="metadata", stage="protocol")
        try:
            self._trace(context, "raw_native_client_request", raw_request, direction="request", stage="client")
            cache_engine = FieldCacheEngine(context.field_cache_rules, store=self.field_cache_store)
            context = await self._inject_metadata(context, cache_engine)
            protocol_context = context.protocol_context()
            request_payload = dict(raw_request)
            request_payload["stream"] = True
            unified_request = protocol.parse_request(request_payload, protocol_context)
            self._trace(context, "parsed_native_unified_request", unified_request, direction="request", stage="protocol")
            await cache_engine.extract("unified_request", serialize_value(unified_request), context.field_cache_context(), transaction_logger=logger)
            self._trace(context, "after_unified_request_field_cache_extraction", {"source": "unified_request"}, direction="request", stage="adapter", snapshot=False)
            unified_request = await self._inject_unified_request(unified_request, context, cache_engine)
            provider_request = protocol.build_request(unified_request, protocol_context)
            self._trace(context, "built_native_provider_request", provider_request, direction="request", stage="protocol")
            adapters = [get_adapter(name) for name in context.adapter_names]
            adapter_context = context.adapter_context()
            adapter_context.transaction_logger = None
            provider_request = await run_adapter_chain(adapters, provider_request, adapter_context, stage="request")
            self._trace(context, "after_request_adapter_chain", provider_request, direction="request", stage="adapter")
            provider_request, _ = await cache_engine.inject(
                "request",
                provider_request,
                context.field_cache_context(),
                transaction_logger=logger,
            )
            self._trace(context, "after_field_cache_injection", provider_request, direction="request", stage="adapter")
            await cache_engine.extract("request", provider_request, context.field_cache_context(), transaction_logger=logger)
            self._trace(context, "after_request_field_cache_extraction", {"source": "request"}, direction="request", stage="adapter", snapshot=False)
            self._trace(context, "native_provider_stream_request", provider_request, direction="request", stage="provider")
            usage_record = extract_usage_record(None, provider=context.provider, model=context.model, source="native_provider_stream")
            async for raw_chunk in transport.stream_json_lines(context.endpoint, headers=context.headers, payload=provider_request):
                self._trace(context, "raw_native_provider_stream_chunk", raw_chunk, direction="stream", stage="provider")
                event = protocol.parse_stream_event(raw_chunk, protocol_context)
                self._trace(context, "parsed_native_unified_stream_event", event, direction="stream", stage="protocol", snapshot=False)
                usage_record = _merge_stream_usage_records(
                    usage_record,
                    extract_usage_record(serialize_value(event), provider=context.provider, model=context.model, source="native_stream_event"),
                    extract_usage_record(raw_chunk, provider=context.provider, model=context.model, source="native_raw_stream_event"),
                )
                if event.type == "done":
                    event_payload = stream_event_payload(event)
                    self._trace(context, "parsed_native_stream_event", event_payload, direction="stream", stage="protocol")
                    break
                adapter_context = context.adapter_context()
                # Native stream traces apply field-cache path redaction below.
                # Suppress generic adapter-chain snapshots here so provider state
                # cannot leak before rule-aware redaction runs.
                adapter_context.transaction_logger = None
                event = await run_adapter_chain(adapters, event, adapter_context, stage="stream_event")
                self._trace(context, "after_stream_event_adapter_chain", event, direction="stream", stage="adapter", snapshot=False)
                await cache_engine.extract("unified_stream_event", serialize_value(event), context.field_cache_context(), transaction_logger=logger)
                self._trace(context, "after_unified_stream_event_field_cache_extraction", {"source": "unified_stream_event"}, direction="stream", stage="adapter", snapshot=False)
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
                response_protocol = get_protocol(context.client_protocol_name) if context.client_protocol_name else protocol
                formatted = response_protocol.format_stream_event(event, context.protocol_context(target_protocol=response_protocol.name))
                self._trace(context, "formatted_client_stream_event", formatted, direction="stream", stage="final", snapshot=False)
                yield formatted
            cost_breakdown = CostCalculator().calculate(usage_record, model=context.model, provider=context.provider)
            self._trace(
                context,
                "usage_accounting_summary",
                {"usage": usage_record.to_dict(), "cost": cost_breakdown.to_dict()},
                direction="metadata",
                stage="final",
                snapshot=False,
            )
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

    @staticmethod
    def _ensure_supported_operation(protocol: Any, context: NativeProviderContext) -> None:
        """Fail before transport when provider and protocol operations disagree."""

        if protocol.supports_operation(context.operation):
            return
        raise ProtocolError(
            f"provider {context.provider} requested unsupported operation {context.operation!r}",
            protocol=protocol.name,
            pass_name="native_operation_check",
            payload={"provider": context.provider, "model": context.model, "operation": context.operation},
        )

    async def _inject_metadata(self, context: NativeProviderContext, cache_engine: FieldCacheEngine) -> NativeProviderContext:
        """Inject cached metadata before protocol/adapter contexts are built."""

        metadata, operations = await cache_engine.inject(
            "metadata",
            dict(context.metadata),
            context.field_cache_context(),
            transaction_logger=context.transaction_logger,
        )
        if operations:
            self._trace(context, "after_metadata_field_cache_injection", metadata, direction="metadata", stage="adapter", snapshot=False)
        if metadata == context.metadata:
            return context
        return replace(context, metadata=metadata)

    async def _inject_unified_request(
        self,
        unified_request: UnifiedRequest,
        context: NativeProviderContext,
        cache_engine: FieldCacheEngine,
    ) -> UnifiedRequest:
        """Inject cached values into a serialized unified request and hydrate it."""

        serialized = serialize_value(unified_request)
        injected, operations = await cache_engine.inject(
            "unified_request",
            serialized,
            context.field_cache_context(),
            transaction_logger=context.transaction_logger,
        )
        if operations:
            self._trace(context, "after_unified_request_field_cache_injection", injected, direction="request", stage="adapter")
        if injected == serialized:
            return unified_request
        return _hydrate_unified_request(unified_request, injected)


def _merge_stream_usage_records(base: Any, event_record: Any, raw_record: Any) -> Any:
    """Merge native stream usage, preserving raw provider cost when needed."""

    selected = event_record if _usage_record_has_token_values(event_record) else base
    if not _usage_record_has_token_values(selected) and _usage_record_has_token_values(raw_record):
        selected = raw_record
    if selected.provider_reported_cost is None and base.provider_reported_cost is not None:
        selected = replace(
            selected,
            provider_reported_cost=base.provider_reported_cost,
            cost_currency=base.cost_currency,
            cost_source=base.cost_source,
        )
    if selected.provider_reported_cost is None and raw_record.provider_reported_cost is not None:
        selected = replace(
            selected,
            provider_reported_cost=raw_record.provider_reported_cost,
            cost_currency=raw_record.cost_currency,
            cost_source=raw_record.cost_source,
        )
    return selected


def _usage_record_has_values(record: Any) -> bool:
    return bool(
        record.input_tokens
        or record.completion_tokens
        or record.reasoning_tokens
        or record.cache_read_tokens
        or record.cache_write_tokens
        or record.raw_total_tokens
        or record.provider_reported_cost is not None
    )


def _usage_record_has_token_values(record: Any) -> bool:
    return bool(
        record.input_tokens
        or record.completion_tokens
        or record.reasoning_tokens
        or record.cache_read_tokens
        or record.cache_write_tokens
        or record.raw_total_tokens
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
        if direction == "metadata" and rule.inject and rule.inject.target == "metadata":
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


def _hydrate_unified_request(original: UnifiedRequest, injected: Any) -> UnifiedRequest:
    """Hydrate common unified-request fields after serialized cache injection.

    Field-cache path injection operates on JSON-like dictionaries. Protocol
    builders still expect `UnifiedRequest`, so this helper copies supported
    top-level fields back onto the dataclass while leaving complex message/tool
    objects untouched unless providers handle them through provider-payload rules.
    """

    if not isinstance(injected, dict):
        return original
    safe_fields = {
        "operation",
        "model",
        "stream",
        "input",
        "modalities",
        "files",
        "generation_params",
        "response_format",
        "previous_response_id",
        "metadata",
        "raw",
        "extra",
    }
    values = {field_name: getattr(original, field_name) for field_name in UnifiedRequest._fields}
    for field_name in safe_fields:
        if field_name in injected:
            values[field_name] = injected[field_name]
    return UnifiedRequest(**values)


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
