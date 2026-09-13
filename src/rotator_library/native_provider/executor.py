# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Opt-in executor for provider-native protocol calls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, AsyncGenerator, Dict, Optional

from ..adapters import get_adapter, run_adapter_chain
from ..field_cache import FieldCacheEngine, InMemoryFieldCacheStore
from ..field_cache.types import is_provider_continuation_path
from ..core.errors import StreamedAPIError, StructuredAPIResponseError, structured_api_response_error
from ..protocols.canonical import family_wire_name
from ..streaming.relay import RelayStreamItem, StreamRepairState
from ..field_cache.paths import FieldCachePathError, PathToken, parse_path
from ..hooks.types import HookAction, TransportView
from ..hooks.runner import PipelineRun, run_slot
from ..protocols import ProtocolError, get_protocol, serialize_value
from ..protocols.types import (
    Annotation,
    BuiltinToolCall,
    ContentBlock,
    ConversionWarning,
    CostDetails,
    MediaSource,
    OutputItem,
    ReasoningBlock,
    ToolCall,
    ToolDefinition,
    ToolResult,
    UnifiedMessage,
    UnifiedRequest,
    UnifiedResponse,
    Usage,
)
from ..transform_trace import REDACTED
from ..usage.accounting import extract_usage_record
from ..usage.costs import CostCalculator
from .context import NativeProviderContext
from .http import NativeHTTPTransport
from .streaming import stream_event_payload

# Terminal vocabulary mirror of client/stream_ops._CHAT_TERMINAL_EVENT_TYPES
# (kept local to avoid a client-layer import cycle; stream_ops owns the
# canonical set). G4: widened so native anthropic/responses terminals take
# the authoritative terminal path, not just chat `done`.
_NATIVE_TERMINAL_EVENT_TYPES = frozenset(
    {
        "done",
        "message_stop",
        "response.completed",
        "response.failed",
        "response.incomplete",
        "completed",
    }
)


class _HookBlock(Exception):
    """Internal carrier for a hook BLOCK verdict escaping a stage."""

    def __init__(self, message: str, error_type: str | None) -> None:
        super().__init__(message or "blocked by hook")
        self.message = message or "blocked by hook"
        self.error_type = error_type


class NativeProviderExecutor:
    """Run one native provider request through protocol/adapter/cache passes.

    The default field-cache store is process-local per executor. That preserves
    provider protocol state across native requests without adding a database;
    production callers can still inject a persistent store when needed.
    """

    def __init__(self, *, field_cache_store: Any = None) -> None:
        self.field_cache_store = field_cache_store or InMemoryFieldCacheStore()

    # -- G2 hookable pipeline ------------------------------------------------

    def _pipeline_run(self, context: NativeProviderContext) -> PipelineRun:
        """Get (or lazily mint) the per-request isolated pipeline run."""

        if context.pipeline_run is None:
            context.pipeline_run = PipelineRun(
                request_id=str(context.metadata.get("request_id", "")),
                provider=context.provider,
                model=context.model,
                credential_id=context.credential_id or "",
                session_id=context.session_id or "",
                scope_key=context.scope_key or "",
                classifier=context.classifier or "",
                operation=context.operation,
                class_hooks=context.hook_class_declarations,
                config_hooks=context.hook_config_declarations,
                global_hooks=context.hook_global_names,
            )
        return context.pipeline_run

    async def _fire(
        self,
        context: NativeProviderContext,
        stage: str,
        payload: Any,
        *,
        direction: str = "request",
        transport_view: TransportView | None = None,
        is_terminal: bool = False,
        event_index: int | None = None,
    ):
        """Run one declared hook/callback boundary; map verdicts to outcomes.

        BLOCK raises ``_HookBlock`` (converted to a structured,
        dialect-rendered error by the caller-facing shim in the client
        executor); RESPOND/DROP/REPLACE are surfaced on the returned outcome
        for the call site to honor. Payloads are only swapped when a hook
        actually modified them, preserving object identity for the D4 raw
        fast-path gates.
        """

        outcome = await run_slot(
            self._pipeline_run(context),
            stage,
            payload,
            direction=direction,
            transport=transport_view,
            is_terminal=is_terminal,
            event_index=event_index,
        )
        if outcome.action is HookAction.BLOCK:
            raise _HookBlock(outcome.message, outcome.error_type)
        if transport_view is not None and transport_view.changed:
            # Surface the rewrite on the wire-visible overlay record too —
            # nothing is invisible (user directive).
            if context.request_transport_overlays is None:
                context.request_transport_overlays = []
            if not any(o.get("kind") == "transport_rewrite" for o in context.request_transport_overlays):
                context.request_transport_overlays.append({
                    "kind": "transport_rewrite",
                    "stage": stage,
                    "endpoint": transport_view.endpoint,
                    "headers": dict(transport_view.headers or {}),
                    "timeout_seconds": transport_view.timeout_seconds,
                })
        return outcome

    async def execute(self, raw_request: dict[str, Any] | UnifiedRequest, context: NativeProviderContext, transport: NativeHTTPTransport) -> dict[str, Any]:
        """Execute a non-streaming native provider request."""

        logger = context.transaction_logger
        provider_protocol = get_protocol(context.protocol_name)
        input_protocol = get_protocol(context.input_protocol_name or context.protocol_name)
        client_protocol = get_protocol(context.client_protocol_name or context.input_protocol_name or context.protocol_name)
        context = _without_provider_continuation_rules(context)
        self._ensure_supported_operation(provider_protocol, context)
        self._trace(context, "native_protocol_selected", {"input_protocol": input_protocol.name, "provider_protocol": provider_protocol.name, "client_protocol": client_protocol.name}, direction="metadata", stage="protocol")
        try:
            self._trace(context, "raw_native_client_request", raw_request, direction="request", stage="client")
            cache_engine = FieldCacheEngine(context.field_cache_rules, store=self.field_cache_store)
            context = await self._inject_metadata(context, cache_engine)
            input_context = context.protocol_context(
                source_protocol=input_protocol.name,
                target_protocol=provider_protocol.name,
                source_provider=context.metadata.get("input_provider"),
                target_provider=context.provider,
            )
            unified_request = deepcopy(raw_request) if isinstance(raw_request, UnifiedRequest) else input_protocol.parse_request(raw_request, input_context)
            unified_request.model = context.model
            self._trace(context, "parsed_native_unified_request", unified_request, direction="request", stage="protocol")
            # G2 R5: hooks own the canonical request post-parse. A modified
            # canonical flips the D4 raw fast path to a rebuild (identity
            # check below), same as field-cache injection.
            outcome = await self._fire(context, "parsed_canonical", unified_request, direction="request")
            if outcome.modified:
                unified_request = outcome.payload
                self._trace(context, "after_parsed_canonical_hooks", unified_request, direction="request", stage="adapter")
            await cache_engine.extract("unified_request", serialize_value(unified_request), context.field_cache_context(), transaction_logger=logger)
            self._trace(context, "after_unified_request_field_cache_extraction", {"source": "unified_request"}, direction="request", stage="adapter", snapshot=False)
            request_before_injection = unified_request
            # G2 R6A: slot before canonical state injection.
            outcome = await self._fire(context, "canonical_state_inject_a", unified_request, direction="request")
            if outcome.modified:
                unified_request = outcome.payload
            unified_request = await self._inject_unified_request(unified_request, context, cache_engine)
            provider_state_compatible = unified_request is not request_before_injection
            # G2 R6B: slot after canonical state injection.
            outcome = await self._fire(context, "canonical_state_inject_b", unified_request, direction="request")
            if outcome.modified:
                unified_request = outcome.payload
                provider_state_compatible = True
            provider_context = context.protocol_context(
                source_protocol=input_protocol.name,
                target_protocol=provider_protocol.name,
                source_provider=context.metadata.get("input_provider"),
                target_provider=context.provider,
                provider_state_compatible=provider_state_compatible,
            )
            # D4 raw fast path: when client and provider speak the same
            # protocol and no semantic edits are pending, the ORIGINAL client
            # payload is the transport basis — no canonical rebuild can strip
            # source-native fields. Every deviation is a traced overlay.
            # Family-aware (G11): sibling variants are the same wire.
            same_protocol = family_wire_name(input_protocol.name) == family_wire_name(provider_protocol.name)
            raw_wire = context.raw_client_request if same_protocol and isinstance(context.raw_client_request, dict) else None
            overlays: list[dict[str, Any]] = []
            raw_basis_used = False
            if raw_wire is not None and unified_request is request_before_injection:
                wire_view = input_protocol.parse_request(deepcopy(raw_wire), input_context)
                if _wire_view_matches_unified(wire_view, unified_request):
                    provider_request = deepcopy(raw_wire)
                    raw_basis_used = True
                    # G3 provider-switch strip: opaque per-provider state
                    # (signatures, encrypted reasoning) that the CLIENT
                    # echoed back is foreign to any other executing
                    # provider — strip it in place as a DISCLOSED edit.
                    # Same provider (or no switch visible) keeps the byte
                    # path verbatim. Cache rows are never touched: the
                    # stripped state stays stored under its origin keys
                    # for the conversation's return home.
                    input_provider = context.metadata.get("input_provider")
                    if input_provider and input_provider != context.provider:
                        from ..protocols.opaque_strip import strip_foreign_opaque_state

                        stripped_fields = strip_foreign_opaque_state(provider_request, input_protocol.name)
                        if stripped_fields:
                            overlays.append({
                                "kind": "foreign_bound_state_stripped",
                                "from_provider": input_provider,
                                "fields": stripped_fields,
                            })
                    if "model" in provider_request and provider_request.get("model") != context.model:
                        # Model overlay: traced, in-body only. Protocols whose
                        # model rides the endpoint (Gemini) keep their native
                        # keyless shape — the endpoint already carries it.
                        overlays.append({"field": "model", "from": provider_request.get("model"), "to": context.model})
                        provider_request["model"] = context.model
                else:
                    provider_request = provider_protocol.build_request(unified_request, provider_context)
                    overlays.append({"kind": "canonical_rebuild", "reason": "payload_divergence"})
            else:
                provider_request = provider_protocol.build_request(unified_request, provider_context)
                if not same_protocol:
                    overlays.append({"kind": "canonical_rebuild", "reason": "cross_protocol"})
                elif provider_state_compatible:
                    overlays.append({"kind": "canonical_rebuild", "reason": "unified_state_injection"})
                else:
                    overlays.append({"kind": "canonical_rebuild", "reason": "no_wire_payload"})
            context.request_transport_overlays = overlays
            self._pipeline_run(context).context.state["transport_basis"] = {
                "basis": "raw" if raw_basis_used else "rebuild",
                "overlays": overlays,
            }
            self._trace(
                context,
                "request_transport_overlays",
                {"basis": "raw" if raw_basis_used else "rebuild", "overlays": overlays},
                direction="metadata",
                stage="protocol",
                snapshot=False,
            )
            if not raw_basis_used:
                self._trace(context, "built_native_provider_request", provider_request, direction="request", stage="protocol")
            else:
                self._trace(context, "raw_fast_path_request", provider_request, direction="request", stage="protocol")
            # G2 R7 (transport_basis_selected) + R8 (provider_built): full
            # power over the wire payload regardless of basis; hook edits are
            # recorded as overlays so nothing is invisible.
            outcome = await self._fire(context, "transport_basis_selected", provider_request, direction="request")
            if outcome.modified:
                provider_request = outcome.payload
                context.request_transport_overlays.append({"kind": "hook_edit", "stage": "transport_basis_selected"})
            outcome = await self._fire(context, "provider_built", provider_request, direction="request")
            if outcome.modified:
                provider_request = outcome.payload
                context.request_transport_overlays.append({"kind": "hook_edit", "stage": "provider_built"})
                self._trace(context, "after_provider_built_hooks", provider_request, direction="request", stage="adapter")
            # G10 Phase B: every transport overlay (foreign-state strip,
            # canonical rebuild, hook edit, transport rewrite) is value-level
            # in the change log — the overlay list has no other consumer.
            self._record_transport_overlays(context, logger)
            request_warnings = list(unified_request.warnings)
            adapters = [get_adapter(name) for name in context.adapter_names]
            adapter_context = context.adapter_context()
            adapter_context.transaction_logger = None
            provider_request = await run_adapter_chain(adapters, provider_request, adapter_context, stage="request")
            self._trace(context, "after_request_adapter_chain", provider_request, direction="request", stage="adapter")
            # G2 R10 (mutated band): hooks see the fully adapted payload.
            outcome = await self._fire(context, "mutated", provider_request, direction="request")
            if outcome.modified:
                provider_request = outcome.payload
                self._trace(context, "after_mutated_hooks", provider_request, direction="request", stage="adapter")
            # G2 R11A: slot before wire state injection.
            outcome = await self._fire(context, "state_inject_a", provider_request, direction="request")
            if outcome.modified:
                provider_request = outcome.payload
            provider_request, _ = await cache_engine.inject(
                "request",
                provider_request,
                context.field_cache_context(),
                transaction_logger=logger,
            )
            self._trace(context, "after_field_cache_injection", provider_request, direction="request", stage="adapter")
            # G2 R11B: slot after wire state injection.
            outcome = await self._fire(context, "state_inject_b", provider_request, direction="request")
            if outcome.modified:
                provider_request = outcome.payload
            await cache_engine.extract("request", provider_request, context.field_cache_context(), transaction_logger=logger)
            self._trace(context, "after_request_field_cache_extraction", {"source": "request"}, direction="request", stage="adapter", snapshot=False)
            # G2 stage correction: the provider finalizer is the LAST
            # pre-send payload edit — after adapters and cache injection
            # (audit R5: finalizer-before-adapters killed envelopes and let
            # cache injection write underneath the wrap).
            provider_request = self._prepare_provider_request(provider_request, context)
            self._trace(context, "provider_native_request_prepared", provider_request, direction="request", stage="provider")
            await self._validate_provider_request(provider_request, context)
            # G2 R12 (validated): veto-capable slot after all mutation.
            outcome = await self._fire(context, "validated", provider_request, direction="request")
            if outcome.modified:
                provider_request = outcome.payload
            self._trace(context, "native_provider_request", provider_request, direction="request", stage="provider")
            # G2 R13 (transport_ready): the transport slot — hooks may
            # rewrite endpoint/headers/timeout; rewrites are recorded as
            # overlays (nothing invisible) and applied to this call only.
            transport_view = TransportView(endpoint=context.endpoint, headers=dict(context.headers))
            outcome = await self._fire(context, "transport_ready", provider_request, direction="request", transport_view=transport_view)
            if outcome.modified:
                provider_request = outcome.payload
            send_endpoint = transport_view.endpoint or context.endpoint
            send_headers = transport_view.headers if transport_view.headers is not None else context.headers
            # Late overlays (transport rewrites land after the first sink).
            self._record_transport_overlays(context, logger)
            send_timeout = transport_view.timeout_seconds
            # G2 R14 (sent): last look at the exact outgoing wire.
            outcome = await self._fire(context, "sent", provider_request, direction="request")
            if outcome.modified:
                provider_request = outcome.payload
            if send_timeout is not None:
                raw_response = await transport.post_json(send_endpoint, headers=send_headers, payload=provider_request, timeout_seconds=send_timeout)
            else:
                raw_response = await transport.post_json(send_endpoint, headers=send_headers, payload=provider_request)
            self._trace(context, "raw_native_provider_response", raw_response, direction="response", stage="provider")
            structured_error = structured_api_response_error(raw_response)
            if structured_error:
                # G10: failed provider payloads are external inputs too —
                # fallback summaries quote them; they land as a boundary
                # before the raise (best-effort, never blocks the error).
                if logger is not None:
                    try:
                        logger.log_provider_response(_redact_field_cache_paths(raw_response, context, "response"))
                    except Exception:
                        pass
                raise structured_error
            # G2 P1 (response_received): full power over the raw provider
            # wire before parsing/adapters.
            outcome = await self._fire(context, "response_received", raw_response, direction="response")
            if outcome.modified:
                raw_response = outcome.payload
                self._trace(context, "after_response_received_hooks", raw_response, direction="response", stage="adapter")
            # G2 P2A: slot before response state extraction.
            outcome = await self._fire(context, "response_state_extract_a", raw_response, direction="response")
            if outcome.modified:
                raw_response = outcome.payload
            # W7 contract: response adapters are WIRE adapters — they run on
            # the provider's native response payload BEFORE parsing, exactly
            # once, in the provider's own dialect (never on the formatted
            # client payload, which varies by client protocol).
            adapter_context = context.adapter_context()
            adapter_context.transaction_logger = None
            raw_response = await run_adapter_chain(adapters, raw_response, adapter_context, stage="response")
            self._trace(context, "after_response_adapter_chain", raw_response, direction="response", stage="adapter")
            await cache_engine.extract("response", raw_response, context.field_cache_context(), transaction_logger=logger)
            self._trace(context, "after_response_field_cache_extraction", {"source": "response", "payload": "raw_provider_response"}, direction="response", stage="adapter", snapshot=False)
            # G2 P2B: slot after response state extraction.
            outcome = await self._fire(context, "response_state_extract_b", raw_response, direction="response")
            if outcome.modified:
                raw_response = outcome.payload
            # Response-side injection (wire target): cached state is restored
            # into the provider's raw response before parsing/formatting. The
            # no-op path is clean — engine.inject deep-copies and reports miss.
            raw_response, _ = await cache_engine.inject(
                "response",
                raw_response,
                context.field_cache_context(),
                transaction_logger=logger,
            )
            self._trace(context, "after_response_field_cache_injection", raw_response, direction="response", stage="adapter")
            response_context = context.protocol_context(
                source_protocol=provider_protocol.name,
                target_protocol=client_protocol.name,
                source_provider=context.provider,
                target_provider=None,
                provider_state_compatible=False,
            )
            unified_response = provider_protocol.parse_response(raw_response, response_context)
            unified_response.model = str(context.metadata.get("public_model") or unified_response.model or context.model)
            for warning in request_warnings:
                if warning not in unified_response.warnings:
                    unified_response.warnings.append(warning)
            self._trace(context, "parsed_native_unified_response", unified_response, direction="response", stage="protocol")
            # G2 P3 (response_parsed): hooks own the canonical response.
            outcome = await self._fire(context, "response_parsed", unified_response, direction="response")
            if outcome.modified:
                unified_response = outcome.payload
                self._trace(context, "after_response_parsed_hooks", unified_response, direction="response", stage="adapter")
            if outcome.action is HookAction.RESPOND and outcome.payload is not None:
                # Synthetic short-circuit: the hook answered the request; the
                # payload is used as-is in the client's shape. The private
                # disclosure channel never survives to a client (a hook
                # echoing a count-tokens payload would otherwise leak it).
                if isinstance(outcome.payload, dict):
                    outcome.payload.pop("_proxy_warnings", None)
                return deepcopy(outcome.payload) if isinstance(outcome.payload, (dict, list)) else outcome.payload
            # Response-side injection (canonical target): restore cached state
            # onto the parsed unified response before it is formatted for the
            # client. Symmetric with request-side unified_request injection.
            unified_response = await self._inject_unified_response(unified_response, context, cache_engine)
            await cache_engine.extract("unified_response", serialize_value(unified_response), context.field_cache_context(), transaction_logger=logger)
            # G10: the provider response is one of the two non-derivable
            # external inputs — always a boundary, never only a trace value.
            # Field-cache paths are redacted exactly like trace values.
            if logger is not None:
                logger.log_provider_response(_redact_field_cache_paths(raw_response, context, "response"))
            self._trace(context, "after_unified_response_field_cache_extraction", {"source": "unified_response"}, direction="response", stage="adapter", snapshot=False)
            self._trace(context, "native_response_protocol_selected", {"protocol": client_protocol.name}, direction="metadata", stage="protocol", snapshot=False)
            if raw_basis_used and client_protocol.name == provider_protocol.name:
                # D4 raw response passthrough: same protocol (request AND
                # response — D1 makes these identical in production; the
                # explicit check is defense-in-depth for hand-built contexts),
                # no proxy semantic edits — the provider's (adapted) response
                # IS the client's response, byte-for-byte (sidecar observation
                # above feeds usage/session/accounting). Response-stage wire
                # adapters already applied upstream of this gate, so they no
                # longer disable the passthrough.
                client_response = deepcopy(raw_response)
                self._trace(context, "raw_fast_path_response", client_response, direction="response", stage="protocol")
            else:
                client_response = client_protocol.format_response(unified_response, response_context)
                self._trace(context, "formatted_native_response", client_response, direction="response", stage="protocol")
            # G10 Phase B: conversion notes live in the change log — never
            # on the client payload, never on the console. This fires on
            # BOTH branches: same-protocol raw passthrough keeps its
            # request-side warnings too (they were silently dropped here
            # before). The count-token private channel is the exception: the
            # facade owns the logger and drains it after the response
            # returns, so logging here would duplicate the records.
            if (
                logger is not None
                and unified_response.warnings
                and not (isinstance(client_response, dict) and "_proxy_warnings" in client_response)
            ):
                logger.log_conversion_warnings(unified_response.warnings)
            # G2 P4 (response_formatted): the ONLY seam that sees the client
            # payload before it leaves the proxy.
            outcome = await self._fire(context, "response_formatted", client_response, direction="response")
            if outcome.modified:
                client_response = outcome.payload
                self._trace(context, "after_response_formatted_hooks", client_response, direction="response", stage="adapter")
            usage_record = extract_usage_record(
                client_response,
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
            # G2 P5 (usage_recorded): hooks own the accounting record.
            outcome = await self._fire(context, "usage_recorded", usage_record, direction="response")
            if outcome.modified:
                usage_record = outcome.payload
                cost_breakdown = CostCalculator().calculate(usage_record, model=context.model, provider=context.provider)
            self._trace(
                context,
                "usage_accounting_summary",
                {"usage": usage_record.to_dict(), "cost": cost_breakdown.to_dict()},
                direction="metadata",
                stage="final",
                snapshot=False,
            )
            self._trace(context, "final_client_response", client_response, direction="response", stage="final")
            return client_response
        except _HookBlock as block:
            raise structured_api_response_error(
                {
                    "error": {
                        "message": block.message,
                        "type": block.error_type or "invalid_request_error",
                        "code": block.error_type,
                    }
                }
            ) or StructuredAPIResponseError(
                block.message,
                error_type=block.error_type or "invalid_request",
                status_code=400,
            )
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

    async def stream(self, raw_request: dict[str, Any] | UnifiedRequest, context: NativeProviderContext, transport: NativeHTTPTransport) -> AsyncGenerator[Any, None]:
        """Execute a streaming native provider request and yield client events."""

        logger = context.transaction_logger
        protocol = get_protocol(context.protocol_name)
        input_protocol = get_protocol(context.input_protocol_name or context.protocol_name)
        client_protocol = get_protocol(context.client_protocol_name or context.input_protocol_name or context.protocol_name)
        # The executor holds the ORIGINAL context object; rebinding below
        # (dataclasses.replace) must not detach per-stream state from it.
        root_context = context
        context = _without_provider_continuation_rules(context)
        self._ensure_supported_operation(protocol, context)
        self._trace(context, "native_protocol_selected", {"input_protocol": input_protocol.name, "provider_protocol": protocol.name, "client_protocol": client_protocol.name}, direction="metadata", stage="protocol")
        try:
            self._trace(context, "raw_native_client_request", raw_request, direction="request", stage="client")
            cache_engine = FieldCacheEngine(context.field_cache_rules, store=self.field_cache_store)
            context = await self._inject_metadata(context, cache_engine)
            input_context = context.protocol_context(
                source_protocol=input_protocol.name,
                target_protocol=protocol.name,
                source_provider=context.metadata.get("input_provider"),
                target_provider=context.provider,
            )
            if isinstance(raw_request, UnifiedRequest):
                unified_request = deepcopy(raw_request)
            else:
                request_payload = dict(raw_request)
                request_payload["stream"] = True
                unified_request = input_protocol.parse_request(request_payload, input_context)
            unified_request.model = context.model
            unified_request.stream = True
            self._trace(context, "parsed_native_unified_request", unified_request, direction="request", stage="protocol")
            # G2 R5: canonical request hooks (stream path).
            outcome = await self._fire(context, "parsed_canonical", unified_request, direction="request")
            if outcome.modified:
                unified_request = outcome.payload
            await cache_engine.extract("unified_request", serialize_value(unified_request), context.field_cache_context(), transaction_logger=logger)
            self._trace(context, "after_unified_request_field_cache_extraction", {"source": "unified_request"}, direction="request", stage="adapter", snapshot=False)
            request_before_injection = unified_request
            # G2 R6A/B around canonical state injection (stream path).
            outcome = await self._fire(context, "canonical_state_inject_a", unified_request, direction="request")
            if outcome.modified:
                unified_request = outcome.payload
            unified_request = await self._inject_unified_request(unified_request, context, cache_engine)
            state_injected = unified_request is not request_before_injection
            outcome = await self._fire(context, "canonical_state_inject_b", unified_request, direction="request")
            if outcome.modified:
                unified_request = outcome.payload
                state_injected = True
            provider_context = context.protocol_context(
                source_protocol=input_protocol.name,
                target_protocol=protocol.name,
                source_provider=context.metadata.get("input_provider"),
                target_provider=context.provider,
                provider_state_compatible=(
                    state_injected
                    or (
                        input_protocol.name == protocol.name
                        and context.metadata.get("input_provider") == context.provider
                    )
                ),
            )
            provider_request = protocol.build_request(unified_request, provider_context)
            # G3 stream-strip parity: opaque per-provider state echoed by the
            # client is foreign to a switched executing provider. The stream
            # path has no raw basis, so the strip runs on the built provider
            # payload — a disclosed edit with the same discipline as the
            # non-stream raw path.
            overlays: list[dict[str, Any]] = list(context.request_transport_overlays or [])
            input_provider = context.metadata.get("input_provider")
            if input_provider and input_provider != context.provider:
                from ..protocols.opaque_strip import strip_foreign_opaque_state

                stripped_fields = strip_foreign_opaque_state(provider_request, input_protocol.name)
                if stripped_fields:
                    overlays.append({
                        "kind": "foreign_bound_state_stripped",
                        "from_provider": input_provider,
                        "fields": stripped_fields,
                    })
                    overlays.append({"kind": "canonical_rebuild", "reason": "foreign_opaque_state_stripped"})
            context.request_transport_overlays = overlays
            self._trace(
                context,
                "request_transport_overlays",
                {"basis": "rebuild", "overlays": overlays},
                direction="metadata",
                stage="protocol",
                snapshot=False,
            )
            self._trace(context, "built_native_provider_request", provider_request, direction="request", stage="protocol")
            # G2 R7/R8 (stream path).
            outcome = await self._fire(context, "transport_basis_selected", provider_request, direction="request")
            if outcome.modified:
                provider_request = outcome.payload
                context.request_transport_overlays.append({"kind": "hook_edit", "stage": "transport_basis_selected"})
            outcome = await self._fire(context, "provider_built", provider_request, direction="request")
            if outcome.modified:
                provider_request = outcome.payload
                context.request_transport_overlays.append({"kind": "hook_edit", "stage": "provider_built"})
                self._trace(context, "after_provider_built_hooks", provider_request, direction="request", stage="adapter")
            self._record_transport_overlays(context, logger)
            # G10 Phase B: stream request-side warnings are recorded too —
            # the non-stream path snapshots them at build; the stream path
            # used to drop them entirely.
            request_warnings = list(unified_request.warnings)
            if logger is not None and request_warnings:
                logger.log_conversion_warnings(request_warnings, stage="stream_request")
            adapters = [get_adapter(name) for name in context.adapter_names]
            adapter_context = context.adapter_context()
            adapter_context.transaction_logger = None
            provider_request = await run_adapter_chain(adapters, provider_request, adapter_context, stage="request")
            self._trace(context, "after_request_adapter_chain", provider_request, direction="request", stage="adapter")
            # G2 R10 + R11A/B (stream path).
            outcome = await self._fire(context, "mutated", provider_request, direction="request")
            if outcome.modified:
                provider_request = outcome.payload
            outcome = await self._fire(context, "state_inject_a", provider_request, direction="request")
            if outcome.modified:
                provider_request = outcome.payload
            provider_request, _ = await cache_engine.inject(
                "request",
                provider_request,
                context.field_cache_context(),
                transaction_logger=logger,
            )
            self._trace(context, "after_field_cache_injection", provider_request, direction="request", stage="adapter")
            outcome = await self._fire(context, "state_inject_b", provider_request, direction="request")
            if outcome.modified:
                provider_request = outcome.payload
            await cache_engine.extract("request", provider_request, context.field_cache_context(), transaction_logger=logger)
            self._trace(context, "after_request_field_cache_extraction", {"source": "request"}, direction="request", stage="adapter", snapshot=False)
            provider_request = self._request_stream_usage(context, provider_request)
            # G2 stage correction (stream path): finalizer LAST pre-send.
            provider_request = self._prepare_provider_request(provider_request, context)
            self._trace(context, "provider_native_request_prepared", provider_request, direction="request", stage="provider")
            await self._validate_provider_request(provider_request, context)
            outcome = await self._fire(context, "validated", provider_request, direction="request")
            if outcome.modified:
                provider_request = outcome.payload
            self._trace(context, "native_provider_stream_request", provider_request, direction="request", stage="provider")
            # G2 R13 (transport_ready, stream path) + S1 (stream_opened).
            transport_view = TransportView(endpoint=context.endpoint, headers=dict(context.headers))
            outcome = await self._fire(context, "transport_ready", provider_request, direction="request", transport_view=transport_view)
            if outcome.modified:
                provider_request = outcome.payload
            await self._fire(context, "stream_opened", provider_request, direction="stream", transport_view=transport_view)
            send_endpoint = transport_view.endpoint or context.endpoint
            send_headers = transport_view.headers if transport_view.headers is not None else context.headers
            # Late overlays (transport rewrites land after the first sink).
            self._record_transport_overlays(context, logger)
            usage_record = extract_usage_record(None, provider=context.provider, model=context.model, source="native_provider_stream")
            response_context = context.protocol_context(
                source_protocol=protocol.name,
                target_protocol=client_protocol.name,
                source_provider=context.provider,
                target_provider=None,
                provider_state_compatible=False,
            )
            event_index = -1
            stream_error: BaseException | None = None
            if root_context.stream_repair_state is None:
                root_context.stream_repair_state = StreamRepairState()
            stream_kwargs: dict[str, Any] = {}
            if transport_view.timeout_seconds is not None:
                stream_kwargs["timeout_seconds"] = transport_view.timeout_seconds
            # G4 conditional re-serialization: frames carry their raw wire
            # text alongside parsed events; the PIPELINE owns the relay
            # decision (protocol match + framing + per-frame edit signals —
            # repair_state.edited_by_hook flips on any hook/adapter edit).
            # Request-side edits (usage overlays) do NOT block relay: they
            # change what we ASK, not the fidelity of the answer's bytes.
            parse_stream_events_plural = getattr(protocol, "parse_stream_events", None)
            # G14: the protocol's declared transport selects the framing
            # decoder. Ollama declares jsonl (NDJSON); the four generative
            # protocols keep the SSE default.
            frame_transport = "jsonl" if protocol.supports_transport("jsonl") else "sse"

            async def _frame_iterator():
                # Prefer the G4 raw-frame seam; legacy transports (custom
                # stream_json_lines implementations) yield parsed chunks
                # wrapped as raw-less frames — relay impossible, formatter
                # path unaffected.
                from .http import RawStreamFrame

                frames = getattr(transport, "stream_raw_frames", None)
                if frames is not None:
                    async for frame in frames(send_endpoint, headers=send_headers, payload=provider_request, transport=frame_transport, **stream_kwargs):
                        yield frame
                    return
                async for chunk in transport.stream_json_lines(send_endpoint, headers=send_headers, payload=provider_request, **stream_kwargs):
                    yield RawStreamFrame(raw=None, parsed=chunk)

            try:
                async for raw_frame in _frame_iterator():
                    if raw_frame.is_comment:
                        # Provider heartbeat frames relay/observe as comments;
                        # they are external wire evidence — captured too.
                        if context.transaction_logger is not None and raw_frame.raw:
                            try:
                                context.transaction_logger.log_provider_frame(raw_frame.raw)
                            except Exception:
                                pass
                        yield RelayStreamItem(events=[], raw=raw_frame.raw, is_comment=True)
                        continue
                    raw_chunk = raw_frame.parsed
                    # G10: provider stream frames are external inputs —
                    # captured as boundary chunks (redacted like traces).
                    if context.transaction_logger is not None and raw_chunk is not None:
                        try:
                            context.transaction_logger.log_provider_frame(
                                _redact_field_cache_paths(raw_chunk, context, "response")
                            )
                        except Exception:
                            pass
                    self._trace(context, "raw_native_provider_stream_chunk", raw_chunk, direction="stream", stage="provider")
                    if parse_stream_events_plural is not None:
                        frame_events = list(parse_stream_events_plural(raw_chunk, response_context))
                    else:
                        frame_events = [protocol.parse_stream_event(raw_chunk, response_context)]
                    for parsed_event in frame_events:
                        self._trace(context, "parsed_native_unified_stream_event", parsed_event, direction="stream", stage="protocol", snapshot=False)
                    error_event = next(
                        (e for e in frame_events if e.type == "error" or e.error is not None),
                        None,
                    )
                    if error_event is not None:
                        error = error_event.error if isinstance(error_event.error, dict) else {"message": str(error_event.error or "Provider stream failed")}
                        raise StreamedAPIError(
                            str(error.get("message") or error.get("type") or error.get("code") or "Provider stream failed"),
                            data={"error": deepcopy(error)},
                        )
                    for parsed_event in frame_events:
                        usage_record = _merge_stream_usage_records(
                            usage_record,
                            extract_usage_record(serialize_value(parsed_event), provider=context.provider, model=context.model, source="native_stream_event"),
                            extract_usage_record(raw_chunk, provider=context.provider, model=context.model, source="native_raw_stream_event"),
                        )
                    terminal_event = next((e for e in frame_events if e.type in _NATIVE_TERMINAL_EVENT_TYPES), None)
                    is_terminal = terminal_event is not None
                    if terminal_event is None:
                        event_index += 1
                    # G4 repair evidence (provider's own reasons survive EOF).
                    for parsed_event in frame_events:
                        if parsed_event.stop_reason:
                            root_context.stream_repair_state.last_provider_reason = parsed_event.stop_reason
                            output_index = getattr(parsed_event, "output_index", None)
                            try:
                                choice_key = int(output_index) if output_index is not None else 0
                            except (TypeError, ValueError):
                                choice_key = 0
                            root_context.stream_repair_state.held_reasons[choice_key] = parsed_event.stop_reason
                        delta = parsed_event.delta or parsed_event.message
                        if delta is not None and getattr(delta, "tool_calls", None):
                            root_context.stream_repair_state.tools_seen = True
                    # G2 S2 (stream_event): hooks own EVERY event, including
                    # the terminal one (D3 — done no longer bypasses slots).
                    fired_events = []
                    hook_modified = False
                    for parsed_event in frame_events:
                        outcome = await self._fire(context, "stream_event", parsed_event, direction="stream", is_terminal=is_terminal, event_index=event_index)
                        if outcome.action is HookAction.DROP:
                            if is_terminal:
                                break
                            continue
                        if outcome.modified:
                            hook_modified = True
                        fired_events.append(outcome.payload)
                    if hook_modified:
                        root_context.stream_repair_state.edited_by_hook = True
                    # W7 contract (plan §2.5): stream adapters run on the NEUTRAL
                    # parsed event — provider frames are SSE-wrapped transport,
                    # not discrete wire payloads; neutral is the protocol-free
                    # seam where adapter edits stay client-agnostic. Terminal
                    # events run the SAME pass (continuation ids live in
                    # response.completed — extraction must see them).
                    for idx, parsed_event in enumerate(fired_events):
                        adapter_context = context.adapter_context()
                        # Native stream traces apply field-cache path redaction below.
                        # Suppress generic adapter-chain snapshots here so provider state
                        # cannot leak before rule-aware redaction runs.
                        adapter_context.transaction_logger = None
                        adapted = parsed_event
                        if terminal_event is None and adapters:
                            # Adapter chain contract: neutral non-terminal
                            # events only (the historical seam); extracts
                            # below still run for terminals. An EMPTY chain
                            # must not flip edited_by_hook — run_adapter_chain
                            # deep-copies even with no adapters.
                            adapted = await run_adapter_chain(adapters, parsed_event, adapter_context, stage="stream_event")
                            if adapted is not parsed_event:
                                root_context.stream_repair_state.edited_by_hook = True
                            fired_events[idx] = adapted
                            self._trace(context, "after_stream_event_adapter_chain", adapted, direction="stream", stage="adapter", snapshot=False)
                        await cache_engine.extract("unified_stream_event", serialize_value(adapted), context.field_cache_context(), transaction_logger=logger)
                        self._trace(context, "after_unified_stream_event_field_cache_extraction", {"source": "unified_stream_event"}, direction="stream", stage="adapter", snapshot=False)
                        event_payload = stream_event_payload(adapted)
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
                    if terminal_event is not None and fired_events:
                        # Terminal provider signal: hand the operational layer
                        # the authoritative events and stop reading the wire.
                        root_context.stream_usage_record = usage_record
                        context.stream_usage_record = usage_record
                        yield RelayStreamItem(events=fired_events, raw=raw_frame.raw)
                        break
                    yield RelayStreamItem(events=fired_events, raw=raw_frame.raw)
                # Post-loop success path: S3 fires INSIDE the try so S4
                # (finally) is always the last stream stage.
                root_context.stream_usage_record = usage_record
                context.stream_usage_record = usage_record
                # G2 S3 (stream_assembled): the assembled stream result — hooks
                # own the authoritative usage record of the whole stream.
                outcome = await self._fire(context, "stream_assembled", usage_record, direction="stream", is_terminal=True)
                if outcome.modified:
                    usage_record = outcome.payload
                    root_context.stream_usage_record = usage_record
                    context.stream_usage_record = usage_record
            except _HookBlock as block:
                raise structured_api_response_error(
                    {
                        "error": {
                            "message": block.message,
                            "type": block.error_type or "invalid_request_error",
                            "code": block.error_type,
                        }
                    }
                ) or StructuredAPIResponseError(
                    block.message,
                    error_type=block.error_type or "invalid_request",
                    status_code=400,
                )
            except BaseException as exc:
                stream_error = exc
                raise
            finally:
                # G2 S4 (stream_closed): guaranteed cleanup path — the stream
                # outcome is observable even on error/abandonment, and the
                # transport generator is always closed. Runs AFTER S3 on the
                # success path (S3 fires inside the try, post-loop).
                terminal_outcome = {
                    "status": "error" if stream_error is not None else "completed",
                    "events": event_index + 1,
                }
                try:
                    await self._fire(context, "stream_closed", terminal_outcome, direction="stream", is_terminal=True, event_index=event_index)
                except _HookBlock:
                    pass  # cleanup path must never fail the stream
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

    def _request_stream_usage(self, context: NativeProviderContext, provider_request: Dict[str, Any]) -> Dict[str, Any]:
        """Ask chat-wire streams for terminal usage (accounting integrity).

        OpenAI-compatible providers only include token usage in the final
        chunk when ``stream_options.include_usage`` is requested; without it
        the stream completes with zero-token accounting. The flag is forced
        true for accounting even when the client explicitly disabled it —
        a provider-required default (finalizer concern), applied after the
        transport basis is chosen and recorded as a traced overlay.
        """

        if context.protocol_name != "openai_chat" or not isinstance(provider_request, dict):
            return provider_request
        stream_options = provider_request.get("stream_options")
        if isinstance(stream_options, dict) and stream_options.get("include_usage"):
            return provider_request
        request = dict(provider_request)
        merged = dict(stream_options) if isinstance(stream_options, dict) else {}
        merged["include_usage"] = True
        request["stream_options"] = merged
        if context.request_transport_overlays is None:
            context.request_transport_overlays = []
        context.request_transport_overlays.append(
            {"field": "stream_options.include_usage", "reason": "stream_usage_accounting"}
        )
        self._trace(
            context,
            "native_stream_usage_overlay",
            {"stream_options": request["stream_options"]},
            direction="request",
            stage="provider",
            snapshot=False,
        )
        return request

    @staticmethod
    def _prepare_provider_request(
        provider_request: dict[str, Any],
        context: NativeProviderContext,
    ) -> dict[str, Any]:
        """Apply narrow provider quirks after protocol-native formatting."""

        prepared = dict(provider_request)
        public_model = context.metadata.get("public_model")
        if public_model:
            prepared["_proxy_model"] = public_model
        if context.request_preparer:
            prepared = dict(
                context.request_preparer(
                    prepared,
                    model=context.model,
                    operation=context.operation,
                )
            )
        prepared.pop("_proxy_model", None)
        return prepared

    @staticmethod
    async def _validate_provider_request(provider_request: dict[str, Any], context: NativeProviderContext) -> None:
        """Run provider validation only after the provider payload exists."""

        validator = context.request_validator
        if not callable(validator):
            return
        result = validator(provider_request, context.model)
        if hasattr(result, "__await__"):
            result = await result
        if result is False:
            raise ProtocolError(
                f"Request validation failed for {context.provider}/{context.model}",
                protocol=context.protocol_name,
                pass_name="provider_validation",
            )
        if isinstance(result, str):
            raise ProtocolError(
                result,
                protocol=context.protocol_name,
                pass_name="provider_validation",
            )

    @staticmethod
    def _record_transport_overlays(context: NativeProviderContext, logger: Any) -> None:
        """Sink each transport overlay into the change log (G10 Phase B).

        Idempotent: overlays already sunk (by identity) are skipped, so a
        second call after ``transport_ready``/send captures late overlays
        (transport rewrites, validated/sent edits) without double records.
        """

        if logger is None:
            return
        sunk = getattr(context, "_sunk_overlays", None)
        if sunk is None:
            sunk = set()
            context._sunk_overlays = sunk
        for overlay in context.request_transport_overlays or []:
            if id(overlay) in sunk:
                continue
            sunk.add(id(overlay))
            logger.log_runtime_event("protocol", "overlay", "transport overlay", overlay)

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
        # G3 reserved-key guard: identity fields the runtime owns can never
        # be overwritten by cache injection — a rule that smuggled a new
        # ``input_provider`` would defeat provider-switch detection.
        for reserved in ("input_provider", "public_model", "execution_profile"):
            if reserved in context.metadata and metadata.get(reserved) != context.metadata[reserved]:
                metadata[reserved] = context.metadata[reserved]
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

    async def _inject_unified_response(
        self,
        unified_response: UnifiedResponse,
        context: NativeProviderContext,
        cache_engine: FieldCacheEngine,
    ) -> UnifiedResponse:
        """Inject cached values into a serialized unified response and hydrate it."""

        serialized = serialize_value(unified_response)
        injected, operations = await cache_engine.inject(
            "unified_response",
            serialized,
            context.field_cache_context(),
            transaction_logger=context.transaction_logger,
        )
        if operations:
            self._trace(context, "after_unified_response_field_cache_injection", injected, direction="response", stage="adapter")
        if injected == serialized:
            return unified_response
        return _hydrate_unified_response(unified_response, injected)


def _wire_view_matches_unified(wire_view: Any, unified_request: Any) -> bool:
    """D4 gate: the pristine wire payload still matches the request we would send.

    Compares the full semantic core of the request — messages (in
    instruction-normalized order, so legal interleaved system messages do not
    spuriously diverge), system, tools, structured output, stream flag,
    generation params, metadata, modalities, files, previous_response_id,
    extensions, extra. Model is excluded (overlaid separately). Any divergence
    means the canonical rebuild (an explicit, traced overlay) must run instead
    of the raw basis — a mutation must never ship silently untraced.
    """

    if _normalized_messages(wire_view) != _normalized_messages(unified_request):
        return False
    if wire_view.system != unified_request.system:
        return False
    if wire_view.tools != unified_request.tools:
        return False
    if wire_view.response_format != unified_request.response_format:
        return False
    if bool(wire_view.stream) != bool(unified_request.stream):
        return False
    if wire_view.generation_params != unified_request.generation_params:
        return False
    if wire_view.metadata != unified_request.metadata:
        return False
    if wire_view.modalities != unified_request.modalities:
        return False
    if wire_view.files != unified_request.files:
        return False
    if wire_view.previous_response_id != unified_request.previous_response_id:
        return False
    if wire_view.extensions != unified_request.extensions:
        return False
    if wire_view.extra != unified_request.extra:
        return False
    if wire_view.operation != unified_request.operation:
        return False
    if wire_view.logical_operation != unified_request.logical_operation:
        return False
    if wire_view.input != unified_request.input:
        return False
    return True


def _normalized_messages(request: Any) -> list[Any]:
    """Instruction-first message order (the canonical normalization).

    Wire payloads may legally interleave system/developer messages; the
    canonical model hoists instructions. Comparing in normalized order keeps
    the D4 raw path for interleaved (still source-native) payloads while
    catching every real content mutation.
    """

    messages = list(request.messages)
    instructions = [m for m in messages if m.role in {"system", "developer"}]
    if not instructions:
        return messages
    conversation = [m for m in messages if m.role not in {"system", "developer"}]
    return instructions + conversation


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


def _without_provider_continuation_rules(context: NativeProviderContext) -> NativeProviderContext:
    """Suppress upstream continuation IDs when proxy history was expanded."""

    if not context.metadata.get("disable_provider_continuation"):
        return context
    rules = tuple(
        rule
        for rule in context.field_cache_rules
        if not (
            (rule.metadata or {}).get("provider_continuation")
            or (rule.inject and is_provider_continuation_path(rule.inject.path))
        )
    )
    return context if rules == context.field_cache_rules else replace(context, field_cache_rules=rules)


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
        if direction == "response":
            if rule.source in {"response", "unified_response"}:
                paths.append(rule.path)
            if rule.inject and rule.inject.target in {"response", "unified_response"}:
                paths.append(rule.inject.path)
        if direction == "stream":
            paths.append(rule.path)
        for path in _trace_redaction_paths(paths, direction=direction):
            try:
                tokens = parse_path(path)
                _redact_path(redacted, tokens)
                _redact_leaf_key(redacted, tokens)
            except (FieldCachePathError, TypeError, ValueError):
                continue
    return redacted


_REQUEST_SCALAR_FIELDS = frozenset(
    {
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
)

_RESPONSE_SCALAR_FIELDS = frozenset(
    {
        "operation",
        "logical_operation",
        "id",
        "model",
        "output",
        "data",
        "content_type",
        "stop_reason",
        "modalities",
        "metadata",
        "source_protocol",
        "extensions",
        "raw",
        "extra",
    }
)


def _hydrate_unified_request(original: UnifiedRequest, injected: Any) -> UnifiedRequest:
    """Hydrate unified-request fields after serialized cache injection.

    Field-cache path injection operates on JSON-like dictionaries. Protocol
    builders still expect ``UnifiedRequest``, so this helper copies supported
    top-level fields back onto the dataclass. Messages, system instructions,
    and tool definitions are hydrated too — injection into those fields is
    legitimate provider state and must never be silently dropped while the
    engine reports success.
    """

    if not isinstance(injected, dict):
        return original
    values = {field_name: getattr(original, field_name) for field_name in UnifiedRequest._fields}
    for field_name in _REQUEST_SCALAR_FIELDS:
        if field_name in injected:
            values[field_name] = injected[field_name]
    if isinstance(injected.get("messages"), list):
        values["messages"] = [
            _hydrate_unified_message(item) for item in injected["messages"] if isinstance(item, dict)
        ]
    if isinstance(injected.get("system"), list):
        values["system"] = [
            _hydrate_content_block(item) for item in injected["system"] if isinstance(item, dict)
        ]
    if isinstance(injected.get("tools"), list):
        values["tools"] = [
            _hydrate_tool_definition(item) for item in injected["tools"] if isinstance(item, dict)
        ]
    return UnifiedRequest(**values)


def _hydrate_unified_response(original: UnifiedResponse, injected: Any) -> UnifiedResponse:
    """Hydrate unified-response fields after serialized cache injection."""

    if not isinstance(injected, dict):
        return original
    values = {field_name: getattr(original, field_name) for field_name in UnifiedResponse._fields}
    for field_name in _RESPONSE_SCALAR_FIELDS:
        if field_name in injected:
            values[field_name] = injected[field_name]
    if isinstance(injected.get("messages"), list):
        values["messages"] = [
            _hydrate_unified_message(item) for item in injected["messages"] if isinstance(item, dict)
        ]
    if isinstance(injected.get("items"), list):
        values["items"] = [
            _hydrate_output_item(item) for item in injected["items"] if isinstance(item, dict)
        ]
    if isinstance(injected.get("usage"), dict):
        values["usage"] = _hydrate_usage(injected["usage"])
    if isinstance(injected.get("warnings"), list):
        values["warnings"] = [
            _hydrate_warning(item) for item in injected["warnings"] if isinstance(item, dict)
        ]
    return UnifiedResponse(**values)


def _hydrate_extra(data: dict[str, Any], known: set[str]) -> dict[str, Any]:
    """Merge serialized ``extra`` with any injected unknown keys."""

    raw_extra = data.get("extra")
    extra = {str(key): deepcopy(value) for key, value in raw_extra.items()} if isinstance(raw_extra, dict) else {}
    for key, value in data.items():
        if key not in known:
            extra[str(key)] = deepcopy(value)
    return extra


def _hydrate_unified_message(data: dict[str, Any]) -> UnifiedMessage:
    return UnifiedMessage(
        role=str(data.get("role") or ""),
        content=[
            _hydrate_content_block(item) for item in data.get("content") or [] if isinstance(item, dict)
        ],
        name=data.get("name"),
        tool_call_id=data.get("tool_call_id"),
        tool_calls=[
            _hydrate_tool_call(item) for item in data.get("tool_calls") or [] if isinstance(item, dict)
        ],
        reasoning=[
            _hydrate_reasoning_block(item)
            for item in data.get("reasoning") or []
            if isinstance(item, dict)
        ],
        index=data.get("index"),
        stop_reason=data.get("stop_reason"),
        raw=deepcopy(data.get("raw")),
        extra=_hydrate_extra(data, set(UnifiedMessage._fields)),
    )


def _hydrate_content_block(data: dict[str, Any]) -> ContentBlock:
    source = data.get("source")
    return ContentBlock(
        type=str(data.get("type") or "unknown"),
        text=data.get("text"),
        source=_hydrate_media_source(source) if isinstance(source, dict) else source,
        tool_call=_hydrate_tool_call(data["tool_call"]) if isinstance(data.get("tool_call"), dict) else None,
        tool_result=_hydrate_tool_result(data["tool_result"]) if isinstance(data.get("tool_result"), dict) else None,
        reasoning=_hydrate_reasoning_block(data["reasoning"]) if isinstance(data.get("reasoning"), dict) else None,
        refusal=data.get("refusal"),
        builtin_tool=_hydrate_builtin_tool(data["builtin_tool"]) if isinstance(data.get("builtin_tool"), dict) else None,
        annotations=[
            _hydrate_annotation(item)
            for item in data.get("annotations") or []
            if isinstance(item, dict)
        ],
        index=data.get("index"),
        raw=deepcopy(data.get("raw")),
        extra=_hydrate_extra(data, set(ContentBlock._fields)),
    )


def _hydrate_reasoning_block(data: dict[str, Any]) -> ReasoningBlock:
    return ReasoningBlock(
        type=str(data.get("type") or "reasoning"),
        text=data.get("text"),
        signature=data.get("signature"),
        encrypted_content=data.get("encrypted_content"),
        redacted=bool(data.get("redacted", False)),
        raw=deepcopy(data.get("raw")),
        extra=_hydrate_extra(data, set(ReasoningBlock._fields)),
    )


def _hydrate_media_source(data: dict[str, Any]) -> MediaSource:
    return MediaSource(
        kind=str(data.get("kind") or "url"),
        media_type=data.get("media_type"),
        url=data.get("url"),
        data=data.get("data"),
        file_id=data.get("file_id"),
        filename=data.get("filename"),
        detail=data.get("detail"),
        transcript=data.get("transcript"),
        raw=deepcopy(data.get("raw")),
        extra=_hydrate_extra(data, set(MediaSource._fields)),
    )


def _hydrate_tool_call(data: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=data.get("id"),
        name=data.get("name"),
        arguments=data.get("arguments"),
        type=str(data.get("type") or "function"),
        index=data.get("index"),
        signature=data.get("signature"),
        raw=deepcopy(data.get("raw")),
        extra=_hydrate_extra(data, set(ToolCall._fields)),
    )


def _hydrate_tool_result(data: dict[str, Any]) -> ToolResult:
    return ToolResult(
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
        content=data.get("content"),
        is_error=data.get("is_error"),
        raw=deepcopy(data.get("raw")),
        extra=_hydrate_extra(data, set(ToolResult._fields)),
    )


def _hydrate_tool_definition(data: dict[str, Any]) -> ToolDefinition:
    return ToolDefinition(
        name=str(data.get("name") or ""),
        description=data.get("description"),
        input_schema=dict(data.get("input_schema") or {}),
        type=str(data.get("type") or "function"),
        extra=_hydrate_extra(data, set(ToolDefinition._fields)),
    )


def _hydrate_annotation(data: dict[str, Any]) -> Annotation:
    return Annotation(
        type=str(data.get("type") or "citation"),
        url=data.get("url"),
        title=data.get("title"),
        citation=data.get("citation"),
        start_index=data.get("start_index"),
        end_index=data.get("end_index"),
        document_index=data.get("document_index"),
        raw=deepcopy(data.get("raw")),
        extra=_hydrate_extra(data, set(Annotation._fields)),
    )


def _hydrate_builtin_tool(data: dict[str, Any]) -> BuiltinToolCall:
    return BuiltinToolCall(
        kind=str(data.get("kind") or "web_search"),
        call_id=data.get("call_id"),
        status=data.get("status"),
        output=data.get("output"),
        raw=deepcopy(data.get("raw")),
        extra=_hydrate_extra(data, set(BuiltinToolCall._fields)),
    )


def _hydrate_output_item(data: dict[str, Any]) -> OutputItem:
    return OutputItem(
        type=str(data.get("type") or ""),
        id=data.get("id"),
        role=data.get("role"),
        content=[
            _hydrate_content_block(item) for item in data.get("content") or [] if isinstance(item, dict)
        ],
        tool_call=_hydrate_tool_call(data["tool_call"]) if isinstance(data.get("tool_call"), dict) else None,
        reasoning=_hydrate_reasoning_block(data["reasoning"]) if isinstance(data.get("reasoning"), dict) else None,
        status=data.get("status"),
        raw=deepcopy(data.get("raw")),
        extra=_hydrate_extra(data, set(OutputItem._fields)),
    )


def _hydrate_cost(data: Any) -> Optional[CostDetails]:
    if not isinstance(data, dict):
        return None
    metadata = data.get("metadata")
    return CostDetails(
        provider_reported_cost=data.get("provider_reported_cost"),
        estimated_cost=data.get("estimated_cost"),
        currency=str(data.get("currency") or "USD"),
        source=data.get("source"),
        metadata=dict(metadata) if isinstance(metadata, dict) else {},
    )


def _hydrate_usage(data: dict[str, Any]) -> Usage:
    def _int(key: str) -> int:
        try:
            return int(data.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    return Usage(
        input_tokens=_int("input_tokens"),
        output_tokens=_int("output_tokens"),
        total_tokens=_int("total_tokens"),
        cache_read_tokens=_int("cache_read_tokens"),
        cache_write_tokens=_int("cache_write_tokens"),
        reasoning_tokens=_int("reasoning_tokens"),
        audio_tokens=_int("audio_tokens"),
        output_audio_tokens=_int("output_audio_tokens"),
        accepted_prediction_tokens=_int("accepted_prediction_tokens"),
        rejected_prediction_tokens=_int("rejected_prediction_tokens"),
        cost=_hydrate_cost(data.get("cost")),
        raw=deepcopy(data.get("raw")),
        extra=_hydrate_extra(data, set(Usage._fields)),
    )


def _hydrate_warning(data: dict[str, Any]) -> ConversionWarning:
    return ConversionWarning(
        code=str(data.get("code") or ""),
        message=str(data.get("message") or ""),
        field=data.get("field"),
        source_protocol=data.get("source_protocol"),
        target_protocol=data.get("target_protocol"),
    )


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



