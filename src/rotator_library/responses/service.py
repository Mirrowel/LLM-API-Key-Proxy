# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Service layer for the OpenAI-compatible Responses API."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, AsyncGenerator, MutableMapping, NoReturn, Optional

from ..protocols import ProtocolContext
from ..protocols.canonical import complete_responses_object
from ..session_tracking import SessionTrackingHints
from ..client.scopes import derive_session_isolation_key
from ..core.errors import StructuredAPIResponseError
from ..usage.accounting import extract_usage_record
from ..usage.costs import CostCalculator
from ..protocols.responses import ResponsesProtocol
from .store import InMemoryResponsesStore, ResponsesStore
from .streaming import ResponsesSSEFormatter, ResponsesStreamEvent
from .types import ResponsesStoreSettings, StoredResponse
from .types import generate_response_id


# Proxy routing controls carried to RequestContextBuilder. Lives here (not the
# retired chat bridge) because the native create/stream paths consume it.
PROXY_ROUTING_KEYS = {"classifier", "api_keys", "providers", "private", "model_filters"}


def responses_session_hints(
    previous_response_id: Optional[str],
) -> Optional[SessionTrackingHints]:
    """Return proxy-internal sticky routing evidence for Responses continuations."""

    if not previous_response_id:
        return None
    anchor = f"responses_previous_response_id:{previous_response_id}"
    return SessionTrackingHints(
        global_strong_anchors=[anchor],
        affinity_key=anchor,
    )


_STORED_REQUEST_FIELDS = {
    "background",
    "conversation",
    "include",
    "input",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "metadata",
    "model",
    "parallel_tool_calls",
    "previous_response_id",
    "prompt",
    "prompt_cache_key",
    "reasoning",
    "safety_identifier",
    "service_tier",
    "store",
    "stream",
    "stream_options",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "truncation",
    "user",
}
_SCOPE_ACCESS_TOKEN_RE = re.compile(
    r"^(classifier:[0-9a-f]{24}|bundle:[0-9a-f]{64})\.([A-Za-z0-9_-]{32,64})$"
)


@dataclass(frozen=True)
class ResponsesRequestScope:
    """Internal scope plus the unforgeable capability returned to HTTP clients."""

    key: str
    access_token: str

    @property
    def access_token_hash(self) -> str:
        return hashlib.sha256(self.access_token.encode("utf-8")).hexdigest()


class ResponsesServiceError(ValueError):
    """Error with an HTTP-compatible status code for proxy routes."""

    def __init__(self, message: str, *, status_code: int = 400, error_type: str = "invalid_request_error") -> None:
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(message)

    def to_protocol_payload(self, protocol: str) -> dict[str, Any]:
        """Format this service failure in an independently selected protocol."""

        if protocol == "responses":
            return {
                "error": {
                    "message": str(self),
                    "type": self.error_type,
                    # Official ResponseError string vocabulary — never the
                    # numeric HTTP status (docs §2.2).
                    "code": _RESPONSES_FAILURE_CODES.get(str(self.error_type), "server_error"),
                }
            }
        normalized = {
            "authentication_error": "authentication",
            "permission_error": "forbidden",
            "rate_limit_error": "rate_limit",
            "invalid_request_error": "invalid_request",
            "not_found_error": "not_found",
        }.get(self.error_type, self.error_type)
        return StructuredAPIResponseError(
            str(self),
            error_type=normalized,
            status_code=self.status_code,
        ).to_protocol_payload(protocol)


class ResponsesService:
    """Create, store, retrieve, cancel, and delete Responses API objects.

    Execution is native: every request is sent to the provider through
    ``client.agenerate(input_protocol="responses")``; the retired chat bridge
    no longer participates in any path.
    """

    def __init__(
        self,
        *,
        protocol: Optional[ResponsesProtocol] = None,
        store: Optional[ResponsesStore] = None,
        store_settings: Optional[ResponsesStoreSettings] = None,
    ) -> None:
        self.store_settings = store_settings or ResponsesStoreSettings()
        self.protocol = protocol or ResponsesProtocol()
        self.store = store or InMemoryResponsesStore(max_items=self.store_settings.max_items)

    @staticmethod
    def request_scope_key(raw_request: dict[str, Any]) -> str:
        """Return the opaque caller/credential domain for a Responses request."""

        return _request_isolation_key(raw_request)

    @staticmethod
    def redact_request_for_logging(raw_request: dict[str, Any]) -> dict[str, Any]:
        """Return a recursive credential-free copy for transport-level logs."""

        redacted = _redact_sensitive_fields(raw_request)
        return redacted if isinstance(redacted, dict) else {}

    def prepare_request_scope(
        self,
        raw_request: dict[str, Any],
    ) -> ResponsesRequestScope:
        """Create the scope capability used by transport-facing retrieval APIs."""

        scope_key = self.request_scope_key(raw_request)
        access_token = (
            "public"
            if scope_key == "public"
            else f"{scope_key}.{secrets.token_urlsafe(32)}"
        )
        return ResponsesRequestScope(scope_key, access_token)

    def _resolve_request_scope(
        self,
        raw_request: dict[str, Any],
        request_scope: Optional[ResponsesRequestScope],
    ) -> ResponsesRequestScope:
        expected_key = self.request_scope_key(raw_request)
        resolved = request_scope or self.prepare_request_scope(raw_request)
        if resolved.key != expected_key:
            raise ResponsesServiceError(
                "Responses request scope does not match routing credentials",
                status_code=400,
            )
        return resolved

    async def close(self) -> None:
        """Close the configured response store and its owned background tasks."""

        close = getattr(self.store, "close", None)
        if close:
            await close()

    async def create_response(
        self,
        raw_request: dict[str, Any],
        client: Any,
        *,
        request: Optional[Any] = None,
        transaction_logger: Optional[Any] = None,
        request_scope: Optional[ResponsesRequestScope] = None,
        previous_response_access_token: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a non-streaming Responses object through native execution."""

        if not raw_request.get("model"):
            raise ResponsesServiceError("'model' is required", status_code=400)
        if raw_request.get("stream"):
            raise ResponsesServiceError("Use stream_response for streaming requests", status_code=400)
        _reject_unsupported_lifecycles(raw_request)

        resolved_scope = self._resolve_request_scope(raw_request, request_scope)
        isolation_key = resolved_scope.key
        safe_request = _safe_stored_request(raw_request)
        self._trace(transaction_logger, "responses_raw_request", safe_request, direction="request", stage="client")
        try:
            unified = self.protocol.parse_request(raw_request, ProtocolContext(source_protocol="responses"))
        except Exception as exc:
            self._log_transform_error(transaction_logger, "responses_parse_request", exc, safe_request)
            raise
        if transaction_logger:
            self._trace(transaction_logger, "responses_parsed_request", _redact_sensitive_fields(unified.to_dict()), direction="request", stage="protocol")

        parent = await self._load_previous_response(
            unified.previous_response_id,
            transaction_logger,
            expected_scope_key=isolation_key,
            access_token=previous_response_access_token,
            provider_passthrough=_provider_continuation_eligible(raw_request),
                raw_request_dict=raw_request,
        )
        try:
            parent_lineage = await self._load_response_lineage(
                parent,
                expected_scope_key=isolation_key,
            )
        except Exception as exc:
            self._log_transform_error(
                transaction_logger,
                "responses_lineage_expand",
                exc,
                _redact_sensitive_fields(unified.to_dict()),
            )
            raise
        session_hints = responses_session_hints(unified.previous_response_id)
        session_info: dict[str, Any] = {
            "scope_access_hash": resolved_scope.access_token_hash,
        }
        native_request = _expanded_responses_request(raw_request, parent_lineage)
        internal_kwargs = _internal_client_kwargs(client, session_hints, session_info)
        self._trace(
            transaction_logger,
            "responses_native_protocol_request",
            _redact_sensitive_fields(native_request),
            direction="request",
            stage="protocol",
            metadata={"lineage_depth": len(parent_lineage), "has_session_hints": bool(session_hints)},
        )
        try:
            response = await client.agenerate(
                native_request,
                input_protocol="responses",
                request=request,
                _disable_provider_continuation=bool(parent_lineage),
                **_routing_kwargs(raw_request),
                **internal_kwargs,
            )
        except StructuredAPIResponseError as exc:
            raise ResponsesServiceError(
                str(exc),
                error_type=exc.error_type,
                status_code=exc.http_status,
            ) from exc
        response_payload = self._response_to_dict(response)
        self._trace(transaction_logger, "responses_native_protocol_response", response_payload, direction="response", stage="provider")
        _record_responses_session_anchor(session_info, response_payload)
        self._trace(transaction_logger, "responses_parsed_response", response_payload, direction="response", stage="protocol")
        self._trace_responses_usage(transaction_logger, response_payload, unified.model, source="responses_response")

        # Store parity with the stream path: failed responses honor
        # store_failed=False (the operator's explicit policy).
        should_store = raw_request.get("store", True) and (
            response_payload.get("status") != "failed" or self.store_settings.store_failed
        )
        if should_store:
            if await self._safe_store(raw_request, response_payload, parent, session_info, transaction_logger, "responses_store_response"):
                self._trace(transaction_logger, "responses_stored_response", response_payload, direction="metadata", stage="final")

        self._trace(transaction_logger, "responses_final_response", response_payload, direction="response", stage="final")
        # W12: the responses route never calls log_response (no chat-shaped
        # body to file), so metadata is finalized here instead.
        if transaction_logger is not None and hasattr(transaction_logger, "finalize_metadata"):
            transaction_logger.finalize_metadata(status_code=200)
        return response_payload

    async def stream_response(
        self,
        raw_request: dict[str, Any],
        client: Any,
        *,
        request: Optional[Any] = None,
        transaction_logger: Optional[Any] = None,
        transport: str = "sse",
        request_scope: Optional[ResponsesRequestScope] = None,
        previous_response_access_token: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a Responses API request as HTTP SSE frames.

        Native-only: every in-tree client exposes ``agenerate``; the provider's
        own Responses frames pass through unchanged.
        """

        async for frame in self._stream_native_response(
            raw_request,
            client,
            request=request,
            transaction_logger=transaction_logger,
            transport=transport,
            request_scope=request_scope,
            previous_response_access_token=previous_response_access_token,
        ):
            yield frame

    async def stream_turn_events(
        self,
        raw_request: dict[str, Any],
        client: Any,
        *,
        request: Optional[Any] = None,
        transaction_logger: Optional[Any] = None,
        request_scope: Optional[ResponsesRequestScope] = None,
        previous_response_access_token: Optional[str] = None,
        local_cache: Optional[MutableMapping[str, StoredResponse]] = None,
    ) -> AsyncGenerator[ResponsesStreamEvent, None]:
        """Yield transport-neutral Responses events for any transport.

        The WebSocket seam: the same storage, anchor, usage, and failure
        semantics as the SSE stream, with formatting left to the transport.
        """

        async for event in self._stream_native_response(
            raw_request,
            client,
            request=request,
            transaction_logger=transaction_logger,
            transport="websocket",
            request_scope=request_scope,
            previous_response_access_token=previous_response_access_token,
            as_events=True,
            local_cache=local_cache,
        ):
            yield event

    async def _stream_native_response(
        self,
        raw_request: dict[str, Any],
        client: Any,
        *,
        request: Optional[Any],
        transaction_logger: Optional[Any],
        transport: str,
        request_scope: Optional[ResponsesRequestScope],
        previous_response_access_token: Optional[str],
        as_events: bool = False,
        local_cache: Optional[MutableMapping[str, StoredResponse]] = None,
    ) -> AsyncGenerator[Any, None]:
        """Stream through the canonical runtime while retaining Responses storage.

        ``as_events`` yields transport-neutral :class:`ResponsesStreamEvent`
        objects (the WebSocket seam) instead of SSE strings; storage,
        anchors, and usage accounting run identically in both modes.
        ``local_cache`` is the WebSocket connection-local continuation cache
        consulted before the global store (store=false / ZDR chains).
        """

        stream_request = dict(raw_request)
        stream_request["stream"] = True
        resolved_scope = self._resolve_request_scope(stream_request, request_scope)
        parent: Optional[StoredResponse] = None
        model = str(raw_request.get("model") or "unknown")
        try:
            unified = self.protocol.parse_request(
                stream_request,
                ProtocolContext(source_protocol="responses", transport=transport),
            )
            model = unified.model
            parent = await self._load_previous_response(
                unified.previous_response_id,
                transaction_logger,
                expected_scope_key=resolved_scope.key,
                access_token=previous_response_access_token,
                local_cache=local_cache,
                provider_passthrough=_provider_continuation_eligible(raw_request),
                raw_request_dict=raw_request,
            )
            parent_lineage = await self._load_response_lineage(
                parent,
                expected_scope_key=resolved_scope.key,
                local_cache=local_cache,
            )
            native_request = _expanded_responses_request(stream_request, parent_lineage)
            session_hints = responses_session_hints(unified.previous_response_id)
            session_info: dict[str, Any] = {
                "scope_access_hash": resolved_scope.access_token_hash,
            }
            self._trace(
                transaction_logger,
                "responses_native_protocol_stream_request",
                _redact_sensitive_fields(native_request),
                direction="request",
                stage="protocol",
                metadata={"lineage_depth": len(parent_lineage)},
            )
            response_stream = await client.agenerate(
                native_request,
                input_protocol="responses",
                request=request,
                _disable_provider_continuation=bool(parent_lineage),
                **_routing_kwargs(raw_request),
                **_internal_client_kwargs(client, session_hints, session_info),
            )
        except Exception as exc:
            # Post-start failures never escape into the transport: the client
            # receives a protocol-valid terminal sequence instead (defect 10).
            async for frame in self._terminal_stream_failure(
                stream_request,
                model,
                parent,
                exc,
                transaction_logger=transaction_logger,
                session_info={"scope_access_hash": resolved_scope.access_token_hash},
                as_events=as_events,
                local_cache=local_cache,
                last_sequence=-1,
            ):
                yield frame
            # Finalize AFTER the terminal frames so the error records the
            # terminal path wrote land in metadata.
            self._finalize_stream_metadata(transaction_logger, error=exc)
            return
        terminal_seen = False
        provider_error: Optional[Exception] = None
        response_id: Optional[str] = None
        # ONE sequence authority per response stream: the highest provider
        # sequence_number observed. A synthesized terminal continues it; the
        # module-global counter is never drawn on.
        last_sequence = -1
        response_context = ProtocolContext(
            model=unified.model,
            source_protocol="responses",
            target_protocol="responses",
            input_protocol="responses",
            provider_protocol="responses",
            client_protocol="responses",
            transport=transport,
            provider_state_compatible=False,
        )
        try:
            async for raw_frame in response_stream:
                if isinstance(raw_frame, str) and raw_frame.lstrip().startswith(":"):
                    # SSE comment heartbeats have no event-mode equivalent.
                    if not as_events:
                        yield raw_frame
                    continue
                event = self.protocol.parse_stream_event(raw_frame, response_context)
                payload = event.extra.get("payload") if isinstance(event.extra, dict) else None
                if isinstance(payload, dict):
                    sequence = payload.get("sequence_number")
                    if isinstance(sequence, int) and sequence > last_sequence:
                        last_sequence = sequence
                response_payload = payload.get("response") if isinstance(payload, dict) and isinstance(payload.get("response"), dict) else None
                if isinstance(response_payload, dict) and isinstance(response_payload.get("id"), str):
                    response_id = response_payload["id"]
                if response_payload and event.type in {"response.created", "response.in_progress"}:
                    # store_in_progress is honored on the native path: the
                    # provider's in-flight object is snapshotted so retrieval
                    # surfaces see the same partial state the bridge used to.
                    await self._store_stream_current_state(
                        stream_request,
                        response_payload,
                        parent,
                        transaction_logger=transaction_logger,
                        session_info=session_info,
                    )
                terminal_event = event.type in {"response.completed", "response.failed", "response.incomplete"}
                if response_payload and terminal_event:
                    _record_responses_session_anchor(session_info, response_payload)
                    self._trace_responses_usage(transaction_logger, response_payload, unified.model, source="responses_stream")
                    stored = await self._store_stream_response(
                        stream_request,
                        response_payload,
                        parent,
                        failed=event.type == "response.failed",
                        transaction_logger=transaction_logger,
                        session_info=session_info,
                        local_cache=local_cache,
                    )
                    self._trace(
                        transaction_logger,
                        "responses_stored_stream_response" if stored else "responses_store_skipped",
                        response_payload if stored else {"response_id": response_payload.get("id")},
                        direction="metadata",
                        stage="final",
                    )
                if as_events:
                    yield ResponsesStreamEvent(
                        str((payload or {}).get("type") or event.type or "response.event"),
                        payload if isinstance(payload, dict) else {"type": event.type or "response.event"},
                    )
                else:
                    yield raw_frame
                if terminal_event:
                    # A provider terminal ENDS the stream — with or without a
                    # nested response object. The synthesized failure only
                    # fires when the provider sent NO terminal at all.
                    terminal_seen = True
                    break
                if event.type == "error":
                    # An out-of-band provider ``error`` event is terminal for
                    # the whole stream too (docs §2.3); pass it through once.
                    terminal_seen = True
                    provider_error = ResponsesServiceError(
                        _stream_error_message(payload if isinstance(payload, dict) else {}),
                        status_code=502,
                        error_type="upstream_error",
                    )
                    break
        except Exception as exc:
            # Post-start failures never escape into the transport: the client
            # receives a protocol-valid terminal sequence instead (defect 10).
            async for frame in self._terminal_stream_failure(
                stream_request,
                unified.model,
                parent,
                exc,
                transaction_logger=transaction_logger,
                session_info=session_info,
                as_events=as_events,
                local_cache=local_cache,
                response_id=response_id,
                last_sequence=last_sequence,
            ):
                yield frame
            # Finalize AFTER the terminal frames so their error records land
            # in metadata (order matters: finalize reads _error_records).
            self._finalize_stream_metadata(transaction_logger, error=exc)
            return
        if not terminal_seen:
            # The stream ended without a terminal event — synthesize one.
            terminal_exc = ResponsesServiceError(
                "Responses stream ended without a terminal response event",
                status_code=502,
                error_type="upstream_error",
            )
            async for frame in self._terminal_stream_failure(
                stream_request,
                unified.model,
                parent,
                terminal_exc,
                transaction_logger=transaction_logger,
                session_info=session_info,
                as_events=as_events,
                local_cache=local_cache,
                response_id=response_id,
                last_sequence=last_sequence,
            ):
                yield frame
            self._finalize_stream_metadata(transaction_logger, error=terminal_exc)
            return
        # Terminal streams still get their L1 summary; a provider error
        # terminal finalizes as an error.
        self._finalize_stream_metadata(transaction_logger, error=provider_error)

    def _finalize_stream_metadata(
        self,
        transaction_logger: Optional[Any],
        *,
        error: Optional[BaseException] = None,
    ) -> None:
        """Best-effort metadata finalize for streamed responses (never raises)."""

        if transaction_logger is None or not hasattr(transaction_logger, "finalize_metadata"):
            return
        try:
            transaction_logger.finalize_metadata(status_code=200 if error is None else 500, error=error)
            if error is not None and hasattr(transaction_logger, "flush_capture_on_error"):
                transaction_logger.flush_capture_on_error(error)
        except Exception:
            pass

    async def _terminal_stream_failure(
        self,
        stream_request: dict[str, Any],
        model: str,
        parent: Optional[StoredResponse],
        exc: Exception,
        *,
        transaction_logger: Optional[Any],
        session_info: dict[str, Any],
        as_events: bool = False,
        local_cache: Optional[MutableMapping[str, StoredResponse]] = None,
        response_id: Optional[str] = None,
        last_sequence: int = -1,
    ) -> AsyncGenerator[Any, None]:
        """Emit the terminal failure frames for a failed native stream.

        Frame emission is isolated from the store attempt: a failing store
        must never cost the client its terminal frames (double-failure
        guard). ``response_id`` correlates the failure with a provider id the
        stream already emitted (no fresh minting); ``last_sequence`` is the
        stream's highest observed provider sequence_number so the synthesized
        terminal continues the SAME counter instead of drawing the module
        global.
        """

        error = _stream_failure_error(exc)
        minted_id = response_id or generate_response_id()
        failed = complete_responses_object(
            {
                "id": minted_id,
                "status": "failed",
                "model": model,
                "output": [],
                "error": error,
            },
            response_id=minted_id,
            model=model,
            status="failed",
        )
        try:
            self._log_transform_error(transaction_logger, "responses_native_stream", exc, stream_request)
            stored = await self._store_stream_response(
                stream_request,
                failed,
                parent,
                failed=True,
                transaction_logger=transaction_logger,
                session_info=session_info,
                local_cache=local_cache,
            )
            self._trace(
                transaction_logger,
                "responses_stored_failed_stream_response" if stored else "responses_store_skipped",
                {"response_id": failed["id"], "status": "failed"},
                direction="metadata",
                stage="final",
            )
        except Exception as store_exc:
            self._trace(
                transaction_logger,
                "responses_store_failed_stream_response_error",
                {"error": str(store_exc)},
                direction="metadata",
                stage="final",
            )
        # Wire convention: the nested {type, response} envelope, matching
        # provider-emitted terminal frames on the same stream. Event mode
        # (WebSocket) yields the neutral event — no [DONE] sentinel exists
        # on that transport; the terminal response.failed closes the turn.
        # The event frame (not the nested response object) carries the
        # monotonic sequence_number per the streaming-events reference.
        failed_event = ResponsesStreamEvent("response.failed", {"type": "response.failed", "response": failed})
        failed_event.payload["sequence_number"] = last_sequence + 1
        if as_events:
            yield failed_event
            return
        formatter = ResponsesSSEFormatter()
        yield formatter.format_stream_event(failed_event)
        yield formatter.format_stream_event(ResponsesStreamEvent("done", {}, terminal=True))

    async def validate_stream_request(
        self,
        raw_request: dict[str, Any],
        *,
        request_scope: Optional[ResponsesRequestScope] = None,
        previous_response_access_token: Optional[str] = None,
        local_cache: Optional[MutableMapping[str, StoredResponse]] = None,
    ) -> None:
        """Validate stream-only preconditions before an HTTP response starts.

        ``local_cache`` is the WebSocket connection-local continuation cache;
        passing it lets warmup validate a chain exactly like a real turn
        (local-first resolution) instead of 404ing on a store=false parent.
        """

        if not raw_request.get("model"):
            raise ResponsesServiceError("'model' is required", status_code=400)
        _reject_unsupported_lifecycles(raw_request)
        previous_response_id = raw_request.get("previous_response_id")
        if previous_response_id:
            resolved_scope = self._resolve_request_scope(raw_request, request_scope)
            await self._load_previous_response(
                str(previous_response_id),
                None,
                expected_scope_key=resolved_scope.key,
                access_token=previous_response_access_token,
                local_cache=local_cache,
                provider_passthrough=_provider_continuation_eligible(raw_request),
                raw_request_dict=raw_request,
            )

    async def get_response(
        self,
        response_id: str,
        *,
        scope_key: str = "public",
    ) -> dict[str, Any]:
        """Return a stored response payload or raise a 404-compatible error."""

        stored = await self._stored_or_not_found(response_id, scope_key)
        return deepcopy(stored.response)

    async def get_response_with_access_token(
        self,
        response_id: str,
        access_token: str = "public",
    ) -> dict[str, Any]:
        """Return a response only when the transport capability is valid."""

        stored = await self._stored_for_access(response_id, access_token)
        return deepcopy(stored.response)

    async def delete_response(
        self,
        response_id: str,
        *,
        scope_key: str = "public",
    ) -> dict[str, Any]:
        """Delete a stored response and return the official deletion object."""

        await self._stored_or_not_found(response_id, scope_key)
        deleted = await self.store.delete(response_id, scope_key)
        if not deleted:
            raise ResponsesServiceError(f"Response not found: {response_id}", status_code=404, error_type="not_found_error")
        return {"id": response_id, "object": "response", "deleted": True}

    async def delete_response_with_access_token(
        self,
        response_id: str,
        access_token: str = "public",
    ) -> dict[str, Any]:
        """Delete a response only when the transport capability is valid."""

        stored = await self._stored_for_access(response_id, access_token)
        deleted = await self.store.delete(response_id, stored.scope_key or "public")
        if not deleted:
            self._raise_response_not_found(response_id)
        return {"id": response_id, "object": "response", "deleted": True}

    async def cancel_response(
        self,
        response_id: str,
        *,
        scope_key: str = "public",
    ) -> dict[str, Any]:
        """Cancel a stored response and return its cancelled response object.

        Best-effort by contract: the stored row is marked ``cancelled`` and
        surfaced. In-flight provider cancellation is a SEAM — no in-tree
        provider exposes a cancel hook, so a provider-owned row only has its
        eligibility checked (provenance + current target family); the durable
        row is always updated locally.
        """

        stored = await self._stored_or_not_found(response_id, scope_key)
        await self._best_effort_provider_cancel(stored)
        return await self._mark_cancelled(stored)

    async def cancel_response_with_access_token(
        self,
        response_id: str,
        access_token: str = "public",
    ) -> dict[str, Any]:
        """Cancel a response only when the transport capability is valid."""

        stored = await self._stored_for_access(response_id, access_token)
        await self._best_effort_provider_cancel(stored)
        return await self._mark_cancelled(stored)

    async def _mark_cancelled(self, stored: StoredResponse) -> dict[str, Any]:
        """Persist the cancelled status and return the client-facing object.

        Terminal states never rewrite history: a completed/failed/incomplete
        response returns its current object unchanged (official cancel is a
        background-response lifecycle operation; this proxy exposes it as a
        foreground extension, so cancelling a finished row is a no-op).
        """

        if str(stored.status) in {"completed", "failed", "incomplete"}:
            response = stored.response if isinstance(stored.response, dict) else {}
            return deepcopy(complete_responses_object(
                response,
                response_id=stored.id,
                model=stored.model,
                status=str(stored.status),
            ))
        stored.status = "cancelled"
        response = stored.response if isinstance(stored.response, dict) else {}
        response = complete_responses_object(
            response,
            response_id=stored.id,
            model=stored.model,
            status="cancelled",
        )
        response["status"] = "cancelled"
        stored.response = response
        try:
            await self.store.save(stored)
        except Exception as exc:
            # Cancellation is best-effort: a store failure never turns the
            # client's cancel into a 500.
            self._log_transform_error(None, "responses_store_cancel", exc, {"response_id": stored.id})
        return deepcopy(response)

    async def _best_effort_provider_cancel(self, stored: StoredResponse) -> bool:
        """Check whether a provider-side cancel would apply (the seam).

        True only when the stored row is provider-owned AND the request's
        current routing target is that same Responses-family provider. No
        in-tree provider implements cancellation, so this reports eligibility
        without a network call; a future provider hook plugs in here.
        """

        metadata = stored.metadata if isinstance(stored.metadata, dict) else {}
        provider = metadata.get("provider")
        if not metadata.get("provider_owned") or not provider:
            return False
        target = _provider_continuation_target({"model": stored.model})
        return target is not None and target == provider

    async def list_input_items(
        self,
        response_id: str,
        *,
        scope_key: str = "public",
        limit: int = 20,
        after: Optional[str] = None,
        order: str = "desc",
    ) -> dict[str, Any]:
        """Return a paginated official input-items list envelope."""

        stored = await self._stored_or_not_found(response_id, scope_key)
        return _input_items_page(stored.input_items, limit=limit, after=after, order=order)

    async def list_input_items_with_access_token(
        self,
        response_id: str,
        access_token: str = "public",
        *,
        limit: int = 20,
        after: Optional[str] = None,
        order: str = "desc",
    ) -> dict[str, Any]:
        """Return input items only when the transport capability is valid."""

        stored = await self._stored_for_access(response_id, access_token)
        return _input_items_page(stored.input_items, limit=limit, after=after, order=order)

    async def _stored_for_access(
        self,
        response_id: str,
        access_token: str,
    ) -> StoredResponse:
        if access_token == "public":
            return await self._stored_or_not_found(response_id, "public")
        match = _SCOPE_ACCESS_TOKEN_RE.fullmatch(str(access_token))
        if not match:
            self._raise_response_not_found(response_id)
        scope_key = match.group(1)
        stored = await self._stored_or_not_found(response_id, scope_key)
        expected_hash = str(stored.metadata.get("scope_access_hash") or "")
        actual_hash = hashlib.sha256(str(access_token).encode("utf-8")).hexdigest()
        if not expected_hash or not hmac.compare_digest(expected_hash, actual_hash):
            self._raise_response_not_found(response_id)
        return stored

    async def _stored_or_not_found(
        self,
        response_id: str,
        scope_key: str,
    ) -> StoredResponse:
        stored = await self.store.get(response_id, scope_key)
        if stored is None or stored.scope_key != scope_key:
            self._raise_response_not_found(response_id)
        return stored

    @staticmethod
    def _raise_response_not_found(response_id: str) -> NoReturn:
        # previous_response_id must reference a response created through
        # this proxy with store enabled (foreign/typo'd ids 404 by design:
        # the store is capability-gated per credential domain — forwarding
        # unknown ids would allow cross-tenant continuation injection).
        raise ResponsesServiceError(
            f"Response not found: {response_id} (previous_response_id must reference a response created through this proxy with store enabled)",
            status_code=404,
            error_type="not_found_error",
        )

    async def _load_response_lineage(
        self,
        parent: Optional[StoredResponse],
        *,
        expected_scope_key: str,
        max_depth: int = 20,
        local_cache: Optional[MutableMapping[str, StoredResponse]] = None,
    ) -> list[StoredResponse]:
        """Return parent continuation lineage from oldest to newest."""

        if parent is None:
            return []
        lineage: list[StoredResponse] = []
        seen: set[str] = set()
        current: Optional[StoredResponse] = parent
        while current is not None and current.id not in seen and len(lineage) < max_depth:
            if current.scope_key != expected_scope_key:
                raise ResponsesServiceError(
                    f"Previous response not found: {current.id}",
                    status_code=404,
                    error_type="not_found_error",
                )
            seen.add(current.id)
            lineage.append(current)
            previous_id = current.request.get("previous_response_id") if isinstance(current.request, dict) else None
            if not previous_id:
                break
            previous_id = str(previous_id)
            if local_cache is not None and previous_id in local_cache:
                current = local_cache[previous_id]
                continue
            current = await self.store.get(previous_id, expected_scope_key)
        return list(reversed(lineage))

    async def _load_previous_response(
        self,
        response_id: Optional[str],
        transaction_logger: Optional[Any],
        *,
        expected_scope_key: str,
        access_token: Optional[str] = None,
        local_cache: Optional[MutableMapping[str, StoredResponse]] = None,
        provider_passthrough: bool = False,
        raw_request_dict: Optional[dict[str, Any]] = None,
    ) -> Optional[StoredResponse]:
        if not response_id:
            return None
        if local_cache is not None and response_id in local_cache:
            # Connection-local cache (WebSocket mode): in-memory only,
            # pre-scoped by construction — the guide's fast continuation
            # path that keeps store=false / ZDR chains working.
            return local_cache[response_id]
        if expected_scope_key == "public":
            stored = await self.store.get(response_id, "public")
            if stored is not None and stored.scope_key == "public":
                # G11 hybrid, second turn: a row the PROVIDER minted (our
                # store is just a mirror) continues on the provider's own
                # chain — replaying locally would sever its hidden prefix
                # (encrypted reasoning, cache keys) behind our stored suffix.
                # Riding the id keeps the provider chain; the provider
                # switched (fallback/redirect) → local replay instead
                # (cross-provider continuation can never succeed).
                if provider_passthrough and stored.metadata.get("provider_owned"):
                    # The row is only safe to ride when the SAME provider is
                    # targeted: a provider-owned row minted elsewhere cannot
                    # resolve on the current provider's chain. Provider
                    # switched (fallback) → fall through to local replay.
                    target_provider = _provider_continuation_target(raw_request_dict or {})
                    if target_provider is not None and stored.metadata.get("provider") == target_provider:
                        return None
                parent = stored
            elif provider_passthrough:
                # G11 hybrid: public-scope miss on a provider that speaks
                # the Responses family — the id may be the provider's own
                # (minted upstream, retained server-side). Return no local
                # parent; the id rides through verbatim and the provider
                # resolves it (preserving its encrypted-reasoning chains
                # and cache keys) or answers its own honest 404.
                return None
            else:
                self._raise_response_not_found(response_id)
                return None  # pragma: no cover - raises
        else:
            parent = await self._stored_for_access(response_id, access_token or "")
            if parent.scope_key != expected_scope_key:
                # Scoped misses never fall through to the provider — the
                # capability gate stays local and ahead of any fallback.
                self._raise_response_not_found(response_id)
        if transaction_logger:
            self._trace(
                transaction_logger,
                "responses_previous_response_loaded",
                parent.to_dict(),
                direction="metadata",
                stage="adapter",
                metadata={
                    "previous_response_id": response_id,
                    "output_count": len(parent.output_items),
                    "input_item_count": len(parent.input_items),
                    "context_expanded": True,
                },
            )
        return parent

    def _stored_response(
        self,
        raw_request: dict[str, Any],
        response_payload: dict[str, Any],
        parent: Optional[StoredResponse] = None,
        *,
        session_info: Optional[dict[str, Any]] = None,
    ) -> StoredResponse:
        # Stream-event payloads nest the response object under "response";
        # direct response payloads are flat. Normalize both.
        if isinstance(response_payload.get("response"), dict):
            response_payload = response_payload["response"]
        session_info = session_info or {}
        provider_created_at = response_payload.get("created_at")
        return StoredResponse(
            id=str(response_payload["id"]),
            # The proxy's receive time is authoritative for eviction/LRU; a
            # provider-stamped old date could otherwise evict a fresh answer
            # early. The provider value stays for diagnostics only.
            created_at=time.time(),
            model=str(response_payload.get("model") or raw_request.get("model") or ""),
            status=str(response_payload.get("status") or "completed"),
            request=_safe_stored_request(raw_request),
            response=deepcopy(response_payload),
            input_items=_input_items(raw_request),
            output_items=deepcopy(response_payload.get("output") or []),
            usage=deepcopy(response_payload.get("usage")) if isinstance(response_payload.get("usage"), dict) else None,
            metadata={
                "previous_response_id": parent.id if parent else raw_request.get("previous_response_id"),
                "response_id": response_payload.get("id"),
                "scope_access_hash": session_info.get("scope_access_hash"),
                # G11 hybrid provenance: which provider served this response —
                # retrieval fallback and continuation routing key on it.
                "provider": session_info.get("provider"),
                "provider_created_at": provider_created_at,
                # G11 hybrid ownership: this row is a local MIRROR of a
                # provider-minted response (we forwarded a provider id with
                # no local lineage and an eligible target). The provider's
                # own chain stays authoritative — later turns ride the id
                # instead of replaying our stored suffix.
                "provider_owned": (
                    parent is None
                    and bool(raw_request.get("previous_response_id"))
                    and _provider_continuation_eligible(raw_request)
                ),
            },
            session_id=session_info.get("session_id") or (parent.session_id if parent else None),
            scope_key=(
                session_info.get("scope_key")
                or (parent.scope_key if parent else None)
                or self.request_scope_key(raw_request)
            ),
            classifier=session_info.get("classifier") or (parent.classifier if parent else None),
            expires_at=_expires_at(self.store_settings),
        )

    @staticmethod
    def _response_to_dict(response: Any) -> Any:
        if isinstance(response, dict):
            return deepcopy(response)
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "dict"):
            return response.dict()
        return repr(response)

    @staticmethod
    def _trace(
        transaction_logger: Optional[Any],
        pass_name: str,
        data: Any,
        *,
        direction: str,
        stage: str,
        metadata: Optional[dict[str, Any]] = None,
        scrub_strings: bool = False,
    ) -> None:
        if not transaction_logger:
            return
        transaction_logger.log_transform_pass(
            pass_name,
            data,
            direction=direction,
            stage=stage,
            protocol="responses",
            metadata=metadata or {},
            scrub_strings=scrub_strings,
        )

    @staticmethod
    def _log_transform_error(transaction_logger: Optional[Any], pass_name: str, error: BaseException, payload: Any) -> None:
        if transaction_logger:
            transaction_logger.log_transform_error(pass_name, error, payload=payload, stage="adapter", protocol="responses")

    def _trace_responses_usage(
        self,
        transaction_logger: Optional[Any],
        response_payload: dict[str, Any],
        model: str,
        *,
        source: str,
    ) -> None:
        """Trace normalized Responses usage without changing stored payloads."""

        if not transaction_logger:
            return
        # Event-shaped payloads nest usage under "response".
        source_payload = response_payload.get("response") if isinstance(response_payload.get("response"), dict) else response_payload
        usage = source_payload.get("usage") if isinstance(source_payload, dict) else None
        if not usage:
            return
        record = extract_usage_record(usage, provider="responses", model=model, source=source)
        cost_breakdown = CostCalculator().calculate(record, model=model, provider="responses")
        self._trace(
            transaction_logger,
            "usage_accounting_summary",
            {"usage": record.to_dict(), "cost": cost_breakdown.to_dict()},
            direction="metadata",
            stage="final",
            metadata={"source": source, "pricing_source": cost_breakdown.pricing_source},
        )

    async def _safe_store(
        self,
        raw_request: dict[str, Any],
        response_payload: dict[str, Any],
        parent: Optional[StoredResponse],
        session_info: Optional[dict[str, Any]],
        transaction_logger: Optional[Any],
        stage: str,
    ) -> bool:
        """Persist one response without ever failing the delivered answer.

        Row construction lives INSIDE the guard: an id-less or non-conformant
        provider payload is a store problem, not a client problem. A storage
        failure is recorded as a transform error plus a store-specific trace
        record; it never propagates. Returns ``True`` only when the write
        actually landed, so callers can still report the observed outcome.
        """

        try:
            stored = self._stored_response(raw_request, response_payload, parent, session_info=session_info)
        except Exception as exc:
            self._log_transform_error(transaction_logger, f"{stage}_build", exc, {"model": raw_request.get("model")})
            return False
        try:
            await self.store.save(stored)
        except Exception as exc:
            try:
                diagnostic = stored.to_dict()
            except Exception:
                diagnostic = {"response_id": getattr(stored, "id", "?")}
            self._log_transform_error(transaction_logger, stage, exc, diagnostic)
            self._trace(
                transaction_logger,
                f"{stage}_error",
                {"response_id": getattr(stored, "id", "?"), "error": str(exc)},
                direction="metadata",
                stage="final",
            )
            return False
        return True

    async def _store_stream_response(
        self,
        raw_request: dict[str, Any],
        response_payload: dict[str, Any],
        parent: Optional[StoredResponse],
        *,
        failed: bool = False,
        transaction_logger: Optional[Any] = None,
        session_info: Optional[dict[str, Any]] = None,
        local_cache: Optional[MutableMapping[str, StoredResponse]] = None,
    ) -> bool:
        build_and_cache = local_cache is not None
        # Failed turns honor the store_failed policy FIRST — on the global
        # store AND the connection-local cache, including store=false (ZDR)
        # turns: caching a failed id anywhere would let a chained turn
        # "succeed" against empty lineage instead of missing.
        if failed and not self.store_settings.store_failed:
            return False
        # Local-cache copies build under the same never-fail contract as the
        # durable save: an id-less payload degrades to no cache entry, never
        # a failed turn.
        cached_row: Optional[StoredResponse] = None
        if build_and_cache:
            try:
                cached_row = self._stored_response(raw_request, response_payload, parent, session_info=session_info)
            except Exception as exc:
                self._log_transform_error(transaction_logger, "responses_store_stream_cache_build", exc, {})
        if not raw_request.get("store", True):
            if build_and_cache and cached_row is not None:
                # WebSocket mode: store=false turns still land in the
                # connection-local in-memory cache (never the global store,
                # never disk) so the next turn's previous_response_id
                # resolves for ZDR-style chains.
                local_cache[cached_row.id] = cached_row
            return False
        saved = await self._safe_store(raw_request, response_payload, parent, session_info, transaction_logger, "responses_store_stream_response")
        if build_and_cache and cached_row is not None:
            local_cache[cached_row.id] = cached_row
        return saved

    async def _store_stream_current_state(
        self,
        raw_request: dict[str, Any],
        response_payload: dict[str, Any],
        parent: Optional[StoredResponse],
        *,
        transaction_logger: Optional[Any],
        session_info: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Optionally persist in-progress stream state for retrieval surfaces."""

        if not self.store_settings.store_in_progress or not raw_request.get("store", True):
            return False
        if not await self._safe_store(raw_request, response_payload, parent, session_info, transaction_logger, "responses_store_stream_current_state"):
            return False
        self._trace(
            transaction_logger,
            "responses_stored_stream_current_state",
            {"response_id": response_payload.get("id"), "status": response_payload.get("status")},
            direction="metadata",
            stage="final",
        )
        return True


def _input_items(raw_request: dict[str, Any]) -> list[Any]:
    value = raw_request.get("input")
    if value is None:
        return []
    return deepcopy(value if isinstance(value, list) else [value])


def _input_item_id(item: Any) -> Optional[str]:
    """Return an input item's id when the item carries one, else ``None``."""

    if isinstance(item, dict) and isinstance(item.get("id"), str):
        return item["id"]
    return None


def _input_items_page(
    items: list[Any],
    *,
    limit: int = 20,
    after: Optional[str] = None,
    order: str = "desc",
) -> dict[str, Any]:
    """Build the official paginated input-items list envelope.

    ``limit`` defaults to 20 and caps at 100; ``after`` is an item id cursor
    (the page starts after that id); ``order`` defaults to ``desc`` — the
    official default — with ``asc`` accepted. An unknown ``after`` cursor
    yields an empty page (never a silent reset to the first page).
    ``first_id``/``last_id`` are the ids of the returned page edges (null
    when the page is empty or the items carry no ids), and ``has_more`` is
    true when another page follows.
    """

    try:
        size = int(limit)
    except (TypeError, ValueError):
        size = 20
    size = max(1, min(size, 100))
    ordered = list(reversed(items)) if str(order).lower() == "desc" else list(items)
    start = 0
    if after:
        start = None
        for index, item in enumerate(ordered):
            if _input_item_id(item) == after:
                start = index + 1
                break
        if start is None:
            ordered = []
            start = 0
    page = ordered[start : start + size]
    return {
        "object": "list",
        "data": deepcopy(page),
        "first_id": _input_item_id(page[0]) if page else None,
        "last_id": _input_item_id(page[-1]) if page else None,
        "has_more": start + size < len(ordered),
    }


def _reject_unsupported_lifecycles(raw_request: dict[str, Any]) -> None:
    """Clear local rejections for spec conflicts this proxy cannot honor."""

    if raw_request.get("previous_response_id") and raw_request.get("conversation"):
        # Spec: previous_response_id and conversation are mutually exclusive.
        raise ResponsesServiceError(
            "previous_response_id cannot be used in conjunction with conversation",
            status_code=400,
        )
    if raw_request.get("background"):
        # Background mode needs the queued/polling lifecycle; executed
        # synchronously it silently breaks the contract — reject explicitly
        # until implemented.
        raise ResponsesServiceError(
            "background mode is not supported by this proxy (no queued/polling lifecycle); omit 'background'",
            status_code=400,
        )


def _provider_continuation_target(raw_request: dict[str, Any]) -> Optional[str]:
    """Resolve the routing target's provider when it speaks Responses, else None.

    True/eligible when the request's routing target resolves to a provider
    whose protocol family is Responses — the provider retains its own chains
    (previous_response_id passthrough, encrypted reasoning, cache keys).
    Conservative by construction: unresolvable routing, unknown aliases, or
    non-Responses families return ``None`` (local replay / honest 404). The
    provider name is returned so continuation can verify a provider-owned row
    was minted by the same provider now targeted.
    """

    model = str(raw_request.get("model") or "")
    if not model:
        return None
    try:
        from ..providers import PROVIDER_PLUGINS
        from ..routing.config import load_routing_config_from_env
        from ..routing.profiles import parse_model_reference, resolve_profile

        target = model
        config = load_routing_config_from_env()
        routes = getattr(config, "model_routes", None) or {}
        target = str(routes.get(model.lower(), target))
        groups = getattr(config, "groups", None) or {}
        group = groups.get(target.removeprefix("group:"))
        if group is not None and getattr(group, "targets", None):
            target = str(group.targets[0])

        reference = parse_model_reference(target)
        plugin_class = PROVIDER_PLUGINS.get(reference.provider)
        if plugin_class is None:
            return None
        # PROVIDER_PLUGINS stores CLASSES (SingletonABCMeta) — unbound
        # method calls bind the model string as self and explode; use the
        # executor convention and talk to the singleton instance.
        plugin = plugin_class()
        # Resolve the profile the WAY ROUTING WOULD for a responses client:
        # the default profile's protocol is NOT the answer when a sibling
        # profile family-matches (bare-name false negative, G11 verify P1).
        declared = getattr(plugin, "transport_profiles", None)
        protocol_name: Optional[str] = None
        if reference.profile:
            protocol_name = plugin.get_protocol_name(reference.model or "", profile=reference.profile)
        elif declared:
            try:
                chosen = resolve_profile(
                    declared_profiles=declared,
                    default_profile=getattr(plugin, "default_profile", None),
                    protocol_name=getattr(plugin, "protocol_name", None),
                    client_protocol="responses",
                    requested_profile=None,
                    provider=reference.provider,
                )
                protocol_name = plugin.get_protocol_name(reference.model or "", profile=chosen)
            except Exception:
                protocol_name = None
        if not protocol_name:
            protocol_name = plugin.get_protocol_name(reference.model or "", profile=None)
        if not protocol_name:
            return None
        from ..protocols.registry import get_protocol_class

        cls = get_protocol_class(protocol_name)
        family = getattr(cls, "base_family", "") or protocol_name
        return reference.provider if family == "responses" else None
    except Exception:
        return None


def _provider_continuation_eligible(raw_request: dict[str, Any]) -> bool:
    """Whether public provider-side continuation is viable (G11 hybrid)."""

    return _provider_continuation_target(raw_request) is not None


def _expanded_responses_request(
    raw_request: dict[str, Any],
    lineage: list[StoredResponse],
) -> dict[str, Any]:
    """Expand proxy-owned continuation history into native Responses input."""

    expanded = {
        key: deepcopy(value)
        for key, value in raw_request.items()
        if key not in PROXY_ROUTING_KEYS
    }
    input_items: list[Any] = []
    for stored in lineage:
        input_items.extend(deepcopy(stored.input_items))
        input_items.extend(deepcopy(stored.output_items))
    input_items.extend(_input_items(raw_request))
    expanded["input"] = input_items
    if lineage:
        # Local lineage replayed inline: the upstream continuation pointer
        # must not ALSO reference the provider's own chain (double context).
        expanded.pop("previous_response_id", None)
    # Empty lineage: the provider's own continuation is the ONLY chain —
    # preserve previous_response_id so server-side state, caching, and the
    # encrypted-reasoning fast path stay on the provider-native path.
    return expanded


def _routing_kwargs(raw_request: dict[str, Any]) -> dict[str, Any]:
    """Carry proxy routing controls to RequestContextBuilder without tracing them."""

    return {
        key: deepcopy(raw_request[key])
        for key in PROXY_ROUTING_KEYS
        if key in raw_request
    }


def _request_isolation_key(raw_request: dict[str, Any]) -> str:
    return derive_session_isolation_key(
        raw_request.get("classifier"),
        raw_request.get("api_keys"),
        raw_request.get("providers"),
        bool(raw_request.get("private", False)),
    )


def _safe_stored_request(raw_request: dict[str, Any]) -> dict[str, Any]:
    """Persist only standard continuation fields, never routing credentials."""

    return {
        key: deepcopy(raw_request[key])
        for key in _STORED_REQUEST_FIELDS
        if key in raw_request
    }


def _redact_sensitive_fields(value: Any) -> Any:
    """Remove routing/auth containers recursively from diagnostic payloads."""

    if isinstance(value, dict):
        return {
            key: _redact_sensitive_fields(item)
            for key, item in value.items()
            if str(key).lower()
            not in {
                "api_key",
                "api_keys",
                "authorization",
                "credential_secrets",
                "provider_config",
                "providers",
                "x-api-key",
            }
        }
    if isinstance(value, list):
        return [_redact_sensitive_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_sensitive_fields(item) for item in value)
    return deepcopy(value)


def _expires_at(settings: ResponsesStoreSettings) -> Optional[float]:
    """Return the expiration timestamp for a new stored response, if enabled."""

    ttl = settings.ttl_seconds
    if ttl is None or ttl <= 0:
        return None
    return time.time() + ttl


def _internal_client_kwargs(client: Any, session_hints: Any, session_info: dict[str, Any]) -> dict[str, Any]:
    """Return hidden kwargs only for the internal RotatingClient path."""

    if not _supports_internal_context_kwargs(client):
        return {}
    kwargs: dict[str, Any] = {"_request_context_callback": _capture_request_context(session_info)}
    if session_hints:
        kwargs["_session_tracking_hints"] = session_hints
    return kwargs


def _supports_internal_context_kwargs(client: Any) -> bool:
    """Return whether a client is the proxy's internal rotating client."""

    return hasattr(client, "_request_builder") and hasattr(client, "_executor")


def _capture_request_context(session_info: dict[str, Any]):
    """Build a callback that records non-secret request context metadata."""

    def capture(context: Any) -> None:
        session_info["session_id"] = getattr(context, "session_id", None)
        session_info["scope_key"] = getattr(context, "session_isolation_key", None)
        session_info["classifier"] = getattr(context, "classifier", None)
        session_info["session_tracker"] = getattr(context, "session_tracker", None)
        session_info["provider"] = getattr(context, "provider", None)
        session_info["model"] = getattr(context, "model", None)
        session_info["tracking_namespace"] = getattr(context, "session_tracking_namespace", None)

    return capture


def _record_responses_session_anchor(session_info: dict[str, Any], response_payload: dict[str, Any]) -> None:
    """Record emitted Responses IDs as response-derived session evidence."""

    tracker = session_info.get("session_tracker")
    session_id = session_info.get("session_id")
    if not tracker or not session_id or not response_payload.get("id"):
        return
    tracker.record_response(
        session_id,
        provider=session_info.get("provider"),
        model=session_info.get("model"),
        scope_key=session_info.get("scope_key"),
        tracking_namespace=session_info.get("tracking_namespace"),
        response={"id": response_payload.get("id"), "object": "response"},
    )


def _stream_error_message(chunk: dict[str, Any]) -> str:
    """Return a compact, client-safe message for upstream stream error chunks."""

    error = chunk.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("type")
        if message:
            return str(message)
    message = chunk.get("message")
    return str(message) if message else "Upstream stream error"


# Official Responses ResponseError.code vocabulary for synthesized failures
# (docs §2.2). Codes are STRINGS; the numeric HTTP status lives on the wire
# status, never in the error object.
_RESPONSES_FAILURE_CODES = {
    "api_connection": "server_error",
    "upstream_error": "server_error",
    "server_error": "server_error",
    "proxy_timeout": "server_error",
    "rate_limit": "rate_limit_exceeded",
    "rate_limit_error": "rate_limit_exceeded",
    "quota_exceeded": "rate_limit_exceeded",
    "invalid_request": "invalid_prompt",
    "invalid_request_error": "invalid_prompt",
    "context_window_exceeded": "invalid_prompt",
    "request_too_large": "invalid_prompt",
    "authentication": "invalid_api_key",
    "authentication_error": "invalid_api_key",
    "forbidden": "invalid_request_error",
    "permission_error": "invalid_request_error",
    "not_found": "previous_response_not_found",
    "not_found_error": "previous_response_not_found",
}


def _nested_response_error_code(exc_type: str) -> str:
    """Map an error type onto the official ``ResponseError.code`` enum.

    The enum admits ``server_error``/``rate_limit_exceeded``/
    ``invalid_prompt`` (plus image/vector-store family codes) — NOT the
    top-level API-error vocabulary (``invalid_api_key``,
    ``previous_response_not_found``, ``invalid_request_error``). A
    synthesized failed ``Response`` must carry an in-enum code or strict
    SDK validators reject the object.
    """

    code = _RESPONSES_FAILURE_CODES.get(str(exc_type), "server_error")
    return code if code in {"server_error", "rate_limit_exceeded", "invalid_prompt"} else "server_error"


def _stream_failure_error(exc: Exception) -> dict[str, Any]:
    """Return a client-safe Responses stream failure object.

    ``code`` uses the official ResponseError string vocabulary; a numeric-only
    code never reaches the Responses surface.
    """

    error_type = getattr(exc, "error_type", None) or exc.__class__.__name__
    result = {
        # In-enum only — the nested Response.error position admits the
        # ResponseError vocabulary, never the top-level API-error codes.
        "code": _nested_response_error_code(str(error_type)),
        "message": str(exc),
        "type": str(error_type),
    }
    if isinstance(exc, ResponsesServiceError) and error_type == "api_connection":
        text = str(exc).lower()
        if "ttfb" in text:
            result["timeout_type"] = "ttfb"
        elif "stall" in text:
            result["timeout_type"] = "stall"
    return result
