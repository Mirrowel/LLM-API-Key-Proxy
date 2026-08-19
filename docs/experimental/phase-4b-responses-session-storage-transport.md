# Phase 4b Plan: Responses API Session, Storage, And Transport Corrections

## Goal

Correct the Phase 4 audit findings without over-claiming full native Responses provider execution. Phase 4 made `/v1/responses` usable through the chat bridge, but the validation pass found that important Responses semantics are still incomplete: `previous_response_id` is only used to replay parent output, not as session-routing evidence; default storage has only incidental expiry support with no runtime TTL policy or max-size pruning; stream generation is SSE-first inside the service instead of cleanly transport-neutral; and route/service tracing/storage should more clearly represent stored/skipped/current-state behavior.

## Non-Goals

- Do not implement a WebSocket FastAPI route in this phase unless the transport abstraction makes it trivial and tests stay small.
- Do not replace the bridge with fully native provider Responses execution here.
- Do not introduce SQLite or a database.
- Do not build multi-user security/admin features.
- Do not change current public SSE event order except where tests show it is wrong.
- Do not commit user-facing reports.
- Do not touch unrelated dirty `ARCHITECTURE.md`, `STRUCTURE.md`, `.opencode/`, `docs/issues/`, or old phase reports.

## Current Code State

- `ResponsesService.create_response()` parses `previous_response_id`, loads the parent from the store, and passes parent output to `ResponsesBridge.to_chat_kwargs()`.
- `ResponsesBridge.to_chat_kwargs()` stores `previous_response_id` only in private `_responses_bridge` metadata used for trace context, then `ResponsesService` pops that metadata before calling `client.acompletion()`.
- The actual `RotatingClient` request context cannot see `previous_response_id`, so `SessionTracker` cannot use it as a strong anchor for credential affinity.
- `StoredResponse` has `expires_at`, and stores check expiry on `get()`, but `ResponsesService` never sets a TTL policy and `InMemoryResponsesStore` has no max-size pruning.
- `ProviderCacheResponsesStore` exists but app startup always creates `ResponsesService()` with default in-memory store.
- `ResponsesService.stream_response()` directly instantiates `ResponsesSSEFormatter()` and yields formatted strings.
- `ResponsesWebSocketFormatter` exists as a placeholder with `NotImplementedError`, which is honest but does not prove that service logic is formatter-neutral.
- Responses stream storage stores only the completed payload after stream success and can skip storage when `store=false`. It does not store in-progress current state.
- `response.failed` events are yielded on exceptions, but failed responses are not stored as failed state even when `store=true`.

## Implementation Plan

1. Add an internal Responses session-hints carrier.
   - Introduce private `_session_tracking_hints` request kwargs consumed by `RequestContextBuilder` and removed before provider execution.
   - Merge service-level hints with provider hints before `SessionTracker.infer_session()`.
   - Preserve provider hints and explain that these hints are proxy-internal evidence, never provider payload.

2. Make `previous_response_id` a strong Responses session anchor.
   - Attach `_session_tracking_hints` for continuation requests with a strong anchor like `responses_previous_response_id:{id}` and deterministic affinity key.
   - Do not fake a strong first-turn anchor from a generated response ID before the request executes.
   - Add tests proving continuation requests expose hidden hints to context construction and the hidden field is not sent to providers.

3. Record generated Responses IDs as response-derived metadata.
   - Store response IDs and parent IDs clearly in `StoredResponse.metadata`.
   - Add a helper for Responses session hints so future native Responses execution can share it.

4. Add runtime storage policy.
   - Add `ResponsesStoreSettings` with `ttl_seconds`, `max_items`, `store_failed`, and `store_in_progress`.
   - Preserve current defaults: no expiry, no max pruning, completed responses stored, in-progress updates disabled.

5. Implement TTL assignment and max-size pruning.
   - Set `StoredResponse.expires_at` in `_stored_response()` from settings.
   - Add max-item pruning to `InMemoryResponsesStore.save()`.
   - Keep provider-cache expiry via `StoredResponse.expires_at`; document max-size limitations if listing is unavailable.

6. Store failed stream responses when `store=true`.
   - Persist `response.failed` payloads when storage policy says failed responses are stored.
   - Keep `store=false` skip behavior.

7. Add a current-state storage seam.
   - Add `store_in_progress` default false.
   - When enabled, save created/intermediate/completed stream state using the same store API.
   - Keep default behavior unchanged.

8. Refactor stream generation to transport-neutral events.
   - Add `ResponsesStreamEvent` and a formatter interface in `responses/streaming.py`.
   - Add `ResponsesService.stream_events()` yielding event objects.
   - Make `stream_response()` a thin SSE wrapper over `stream_events()`.
   - Keep `ResponsesWebSocketFormatter` honest: either pure JSON event formatting or explicit route-level future support, but service logic must not be SSE-specific.

9. Tighten trace boundaries around the event pipeline.
   - Preserve existing `responses_stream_event_*` pass names.
   - Avoid trace-only conversions when no transaction logger is present.
   - Add trace pass for current-state storage updates if enabled.

10. Update FastAPI route wiring minimally.
    - Keep current `/v1/responses` route behavior.
    - Continue using `service.stream_response()` for SSE.
    - Defer provider-cache/durable config wiring to config work unless a safe existing setting exists.

## Tests

- Responses service tests for hidden session hints, TTL, max pruning, and metadata.
- Responses streaming tests for `stream_events()` order, SSE wrapper compatibility, failed storage, `store=false`, and WebSocket formatter seam.
- Responses store tests for memory max pruning and provider-cache expiry behavior.
- Request-builder/session tracking tests for `_session_tracking_hints` consumption and provider payload cleanup.
- Route regression tests for current HTTP behavior.

## Acceptance Criteria

- `previous_response_id` becomes strong session-routing evidence for continuation requests without leaking internal hints to provider payloads.
- Responses storage has explicit TTL/max-size policy and failed-stream storage behavior.
- Streaming service logic can emit transport-neutral event objects, with SSE as a formatter wrapper and WebSocket support no longer requiring service/protocol rewrite.
- Existing `/v1/responses`, retrieve, delete, input_items, and SSE behavior remain compatible.
- Trace-disabled paths avoid unnecessary trace-only conversions.
- Focused tests and dual-agent review pass.
