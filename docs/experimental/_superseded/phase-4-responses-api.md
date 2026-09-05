# Phase 4 Plan: Responses API

## Goal

Add a first-class OpenAI-compatible Responses API surface with durable response storage, `previous_response_id` continuation support, HTTP SSE streaming, and an explicit WebSocket transport seam for future implementation. Phase 4 should use the Phase 1 Responses protocol, Phase 2 transform trace logging, and Phase 3 field-cache/adapter foundations without forcing all providers onto native protocols yet.

## Non-Goals

- Do not migrate every provider to native protocol execution in this phase.
- Do not remove or replace `/v1/chat/completions`.
- Do not replace `UsageManager`, `SessionTracker`, retry-after parsing, or current credential rotation.
- Do not implement full multi-user/security controls.
- Do not add SQLite or any database.
- Do not implement full provider-specific Responses APIs for Claude Code/Codex/Copilot yet.
- Do not implement WebSocket runtime handling yet; provide transport interfaces/seams and tests for HTTP SSE behavior.

## Current Code Context

- FastAPI routes live in `src/proxy_app/main.py`.
- `RotatingClient.acompletion()` is the stable execution entry point today.
- `ResponsesProtocol` already parses/formats Responses request/response/stream shapes and declares future `websocket` support.
- `TransactionLogger` can record new transform passes and errors.
- `FieldCacheEngine` and `ProviderCacheFieldStore` can preserve response IDs/output items but need targeted key deletion for Responses delete endpoints.
- Existing stream wrappers aggregate chat-completions SSE chunks, but Responses SSE needs its own event formatting instead of chat completion chunk output.
- Runtime execution should stay conservative: Phase 4 can bridge Responses requests to the existing completion path until native provider protocols are wired in later phases.

## Files To Add

- `src/rotator_library/responses/__init__.py`
- `src/rotator_library/responses/types.py`
- `src/rotator_library/responses/store.py`
- `src/rotator_library/responses/bridge.py`
- `src/rotator_library/responses/service.py`
- `src/rotator_library/responses/streaming.py`
- `tests/test_responses_store.py`
- `tests/test_responses_bridge.py`
- `tests/test_responses_service.py`
- `tests/test_responses_routes.py`
- `tests/test_responses_streaming.py`

## Files Likely To Touch

- `src/proxy_app/main.py`
- `src/rotator_library/client/rotating_client.py`
- `src/rotator_library/field_cache/store.py` if targeted delete is added to the store protocol.
- `src/rotator_library/__init__.py` only if lazy public exports are useful.
- Existing protocol tests only if the Responses adapter needs an additive event/output fix.

## Route/API Scope

`POST /v1/responses`:

- Accept raw Responses JSON.
- Support non-streaming response.
- Support `stream: true` with HTTP SSE.
- Validate required `model`.
- Accept `input`, `instructions`, `tools`, `tool_choice`, `reasoning`, `metadata`, `include`, `store`, `previous_response_id`, and common generation params.

`GET /v1/responses/{response_id}`:

- Return stored response object when available.
- Return 404-compatible error if missing or deleted.

`DELETE /v1/responses/{response_id}`:

- Delete stored response metadata/output.
- Return `{ "id": ..., "object": "response.deleted", "deleted": true }` or equivalent compatible shape.

`GET /v1/responses/{response_id}/input_items`:

- Return stored input items for clients that need continuation inspection.

Optional if simple:

- `POST /v1/responses/{response_id}/cancel` for active stream cancellation can be planned but not fully implemented unless active stream tracking lands cleanly.

## Response Storage

`StoredResponse` fields:

- `id`
- `created_at`
- `model`
- `status`
- `request`
- `response`
- `input_items`
- `output_items`
- `usage`
- `metadata`
- `session_id`
- `scope_key`
- `classifier`
- `expires_at` optional

`ResponsesStore` protocol:

- `save(stored_response)`
- `get(response_id)`
- `delete(response_id)`
- `list_input_items(response_id)`

Store implementations:

- `InMemoryResponsesStore` for tests and runtime default if no persistent cache is configured.
- `ProviderCacheResponsesStore` as an optional wrapper around injected `ProviderCache`.
- JSON serialization through `serialize_value()`.
- No global `ProviderCache` construction at import time.
- No SQLite.

Targeted deletion:

- Add `delete(key)` to `FieldCacheStore` only if reused directly.
- Prefer a dedicated Responses store so field cache remains small and focused.

## Response IDs

- Generate IDs like `resp_<safe_random>` when upstream response lacks an ID.
- Preserve upstream IDs when present.
- Store every response when request `store` is true or omitted if compatibility expects default storage.
- If `store: false`, return the response but do not persist it; `previous_response_id` cannot refer to it later.
- Include `previous_response_id` in stored metadata for lineage/debugging.

## `previous_response_id`

- On request with `previous_response_id`, load the stored parent response.
- If not found, return a 404/400-compatible invalid request error rather than silently ignoring.
- Build continuation context conservatively:
  - Keep original current request input.
  - Add parent response output items into protocol/service metadata for traceability.
  - For bridge execution through chat completions, convert parent response message items into prior assistant messages where safe.
  - Do not inject unknown output item types into chat messages unless `ResponsesProtocol` can represent them safely.
- Record transform trace pass:
  - `responses_previous_response_loaded`
  - Include parent ID, output count, input item count, and whether bridge context was expanded.
- This should line up with Phase 3 field-cache storage but does not need to fully rely on field-cache rules yet.

## Bridge Execution

`ResponsesBridge` responsibilities:

- Convert `UnifiedRequest` from `ResponsesProtocol.parse_request()` into chat-completions kwargs for current `RotatingClient.acompletion()`.
- Map `model` to `model`.
- Map `instructions`/system blocks to a system message.
- Map input messages to chat messages.
- Map tool definitions to OpenAI tool shape when possible.
- Map generation params to existing compatible kwargs.
- Map `stream` to `stream`.
- Preserve unsupported Responses fields in trace metadata and request metadata, not silently discard them.
- Rebuild Responses output from chat completion responses using `ResponsesProtocol.format_response()`.

This bridge is temporary compatibility, not the final native provider path. Docstrings/comments should explain that native provider execution will replace the bridge for covered providers in later phases.

## Service

`ResponsesService` responsibilities:

- Own `ResponsesProtocol`, `ResponsesBridge`, `ResponsesStore`, and optional transform logging.
- `create_response(raw_request, client, request=None)`
- `stream_response(raw_request, client, request=None)`
- `get_response(response_id)`
- `delete_response(response_id)`
- `list_input_items(response_id)`

Service transform passes:

- `raw_responses_request`
- `parsed_unified_request`
- `responses_previous_response_loaded`
- `responses_bridge_chat_request`
- `raw_responses_provider_response` or `raw_chat_bridge_response`
- `parsed_unified_response`
- `stored_responses_response`
- `final_responses_response`

Errors use `log_transform_error()`.

## Streaming

- HTTP SSE first.
- Convert chat-completions stream chunks into Responses SSE events.
- Required event names:
  - `response.created`
  - `response.output_item.added`
  - `response.output_text.delta`
  - `response.output_item.done`
  - `response.completed`
  - `response.failed` on errors
  - final `data: [DONE]` only if compatibility requires it; prefer Responses-style events and document the chosen behavior.
- Preserve raw stream chunk trace:
  - `raw_chat_bridge_stream_chunk`
  - `parsed_unified_stream_event`
  - `formatted_responses_stream_event`
  - `stored_responses_stream_response`
- Accumulate final output items so streamed responses are retrievable by `GET /v1/responses/{id}` after completion.
- Handle client disconnect without crashing the server.
- Do not add the broader stream stall/backpressure overhaul; that belongs to the later streaming phase.

## WebSocket Seam

- Add transport-neutral interfaces:
  - `ResponsesTransport` or small formatter abstraction.
  - `ResponsesSSEFormatter`.
  - `ResponsesWebSocketFormatter` placeholder or protocol with `NotImplementedError`.
- Keep protocol/service logic independent from `StreamingResponse`.
- Add tests that formatter/service API can select `sse` now and recognizes `websocket` as a future transport without implementing a WebSocket route.
- Do not expose a WebSocket endpoint yet unless it is a no-op documented placeholder; prefer no route over a misleading route.

## FastAPI Integration

- Add routes near existing OpenAI-compatible routes in `main.py`.
- Use existing `verify_api_key`.
- Use existing raw request logging where applicable.
- Use `JSONResponse` for normal responses.
- Use `StreamingResponse(..., media_type="text/event-stream")` for streams.
- Use OpenAI-compatible error response shape for validation/missing response IDs.
- Store service on app state during lifespan if it needs shared storage, or construct module-level in-memory store carefully if no async lifecycle is needed. Prefer `app.state.responses_service`.

## Tests

Store tests:

- save/get/delete response
- input items list
- missing response returns None
- JSON serialization with provider-cache wrapper if implemented
- no SQLite/import-time cache construction

Bridge tests:

- Responses input string/list converts to chat messages
- instructions become system message
- tool definitions preserve function call shape
- unsupported fields are preserved in metadata/trace
- chat completion response formats back to Responses output

Service tests:

- non-stream create stores response
- `store: false` does not persist
- `previous_response_id` loads parent
- missing previous response raises useful error
- transform trace passes emitted

Route tests:

- `POST /v1/responses` non-stream success with fake client
- `POST /v1/responses` missing model returns 400
- `GET /v1/responses/{id}` success and 404
- `DELETE /v1/responses/{id}` success
- `GET /v1/responses/{id}/input_items`

Streaming tests:

- stream emits Responses SSE event names in order
- deltas aggregate into stored final response
- stream errors emit `response.failed`
- client disconnect path is safe if testable

Regression tests:

- Phase 1 protocol tests
- Phase 2 transform logging tests
- Phase 3 adapter/cache tests
- `test_session_tracking.py`
- `test_selection_engine.py`

## Commit Checkpoints

1. Add Responses store and tests.
2. Add Responses bridge and tests.
3. Add Responses service with non-stream create/get/delete/input-items and tests.
4. Add HTTP routes and route tests.
5. Add HTTP SSE formatter/streaming conversion and tests.
6. Add WebSocket transport seam tests/docstrings without runtime route.
7. Run focused and regression tests.
8. Review with `explore` and `explore-heavy`, fix findings, and write the uncommitted Phase 4 report.

## Risks And Mitigations

- Responses API is broader than chat completions. Keep the bridge explicit about unsupported fields and preserve them in metadata/trace rather than pretending full native support.
- `previous_response_id` can leak context across users/scopes if storage keys are global. Include request/session/scope/classifier metadata and leave full multi-user isolation for later, but avoid credential secrets in storage keys.
- Streaming conversion can lose item fidelity. Accumulate output text and tool/reasoning items explicitly; preserve raw chunks in trace.
- In-memory store will not survive restarts. This is acceptable for Phase 4 default; provider-cache-backed persistence can be enabled where lifecycle is available.
- WebSocket support can be over-promised. Expose the abstraction seam only, not a functioning route, until the later transport phase.
