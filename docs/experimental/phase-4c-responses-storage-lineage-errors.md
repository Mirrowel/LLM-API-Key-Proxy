# Phase 4c: Responses Storage, Continuation Lineage, And Error Shape

## Goal

Close the Phase 4/4b third-pass Responses findings while keeping the existing bridge architecture and no-SQLite constraint.

## Scope

- Wire configurable Responses storage at app startup.
- Keep process-local memory storage as the default.
- Add provider-cache-backed JSON storage as an opt-in durable backend.
- Preserve existing `ResponsesStoreSettings` policy for TTL, maximum items, failed storage, and in-progress storage.
- Extend `previous_response_id` continuation context to replay parent input and output lineage, oldest-to-newest.
- Return top-level OpenAI-compatible `error` bodies from Responses routes.

## Non-Goals

- Do not introduce SQLite or a new database.
- Do not replace the chat-completions bridge with native Responses execution.
- Do not add a WebSocket route.
- Do not implement a cancel endpoint.
- Do not perfect every rich Responses item type in the bridge; cover known text/message/tool-call cases and preserve unsupported fields.

## Implementation Plan

1. Configurable Responses store backend.
   - Add runtime config helpers for `memory` and `provider_cache` backends.
   - Use existing provider-cache JSON storage for durable mode.
   - Keep memory backend as the default.

2. Startup wiring.
   - Construct `ResponsesService(store=..., store_settings=...)` at proxy startup and fallback service creation.
   - Keep direct `ResponsesService()` defaults unchanged for tests and embedding users.

3. Continuation lineage.
   - Load parent chains through stored `request.previous_response_id` with depth and cycle guards.
   - Pass oldest-to-newest lineage into the bridge.

4. Bridge replay.
   - Replay parent input items as user messages when convertible.
   - Replay parent output message items as assistant messages.
   - Keep existing parent-output behavior for external callers using the old argument.

5. Top-level route errors.
   - Return `JSONResponse(status_code=..., content={"error": ...})` for Responses service errors.
   - Apply to create validation, retrieve, delete, and input-items routes.

6. Tests.
   - Config/env tests for storage backend selection.
   - Store tests for provider-cache-backed persistence.
   - Bridge/service tests for parent input + output lineage replay.
   - Route tests for top-level error bodies.

## Acceptance Criteria

- App startup can use durable provider-cache-backed Responses storage via config/env.
- Existing memory storage remains default.
- Continuations replay parent input and output lineage, not just parent output.
- Responses route errors use top-level OpenAI-compatible `error` bodies.
- Focused tests pass and both reviewers report no blockers/highs/mediums.
