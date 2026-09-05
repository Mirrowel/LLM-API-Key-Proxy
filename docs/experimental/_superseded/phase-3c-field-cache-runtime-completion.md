# Phase 3c: Field-Cache Runtime Completion And Trace Safety

## Goal

Close the Phase 3/3b third-pass findings around field-cache runtime coverage, adapter-chain trace safety, and credential-scoped isolation.

## Scope

- Execute all declared field-cache sources used by native runtime: `request`, `response`, `stream_event`, `unified_request`, `unified_response`, and `unified_stream_event`.
- Execute declared injection targets used by native runtime: `request`, `unified_request`, and `metadata`.
- Suppress generic adapter-chain payload traces in native execution where field-cache-aware redaction has not yet run, and rely on native executor traces that apply rule-aware redaction.
- Fail closed when a rule includes `credential` scope but the runtime lacks a credential identifier.
- Add focused tests for each corrected behavior.

## Non-Goals

- Do not replace the field-cache store or add a database.
- Do not make native streaming inject into stream events; streaming extraction remains expected behavior.
- Do not make Copilot declare field-cache rules when none are required.
- Do not implement custom provider-specific turn inference beyond existing metadata hints.

## Implementation Plan

1. Credential scope isolation.
   - Update cache-key construction so missing `credential` scope returns `None`.
   - Keep existing `allow_missing_session` behavior only for `session` scope.
   - Add tests proving credential-scoped extraction/injection skips instead of using a shared missing bucket.

2. Unified request injection and extraction.
   - In native non-streaming and streaming paths, create the field-cache engine before protocol build.
   - Extract `unified_request` after parsing client payload.
   - Inject `metadata` before building protocol context when metadata rules exist.
   - Inject `unified_request` before `protocol.build_request()`.
   - Hydrate injected serialized unified-request dictionaries back into the current `UnifiedRequest` for supported common fields.
   - Trace `after_unified_request_field_cache_injection` and `after_metadata_field_cache_injection`.

3. Unified response extraction.
   - Extract `unified_response` after protocol response parsing and before formatting.
   - Keep existing provider-response extraction after response adapter chain.
   - Trace `after_unified_response_field_cache_extraction`.

4. Unified stream-event extraction.
   - Extract `unified_stream_event` after native stream-event adapter chain and before formatting.
   - Keep existing provider stream-event extraction from formatted event payload.
   - Trace `after_unified_stream_event_field_cache_extraction`.

5. Native adapter trace safety.
   - Disable generic adapter-chain traces inside native request/response paths, matching the stream-event path.
   - Keep native executor traces after adapter chains, where configured field-cache paths are redacted.
   - Add tests with arbitrary configured provider-state paths proving generic `before_adapter_chain`/`after_adapter` traces are not emitted in native execution and native traces redact configured paths.

6. Tests.
   - Field-cache engine credential-scope fail-closed tests.
   - Native executor unified request source/target tests.
   - Native executor metadata injection tests.
   - Native executor unified response extraction tests.
   - Native stream unified event extraction tests.
   - Native adapter trace redaction/suppression tests.

## Acceptance Criteria

- Native runtime executes all declared field-cache sources/targets that it accepts, except stream-event injection which remains intentionally unsupported.
- Missing credential scope never shares a `_none` cache key.
- Native adapter chain traces cannot leak arbitrary configured provider-state fields before rule-aware redaction.
- Existing field-cache and native provider tests continue to pass.
- Both `explore` and `explore-heavy` reviewers report no blockers/highs/mediums.
