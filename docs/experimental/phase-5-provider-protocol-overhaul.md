# Phase 5 Plan: Native Provider Protocol Overhaul

## Goal

Make provider implementations opt into the native protocol, adapter, field-cache, and Responses foundations added in Phases 1-4, then add or restore the priority providers in the requested order: Claude Code, Codex, Copilot, Antigravity, and Gemini CLI parity review. This phase should create a reusable provider-native execution seam first, then implement providers incrementally behind explicit declarations so LiteLLM remains fallback-only for uncovered cases.

## Non-Goals

- Do not replace `UsageManager`, `SessionTracker`, `SelectionEngine`, or retry-after parsing.
- Do not implement routing/fallback groups yet; Phase 6 owns ordered fallback chains.
- Do not implement full multi-user/security isolation.
- Do not add SQLite or any database.
- Do not rewrite the entire executor in one pass.
- Do not migrate every existing provider at once.
- Do not copy obsolete retired-provider behavior blindly.
- Do not remove LiteLLM fallback; make fallback explicit and observable.

## Current Code Context

- Phase 1 provides reusable protocol adapters: OpenAI Chat, Anthropic Messages, Gemini, Responses, and LiteLLM fallback.
- Phase 2 provides transaction transform tracing and provider trace entries.
- Phase 3 provides provider declarations, adapter chains, field-cache rules, and scoped cache engine.
- Phase 4 provides `/v1/responses`, response storage, HTTP SSE, and WebSocket seam.
- `ProviderInterface` already has optional `get_protocol_name()`, `get_adapter_names()`, `get_adapter_config()`, and `get_field_cache_rules()`.
- Existing custom providers with `has_custom_logic()` include Gemini CLI, Deepseek, and retired Antigravity/IFlow/Qwen-style providers.
- Gemini CLI is large and custom; Phase 5 should review and target improvements rather than destabilize it.
- Antigravity exists only under `src/rotator_library/providers/_retired/` and must be compared against current behavior before restoration.
- No Claude Code, Codex, or Copilot provider files exist in active providers today.

## Files To Add

- `src/rotator_library/native_provider/__init__.py`
- `src/rotator_library/native_provider/context.py`
- `src/rotator_library/native_provider/executor.py`
- `src/rotator_library/native_provider/http.py`
- `src/rotator_library/native_provider/streaming.py`
- `src/rotator_library/providers/claude_code_provider.py`
- `src/rotator_library/providers/codex_provider.py`
- `src/rotator_library/providers/copilot_provider.py`
- Possibly `src/rotator_library/providers/antigravity_provider.py`
- `tests/test_native_provider_executor.py`
- `tests/test_native_provider_streaming.py`
- `tests/test_claude_code_provider.py`
- `tests/test_codex_provider.py`
- `tests/test_copilot_provider.py`
- `tests/test_antigravity_provider_restore.py` if restored in this phase
- `tests/test_gemini_cli_protocol_declarations.py`

## Files Likely To Touch

- `src/rotator_library/providers/provider_interface.py` only for small optional native execution methods if needed.
- `src/rotator_library/client/executor.py` only if a minimal seam is needed to route declared native providers without breaking current providers.
- `src/rotator_library/client/models.py` only if model/provider resolution needs explicit provider declarations.
- `src/rotator_library/client/request_builder.py` only if provider declarations need context fields before execution.
- `src/rotator_library/responses/service.py` only if native Responses-capable providers can bypass the chat bridge cleanly.
- Existing provider files only for declaration additions or targeted Gemini CLI fixes.

## Native Provider Execution Seam

- Add `NativeProviderContext` with provider, model, credential identifier, headers, request context, protocol name, adapter names, field-cache rules, transport, transaction logger, session fields, scope key, classifier, and provider metadata.
- Add `NativeProviderExecutor` that can:
  - Resolve protocol via `provider.get_protocol_name(model)`.
  - Build `AdapterContext` and run `before_adapter_chain` / `after_adapter` / `after_adapter_chain`.
  - Run `FieldCacheEngine.inject()` before provider request.
  - Build native HTTP request payload from the selected protocol.
  - Send request through a small HTTP transport wrapper.
  - Parse provider response through selected protocol.
  - Run `FieldCacheEngine.extract()` after response or stream events.
  - Format back to the requested client protocol.
  - Emit transform trace passes for every step.
- The executor must be provider-opt-in. If `get_protocol_name()` returns `None` or `"litellm_fallback"`, current behavior stays unchanged.
- It must be independently testable with mocked HTTP clients and fake providers.

## Provider Interface Additions

Prefer using existing declarations first. Add optional methods only if needed:

- `get_native_endpoint(model, operation)`
- `get_native_headers(credential_identifier, model, operation)`
- `build_native_request_options(model, operation)`
- `supports_native_operation(operation, model)`
- `should_use_native_protocol(operation, model)`

Defaults must preserve current behavior. Docstrings must explain provider overrides are encouraged when a base protocol is close but not exact.

## HTTP Transport

- Use injected `httpx.AsyncClient`.
- Support JSON POST first.
- Support SSE response iteration for streaming.
- Preserve raw request/response bodies for transform trace.
- Do not add WebSocket runtime transport yet; keep the seam compatible with Phase 4 WebSocket formatter.
- Convert provider HTTP errors into existing error classification paths where possible.

## Streaming

- Native stream parser should call protocol stream parsing where available.
- Emit transform passes:
  - `native_provider_request`
  - `raw_native_provider_stream_chunk`
  - `parsed_native_stream_event`
  - `after_field_cache_stream_extraction`
  - `formatted_client_stream_event`
- Do not fallback after visible output. Phase 6 owns fallback policy; Phase 5 only makes stream behavior observable and safe.

## Responses Integration

- For providers with native Responses support, allow `ResponsesService` to bypass `ResponsesBridge`.
- Add a service-level native executor hook only if it can be done without coupling service to provider internals.
- If not clean in Phase 5, leave `/v1/responses` bridge as default and expose provider-native Responses in provider tests/foundation for Phase 6/8 wiring.
- Preserve `/v1/responses` route behavior from Phase 4.

## Priority Provider Plan

### Claude Code

- Add provider first.
- Determine whether it is Anthropic Messages-compatible, OpenAI Chat-compatible, or a dedicated endpoint.
- Start with declarations and mocked native execution tests.
- Preserve Claude reasoning/thinking fields via field-cache rules if present.
- Suppress/transform unsupported roles through adapters instead of bespoke monolithic code.
- Add tests for auth headers, model list behavior, request transform, response transform, streaming text, and field-cache rule declarations.
- If live API details are uncertain, implement an integration path with explicit env names and mocked endpoint behavior rather than guessing secrets or undocumented flows.

### Codex

- Add provider second.
- Treat as likely OpenAI/Responses-compatible until provider-specific evidence says otherwise.
- Prefer native Responses protocol if supported; otherwise OpenAI Chat protocol.
- Add tests for protocol selection, Responses bypass or bridge compatibility, auth headers, model naming, and explicit no-LiteLLM path when native mode is declared.

### Copilot

- Add provider third.
- Use native protocol declarations and OAuth/header helpers.
- Keep credential acquisition/refresh minimal and mocked unless existing credential machinery already supports it.
- Add field-cache rules for conversation/session IDs if required by provider behavior.
- Add tests for model list, auth header, protocol selection, request conversion, and streaming.

### Antigravity

- Compare `src/rotator_library/providers/_retired/antigravity_provider.py`, auth base, quota tracker, and device profile utilities before restoring.
- Restore only valid/current behavior.
- Avoid fragile or obsolete device-profile behavior unless current service behavior requires it.
- Extract reusable pieces:
  - model mapping
  - auth headers
  - schema cleanup
  - thinking/reasoning preservation
  - quota parsing
  - SSE handling
- Do not resurrect the whole monolith unchanged.
- Add a restored provider only after tests describe the stable subset.
- If the current service cannot be validated safely, write an explicit integration-path provider skeleton and defer live-specific behavior.

### Gemini CLI Parity Review

- Review active `gemini_cli_provider.py` against the new protocol/adapter/cache foundation.
- Add provider declarations where safe:
  - likely protocol `gemini`
  - adapters for model override / field rename / role suppression only if needed
  - field-cache rules for thought signatures and provider session fields if they match current behavior.
- Do not rewrite Gemini CLI in this phase.
- Add targeted tests proving declarations do not change current behavior.
- Fix only clear parity gaps found during review.

## LiteLLM Fallback Policy

- Providers with native declarations should not silently fall back to LiteLLM on the primary path.
- If fallback is allowed, it must be explicit:
  - protocol name `"litellm_fallback"`
  - trace pass `native_provider_litellm_fallback`
  - metadata reason.
- Tests must assert no accidental LiteLLM fallback for providers declared native.

## Transform Trace Requirements

Provider-native calls should log:

- `native_protocol_selected`
- `before_adapter_chain`
- `after_adapter`
- `after_field_cache_injection`
- `native_provider_request`
- `raw_native_provider_response`
- `parsed_native_provider_response`
- `after_field_cache_extraction`
- `final_client_response`

Provider-native stream calls should log:

- `native_protocol_selected`
- `native_provider_stream_request`
- `raw_native_provider_stream_chunk`
- `parsed_native_stream_event`
- `after_field_cache_stream_extraction`
- `formatted_client_stream_event`

Errors should use `log_transform_error()` with provider/protocol/pass context.

## Field-Cache Requirements

- Use Phase 3 rules for:
  - reasoning content
  - thought signatures
  - provider session IDs
  - prompt cache keys
  - previous/response IDs where provider-specific
- Scope must include at least provider + model + classifier + session for conversation-affecting values.
- Credential scope must be added for values tied to an account/token.
- Tests must prove no cross-provider/model/session/credential leakage for provider rules.

## Testing Plan

Native executor tests:

- protocol selection
- adapter chain order
- field-cache injection/extraction
- non-stream request/response trace passes
- stream trace passes
- explicit fallback behavior
- provider opt-in leaves current fallback path untouched for undeclared providers

Provider tests:

- registration/discovery for Claude Code, Codex, Copilot, and restored Antigravity if added.
- env var naming and auth header construction via mocks.
- model list parsing via mocked HTTP.
- request payload build and response parse via mocked HTTP.
- SSE stream conversion via mocked chunks.
- field-cache declarations.
- no live credentials required.

Regression tests:

- Phase 1 protocol tests.
- Phase 2 logging tests.
- Phase 3 adapter/field-cache tests.
- Phase 4 Responses tests.
- `test_session_tracking.py`.
- `test_selection_engine.py`.

## Commit Checkpoints

1. Add native provider context/executor/HTTP transport with tests.
2. Add native streaming support with tests.
3. Add Claude Code provider or integration skeleton with tests.
4. Add Codex provider or integration skeleton with tests.
5. Add Copilot provider or integration skeleton with tests.
6. Compare and restore the safe Antigravity subset, or write a documented deferral with tests.
7. Add Gemini CLI declaration/parity fixes with tests.
8. Run focused and regression tests.
9. Review with `explore` and `explore-heavy`, fix findings, and write the uncommitted Phase 5 report.

## Risks And Mitigations

- Provider APIs may be undocumented or volatile. Use mocked behavior and explicit integration seams rather than guessing hidden flows.
- A native executor could destabilize current traffic. Keep native execution opt-in and leave undeclared providers unchanged.
- Restoring Antigravity could reintroduce brittle device/profile logic. Restore only tested stable subsets.
- Streaming fidelity for tool/reasoning deltas may vary by provider. Preserve raw chunks in trace and keep provider-specific parsers override-friendly.
- Field-cache leakage would be serious. Add scope tests per provider rule.
- LiteLLM fallback could hide native failures. Make fallback explicit, traced, and tested.
