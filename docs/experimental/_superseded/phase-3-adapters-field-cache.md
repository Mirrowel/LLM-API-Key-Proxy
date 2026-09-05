# Phase 3 Plan: Adapter And Field-Cache System

## Goal

Add the configurable adapter and field-cache foundation that lets protocol/provider implementations preserve and re-inject provider-specific state without hardcoding every provider. This phase should make it possible to define rules for values like reasoning content, thought signatures, prompt cache keys, provider session IDs, and response IDs, with strict provider/model/credential/session/classifier scoping and transform trace visibility.

## Non-Goals

- Do not migrate all existing providers to native protocols yet.
- Do not add `/v1/responses` routes yet.
- Do not replace `ProviderTransforms` yet.
- Do not implement JSON config loading for these rules yet; Phase 10 owns config polish.
- Do not instantiate SQLite or any DB.
- Do not implement provider-specific Claude/Codex/Copilot/Antigravity behavior yet.
- Do not make field cache a replacement for `SessionTracker`.

## Current Code Context

- `ProviderTransforms` is still the active runtime transform path and should remain untouched unless a small hook is needed.
- Phase 1 provides unified request/response/stream dataclasses and protocol adapters.
- Phase 2 provides `log_transform_pass()` and `log_transform_error()` for request/response/stream trace states.
- `ProviderCache` already exists and supports async `store_async()` / `retrieve_async()` with memory+disk TTL behavior, but it stores strings and starts async background tasks on construction.
- `RequestContext` has session, scope, classifier, provider, model, and credential metadata needed for cache isolation.
- Existing runtime behavior should stay stable while the new adapter/cache foundation is built and tested in isolation.

## Files To Add

- `src/rotator_library/adapters/__init__.py`
- `src/rotator_library/adapters/base.py`
- `src/rotator_library/adapters/registry.py`
- `src/rotator_library/adapters/builtin.py`
- `src/rotator_library/field_cache/__init__.py`
- `src/rotator_library/field_cache/types.py`
- `src/rotator_library/field_cache/paths.py`
- `src/rotator_library/field_cache/store.py`
- `src/rotator_library/field_cache/engine.py`
- `tests/test_adapter_registry.py`
- `tests/test_field_cache_paths.py`
- `tests/test_field_cache_engine.py`
- `tests/test_field_cache_trace.py`

## Files Possibly Touched

- `src/rotator_library/providers/provider_interface.py` to add optional provider declarations for adapter names and field-cache rules.
- `src/rotator_library/client/transforms.py` only if adding a no-op future hook is cleaner.
- `src/rotator_library/__init__.py` only if public lazy exports are useful.
- Avoid `RequestExecutor` runtime wiring in the first adapter/cache commits unless tests prove the hook is harmless and necessary.

## Adapter System

- Create a protocol-neutral `Adapter` base class.
- Adapters operate on raw dicts, unified dataclasses, or stream events depending on `supported_stages`.
- Adapter methods are override-friendly:
  - `transform_request(payload, context) -> payload`
  - `transform_response(payload, context) -> payload`
  - `transform_stream_event(payload, context) -> payload`
- Adapter context should include provider, model, protocol, credential_id, session_id, scope_key, classifier, transport, metadata, and optional `transaction_logger`.
- Registry should support:
  - explicit registration
  - alias resolution
  - auto-discovery from `src/rotator_library/adapters/`
  - duplicate/collision checks similar to protocol registry
  - order-preserving adapter chain resolution

Built-in adapters:

- `reasoning_rewrite`: copy/rename reasoning fields between configured paths.
- `reasoning_content`: normalize common reasoning content field names.
- `suppress_developer_role`: convert or remove developer-role messages for providers that reject it.
- `model_override`: replace model in outbound payload.
- `field_rename`: generic configured source-path to target-path copy/move.

These adapters are bases, not gospel. Providers can subclass/copy/mutate adapters or override provider methods.

## Field-Cache Rules

`FieldCacheRule` fields:

- `name`
- `source`: `request`, `response`, `stream_event`, `unified_request`, `unified_response`, `unified_stream_event`
- `path`: JSON-path-like selector
- `mode`: `last`, `all`, `last_user_turn`, `last_assistant_turn`, `per_tool_call`
- `scope`: list/tuple of scope dimensions
- `inject`: optional `FieldCacheInjection`
- `enabled`
- `ttl_seconds` optional, for future store implementations
- `metadata` for provider/model notes

`FieldCacheInjection` fields:

- `target`: `request`, `unified_request`, `metadata`, or provider-specific future target
- `path`
- `when_missing_only`
- `insert`
- `as_list`

Scope dimensions:

- `provider`
- `model`
- `credential`
- `session`
- `conversation`
- `classifier`

Default scope for conversation-affecting rules:

- provider + model + classifier + session.

Cache keys must include rule name and selected scope values. Missing optional values should use stable placeholders only where safe; for session-scoped rules with no session, default to no-op unless a rule explicitly allows fallback.

## Path Engine

Support dotted paths:

- `choices.0.message.reasoning_content`
- `choices.*.message.reasoning_content`
- `messages[-1].reasoning_content`
- `candidates.*.content.parts.*.thoughtSignature`

Behavior:

- Supports dict keys, list indices, `*`, and `[-1]`.
- Extraction returns all matches in stable traversal order.
- Injection creates missing dict containers when unambiguous.
- Injection does not create containers through wildcard paths.
- Malformed paths raise `FieldCachePathError` with useful messages.
- Missing paths are no-op, not errors.

## Store Plan

- `FieldCacheStore` protocol/interface with async `get`, `set`, `append`, and `clear`.
- `InMemoryFieldCacheStore` for tests and simple runtime.
- `ProviderCacheFieldStore` wrapper around an injected `ProviderCache`, JSON serializing values.
- Do not instantiate `ProviderCache` in module import paths because it starts async tasks.
- No SQLite.

## Engine Plan

`FieldCacheEngine` responsibilities:

- hold rules and store
- validate rule names and paths
- `extract(source, payload, context, transaction_logger=None)`
- `inject(target, payload, context, transaction_logger=None)`

Extraction:

- select rules matching source
- extract values by path
- apply mode
- store under scoped key
- trace `after_field_cache_extraction` with rule name, match count, mode, scope key, and sanitized values summary

Injection:

- select rules with matching injection target
- retrieve scoped values
- apply mode result
- inject into a payload copy by path
- trace `after_field_cache_injection` with rule name, hit/miss, target path, and changed flag

Engine rules:

- Do not mutate caller payload unless explicitly requested; default returns a deep copy.
- Failures should call `log_transform_error()` with failed rule/pass and then raise validation errors in tests. Runtime integration later can choose fail-open or fail-closed per provider.

## Modes

- `last`: store latest extracted value.
- `all`: append extracted values preserving order.
- `last_user_turn`: use metadata/turn marker when available; for raw messages fallback to latest user-associated match if path includes messages.
- `last_assistant_turn`: same for assistant.
- `per_tool_call`: requires a tool-call ID path or extracted object with obvious `id` / `tool_call_id`; otherwise validation error.

Phase 3 should implement `last` and `all` fully, and provide validated structure for turn/tool modes with tests for validation and simple cases. If a mode needs runtime conversation indexing not yet available, document it as present but limited.

## Transform Trace Requirements

Every adapter invocation should be traceable:

- `before_adapter_chain`
- `after_adapter`
- `after_adapter_chain`

Every field-cache injection/extraction should be traceable:

- `before_field_cache_injection`
- `after_field_cache_injection`
- `before_field_cache_extraction`
- `after_field_cache_extraction`

Errors use Phase 2 `transform_log_error`.

Trace metadata should include:

- adapter name
- rule name
- source/target
- path
- mode
- scope dimensions
- cache key hash or readable scoped key with no secrets
- match count
- changed flag

Do not log huge cached payloads by default; log summaries and sanitized sample values.

## Provider Declaration Plan

Add optional provider methods/class attributes only if needed:

- `protocol_name`
- `adapter_names`
- `field_cache_rules`
- `get_adapter_config(model)`
- `get_field_cache_rules(model)`

Keep defaults empty/no-op to avoid provider changes. These declarations are for later provider migration.

## Tests

Adapter registry:

- auto-discovers built-ins
- aliases resolve
- duplicate names/collisions are deterministic
- chain order is preserved
- no-op adapter preserves payload

Built-in adapters:

- `model_override`
- `suppress_developer_role`
- `field_rename`
- reasoning field copy/rename

Path engine:

- dict path extraction
- list index extraction
- wildcard extraction
- `[-1]` extraction
- injection into simple dict path
- missing path no-op
- wildcard injection rejected
- malformed path error

Field-cache engine:

- extract response value and inject into next request
- `last` overwrites
- `all` appends
- scope isolation by provider/model/session/classifier/credential
- missing path no-op
- malformed rule validation
- stream_event extraction
- trace entries emitted for injection/extraction
- transform errors emitted for rule failures

Regression:

- Phase 1 protocol tests.
- Phase 2 logging tests.
- core session/selection tests.

## Commit Checkpoints

1. Add adapter base/registry and built-in no-op/simple adapters with tests.
2. Add field-cache rule dataclasses and path engine with tests.
3. Add field-cache stores and scoped key builder with tests.
4. Add field-cache engine extraction/injection with trace tests.
5. Add optional provider declaration methods if needed.
6. Run focused and regression tests.
7. Review with `explore` and `explore-heavy`, fix findings, and write the uncommitted Phase 3 report.

## Risks And Mitigations

- JSON path scope could grow too large. Keep Phase 3 path syntax intentionally small and explicit.
- Cache leaks across providers/models/sessions would be severe. Scope-key tests must be extensive.
- Storing huge model outputs can bloat caches. Engine should store only matched values and trace summaries.
- ProviderCache lifecycle can be awkward because it starts async tasks. Wrap injected instances; do not instantiate globally.
- Turn/tool modes can be under-specified. Validate and implement safe subsets rather than guessing.
- Runtime integration can destabilize requests. Keep Phase 3 mostly isolated until review confirms the foundation.
