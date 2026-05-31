# Phase 3b: Field-Cache Runtime Semantics And Persistence

## Goal

Correct the Phase 3 audit gap. Phase 3 built the adapter and field-cache foundation, but the audit found that the system is still partly declarative: `last_user_turn`, `last_assistant_turn`, and `per_tool_call` are validated/declared but not semantically implemented enough; the native provider executor creates a fresh `InMemoryFieldCacheStore` by default so cached values do not persist across native requests; JSON-configured field-cache rules are parsed but not merged into runtime provider declarations.

## Non-Goals

- Do not introduce SQLite or any database.
- Do not replace `SessionTracker`, `UsageManager`, or provider retry logic.
- Do not wire non-native LiteLLM paths through field-cache rules in this phase.
- Do not make field cache a substitute for session affinity. It preserves provider protocol state only.
- Do not implement a full external admin UI/config editor.
- Do not commit user-facing reports.
- Do not touch unrelated dirty files (`ARCHITECTURE.md`, `STRUCTURE.md`, `.opencode/`, `docs/issues/`, old phase reports).

## Current Code State

- `FieldCacheRule.mode` allows `last`, `all`, `last_user_turn`, `last_assistant_turn`, and `per_tool_call`.
- `FieldCacheEngine._store_values()` currently treats `last_user_turn` and `last_assistant_turn` exactly like `last`.
- `per_tool_call` currently extracts tool IDs from each extracted value, which only works when the selected value itself contains the tool ID. It cannot correlate values with a sibling or parent tool-call ID from the original payload.
- `FieldCacheInjection` has `as_list`, `when_missing_only`, and `insert`, but engine injection does not use `insert` and does not support selecting a per-tool-call value at injection time.
- `NativeProviderExecutor.__init__(field_cache_store=None)` passes `None` into each `FieldCacheEngine`, causing each request to get a new `InMemoryFieldCacheStore`. That means default native field-cache persistence is per-engine/per-request, not process-level.
- `ProviderCacheFieldStore` exists and no SQLite is present.
- `ExperimentalConfig.parse_field_cache_rules()` can parse JSON rules, but `RequestExecutor._native_context()` currently only uses `plugin.get_field_cache_rules(model)`.
- `NativeProviderContext.field_cache_context()` includes provider/model/credential/session/conversation/classifier plus metadata; metadata is the right place to carry current request payload/turn/tool-call hints without expanding core scope dimensions.
- Phase 2b hardened traces so field-cache logs expose shapes and counts, not raw cached values.

## Implementation Plan

### 1. Process-Local Default Native Store

- Change `NativeProviderExecutor` so a missing `field_cache_store` creates one shared `InMemoryFieldCacheStore` in the executor instance, not a new store inside every `FieldCacheEngine`.
- Keep injected stores honored exactly as before.
- Add tests proving two calls through the same executor can extract on response and inject on a later request using the default store.
- Do not persist to disk by default; process-local is enough for runtime continuity and respects the no-SQLite rule.

### 2. Value Envelopes For Contextual Modes

- Add a small internal stored representation for contextual modes, likely dicts with fields such as `value`, `role`, `turn_index`, `tool_call_id`, and `source_path`.
- Keep stored values JSON-serializable for `ProviderCacheFieldStore`.
- Do not expose envelopes to injected payloads except where needed; injection should unwrap `value`.
- Preserve current external behavior for `last` and `all` as much as possible.

### 3. Turn-Aware Extraction

- Implement real `last_user_turn` and `last_assistant_turn` semantics.
- Minimum supported shapes:
  - OpenAI/Responses-like `messages[*]` objects with `role`.
  - Anthropic-like content lists nested under message objects with `role`.
  - Generic payloads where a rule provides metadata hints.
- Add rule metadata hints:
  - `turn_container_path`: path to turn/message list, default inferred from common roots such as `messages`.
  - `turn_role_path`: role field inside each turn, default `role`.
  - `turn_value_path`: value path relative to each turn when inference is better than global path.
- If no turn context can be inferred, skip with `reason="turn_context_not_found"` rather than silently behaving like `last`.
- Store only values from the latest matching user/assistant turn.
- Add tests for user turn, assistant turn, skip/no-op, and metadata-configured relative extraction.

### 4. Per-Tool-Call Correlation And Injection

- Keep requiring `metadata.tool_call_id_path` for `per_tool_call`.
- Support existing value-relative extraction when each selected value contains the tool ID.
- Add payload-relative correlation with metadata:
  - `tool_container_path`
  - `tool_call_id_path` relative to each tool container
  - `tool_value_path` relative to each tool container
- Store a mapping of `tool_call_id -> value`.
- Add injection selection:
  - If `metadata.inject_tool_call_id_path` exists, extract current tool IDs from target payload and inject matching values.
  - If `FieldCacheContext.metadata["tool_call_id"]` exists, inject that value.
  - If `inject.as_list=True`, inject all matching values as a list.
  - Otherwise skip ambiguous maps with `reason="tool_call_id_not_found"` or `reason="ambiguous_tool_call_values"`.
- Add tests for sibling tool ID/value extraction and target-specific injection.

### 5. Honor `FieldCacheInjection.insert`

- Wire `rule.inject.insert` into `inject_path()`.
- Add tests for list insertion where path targets a list index or append-like position if supported by the path engine.
- If the path engine does not support safe insertion yet, add minimal support or explicitly validate/reject unsafe insert paths with clear errors.

### 6. TTL Handling Without New Persistence

- `FieldCacheRule.ttl_seconds` exists but stores ignore it.
- Add TTL support to `InMemoryFieldCacheStore`.
- Keep `ProviderCacheFieldStore` compatible:
  - If the injected provider cache supports TTL arguments, pass them.
  - If not, wrap values with expiry metadata and enforce expiry on `get()`.
- Update `FieldCacheStore` protocol with optional TTL on `set`/`append` only if safe; otherwise add internal engine helper methods to avoid breaking third-party stores.
- Add tests for memory-store expiry using a fake clock if possible.
- No SQLite.

### 7. Merge JSON-Configured Rules Into Native Runtime Context

- In `RequestExecutor._native_context()`, load optional `ExperimentalConfig` and merge `parse_field_cache_rules(config, provider, model)` with provider-declared rules.
- Provider-declared rules should come first; JSON rules append and can add/override by rule name.
- Override policy:
  - If a JSON rule name matches a provider rule name, JSON replaces that rule.
  - Otherwise append.
- Environment remains primary for config path and JSON secrets remain rejected by the existing loader.
- Add tests for native context rule merge without real network calls or credentials.
- Avoid top-level imports that create cycles; use local imports where needed.

### 8. Tests

- Engine semantics tests for contextual modes.
- Native executor test proving default process-local persistence across separate `execute()` calls.
- Config merge test for JSON field-cache rules.
- TTL store tests.
- Existing trace and protocol regression tests where touched.

Focused suites:

- `tests/test_field_cache_paths.py`
- `tests/test_field_cache_engine.py`
- `tests/test_field_cache_trace.py`
- `tests/test_native_provider_executor.py`
- `tests/test_experimental_config.py`
- relevant provider declaration tests
- Phase 1b/2b regression subsets if field-cache/protocol trace serialization is touched

### 9. Documentation And Comments

- Update docstrings in `FieldCacheEngine`, `FieldCacheRule`, and store classes to explain mode semantics.
- Document when turn/tool modes skip rather than fallback.
- Document process-local default store vs injected production store.
- Keep comments focused on algorithmic rationale: turn context inference, tool-call correlation, TTL handling, and JSON rule merge precedence.

## Commit Checkpoints

1. `docs(experimental): plan field cache runtime correction`.
2. Default persistent native store.
3. Turn/tool-call mode semantics.
4. TTL/store behavior.
5. JSON rule merge/runtime wiring.
6. Review fixes after `explore` and `explore-heavy`.
7. User-facing Phase 3b report, uncommitted.

## Acceptance Criteria

- Native field cache persists across requests by default within one executor instance.
- `last_user_turn` and `last_assistant_turn` have real turn-aware semantics and skip safely when context is unavailable.
- `per_tool_call` can correlate tool IDs with sibling values and inject the right cached value by current tool ID.
- `insert`, `ttl_seconds`, and JSON-configured rules are implemented or explicitly validated with clear errors if a requested shape is unsafe.
- No raw cached values leak in traces after Phase 2b hardening.
- No SQLite or new database is introduced.
- Focused tests and dual-agent review pass.
