# Codebase Structure

## Directory Layout

```
[project-root]/
├── src/
│   ├── proxy_app/              # FastAPI proxy server (API surface)
│   └── rotator_library/        # Core resilience engine (library)
├── tests/                      # Test suite (pytest)
├── tools/                      # Offline utility scripts (trace reconstruction from L1 transaction artifacts)
├── stuff/                      # Related projects (Antigravity-Manager, CLIProxyAPI, etc.)
├── cache/                      # Runtime caches (device profiles, provider data)
├── logs/                       # Transaction logs and debug logs
├── usage/                      # Per-provider usage JSON files
├── oauth_creds/                # OAuth credential files
├── docs/                       # Additional documentation
├── .env                        # Environment configuration (do not commit)
├── .env.example                # Example environment template
├── Dockerfile                  # Container build definition
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Python dependencies
├── decrypt_share_link.py       # Mirrobot share-link decryptor (CI tooling)
├── DOCUMENTATION.md            # Detailed technical documentation
└── README.md                   # Project overview
```

## Directory Purposes

**`src/proxy_app/`:**
- Purpose: FastAPI application serving as the user-facing proxy gateway
- Contains: Route handlers, Pydantic models, TUI tools, startup/lifespan logic
- Key files: `main.py` (thin route surface), `route_helpers.py` (stream framing with in-band terminal error frames, request overrides, embedding fan-out), `startup.py` (OAuth credential bootstrap), `launcher_tui.py`, `quota_viewer.py`, `batch_manager.py`, `request_logger.py`, `detailed_logger.py`, `build.py`, `provider_urls.py`, `settings_tool.py`, `model_filter_gui.py`

**`src/rotator_library/`:**
- Purpose: Portable resilience library for multi-provider API key rotation
- Contains: Client facade, routing and fallback targets, protocol adapters, payload adapters, native provider execution, field cache, provider plugins, usage tracking, credential management, session tracking, leveled transaction logging
- Key files: `__init__.py`, `rotating_client.py` (in `client/`), `provider_interface.py` (in `providers/`), `usage_manager.py`, `session_tracking.py`, `transaction_logger.py`, `error_handler.py`

**`src/rotator_library/session_tracking.py`:**
- Purpose: Evidence-based session inference with scoped anchors, confidence scoring, compaction probe detection, and deterministic affinity routing
- Contains: `SessionTracker`, `SessionAnchor`, `SessionTrackingHints`, `SessionInference`, `_MatchCandidate` (with `response_groups`/`request_groups`/`matched_probe_groups` tracking and `last_seen` tiebreaker), `_CompactionDecision` (validated parent lineage with retained-history ratio and `context_probe_groups`)
- Key data types: `SessionAnchor` (evidence with strength/source/group), `SessionTrackingHints` (provider-supplied evidence), `SessionInference` (result with session_id, affinity_key, confidence, lineage_parent_session_id, namespace)
- Anchor strength levels: `strong` (trusted explicit IDs, provider affinity keys, response global IDs), `medium` (message content hashes, response anchors), `weak` (first-user text, raw tool-call IDs, untrusted explicit IDs)
- Scope isolation: Namespaces are `scope:{scope_key}:provider:{provider}:model:{model}` to prevent credential pool leakage; namespaces are immutable per session (`_refresh_and_bridge()` rejects drift, `record_response()` normalizes fallback callbacks to the original namespace), and eviction ranks `compaction_context`/`compaction_replay` anchors above ordinary evidence before falling back to deterministic value tie-breaking
- Compaction probes: Separate anchor path (`_build_compaction_probe_anchors()`) probes only early user/system/developer messages (assistant/tool/function-result history excluded) and requires structural replacement of more than half the parent's high-water request history via `_evaluate_compaction()` (`_retained_history_ratio()`); unmarked summaries must additionally overlap at least two distinct response events (`_MIN_UNMARKED_RESPONSE_GROUPS`) plus a retained request group; authoritative identity (`_is_authoritative_identity_anchor()`: trusted explicit or provider) takes precedence and suppresses unrelated compaction lineage; exact resends reuse the validated child session via opaque `compaction_replay` anchors while changed-tail continuations bind via `compaction_context` anchors (`_find_compaction_context()` / `_compaction_context_anchor()`) minted only from probe groups that matched parent response evidence; probe indexes are suppressed from normal continuity anchors; system/developer prompts excluded from continuity evidence
- Persistence: Schema-versioned JSON disk storage (v3) via `ResilientStateWriter` with generation-based write deduplication (`_dirty_generation` / `_save_io_lock`), dirty state retained on failed writes, stale delayed generations rejected, anchor ownership rebuilt on load (rejecting malformed containers, non-finite timestamps, expired sessions, orphan anchors, namespace mismatches, invalid strengths, and unsupported schemas), and configurable flush interval
- Configuration: `TRUSTED_SESSION_ID_FIELDS` env var for trusted explicit ID fields; `SESSION_PERSISTENCE_ENABLED` / `SESSION_PERSISTENCE_FLUSH_INTERVAL_SECONDS` env vars on `RotatingClient` for restart persistence; `max_anchor_records`, `max_anchors_per_session`, `persistence_flush_interval_seconds` constructor args

**`src/rotator_library/transaction_logger.py`:**
- Purpose: Unified transaction logging between the OpenAI-compatible client layer and provider implementations; each API transaction gets a unique directory with client-level and provider-level I/O, leveled by detail tier (W12)
- Contains: `TransactionLogger` class with tier plumbing (`TRANSACTION_LOG_LEVEL`: 1 = boundaries + metadata (default), 2 = + intermediates, 3 = verbose per-frame); capture-on-error via `flush_capture_on_error()` / `_error_qualifies_for_capture()` (request-related failures — 400-class payload/protocol problems, unexpected crashes — archive buffered intermediates; rotation-class failures — rate-limit, quota, auth, transport timeout — do not); L1 retention bound `_prune_old_transactions()` (`TRANSACTION_LOG_RETENTION`, newest N directories); `_make_json_safe` recursive converter for Pydantic/dataclass/`Path`/timestamp objects with circular-reference tracking; helpers `assemble_streaming_response()`, `_strip_framework_keys()`; metadata v2 records correlation, tier, and reconstruction notes
- Output layout: `logs/transactions/MMDD_HHMMSS_{provider}_{model}_{request_id}/` containing `request.json`, `response.json`, `streaming_chunks.jsonl`, `metadata.json`, an optional `capture/captured_trace.json` (error-archived intermediates), and an optional `provider/` subdir (`request_payload.json`, `response_stream.log`, `final_response.json`, `error.log`); JSON artifacts carry a `.zst` suffix when zstd compression is active
- Compression: `src/rotator_library/utils/zstd_io.py` owns the behavior — zstd when `zstandard` is installed, plain files with a metadata flag otherwise; logging never fails a request and never blocks startup
- Redaction: trace snapshots from `transform_trace.py` (`TransformTraceWriter`) scrub secrets and provider protocol-state fields (reasoning content, thought signatures, opaque state) at L1 boundaries
- Integration: Instantiated by `RequestContextBuilder` and `AnthropicHandler` when request logging is enabled; threaded through `RequestContext.transaction_logger` into the executor, streaming handler, field cache engine, and adapter base; the responses route finalizes metadata itself (`finalize_metadata`) since it files no chat-shaped body
- Offline reconstruction: `tools/reconstruct_traces.py` replays the deterministic pipeline (parse → neutral canonical → provider build) over L1 artifacts to regenerate L2-style trace reports without contacting a provider
- Toggle: Enabled via proxy `--enable-request-logging` flag

**`src/rotator_library/client/`:**
- Purpose: Client-side request execution with retry, rotation, and protocol-neutral entry points (`agenerate()` executes a request in the client's own wire protocol)
- Contains: `RotatingClient` facade and extracted components
- Key files: `rotating_client.py`, `executor.py`, `stream_ops.py` (neutral-event stream pipeline), `streaming.py` (legacy SSE parsing helpers), `filters.py`, `models.py`, `transforms.py`, `anthropic.py`, `gemini.py`, `protocol_selection.py`, `request_builder.py`, `quota.py`, `usage_managers.py`, `scopes.py`, `model_discovery.py`, `stream_retry_policy.py`, `types.py`

**`src/rotator_library/client/request_builder.py`:**
- Purpose: Build `RequestContext` with session inference and provider hints
- Contains: `RequestContextBuilder` — resolves provider via `get_session_tracking_hints()`, runs `SessionTracker.infer_session()`, populates session affinity and namespace fields on `RequestContext`

**`src/rotator_library/routing/`:**
- Purpose: Resolve model names to ordered execution targets — direct `provider/model` references, env-configured fallback groups, and `provider:profile/model` transport-profile addressing; identity stays provider-level (usage pools, cooldowns, classifiers, session namespaces, cache provenance key on the bare provider name — the profile only steers transport)
- Key files: `types.py` (`RouteTarget` with `profile` and `execution` ∈ {`auto`, `native`, `custom`, `litellm_fallback`}, `FallbackGroup`, `RoutingDecision`, failover/stop error vocabularies `DEFAULT_FAILOVER_ON` / `DEFAULT_STOP_ON` / `HARD_STOP_ON`), `config.py` (`parse_route_target()` — `provider/model[@execution]`, profile split on the provider segment only so model names keep their colons; `load_routing_config_from_env()` — env vars are the final override layer), `resolver.py` (`FallbackResolver` — model routes, `group:` aliases, requested-target promotion), `executor.py` (`FallbackAttemptRunner`, `FallbackExhaustedError`), `policy.py` (`FallbackPolicy` — error alias normalization and failover/stop decisions), `attempts.py` (`clone_context_for_target()` — per-target context copies preserving the original for traceability), `profiles.py` (`parse_model_reference()`, `resolve_profile()`, `split_profile_from_provider()` — explicit profiles must exist; bare names pick the default profile or the unique profile matching the client protocol, else a fail-fast error, never silent conversion)
- Integration: `RequestContextBuilder._resolve_routing_decision()` stamps `routing_targets` / `routing_group_name` / `routing_group` / `routing_target_index` / `routing_attempt_history` on `RequestContext`; `RequestExecutor` dispatches each target with its `execution` mode

**`src/rotator_library/providers/`:**
- Purpose: Provider-specific implementations and plugin discovery
- Contains: One file per provider implementing `ProviderInterface`, shared utilities, retired providers
- Key files: `provider_interface.py`, `__init__.py` (auto-discovery), `gemini_provider.py`, `openai_provider.py`, `openai_compatible_provider.py`, `openrouter_provider.py`, `deepseek_provider.py`, `nvidia_provider.py`, `mistral_provider.py`, `cohere_provider.py`, `groq_provider.py`, `chutes_provider.py`, `firmware_provider.py`, `nanogpt_provider.py`, `provider_cache.py`, `example_provider.py`

**`src/rotator_library/providers/utilities/`:**
- Purpose: Shared provider utility modules for quota tracking and credential management
- Key files: `base_quota_tracker.py`, `nanogpt_quota_tracker.py`, `firmware_quota_tracker.py`, `chutes_quota_tracker.py`

**`src/rotator_library/protocols/`:**
- Purpose: Native protocol adapters converting between client wire protocols and the neutral canonical model (`UnifiedRequest` / `UnifiedMessage` / `UnifiedResponse` / `UnifiedStreamEvent`); the client response protocol always equals the client request protocol
- Contains: `ProtocolAdapter` base (`base.py`, override-friendly defaults for providers with near-standard protocol quirks), registry with auto-discovery (`registry.py` — `PROTOCOL_PLUGINS`, `get_protocol`), cross-protocol canonical semantics (`canonical.py`), destination capability validation (`validation.py`), neutral stream formatting (`streaming.py` — `format_canonical_stream_event`), operation vocabulary (`operation.py`), shared types (`types.py`), named transforms for portable cached state (`transforms.py` — `register_transform()` / `get_transform()`; referenced from field-cache rule metadata so e.g. plain chat-completions reasoning text restores as an Anthropic thinking block and vice versa)
- Key files: `openai_chat.py`, `responses.py`, `anthropic_messages.py`, `gemini.py`, `ollama.py`, `openai_audio.py`, `openai_embeddings.py`, `openai_images.py`, `mcp.py`, `litellm_fallback.py`, `transforms.py`
- Capability model: destination protocols declare supported content capabilities (`validation.py`), and tool-type support is per destination — `openai_chat` accepts `custom`, `anthropic_messages` accepts Anthropic-native server/hosted tools (identity is the versioned type; they round-trip verbatim with no cross-protocol mapping), `responses` accepts its native ToolParam union (hosted tools are first-class), `gemini` maps hosted web-search/googleSearch/codeExecution/urlContext/googleMaps server tools and rejects unmapped hosted families, all others accept `function` only; refusal evidence degrades to text with refusal stop semantics rather than blocking conversion

**`src/rotator_library/adapters/`:**
- Purpose: Composable payload adapters running on the raw provider wire between the protocol layer and providers — request, response, and stream-event stages (W7 contract: response adapters are WIRE adapters, applied upstream of canonical conversion in declared order)
- Key files: `base.py` (`PayloadAdapter`, `AdapterContext`, `run_adapter_chain` with trace entries around the chain), `registry.py` (auto-discovery — `register_adapter()`, `get_adapter()`, `list_adapters()`), `builtin.py` (`NoOpAdapter`, `ModelOverrideAdapter`, `SuppressDeveloperRoleAdapter`, `ReasoningContentAdapter`, `FieldRenameAdapter`, `AntigravityEnvelopeAdapter`)
- Integration: providers reference adapters by name via `adapter_names` / `get_adapter_names()` with per-adapter config from `get_adapter_config()`; the native executor resolves and runs the chain on raw payloads before field-cache extraction

**`src/rotator_library/native_provider/`:**
- Purpose: Provider-native HTTP execution through protocol adapters, bypassing litellm; native is the default for providers that declare a native protocol — selected via the `native` execution mode (`RouteTarget.execution`) or `auto` mode when `ProviderInterface.should_use_native_protocol()` allows it, with LiteLLM as an explicit, logged `litellm_fallback`
- Contains: `NativeProviderExecutor` (`executor.py`), `NativeProviderContext` (`context.py` — carries the pristine same-protocol client payload as `raw_client_request`, transport overlay records, and the wire-exhaustive `stream_usage_record` adopted by the operational stream layer), `NativeHTTPTransport` (`http.py`)
- Key files: `executor.py`, `context.py`, `http.py`, `streaming.py`

**`src/rotator_library/field_cache/`:**
- Purpose: Extract and re-inject provider-protocol state (reasoning content, thought signatures, prompt-cache keys, response IDs) on the native execution path only — explicit LiteLLM fallback or a custom provider's own path neither caches nor injects; rules are protocol/provider extensions, not session-affinity logic (session tracking decides continuity, field cache preserves protocol state)
- Key files: `engine.py` (`FieldCacheEngine` — extraction/injection passes with `FieldCacheOperation` summaries, each recording the transport profile that served the request, and `build_cache_key`), `types.py` (`FieldCacheRule` — source/mode/scope/inject declarations; provider+model are the required identity with credential and session as optional refinements), `paths.py` (`parse_path()` / `extract_path()` / `inject_path()`), `store.py` (`InMemoryFieldCacheStore`, `ProviderCacheFieldStore`), `replay.py` (declarative cache-and-replay: env `<NAME>_CACHE_REPLAY` or the provider's `cache_replay` class attribute compiles to ordinary `FieldCacheRule`s — one surface, one engine; extraction always runs on the finalized, assembled response), `compat.py` (compatibility classes: `bound` fields are provider-locked opaque state restoring only to the exact provider+model; `portable` fields inherit within declared compatibility groups via `CompatibilityRegistry` and the `FIELD_CACHE_COMPAT_GROUPS` env var — unknown pairs are denied by default), `__init__.py`
- Integration: cross-format restores run through named transforms registered in `src/rotator_library/protocols/transforms.py`; rule precedence per name is JSON runtime config > env `<NAME>_CACHE_REPLAY` (operator) > class `cache_replay` (code author) > provider-declared `field_cache_rules`, with a uniform behavior-weakening guard so isolation is never silently narrowed

**`src/rotator_library/usage/`:**
- Purpose: Usage tracking, limit enforcement, and credential selection
- Contains: `UsageManager` facade, sub-packages for identity, tracking, limits, selection, persistence, integration
- Key files: `__init__.py`, `manager.py`, `config.py`, `types.py`

**`src/rotator_library/usage/config.py`:**
- Purpose: Per-provider usage configuration with session sticky settings and quota-exhaustion policies
- Contains: `ProviderUsageConfig` with session sticky controls (`session_sticky_wait_seconds`, `session_sticky_entry_ttl_seconds`, `session_sticky_max_entries`) and no-reset exhaustion controls (`no_reset_exhaustion_policy` ∈ {`warn_only`, `cooldown`, `disable_scope`}, `no_reset_exhaustion_cooldown_seconds`)
- Configuration: Per-provider `SESSION_STICKY_WAIT_SECONDS_{PROVIDER}` or global `SESSION_STICKY_WAIT_SECONDS` env vars; similarly for `SESSION_STICKY_ENTRY_TTL_SECONDS` and `SESSION_STICKY_MAX_ENTRIES`. Per-provider `QUOTA_NO_RESET_EXHAUSTION_POLICY_{PROVIDER}` / global `QUOTA_NO_RESET_EXHAUSTION_POLICY`, and `QUOTA_NO_RESET_COOLDOWN_SECONDS_{PROVIDER}` / global `QUOTA_NO_RESET_COOLDOWN_SECONDS`; provider classes may set `default_no_reset_exhaustion_policy` / `default_no_reset_exhaustion_cooldown_seconds` as baseline

**`src/rotator_library/usage/tracking/`:**
- Purpose: Usage recording engine and window management
- Key files: `engine.py`, `windows.py`

**`src/rotator_library/usage/limits/`:**
- Purpose: Limit checking and enforcement modules
- Key files: `engine.py`, `base.py`, `concurrent.py`, `cooldowns.py`, `custom_caps.py`, `fair_cycle.py`, `window_limits.py`

**`src/rotator_library/usage/selection/`:**
- Purpose: Credential selection with pluggable strategies
- Key files: `engine.py`, `strategies/balanced.py`, `strategies/sequential.py`

**`src/rotator_library/usage/selection/strategies/sequential.py`:**
- Purpose: Sequential credential rotation with TTL-based sticky entries and affinity-based placement
- Contains: `SequentialStrategy` with `_StickyEntry` (credential + last_seen), TTL pruning, max-entry trimming, `session_affinity_key` for deterministic first-pick, and `threading.RLock` for thread-safe access across `select`, `mark_exhausted`, `get_current`, `clear_sticky`

**`src/rotator_library/usage/identity/`:**
- Purpose: Stable credential identity management
- Key files: `registry.py`

**`src/rotator_library/usage/persistence/`:**
- Purpose: JSON file persistence for usage data
- Key files: `storage.py`

**`src/rotator_library/usage/integration/`:**
- Purpose: Integration hooks and API for external consumers
- Key files: `api.py`, `hooks.py`

**`src/rotator_library/responses/`:**
- Purpose: OpenAI Responses API compatibility — object creation, retrieval, deletion, and streaming with `previous_response_id` continuation
- Contains: `ResponsesService` (orchestrator) bridging through the chat-completions executor; `ResponsesBridge` (Responses ↔ chat translation via `ResponsesProtocol`; structured output survives the bridge as a legal chat `response_format`); `ResponsesStore` protocol with `InMemoryResponsesStore` (default) and `ProviderCacheResponsesStore` (durable JSON cache — corrupt rows are cache misses, never 500s) backends plus `create_configured_responses_store` factory; `ResponsesSSEFormatter` / `ResponsesStreamEvent` / `ResponsesStreamState` SSE streaming helpers; `websocket.py` (`ResponsesWebSocketSession` / `ResponsesWebSocketFormatter` — WebSocket Mode: response.create turns, connection-local ZDR continuation cache, spec-shaped error frames) (monotonic `sequence_number` on every event, full created → in_progress → items → terminal lifecycle); `StoredResponse`, `ResponsesStoreSettings` (incl. `store_failed` policy), `generate_response_id` types
- Key files: `service.py`, `bridge.py`, `store.py`, `streaming.py`, `websocket.py`, `types.py`, `__init__.py`
- Scope isolation: `StoredResponse` records are keyed by session isolation key (`derive_session_isolation_key`) so `previous_response_id` continuation cannot cross credential pools; stores enforce TTL (`ResponsesStoreSettings.ttl_seconds`), bounded memory (`max_items`), and prune expired/overflow entries
- Storage backends: Memory is the default; provider-cache backend reuses the existing JSON `ProviderCache` (SHA-256 scoped keys) for durable storage without a new database — selection via `config.experimental` runtime settings

**`src/rotator_library/core/`:**
- Purpose: Shared types, constants, utilities, and error definitions
- Key files: `types.py` (`RequestContext` with session tracking fields: `session_affinity_key`, `session_tracker`, `session_possible_compaction`, `session_lineage_parent_id`, `session_tracking_namespace`), `config.py`, `constants.py`, `errors.py`, `utils.py`

**`src/rotator_library/config/`:**
- Purpose: Centralized configuration defaults
- Key files: `__init__.py`, `defaults.py`

**`src/rotator_library/utils/`:**
- Purpose: Shared utility modules
- Key files: `paths.py`, `resilient_io.py`, `reauth_coordinator.py`, `headless_detection.py`, `suppress_litellm_warnings.py`, `zstd_io.py` (zstd-compressed JSON/JSONL writing with graceful degradation to plain files when `zstandard` is missing)

**`tests/`:**
- Purpose: Test suite organized by feature area
- Contains: Unit and integration tests for the rotator library
- Key files: `test_selection_engine.py`, `test_fair_cycle_and_custom_caps.py`, `test_fallback_groups.py`, `test_error_handler.py`, `test_executor_session_forwarding.py`, `test_session_tracking.py`, `test_protocol_streaming_matrix.py`, `test_w2_neutral_completeness.py`, `test_w3_same_protocol_fidelity.py`, `test_w4_cross_protocol_conversion.py`, `test_w5_stream_parity.py`, `test_w7_adapter_staging.py`, `test_w11_native_default.py`, `test_w12_transaction_tiers.py`, `test_w13_cache_replay.py`, `test_w_prof_profiles.py`, `test_anthropic_transform_tracing.py`, `test_transaction_logger_json_safety.py`

**`tests/refactor/`:**
- Purpose: Tests verifying parity after refactoring from monolithic client.py
- Contains: Tests for executor, streaming handler, failure logging, usage tracking parity
- Key files: `test_executor_streaming_parity.py`, `test_executor_non_streaming_parity.py`, `test_streaming_handler_behavior.py`

## Key File Locations

**Entry Points:** `src/proxy_app/main.py`: FastAPI server, TUI launcher, credential tool
**Configuration:** `src/rotator_library/config/defaults.py`: All tunable defaults (rotation mode, cooldowns, fair cycle, concurrency)
**Core Logic:** `src/rotator_library/client/executor.py`: Unified retry/rotation engine with execution-mode dispatch (~3700 lines)
**Session Tracking:** `src/rotator_library/session_tracking.py`: Evidence-based session inference with scoped anchors (~2000 lines)
**Provider Interface:** `src/rotator_library/providers/provider_interface.py`: ABC for all providers (~1100 lines)
**Usage Facade:** `src/rotator_library/usage/manager.py`: Usage tracking + credential selection facade (~2300 lines)
**Tests:** `tests/`: Root-level for integration tests; `tests/refactor/` for parity tests

## Naming Conventions

**Files:** `snake_case.py` — provider files follow `{provider_name}_provider.py` pattern (e.g., `openai_provider.py`)
**Directories:** `snake_case` — package directories match their Python module purpose
**Providers:** Named by stripping `_provider` suffix from filename; `nvidia_provider.py` remapped to key `nvidia_nim`
**Tests:** `test_{feature_name}.py` — co-located in `tests/` directory

## Where to Add New Code

**New provider:** `src/rotator_library/providers/{name}_provider.py` — extend `ProviderInterface`, auto-discovered by `__init__.py`
**New provider native transport:** Declare class-level runtime config on the provider — `protocol_name`, `adapter_names`, `field_cache_rules`, `native_streaming_supported`; multi-profile providers set `transport_profiles` + `default_profile` and are addressed as `provider:profile/model`
**New payload adapter:** `src/rotator_library/adapters/{name}.py` — extend `PayloadAdapter`, register via `register_adapter()`, reference from the provider's `adapter_names`
**New field-cache transform:** `src/rotator_library/protocols/transforms.py` — decorate with `@register_transform(name)` and reference from field-cache rule metadata `{"transform": "<name>"}` (portable rules only)
**New cache-and-replay rule:** env `<NAME>_CACHE_REPLAY` JSON or the provider's `cache_replay` class attribute — compiled to ordinary `FieldCacheRule`s via `src/rotator_library/field_cache/replay.py`; precedence per rule name is JSON runtime config > env > class > provider-declared
**New fallback route:** `load_routing_config_from_env()` in `src/rotator_library/routing/config.py` — fallback groups and model-route aliases; target grammar is `provider/model[@execution]` with optional `provider:profile/model` addressing
**New provider session evidence:** Override `get_session_tracking_hints()` on `ProviderInterface` — return `SessionTrackingHints` with anchors, affinity key, or scope
**New provider utility:** `src/rotator_library/providers/utilities/{name}_quota_tracker.py` — for quota tracking or credential management
**New protocol adapter:** `src/rotator_library/protocols/{name}.py` — extend `ProtocolAdapter`, register via `register_protocol()`; auto-discovered through `PROTOCOL_PLUGINS`
**New rotation strategy:** `src/rotator_library/usage/selection/strategies/{name}.py` — implement strategy interface, register in `SelectionEngine`
**New limit checker:** `src/rotator_library/usage/limits/{name}.py` — extend limit engine
**New proxy endpoint:** `src/proxy_app/main.py` — add route handler to the FastAPI app
**New Anthropic behavior:** `src/rotator_library/protocols/anthropic_messages.py` — extend the `anthropic_messages` protocol adapter; all Anthropic ↔ canonical translation lives there
**New Responses store backend:** Implement the `ResponsesStore` protocol in `src/rotator_library/responses/store.py` and select it in `create_configured_responses_store()` — keep scope-keyed retrieval and TTL/overflow pruning
**New shared type:** `src/rotator_library/core/types.py` — for types used across multiple packages
**New config default:** `src/rotator_library/config/defaults.py` — export from `config/__init__.py`
**New utility:** `src/rotator_library/utils/` — for cross-cutting utilities (paths, IO, detection)
**Tests:** `tests/test_{feature_name}.py` — for new feature tests; `tests/refactor/` for refactoring parity tests
**Retired provider:** `src/rotator_library/providers/_retired/` — keep out of auto-discovery (files starting with `_` are skipped)
