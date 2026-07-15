# Codebase Structure

## Directory Layout

```
[project-root]/
├── src/
│   ├── proxy_app/              # FastAPI proxy server (API surface)
│   └── rotator_library/        # Core resilience engine (library)
├── tests/                      # Test suite (pytest)
├── tools/                      # Utility scripts (litellm scraper, vivgrid signup)
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
├── DOCUMENTATION.md            # Detailed technical documentation
└── README.md                   # Project overview
```

## Directory Purposes

**`src/proxy_app/`:**
- Purpose: FastAPI application serving as the user-facing proxy gateway
- Contains: Route handlers, Pydantic models, TUI tools, startup/lifespan logic
- Key files: `main.py`, `launcher_tui.py`, `quota_viewer.py`, `batch_manager.py`, `request_logger.py`, `detailed_logger.py`, `build.py`, `provider_urls.py`, `settings_tool.py`, `model_filter_gui.py`

**`src/rotator_library/`:**
- Purpose: Portable resilience library for multi-provider API key rotation
- Contains: Client facade, provider plugins, usage tracking, Anthropic compatibility, credential management, session tracking, transaction logging
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
- Purpose: Unified transaction logging between the OpenAI-compatible client layer and provider implementations; each API transaction gets a unique directory with client-level and provider-level I/O
- Contains: `TransactionLogger` class; `_make_json_safe` recursive converter for Pydantic/dataclass/`Path`/timestamp objects with circular-reference tracking; helpers `assemble_streaming_response()`, `_strip_framework_keys()`
- Output layout: `logs/transactions/MMDD_HHMMSS_{provider}_{model}_{request_id}/` containing `request.json`, `response.json`, `streaming_chunks.jsonl`, `metadata.json`, and an optional `provider/` subdir (`request_payload.json`, `response_stream.log`, `final_response.json`, `error.log`)
- Integration: Instantiated by `RequestContextBuilder` and `AnthropicHandler` when request logging is enabled; threaded through `RequestContext.transaction_logger` into the executor, streaming handler, field cache engine, and adapter base
- Toggle: Enabled via proxy `--enable-request-logging` flag

**`src/rotator_library/client/`:**
- Purpose: Client-side request execution with retry and rotation
- Contains: `RotatingClient` facade and extracted components
- Key files: `rotating_client.py`, `executor.py`, `streaming.py`, `filters.py`, `models.py`, `transforms.py`, `anthropic.py`, `request_builder.py`, `quota.py`, `usage_managers.py`, `scopes.py`, `model_discovery.py`, `stream_retry_policy.py`, `types.py`

**`src/rotator_library/client/request_builder.py`:**
- Purpose: Build `RequestContext` with session inference and provider hints
- Contains: `RequestContextBuilder` — resolves provider via `get_session_tracking_hints()`, runs `SessionTracker.infer_session()`, populates session affinity and namespace fields on `RequestContext`

**`src/rotator_library/providers/`:**
- Purpose: Provider-specific implementations and plugin discovery
- Contains: One file per provider implementing `ProviderInterface`, shared utilities, retired providers
- Key files: `provider_interface.py`, `__init__.py` (auto-discovery), `gemini_provider.py`, `openai_provider.py`, `openai_compatible_provider.py`, `openrouter_provider.py`, `deepseek_provider.py`, `nvidia_provider.py`, `mistral_provider.py`, `cohere_provider.py`, `groq_provider.py`, `chutes_provider.py`, `firmware_provider.py`, `nanogpt_provider.py`, `provider_cache.py`, `example_provider.py`

**`src/rotator_library/providers/utilities/`:**
- Purpose: Shared provider utility modules for quota tracking and credential management
- Key files: `base_quota_tracker.py`, `nanogpt_quota_tracker.py`, `firmware_quota_tracker.py`, `chutes_quota_tracker.py`

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

**`src/rotator_library/anthropic_compat/`:**
- Purpose: Anthropic Messages API ↔ OpenAI Chat Completions API translation
- Key files: `translator.py`, `models.py`, `streaming.py`

**`src/rotator_library/responses/`:**
- Purpose: OpenAI Responses API compatibility — object creation, retrieval, deletion, and streaming with `previous_response_id` continuation
- Contains: `ResponsesService` (orchestrator) bridging through the chat-completions executor; `ResponsesBridge` (Responses ↔ chat translation via `ResponsesProtocol`); `ResponsesStore` protocol with `InMemoryResponsesStore` (default) and `ProviderCacheResponsesStore` (durable JSON cache) backends plus `create_configured_responses_store` factory; `ResponsesSSEFormatter` / `ResponsesWebSocketFormatter` / `ResponsesStreamEvent` / `ResponsesStreamState` streaming helpers; `StoredResponse`, `ResponsesStoreSettings`, `generate_response_id` types
- Key files: `service.py`, `bridge.py`, `store.py`, `streaming.py`, `types.py`, `__init__.py`
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
- Key files: `paths.py`, `resilient_io.py`, `reauth_coordinator.py`, `headless_detection.py`, `suppress_litellm_warnings.py`

**`tests/`:**
- Purpose: Test suite organized by feature area
- Contains: Unit and integration tests for the rotator library
- Key files: `test_selection_engine.py`, `test_fair_cycle_and_custom_caps.py`, `test_fallback_groups.py`, `test_error_handler.py`, `test_executor_session_forwarding.py`, `test_session_tracking.py`

**`tests/refactor/`:**
- Purpose: Tests verifying parity after refactoring from monolithic client.py
- Contains: Tests for executor, streaming handler, failure logging, usage tracking parity
- Key files: `test_executor_streaming_parity.py`, `test_executor_non_streaming_parity.py`, `test_streaming_handler_behavior.py`

## Key File Locations

**Entry Points:** `src/proxy_app/main.py`: FastAPI server, TUI launcher, credential tool
**Configuration:** `src/rotator_library/config/defaults.py`: All tunable defaults (rotation mode, cooldowns, fair cycle, concurrency)
**Core Logic:** `src/rotator_library/client/executor.py`: Unified retry/rotation engine (~1500 lines)
**Session Tracking:** `src/rotator_library/session_tracking.py`: Evidence-based session inference with scoped anchors (~1900 lines)
**Provider Interface:** `src/rotator_library/providers/provider_interface.py`: ABC for all providers (~800 lines)
**Usage Facade:** `src/rotator_library/usage/manager.py`: Usage tracking + credential selection facade (~2200 lines)
**Tests:** `tests/`: Root-level for integration tests; `tests/refactor/` for parity tests

## Naming Conventions

**Files:** `snake_case.py` — provider files follow `{provider_name}_provider.py` pattern (e.g., `openai_provider.py`)
**Directories:** `snake_case` — package directories match their Python module purpose
**Providers:** Named by stripping `_provider` suffix from filename; `nvidia_provider.py` remapped to key `nvidia_nim`
**Tests:** `test_{feature_name}.py` — co-located in `tests/` directory

## Where to Add New Code

**New provider:** `src/rotator_library/providers/{name}_provider.py` — extend `ProviderInterface`, auto-discovered by `__init__.py`
**New provider session evidence:** Override `get_session_tracking_hints()` on `ProviderInterface` — return `SessionTrackingHints` with anchors, affinity key, or scope
**New provider utility:** `src/rotator_library/providers/utilities/{name}_quota_tracker.py` — for quota tracking or credential management
**New rotation strategy:** `src/rotator_library/usage/selection/strategies/{name}.py` — implement strategy interface, register in `SelectionEngine`
**New limit checker:** `src/rotator_library/usage/limits/{name}.py` — extend limit engine
**New proxy endpoint:** `src/proxy_app/main.py` — add route handler to the FastAPI app
**New Anthropic translation:** `src/rotator_library/anthropic_compat/` — add models or translation logic
**New Responses store backend:** Implement the `ResponsesStore` protocol in `src/rotator_library/responses/store.py` and select it in `create_configured_responses_store()` — keep scope-keyed retrieval and TTL/overflow pruning
**New shared type:** `src/rotator_library/core/types.py` — for types used across multiple packages
**New config default:** `src/rotator_library/config/defaults.py` — export from `config/__init__.py`
**New utility:** `src/rotator_library/utils/` — for cross-cutting utilities (paths, IO, detection)
**Tests:** `tests/test_{feature_name}.py` — for new feature tests; `tests/refactor/` for refactoring parity tests
**Retired provider:** `src/rotator_library/providers/_retired/` — keep out of auto-discovery (files starting with `_` are skipped)
