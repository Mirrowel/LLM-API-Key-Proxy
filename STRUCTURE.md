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
- Contains: Client facade, provider plugins, usage tracking, Anthropic compatibility, credential management, session tracking
- Key files: `__init__.py`, `rotating_client.py` (in `client/`), `provider_interface.py` (in `providers/`), `usage_manager.py`, `session_tracking.py`

**`src/rotator_library/session_tracking.py`:**
- Purpose: Evidence-based session inference with scoped anchors, confidence scoring, compaction probe detection, and deterministic affinity routing
- Contains: `SessionTracker`, `SessionAnchor`, `SessionTrackingHints`, `SessionInference`, `_MatchCandidate` (with `last_seen` tiebreaker)
- Key data types: `SessionAnchor` (evidence with strength/source/group), `SessionTrackingHints` (provider-supplied evidence), `SessionInference` (result with session_id, affinity_key, confidence, lineage_parent_session_id, namespace)
- Anchor strength levels: `strong` (tool-call IDs, provider affinity keys), `medium` (message content hashes, response anchors), `weak` (first-user text, untrusted explicit IDs)
- Scope isolation: Namespaces are `scope:{scope_key}:provider:{provider}:model:{model}` to prevent credential pool leakage
- Compaction probes: Separate anchor path (`_build_compaction_probe_anchors()`) identifies lineage parents from large early messages or explicit summarization markers; probe indexes are suppressed from normal continuity anchors; system/developer prompts excluded from continuity evidence
- Persistence: Schema-versioned JSON disk storage via `ResilientStateWriter` with generation-based write deduplication (`_dirty_generation` / `_save_io_lock`) and configurable flush interval
- Configuration: `TRUSTED_SESSION_ID_FIELDS` env var for trusted explicit ID fields; `max_anchor_records`, `max_anchors_per_session`, `persistence_flush_interval_seconds` constructor args

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
- Key files: `provider_interface.py`, `__init__.py` (auto-discovery), `gemini_cli_provider.py`, `gemini_provider.py`, `gemini_auth_base.py`, `google_oauth_base.py`, `openai_provider.py`, `openai_compatible_provider.py`, `openrouter_provider.py`, `deepseek_provider.py`, `nvidia_provider.py`, `mistral_provider.py`, `cohere_provider.py`, `groq_provider.py`, `chutes_provider.py`, `nanogpt_provider.py`, `provider_cache.py`, `example_provider.py`

**`src/rotator_library/providers/utilities/`:**
- Purpose: Shared provider utility modules for quota tracking and credential management
- Key files: `gemini_credential_manager.py`, `gemini_cli_quota_tracker.py`, `gemini_tool_handler.py`, `gemini_shared_utils.py`, `base_quota_tracker.py`, `nanogpt_quota_tracker.py`, `chutes_quota_tracker.py`

**`src/rotator_library/usage/`:**
- Purpose: Usage tracking, limit enforcement, and credential selection
- Contains: `UsageManager` facade, sub-packages for identity, tracking, limits, selection, persistence, integration
- Key files: `__init__.py`, `manager.py`, `config.py`, `types.py`

**`src/rotator_library/usage/config.py`:**
- Purpose: Per-provider usage configuration with session sticky settings
- Contains: `ProviderUsageConfig` with session sticky controls: `session_sticky_wait_seconds`, `session_sticky_entry_ttl_seconds`, `session_sticky_max_entries`
- Configuration: Per-provider `SESSION_STICKY_WAIT_SECONDS_{PROVIDER}` or global `SESSION_STICKY_WAIT_SECONDS` env vars; similarly for `SESSION_STICKY_ENTRY_TTL_SECONDS` and `SESSION_STICKY_MAX_ENTRIES`

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
**Session Tracking:** `src/rotator_library/session_tracking.py`: Evidence-based session inference with scoped anchors (~900 lines)
**Provider Interface:** `src/rotator_library/providers/provider_interface.py`: ABC for all providers (~800 lines)
**Usage Facade:** `src/rotator_library/usage/manager.py`: Usage tracking + credential selection facade (~2200 lines)
**Tests:** `tests/`: Root-level for integration tests; `tests/refactor/` for parity tests

## Naming Conventions

**Files:** `snake_case.py` — provider files follow `{provider_name}_provider.py` pattern (e.g., `gemini_cli_provider.py`)
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
**New shared type:** `src/rotator_library/core/types.py` — for types used across multiple packages
**New config default:** `src/rotator_library/config/defaults.py` — export from `config/__init__.py`
**New utility:** `src/rotator_library/utils/` — for cross-cutting utilities (paths, IO, detection)
**Tests:** `tests/test_{feature_name}.py` — for new feature tests; `tests/refactor/` for refactoring parity tests
**Retired provider:** `src/rotator_library/providers/_retired/` — keep out of auto-discovery (files starting with `_` are skipped)
