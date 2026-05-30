# Architecture

## Pattern Overview

**Overall:** Layered proxy gateway with plugin-based provider system and intelligent credential rotation

**Key Characteristics:**
- Two-layer separation: FastAPI proxy (`proxy_app`) provides the API surface; `rotator_library` provides all resilience logic
- Plugin-based provider discovery: providers auto-register from `src/rotator_library/providers/` files plus dynamic `*_API_BASE` environment variables
- Singleton providers via `SingletonABCMeta` metaclass — one instance per provider class shared across all components
- Lazy imports at package boundaries (`__getattr__`) to keep startup fast
- All state is per-provider, per-credential, windowed and persisted to JSON files
- Evidence-based session tracking with scoped anchor namespaces, confidence scoring, compaction detection, and deterministic affinity keys — `src/rotator_library/session_tracking.py`

## Layers

**Proxy Application Layer:**
- Purpose: Expose OpenAI- and Anthropic-compatible HTTP endpoints, handle auth, logging, TUI
- Location: `src/proxy_app/`
- Contains: FastAPI app, route handlers, Pydantic request/response models, launcher TUI, quota viewer
- Depends on: `rotator_library`, `litellm`, `fastapi`, `uvicorn`
- Used by: External API clients (Claude Code, Gemini CLI, OpenAI SDK, curl)

**Client Facade Layer:**
- Purpose: Provide a single `RotatingClient` entry point that orchestrates retries, rotation, and streaming
- Location: `src/rotator_library/client/`
- Contains: `RotatingClient` (facade), `RequestExecutor`, `CredentialFilter`, `ModelResolver`, `ProviderTransforms`, `StreamingHandler`, `AnthropicHandler`, `RequestContextBuilder` (resolves provider hints for session evidence)
- Depends on: `rotator_library.usage`, `rotator_library.providers`, `rotator_library.core`
- Used by: `proxy_app` via `from rotator_library import RotatingClient`

**Provider Plugin Layer:**
- Purpose: Abstract provider-specific behavior (model discovery, auth, transforms, quota tracking, background jobs, session evidence)
- Location: `src/rotator_library/providers/`
- Contains: `ProviderInterface` (ABC) and one file per provider (`*_provider.py`)
- Depends on: `litellm`, provider utility modules
- Used by: Client layer via `PROVIDER_PLUGINS` dict, auto-discovered at import time

**Usage Tracking Layer:**
- Purpose: Track per-credential usage, enforce limits, select credentials, persist state
- Location: `src/rotator_library/usage/`
- Contains: `UsageManager` (facade), `TrackingEngine`, `LimitEngine`, `SelectionEngine`, `WindowManager`, `CredentialRegistry`, `UsageStorage`, `UsageAPI`, `HookDispatcher`
- Depends on: `rotator_library.core`, provider config
- Used by: Client layer for credential selection and usage recording

**Anthropic Compatibility Layer:**
- Purpose: Translate between Anthropic Messages API and OpenAI Chat Completions API formats
- Location: `src/rotator_library/anthropic_compat/`
- Contains: Data models (`models.py`), request/response translator (`translator.py`), streaming wrapper (`streaming.py`)
- Depends on: `rotator_library.core`
- Used by: Client layer's `AnthropicHandler`, proxy routes `/v1/messages`, `/v1/messages/count_tokens`

**Core Types & Config Layer:**
- Purpose: Shared type definitions, constants, error classification, config defaults
- Location: `src/rotator_library/core/`, `src/rotator_library/config/`
- Contains: `RequestContext`, `CredentialInfo`, `CustomCapConfig`, `FairCycleConfig`, error classifiers, cooldown constants
- Depends on: Nothing internal (leaf layer)
- Used by: All other layers

## Data Flow

**Chat Completion Request:**

1. Client sends POST to `/v1/chat/completions` — `src/proxy_app/main.py`
2. FastAPI handler calls `client.chat_completions()` — `src/rotator_library/client/rotating_client.py`
3. `RequestContextBuilder` resolves provider hints, runs session inference, builds a `RequestContext` with session affinity key and namespace — `src/rotator_library/client/request_builder.py`
4. `ModelResolver` resolves model name to provider + litellm format — `src/rotator_library/client/models.py`
5. `UsageManager.acquire_credential()` selects best credential via `SelectionEngine` — `src/rotator_library/usage/manager.py`
6. `RequestExecutor` executes with retry/rotation logic, calling litellm — `src/rotator_library/client/executor.py`
7. For streaming, `StreamingHandler` processes chunks and tracks usage — `src/rotator_library/client/streaming.py`
8. On completion, `UsageManager` records success/failure, `SessionTracker.record_response()` records response-derived anchors — `src/rotator_library/usage/manager.py`, `src/rotator_library/session_tracking.py`

**Anthropic Messages Request:**

1. Client sends POST to `/v1/messages` — `src/proxy_app/main.py`
2. Proxy translates Anthropic format to OpenAI format via `anthropic_compat.translator` — `src/rotator_library/anthropic_compat/translator.py`
3. Standard chat completion flow follows (steps 2–8 above)
4. Response is translated back to Anthropic format — `src/rotator_library/anthropic_compat/translator.py`
5. For streaming, `anthropic_streaming_wrapper` wraps the SSE stream — `src/rotator_library/anthropic_compat/streaming.py`

**Provider Discovery:**

1. `providers/__init__.py` scans all `*_provider.py` files in `src/rotator_library/providers/`
2. Each module's `ProviderInterface` subclass is registered in `PROVIDER_PLUGINS` dict keyed by provider name
3. Environment variables matching `*_API_BASE` create dynamic `DynamicOpenAICompatibleProvider` entries
4. `nvidia_provider.py` is remapped to key `nvidia_nim` to match litellm's naming

**Session Inference:**

1. `RequestContextBuilder` calls `provider.get_session_tracking_hints()` to collect provider-specific evidence — `src/rotator_library/client/request_builder.py`
2. `SessionTracker.infer_session()` builds scoped anchors from explicit IDs, message content, tool-call IDs, and provider hints — `src/rotator_library/session_tracking.py`
3. Anchors are namespaced by scope/provider/model so sticky evidence never leaks between credential pools
4. `_best_match()` scores anchor overlap against live sessions; confidence is `strong` (any strong match), `probable` (diverse medium evidence), `weak`, or `none`
5. Compaction detection checks system prompts for summarization markers; compaction lineage is tracked but does not force sticky continuation
6. Returns `SessionInference` with `session_id`, `affinity_key` (deterministic first-pick hint), confidence, and namespace

**Credential Selection:**

1. `SelectionEngine` receives all credentials for a provider — `src/rotator_library/usage/selection/engine.py`
2. `LimitEngine` filters out credentials at capacity — `src/rotator_library/usage/limits/engine.py`
3. Fair cycle modifier filters exhausted credentials — `src/rotator_library/usage/limits/fair_cycle.py`
4. Strategy (`BalancedStrategy` or `SequentialStrategy`) picks from remaining, using `session_affinity_key` for deterministic placement when evidence is strong enough — `src/rotator_library/usage/selection/strategies/`
5. `SequentialStrategy` maintains sticky entries with TTL-based expiry and max-entry trimming — `src/rotator_library/usage/selection/strategies/sequential.py`

## Key Abstractions

**ProviderInterface:**
- Purpose: Abstract base class defining the contract for all provider plugins
- Location: `src/rotator_library/providers/provider_interface.py`
- Pattern: Abstract base class with singleton metaclass (`SingletonABCMeta`), template method pattern
- Key methods: `get_models()`, `get_model_options()`, `has_custom_logic()`, `get_auth_header()`, `initialize_token()`, `get_background_job_config()`, `get_session_tracking_hints()`

**RotatingClient:**
- Purpose: Slim facade that delegates to modular components for request execution
- Location: `src/rotator_library/client/rotating_client.py`
- Pattern: Facade pattern — ~300 lines delegating to `RequestExecutor`, `CredentialFilter`, `ModelResolver`, `ProviderTransforms`, `StreamingHandler`

**UsageManager:**
- Purpose: Facade for usage tracking, credential selection, and persistence
- Location: `src/rotator_library/usage/manager.py`
- Pattern: Facade + context manager (`CredentialContext`) — composes `TrackingEngine`, `LimitEngine`, `SelectionEngine`, `WindowManager`, `CredentialRegistry`, `UsageStorage`

**RequestContext:**
- Purpose: Immutable data bag carrying all state for a single request attempt
- Location: `src/rotator_library/core/types.py`
- Pattern: Dataclass value object with session tracking fields: `session_id`, `session_affinity_key`, `session_tracker`, `session_possible_compaction`, `session_lineage_parent_id`, `session_tracking_namespace`

**SessionTracker:**
- Purpose: TTL-based session inference using scoped, compounding evidence anchors with confidence scoring
- Location: `src/rotator_library/session_tracking.py`
- Pattern: Evidence accumulator with thread-safe anchor store, namespace isolation, and optional JSON persistence via `ResilientStateWriter`
- Key types: `SessionAnchor` (evidence with strength/source), `SessionTrackingHints` (provider evidence), `SessionInference` (result with confidence + affinity)
- Key methods: `infer_session()`, `record_response()`, `flush()`

## Entry Points

**Proxy Server:**
- Location: `src/proxy_app/main.py`
- Triggers: `python src/proxy_app/main.py` (no args = TUI mode), `--host`, `--port`, `--enable-request-logging`, `--add-credential`
- Responsibilities: Parse args, load `.env` files, configure logging, initialize `RotatingClient`, mount FastAPI routes, start `BackgroundRefresher` and `ModelInfoService`

**TUI Launcher:**
- Location: `src/proxy_app/launcher_tui.py`
- Triggers: Running `main.py` with no arguments
- Responsibilities: Interactive terminal UI for selecting proxy configuration before startup

**Credential Tool:**
- Location: `src/rotator_library/credential_tool.py`
- Triggers: `--add-credential` flag
- Responsibilities: Interactive tool for adding OAuth credentials to the proxy

**Quota Viewer:**
- Location: `src/proxy_app/quota_viewer.py`
- Triggers: Standalone script connecting to running proxy
- Responsibilities: TUI dashboard for viewing credential quotas and usage statistics

## Error Handling

**Strategy:** Classify errors into categories (auth, rate-limit, transient, permanent) and take appropriate action (rotate credential, retry same key, abort)

- Error classification: `src/rotator_library/core/errors.py` — `classify_error()`, `should_rotate_on_error()`, `should_retry_same_key()`
- Error handler with cooldown parsing: `src/rotator_library/error_handler.py` — parses retry-after headers, duration strings, sets provider cooldowns
- Streaming errors: `StreamedAPIError` raised mid-stream to trigger credential rotation
- Credential reauth: `CredentialNeedsReauthError` triggers background OAuth refresh

## Cross-Cutting Concerns

**Logging:** Dual-sink approach — colorized console (INFO+) via `colorlog`, file logging to `logs/proxy.log` (INFO+) and `logs/proxy_debug.log` (DEBUG from `rotator_library` only). LiteLLM logger silenced on console.

**Caching:** Provider instances are singletons via `SingletonABCMeta`. Provider-level HTTP caching via `provider_cache.py`. Model info cached by `ModelInfoService` with async refresh.

**Storage:** JSON file persistence for usage data (`usage/usage_*.json`), OAuth credentials in `oauth_creds/`, transaction logs in `logs/transactions/`. Session state persisted to JSON via `ResilientStateWriter` when disk persistence is enabled. Config via `.env` files and environment variables.

**Background Tasks:** `BackgroundRefresher` manages periodic OAuth token refresh (default 10 min) and provider-specific background jobs (quota refresh, etc.) with independent timers.

**Session Tracking:** Thread-safe (`threading.RLock`), scoped by usage scope/provider/model. Anchor strength levels: `strong` (tool IDs, provider affinity), `medium` (message content hashes, response anchors), `weak` (first-user text, untrusted explicit IDs). Compaction detection tracks lineage without forcing sticky continuation. Configurable via `TRUSTED_SESSION_ID_FIELDS` env var and per-provider `SESSION_STICKY_*` env vars.
