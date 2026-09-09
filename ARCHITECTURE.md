# Architecture

## Pattern Overview

**Overall:** Layered proxy gateway with plugin-based provider system and intelligent credential rotation

**Key Characteristics:**
- Two-layer separation: FastAPI proxy (`proxy_app`) provides the API surface; `rotator_library` provides all resilience logic
- Protocol-neutral execution: every wire protocol parses once into the neutral canonical model (`src/rotator_library/protocols/`); providers receive only their declared native format, the client response protocol always equals the client request protocol, and same-protocol requests use the pristine client payload as the raw transport basis (`src/rotator_library/native_provider/`) — native execution is the default for providers that declare a native protocol (`auto` mode); LiteLLM is an explicit, logged fallback (`litellm_fallback`), never a silent one
- Plugin-based provider discovery: providers auto-register from `src/rotator_library/providers/` files plus dynamic `*_API_BASE` environment variables
- Singleton providers via `SingletonABCMeta` metaclass — one instance per provider class shared across all components
- Lazy imports at package boundaries (`__getattr__`) to keep startup fast
- All state is per-provider, per-credential, windowed and persisted to JSON files
- Evidence-based session tracking with scoped anchor namespaces, confidence scoring, compaction detection, and deterministic affinity keys — `src/rotator_library/session_tracking.py`

## Layers

**Proxy Application Layer:**
- Purpose: Expose OpenAI Chat Completions, OpenAI Responses, and Anthropic-compatible HTTP endpoints, handle auth, logging, TUI
- Location: `src/proxy_app/`
- Contains: FastAPI app, route handlers, Pydantic request/response models, launcher TUI, quota viewer; `main.py` stays a thin route surface — OAuth credential bootstrap lives in `startup.py` and route-support behaviors (stream framing with in-band terminal error frames, request overrides, embedding fan-out) in `route_helpers.py`
- Depends on: `rotator_library`, `litellm`, `fastapi`, `uvicorn`
- Used by: External API clients (Claude Code, Gemini CLI, OpenAI SDK, curl)

**Client Facade Layer:**
- Purpose: Provide a single `RotatingClient` entry point that orchestrates retries, rotation, and streaming
- Location: `src/rotator_library/client/`
- Contains: `RotatingClient` (facade — `agenerate()` is the protocol-aware entry point, `acompletion()` the raw chat-kwargs seam), `RequestExecutor`, `CredentialFilter`, `ModelResolver`, `ProviderTransforms`, `NeutralStreamPipeline` / `ChatWireStreamAdapter` / `StreamUsageTracker` (`stream_ops.py`), `StreamingHandler` (legacy SSE parsing helpers), `AnthropicHandler` (thin facade — routes the raw `/v1/messages` body through the protocol runtime, stamps the response id for non-streaming calls, traces the boundary, counts tokens locally via the adapter parse → Chat projection), `GeminiHandler`, `RequestContextBuilder` (resolves env-configured routing targets and provider hints for session evidence)
- Depends on: `rotator_library.usage`, `rotator_library.providers`, `rotator_library.protocols`, `rotator_library.core`
- Used by: `proxy_app` via `from rotator_library import RotatingClient`

**Routing & Fallback Layer:**
- Purpose: Resolve model names to ordered execution targets — direct `provider/model` references, env-configured fallback groups, and `provider:profile/model` transport-profile addressing; identity stays provider-level (usage pools, cooldowns, classifiers, session namespaces, and cache provenance key on the bare provider name — the profile only steers transport)
- Location: `src/rotator_library/routing/`
- Contains: `RouteTarget` / `FallbackGroup` / `RoutingDecision` with failover/stop error vocabularies (`types.py`), env parser `parse_route_target()` (`provider/model[@execution]`) and `load_routing_config_from_env()` (`config.py`), `FallbackResolver` (model routes, `group:` aliases, requested-target promotion), `FallbackAttemptRunner` + `FallbackPolicy` (ordered attempts with failover/stop decisions on classified error types), `clone_context_for_target()` (`attempts.py` — per-target context copies that preserve the original for traceability), profile grammar (`profiles.py` — `parse_model_reference()` splits only the provider segment so model names keep their colons; `resolve_profile()` requires explicit profiles to exist and resolves bare names to the default profile or the unique profile matching the client protocol, else fails fast — never silent conversion)
- Depends on: `rotator_library.core` (`RequestContext`)
- Used by: `RequestContextBuilder` (`_resolve_routing_decision()` stamps `routing_targets` on `RequestContext`); `RequestExecutor` executes each target with its `execution` mode

**Protocol Layer:**
- Purpose: Convert between client wire protocols (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Gemini, Ollama, audio/embeddings/images, MCP) and the provider-neutral canonical model — parse once into `UnifiedRequest` / `UnifiedMessage` / `UnifiedResponse` / `UnifiedStreamEvent`, build and format any client protocol from it; the client response protocol always equals the client request protocol
- Location: `src/rotator_library/protocols/`
- Contains: `ProtocolAdapter` base (`base.py`, override-friendly defaults), per-protocol adapters auto-registered in `PROTOCOL_PLUGINS` (`registry.py`), cross-protocol canonical semantics (`canonical.py` — stop reasons (extended Gemini finishReason vocabulary: payload-shape failures map to incomplete, image/model-armor refusals to content_filter), instruction placement and merge records, reasoning-control vocabulary with effort ↔ budget-token mapping plus dynamic thinking (`thinkingBudget: -1`), source-aware passthrough, namespaced tool-choice round-trips (Responses-native variants replay verbatim; foreign targets narrow to the mode with a recorded warning), cache-inclusive usage totals: canonical `input_tokens` already includes cached tokens, so every destination formatter emits the same prompt total from one canonical `Usage` and never adds cached tokens on top — Anthropic folds/unfolds its sibling reporting), destination capability validation (`validation.py` — per-destination hosted-tool allowlists: `responses` accepts its native ToolParam union, `gemini` maps hosted web-search/googleSearch/codeExecution/urlContext/googleMaps server tools and rejects unmapped families), neutral-event stream formatting (`streaming.py` — stable block identity across deltas, refusal deltas, per-stream `sequence` numbers on Responses frames, and same-protocol opaque-state replay gating: `thoughtSignature`/signatures stream back only to the source protocol), operation vocabulary (`operation.py`), shared types (`types.py` — `ToolCall.signature` carries provider thought signatures as D8 opaque state), `litellm_fallback.py` passthrough for LiteLLM-shaped payloads
- Depends on: `rotator_library.core`, `rotator_library.streaming`
- Used by: Client layer (`request_builder.py`, `executor.py`, `stream_ops.py`), native provider executor, Responses layer, proxy error formatting

**Native Provider Execution Layer:**
- Purpose: Execute requests against provider-native HTTP endpoints through protocol adapters; native execution is the default for providers that declare a native protocol (`auto` mode) — LiteLLM remains an explicit, logged fallback. When the client protocol equals the provider protocol and no semantic edits are required, the pristine client wire payload is the transport basis (same-protocol raw fast path) instead of a canonical rebuild
- Location: `src/rotator_library/native_provider/`
- Contains: `NativeProviderExecutor` (`executor.py` — runs protocol/adapter/field-cache passes: adapter chains run on the raw provider wire before field-cache extraction, and cached provider state is injected before the raw payload is sent), `NativeProviderContext` (`context.py` — carries `raw_client_request`, transport overlay records, and the wire-exhaustive `stream_usage_record`), `NativeHTTPTransport` (`http.py`)
- Depends on: `rotator_library.protocols`, `rotator_library.adapters`, `rotator_library.field_cache`
- Used by: Client layer's `RequestExecutor` for the `native` execution mode and for `auto`-mode requests where the provider declares a native protocol

**Provider Plugin Layer:**
- Purpose: Abstract provider-specific behavior (model discovery, auth, transforms, quota tracking, background jobs, session evidence)
- Location: `src/rotator_library/providers/`
- Contains: `ProviderInterface` (ABC) and one file per provider (`*_provider.py`); providers declare native transport as class-level runtime config attributes (`protocol_name`, `adapter_names`, `field_cache_rules`, `native_streaming_supported`, `transport_profiles`, `default_profile`, `cache_replay`) merged with the JSON provider config through `bind_runtime_config()` / `_get_runtime_config()` so operator overrides beat code defaults
- Depends on: `litellm`, provider utility modules
- Used by: Client layer via `PROVIDER_PLUGINS` dict, auto-discovered at import time

**Usage Tracking Layer:**
- Purpose: Track per-credential usage, enforce limits, select credentials, persist state
- Location: `src/rotator_library/usage/`
- Contains: `UsageManager` (facade), `TrackingEngine`, `LimitEngine`, `SelectionEngine`, `WindowManager`, `CredentialRegistry`, `UsageStorage`, `UsageAPI`, `HookDispatcher`
- Depends on: `rotator_library.core`, provider config
- Used by: Client layer for credential selection and usage recording

**Responses API Layer:**
- Purpose: OpenAI Responses API compatibility — create, store, retrieve, and stream response objects with `previous_response_id` continuation
- Location: `src/rotator_library/responses/`
- Contains: `ResponsesService` (orchestrator + `ResponsesServiceError` with string error codes), `ResponsesBridge` (Responses ↔ chat-completions translation via `ResponsesProtocol`; fallback seam for clients without `agenerate` — structured output survives the bridge as a legal chat `response_format`), `ResponsesStore` protocol with `InMemoryResponsesStore` / `ProviderCacheResponsesStore` backends (corrupt cache rows are cache misses, never 500s) and `create_configured_responses_store` factory, `ResponsesSSEFormatter` / `ResponsesStreamEvent` (SSE streaming — monotonic `sequence_number` on every event), `ResponsesWebSocketSession` / `ResponsesWebSocketFormatter` (`websocket.py` — WebSocket Mode: response.create turns, connection-local ZDR continuation cache, spec-shaped error frames), `StoredResponse` / `ResponsesStoreSettings` (incl. `store_failed` policy) / `generate_response_id` (types)
- Depends on: `rotator_library.protocols`, `rotator_library.streaming`, `rotator_library.usage` (costs/accounting), `rotator_library.client` via injected `RotatingClient`
- Used by: Proxy routes `/v1/responses`, `/v1/responses/{response_id}`, `/v1/responses/{response_id}/input_items`

**Core Types & Config Layer:**
- Purpose: Shared type definitions, constants, error classification, config defaults
- Location: `src/rotator_library/core/`, `src/rotator_library/config/`
- Contains: `RequestContext`, `CredentialInfo`, `CustomCapConfig`, `FairCycleConfig`, error classifiers, cooldown constants
- Depends on: Nothing internal (leaf layer)
- Used by: All other layers

## Data Flow

**Chat Completion Request:**

1. Client sends POST to `/v1/chat/completions` — `src/proxy_app/main.py`
2. FastAPI handler calls `client.agenerate()` with `input_protocol="openai_chat"` — `src/proxy_app/main.py`, `src/rotator_library/client/rotating_client.py`
3. `RequestContextBuilder` resolves the input protocol, env-configured routing decision (fallback targets, `provider:profile/model` transport profile), provider hints, and session inference, and builds a `RequestContext` with session affinity key and namespace — `src/rotator_library/client/request_builder.py`
4. `ModelResolver` resolves model name to provider + litellm format — `src/rotator_library/client/models.py`
5. `UsageManager.acquire_credential()` selects best credential via `SelectionEngine` — `src/rotator_library/usage/manager.py`
6. `RequestExecutor` executes with retry/rotation logic, dispatching on the routing target's execution mode: `native` (`NativeProviderExecutor` executes the canonical request against the provider's native protocol — same-protocol requests use the raw client payload as transport basis; under `auto` this is the default whenever the provider declares a native protocol), `custom` (provider plugin `acompletion()`), and `litellm_fallback` (explicit, logged fallback — `_record_litellm_fallback_identity()` stamps the fallback identity in metadata and warns once per request when a native protocol was available); `provider:profile/model` addressing steers the transport profile without changing provider identity — `src/rotator_library/client/executor.py`
7. For streaming, `NeutralStreamPipeline` runs timing/disconnect/heartbeat/usage/session-evidence gates on neutral `UnifiedStreamEvent`s and formats the client-protocol stream exactly once at the tail; `ChatWireStreamAdapter` parses LiteLLM/custom chat-wire chunks into neutral events — `src/rotator_library/client/stream_ops.py`
8. On completion, `UsageManager` records success/failure, `SessionTracker.record_response()` records response-derived anchors — `src/rotator_library/usage/manager.py`, `src/rotator_library/session_tracking.py`

**Anthropic Messages Request:**

1. Client sends POST to `/v1/messages` — `src/proxy_app/main.py`
2. `AnthropicHandler.messages()` routes the raw `/v1/messages` body through `client.agenerate()` with `input_protocol="anthropic_messages"`; the `anthropic_messages` adapter owns validation, and unknown fields plus explicit nulls transport verbatim on the raw fast path — `src/rotator_library/client/anthropic.py`
3. Standard execution flow follows (protocol parse → neutral canonical → provider protocol, execution-mode selection, steps 3–8 above)
4. Response is formatted back to Anthropic Messages format by the `anthropic_messages` protocol adapter — `src/rotator_library/protocols/anthropic_messages.py`
5. `POST /v1/messages/count_tokens` counts locally: the `anthropic_messages` adapter parses the payload, `openai_chat.build_request()` projects it, and `RotatingClient.token_count()` tallies messages plus tools; the route returns protocol-formatted errors, not FastAPI detail envelopes, and marks the result `x-proxy-estimate: local-projection` since the local projection approximates images/PDFs and may count prior-turn thinking the upstream counter ignores — `src/rotator_library/client/anthropic.py`, `src/proxy_app/main.py`

**Responses API Request:**

1. Client sends POST to `/v1/responses` (streaming or non-streaming) — `src/proxy_app/main.py`
2. `ResponsesService.create_response()` / `stream_response()` parses the payload via `ResponsesProtocol`; `_reject_unsupported_lifecycles()` rejects `previous_response_id` + `conversation` (mutually exclusive) and `background` mode (no queued/polling lifecycle) with clear 400s — `src/rotator_library/responses/service.py`
3. `previous_response_id` resolves the parent `StoredResponse` and its lineage from the scoped `ResponsesStore` for continuation; unresolved ids 404 with a hint that continuation must reference responses created through this proxy with store enabled — `src/rotator_library/responses/service.py`, `src/rotator_library/responses/store.py`
4. The request is expanded with parent-lineage items (`_expanded_responses_request()`) and executed through `RotatingClient.agenerate()` with `input_protocol="responses"`; the response returns already in Responses format (client protocol equals request protocol) — `src/rotator_library/responses/service.py`
5. Execution reuses the standard retry/rotation/session-tracking path; usage is recorded via `extract_usage_record` and cost via `CostCalculator` — `src/rotator_library/client/executor.py`, `src/rotator_library/usage/`
6. `ResponsesBridge.to_chat_kwargs()` remains as the fallback seam for clients without `agenerate` (`stream_events()` runs the chat-kwargs path through `client.acompletion()`) — `src/rotator_library/responses/bridge.py`
7. When `store` is true, a `StoredResponse` is persisted scoped by the session isolation key for later retrieval and continuation; failed responses honor the `store_failed` policy on both the create and stream paths — `src/rotator_library/responses/store.py`
8. Streaming requests stream through the canonical runtime while retaining Responses storage, emitting Responses SSE events with monotonic `sequence_number`s (`response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`, `response.output_text.done`, `response.content_part.done`, `response.output_item.done`, `response.completed`) via `ResponsesSSEFormatter`; the bridge fallback path buffers non-text deltas (tool calls, refusal, reasoning) and flushes them as native output items with stable output indexes before the terminal; post-start failures (lineage/parse errors, terminal-less streams) never escape into the transport — the stream ends in a protocol-valid `response.failed` + `[DONE]` sequence (open items close as incomplete first) with a failed `StoredResponse` via `_terminal_stream_failure()` — `src/rotator_library/responses/service.py`, `src/rotator_library/responses/streaming.py`

**Provider Discovery:**

1. `providers/__init__.py` scans all `*_provider.py` files in `src/rotator_library/providers/`
2. Each module's `ProviderInterface` subclass is registered in `PROVIDER_PLUGINS` dict keyed by provider name
3. Environment variables matching `*_API_BASE` create dynamic `DynamicOpenAICompatibleProvider` entries
4. `nvidia_provider.py` is remapped to key `nvidia_nim` to match litellm's naming

**Session Inference:**

1. `RequestContextBuilder` calls `provider.get_session_tracking_hints()` to collect provider-specific evidence — `src/rotator_library/client/request_builder.py`
2. `SessionTracker.infer_session()` builds scoped anchors from explicit IDs, message content, tool-call IDs, and provider hints — `src/rotator_library/session_tracking.py`
3. System/developer prompts are excluded from continuity anchors to prevent harness-level system prompts from cross-binding independent sessions
4. Compaction probe anchors are built separately (`_build_compaction_probe_anchors()`) from early user/system/developer messages only (assistant, tool, and function-result history is never probed); these identify lineage parents but are not stored on the new child session
5. Normal anchors suppress compaction probe indexes to avoid double-counting summary text as continuity evidence
6. Anchors are namespaced by scope/provider/model so sticky evidence never leaks between credential pools
7. `_best_match()` scores anchor overlap against live sessions with comprehensive tiebreaker (score, strong matches, medium matches, group diversity, response matches, last_seen, session_id); confidence is `strong` (any strong match), `probable` (diverse medium evidence), `weak`, or `none`
8. `_evaluate_compaction()` requires structural replacement of more than half the parent's high-water request history (`_retained_history_ratio()` below `_COMPACTION_MAX_RETAINED_HISTORY_RATIO`); unmarked summaries must additionally overlap at least two distinct response events (`_MIN_UNMARKED_RESPONSE_GROUPS`) plus a retained request group, while explicit marker probes qualify on any score > 0
9. Compaction lineage is tracked via `lineage_parent_session_id` on `SessionInference` but does not force sticky continuation of the parent; exact resends of a validated compacted payload reuse the already-created child session via opaque `compaction_replay` anchors (`_find_compaction_replay()` / `_compaction_replay_anchor()`), and changing non-probe history invalidates the replay key; post-compaction requests that retain the validated compacted base context but extend or alter the tail continue the child via `compaction_context` anchors (`_find_compaction_context()` / `_compaction_context_anchor()`) minted only from probe groups that matched parent response evidence; authoritative identity (trusted explicit IDs or provider affinity, via `_is_authoritative_identity_anchor()`) takes precedence over replay/context bindings and suppresses unrelated compaction lineage
10. Returns `SessionInference` with `session_id`, `affinity_key` (deterministic first-pick hint), confidence, and namespace

**Credential Selection:**

1. `SelectionEngine` receives all credentials for a provider — `src/rotator_library/usage/selection/engine.py`
2. `LimitEngine` filters out credentials at capacity — `src/rotator_library/usage/limits/engine.py`
3. Fair cycle modifier filters exhausted credentials — `src/rotator_library/usage/limits/fair_cycle.py`
4. Strategy (`BalancedStrategy` or `SequentialStrategy`) picks from remaining, using `session_affinity_key` for deterministic placement when evidence is strong enough — `src/rotator_library/usage/selection/strategies/`
5. `SequentialStrategy` maintains sticky entries with TTL-based expiry, max-entry trimming, and thread-safe access (`threading.RLock`) — `src/rotator_library/usage/selection/strategies/sequential.py`

## Key Abstractions

**ProviderInterface:**
- Purpose: Abstract base class defining the contract for all provider plugins
- Location: `src/rotator_library/providers/provider_interface.py`
- Pattern: Abstract base class with singleton metaclass (`SingletonABCMeta`), template method pattern
- Key methods: `get_models()`, `get_model_options()`, `has_custom_logic()`, `acompletion()`, `get_auth_header()`, `get_background_job_config()`, `get_session_tracking_hints()`, native hooks (`get_protocol_name()`, `should_use_native_protocol()`, `supports_native_operation()`, `get_native_operation()`, `get_native_endpoint()`, `prepare_native_request()`, `supports_native_streaming()`), adapter and field-cache declarations (`get_adapter_names()`, `get_adapter_config()`, `get_field_cache_rules()`), multi-profile transport (`transport_profiles` / `default_profile` class attributes)

**RotatingClient:**
- Purpose: Slim facade that delegates to modular components for request execution
- Location: `src/rotator_library/client/rotating_client.py`
- Pattern: Facade pattern — ~300 lines delegating to `RequestExecutor`, `CredentialFilter`, `ModelResolver`, `ProviderTransforms`, `NeutralStreamPipeline`; `agenerate()` is the protocol-aware entry (records `_input_protocol`), `acompletion()` executes raw chat kwargs

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
- Pattern: Evidence accumulator with thread-safe anchor store (`threading.RLock` for state, separate `threading.Lock` for save I/O), namespace isolation, schema-versioned JSON persistence via `ResilientStateWriter`
- Key types: `SessionAnchor` (evidence with strength/source/group), `SessionTrackingHints` (provider evidence), `SessionInference` (result with confidence + affinity + lineage), `_MatchCandidate` (scored candidate with `response_groups`/`request_groups`/`matched_probe_groups` tracking and `last_seen` tiebreaker), `_CompactionDecision` (validated parent lineage with retained-history ratio and `context_probe_groups`)
- Key methods: `infer_session()`, `record_response()`, `flush()`, `_build_compaction_probe_anchors()`, `_evaluate_compaction()`, `_find_compaction_replay()`, `_find_compaction_context()`, `_compaction_replay_anchor()`, `_compaction_context_anchor()`, `_is_authoritative_identity_anchor()`, `_compaction_marker_probe_groups()`, `_log_inference_decision()`, `_retained_history_ratio()`, `_prepare_save_locked()`, `_write_save_job()`
- Save I/O: Dirty generation counter (`_dirty_generation`) decouples state mutations from disk writes; `_prepare_save_locked()` snapshots under state lock, `_write_save_job()` writes under separate I/O lock to avoid blocking inference

## Entry Points

**Proxy Server:**
- Location: `src/proxy_app/main.py`
- Triggers: `python src/proxy_app/main.py` (no args = TUI mode), `--host`, `--port`, `--enable-request-logging`, `--add-credential`
- Responsibilities: Parse args, load `.env` files, configure logging, initialize `RotatingClient` (OAuth credential bootstrap via `startup.py`), mount FastAPI routes, start `BackgroundRefresher` and `ModelInfoService`

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
- No-reset quota exhaustion: Authoritative quota APIs that report an exhausted bucket with no reset timestamp (e.g., account lacks model-group entitlement) are handled by `UsageManager._handle_no_reset_quota_exhaustion()` in `src/rotator_library/usage/manager.py` — policy `warn_only` | `cooldown` | `disable_scope` from `ProviderUsageConfig.no_reset_exhaustion_policy` applies a scoped fallback cooldown instead of repeated retries
- Streaming errors: `StreamedAPIError` raised mid-stream to trigger credential rotation
- Credential reauth: `CredentialNeedsReauthError` triggers background OAuth refresh

## Cross-Cutting Concerns

**Logging:** Dual-sink approach — colorized console (INFO+) via `colorlog`, file logging to `logs/proxy.log` (INFO+) and `logs/proxy_debug.log` (DEBUG from `rotator_library` only). LiteLLM logger silenced on console.

**Caching:** Provider instances are singletons via `SingletonABCMeta`. Provider-level HTTP caching via `provider_cache.py`. Model info cached by `ModelInfoService` with async refresh. Provider-protocol state (reasoning content, thought signatures, prompt-cache keys, response IDs) is cached and re-injected by the field cache (`src/rotator_library/field_cache/`) on the native execution path only — declarative `cache_replay` rules compile to ordinary `FieldCacheRule`s, bound fields restore only to the exact provider+model that produced them while portable fields inherit within declared compatibility groups, and cross-format restores run through named transforms (`src/rotator_library/protocols/transforms.py`); rules are identity-normalized to the bare provider but every `FieldCacheOperation` records the transport profile that served the request so cross-profile sharing stays visible per operation and trace.

**Storage:** JSON file persistence for usage data (`usage/usage_*.json`), OAuth credentials in `oauth_creds/`, transaction logs in `logs/transactions/` written by `TransactionLogger` (`src/rotator_library/transaction_logger.py`) with per-request directories containing client/provider I/O and a JSON-safe payload converter (`_make_json_safe`) for Pydantic/dataclass/timestamp objects. Transaction logging is leveled by `TRANSACTION_LOG_LEVEL` (1 = boundaries + metadata, default; 2 = + intermediates; 3 = verbose per-frame) with artifacts zstd-compressed via `utils/zstd_io.py` when `zstandard` is installed (plain files with a metadata flag otherwise), L1 disk usage bounded by `TRANSACTION_LOG_RETENTION` (newest N directories kept), request-related failures archiving buffered intermediates to `capture/captured_trace.json` while rotation-class failures (rate-limit, quota, auth, timeout) do not, and `tools/reconstruct_traces.py` regenerating L2-style intermediates offline from L1 artifacts. Session state persisted to JSON via `ResilientStateWriter` when disk persistence is enabled. Config via `.env` files and environment variables.

**Background Tasks:** `BackgroundRefresher` manages periodic OAuth token refresh (default 10 min) and provider-specific background jobs (quota refresh, etc.) with independent timers.

**Session Tracking:** Thread-safe with two-lock design (`threading.RLock` for anchor store, `threading.Lock` for save I/O), scoped by usage scope/provider/model. Anchor strength levels: `strong` (trusted explicit IDs, provider affinity keys, response global IDs), `medium` (message content hashes, response anchors), `weak` (first-user text, raw tool-call IDs, untrusted explicit IDs). System/developer prompts excluded from continuity anchors. Compaction detection uses separate probe anchors (`_build_compaction_probe_anchors()`) restricted to early user/system/developer messages and requires structural replacement of more than half the parent's high-water request history via `_evaluate_compaction()`; unmarked summaries must additionally overlap at least two distinct response events plus a retained request group; authoritative identity (trusted explicit or provider, via `_is_authoritative_identity_anchor()`) suppresses unrelated compaction lineage; exact resends of a validated compacted payload reuse the child session via opaque `compaction_replay` anchors while changed-tail continuations bind via `compaction_context` anchors (`_find_compaction_context()`) minted only from probe groups that matched parent response evidence. Lineage tracked via `lineage_parent_session_id` without forcing sticky continuation of the parent. Trimming and TTL pruning evict weak/ordinary evidence before replay/context identity (`_anchor_eviction_key()` ranks `compaction_context` and `compaction_replay` sources above ordinary anchors) with deterministic value tie-breaking, and late responses cannot resurrect an expired session; session namespaces are immutable — `_refresh_and_bridge()` rejects namespace drift and `record_response()` normalizes fallback callbacks to the session's original namespace. Streaming response identity is recorded only after an explicit provider completion signal (`_sse_has_completion_signal()`: usage-backed final chunk, `finish_reason` paired with usage, or `[DONE]`). Persistence uses schema-versioned JSON (`_PERSISTENCE_SCHEMA_VERSION`) with generation-based write deduplication, dirty state retained on failed writes for retry, stale delayed generations rejected, and anchor ownership rebuilt on load (rejecting malformed containers, non-finite timestamps, expired sessions, orphan anchors, namespace mismatches, invalid strengths, and unsupported schemas); rebuilt sessions are flagged `loaded_from_persistence` for lineage diagnostics. Configurable via `TRUSTED_SESSION_ID_FIELDS` env var, per-provider `SESSION_STICKY_*` env vars, and `SESSION_PERSISTENCE_ENABLED` / `SESSION_PERSISTENCE_FLUSH_INTERVAL_SECONDS` env vars on `RotatingClient`; every inference emits a temporary warning-level `Session tracker decision` line (`_log_inference_decision()`) with action, session IDs, namespace, confidence, score, persistence origin, and compaction evidence.
