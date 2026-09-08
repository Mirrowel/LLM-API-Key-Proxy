# Configuration Reference

Every knob the proxy reads, organized by subsystem. Configuration is startup-loaded only — no hot reload; a restart applies changes. Environment variables are the final override layer over the JSON config (`ROTATOR_LIBRARY_CONFIG`).

**Addressing conventions:** `<PROVIDER>` in a variable name means the uppercased provider key (`GEMINI`, `OPENAI`, `GEMINI_CLI`, `NVIDIA_NIM`, ...). Names normalize `-` to `_`.

---

## 1. Server & Auth

| Variable | Default | Purpose |
|---|---|---|
| `PROXY_API_KEY` | — | Bearer token clients must send to this proxy. Unset = no auth. |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | Bind address (CLI flags override). |
| `GLOBAL_TIMEOUT` | `600` | Per-request overall timeout (seconds). |
| `REQUEST_LOGGING_ENABLED` | `false` | Master switch for the request/transaction logger (CLI `--enable-request-logging` sets it). |

## 2. Providers & Credentials

| Pattern | Purpose |
|---|---|
| `<NAME>_API_KEY`, `<NAME>_API_KEY_2`, ... | Static API keys for provider `<NAME>`. Numbered suffixes add more credentials. |
| `<NAME>_API_BASE` | Base URL override for provider `<NAME>`; **also creates a dynamic OpenAI-compatible provider** when no built-in `<NAME>` exists. |
| `IGNORE_MODELS_<PROVIDER>` | Comma-separated exclusion list for a provider's models (`*` = all). |
| `WHITELIST_MODELS_<PROVIDER>` | Comma-separated always-include list (overrides the ignore list). |

**OAuth:** credentials live in `oauth_creds/` (local-first). One-time import via `GEMINI_CLI_OAUTH_1` (path to an existing credential file); afterwards only the local directory is read. `--add-credential` runs the interactive importer. `OAUTH_REFRESH_INTERVAL` (default `600`s) paces background token refresh; `SKIP_OAUTH_INITIALIZATION=true` bypasses the startup bootstrap.

**Structured providers:** `ROTATOR_LIBRARY_CONFIG` points to a JSON file (or holds inline JSON) declaring custom providers — `protocol_name` (one of `openai_chat`, `responses`, `anthropic_messages`, `gemini`), `api_base`, `endpoint_paths` (same-origin absolute paths, startup-validated), `auth_mode` (`bearer` | `x-api-key` | `x-goog-api-key` | custom header | `none`), `models`, `adapters`, `field_cache` rules. Credentials NEVER live in this JSON — they come from the env patterns above. `auth_mode: none` gets an internal non-secret credential slot for selection/accounting. See §4 of `00-final-plan.md`.

## 3. Rotation, Concurrency & Cooldowns

| Variable | Default | Purpose |
|---|---|---|
| `ROTATION_MODE_<PROVIDER>` | `sequential` | `balanced` (spread load) or `sequential` (use until exhausted, then next). |
| `MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>` | unlimited | Hard per-key concurrency ceiling. `0`/negative = unlimited. Optional `_<MODE>` suffix scopes to `BALANCED`/`SEQUENTIAL`. |
| `OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>` | unlimited | Soft rotation target for balanced mode. |
| `CONCURRENCY_MULTIPLIER_<PROVIDER>_PRIORITY_<N>` | provider class | Tier multiplier on the hard cap (paid tiers outrank free). |
| `QUOTA_GROUPS_<PROVIDER>_<GROUP>` | — | Models sharing a quota/cooldown window (comma-separated). |
| `SMALL_COOLDOWN_RETRY_THRESHOLD` | see defaults | Retry pacing for short cooldowns. |
| `TRANSIENT_RETRY_DELAY` / `TRANSIENT_RETRY_JITTER` | see defaults | Transient-error retry pacing. |
| `STREAM_RETRY_ON_REASONING_ONLY` | see defaults | Whether reasoning-only stream progress blocks a retry-on-stall. |

**Fair cycle** (sequential mode): each credential exhausts once before reuse. `FAIR_CYCLE_<PROVIDER>` (`true`/`false`), `FAIR_CYCLE_TRACKING_MODE_<PROVIDER>` (`credential` | `model_group`), `FAIR_CYCLE_CROSS_TIER_<PROVIDER>`, `FAIR_CYCLE_DURATION_<PROVIDER>` (default `86400`), `FAIR_CYCLE_EXHAUSTION_THRESHOLD_<PROVIDER>` (default `300`). Custom caps: `CUSTOM_CAP_<PROVIDER>` JSON (`max_requests` + `max_requests_mode` = `absolute`|`percentage`).

**No-reset quota exhaustion** (authoritative quota APIs reporting an exhausted bucket with no reset time): `QUOTA_NO_RESET_EXHAUSTION_POLICY` / `QUOTA_NO_RESET_EXHAUSTION_POLICY_<PROVIDER>` ∈ `warn_only` | `cooldown` | `disable_scope`, plus `QUOTA_NO_RESET_COOLDOWN_SECONDS[_<PROVIDER>]`.

## 4. Routing & Fallback

Target grammar: `provider/model[@execution]`, `provider:profile/model[@execution]` (profile steering only — identity stays provider-level). `execution` ∈ `auto` (default; native when the provider declares a protocol) | `native` | `custom` | `litellm_fallback` (explicit, logged).

| Variable | Purpose |
|---|---|
| `FALLBACK_GROUPS` | Comma-separated group names to define. |
| `FALLBACK_GROUP_<NAME>` | Comma-separated ordered targets for the group. |
| `FALLBACK_GROUP_<NAME>_FAILOVER_ON` | Error classes that advance to the next target (default set in `routing/types.py`). |
| `FALLBACK_GROUP_<NAME>_STOP_ON` | Error classes that abort the chain (may include hard-stop classes). |
| `FALLBACK_GROUP_<NAME>_STREAMING_POLICY` | `pre_output_only` (default) or `never`. |
| `MODEL_ROUTE_<alias>` | Model alias → `provider/model`, `provider:profile/model`, or `group:<name>`. |

The same structures are declarable in the JSON config under `routing.fallback_groups` / `routing.model_routes`; env vars win.

## 5. Session Tracking & Stickiness

| Variable | Default | Purpose |
|---|---|---|
| `TRUSTED_SESSION_ID_FIELDS` | — | Promote explicit request fields to trusted strong session evidence (comma-separated). |
| `SESSION_STICKY_WAIT_SECONDS[_<PROVIDER>]` | `15` | Wait-for-blocked-session-credential window; `0` rotates immediately. |
| `SESSION_STICKY_ENTRY_TTL_SECONDS[_<PROVIDER>]` | `300` | Idle lifetime of session→credential bindings; `0` disables stickiness. |
| `SESSION_STICKY_MAX_ENTRIES[_<PROVIDER>]` | `10000` | Sticky-map size cap. |
| `SESSION_PERSISTENCE_ENABLED` | `false` | Persist inferred session lineage across restarts (schema-3 content-free hashes in `session_stickiness.json`). |
| `SESSION_PERSISTENCE_FLUSH_INTERVAL_SECONDS` | `5.0` | Write throttling (`0` = immediate). |

## 6. Field Cache (provider-protocol state)

Scope: `provider` + `model` required (D11); `credential`/`session` optional refinements; `classifier` fail-closed. Compatibility classes: `bound` fields restore only to the exact provider+model; `portable` fields inherit within declared groups.

| Variable | Purpose |
|---|---|
| `FIELD_CACHE_COMPAT_GROUPS` | JSON object of compatibility groups (`{"open-reasoning": ["glm/glm-5.3", "moonshot/kimi-k3"]}`); unknown pairs are denied by default. |
| `<NAME>_CACHE_REPLAY` | Declarative cache-and-replay rules (D14) for provider `<NAME>` — JSON list of rules (`source`, `path`, `keep` = `last`\|`all`\|`turn`\|`turns:N`\|`per_tool_call`, `inject` = `auto`\|`always`, `scope`, optional `transform` for portable cross-format restore). Compiles to ordinary field-cache rules; provider classes may declare the same surface via the `cache_replay` attribute. |

Rule precedence per name: JSON runtime config > env > provider class > provider-declared rules; same-name replacements may never weaken isolation (scope/compatibility/transform guard).

## 7. Transaction Logging (W12 tiers)

Enabled via `--enable-request-logging` / `REQUEST_LOGGING_ENABLED`.

| Variable | Default | Purpose |
|---|---|---|
| `TRANSACTION_LOG_LEVEL` | `1` | `1` = L1 boundaries + metadata + raw stream chunks; `2` = + intermediates (transform traces); `3` = reserved verbose tier. |
| `TRANSACTION_LOG_RETENTION` | `1000` | Newest N transaction directories kept on disk; `0` = unlimited. |

Capture-on-error is on by default: request-relevant failures (400-class, unexpected crashes) archive buffered intermediates under `capture/`; rotation-class failures (429/quota/auth/timeout) never do. All artifacts are zstd-compressed when `zstandard` is installed. `tools/reconstruct_traces.py` regenerates intermediate traces offline from L1 artifacts.

## 8. Responses API

| Variable | Default | Purpose |
|---|---|---|
| `RESPONSES_STORE_BACKEND` | `memory` | `memory` or `provider_cache` (durable JSON cache). |
| `RESPONSES_STORE_TTL_SECONDS` | see types | Stored-response TTL. |
| `RESPONSES_STORE_MAX_ITEMS` | see types | Bounded store memory. |
| `RESPONSES_STORE_FAILED` | `false` | Persist failed responses (`previous_response_id` chains past failures when true). |

WebSocket Mode serves the same Responses surface at `/v1/responses/ws` (response.create turns, connection-local ZDR continuation, shared sequence domain).

## 9. Native Execution (W11)

Providers declaring `protocol_name` execute natively by default (`auto`); LiteLLM runs only as an explicit, logged fallback (`litellm_fallback` execution identity in transaction metadata). `<NAME>_API_BASE` overrides a built-in provider's native transport base. `deepseek` keeps its verified custom path (`has_custom_logic()` → custom-first). Multi-profile providers (D13) are addressed as `provider:profile/model`; bare names fast-path to the profile matching the client protocol or fail with "endpoint does not exist" — never silent conversion.

## 10. Observability & Misc

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_INFO_REFRESH_INTERVAL` | `3600` | Model catalog refresh cadence. |
| `ADVISORY_MODEL_PRICING` | — | Optional pricing overrides JSON (cost estimates). |
| `SUPPRESS_LITELLM_SERIALIZATION_WARNINGS` | `true` | Silence litellm console noise. |
| `ROTATOR_LIBRARY_FINGERPRINT_KEY` | derived | Cache fingerprint secret. |
| `CHUTES_QUOTA_REFRESH_INTERVAL` / `NANOGPT_QUOTA_REFRESH_INTERVAL` / `FIRMWARE_QUOTA_REFRESH_INTERVAL` | provider defaults | Provider quota-poll cadence. |

Streaming observability (`STREAM_TTFB_TIMEOUT_SECONDS`, stall windows) is documented in `src/rotator_library/config/defaults.py` alongside every tunable default — that module remains the canonical source; this file is the index.
