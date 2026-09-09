# Technical Documentation: Universal LLM API Proxy & Resilience Library

This document provides a detailed technical explanation of the project's architecture, internal components, and data flows. It is intended for developers who want to understand how the system achieves high availability and resilience.

## 1. Architecture Overview

The project is a monorepo containing two primary components:

1.  **The Proxy Application (`proxy_app`)**: This is the user-facing component. It's a FastAPI application that acts as a universal gateway. It uses `litellm` to translate requests to various provider formats and includes:
    *   **Batch Manager**: Optimizes high-volume embedding requests.
    *   **Detailed Logger**: Provides per-request file logging for debugging.
    *   **OpenAI-Compatible Endpoints**: `/v1/chat/completions`, `/v1/embeddings`, etc.
    *   **Anthropic-Compatible Endpoints**: `/v1/messages`, `/v1/messages/count_tokens` for Claude Code and other Anthropic API clients.
    *   **Model Filter GUI**: Visual interface for configuring model ignore/whitelist rules per provider (see Section 6).
2.  **The Resilience Library (`rotator_library`)**: This is the core engine that provides high availability. It is consumed by the proxy app to manage a pool of API keys, handle errors gracefully, and ensure requests are completed successfully even when individual keys or provider endpoints face issues.

This architecture cleanly separates the API interface from the resilience logic, making the library a portable and powerful tool for any application needing robust API key management.

---

## 2. `rotator_library` - The Resilience Engine

This library is the heart of the project, containing all the logic for managing a pool of API keys, tracking their usage, and handling provider interactions to ensure application resilience.

### 2.1. `client/rotating_client.py` - The `RotatingClient`

The `RotatingClient` is the central class that orchestrates all operations. It is now a slim facade that delegates to modular components (executor, filters, transforms) while remaining a long-lived, async-native object.

#### Initialization

The client is initialized with your provider API keys, retry settings, and a new `global_timeout`.

```python
client = RotatingClient(
    api_keys=api_keys,
    oauth_credentials=oauth_credentials,
    max_retries=2,
    usage_file_path="usage.json",
    configure_logging=True,
    global_timeout=30,
    abort_on_callback_error=True,
    litellm_provider_params={},
    ignore_models={},
    whitelist_models={},
    enable_request_logging=False,
    max_concurrent_requests_per_key={}
)
```

-   `api_keys` (`Optional[Dict[str, List[str]]]`, default: `None`): A dictionary mapping provider names to a list of API keys.
-   `oauth_credentials` (`Optional[Dict[str, List[str]]]`, default: `None`): A dictionary mapping provider names to a list of file paths to OAuth credential JSON files.
-   `max_retries` (`int`, default: `2`): The number of times to retry a request with the *same key* if a transient server error occurs.
-   `usage_file_path` (`str`, optional): Base path for usage persistence (defaults to `usage/` in the data directory). The client stores per-provider files under `usage/usage_<provider>.json`.
-   `configure_logging` (`bool`, default: `True`): If `True`, configures the library's logger to propagate logs to the root logger.
-   `global_timeout` (`int`, default: `30`): A hard time limit (in seconds) for the entire request lifecycle.
-   `abort_on_callback_error` (`bool`, default: `True`): If `True`, any exception raised by `pre_request_callback` will abort the request.
-   `litellm_provider_params` (`Optional[Dict[str, Any]]`, default: `None`): Extra parameters to pass to `litellm` for specific providers.
-   `ignore_models` (`Optional[Dict[str, List[str]]]`, default: `None`): Blacklist of models to exclude (supports wildcards).
-   `whitelist_models` (`Optional[Dict[str, List[str]]]`, default: `None`): Whitelist of models to always include, overriding `ignore_models`.
-   `enable_request_logging` (`bool`, default: `False`): If `True`, enables detailed per-request file logging.
-   `max_concurrent_requests_per_key` (`Optional[Dict[str, int]]`, default: `None`): Max concurrent requests allowed for a single API key per provider.
-   `rotation_tolerance` (`float`, default: `3.0`): Controls the credential rotation strategy. See Section 2.2 for details.

#### Core Responsibilities

*   **Lifecycle Management**: Manages a shared `httpx.AsyncClient` for all non-blocking HTTP requests.
*   **Key Management**: Interfacing with the `UsageManager` to acquire and release API keys based on load and health.
*   **Plugin System**: Dynamically loading and using provider-specific plugins from the `providers/` directory.
*   **Execution Logic**: Executing API calls via `litellm` with a robust, **deadline-driven** retry and key selection strategy.
*   **Streaming Safety**: Providing a safe, stateful wrapper (`_safe_streaming_wrapper`) for handling streaming responses, buffering incomplete JSON chunks, and detecting mid-stream errors.
*   **Model Filtering**: Filtering available models using configurable whitelists and blacklists.
*   **Classifier-Scoped Routing**: Resolving per-request or registered classifier scopes so user-owned provider keys do not mix with platform/default pools.
*   **Request Sanitization**: Automatically cleaning invalid parameters (like `dimensions` for non-OpenAI models) via `request_sanitizer.py`.

#### Model Filtering Logic

The `RotatingClient` provides fine-grained control over which models are exposed via the `/v1/models` endpoint. This is handled by the `get_available_models` method.

The logic applies in the following order:
1.  **Whitelist Check**: If a provider has a whitelist defined (`WHITELIST_MODELS_<PROVIDER>`), any model on that list will **always be available**, even if it matches a blacklist pattern. This acts as a definitive override.
2.  **Blacklist Check**: For any model *not* on the whitelist, the client checks the blacklist (`IGNORE_MODELS_<PROVIDER>`). If the model matches a blacklist pattern (supports wildcards like `*-preview`), it is excluded.
3.  **Default**: If a model is on neither list, it is included.

For classifier-scoped model discovery, global filters do not propagate by default. Scoped callers can pass `model_filters` to `get_available_models()` or `get_all_available_models()`; these filters are applied only to that classified model-listing operation and do not block completions.

#### Classifier-Scoped Routing

Classifier-scoped routing is the multi-tenant isolation layer in `RotatingClient`. It lets one long-lived client serve both global/platform models and user-owned provider credentials without constructing a separate `RotatingClient` per user.

The isolation key is:

```text
classifier + provider
```

No classifier preserves the original behavior:

```text
provider -> global/default credential pool -> usage/usage_<provider>.json
```

With a classifier, the resolver builds an effective scope from:

```text
global/default provider templates
+ registered classifier state
+ request overlay
= effective request scope
```

Credential resolution for classified requests is intentionally stricter:

```text
request api_keys[provider]
or registered classifier credentials[provider]
or no credentials
```

Global/default API keys are never inherited by classified requests. This is the primary guard that prevents a user-owned model request from accidentally charging or exposing platform credentials.

Provider config resolution is more permissive:

```text
request providers[provider]
or registered classifier provider config
or global/default provider API base/template
or built-in provider behavior
```

This allows a classifier to use a globally known custom provider alias such as `logfare` while still supplying only user-owned keys.

The scoped request metadata is carried on `RequestContext`:

```text
usage_manager_key     -> lookup key for classifier/provider usage manager
provider_config       -> request-local provider base URL/protocol override
credential_secrets    -> safe credential id -> raw secret map for this request
classifier            -> external isolation label
```

The executor continues to rotate over `context.credentials`, but for private scoped credentials those values are safe identifiers such as `private:<fingerprint>`. The raw secret is resolved from `context.credential_secrets` only when injecting `api_key` or `credential_identifier` into the provider call.

#### Stateless Scoped Calls

Stateless scoped calls pass everything needed for the operation:

```python
await client.acompletion(
    model="logfare/my-model",
    messages=[{"role": "user", "content": "Hello"}],
    classifier="user_123",
    api_keys={"logfare": ["user-key"]},
    providers={"logfare": {"base_url": "https://logfare.example/v1"}},
    private=True,
)
```

The call uses only the `logfare` keys supplied for the operation. Keys for unrelated providers in the `api_keys` dict are ignored.

#### Registered Scoped Calls

Registered scope state is held in memory and can be managed externally:

```python
await client.register_scope(
    "user_123",
    providers={"logfare": {"base_url": "https://logfare.example/v1"}},
    api_keys={"logfare": ["user-key"]},
    private=True,
)

await client.acompletion(
    model="logfare/my-model",
    messages=[{"role": "user", "content": "Hello"}],
    classifier="user_123",
)
```

Registered management methods include whole-scope operations and provider/credential operations:

```text
register_scope, update_scope, get_scope, remove_scope
add_scope_provider, update_scope_provider, remove_scope_provider, list_scope_providers
add_scope_credentials, set_scope_credentials, remove_scope_credentials, list_scope_credentials
```

`get_scope()` and `list_scope_credentials()` hide secrets unless `include_secrets=True` is requested. The library does not persist registered secrets; host applications remain responsible for user identity, authorization, encrypted durable storage, and billing.

#### Private Credential Persistence

For `private=True`, raw API keys are converted to HMAC fingerprints before entering usage tracking:

```text
private:<HMAC-SHA256 fingerprint>
```

The HMAC key is `ROTATOR_LIBRARY_FINGERPRINT_KEY` when set, otherwise the resolved data directory path. Private usage files contain only the safe identifier, not the raw API key.

Private credential persistence behavior:

```text
usage/classifiers/<safe_classifier>/usage_<provider>.json
credentials[*].accessor = private:<fingerprint>
credentials[*].private = true
accessor_index excludes private credentials
stats credentials[*].full_path = null
```

Legacy/global non-private credentials keep the previous persistence shape for backward compatibility.

#### Scoped Model Discovery And Stats

`get_available_models()` and `get_all_available_models()` accept the same scope arguments as requests: `classifier`, `api_keys`, `providers`, `private`, plus `model_filters` and `force_refresh`.

Cache keys include classifier, provider, provider override, and model filters so different users can reuse the same provider name without sharing model discovery results.

`get_quota_stats(classifier=...)` selects only usage managers under that classifier. Without `classifier`, stats skip classifier-scoped managers to preserve previous global/default behavior.

#### Request Lifecycle: A Deadline-Driven Approach

The request lifecycle has been designed around a single, authoritative time budget to ensure predictable performance:

1.  **Deadline Establishment**: The moment `acompletion` or `aembedding` is called, a `deadline` is calculated: `time.time() + self.global_timeout`. This `deadline` is the absolute point in time by which the entire operation must complete.
2.  **Deadline-Aware Key Selection**: The main loop checks this deadline before every key acquisition attempt. If the deadline is exceeded, the request fails immediately.
3.  **Deadline-Aware Key Acquisition**: The `UsageManager` itself takes this `deadline`. It will only wait for a key (if all are busy) until the deadline is reached.
4.  **Deadline-Aware Retries**: If a transient error occurs (like a 500 or 429), the client calculates the backoff time. If waiting would push the total time past the deadline, the wait is skipped, and the client immediately rotates to the next key.

#### Streaming Resilience

The `_safe_streaming_wrapper` is a critical component for stability. It:
*   **Buffers Fragments**: Reads raw chunks from the stream and buffers them until a valid JSON object can be parsed. This handles providers that may split JSON tokens across network packets.
*   **Error Interception**: Detects if a chunk contains an API error (like a quota limit) instead of content, and raises a specific `StreamedAPIError`.
*   **Quota Handling**: If a specific "quota exceeded" error is detected mid-stream multiple times, it can terminate the stream gracefully to prevent infinite retry loops on oversized inputs.

### 2.2. `usage/manager.py` - Stateful Concurrency & Usage Management

This class is the stateful core of the library, managing concurrency, usage tracking, cooldowns, and quota resets. Usage tracking now lives in the `rotator_library/usage/` package with per-provider managers and `usage/usage_<provider>.json` storage.

#### Key Concepts

*   **Async-Native & Lazy-Loaded**: Fully asynchronous, using `aiofiles` for non-blocking file I/O. Usage data is loaded only when needed.
*   **Fine-Grained Locking**: Each API key has its own `asyncio.Lock` and `asyncio.Condition`. This allows for highly granular control.
*   **Multiple Reset Modes**: Supports three reset strategies:
    - **per_model**: Each model has independent usage window with authoritative `quota_reset_ts` (from provider errors)
    - **credential**: One window per credential with custom duration (e.g., 5 hours, 7 days)
    - **daily**: Legacy daily reset at `daily_reset_time_utc`
*   **Model Quota Groups**: Models can be grouped to share quota limits. When one model in a group hits quota, all receive the same reset timestamp.

#### Capacity-Phase Key Acquisition Strategy

The `acquire_key` method uses a sophisticated strategy to balance load:

1.  **Filtering**: Keys currently on cooldown (global or model-specific) are excluded.
2.  **Rotation Mode**: Determines credential selection strategy:
    *   **Sequential Mode** (default): Reuses the selected credential until it errors/exhausts to preserve provider-side cache locality
    *   **Balanced Mode**: Distributes requests across credentials for even load when explicitly configured
3.  **Capacity Phases**: Valid keys are first filtered by hard blockers, then grouped by soft concurrency capacity:
    *   **Below Optimal**: Healthy credentials with `active_requests < optimal_concurrent` are preferred.
    *   **Stacking Fallback**: If every healthy credential is at or above `optimal_concurrent`, credentials remain usable until they hit `max_concurrent`.
    *   **Blocked**: Credentials on cooldown, over quota, disallowed for the model, or at/above a positive `max_concurrent` are not selected.
4.  **Selection Strategy** (configurable via `rotation_tolerance`):
    *   **Deterministic (tolerance=0.0)**: Within the current capacity phase, keys are sorted by daily usage count and the least-used key is always selected. This provides perfect load balance but predictable patterns.
    *   **Weighted Random (tolerance>0, default)**: Keys are selected randomly with weights biased toward less-used ones:
        - Formula: `weight = (max_usage - credential_usage) + tolerance + 1`
        - `tolerance=2.0` (recommended): Balanced randomness - credentials within 2 uses of the maximum can still be selected with reasonable probability
        - `tolerance=5.0+`: High randomness - even heavily-used credentials have significant probability
        - **Security Benefit**: Unpredictable selection patterns make rate limit detection and fingerprinting harder
        - **Load Balance**: Lower-usage credentials still preferred, maintaining reasonable distribution
5.  **Concurrency Controls**: `optimal_concurrent` is a soft spread-before-stacking target. `max_concurrent` is the hard safety ceiling (with priority multipliers applied); values `<= 0` mean unlimited.
6.  **Priority Groups**: When credential prioritization is enabled, higher-tier credentials (lower priority numbers) are tried first before moving to lower tiers.

Balanced mode defaults to `optimal_concurrent=1` and `max_concurrent=-1`, which spreads first but does not artificially block when all credentials are busy. Sequential mode defaults to `optimal_concurrent=-1` and `max_concurrent=-1`, preserving sticky behavior unless a provider or environment override constrains it. Provider-wide and mode-specific environment variables are available as `OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>`, `MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>`, and the `_BALANCED`/`_SEQUENTIAL` variants.

#### Failure Handling & Cooldowns

*   **Escalating Backoff**: When a failure occurs, the key gets a temporary cooldown for that specific model. Consecutive failures increase this time (10s -> 30s -> 60s -> 120s).
*   **Key-Level Lockouts**: If a key accumulates failures across multiple distinct models (3+), it is assumed to be dead/revoked and placed on a global 5-minute lockout.
*   **Authentication Errors**: Immediate 5-minute global lockout.
*   **Quota Exhausted Errors**: When a provider returns a quota exhausted error with an authoritative reset timestamp:
    - The `quota_reset_ts` is extracted from the error response (via provider's `parse_quota_error()` method)
    - Applied to the affected model (and all models in its quota group if defined)
    - Cooldown preserved even during daily/window resets until the actual quota reset time
    - Logs show the exact reset time in local timezone with ISO format

### 2.3. `batch_manager.py` - Efficient Request Aggregation

The `EmbeddingBatcher` class optimizes high-throughput embedding workloads.

*   **Mechanism**: It uses an `asyncio.Queue` to collect incoming requests.
*   **Triggers**: A batch is dispatched when either:
    1.  The queue size reaches `batch_size` (default: 64).
    2.  A time window (`timeout`, default: 0.1s) elapses since the first request in the batch.
*   **Efficiency**: This reduces dozens of HTTP calls to a single API request, significantly reducing overhead and rate limit usage.

### 2.4. `background_refresher.py` - Automated Token Maintenance & Provider Jobs

The `BackgroundRefresher` manages background tasks for the proxy, including OAuth token refresh and provider-specific periodic jobs.

#### OAuth Token Refresh

*   **Periodic Checks**: It runs a background task that wakes up at a configurable interval (default: 600 seconds/10 minutes via `OAUTH_REFRESH_INTERVAL`).
*   **Proactive Refresh**: It iterates through all loaded OAuth credentials and calls their `proactively_refresh` method to ensure tokens are valid before they are needed.

#### Provider-Specific Background Jobs

Providers can define their own background jobs that run on independent schedules:

*   **Independent Timers**: Each provider's job runs on its own interval, separate from the OAuth refresh cycle.
*   **Configuration**: Providers implement `get_background_job_config()` to define their job settings.
*   **Execution**: Providers implement `run_background_job()` to execute the periodic task.

**Provider Job Configuration:**
```python
def get_background_job_config(self) -> Optional[Dict[str, Any]]:
    """Return configuration for provider-specific background job."""
    return {
        "interval": 300,      # seconds between runs
        "name": "quota_refresh",  # for logging
        "run_on_start": True,  # whether to run immediately at startup
    }

async def run_background_job(
    self,
    usage_manager: "UsageManager",
    credentials: List[str],
) -> None:
    """Execute the provider's periodic background job."""
    # Provider-specific logic here
    pass
```

**Current Provider Jobs:**

| Provider | Job Name | Default Interval | Purpose |
|----------|----------|------------------|---------|
| Gemini CLI | `gemini_cli_quota_refresh` | 300s (5 min) | Fetches quota status from `retrieveUserQuota` API to update remaining quota estimates |

### 2.6. Credential Management Architecture

The `CredentialManager` class (`credential_manager.py`) centralizes the lifecycle of all API credentials. It adheres to a "Local First" philosophy.

#### 2.6.1. Automated Discovery & Preparation

On startup (unless `SKIP_OAUTH_INIT_CHECK=true`), the manager performs a comprehensive sweep:

1. **System-Wide Scan**: Searches for OAuth credential files in standard locations:
   - `~/.gemini/` → All `*.json` files (typically `credentials.json`)

2. **Local Import**: Valid credentials are **copied** (not moved) to the project's `oauth_creds/` directory with standardized names:
   -  `gemini_cli_oauth_1.json`, `gemini_cli_oauth_2.json`, etc.

3. **Intelligent Deduplication**: 
   - The manager inspects each credential file for a `_proxy_metadata` field containing the user's email or ID
   - If this field doesn't exist, it's added during import using provider-specific APIs (e.g., fetching Google account email for Gemini)
   - Duplicate accounts (same email/ID) are detected and skipped with a warning log
   - Prevents the same account from being added multiple times, even if the files are in different locations

4. **Isolation**: The project's credentials in `oauth_creds/` are completely isolated from system-wide credentials, preventing cross-contamination

#### 2.6.2. Credential Loading & Stateless Operation

The manager supports loading credentials from two sources, with a clear priority:

**Priority 1: Local Files** (`oauth_creds/` directory)
- Standard `.json` files are loaded first
- Naming convention: `{provider}_oauth_{number}.json`
- Example: `oauth_creds/gemini_cli_oauth_1.json`

**Priority 2: Environment Variables** (Stateless Deployment)
- If no local files are found, the manager checks for provider-specific environment variables
- This is the key to "Stateless Deployment" for platforms like Railway, Render, Heroku
- Credentials are referenced internally using `env://` URIs (e.g., `env://gemini_cli/1`)

**Gemini CLI Environment Variables:**

Single credential (legacy format):
```
GEMINI_CLI_ACCESS_TOKEN
GEMINI_CLI_REFRESH_TOKEN
GEMINI_CLI_EXPIRY_DATE
GEMINI_CLI_EMAIL
GEMINI_CLI_PROJECT_ID (optional)
GEMINI_CLI_TIER (optional: standard-tier or free-tier)
```

Multiple credentials (use `_N_` suffix where N is 1, 2, 3...):
```
GEMINI_CLI_1_ACCESS_TOKEN
GEMINI_CLI_1_REFRESH_TOKEN
GEMINI_CLI_1_EXPIRY_DATE
GEMINI_CLI_1_EMAIL
GEMINI_CLI_1_PROJECT_ID (optional)
GEMINI_CLI_1_TIER (optional)

GEMINI_CLI_2_ACCESS_TOKEN
GEMINI_CLI_2_REFRESH_TOKEN
...
```

**How it works:**
- If the manager finds (e.g.) `GEMINI_CLI_ACCESS_TOKEN` or `GEMINI_CLI_1_ACCESS_TOKEN`, it constructs an in-memory credential object that mimics the file structure
- The credential is referenced internally as `env://gemini_cli/0` (legacy) or `env://gemini_cli/1` (numbered)
- The credential behaves exactly like a file-based credential (automatic refresh, expiry detection, etc.)
- No physical files are created or needed on the host system
- Perfect for ephemeral containers or read-only filesystems

**env:// URI Format:**
```
env://{provider}/{index}

Examples:
- env://gemini_cli/1  → GEMINI_CLI_1_ACCESS_TOKEN, etc.
- env://gemini_cli/0  → GEMINI_CLI_ACCESS_TOKEN (legacy single credential)
```

#### 2.6.3. Credential Tool Integration

The `credential_tool.py` provides a user-friendly CLI interface to the `CredentialManager`:

**Key Functions:**
1. **OAuth Setup**: Wraps provider-specific auth classes to handle interactive login flows
2. **Credential Export**: Reads local `.json` files and generates `.env` format output for stateless deployment
3. **API Key Management**: Adds or updates `PROVIDER_API_KEY_N` entries in the `.env` file

---

### 2.7. Request Sanitizer (`request_sanitizer.py`)

The `sanitize_request_payload` function ensures requests are compatible with each provider's specific requirements:

**Parameter Cleaning Logic:**

1. **`dimensions` Parameter**:
   - Only supported by OpenAI's `text-embedding-3-small` and `text-embedding-3-large` models
   - Automatically removed for all other models to prevent `400 Bad Request` errors

2. **`thinking` Parameter** (Gemini-specific):
   - Format: `{"type": "enabled", "budget_tokens": -1}`
   - Only valid for `gemini/gemini-2.5-pro` and `gemini/gemini-2.5-flash`
   - Removed for all other models

---

### 2.8. Error Classification (`error_handler.py`)

The `ClassifiedError` class wraps all exceptions from `litellm` and categorizes them for intelligent handling:

**Error Types:**
```python
class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"           # 429 errors, temporary backoff needed
    AUTHENTICATION = "authentication"    # 401/403, invalid/revoked key
    SERVER_ERROR = "server_error"       # 500/502/503, provider infrastructure issues
    QUOTA = "quota"                      # Daily/monthly quota exceeded
    CONTEXT_LENGTH = "context_length"    # Input too long for model
    CONTENT_FILTER = "content_filter"    # Request blocked by safety filters
    NOT_FOUND = "not_found"              # Model/endpoint doesn't exist
    TIMEOUT = "timeout"                  # Request took too long
    UNKNOWN = "unknown"                  # Unclassified error
```

**Classification Logic:**

1. **Status Code Analysis**: Primary classification method
   - `401`/`403` → `AUTHENTICATION`
   - `429` → `RATE_LIMIT`
   - `400` with "context_length" or "tokens" → `CONTEXT_LENGTH`
   - `400` with "quota" → `QUOTA`
   - `500`/`502`/`503` → `SERVER_ERROR`

2. **Special Exception Types**:
   - `EmptyResponseError` → `SERVER_ERROR` (status 503, rotatable)
   - `TransientQuotaError` → `SERVER_ERROR` (status 503, rotatable - bare 429 without retry info)

3. **Message Analysis**: Fallback for ambiguous errors
   - Searches for keywords like "quota exceeded", "rate limit", "invalid api key"

4. **Provider-Specific Overrides**: Some providers use non-standard error formats

**Usage in Client:**
- `AUTHENTICATION` → Immediate 5-minute global lockout
- `RATE_LIMIT`/`QUOTA` → Escalating per-model cooldown
- `SERVER_ERROR` → Retry with same key (up to `max_retries`), then rotate
- `CONTEXT_LENGTH`/`CONTENT_FILTER` → Immediate failure (user needs to fix request)

---

### 2.9. Cooldown Management (`cooldown_manager.py`)

The `CooldownManager` handles IP or account-level rate limiting that affects all keys for a provider:

**Purpose:**
- Some providers (like NVIDIA NIM) have rate limits tied to account/IP rather than API key
- When a 429 error occurs, ALL keys for that provider must be paused

**Key Methods:**

1. **`is_cooling_down(provider: str) -> bool`**:
   - Checks if a provider is currently in a global cooldown period
   - Returns `True` if the current time is still within the cooldown window

2. **`start_cooldown(provider: str, duration: int)`**:
   - Initiates or extends a cooldown for a provider
   - Duration is typically 60-120 seconds for 429 errors

3. **`get_cooldown_remaining(provider: str) -> float`**:
   - Returns remaining cooldown time in seconds
   - Used for logging and diagnostics

**Integration with UsageManager:**
- When a key fails with `RATE_LIMIT` error type, the client checks if it's likely an IP-level limit
- If so, `CooldownManager.start_cooldown()` is called for the entire provider
- All subsequent `acquire_key()` calls for that provider will wait until the cooldown expires


### 2.10. Credential Prioritization System (`client/rotating_client.py` & `usage/manager.py`)

The library now includes an intelligent credential prioritization system that automatically detects credential tiers and ensures optimal credential selection for each request.

**Key Concepts:**

- **Provider-Level Priorities**: Providers can implement `get_credential_priority()` to return a priority level (1=highest, 10=lowest) for each credential
- **Model-Level Requirements**: Providers can implement `get_model_tier_requirement()` to specify minimum priority required for specific models
- **Automatic Filtering**: The client automatically filters out incompatible credentials before making requests
- **Priority-Aware Selection**: The `UsageManager` prioritizes higher-tier credentials (lower numbers) within the same priority group

**Implementation Example (Gemini CLI):**

```python
def get_credential_priority(self, credential: str) -> Optional[int]:
    """Returns priority based on Gemini tier."""
    tier = self.project_tier_cache.get(credential)
    if not tier:
        return None  # Not yet discovered
    
    # Paid tiers get highest priority
    if tier not in ['free-tier', 'legacy-tier', 'unknown']:
        return 1
    
    # Free tier gets lower priority
    if tier == 'free-tier':
        return 2
    
    return 10

def get_model_tier_requirement(self, model: str) -> Optional[int]:
    """Returns minimum priority required for model."""
    if model.startswith("gemini-3-"):
        return 1  # Only paid tier (priority 1) credentials
    
    return None  # All other models have no restrictions
```

**Provider Support:**

The following providers implement credential prioritization:

- **Gemini CLI**: Paid tier (priority 1), Free tier (priority 2), Legacy/Unknown (priority 10). Gemini 3 models require paid tier.

**Usage Manager Integration:**

The `acquire_key()` method has been enhanced to:
1. Group credentials by priority level
2. Try highest priority group first (priority 1, then 2, etc.)
3. Within each group, use existing tier1/tier2 logic (idle keys first, then busy keys)
4. Load balance within priority groups by usage count
5. Only move to next priority if all higher-priority credentials are exhausted

**Benefits:**

- Ensures paid-tier credentials are always used for premium models
- Prevents failed requests due to tier restrictions
- Optimal cost distribution (free tier used when possible, paid when required)
- Graceful fallback if primary credentials are unavailable

---

### 2.11. Provider Cache System (`providers/provider_cache.py`)

A modular, shared caching system for providers to persist conversation state across requests.

**Architecture:**

- **Dual-TTL Design**: Short-lived memory cache (default: 1 hour) + longer-lived disk persistence (default: 24 hours)
- **Background Persistence**: Batched disk writes every 60 seconds (configurable)
- **Automatic Cleanup**: Background task removes expired entries from memory cache

### 2.15. TransientQuotaError (`error_handler.py`)

A new error type for handling bare 429 responses without retry timing information.

**When Raised:**
- Provider returns HTTP 429 status code
- Response doesn't contain retry timing info (no `quotaResetTimeStamp` or `retryDelay`)
- After internal retry attempts are exhausted

**Behavior:**
- Classified as `server_error` (status 503) rather than quota exhaustion
- Causes credential rotation to try the next credential
- Does NOT trigger long-term quota cooldowns

**Rationale:**
Some 429 responses are transient rate limits rather than true quota exhaustion. These occur when the API is temporarily overloaded but the credential still has quota available. Retrying internally before rotating credentials provides better resilience.

### 2.16. Gemini CLI Quota Tracker (`providers/utilities/gemini_cli_quota_tracker.py`)

A mixin class providing quota tracking functionality for the Gemini CLI provider. This enables accurate remaining quota estimation based on API-fetched baselines and local request counting.

#### Core Concepts

**Quota Baseline Tracking:**
- Periodically fetches quota status from the `retrieveUserQuota` API endpoint
- Stores the remaining fraction as a baseline in UsageManager
- Tracks requests since baseline to estimate current remaining quota
- Syncs local request count with API's authoritative values

**Quota Cost Constants:**
Based on empirical testing, quota limits are known per model and tier:

| Tier | Model Group | Max Requests per 100% |
|------|-------------|----------------------|
| standard-tier | Pro (gemini-2.5-pro, gemini-3-pro-preview) | 250 |
| standard-tier | 2.5-Flash (gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-flash-lite) | 1500 |
| standard-tier | 3-Flash (gemini-3-flash-preview) | 1500 |
| free-tier | Pro | 100 |
| free-tier | 2.5-Flash | 1000 |
| free-tier | 3-Flash | 1000 |

**Reset Windows:**
- All tiers use 24-hour fixed windows from first request (verified 2026-01-07)
- The reset time is set when the first request is made and does NOT roll forward

**Model Quota Groups:**
Models that share quota limits are grouped together:
- `pro`: `gemini-2.5-pro`, `gemini-3-pro-preview`
- `25-flash`: `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`
- `3-flash`: `gemini-3-flash-preview`

Groups can be overridden via environment variables: `QUOTA_GROUPS_GEMINI_CLI_{GROUP}="model1,model2"`

#### Key Methods

**`retrieve_user_quota(credential_path)`:**
Fetches current quota status from the Gemini CLI `retrieveUserQuota` API. Returns remaining fraction and reset times for all models.

**`get_all_quota_info(credential_paths, oauth_base_dir, usage_data, include_estimates)`:**
Gets structured quota info for all credentials, suitable for the TUI quota viewer and stats endpoint.

**`get_max_requests_for_model(model, tier)`:**
Returns the maximum number of requests for a model/tier combination. Uses learned values if available, otherwise falls back to defaults.

**`discover_quota_costs(credential_path, models_to_test)`:**
Manual utility to discover quota costs by making test requests and measuring before/after quota. Saves learned costs to `cache/gemini_cli/learned_quota_costs.json`.

#### Integration with Background Jobs

The Gemini CLI provider defines a background job for quota baseline refresh:

```python
def get_background_job_config(self) -> Optional[Dict[str, Any]]:
    return {
        "interval": 300,  # 5 minutes (configurable via GEMINI_CLI_QUOTA_REFRESH_INTERVAL)
        "name": "gemini_cli_quota_refresh",
        "run_on_start": True,
    }
```

This job:
1. On first run: Fetches quota for ALL credentials to establish baselines
2. On subsequent runs: Only fetches for credentials used since last refresh
3. Updates baselines in UsageManager for accurate estimation

#### Data Storage

Quota baselines are stored in UsageManager's per-model data:

```json
{
  "credential_path": {
    "models": {
      "gemini_cli/gemini-2.5-pro": {
        "request_count": 15,
        "baseline_remaining_fraction": 0.94,
        "baseline_fetched_at": 1734567890.0,
        "requests_at_baseline": 15,
        "quota_max_requests": 250,
        "quota_display": "15/250"
      }
    }
  }
}
```

#### Environment Variables

```env
# Background job interval in seconds (default: 300 = 5 min)
GEMINI_CLI_QUOTA_REFRESH_INTERVAL=300

# Override default quota groups
QUOTA_GROUPS_GEMINI_CLI_PRO="gemini-2.5-pro,gemini-3-pro-preview"
QUOTA_GROUPS_GEMINI_CLI_25_FLASH="gemini-2.0-flash,gemini-2.5-flash,gemini-2.5-flash-lite"
QUOTA_GROUPS_GEMINI_CLI_3_FLASH="gemini-3-flash-preview"
```

### 2.17. Shared Gemini OAuth Utilities (`providers/utilities/`)

Shared Gemini CLI logic lives in reusable utility modules:

| Module | Purpose |
|--------|---------|
| `gemini_shared_utils.py` | Shared constants (FINISH_REASON_MAP, DEFAULT_SAFETY_SETTINGS, CODE_ASSIST_ENDPOINT), helper functions (env_bool, env_int, inline_schema_refs, recursively_parse_json_strings) |
| `base_quota_tracker.py` | Abstract base class for quota tracking with learned costs, credential discovery, and baseline management |
| `gemini_credential_manager.py` | Mixin for OAuth credential tier management, initialization, and background job interface |
| `gemini_file_logger.py` | Transaction-level file logging for debugging API requests and responses |
| `gemini_tool_handler.py` | Tool schema transformation and Gemini 3 tool fix logic |

**Benefits:**
- Keeps Gemini CLI provider logic modular
- Single source of truth for shared constants and logic
- Easier maintenance and bug fixes
- Consistent behavior across Google OAuth-based providers

### 2.18. Fair Cycle Rotation

Fair Cycle Rotation ensures each credential is used at least once before any credential can be reused within a tier. This prevents a single credential from being repeatedly used and exhausted while others sit idle.

**Problem Solved:**
- In sequential mode, the same high-priority credential might be used repeatedly
- When exhausted, it gets a cooldown, but after cooldown expires, it's used again
- Other credentials of the same tier never get used

**Solution:**
- When a credential hits a long cooldown (> threshold), mark it as "exhausted"
- Exhausted credentials are skipped until ALL credentials in the tier exhaust
- Once all exhaust OR cycle duration expires, the cycle resets

**Configuration (Environment Variables):**

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `FAIR_CYCLE_{PROVIDER}` | bool | sequential only | Enable/disable fair cycle |
| `FAIR_CYCLE_TRACKING_MODE_{PROVIDER}` | string | `model_group` | `model_group` or `credential` |
| `FAIR_CYCLE_CROSS_TIER_{PROVIDER}` | bool | `false` | Track across all tiers |
| `FAIR_CYCLE_DURATION_{PROVIDER}` | int | `86400` | Cycle duration in seconds |
| `EXHAUSTION_COOLDOWN_THRESHOLD_{PROVIDER}` | int | `300` | Threshold in seconds |

**Defaults:** All defaults are defined in `src/rotator_library/config/defaults.py`.

**Logging Format:**
```
Acquiring key for model gemini_cli/gemini-2.5-pro. Tried keys: 0/12(17,cd:3,fc:2)
# Breakdown: 0 tried, 12 available, 17 total, 3 on cooldown, 2 fair-cycle excluded
```

**Persistence:**
Cycle state is persisted alongside usage data in `usage/usage_<provider>.json`.

### 2.19. Custom Caps

Custom Caps allow setting custom usage limits per tier, per model/group that are MORE restrictive than actual API limits. When the custom cap is reached, the credential is put on cooldown BEFORE hitting the actual API limit.

**Use Cases:**
- Pace usage across quota window (don't burn 150 requests in first hour)
- Reserve capacity for certain times of day
- Add safety buffer (stop at 120/150 to avoid edge cases)
- Extend cooldown beyond natural reset for pacing

**Key Principle: More Restrictive Only**
- Custom cap is always <= actual max (clamped if set higher)
- Custom cooldown is always >= natural reset time (clamped if set shorter)

**Configuration (Environment Variables):**

```bash
# Format
CUSTOM_CAP_{PROVIDER}_T{TIER}_{MODEL_OR_GROUP}=<value>
CUSTOM_CAP_COOLDOWN_{PROVIDER}_T{TIER}_{MODEL_OR_GROUP}=<mode>:<value>

# Examples
CUSTOM_CAP_GEMINI_CLI_T2_PRO=100
CUSTOM_CAP_COOLDOWN_GEMINI_CLI_T2_PRO=quota_reset
```

**Cap Values:**
- Absolute number: `100`
- Percentage of actual max: `"80%"`

**Cooldown Modes:**

| Mode | Formula | Use Case |
|------|---------|----------|
| `quota_reset` | `quota_reset_ts` | Same as natural behavior |
| `offset` | `quota_reset_ts + value` | Add buffer time |
| `fixed` | `window_start_ts + value` | Fixed window from start |

**Resolution Priority:**
1. Tier + Model (most specific)
2. Tier + Group (model's quota group)
3. Default + Model
4. Default + Group
5. No custom cap (use actual API limits)

**Integration with Fair Cycle:**
When a custom cap triggers a cooldown longer than the exhaustion threshold, it also marks the credential as exhausted for fair cycle rotation.

**Defaults:** See `src/rotator_library/config/defaults.py` for all configurable defaults.

### 2.21. Anthropic API Compatibility

The legacy `anthropic_compat/` translation package is retired. The
`/v1/messages` and `/v1/messages/count_tokens` surfaces run on the
`anthropic_messages` protocol adapter through the canonical runtime:
the raw request body transports verbatim (unknown fields and explicit
nulls survive), validation errors return Anthropic-shaped errors, and
`client.anthropic_messages(payload)` accepts the raw dict.

