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

### 2.21. Anthropic API Compatibility (`anthropic_compat/`)

A translation layer that enables Anthropic API clients (like Claude Code) to use any OpenAI-compatible provider through the proxy.

#### Architecture

The module consists of three components:

| File | Purpose |
|------|---------|
| `models.py` | Pydantic models for Anthropic request/response formats (`AnthropicMessagesRequest`, `AnthropicMessage`, `AnthropicTool`, etc.) |
| `translator.py` | Bidirectional format translation functions |
| `streaming.py` | SSE format conversion for streaming responses |

#### Request Translation (`translate_anthropic_request`)

Converts Anthropic Messages API requests to OpenAI Chat Completions format:

**Message Conversion:**
- Anthropic `system` field → OpenAI system message
- `content` blocks (text, image, tool_use, tool_result) → OpenAI format
- Image blocks with base64 data → OpenAI `image_url` with data URI
- Document blocks (PDF, etc.) → OpenAI `image_url` format

**Tool Conversion:**
- Anthropic `tools` with `input_schema` → OpenAI `tools` with `parameters`
- `tool_choice.type: "any"` → `"required"`
- `tool_choice.type: "tool"` → `{"type": "function", "function": {"name": ...}}`

**Thinking Configuration:**
- `thinking.type: "enabled"` → `reasoning_effort: "high"` + `thinking_budget`
- `thinking.type: "disabled"` → `reasoning_effort: "disable"`
- Opus models default to thinking enabled

**Special Handling:**
- Reorders assistant content blocks: thinking → text → tool_use
- Injects `[Continue]` prompt for fresh thinking turns
- Preserves thinking signatures for multi-turn conversations

#### Response Translation (`openai_to_anthropic_response`)

Converts OpenAI Chat Completions responses to Anthropic Messages format:

**Content Blocks:**
- `reasoning_content` → thinking block with signature
- `content` → text block
- `tool_calls` → tool_use blocks with parsed JSON input

**Field Mapping:**
- `finish_reason: "stop"` → `stop_reason: "end_turn"`
- `finish_reason: "length"` → `stop_reason: "max_tokens"`
- `finish_reason: "tool_calls"` → `stop_reason: "tool_use"`

**Usage Translation:**
- `prompt_tokens` minus `cached_tokens` → `input_tokens`
- `completion_tokens` → `output_tokens`
- `prompt_tokens_details.cached_tokens` → `cache_read_input_tokens`

#### Streaming Wrapper (`anthropic_streaming_wrapper`)

Converts OpenAI SSE streaming format to Anthropic's event-based format:

**Event Types Generated:**
```
message_start      → Initial message metadata
content_block_start → Start of text/thinking/tool_use block
content_block_delta → Incremental content (text_delta, thinking_delta, input_json_delta)
content_block_stop  → End of content block
message_delta      → Final metadata (stop_reason, usage)
message_stop       → End of message
```

**Features:**
- Accumulates tool call arguments across chunks
- Handles thinking/reasoning content from `delta.reasoning_content`
- Proper block indexing for multiple content blocks
- Cache token handling in usage statistics
- Error recovery with proper message structure

#### Client Integration

The `RotatingClient` provides two methods for Anthropic compatibility:

```python
async def anthropic_messages(self, request, raw_request=None, pre_request_callback=None):
    """Handle Anthropic Messages API requests."""
    # 1. Translate Anthropic request to OpenAI format
    # 2. Call acompletion() with translated request
    # 3. Convert response back to Anthropic format
    # 4. For streaming: wrap with anthropic_streaming_wrapper

async def anthropic_count_tokens(self, request):
    """Count tokens for Anthropic-format request."""
    # Translates messages and tools, then uses token_count()
```

#### Authentication

The proxy accepts both Anthropic and OpenAI authentication styles:
- `x-api-key` header (Anthropic style)
- `Authorization: Bearer` header (OpenAI style)

- **Atomic Disk Writes**: Uses temp-file-and-move pattern to prevent corruption

**Key Methods:**

1. **`store(key, value)`**: Synchronously queues value for storage (schedules async write)
2. **`retrieve(key)`**: Synchronously retrieves from memory, optionally schedules disk fallback
3. **`store_async(key, value)`**: Awaitable storage for guaranteed persistence
4. **`retrieve_async(key)`**: Awaitable retrieval with disk fallback

**Use Cases:**

- **Gemini 3 ThoughtSignatures**: Caching tool call signatures for multi-turn conversations
- **Claude Thinking**: Preserving thinking content for consistency across conversation turns
- **Any Transient State**: Generic key-value storage for provider-specific needs

**Configuration (Environment Variables):**

```env
# Cache control (prefix can be customized per cache instance)
PROVIDER_CACHE_ENABLE=true
PROVIDER_CACHE_WRITE_INTERVAL=60  # seconds between disk writes
PROVIDER_CACHE_CLEANUP_INTERVAL=1800  # 30 min between cleanups

# Gemini 3 specific
GEMINI_CLI_SIGNATURE_CACHE_ENABLE=true
GEMINI_CLI_SIGNATURE_CACHE_TTL=3600  # 1 hour memory TTL
GEMINI_CLI_SIGNATURE_DISK_TTL=86400  # 24 hours disk TTL
```

**File Structure:**

```
cache/
└── gemini_cli/
    └── gemini3_signatures.json
```

---

### 2.13. Sequential Rotation & Per-Model Quota Tracking

A comprehensive credential rotation and quota management system introduced in PR #31.

#### Rotation Modes

Two rotation strategies are available per provider:

**Balanced Mode**:
- Distributes load evenly across all credentials
- Least-used credentials selected first
- Best for providers with per-minute rate limits
- Prevents any single credential from being overused

**Sequential Mode (Default)**:
- Uses one credential until it's exhausted (429 quota error)
- Switches to next credential only after current one fails
- Most-used credentials selected first (sticky behavior)
- Best for providers with daily/weekly quotas
- Maximizes provider-side cache locality
- Default for all providers unless overridden

**Configuration**:
```env
# Set per provider
ROTATION_MODE_GEMINI=sequential
ROTATION_MODE_OPENAI=balanced

# Optional sequential session-stickiness controls
SESSION_STICKY_WAIT_SECONDS=15
SESSION_STICKY_ENTRY_TTL_SECONDS=3600
SESSION_STICKY_MAX_ENTRIES=10000
```

#### Session Stickiness

Sequential routing uses scoped session evidence to keep related chat requests on
the same credential where possible. Evidence is always bound to the resolved
credential scope, provider, and model/session scope, so classifier/private scopes
do not share sticky sessions. Explicit request IDs are weak by default because
some clients generate random IDs per request; set `TRUSTED_SESSION_ID_FIELDS` only
when the client-provided fields are known stable.

Provider plugins may implement `get_session_tracking_hints()` to contribute
provider-specific anchors or a provider session scope. The hook supplies evidence
only; credential selection remains centralized in the rotator.

#### Per-Model Quota Tracking

Instead of tracking usage at the credential level, the system now supports granular per-model tracking:

**Data Structure** (when `mode="per_model"`):
```json
{
  "credential_id": {
    "models": {
      "gemini-2.5-pro": {
        "window_start_ts": 1733678400.0,
        "quota_reset_ts": 1733696400.0,
        "success_count": 15,
        "prompt_tokens": 5000,
        "completion_tokens": 1000,
        "approx_cost": 0.05,
        "window_started": "2025-12-08 14:00:00 +0100",
        "quota_resets": "2025-12-08 19:00:00 +0100"
      }
    },
    "global": {...},
    "model_cooldowns": {...}
  }
}
```

**Key Features**:
- Each model tracks its own usage window independently
- `window_start_ts`: When the current quota period started
- `quota_reset_ts`: Authoritative reset time from provider error response
- Human-readable timestamps added for debugging
- Supports custom window durations (5h, 7d, etc.)

#### Provider-Specific Quota Parsing

Providers can implement `parse_quota_error()` to extract precise reset times from error responses:

```python
@staticmethod
def parse_quota_error(error, error_body) -> Optional[Dict]:
    """Extract quota reset timestamp from provider error.
    
    Returns:
        {
            'quota_reset_timestamp': 1733696400.0,  # Unix timestamp
            'retry_after': 18000  # Seconds until reset
        }
    """
```

**Google RPC Format** (Gemini CLI):
- Parses `RetryInfo` and `ErrorInfo` from error details
- Handles duration strings: `"143h4m52.73s"` or `"515092.73s"`
- Extracts `quotaResetTimeStamp` and converts to Unix timestamp
- Falls back to `quotaResetDelay` if timestamp not available

**Example Error Response**:
```json
{
  "error": {
    "code": 429,
    "message": "Quota exceeded",
    "details": [{
      "@type": "type.googleapis.com/google.rpc.RetryInfo",
      "retryDelay": "143h4m52.73s"
    }, {
      "@type": "type.googleapis.com/google.rpc.ErrorInfo",
      "metadata": {
        "quotaResetTimeStamp": "2025-12-08T19:00:00Z"
      }
    }]
  }
}
```

#### Model Quota Groups

Models that share the same quota limits can be grouped:

**Configuration**:
```env
# Models in a group share quota/cooldown timing
QUOTA_GROUPS_GEMINI_CLI_PRO="gemini-2.5-pro,gemini-3-pro-preview"
```

**Behavior**:
- When one model hits quota, all models in the group receive the same `quota_reset_ts`
- Group resets only when ALL models' quotas have reset
- Preserves unexpired cooldowns during other resets

#### Priority-Based Concurrency Multipliers

Credentials can be assigned to priority tiers with configurable concurrency limits:

**Configuration**:
```env
# Universal multipliers (all modes)
CONCURRENCY_MULTIPLIER_GEMINI_CLI_PRIORITY_1=5
CONCURRENCY_MULTIPLIER_GEMINI_CLI_PRIORITY_2=3

# Mode-specific overrides
CONCURRENCY_MULTIPLIER_GEMINI_CLI_PRIORITY_2_BALANCED=1  # Lower in balanced mode
```

**How it works**:
```python
effective_concurrent_limit = MAX_CONCURRENT_REQUESTS_PER_KEY * tier_multiplier
```

**Benefits**:
- Paid credentials handle more load without manual configuration
- Different concurrency for different rotation modes
- Automatic tier detection based on credential properties

#### Reset Window Configuration

Providers can specify custom reset windows per priority tier:

**Supported Modes**:
- `per_model`: Independent window per model with authoritative reset times
- `credential`: Single window per credential (legacy)
- `daily`: Daily reset at configured UTC hour (legacy)

#### Usage Flow

1. **Request arrives** for model X with credential Y
2. **Check rotation mode**: Sequential or balanced?
3. **Select credential**:
   - Filter by priority tier requirements
   - Apply concurrency multiplier for effective limit
   - Sort by rotation mode strategy
4. **Check quota**:
   - Load model's usage data
   - Check if within window (window_start_ts to quota_reset_ts)
   - Check model quota groups for combined usage
5. **Execute request**
6. **On success**: Increment model usage count
7. **On quota error**:
   - Parse error for `quota_reset_ts`
   - Apply to model (and quota group)
   - Credential remains on cooldown until reset time
8. **On window expiration**:
   - Archive model data to global stats
   - Start fresh window with new `window_start_ts`
   - Preserve unexpired quota cooldowns

---

### 2.12. Google OAuth Base (`providers/google_oauth_base.py`)

A refactored, reusable OAuth2 base class that eliminates code duplication across Google-based providers.

**Refactoring Benefits:**

- **Single Source of Truth**: All OAuth logic centralized in one class
- **Easy Provider Addition**: New providers only need to override constants
- **Consistent Behavior**: Token refresh, expiry handling, and validation work identically across providers
- **Maintainability**: OAuth bugs fixed once apply to all inheriting providers

**Inherited Features:**

- Automatic token refresh with exponential backoff
- Invalid grant re-authentication flow
- Stateless deployment support (env var loading)
- Atomic credential file writes
- Headless environment detection
- Sequential refresh queue processing

#### OAuth Callback Port Configuration

Each OAuth provider uses a local callback server during authentication. The callback port can be customized via environment variables to avoid conflicts with other services.

**Default Ports:**

| Provider | Default Port | Environment Variable |
|----------|-------------|---------------------|
| Gemini CLI | 8085 | `GEMINI_CLI_OAUTH_PORT` |

**Configuration Methods:**

1. **Via TUI Settings Menu:**
   - Main Menu → `4. View Provider & Advanced Settings` → `1. Launch Settings Tool`
   - Select the provider (Gemini CLI)
   - Modify the `*_OAUTH_PORT` setting
   - Use "Reset to Default" to restore the original port

2. **Via `.env` file:**
   ```env
   # Custom OAuth callback ports (optional)
   GEMINI_CLI_OAUTH_PORT=8085
   ```

**When to Change Ports:**

- If the default port conflicts with another service on your system
- If running multiple proxy instances on the same machine
- If firewall rules require specific port ranges

**Note:** Port changes take effect on the next OAuth authentication attempt. Existing tokens are not affected.

---

### 2.14. HTTP Timeout Configuration (`timeout_config.py`)

Centralized timeout configuration for all HTTP requests to LLM providers.

#### Purpose

The `TimeoutConfig` class provides fine-grained control over HTTP timeouts for streaming and non-streaming LLM requests. This addresses the common issue of proxy hangs when upstream providers stall during connection establishment or response generation.

#### Timeout Types Explained

| Timeout | Description |
|---------|-------------|
| **connect** | Maximum time to establish a TCP/TLS connection to the upstream server |
| **read** | Maximum time to wait between receiving data chunks (resets on each chunk for streaming) |
| **write** | Maximum time to wait while sending the request body |
| **pool** | Maximum time to wait for a connection from the connection pool |

#### Default Values

| Setting | Streaming | Non-Streaming | Rationale |
|---------|-----------|---------------|-----------|
| **connect** | 30s | 30s | Fast fail if server is unreachable |
| **read** | 180s (3 min) | 600s (10 min) | Streaming expects periodic chunks; non-streaming may wait for full generation |
| **write** | 30s | 30s | Request bodies are typically small |
| **pool** | 60s | 60s | Reasonable wait for connection pool |

#### Environment Variable Overrides

All timeout values can be customized via environment variables:

```env
# Connection establishment timeout (seconds)
TIMEOUT_CONNECT=30

# Request body send timeout (seconds)
TIMEOUT_WRITE=30

# Connection pool acquisition timeout (seconds)
TIMEOUT_POOL=60

# Read timeout between chunks for streaming requests (seconds)
# If no data arrives for this duration, the connection is considered stalled
TIMEOUT_READ_STREAMING=180

# Read timeout for non-streaming responses (seconds)
# Longer to accommodate models that take time to generate full responses
TIMEOUT_READ_NON_STREAMING=600
```

#### Streaming vs Non-Streaming Behavior

**Streaming Requests** (`TimeoutConfig.streaming()`):
- Uses shorter read timeout (default 3 minutes)
- Timer resets every time a chunk arrives
- If no data for 3 minutes → connection considered dead → failover to next credential
- Appropriate for chat completions where tokens should arrive periodically

**Non-Streaming Requests** (`TimeoutConfig.non_streaming()`):
- Uses longer read timeout (default 10 minutes)
- Server may take significant time to generate the complete response before sending anything
- Complex reasoning tasks or large outputs may legitimately take several minutes
- Used by providers with true non-streaming paths

#### Provider Usage

The following providers use `TimeoutConfig`:

| Provider | Method | Timeout Type |
|----------|--------|--------------|
| `gemini_cli_provider.py` | `acompletion()` | `streaming()` |

**Note:** Gemini CLI uses streaming internally (even for non-streaming requests), aggregating chunks into a complete response.

#### Tuning Recommendations

| Use Case | Recommendation |
|----------|----------------|
| **Long thinking tasks** | Increase `TIMEOUT_READ_STREAMING` to 300-360s |
| **Unstable network** | Increase `TIMEOUT_CONNECT` to 60s |
| **High concurrency** | Increase `TIMEOUT_POOL` if seeing pool exhaustion |
| **Large context/output** | Increase `TIMEOUT_READ_NON_STREAMING` to 900s+ |

#### Example Configuration

```env
# For environments with complex reasoning tasks
TIMEOUT_READ_STREAMING=300
TIMEOUT_READ_NON_STREAMING=900

# For unstable network conditions
TIMEOUT_CONNECT=60
TIMEOUT_POOL=120
```

---


---

## 3. Provider Specific Implementations

The library handles provider idiosyncrasies through specialized "Provider" classes in `src/rotator_library/providers/`.

### 3.1. Gemini CLI (`gemini_cli_provider.py`)

The `GeminiCliProvider` is the most complex implementation, mimicking the Google Cloud Code extension.

**New in PR #62**:
- **Quota Baseline Tracking**: Background job fetches quota status from API (`retrieveUserQuota`) to provide accurate remaining quota estimates
- **GeminiCliQuotaTracker Mixin**: Inherits from `BaseQuotaTracker` for shared quota infrastructure
- **env:// Credential Support**: Environment-based credentials are detected and loaded via `env://gemini_cli/N` URIs
- **Quota Groups**: Models sharing quota are grouped (`pro`, `25-flash`, `3-flash`) for accurate cooldown propagation
- **24-Hour Fixed Windows**: All tiers use fixed 24-hour windows from first request (verified 2026-01-07)

**From PR #31**:
- **Quota Parsing**: Implements `parse_quota_error()` using Google RPC format parser
- **Tier Configuration**: Defines `tier_priorities` and `usage_reset_configs` for automatic priority resolution
- **Sequential Rotation**: Defaults to sequential mode (uses credentials until quota exhausted)
- **Priority Multipliers**: P1: 5x, P2: 3x, others: 1x

#### Authentication (`gemini_auth_base.py`)

 *   **Device Flow**: Uses a standard OAuth 2.0 flow. The `credential_tool` spins up a local web server (default: `localhost:8085`, configurable via `GEMINI_CLI_OAUTH_PORT`) to capture the callback from Google's auth page.
 *   **Token Lifecycle**:
    *   **Proactive Refresh**: Tokens are refreshed 5 minutes before expiry.
    *   **Atomic Writes**: Credential files are updated using a temp-file-and-move strategy to prevent corruption during writes.
    *   **Revocation Handling**: If a `400` or `401` occurs during refresh, the token is marked as revoked, preventing infinite retry loops.

#### Project ID Discovery (Zero-Config)

The provider employs a sophisticated, cached discovery mechanism to find a valid Google Cloud Project ID:
1.  **Configuration**: Checks `GEMINI_CLI_PROJECT_ID` first.
2.  **Code Assist API**: Tries `CODE_ASSIST_ENDPOINT:loadCodeAssist`. This returns the project associated with the Cloud Code extension.
3.  **Onboarding Flow**: If step 2 fails, it triggers the `onboardUser` endpoint. This initiates a Long-Running Operation (LRO) that automatically provisions a free-tier Google Cloud Project for the user. The proxy polls this operation for up to 5 minutes until completion.
4.  **Resource Manager**: As a final fallback, it lists all active projects via the Cloud Resource Manager API and selects the first one.

#### Rate Limit Handling

*   **Internal Endpoints**: Uses `https://cloudcode-pa.googleapis.com/v1internal`, which typically has higher quotas than the public API.
*   **Smart Fallback**: If `gemini-2.5-pro` hits a rate limit (`429`), the provider transparently retries the request using `gemini-2.5-pro-preview-06-05`. This fallback chain is configurable in code.

#### Quota Tracking

The provider implements quota tracking via the `GeminiCliQuotaTracker` mixin (see Section 2.17):

*   **Real-Time Quota API**: Fetches quota status from `retrieveUserQuota` endpoint
*   **Background Refresh**: Configurable interval (default: 5 minutes) via `GEMINI_CLI_QUOTA_REFRESH_INTERVAL`
*   **Model Quota Groups**: Pro models share quota, Flash 2.x models share quota, Flash 3 is standalone

**Default Quota Groups:**

| Group Name | Models | Verified Sharing |
|------------|--------|------------------|
| `pro` | gemini-2.5-pro, gemini-3-pro-preview | Yes (same bucket) |
| `25-flash` | gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-flash-lite | Yes (same bucket) |
| `3-flash` | gemini-3-flash-preview | Standalone |

**Quota Limits by Tier:**

| Tier | Pro Group | Flash Groups |
|------|-----------|--------------|
| standard-tier | 250 requests/24h | 1500 requests/24h |
| free-tier | 100 requests/24h | 1000 requests/24h |

#### Configuration (Environment Variables)

```env
# Quota tracking
GEMINI_CLI_QUOTA_REFRESH_INTERVAL=300  # Background refresh interval (default: 5 min)

# Override quota groups
QUOTA_GROUPS_GEMINI_CLI_PRO="gemini-2.5-pro,gemini-3-pro-preview"
QUOTA_GROUPS_GEMINI_CLI_25_FLASH="gemini-2.0-flash,gemini-2.5-flash,gemini-2.5-flash-lite"
QUOTA_GROUPS_GEMINI_CLI_3_FLASH="gemini-3-flash-preview"
```

### 3.2. Google Gemini (`gemini_provider.py`)

*   **Thinking Parameter**: Automatically handles the `thinking` parameter transformation required for Gemini 2.5 models (`thinking` -> `gemini-2.5-pro` reasoning parameter).
*   **Safety Settings**: Ensures default safety settings (blocking nothing) are applied if not provided, preventing over-sensitive refusals.

---

## 4. Logging & Debugging

### `detailed_logger.py`

To facilitate robust debugging, the proxy includes a comprehensive transaction logging system.

*   **Unique IDs**: Every request generates a UUID.
*   **Directory Structure**: Logs are stored in `logs/detailed_logs/YYYYMMDD_HHMMSS_{uuid}/`.
*   **Artifacts**:
    *   `request.json`: The exact payload sent to the proxy.
    *   `final_response.json`: The complete reassembled response.
    *   `streaming_chunks.jsonl`: A line-by-line log of every SSE chunk received from the provider.
    *   `metadata.json`: Performance metrics (duration, token usage, model used).

This level of detail allows developers to trace exactly why a request failed or why a specific key was rotated.

---

## 5. Runtime Resilience

The proxy is engineered to maintain high availability even in the face of runtime filesystem disruptions. This "Runtime Resilience" capability ensures that the service continues to process API requests even if data files or directories are deleted while the application is running.

### 5.1. Centralized Resilient I/O (`resilient_io.py`)

All file operations are centralized in a single utility module that provides consistent error handling, graceful degradation, and automatic retry with shutdown flush:

#### `BufferedWriteRegistry` (Singleton)

Global registry for buffered writes with periodic retry and shutdown flush. Ensures critical data is saved even if disk writes fail temporarily:

- **Per-file buffering**: Each file path has its own pending write (latest data always wins)
- **Periodic retries**: Background thread retries failed writes every 30 seconds
- **Shutdown flush**: `atexit` hook ensures final write attempt on app exit (Ctrl+C)
- **Thread-safe**: Safe for concurrent access from multiple threads

```python
# Get the singleton instance
registry = BufferedWriteRegistry.get_instance()

# Check pending writes (for monitoring)
pending_count = registry.get_pending_count()
pending_files = registry.get_pending_paths()

# Manual flush (optional - atexit handles this automatically)
results = registry.flush_all()  # Returns {path: success_bool}

# Manual shutdown (if needed before atexit)
results = registry.shutdown()
```

#### `ResilientStateWriter`

For stateful files that must persist (usage stats):
- **Memory-first**: Always updates in-memory state before attempting disk write
- **Atomic writes**: Uses tempfile + move pattern to prevent corruption
- **Automatic retry with backoff**: If disk fails, waits `retry_interval` seconds before trying again
- **Shutdown integration**: Registers with `BufferedWriteRegistry` on failure for final flush
- **Health monitoring**: Exposes `is_healthy` property for monitoring

```python
writer = ResilientStateWriter("data.json", logger, retry_interval=30.0)
writer.write({"key": "value"})  # Always succeeds (memory update)
if not writer.is_healthy:
    logger.warning("Disk writes failing, data in memory only")
# On next write() call after retry_interval, disk write is attempted again
# On app exit (Ctrl+C), BufferedWriteRegistry attempts final save
```

#### `safe_write_json()`

For JSON writes with configurable options (credentials, cache):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | required | File path to write to |
| `data` | required | JSON-serializable data |
| `logger` | required | Logger for warnings |
| `atomic` | `True` | Use atomic write pattern (tempfile + move) |
| `indent` | `2` | JSON indentation level |
| `ensure_ascii` | `True` | Escape non-ASCII characters |
| `secure_permissions` | `False` | Set file permissions to 0o600 |
| `buffer_on_failure` | `False` | Register with BufferedWriteRegistry on failure |

When `buffer_on_failure=True`:
- Failed writes are registered with `BufferedWriteRegistry`
- Data is retried every 30 seconds in background
- On app exit, final write attempt is made automatically
- Success unregisters the pending write

```python
# For critical data (auth tokens) - use buffer_on_failure
safe_write_json(path, creds, logger, secure_permissions=True, buffer_on_failure=True)

# For non-critical data (logs) - no buffering needed
safe_write_json(path, data, logger)
```

#### `safe_log_write()`

For log files where occasional loss is acceptable:
- Fire-and-forget pattern
- Creates parent directories if needed
- Returns `True`/`False`, never raises
- **No buffering** - logs are dropped on failure

#### `safe_mkdir()`

For directory creation with error handling.

### 5.2. Resilience Hierarchy

The system follows a strict hierarchy of survival:

1. **Core API Handling (Level 1)**: The Python runtime keeps all necessary code in memory. Deleting source code files while the proxy is running will **not** crash active requests.

2. **Credential Management (Level 2)**: OAuth tokens are cached in memory first. If credential files are deleted, the proxy continues using cached tokens. If a token refresh succeeds but the file cannot be written, the new token is buffered for retry and saved on shutdown.

3. **Usage Tracking (Level 3)**: Usage statistics (`usage/usage_<provider>.json`) are maintained in memory via `ResilientStateWriter`. If the file is deleted, the system tracks usage internally and attempts to recreate the file on the next save interval. Pending writes are flushed on shutdown.

4. **Provider Cache (Level 4)**: The provider cache tracks disk health and continues operating in memory-only mode if disk writes fail. Has its own shutdown mechanism.

5. **Logging (Level 5)**: Logging is treated as non-critical. If the `logs/` directory is removed, the system attempts to recreate it. If creation fails, logging degrades gracefully without interrupting the request flow. **No buffering or retry**.

### 5.3. Component Integration

| Component | Utility Used | Behavior on Disk Failure | Shutdown Flush |
|-----------|--------------|--------------------------|----------------|
| `UsageManager` | `ResilientStateWriter` | Continues in memory, retries after 30s | Yes (via registry) |
| `GoogleOAuthBase` | `safe_write_json(buffer_on_failure=True)` | Memory cache preserved, buffered for retry | Yes (via registry) |
| `ProviderCache` | `safe_write_json` + own shutdown | Retries via own background loop | Yes (own mechanism) |
| `DetailedLogger` | `safe_write_json` | Logs dropped, no crash | No |
| `failure_logger` | Python `logging.RotatingFileHandler` | Falls back to NullHandler | No |

### 5.4. Shutdown Behavior

When the application exits (including Ctrl+C):

1. **atexit handler fires**: `BufferedWriteRegistry._atexit_handler()` is called
2. **Pending writes counted**: Registry checks how many files have pending writes
3. **Flush attempted**: Each pending file gets a final write attempt
4. **Results logged**:
   - Success: `"Shutdown flush: all N write(s) succeeded"`
   - Partial: `"Shutdown flush: X succeeded, Y failed"` with failed file names

**Console output example:**
```
INFO:rotator_library.resilient_io:Flushing 2 pending write(s) on shutdown...
INFO:rotator_library.resilient_io:Shutdown flush: all 2 write(s) succeeded
```

### 5.5. "Develop While Running"

This architecture supports a robust development workflow:

- **Log Cleanup**: You can safely run `rm -rf logs/` while the proxy is serving traffic. The system will recreate the directory structure on the next request.
- **Config Reset**: Deleting `usage/usage_<provider>.json` resets the persistence layer, but the running instance preserves its current in-memory counts for load balancing consistency.
- **File Recovery**: If you delete a critical file, the system attempts directory auto-recreation before every write operation.
- **Safe Exit**: Ctrl+C triggers graceful shutdown with final data flush attempt.

### 5.6. Graceful Degradation & Data Loss

While functionality is preserved, persistence may be compromised during filesystem failures:

- **Logs**: If disk writes fail, detailed request logs may be lost (no buffering).
- **Usage Stats**: Buffered in memory and flushed on shutdown. Data loss only if shutdown flush also fails.
- **Credentials**: Buffered in memory and flushed on shutdown. Re-authentication only needed if shutdown flush fails.
- **Cache**: Provider cache entries may need to be regenerated after restart if its own shutdown mechanism fails.

### 5.7. Monitoring Disk Health

Components expose health information for monitoring:

```python
# BufferedWriteRegistry
registry = BufferedWriteRegistry.get_instance()
pending = registry.get_pending_count()  # Number of files with pending writes
files = registry.get_pending_paths()    # List of pending file names

# UsageManager
writer = usage_manager._state_writer
health = writer.get_health_info()
# Returns: {"healthy": True, "failure_count": 0, "last_success": 1234567890.0, ...}

# ProviderCache
stats = cache.get_stats()
# Includes: {"disk_available": True, "disk_errors": 0, ...}
```

---

## 6. Model Filter GUI

The Model Filter GUI (`model_filter_gui.py`) provides a visual interface for configuring model ignore and whitelist rules per provider. It replaces the need to manually edit `IGNORE_MODELS_*` and `WHITELIST_MODELS_*` environment variables.

### 6.1. Overview

**Purpose**: Visually manage which models are exposed via the `/v1/models` endpoint for each provider.

**Launch**: 
```bash
python -c "from src.proxy_app.model_filter_gui import run_model_filter_gui; run_model_filter_gui()"
```

Or via the launcher TUI if integrated.

### 6.2. Features

#### Core Functionality

- **Provider Selection**: Dropdown to switch between available providers with automatic model fetching
- **Ignore Rules**: Pattern-based rules (supports wildcards like `*-preview`, `gpt-4*`) to exclude models
- **Whitelist Rules**: Pattern-based rules to explicitly include models, overriding ignore rules
- **Real-time Preview**: Typing in rule input fields highlights affected models before committing
- **Rule-Model Linking**: Click a model to highlight the affecting rule; click a rule to highlight all affected models
- **Persistence**: Rules saved to `.env` file in standard `IGNORE_MODELS_<PROVIDER>` and `WHITELIST_MODELS_<PROVIDER>` format

#### Dual-Pane Model View

The interface displays two synchronized lists:

| Left Pane | Right Pane |
|-----------|------------|
| All fetched models (plain text) | Same models with color-coded status |
| Shows total count | Shows available/ignored count |
| Scrolls in sync with right pane | Color indicates affecting rule |

**Color Coding**:
- **Green**: Model is available (no rule affects it, or whitelisted)
- **Red/Orange tones**: Model is ignored (color matches the specific ignore rule)
- **Blue/Teal tones**: Model is explicitly whitelisted (color matches the whitelist rule)

#### Rule Management

- **Comma-separated input**: Add multiple rules at once (e.g., `*-preview, *-beta, gpt-3.5*`)
- **Wildcard support**: `*` matches any characters (e.g., `gemini-*-preview`)
- **Affected count**: Each rule shows how many models it affects
- **Tooltips**: Hover over a rule to see the list of affected models
- **Instant delete**: Click the × button to remove a rule immediately

### 6.3. Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+S` | Save changes to `.env` |
| `Ctrl+R` | Refresh models from provider |
| `Ctrl+F` | Focus search field |
| `F1` | Show help dialog |
| `Escape` | Clear search / Clear highlights |

### 6.4. Context Menu

Right-click on any model to access:

- **Add to Ignore List**: Creates an ignore rule for the exact model name
- **Add to Whitelist**: Creates a whitelist rule for the exact model name
- **View Affecting Rule**: Highlights the rule that affects this model
- **Copy Model Name**: Copies the full model ID to clipboard

### 6.5. Integration with Proxy

The GUI modifies the same environment variables that the `RotatingClient` reads:

1. **GUI saves rules** → Updates `.env` file
2. **Proxy reads on startup** → Loads `IGNORE_MODELS_*` and `WHITELIST_MODELS_*`
3. **Proxy applies rules** → `get_available_models()` filters based on rules

**Note**: The proxy must be restarted to pick up rule changes made via the GUI (or use the Launcher TUI's reload functionality if available).

