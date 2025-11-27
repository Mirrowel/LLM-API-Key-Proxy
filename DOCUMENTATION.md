# Technical Documentation: Universal LLM API Proxy & Resilience Library

This document provides a detailed technical explanation of the project's architecture, internal components, and data flows. It is intended for developers who want to understand how the system achieves high availability and resilience.

## 1. Architecture Overview

The project is a monorepo containing two primary components:

1.  **The Proxy Application (`proxy_app`)**: This is the user-facing component. It's a FastAPI application that acts as a universal gateway. It uses `litellm` to translate requests to various provider formats and includes:
    *   **Batch Manager**: Optimizes high-volume embedding requests.
    *   **Detailed Logger**: Provides per-request file logging for debugging.
    *   **OpenAI-Compatible Endpoints**: `/v1/chat/completions`, `/v1/embeddings`, etc.
2.  **The Resilience Library (`rotator_library`)**: This is the core engine that provides high availability. It is consumed by the proxy app to manage a pool of API keys, handle errors gracefully, and ensure requests are completed successfully even when individual keys or provider endpoints face issues.

This architecture cleanly separates the API interface from the resilience logic, making the library a portable and powerful tool for any application needing robust API key management.

---

## 2. `rotator_library` - The Resilience Engine

This library is the heart of the project, containing all the logic for managing a pool of API keys, tracking their usage, and handling provider interactions to ensure application resilience.

### 2.1. `client.py` - The `RotatingClient`

The `RotatingClient` is the central class that orchestrates all operations. It is designed as a long-lived, async-native object.

#### Initialization

The client is initialized with your provider API keys, retry settings, and a new `global_timeout`.

```python
client = RotatingClient(
    api_keys=api_keys,
    oauth_credentials=oauth_credentials,
    max_retries=2,
    usage_file_path="key_usage.json",
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
-   `usage_file_path` (`str`, default: `"key_usage.json"`): The path to the JSON file where usage statistics are persisted.
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
*   **Request Sanitization**: Automatically cleaning invalid parameters (like `dimensions` for non-OpenAI models) via `request_sanitizer.py`.

#### Model Filtering Logic

The `RotatingClient` provides fine-grained control over which models are exposed via the `/v1/models` endpoint. This is handled by the `get_available_models` method.

The logic applies in the following order:
1.  **Whitelist Check**: If a provider has a whitelist defined (`WHITELIST_MODELS_<PROVIDER>`), any model on that list will **always be available**, even if it matches a blacklist pattern. This acts as a definitive override.
2.  **Blacklist Check**: For any model *not* on the whitelist, the client checks the blacklist (`IGNORE_MODELS_<PROVIDER>`). If the model matches a blacklist pattern (supports wildcards like `*-preview`), it is excluded.
3.  **Default**: If a model is on neither list, it is included.

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

### 2.2. `usage_manager.py` - Stateful Concurrency & Usage Management

This class is the stateful core of the library, managing concurrency, usage tracking, and cooldowns.

#### Key Concepts

*   **Async-Native & Lazy-Loaded**: Fully asynchronous, using `aiofiles` for non-blocking file I/O. Usage data is loaded only when needed.
*   **Fine-Grained Locking**: Each API key has its own `asyncio.Lock` and `asyncio.Condition`. This allows for highly granular control.

#### Tiered Key Acquisition Strategy

The `acquire_key` method uses a sophisticated strategy to balance load:

1.  **Filtering**: Keys currently on cooldown (global or model-specific) are excluded.
2.  **Tiering**: Valid keys are split into two tiers:
    *   **Tier 1 (Ideal)**: Keys that are completely idle (0 concurrent requests).
    *   **Tier 2 (Acceptable)**: Keys that are busy but still under their configured `MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>` limit for the requested model. This allows a single key to be used multiple times for the same model, maximizing throughput.
3.  **Selection Strategy** (configurable via `rotation_tolerance`):
    *   **Deterministic (tolerance=0.0)**: Within each tier, keys are sorted by daily usage count and the least-used key is always selected. This provides perfect load balance but predictable patterns.
    *   **Weighted Random (tolerance>0, default)**: Keys are selected randomly with weights biased toward less-used ones:
        - Formula: `weight = (max_usage - credential_usage) + tolerance + 1`
        - `tolerance=2.0` (recommended): Balanced randomness - credentials within 2 uses of the maximum can still be selected with reasonable probability
        - `tolerance=5.0+`: High randomness - even heavily-used credentials have significant probability
        - **Security Benefit**: Unpredictable selection patterns make rate limit detection and fingerprinting harder
        - **Load Balance**: Lower-usage credentials still preferred, maintaining reasonable distribution
4.  **Concurrency Limits**: Checks against `max_concurrent` limits to prevent overloading a single key.
5.  **Priority Groups**: When credential prioritization is enabled, higher-tier credentials (lower priority numbers) are tried first before moving to lower tiers.

#### Failure Handling & Cooldowns

*   **Escalating Backoff**: When a failure occurs, the key gets a temporary cooldown for that specific model. Consecutive failures increase this time (10s -> 30s -> 60s -> 120s).
*   **Key-Level Lockouts**: If a key accumulates failures across multiple distinct models (3+), it is assumed to be dead/revoked and placed on a global 5-minute lockout.
*   **Authentication Errors**: Immediate 5-minute global lockout.

### 2.3. `batch_manager.py` - Efficient Request Aggregation

The `EmbeddingBatcher` class optimizes high-throughput embedding workloads.

*   **Mechanism**: It uses an `asyncio.Queue` to collect incoming requests.
*   **Triggers**: A batch is dispatched when either:
    1.  The queue size reaches `batch_size` (default: 64).
    2.  A time window (`timeout`, default: 0.1s) elapses since the first request in the batch.
*   **Efficiency**: This reduces dozens of HTTP calls to a single API request, significantly reducing overhead and rate limit usage.

### 2.4. `background_refresher.py` - Automated Token Maintenance

The `BackgroundRefresher` ensures that OAuth tokens (for providers like Gemini CLI, Qwen, iFlow) never expire while the proxy is running.

*   **Periodic Checks**: It runs a background task that wakes up at a configurable interval (default: 3600 seconds/1 hour).
*   **Proactive Refresh**: It iterates through all loaded OAuth credentials and calls their `proactively_refresh` method to ensure tokens are valid before they are needed.

### 2.6. Credential Management Architecture

The `CredentialManager` class (`credential_manager.py`) centralizes the lifecycle of all API credentials. It adheres to a "Local First" philosophy.

#### 2.6.1. Automated Discovery & Preparation

On startup (unless `SKIP_OAUTH_INIT_CHECK=true`), the manager performs a comprehensive sweep:

1. **System-Wide Scan**: Searches for OAuth credential files in standard locations:
   - `~/.gemini/` → All `*.json` files (typically `credentials.json`)
   - `~/.qwen/` → All `*.json` files (typically `oauth_creds.json`)
   - `~/.iflow/` → All `*. json` files

2. **Local Import**: Valid credentials are **copied** (not moved) to the project's `oauth_creds/` directory with standardized names:
   -  `gemini_cli_oauth_1.json`, `gemini_cli_oauth_2.json`, etc.
   - `qwen_code_oauth_1.json`, `qwen_code_oauth_2.json`, etc.
   - `iflow_oauth_1.json`, `iflow_oauth_2.json`, etc.

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

**Gemini CLI Environment Variables:**
```
GEMINI_CLI_ACCESS_TOKEN
GEMINI_CLI_REFRESH_TOKEN
GEMINI_CLI_E XPIRY_DATE
GEMINI_CLI_EMAIL
GEMINI_CLI_PROJECT_ID (optional)
GEMINI_CLI_CLIENT_ID (optional)
```

**Qwen Code Environment Variables:**
```
QWEN_CODE_ACCESS_TOKEN
QWEN_CODE_REFRESH_TOKEN
QWEN_CODE_EXPIRY_DATE
QWEN_CODE_EMAIL
```

**iFlow Environment Variables:**
```
IFLOW_ACCESS_TOKEN
IFLOW_REFRESH_TOKEN
IFLOW_EXPIRY_DATE
IFLOW_EMAIL
IFLOW_API_KEY
```

**How it works:**
- If the manager finds (e.g.) `GEMINI_CLI_ACCESS_TOKEN`, it constructs an in-memory credential object that mimics the file structure
- The credential behaves exactly like a file-based credential (automatic refresh, expiry detection, etc.)
- No physical files are created or needed on the host system
- Perfect for ephemeral containers or read-only filesystems

#### 2.6.3. Credential Tool Integration

The `credential_tool.py` provides a user-friendly CLI interface to the `CredentialManager`:

**Key Functions:**
1. **OAuth Setup**: Wraps provider-specific `AuthBase` classes (`GeminiAuthBase`, `QwenAuthBase`, `IFlowAuthBase`) to handle interactive login flows
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

**Provider-Specific Tool Schema Cleaning:**

Implemented in individual provider classes (`QwenCodeProvider`, `IFlowProvider`):

- **Recursively removes** unsupported properties from tool function schemas:
  - `strict`: OpenAI-specific, causes validation errors on Qwen/iFlow
  - `additionalProperties`: Same issue
- **Prevents `400 Bad Request` errors** when using complex tool definitions
- Applied automatically before sending requests to the provider

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

2. **Message Analysis**: Fallback for ambiguous errors
   - Searches for keywords like "quota exceeded", "rate limit", "invalid api key"

3. **Provider-Specific Overrides**: Some providers use non-standard error formats

**Usage in Client:**
- `AUTHENTICATION` → Immediate 5-minute global lockout
- `RATE_LIMIT`/`QUOTA` → Escalating per-model cooldown
- `SERVER_ERROR` → Retry with same key (up to `max_retries`)
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


### 2.10. Credential Prioritization System (`client.py` & `usage_manager.py`)

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

### 3.5. Antigravity (`antigravity_provider.py`)

The most sophisticated provider implementation, supporting Google's internal Antigravity API for Gemini and Claude models.

#### Architecture

- **Unified Streaming/Non-Streaming**: Single code path handles both response types with optimal transformations
- **Thought Signature Caching**: Server-side caching of encrypted signatures for multi-turn Gemini 3 conversations
- **Model-Specific Logic**: Automatic configuration based on model type (Gemini 2.5, Gemini 3, Claude)

#### Model Support

**Gemini 2.5 (Pro/Flash):**
- Uses `thinkingBudget` parameter (integer tokens: -1 for auto, 0 to disable, or specific value)
- Standard safety settings and toolConfig
- Stream processing with thinking content separation

**Gemini 3 (Pro/Image):**
- Uses `thinkingLevel` parameter (string: "low" or "high")
- **Tool Hallucination Prevention**:
  - Automatic system instruction injection explaining custom tool schema rules
  - Parameter signature injection into tool descriptions (e.g., "STRICT PARAMETERS: files (ARRAY_OF_OBJECTS[path: string REQUIRED, ...])")
  - Namespace prefix for tool names (`gemini3_` prefix) to avoid training data conflicts
  - Malformed JSON auto-correction (handles extra trailing braces)
- **ThoughtSignature Management**:
  - Caching signatures from responses for reuse in follow-up messages
  - Automatic injection into functionCalls for multi-turn conversations
  - Fallback to bypass value if signature unavailable

**Claude Sonnet 4.5:**
- Proxied through Antigravity API (uses internal model name `claude-sonnet-4-5-thinking`)
- Uses `thinkingBudget` parameter like Gemini 2.5
- **Thinking Preservation**: Caches thinking content using composite keys (tool_call_id + text_hash)
- **Schema Cleaning**: Removes unsupported properties (`$schema`, `additionalProperties`, `const` → `enum`)

#### Base URL Fallback

Automatic fallback chain for resilience:
1. `daily-cloudcode-pa.sandbox.googleapis.com` (primary sandbox)
2. `autopush-cloudcode-pa.sandbox.googleapis.com` (fallback sandbox)
3. `cloudcode-pa.googleapis.com` (production fallback)

#### Message Transformation

**OpenAI → Gemini Format:**
- System messages → `systemInstruction` with parts array
- Multi-part content (text + images) → `inlineData` format
- Tool calls → `functionCall` with args and id
- Tool responses → `functionResponse` with name and response
- ThoughtSignatures preserved/injected as needed

**Tool Response Grouping:**
- Converts linear format (call, response, call, response) to grouped format
- Groups all function calls in one `model` message
- Groups all responses in one `user` message
- Required for Antigravity API compatibility

#### Configuration (Environment Variables)

```env
# Cache control
ANTIGRAVITY_SIGNATURE_CACHE_TTL=3600  # Memory cache TTL
ANTIGRAVITY_SIGNATURE_DISK_TTL=86400  # Disk cache TTL
ANTIGRAVITY_ENABLE_SIGNATURE_CACHE=true

# Feature flags
ANTIGRAVITY_PRESERVE_THOUGHT_SIGNATURES=true  # Include signatures in client responses
ANTIGRAVITY_ENABLE_DYNAMIC_MODELS=false  # Use API model discovery
ANTIGRAVITY_GEMINI3_TOOL_FIX=true  # Enable Gemini 3 hallucination prevention

# Gemini 3 tool fix customization
ANTIGRAVITY_GEMINI3_TOOL_PREFIX="gemini3_"  # Namespace prefix
ANTIGRAVITY_GEMINI3_DESCRIPTION_PROMPT="\n\nSTRICT PARAMETERS: {params}."
ANTIGRAVITY_GEMINI3_SYSTEM_INSTRUCTION="..."  # Full system prompt
```

#### File Logging

Optional transaction logging for debugging:
- Enabled via `enable_request_logging` parameter
- Creates `logs/antigravity_logs/TIMESTAMP_MODEL_UUID/` directory per request
- Logs: `request_payload.json`, `response_stream.log`, `final_response.json`, `error.log`

---


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
├── gemini_cli/
│   └── gemini3_signatures.json
└── antigravity/
    ├── gemini3_signatures.json
    └── claude_thinking.json
```

---

### 2.12. Google OAuth Base (`providers/google_oauth_base.py`)

A refactored, reusable OAuth2 base class that eliminates code duplication across Google-based providers.

**Refactoring Benefits:**

- **Single Source of Truth**: All OAuth logic centralized in one class
- **Easy Provider Addition**: New providers only need to override constants
- **Consistent Behavior**: Token refresh, expiry handling, and validation work identically across providers
- **Maintainability**: OAuth bugs fixed once apply to all inheriting providers

**Provider Implementation:**

```python
class AntigravityAuthBase(GoogleOAuthBase):
    # Required overrides
    CLIENT_ID = "antigravity-client-id"
    CLIENT_SECRET = "antigravity-secret"
    OAUTH_SCOPES = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/cclog",  # Antigravity-specific
        "https://www.googleapis.com/auth/experimentsandconfigs",
    ]
    ENV_PREFIX = "ANTIGRAVITY"  # Used for env var loading
    
    # Optional overrides (defaults provided)
    CALLBACK_PORT = 51121
    CALLBACK_PATH = "/oauthcallback"
```

**Inherited Features:**

- Automatic token refresh with exponential backoff
- Invalid grant re-authentication flow
- Stateless deployment support (env var loading)
- Atomic credential file writes
- Headless environment detection
- Sequential refresh queue processing

---


---

## 3. Provider Specific Implementations

The library handles provider idiosyncrasies through specialized "Provider" classes in `src/rotator_library/providers/`.

### 3.1. Gemini CLI (`gemini_cli_provider.py`)

The `GeminiCliProvider` is the most complex implementation, mimicking the Google Cloud Code extension.

#### Authentication (`gemini_auth_base.py`)

 *   **Device Flow**: Uses a standard OAuth 2.0 flow. The `credential_tool` spins up a local web server (`localhost:8085`) to capture the callback from Google's auth page.
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

### 3.2. Qwen Code (`qwen_code_provider.py`)

*   **Dual Auth**: Supports both standard API keys (direct) and OAuth (via `QwenAuthBase`).
*   **Device Flow**: Implements the OAuth Device Authorization Grant (RFC 8628). It displays a code to the user and polls the token endpoint until the user authorizes the device in their browser.
*   **Dummy Tool Injection**: To work around a Qwen API bug where streams hang if `tools` is empty but `tool_choice` logic is present, the provider injects a benign `do_not_call_me` tool.
*   **Schema Cleaning**: Recursively removes `strict` and `additionalProperties` from tool schemas, as Qwen's validation is stricter than OpenAI's.
*   **Reasoning Parsing**: Detects `<think>` tags in the raw stream and redirects their content to a separate `reasoning_content` field in the delta, mimicking the OpenAI o1 format.

### 3.3. iFlow (`iflow_provider.py`)

*   **Hybrid Auth**: Uses a custom OAuth flow (Authorization Code) to obtain an `access_token`. However, the *actual* API calls use a separate `apiKey` that is retrieved from the user's profile (`/api/oauth/getUserInfo`) using the access token.
*   **Callback Server**: The auth flow spins up a local server on port `11451` to capture the redirect.
*   **Token Management**: Automatically refreshes the OAuth token and re-fetches the API key if needed.
*   **Schema Cleaning**: Similar to Qwen, it aggressively sanitizes tool schemas to prevent 400 errors.
*   **Dedicated Logging**: Implements `_IFlowFileLogger` to capture raw chunks for debugging proprietary API behaviors.

### 3.4. Google Gemini (`gemini_provider.py`)

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


