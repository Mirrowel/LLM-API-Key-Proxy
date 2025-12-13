# Technical Documentation: Universal LLM API Proxy & Resilience Library

This document provides a detailed technical explanation of the project's architecture, internal components, and data flows. It is intended for developers who want to understand how the system achieves high availability and resilience.

## 1. Architecture Overview

The project is a monorepo containing two primary components:

1.  **The Proxy Application (`proxy_app`)**: This is the user-facing component. It's a FastAPI application that acts as a universal gateway. It uses `litellm` to translate requests to various provider formats and includes:
    *   **Batch Manager**: Optimizes high-volume embedding requests.
    *   **Detailed Logger**: Provides per-request file logging for debugging.
    *   **OpenAI-Compatible Endpoints**: `/v1/chat/completions`, `/v1/embeddings`, etc.
    *   **Model Filter GUI**: Visual interface for configuring model ignore/whitelist rules per provider (see Section 6).
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

This class is the stateful core of the library, managing concurrency, usage tracking, cooldowns, and quota resets.

#### Key Concepts

*   **Async-Native & Lazy-Loaded**: Fully asynchronous, using `aiofiles` for non-blocking file I/O. Usage data is loaded only when needed.
*   **Fine-Grained Locking**: Each API key has its own `asyncio.Lock` and `asyncio.Condition`. This allows for highly granular control.
*   **Multiple Reset Modes**: Supports three reset strategies:
    - **per_model**: Each model has independent usage window with authoritative `quota_reset_ts` (from provider errors)
    - **credential**: One window per credential with custom duration (e.g., 5 hours, 7 days)
    - **daily**: Legacy daily reset at `daily_reset_time_utc`
*   **Model Quota Groups**: Models can be grouped to share quota limits. When one model in a group hits quota, all receive the same reset timestamp.

#### Tiered Key Acquisition Strategy

The `acquire_key` method uses a sophisticated strategy to balance load:

1.  **Filtering**: Keys currently on cooldown (global or model-specific) are excluded.
2.  **Rotation Mode**: Determines credential selection strategy:
    *   **Balanced Mode** (default): Credentials sorted by usage count - least-used first for even distribution
    *   **Sequential Mode**: Credentials sorted by usage count descending - most-used first to maintain sticky behavior until exhausted
3.  **Tiering**: Valid keys are split into two tiers:
    *   **Tier 1 (Ideal)**: Keys that are completely idle (0 concurrent requests).
    *   **Tier 2 (Acceptable)**: Keys that are busy but still under their configured `MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>` limit for the requested model. This allows a single key to be used multiple times for the same model, maximizing throughput.
4.  **Selection Strategy** (configurable via `rotation_tolerance`):
    *   **Deterministic (tolerance=0.0)**: Within each tier, keys are sorted by daily usage count and the least-used key is always selected. This provides perfect load balance but predictable patterns.
    *   **Weighted Random (tolerance>0, default)**: Keys are selected randomly with weights biased toward less-used ones:
        - Formula: `weight = (max_usage - credential_usage) + tolerance + 1`
        - `tolerance=2.0` (recommended): Balanced randomness - credentials within 2 uses of the maximum can still be selected with reasonable probability
        - `tolerance=5.0+`: High randomness - even heavily-used credentials have significant probability
        - **Security Benefit**: Unpredictable selection patterns make rate limit detection and fingerprinting harder
        - **Load Balance**: Lower-usage credentials still preferred, maintaining reasonable distribution
5.  **Concurrency Limits**: Checks against `max_concurrent` limits (with priority multipliers applied) to prevent overloading a single key.
6.  **Priority Groups**: When credential prioritization is enabled, higher-tier credentials (lower priority numbers) are tried first before moving to lower tiers.

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

**Provider Support:**

The following providers implement credential prioritization:

- **Gemini CLI**: Paid tier (priority 1), Free tier (priority 2), Legacy/Unknown (priority 10). Gemini 3 models require paid tier.
- **Antigravity**: Same priority system as Gemini CLI. No model-tier restrictions (all models work on all tiers). Paid tier resets every 5 hours, free tier resets weekly.

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

The most sophisticated provider implementation, supporting Google's internal Antigravity API for Gemini 3 and Claude models (including **Claude Opus 4.5**, Anthropic's most powerful model).

#### Architecture

- **Unified Streaming/Non-Streaming**: Single code path handles both response types with optimal transformations
- **Thought Signature Caching**: Server-side caching of encrypted signatures for multi-turn Gemini 3 conversations
- **Model-Specific Logic**: Automatic configuration based on model type (Gemini 3, Claude Sonnet, Claude Opus)
- **Credential Prioritization**: Automatic tier detection with paid credentials prioritized over free (paid tier resets every 5 hours, free tier resets weekly)
- **Sequential Rotation Mode**: Default rotation mode is sequential (use credentials until exhausted) to maximize thought signature cache hits
- **Per-Model Quota Tracking**: Each model tracks independent usage windows with authoritative reset timestamps from quota errors
- **Quota Groups**: Claude models (Sonnet 4.5 + Opus 4.5) can be grouped to share quota limits (disabled by default, configurable via `QUOTA_GROUPS_ANTIGRAVITY_CLAUDE`)
- **Priority Multipliers**: Paid tier credentials get higher concurrency limits (Priority 1: 5x, Priority 2: 3x, Priority 3+: 2x in sequential mode)

#### Model Support

**Gemini 3 Pro:**
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

**Claude Opus 4.5 (NEW!):**
- Anthropic's most powerful model, now available via Antigravity proxy
- **Always uses thinking variant** - `claude-opus-4-5-thinking` is the only available variant (non-thinking version doesn't exist)
- Uses `thinkingBudget` parameter for extended thinking control (-1 for auto, 0 to disable, or specific token count)
- Full support for tool use with schema cleaning
- Same thinking preservation and sanitization features as Sonnet
- Increased default max output tokens to 64000 to accommodate thinking output

**Claude Sonnet 4.5:**
- Proxied through Antigravity API
- **Supports both thinking and non-thinking modes**:
  - With `reasoning_effort`: Uses `claude-sonnet-4-5-thinking` variant with `thinkingBudget`
  - Without `reasoning_effort`: Uses standard `claude-sonnet-4-5` variant
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
ANTIGRAVITY_CLAUDE_THINKING_SANITIZATION=true  # Enable Claude thinking mode auto-correction

# Gemini 3 tool fix customization
ANTIGRAVITY_GEMINI3_TOOL_PREFIX="gemini3_"  # Namespace prefix
ANTIGRAVITY_GEMINI3_DESCRIPTION_PROMPT="\n\nSTRICT PARAMETERS: {params}."
ANTIGRAVITY_GEMINI3_SYSTEM_INSTRUCTION="..."  # Full system prompt
```

#### Claude Extended Thinking Sanitization

The provider now includes robust automatic sanitization for Claude's extended thinking mode, handling all common error scenarios with conversation history.

**Problem**: Claude's extended thinking API requires strict consistency in thinking blocks:
- If thinking is enabled, the final assistant turn must start with a thinking block
- If thinking is disabled, no thinking blocks can be present in the final turn
- Tool use loops are part of a single "assistant turn"
- You **cannot** toggle thinking mode mid-turn (this is invalid per Claude API)

**Scenarios Handled**:

| Scenario | Action |
|----------|--------|
| Tool loop WITH thinking + thinking enabled | Preserve thinking, continue normally |
| Tool loop WITHOUT thinking + thinking enabled | **Inject synthetic closure** to start fresh turn with thinking |
| Thinking disabled | Strip all thinking blocks |
| Normal conversation (no tool loop) | Strip old thinking, new response adds thinking naturally |
| Function call ID mismatch | Three-tier recovery: ID match → name match → fallback |
| Missing tool responses | Automatic placeholder injection |
| Compacted/cached conversations | Recover thinking from cache post-transformation |

**Key Implementation Details**:

The `_sanitize_thinking_for_claude()` method now:
- Operates on Gemini-format messages (`parts[]` with `"thought": true` markers)
- Detects tool results as user messages with `functionResponse` parts
- Uses `_analyze_turn_state()` to classify conversation state on Gemini format
- Recovers thinking from cache when client strips reasoning_content
- When enabling thinking in a tool loop started without thinking:
  - Injects synthetic assistant message to close the previous turn
  - Allows Claude to start fresh turn with thinking capability

**Function Call Response Grouping**:

The enhanced pairing system ensures conversation history integrity:
```
Problem: Client/proxy may mutate response IDs or lose responses during context processing

Solution:
1. Try direct ID match (tool_call_id == response.id)
2. If no match, try function name match (tool.name == response.name)
3. If still no match, use order-based fallback (nth tool → nth response)
4. Repair "unknown_function" responses with correct names
5. Create placeholders for completely missing responses
```

**Configuration**:
```env
ANTIGRAVITY_CLAUDE_THINKING_SANITIZATION=true  # Enable/disable auto-correction (default: true)
```

**Note**: These fixes ensure Claude thinking mode works seamlessly with tool use, model switching, context compression, and cached conversations. No manual intervention required.

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

### 2.13. Sequential Rotation & Per-Model Quota Tracking

A comprehensive credential rotation and quota management system introduced in PR #31.

#### Rotation Modes

Two rotation strategies are available per provider:

**Balanced Mode (Default)**:
- Distributes load evenly across all credentials
- Least-used credentials selected first
- Best for providers with per-minute rate limits
- Prevents any single credential from being overused

**Sequential Mode**:
- Uses one credential until it's exhausted (429 quota error)
- Switches to next credential only after current one fails
- Most-used credentials selected first (sticky behavior)
- Best for providers with daily/weekly quotas
- Maximizes cache hit rates (e.g., Antigravity thought signatures)
- Default for Antigravity provider

**Configuration**:
```env
# Set per provider
ROTATION_MODE_GEMINI=sequential
ROTATION_MODE_OPENAI=balanced
ROTATION_MODE_ANTIGRAVITY=balanced  # Override default
```

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

**Google RPC Format** (Antigravity, Gemini CLI):
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
QUOTA_GROUPS_ANTIGRAVITY_CLAUDE="claude-sonnet-4-5,claude-opus-4-5"

# To disable a default group:
QUOTA_GROUPS_ANTIGRAVITY_CLAUDE=""
```

**Behavior**:
- When one model hits quota, all models in the group receive the same `quota_reset_ts`
- Combined weighted usage for credential selection (e.g., Opus counts 2x vs Sonnet)
- Group resets only when ALL models' quotas have reset
- Preserves unexpired cooldowns during other resets

**Provider Implementation**:
```python
class AntigravityProvider(ProviderInterface):
    model_quota_groups = {
        "claude": ["claude-sonnet-4-5", "claude-opus-4-5"]
    }
    
    model_usage_weights = {
        "claude-opus-4-5": 2  # Opus counts 2x vs Sonnet
    }
```

#### Priority-Based Concurrency Multipliers

Credentials can be assigned to priority tiers with configurable concurrency limits:

**Configuration**:
```env
# Universal multipliers (all modes)
CONCURRENCY_MULTIPLIER_ANTIGRAVITY_PRIORITY_1=10
CONCURRENCY_MULTIPLIER_ANTIGRAVITY_PRIORITY_2=3

# Mode-specific overrides
CONCURRENCY_MULTIPLIER_ANTIGRAVITY_PRIORITY_2_BALANCED=1  # Lower in balanced mode
```

**How it works**:
```python
effective_concurrent_limit = MAX_CONCURRENT_REQUESTS_PER_KEY * tier_multiplier
```

**Provider Defaults** (Antigravity):
- Priority 1 (paid ultra): 5x multiplier
- Priority 2 (standard paid): 3x multiplier  
- Priority 3+ (free): 2x (sequential mode) or 1x (balanced mode)

**Benefits**:
- Paid credentials handle more load without manual configuration
- Different concurrency for different rotation modes
- Automatic tier detection based on credential properties

#### Reset Window Configuration

Providers can specify custom reset windows per priority tier:

```python
class AntigravityProvider(ProviderInterface):
    usage_reset_configs = {
        frozenset([1, 2]): UsageResetConfigDef(
            mode="per_model",
            window_hours=5,  # 5-hour rolling window for paid tiers
            field_name="5h_window"
        ),
        frozenset([3, 4, 5]): UsageResetConfigDef(
            mode="per_model",
            window_hours=168,  # 7-day window for free tier
            field_name="7d_window"
        )
    }
```

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

#### OAuth Callback Port Configuration

Each OAuth provider uses a local callback server during authentication. The callback port can be customized via environment variables to avoid conflicts with other services.

**Default Ports:**

| Provider | Default Port | Environment Variable |
|----------|-------------|---------------------|
| Gemini CLI | 8085 | `GEMINI_CLI_OAUTH_PORT` |
| Antigravity | 51121 | `ANTIGRAVITY_OAUTH_PORT` |
| iFlow | 11451 | `IFLOW_OAUTH_PORT` |

**Configuration Methods:**

1. **Via TUI Settings Menu:**
   - Main Menu → `4. View Provider & Advanced Settings` → `1. Launch Settings Tool`
   - Select the provider (Gemini CLI, Antigravity, or iFlow)
   - Modify the `*_OAUTH_PORT` setting
   - Use "Reset to Default" to restore the original port

2. **Via `.env` file:**
   ```env
   # Custom OAuth callback ports (optional)
   GEMINI_CLI_OAUTH_PORT=8085
   ANTIGRAVITY_OAUTH_PORT=51121
   IFLOW_OAUTH_PORT=11451
   ```

**When to Change Ports:**

- If the default port conflicts with another service on your system
- If running multiple proxy instances on the same machine
- If firewall rules require specific port ranges

**Note:** Port changes take effect on the next OAuth authentication attempt. Existing tokens are not affected.

---


---

## 3. Provider Specific Implementations

The library handles provider idiosyncrasies through specialized "Provider" classes in `src/rotator_library/providers/`.

### 3.1. Gemini CLI (`gemini_cli_provider.py`)

The `GeminiCliProvider` is the most complex implementation, mimicking the Google Cloud Code extension.

**New in PR #31**:
- **Quota Parsing**: Implements `parse_quota_error()` using Google RPC format parser
- **Tier Configuration**: Defines `tier_priorities` and `usage_reset_configs` for automatic priority resolution
- **Balanced Rotation**: Defaults to balanced mode (unlike Antigravity which uses sequential)
- **Priority Multipliers**: Same as Antigravity (P1: 5x, P2: 3x, others: 1x)

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

### 3.2. Qwen Code (`qwen_code_provider.py`)

*   **Dual Auth**: Supports both standard API keys (direct) and OAuth (via `QwenAuthBase`).
*   **Device Flow**: Implements the OAuth Device Authorization Grant (RFC 8628). It displays a code to the user and polls the token endpoint until the user authorizes the device in their browser.
*   **Dummy Tool Injection**: To work around a Qwen API bug where streams hang if `tools` is empty but `tool_choice` logic is present, the provider injects a benign `do_not_call_me` tool.
*   **Schema Cleaning**: Recursively removes `strict` and `additionalProperties` from tool schemas, as Qwen's validation is stricter than OpenAI's.
*   **Reasoning Parsing**: Detects `<think>` tags in the raw stream and redirects their content to a separate `reasoning_content` field in the delta, mimicking the OpenAI o1 format.

### 3.3. iFlow (`iflow_provider.py`)

*   **Hybrid Auth**: Uses a custom OAuth flow (Authorization Code) to obtain an `access_token`. However, the *actual* API calls use a separate `apiKey` that is retrieved from the user's profile (`/api/oauth/getUserInfo`) using the access token.
*   **Callback Server**: The auth flow spins up a local server (default: port `11451`, configurable via `IFLOW_OAUTH_PORT`) to capture the redirect.
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

3. **Usage Tracking (Level 3)**: Usage statistics (`key_usage.json`) are maintained in memory via `ResilientStateWriter`. If the file is deleted, the system tracks usage internally and attempts to recreate the file on the next save interval. Pending writes are flushed on shutdown.

4. **Provider Cache (Level 4)**: The provider cache tracks disk health and continues operating in memory-only mode if disk writes fail. Has its own shutdown mechanism.

5. **Logging (Level 5)**: Logging is treated as non-critical. If the `logs/` directory is removed, the system attempts to recreate it. If creation fails, logging degrades gracefully without interrupting the request flow. **No buffering or retry**.

### 5.3. Component Integration

| Component | Utility Used | Behavior on Disk Failure | Shutdown Flush |
|-----------|--------------|--------------------------|----------------|
| `UsageManager` | `ResilientStateWriter` | Continues in memory, retries after 30s | Yes (via registry) |
| `GoogleOAuthBase` | `safe_write_json(buffer_on_failure=True)` | Memory cache preserved, buffered for retry | Yes (via registry) |
| `QwenAuthBase` | `safe_write_json(buffer_on_failure=True)` | Memory cache preserved, buffered for retry | Yes (via registry) |
| `IFlowAuthBase` | `safe_write_json(buffer_on_failure=True)` | Memory cache preserved, buffered for retry | Yes (via registry) |
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
- **Config Reset**: Deleting `key_usage.json` resets the persistence layer, but the running instance preserves its current in-memory counts for load balancing consistency.
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

