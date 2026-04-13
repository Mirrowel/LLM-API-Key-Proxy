# LLM API Key Proxy (Fork)

A personal fork of [Mirrowel/LLM-API-Key-Proxy](https://github.com/Mirrowel/LLM-API-Key-Proxy) with additional providers, fixes, and tooling.

> **For full documentation**, see the [upstream repository](https://github.com/Mirrowel/LLM-API-Key-Proxy).

---

## Fork-Specific Features

- **Universal Compatibility** — Works with any app supporting OpenAI or Anthropic APIs: Claude Code, Opencode, Continue, Roo/Kilo Code, Cursor, JanitorAI, SillyTavern, custom applications, and more
- **One Endpoint, Many Providers** — Configure Gemini, OpenAI, Anthropic, and [any LiteLLM-supported provider](https://docs.litellm.ai/docs/providers) once. Access them all through a single API key
- **Anthropic API Compatible** — Use Claude Code or any Anthropic SDK client with non-Anthropic providers like Gemini, OpenAI, or custom models
- **Built-in Resilience** — Automatic key rotation, failover on errors, rate limit handling, and intelligent cooldowns
- **Classifier-Scoped Routing** — Use isolated per-user/provider credential pools in the library without leaking user keys into global rotation
- **Exclusive Provider Support** — Includes custom providers not available elsewhere, including **Gemini CLI**

### Additional Providers

| Provider | Description |
|----------|-------------|
| **GitHub Copilot** | OAuth Device Flow with plan-based model filtering (free/pro/business/enterprise), premium interaction quota tracking |
| **NanoGPT** | Native Anthropic message routing, streaming fallback, embedding dispatch |
| **Kilocode** | OpenAI-compatible provider with credit balance tracking via web session cookie |
| **Chutes** | Dollar credit quota tracking with sliding window, tool-calling support |
| **Vertex AI** | Express Mode API key auth via `x-goog-api-key`, curated model list (Vertex has no `/v1/models` endpoint) |
| **Opencode Go** | 3-window quota tracking (`5hr`, `weekly`, `monthly`) via SolidJS scraping, custom OpenAI routing |
| **Command Code** | Bypasses standard subscription tier limits on chat completions by routing to the CLI endpoint (`/alpha/generate`). Supports dollar credits tracking mapped to cents baseline, 5-minute background refresh, and reasoning/thinking stream translation for `deepseek-v4-pro` and `mimo-v2.5-pro` |

### Smart "Latest" Model Aliases

Resolve virtual `latest` model names to the current best-available model at request time:

```env
# Automatically resolves at request time based on available models
MODEL_LATEST_nanogpt=nanogpt/glm-5  # "latest" resolves to current best GLM-5
```

- Cost-based tiebreaking when multiple candidates match
- On-demand model cache warming for cold starts
- Configurable per-provider resolution rules

### Usage & Quota Stats

- **Current period** vs **global/lifetime** quota split — TUI toggle between windows
- **Cached token pricing** — correct discounted rates for cached input tokens in streaming cost calculations
- **Identity-based deduplication** — OAuth credential dedup handles GitHub login (not just email)

### OpenAI Responses API Compatibility

`POST /v1/responses` — accepts the Responses API format used by codex-cli and the OpenAI Python SDK, and transparently converts it to/from Chat Completions internally. Supports streaming, tool calling, and multi-turn conversations.

### Monitoring & Health Endpoints

- `GET /v1/health` — status, uptime, provider/credential counts (add `?detail=full` for per-model window stats and error summary)
- `GET /v1/health/errors` — recent errors with optional `?provider=` and `?model=` filters
- Both endpoints are gated by `PROXY_API_KEY`

### High-Throughput Embedding Support

The proxy fully supports text embeddings under the `/v1/embeddings` OpenAI-compatible endpoint. Features include:
- **Resilient Key Rotation & Cooldowns**: Embedding requests leverage the exact same key management, error tracking, and rotation mechanics as chat completions.
- **Server-Side Batching**: Enable `USE_EMBEDDING_BATCHER=true` in `.env` to transparently queue and batch individual incoming embedding requests at the proxy layer, maximizing API throughput and key efficiency.
- **Multi-Provider Support**: Fully compatible with Google AI Studio (`google/gemini-embedding-2`, `google/gemini-embedding-001`), OpenAI, Voyage, Cohere, and other major providers.

### Quota Guards

#### Monthly Budget

Per-credential monthly spending cap. Tracks cumulative `approx_cost` across all models and blocks the credential once the budget is reached. Resets on a configurable day of the month.

Activated by setting the environment variable — **no defaults are applied**:

```bash
MONTHLY_BUDGET_VERTEX=200          # $200/month cap for all Vertex credentials
MONTHLY_BUDGET_RESET_DAY_VERTEX=1  # reset on the 1st (default, range 1-28)
```

The budget and remaining spend appear in `/v1/quota-stats` under each credential's `monthly_budget` field.

#### RPD (Requests Per Day) Limits

Per-model daily request caps, tracked per-credential. Fully configured via environment variables — no defaults are hardcoded. Counters reset at a configurable time (default: midnight Pacific).

```bash
# Per-model limits: RPD_LIMIT_{PROVIDER}_{MODEL}=limit
# Model name: uppercase, hyphens become underscores
RPD_LIMIT_GOOGLE_GEMINI_FLASH_LATEST=20
RPD_LIMIT_GOOGLE_GEMINI_FLASH_LITE_LATEST=500
RPD_LIMIT_GOOGLE_GEMINI_EMBEDDING_2=1000
RPD_LIMIT_GOOGLE_GEMMA_4_31B_IT=1500

# Model aliases: RPD_ALIAS_{PROVIDER}_{ALIAS}=canonical_name
# Aliases let "latest" model names share a counter with their resolved name
RPD_ALIAS_GOOGLE_GEMINI_FLASH_LATEST=gemini-3.5-flash
RPD_ALIAS_GOOGLE_GEMINI_FLASH_LITE_LATEST=gemini-3.1-flash-lite

# Reset settings (optional, defaults shown)
RPD_RESET_TZ_GOOGLE=America/Los_Angeles
RPD_RESET_HOUR_GOOGLE=0
```

RPD status appears in `/v1/quota-stats` under each credential's `rpd_limits` field, in the TUI summary page, and in the WebUI credential cards.

### Tooling

- **Transaction Log Viewer TUI** — Browse and inspect API request/response logs

---

## Quick Start (Docker)

```bash
docker-compose up -d
```

Or use the Komodo stack for deployment.

### Environment Variables

### The `.env` File

Credentials are stored in a `.env` file. You can edit it directly or use the TUI:

```env
# Required: Authentication key for YOUR proxy
PROXY_API_KEY="your-secret-proxy-key"

# Provider API Keys (add multiple with _1, _2, etc.)
GEMINI_API_KEY_1="your-gemini-key"
GEMINI_API_KEY_2="another-gemini-key"
OPENAI_API_KEY_1="your-openai-key"
ANTHROPIC_API_KEY_1="your-anthropic-key"
```

> Copy `.env.example` to `.env` as a starting point.

---

## The Resilience Library

The proxy is powered by a standalone Python library that you can use directly in your own applications.

### Key Features

- **Async-native** with `asyncio` and `httpx`
- **Intelligent key selection** with tiered, model-aware locking
- **Deadline-driven requests** with configurable global timeout
- **Automatic failover** between keys on errors
- **OAuth support** for Gemini CLI, Codex, Anthropic, and Copilot
- **Stateless deployment ready** — load credentials from environment variables

### Basic Usage

```python
from rotator_library import RotatingClient

client = RotatingClient(
    api_keys={"gemini": ["key1", "key2"], "openai": ["key3"]},
    global_timeout=30,
    max_retries=2
)

async with client:
    response = await client.acompletion(
        model="gemini/gemini-2.5-flash",
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

### Classifier-Scoped Routing

Applications that use `rotator_library` directly can isolate user-owned provider keys with a `classifier`. This keeps platform/global keys separate from user-owned credentials while reusing the same retry, rotation, streaming, model discovery, and usage tracking machinery.

```python
response = await client.acompletion(
    model="logfare/my-model",
    messages=[{"role": "user", "content": "Use my connected provider"}],
    classifier="user_123",
    api_keys={"logfare": ["user-logfare-key"]},
    providers={
        "logfare": {
            "base_url": "https://user-logfare.example/v1",
            "protocol": "openai_chat_completions",
        }
    },
    private=True,
)
```

Key behavior:

- No `classifier`: existing global/provider pool behavior is preserved.
- With `classifier`: only credentials supplied for that classifier/request or registered classifier are used.
- Classified requests never fall back to global API keys.
- `private=True` stores safe `private:<fingerprint>` identifiers in usage files instead of raw API keys.
- Usage files are isolated under `usage/classifiers/<safe_classifier>/usage_<provider>.json`.
- Scoped model discovery uses the same classifier/provider/key isolation and separate cache entries, including a safe fingerprint of the scoped credential set.
- Registered scope APIs let host apps add, update, fetch, and remove classifier provider configs and credentials at runtime.

See [Classifier-Scoped Routing](docs/CLASSIFIER_SCOPED_ROUTING.md) for the full API, examples, privacy rules, model discovery behavior, and limitations.

### Library Documentation

See the [Library README](src/rotator_library/README.md) for complete documentation including:
- All initialization parameters
- Streaming support
- Error handling and cooldown strategies
- Provider plugin system
- Credential prioritization
- Classifier-scoped routing and registered scope management

---

## Interactive TUI

The proxy includes a powerful text-based UI for configuration and management.

<!-- TODO: Add TUI main menu screenshot here -->

### TUI Features

- **🚀 Run Proxy** — Start the server with saved settings
- **⚙️ Configure Settings** — Host, port, API key, request logging, raw I/O logging
- **🔑 Manage Credentials** — Add/edit API keys and OAuth credentials
- **📊 View Provider & Advanced Settings** — Inspect providers and launch the settings tool
- **📈 View Quota & Usage Stats (Alpha)** — Usage, quota windows, fair-cycle status
- **🔄 Reload Configuration** — Refresh settings without restarting

### Configuration Files

| File | Contents |
|------|----------|
| `.env` | All credentials and advanced settings |
| `launcher_config.json` | TUI-specific settings (host, port, logging) |
| `quota_viewer_config.json` | Quota viewer remotes + per-provider display toggles |
| `usage/usage_<provider>.json` | Usage persistence per provider |
| `usage/classifiers/<safe_classifier>/usage_<provider>.json` | Classifier-scoped library usage persistence |

---

## Features

### Core Capabilities

- **Universal OpenAI-compatible endpoint** for all providers
- **Multi-provider support** via [LiteLLM](https://docs.litellm.ai/docs/providers) fallback
- **Automatic key rotation** and load balancing
- **Interactive TUI** for easy configuration
- **Detailed request logging** for debugging

<details>
<summary><b>🛡️ Resilience & High Availability</b></summary>

- **Global timeout** with deadline-driven retries
- **Escalating cooldowns** per model (10s → 30s → 60s → 120s)
- **Key-level lockouts** for consistently failing keys
- **Stream error detection** and graceful recovery
- **Batch embedding aggregation** for improved throughput
- **Automatic daily resets** for cooldowns and usage stats

</details>

<details>
<summary><b>🔑 Credential Management</b></summary>

- **Auto-discovery** of API keys from environment variables
- **OAuth discovery** from standard paths (`~/.gemini/`)
- **Duplicate detection** warns when same account added multiple times
- **Credential prioritization** — paid tier used before free tier
- **Stateless deployment** — export OAuth to environment variables
- **Local-first storage** — credentials isolated in `oauth_creds/` directory

</details>

<details>
<summary><b>⚙️ Advanced Configuration</b></summary>

- **Model whitelists/blacklists** with wildcard support
- **Per-provider concurrency controls** (`OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>` and `MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>`)
- **Rotation modes** — balanced (distribute load) or sequential (use until exhausted)
- **Priority multipliers** — higher concurrency for paid credentials
- **Model quota groups** — shared cooldowns for related models
- **Temperature override** — prevent tool hallucination issues
- **Weighted random rotation** — unpredictable selection patterns

</details>

<details>
<summary><b>🔌 Provider-Specific Features</b></summary>

**Gemini CLI:**

- Zero-config Google Cloud project discovery
- Internal API access with higher rate limits
- Automatic fallback to preview models on rate limit
- Paid vs free tier detection

**NVIDIA NIM:**

- Dynamic model discovery
- DeepSeek thinking support

</details>

<details>
<summary><b>📝 Logging & Debugging</b></summary>

- **Per-request file logging** with `--enable-request-logging`
- **Raw I/O logging** with `--enable-raw-logging` (proxy boundary payloads)
- **Unique request directories** with full transaction details
- **Streaming chunk capture** for debugging
- **Performance metadata** (duration, tokens, model used)
- **Provider-specific logs** for active custom providers

</details>

---

## Advanced Configuration

<details>
<summary><b>Environment Variables Reference</b></summary>

### Proxy Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `PROXY_API_KEY` | Authentication key for your proxy | Required |
| `OAUTH_REFRESH_INTERVAL` | Token refresh check interval (seconds) | `600` |
| `SKIP_OAUTH_INIT_CHECK` | Skip interactive OAuth setup on startup | `false` |

### Per-Provider Settings

| Pattern | Description | Example |
|---------|-------------|---------|
| `<PROVIDER>_API_KEY_<N>` | API key for provider | `GEMINI_API_KEY_1` |
| `OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>` | Soft spread-before-stacking target | `OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_OPENAI=1` |
| `MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>` | Hard concurrent request ceiling (`<=0` means unlimited) | `MAX_CONCURRENT_REQUESTS_PER_KEY_OPENAI=-1` |
| `ROTATION_MODE_<PROVIDER>` | `balanced` or `sequential` | `ROTATION_MODE_GEMINI=sequential` |
| `IGNORE_MODELS_<PROVIDER>` | Blacklist (comma-separated, supports `*`) | `IGNORE_MODELS_OPENAI=*-preview*` |
| `WHITELIST_MODELS_<PROVIDER>` | Whitelist (overrides blacklist) | `WHITELIST_MODELS_GEMINI=gemini-2.5-pro` |

### Advanced Features

| Variable | Description |
|----------|-------------|
| `ROTATION_TOLERANCE` | `0.0`=deterministic, `3.0`=weighted random (default) |
| `CONCURRENCY_MULTIPLIER_<PROVIDER>_PRIORITY_<N>` | Concurrency multiplier per priority tier |
| `QUOTA_GROUPS_<PROVIDER>_<GROUP>` | Models sharing quota limits |
| `SESSION_STICKY_WAIT_SECONDS[_<PROVIDER>]` | Sequential-mode wait for a session-bound key blocked only by concurrency |
| `SESSION_STICKY_ENTRY_TTL_SECONDS[_<PROVIDER>]` | TTL for in-memory session-to-credential sticky entries |
| `SESSION_STICKY_MAX_ENTRIES[_<PROVIDER>]` | Max in-memory sequential sticky entries before LRU pruning |
| `TRUSTED_SESSION_ID_FIELDS` | Comma-separated explicit request ID fields to trust as strong evidence; unset by default |
| `OVERRIDE_TEMPERATURE_ZERO` | `remove` or `set` to prevent tool hallucination |
| `GEMINI_CLI_QUOTA_REFRESH_INTERVAL` | Quota baseline refresh interval in seconds (default: 300) |

</details>

<details>
<summary><b>Model Filtering (Whitelists & Blacklists)</b></summary>

Control which models are exposed through your proxy.

### Blacklist Only

```env
# Hide all preview models
IGNORE_MODELS_OPENAI="*-preview*"
```

### Pure Whitelist Mode

```env
# Block all, then allow specific models
IGNORE_MODELS_GEMINI="*"
WHITELIST_MODELS_GEMINI="gemini-2.5-pro,gemini-2.5-flash"
```

### Exemption Mode

```env
# Block preview models, but allow one specific preview
IGNORE_MODELS_OPENAI="*-preview*"
WHITELIST_MODELS_OPENAI="gpt-4o-2024-08-06-preview"
```

**Logic order:** Whitelist check → Blacklist check → Default allow

</details>

<details>
<summary><b>Concurrency & Rotation Settings</b></summary>

### Concurrency Limits

```env
# Balanced mode defaults to optimal=1 and max=-1, so it spreads first
# but will stack on busy keys instead of blocking when every key is busy.
ROTATION_MODE_OPENAI=balanced
OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_OPENAI=1
MAX_CONCURRENT_REQUESTS_PER_KEY_OPENAI=-1

# Sequential mode defaults to optimal=-1 and max=-1 for sticky/unlimited use.
ROTATION_MODE_GEMINI=sequential
OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_GEMINI=-1
MAX_CONCURRENT_REQUESTS_PER_KEY_GEMINI=-1

# Constrained providers can set optimal and max to the same value.
MAX_CONCURRENT_REQUESTS_PER_KEY_GEMINI_CLI=1
OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_GEMINI_CLI=1

# Mode-specific forms override provider-wide values only for that mode.
MAX_CONCURRENT_REQUESTS_PER_KEY_OPENAI_BALANCED=-1
OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_OPENAI_BALANCED=1
```

`optimal` is a soft target used for capacity phases: the rotator prefers credentials below optimal, then stacks on healthy credentials when no below-optimal credential remains. `max` is the hard safety ceiling; `0` or any negative value means unlimited.

### Rotation Modes

```env
# sequential (default): Use one key until it errors/exhausts, preserving provider-side cache locality
ROTATION_MODE_GEMINI=sequential

# balanced: Distribute load evenly - opt in for per-minute rate limits
ROTATION_MODE_OPENAI=balanced
```

### Session Stickiness

Sequential mode can keep related chat requests on the same credential when the session tracker has enough scoped evidence. Tracking is provider/model/scope-bound, so classifier/private credential pools do not share sticky sessions.

```env
# Wait up to 15s for a session-bound credential if it is only busy by concurrency.
SESSION_STICKY_WAIT_SECONDS=15
SESSION_STICKY_WAIT_SECONDS_GEMINI_CLI=15

# Bound the in-memory sticky session cache.
SESSION_STICKY_ENTRY_TTL_SECONDS=3600
SESSION_STICKY_MAX_ENTRIES=10000

# Optional and conservative: only set this if your clients send stable IDs.
TRUSTED_SESSION_ID_FIELDS="conversation_id,thread_id"
```

### Priority Multipliers

Paid credentials can handle more concurrent requests. Legacy priority multipliers apply to hard max concurrency; provider-specific optimal multipliers can also raise the soft target where supported:

```env
# Priority 1: 10x concurrency
CONCURRENCY_MULTIPLIER_GEMINI_CLI_PRIORITY_1=10

# Priority 2: 3x
CONCURRENCY_MULTIPLIER_GEMINI_CLI_PRIORITY_2=3
```

### Model Quota Groups

Models sharing quota limits:

```env
# Example: group provider models that share quota
QUOTA_GROUPS_GEMINI_CLI_PRO="gemini-2.5-pro,gemini-3-pro-preview"
```

</details>

<details>
<summary><b>Timeout Configuration</b></summary>

Fine-grained control over HTTP timeouts:

```env
TIMEOUT_CONNECT=30              # Connection establishment
TIMEOUT_WRITE=30                # Request body send
TIMEOUT_POOL=60                 # Connection pool acquisition
TIMEOUT_READ_STREAMING=180      # Between streaming chunks (3 min)
TIMEOUT_READ_NON_STREAMING=600  # Full response wait (10 min)
```

**Recommendations:**

- Long thinking tasks: Increase `TIMEOUT_READ_STREAMING` to 300-360s
- Unstable network: Increase `TIMEOUT_CONNECT` to 60s
- Large outputs: Increase `TIMEOUT_READ_NON_STREAMING` to 900s+

</details>

---

## OAuth Providers

<details>
<summary><b>Gemini CLI</b></summary>

Uses Google OAuth to access internal Gemini endpoints with higher rate limits.

**Setup:**

1. Run `python -m rotator_library.credential_tool`
2. Select "Add OAuth Credential" → "Gemini CLI"
3. Complete browser authentication
4. Credentials saved to `oauth_creds/gemini_cli_oauth_1.json`

**Features:**

- Zero-config project discovery
- Automatic free-tier project onboarding
- Paid vs free tier detection
- Smart fallback on rate limits
- Quota baseline tracking with background refresh (accurate remaining quota estimates)
- Sequential rotation mode (uses credentials until quota exhausted)

**Quota Groups:** Models that share quota are automatically grouped:
- **Pro**: `gemini-2.5-pro`, `gemini-3-pro-preview`
- **2.5-Flash**: `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`
- **3-Flash**: `gemini-3-flash-preview`

All models in a group deplete the shared quota equally. 24-hour per-model quota windows.

**Environment Variables (for stateless deployment):**

Single credential (legacy):
```env
GEMINI_CLI_ACCESS_TOKEN="ya29.your-access-token"
GEMINI_CLI_REFRESH_TOKEN="1//your-refresh-token"
GEMINI_CLI_EXPIRY_DATE="1234567890000"
GEMINI_CLI_EMAIL="your-email@gmail.com"
GEMINI_CLI_PROJECT_ID="your-gcp-project-id"  # Optional
GEMINI_CLI_TIER="standard-tier"  # Optional: standard-tier or free-tier
```

Multiple credentials (use `_N_` suffix where N is 1, 2, 3...):
```env
GEMINI_CLI_1_ACCESS_TOKEN="ya29.first-token"
GEMINI_CLI_1_REFRESH_TOKEN="1//first-refresh"
GEMINI_CLI_1_EXPIRY_DATE="1234567890000"
GEMINI_CLI_1_EMAIL="first@gmail.com"
GEMINI_CLI_1_PROJECT_ID="project-1"
GEMINI_CLI_1_TIER="standard-tier"

GEMINI_CLI_2_ACCESS_TOKEN="ya29.second-token"
GEMINI_CLI_2_REFRESH_TOKEN="1//second-refresh"
GEMINI_CLI_2_EXPIRY_DATE="1234567890000"
GEMINI_CLI_2_EMAIL="second@gmail.com"
GEMINI_CLI_2_PROJECT_ID="project-2"
GEMINI_CLI_2_TIER="free-tier"
```

**Feature Toggles:**
```env
GEMINI_CLI_QUOTA_REFRESH_INTERVAL=300  # Quota refresh interval in seconds (default: 300 = 5 min)
```

</details>

<details>
<summary><b>Stateless Deployment (Export to Environment Variables)</b></summary>

For platforms without file persistence (Railway, Render, Vercel):

1. **Set up credentials locally:**

   ```bash
   python -m rotator_library.credential_tool
   # Complete OAuth flows
   ```

2. **Export to environment variables:**

   ```bash
   python -m rotator_library.credential_tool
   # Select "Export [Provider] to .env"
   ```

3. **Copy generated variables to your platform:**
   The tool creates files like `gemini_cli_credential_1.env` containing all necessary variables.

4. **Set `SKIP_OAUTH_INIT_CHECK=true`** to skip interactive validation on startup.

</details>

<details>
<summary><b>OAuth Callback Port Configuration</b></summary>

Customize OAuth callback ports if defaults conflict:

| Provider    | Default Port | Environment Variable     |
| ----------- | ------------ | ------------------------ |
| Gemini CLI  | 8085         | `GEMINI_CLI_OAUTH_PORT`  |
| Codex       | 1455         | `CODEX_OAUTH_PORT`       |

</details>

---

## Deployment

<details>
<summary><b>Command-Line Arguments</b></summary>

```bash
# GitHub Copilot (OAuth Device Flow — use credential tool to authenticate)
# Credentials stored in oauth_creds/copilot_oauth_*.json

# NanoGPT
NANOGPT_API_KEY_1=your-nanogpt-key

# Global concurrency default (max concurrent requests per key across all providers)
# Per-provider override: MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>=N
MAX_CONCURRENT_REQUESTS_PER_KEY=1

# Vertex AI (Express Mode API key)
VERTEX_PROJECT=your-default-project-id  # optional if keys embed project
VERTEX_LOCATION=global
VERTEX_API_KEY_1=your-vertex-express-key             # uses VERTEX_PROJECT
VERTEX_API_KEY_2=other-project:your-other-key        # project embedded in key

# Vertex does not expose a v1/models endpoint, so model discovery is
# not possible at runtime. A curated set of known-active models is
# provided as defaults. To override (e.g. when Google ships a new model),
# set VERTEX_MODELS to a comma-separated list of bare model names:
# VERTEX_MODELS=gemini-2.5-pro,gemini-3-flash-preview,my-new-model

# Opencode Go (scraped quota tracking)
# Format: sk-key (required) or api_key:workspace_id:auth_cookie (workspace and cookie optional)
OPENCODE_GO_API_KEY_1=sk-...
OPENCODE_GO_API_KEY_2=sk-...:wrk_...:auth=...

# Command Code
COMMAND_API_KEY_1=user_...  # Long-lived API key from CLI authentication

# KiloCode credit balance tracking (optional — proxy still works without it)
# Get token from browser cookie __Secure-next-auth.session-token on app.kilo.ai
KILO_SESSION_TOKEN=...  # Auto-refreshes on use, ~30-day TTL
KILO_QUOTA_REFRESH_INTERVAL=600  # optional, default 600s

# Per-provider retry overrides
MAX_RETRIES_NANOGPT=2

# Log rotation (set in main.py automatically)
# scripts/cleanup-logs.sh for transaction directory cleanup
```

---

## Fork Strategy

This fork is maintained as a **linear commit stack** on top of `upstream/dev` — one squashed commit per feature area, no merge commits. Changes are folded into the correct commit using `fixup!` + `git rebase --autosquash`. See `AGENTS.md` for the full workflow.

---

## License

Same as upstream — see [LICENSE](LICENSE).
