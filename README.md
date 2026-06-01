# Universal LLM API Proxy & Resilience Library 
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/C0C0UZS4P)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Mirrowel/LLM-API-Key-Proxy) [![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Mirrowel/LLM-API-Key-Proxy)

**One proxy. Any LLM provider. Zero code changes.**

A self-hosted proxy that provides OpenAI and Anthropic compatible API endpoints for all your LLM providers. Works with any application that supports custom OpenAI or Anthropic base URLs—including Claude Code, Opencode,  and more—no code changes required in your existing tools.

This project consists of two components:

1. **The API Proxy** — A FastAPI application providing universal `/v1/chat/completions` (OpenAI) and `/v1/messages` (Anthropic) endpoints
2. **The Resilience Library** — A reusable Python library for intelligent API key management, rotation, and failover

---

## Why Use This?

- **Universal Compatibility** — Works with any app supporting OpenAI or Anthropic APIs: Claude Code, Opencode, Continue, Roo/Kilo Code, Cursor, JanitorAI, SillyTavern, custom applications, and more
- **One Endpoint, Many Providers** — Configure Gemini, OpenAI, Anthropic, and [any LiteLLM-supported provider](https://docs.litellm.ai/docs/providers) once. Access them all through a single API key
- **Anthropic API Compatible** — Use Claude Code or any Anthropic SDK client with non-Anthropic providers like Gemini, OpenAI, or custom models
- **Built-in Resilience** — Automatic key rotation, failover on errors, rate limit handling, and intelligent cooldowns
- **Classifier-Scoped Routing** — Use isolated per-user/provider credential pools in the library without leaking user keys into global rotation
- **Exclusive Provider Support** — Includes custom providers not available elsewhere, including **Gemini CLI**

---

| Provider | Description |
|----------|-------------|
| **GitHub Copilot** | OAuth Device Flow with plan-based model filtering (free/pro/business/enterprise), premium interaction quota tracking |
| **NanoGPT** | Native Anthropic message routing, streaming fallback, embedding dispatch |
| **Kilocode** | OpenAI-compatible provider with frequent free model offerings |
| **Chutes** | Dollar credit quota tracking with sliding window, tool-calling support |
| **Lightning AI** | Dollar credit quotas with date-based parsing |
| **Vertex AI** | Express Mode API key auth via `x-goog-api-key`, curated model list (Vertex has no `/v1/models` endpoint) |
| **Opencode Go** | 3-window quota tracking (`5hr`, `weekly`, `monthly`) via SolidJS scraping, custom OpenAI routing |
| **Command Code** | Bypasses standard subscription tier limits on chat completions by routing to the CLI endpoint (`/alpha/generate`). Supports dollar credits tracking mapped to cents baseline, 5-minute background refresh, and reasoning/thinking stream translation for `deepseek-v4-pro` and `mimo-v2.5-pro` |

## Quick Start

### Windows

1. **Download** the latest release from [GitHub Releases](https://github.com/Mirrowel/LLM-API-Key-Proxy/releases/latest)
2. **Unzip** the downloaded file
3. **Run** `proxy_app.exe` — the interactive TUI launcher opens

<!-- TODO: Add TUI main menu screenshot here -->

### macOS / Linux

```bash
# Download and extract the release for your platform
chmod +x proxy_app
./proxy_app
```

### Docker

**Using the pre-built image (recommended):**

```bash
# Pull and run directly
docker run -d \
  --name llm-api-proxy \
  -p 8000:8000 \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/oauth_creds:/app/oauth_creds \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/usage:/app/usage \
  -e SKIP_OAUTH_INIT_CHECK=true \
  ghcr.io/mirrowel/llm-api-key-proxy:latest
```

**Using Docker Compose:**

```bash
# Create your .env file and usage directory first, then:
cp .env.example .env
mkdir usage
docker compose up -d
```

> **Important:** Create the `usage/` directory before running Docker Compose so usage stats persist on the host.

> **Note:** For OAuth providers, complete authentication locally first using the credential tool, then mount the `oauth_creds/` directory or export credentials to environment variables.

### From Source

```bash
git clone https://github.com/Mirrowel/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/proxy_app/main.py
```

> **Tip:** Running with command-line arguments (e.g., `--host 0.0.0.0 --port 8000`) bypasses the TUI and starts the proxy directly.

---

## Connecting to the Proxy

Once the proxy is running, configure your application with these settings:

| Setting | Value |
|---------|-------|
| **Base URL / API Endpoint** | `http://127.0.0.1:8000/v1` |
| **API Key** | Your `PROXY_API_KEY` |

### Model Format: `provider/model_name`

**Important:** Models must be specified in the format `provider/model_name`. The `provider/` prefix tells the proxy which backend to route the request to.

```
gemini/gemini-2.5-flash          ← Gemini API
openai/gpt-4o                    ← OpenAI API
anthropic/claude-3-5-sonnet      ← Anthropic API
openrouter/anthropic/claude-3-opus  ← OpenRouter
gemini_cli/gemini-2.5-pro        ← Gemini CLI (OAuth)
```

### Usage Examples

<details>
<summary><b>Python (OpenAI Library)</b></summary>

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="your-proxy-api-key"
)

response = client.chat.completions.create(
    model="gemini/gemini-2.5-flash",  # provider/model format
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

</details>

<details>
<summary><b>curl</b></summary>

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "gemini/gemini-2.5-flash",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

</details>

<details>
<summary><b>JanitorAI / SillyTavern / Other Chat UIs</b></summary>

1. Go to **API Settings**
2. Select **"Proxy"** or **"Custom OpenAI"** mode
3. Configure:
   - **API URL:** `http://127.0.0.1:8000/v1`
   - **API Key:** Your `PROXY_API_KEY`
   - **Model:** `provider/model_name` (e.g., `gemini/gemini-2.5-flash`)
4. Save and start chatting

</details>

<details>
<summary><b>Continue / Cursor / IDE Extensions</b></summary>

In your configuration file (e.g., `config.json`):

```json
{
  "models": [
    {
      "title": "Gemini via Proxy",
      "provider": "openai",
      "model": "gemini/gemini-2.5-flash",
      "apiBase": "http://127.0.0.1:8000/v1",
      "apiKey": "your-proxy-api-key"
    }
  ]
}
```

</details>

<details>
<summary><b>Claude Code</b></summary>

Claude Code natively supports custom Anthropic API endpoints. The recommended setup is to edit your Claude Code `settings.json`:

```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "your-proxy-api-key",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "gemini/gemini-3-pro",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "gemini/gemini-3-flash",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "openai/gpt-5-mini"
  }
}
```

Now you can use Claude Code with Gemini, OpenAI, or any other configured provider.

</details>

<details>
<summary><b>Anthropic Python SDK</b></summary>

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://127.0.0.1:8000",
    api_key="your-proxy-api-key"
)

# Use any provider through Anthropic's API format
response = client.messages.create(
    model="gemini/gemini-3-flash",  # provider/model format
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.content[0].text)
```

</details>

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Status check — confirms proxy is running |
| `POST /v1/chat/completions` | Chat completions (OpenAI format) |
| `POST /v1/messages` | Chat completions (Anthropic format) — Claude Code compatible |
| `POST /v1/messages/count_tokens` | Count tokens for Anthropic-format requests |
| `POST /v1/embeddings` | Text embeddings |
| `GET /v1/models` | List all available models with pricing & capabilities |
| `GET /v1/models/{model_id}` | Get details for a specific model |
| `GET /v1/providers` | List configured providers |
| `POST /v1/token-count` | Calculate token count for a payload |
| `POST /v1/cost-estimate` | Estimate cost based on token counts |

> **Tip:** The `/v1/models` endpoint is useful for discovering available models in your client. Many apps can fetch this list automatically. Add `?enriched=false` for a minimal response without pricing data.

---

## Managing Credentials

The proxy includes an interactive tool for managing all your API keys and OAuth credentials.

### Using the TUI

<!-- TODO: Add TUI credentials menu screenshot here -->

1. Run the proxy without arguments to open the TUI
2. Select **"🔑 Manage Credentials"**
3. Choose to add API keys or OAuth credentials

### Using the Command Line

```bash
python -m rotator_library.credential_tool
```

### Credential Types

| Type | Providers | How to Add |
|------|-----------|------------|
| **API Keys** | Gemini, OpenAI, Anthropic, OpenRouter, Groq, Mistral, NVIDIA, Cohere, Chutes | Enter key in TUI or add to `.env` |
| **OAuth** | Gemini CLI | Interactive browser login via credential tool |

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
- **OAuth support** for Gemini CLI
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

</details>

---

## Deployment

<details>
<summary><b>Command-Line Arguments</b></summary>

```bash
python src/proxy_app/main.py [OPTIONS]

Options:
  --host TEXT                Host to bind (default: 0.0.0.0)
  --port INTEGER             Port to run on (default: 8000)
  --enable-request-logging   Enable detailed per-request logging
  --enable-raw-logging       Capture raw proxy I/O payloads
  --add-credential           Launch interactive credential setup tool
```

**Examples:**

```bash
# Run on custom port
python src/proxy_app/main.py --host 127.0.0.1 --port 9000

# Run with logging
python src/proxy_app/main.py --enable-request-logging

# Run with raw I/O logging
python src/proxy_app/main.py --enable-raw-logging

# Add credentials without starting proxy
python src/proxy_app/main.py --add-credential
```

</details>

<details>
<summary><b>Render / Railway / Vercel</b></summary>

See the [Deployment Guide](Deployment%20guide.md) for complete instructions.

**Quick Setup:**

1. Fork the repository
2. Create a `.env` file with your credentials
3. Create a new Web Service pointing to your repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn src.proxy_app.main:app --host 0.0.0.0 --port $PORT`
6. Upload `.env` as a secret file

**OAuth Credentials:**
Export OAuth credentials to environment variables using the credential tool, then add them to your platform's environment settings.

</details>

<details>
<summary><b>Docker</b></summary>

The proxy is available as a multi-architecture Docker image (amd64/arm64) from GitHub Container Registry.

**Quick Start with Docker Compose:**

```bash
# 1. Create your .env file with PROXY_API_KEY and provider keys
cp .env.example .env
nano .env

# 2. Create usage directory (usage_*.json files are created automatically)
mkdir usage

# 3. Start the proxy
docker compose up -d

# 4. Check logs
docker compose logs -f
```

> **Important:** Create the `usage/` directory before running Docker Compose so usage stats persist on the host.

**Manual Docker Run:**

```bash
# Create usage directory if it doesn't exist
mkdir usage

docker run -d \
  --name llm-api-proxy \
  --restart unless-stopped \
  -p 8000:8000 \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/oauth_creds:/app/oauth_creds \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/usage:/app/usage \
  -e SKIP_OAUTH_INIT_CHECK=true \
  -e PYTHONUNBUFFERED=1 \
  ghcr.io/mirrowel/llm-api-key-proxy:latest
```

**Development with Local Build:**

```bash
# Build and run locally
docker compose -f docker-compose.dev.yml up -d --build
```

**Volume Mounts:**

| Path             | Purpose                                |
| ---------------- | -------------------------------------- |
| `.env`           | Configuration and API keys (read-only) |
| `oauth_creds/`   | OAuth credential files (persistent)    |
| `logs/`          | Request logs and detailed logging      |
| `usage/`       | Usage statistics persistence (`usage_*.json`) |

**Image Tags:**

| Tag                     | Description                                |
| ----------------------- | ------------------------------------------ |
| `latest`                | Latest stable from `main` branch           |
| `dev-latest`            | Latest from `dev` branch                   |
| `YYYYMMDD-HHMMSS-<sha>` | Specific version with timestamp and commit |

**OAuth with Docker:**

For OAuth providers such as Gemini CLI, you must authenticate locally first:

1. Run `python -m rotator_library.credential_tool` on your local machine
2. Complete OAuth flows in browser
3. Either:
   - Mount `oauth_creds/` directory to container, or
   - Export credentials to `.env` using the export option

</details>

<details>
<summary><b>Custom VPS / Systemd</b></summary>

**Option 1: Authenticate locally, deploy credentials**

1. Complete OAuth flows on your local machine
2. Export to environment variables
3. Deploy `.env` to your server

**Option 2: SSH Port Forwarding**

```bash
# Forward callback ports through SSH
ssh -L 51121:localhost:51121 -L 8085:localhost:8085 user@your-vps

# Then run credential tool on the VPS
```

**Systemd Service:**

```ini
[Unit]
Description=LLM API Key Proxy
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/LLM-API-Key-Proxy
ExecStart=/path/to/python -m uvicorn src.proxy_app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

See [VPS Deployment](Deployment%20guide.md#appendix-deploying-to-a-custom-vps) for complete guide.

</details>

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `401 Unauthorized` | Verify `PROXY_API_KEY` matches your `Authorization: Bearer` header exactly |
| `500 Internal Server Error` | Check provider key validity; enable `--enable-request-logging` for details |
| All keys on cooldown | All keys failed recently; check `logs/detailed_logs/` for upstream errors |
| Model not found | Verify format is `provider/model_name` (e.g., `gemini/gemini-2.5-flash`) |
| OAuth callback failed | Ensure callback port (8085, 51121, 11451) isn't blocked by firewall |
| Streaming hangs | Increase `TIMEOUT_READ_STREAMING`; check provider status |

**Detailed Logs:**

When `--enable-request-logging` is enabled, check `logs/detailed_logs/` for:

- `request.json` — Exact request payload
- `final_response.json` — Complete response or error
- `streaming_chunks.jsonl` — All SSE chunks received
- `metadata.json` — Performance metrics

---

## Documentation

| Document | Description |
|----------|-------------|
| [Technical Documentation](DOCUMENTATION.md) | Architecture, internals, provider implementations |
| [Library README](src/rotator_library/README.md) | Using the resilience library directly |
| [Deployment Guide](Deployment%20guide.md) | Hosting on Render, Railway, VPS |
| [.env.example](.env.example) | Complete environment variable reference |

---

## License

This project is dual-licensed:

- **Proxy Application** (`src/proxy_app/`) — [MIT License](src/proxy_app/LICENSE)
- **Resilience Library** (`src/rotator_library/`) — [LGPL-3.0](src/rotator_library/COPYING.LESSER)
