# Universal LLM API Proxy & Resilience Library

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/C0C0UZS4P)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Mirrowel/LLM-API-Key-Proxy) [![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Mirrowel/LLM-API-Key-Proxy)

**One proxy. Any LLM provider. Zero code changes.**

A self-hosted proxy that provides a single, OpenAI-compatible API endpoint for all your LLM providers. Works with any application that supports custom OpenAI base URLs—no code changes required in your existing tools.

This project consists of two components:

1. **The API Proxy** — A FastAPI application providing a universal `/v1/chat/completions` endpoint
2. **The Resilience Library** — A reusable Python library for intelligent API key management, rotation, and failover

---

## Why Use This?

- **Universal Compatibility** — Works with any app supporting OpenAI-compatible APIs: Opencode, Continue, Roo/Kilo Code, JanitorAI, SillyTavern, custom applications, and more
- **One Endpoint, Many Providers** — Configure Gemini, OpenAI, Anthropic, and [any LiteLLM-supported provider](https://docs.litellm.ai/docs/providers) once. Access them all through a single API key
- **Built-in Resilience** — Automatic key rotation, failover on errors, rate limit handling, and intelligent cooldowns
- **Exclusive Provider Support** — Includes custom providers not available elsewhere: **Antigravity** (Gemini 3 + Claude Sonnet/Opus 4.5), **Gemini CLI**, **Qwen Code**, and **iFlow**

---

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
  -e SKIP_OAUTH_INIT_CHECK=true \
  ghcr.io/mirrowel/llm-api-key-proxy:latest
```

**Using Docker Compose:**

```bash
# Create your .env file and key_usage.json first, then:
cp .env.example .env
touch key_usage.json
docker compose up -d
```

> **Important:** You must create both `.env` and `key_usage.json` files before running Docker Compose. If `key_usage.json` doesn't exist, Docker will create it as a directory instead of a file, causing errors.

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

| Setting                     | Value                      |
| --------------------------- | -------------------------- |
| **Base URL / API Endpoint** | `http://127.0.0.1:8000/v1` |
| **API Key**                 | Your `PROXY_API_KEY`       |

### Model Format: `provider/model_name`

**Important:** Models must be specified in the format `provider/model_name`. The `provider/` prefix tells the proxy which backend to route the request to.

```
gemini/gemini-2.5-flash          ← Gemini API
openai/gpt-4o                    ← OpenAI API
anthropic/claude-3-5-sonnet      ← Anthropic API
openrouter/anthropic/claude-3-opus  ← OpenRouter
gemini_cli/gemini-2.5-pro        ← Gemini CLI (OAuth)
antigravity/gemini-3-pro-preview ← Antigravity (Gemini 3, Claude Opus 4.5)
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

### API Endpoints

| Endpoint                    | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| `GET /`                     | Status check — confirms proxy is running              |
| `POST /v1/chat/completions` | Chat completions (main endpoint)                      |
| `POST /v1/embeddings`       | Text embeddings                                       |
| `GET /v1/models`            | List all available models with pricing & capabilities |
| `GET /v1/models/{model_id}` | Get details for a specific model                      |
| `GET /v1/providers`         | List configured providers                             |
| `POST /v1/token-count`      | Calculate token count for a payload                   |
| `POST /v1/cost-estimate`    | Estimate cost based on token counts                   |

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

| Type         | Providers                                                                    | How to Add                                    |
| ------------ | ---------------------------------------------------------------------------- | --------------------------------------------- |
| **API Keys** | Gemini, OpenAI, Anthropic, OpenRouter, Groq, Mistral, NVIDIA, Cohere, Chutes | Enter key in TUI or add to `.env`             |
| **OAuth**    | Gemini CLI, Antigravity, Qwen Code, iFlow                                    | Interactive browser login via credential tool |

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
- **OAuth support** for Gemini CLI, Antigravity, Qwen, iFlow
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

### Library Documentation

See the [Library README](src/rotator_library/README.md) for complete documentation including:

- All initialization parameters
- Streaming support
- Error handling and cooldown strategies
- Provider plugin system
- Credential prioritization

---

## Interactive TUI

The proxy includes a powerful text-based UI for configuration and management.

<!-- TODO: Add TUI main menu screenshot here -->

### TUI Features

- **🚀 Run Proxy** — Start the server with saved settings
- **⚙️ Configure Settings** — Host, port, API key, request logging
- **🔑 Manage Credentials** — Add/edit API keys and OAuth credentials
- **📊 View Status** — See configured providers and credential counts
- **🔧 Advanced Settings** — Custom providers, model definitions, concurrency

### Configuration Files

| File                   | Contents                                    |
| ---------------------- | ------------------------------------------- |
| `.env`                 | All credentials and advanced settings       |
| `launcher_config.json` | TUI-specific settings (host, port, logging) |

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
- **OAuth discovery** from standard paths (`~/.gemini/`, `~/.qwen/`, `~/.iflow/`)
- **Duplicate detection** warns when same account added multiple times
- **Credential prioritization** — paid tier used before free tier
- **Stateless deployment** — export OAuth to environment variables
- **Local-first storage** — credentials isolated in `oauth_creds/` directory

</details>

<details>
<summary><b>⚙️ Advanced Configuration</b></summary>

- **Model whitelists/blacklists** with wildcard support
- **Per-provider concurrency limits** (`MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>`)
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

**Antigravity:**

- Gemini 3 Pro with `thinkingLevel` support
- Gemini 2.5 Flash/Flash Lite with thinking mode
- Claude Opus 4.5 (thinking mode)
- Claude Sonnet 4.5 (thinking and non-thinking)
- GPT-OSS 120B Medium
- Thought signature caching for multi-turn conversations
- Tool hallucination prevention
- Quota baseline tracking with background refresh
- Parallel tool usage instruction injection
- **Quota Groups**: Models that share quota are automatically grouped:
  - Claude/GPT-OSS: `claude-sonnet-4-5`, `claude-opus-4-5`, `gpt-oss-120b-medium`
  - Gemini 3 Pro: `gemini-3-pro-high`, `gemini-3-pro-low`, `gemini-3-pro-preview`
  - Gemini 2.5 Flash: `gemini-2.5-flash`, `gemini-2.5-flash-thinking`, `gemini-2.5-flash-lite`
  - All models in a group deplete the usage of the group equally. So in claude group - it is beneficial to use only Opus, and forget about Sonnet and GPT-OSS.

**Qwen Code:**

- Dual auth (API key + OAuth Device Flow)
- `<think>` tag parsing as `reasoning_content`
- Tool schema cleaning

**iFlow:**

- Dual auth (API key + OAuth Authorization Code)
- Hybrid auth with separate API key fetch
- Tool schema cleaning

**NVIDIA NIM:**

- Dynamic model discovery
- DeepSeek thinking support

</details>

<details>
<summary><b>📝 Logging & Debugging</b></summary>

- **Per-request file logging** with `--enable-request-logging`
- **Unique request directories** with full transaction details
- **Streaming chunk capture** for debugging
- **Performance metadata** (duration, tokens, model used)
- **Provider-specific logs** for Qwen, iFlow, Antigravity

</details>

---

## Advanced Configuration

<details>
<summary><b>Environment Variables Reference</b></summary>

### Proxy Settings

| Variable                 | Description                             | Default  |
| ------------------------ | --------------------------------------- | -------- |
| `PROXY_API_KEY`          | Authentication key for your proxy       | Required |
| `OAUTH_REFRESH_INTERVAL` | Token refresh check interval (seconds)  | `600`    |
| `SKIP_OAUTH_INIT_CHECK`  | Skip interactive OAuth setup on startup | `false`  |

### Per-Provider Settings

| Pattern                                      | Description                               | Example                                    |
| -------------------------------------------- | ----------------------------------------- | ------------------------------------------ |
| `<PROVIDER>_API_KEY_<N>`                     | API key for provider                      | `GEMINI_API_KEY_1`                         |
| `MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>` | Concurrent request limit                  | `MAX_CONCURRENT_REQUESTS_PER_KEY_OPENAI=3` |
| `ROTATION_MODE_<PROVIDER>`                   | `balanced` or `sequential`                | `ROTATION_MODE_GEMINI=sequential`          |
| `IGNORE_MODELS_<PROVIDER>`                   | Blacklist (comma-separated, supports `*`) | `IGNORE_MODELS_OPENAI=*-preview*`          |
| `WHITELIST_MODELS_<PROVIDER>`                | Whitelist (overrides blacklist)           | `WHITELIST_MODELS_GEMINI=gemini-2.5-pro`   |

### Advanced Features

| Variable                                         | Description                                          |
| ------------------------------------------------ | ---------------------------------------------------- |
| `ROTATION_TOLERANCE`                             | `0.0`=deterministic, `3.0`=weighted random (default) |
| `CONCURRENCY_MULTIPLIER_<PROVIDER>_PRIORITY_<N>` | Concurrency multiplier per priority tier             |
| `QUOTA_GROUPS_<PROVIDER>_<GROUP>`                | Models sharing quota limits                          |
| `OVERRIDE_TEMPERATURE_ZERO`                      | `remove` or `set` to prevent tool hallucination      |

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
# Allow 3 concurrent requests per OpenAI key
MAX_CONCURRENT_REQUESTS_PER_KEY_OPENAI=3

# Default is 1 (no concurrency)
MAX_CONCURRENT_REQUESTS_PER_KEY_GEMINI=1
```

### Rotation Modes

```env
# balanced (default): Distribute load evenly - best for per-minute rate limits
ROTATION_MODE_OPENAI=balanced

# sequential: Use until exhausted - best for daily/weekly quotas
ROTATION_MODE_GEMINI=sequential
```

### Priority Multipliers

Paid credentials can handle more concurrent requests:

```env
# Priority 1 (paid ultra): 10x concurrency
CONCURRENCY_MULTIPLIER_ANTIGRAVITY_PRIORITY_1=10

# Priority 2 (standard paid): 3x
CONCURRENCY_MULTIPLIER_ANTIGRAVITY_PRIORITY_2=3
```

### Model Quota Groups

Models sharing quota limits:

```env
# Claude models share quota - when one hits limit, both cool down
QUOTA_GROUPS_ANTIGRAVITY_CLAUDE="claude-sonnet-4-5,claude-opus-4-5"
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

**Environment Variables (for stateless deployment):**

```env
GEMINI_CLI_ACCESS_TOKEN="ya29.your-access-token"
GEMINI_CLI_REFRESH_TOKEN="1//your-refresh-token"
GEMINI_CLI_EXPIRY_DATE="1234567890000"
GEMINI_CLI_EMAIL="your-email@gmail.com"
GEMINI_CLI_PROJECT_ID="your-gcp-project-id"  # Optional
```

</details>

<details>
<summary><b>Antigravity (Gemini 3 + Claude Opus 4.5)</b></summary>

Access Google's internal Antigravity API for cutting-edge models.

**Supported Models:**

- **Gemini 3 Pro** — with `thinkingLevel` support (low/high)
- **Gemini 2.5 Flash** — with thinking mode support
- **Gemini 2.5 Flash Lite** — configurable thinking budget
- **Claude Opus 4.5** — Anthropic's most powerful model (thinking mode only)
- **Claude Sonnet 4.5** — supports both thinking and non-thinking modes
- **GPT-OSS 120B** — OpenAI-compatible model

**Setup:**

1. Run `python -m rotator_library.credential_tool`
2. Select "Add OAuth Credential" → "Antigravity"
3. Complete browser authentication

**Advanced Features:**

- Thought signature caching for multi-turn conversations
- Tool hallucination prevention via parameter signature injection
- Automatic thinking block sanitization for Claude
- Credential prioritization (paid resets every 5 hours, free weekly)
- Quota baseline tracking with background refresh (accurate remaining quota estimates)
- Parallel tool usage instruction injection for Claude

**Environment Variables:**

```env
ANTIGRAVITY_ACCESS_TOKEN="ya29.your-access-token"
ANTIGRAVITY_REFRESH_TOKEN="1//your-refresh-token"
ANTIGRAVITY_EXPIRY_DATE="1234567890000"
ANTIGRAVITY_EMAIL="your-email@gmail.com"

# Feature toggles
ANTIGRAVITY_ENABLE_SIGNATURE_CACHE=true
ANTIGRAVITY_GEMINI3_TOOL_FIX=true
ANTIGRAVITY_QUOTA_REFRESH_INTERVAL=300  # Quota refresh interval (seconds)
ANTIGRAVITY_PARALLEL_TOOL_INSTRUCTION_CLAUDE=true  # Parallel tool instruction for Claude
```

> **Note:** Gemini 3 models require a paid-tier Google Cloud project.

</details>

<details>
<summary><b>Qwen Code</b></summary>

Uses OAuth Device Flow for Qwen/Dashscope APIs.

**Setup:**

1. Run the credential tool
2. Select "Add OAuth Credential" → "Qwen Code"
3. Enter the code displayed in your browser
4. Or add API key directly: `QWEN_CODE_API_KEY_1="your-key"`

**Features:**

- Dual auth (API key or OAuth)
- `<think>` tag parsing as `reasoning_content`
- Automatic tool schema cleaning
- Custom models via `QWEN_CODE_MODELS` env var

</details>

<details>
<summary><b>iFlow</b></summary>

Uses OAuth Authorization Code flow with local callback server.

**Setup:**

1. Run the credential tool
2. Select "Add OAuth Credential" → "iFlow"
3. Complete browser authentication (callback on port 11451)
4. Or add API key directly: `IFLOW_API_KEY_1="sk-your-key"`

**Features:**

- Dual auth (API key or OAuth)
- Hybrid auth (OAuth token fetches separate API key)
- Automatic tool schema cleaning
- Custom models via `IFLOW_MODELS` env var

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
| Antigravity | 51121        | `ANTIGRAVITY_OAUTH_PORT` |
| iFlow       | 11451        | `IFLOW_OAUTH_PORT`       |

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
  --add-credential           Launch interactive credential setup tool
```

**Examples:**

```bash
# Run on custom port
python src/proxy_app/main.py --host 127.0.0.1 --port 9000

# Run with logging
python src/proxy_app/main.py --enable-request-logging

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

# 2. Create key_usage.json file (required before first run)
touch key_usage.json

# 3. Start the proxy
docker compose up -d

# 4. Check logs
docker compose logs -f
```

> **Important:** You must create `key_usage.json` before running Docker Compose. If this file doesn't exist on the host, Docker will create it as a directory instead of a file, causing the container to fail.

**Manual Docker Run:**

```bash
# Create key_usage.json if it doesn't exist
touch key_usage.json

docker run -d \
  --name llm-api-proxy \
  --restart unless-stopped \
  -p 8000:8000 \
  -v $(pwd)/.env:/app/.env:ro \
  -v $(pwd)/oauth_creds:/app/oauth_creds \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/key_usage.json:/app/key_usage.json \
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
| `key_usage.json` | Usage statistics persistence           |

**Image Tags:**

| Tag                     | Description                                |
| ----------------------- | ------------------------------------------ |
| `latest`                | Latest stable from `main` branch           |
| `dev-latest`            | Latest from `dev` branch                   |
| `YYYYMMDD-HHMMSS-<sha>` | Specific version with timestamp and commit |

**OAuth with Docker:**

For OAuth providers (Antigravity, Gemini CLI, etc.), you must authenticate locally first:

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

| Issue                       | Solution                                                                   |
| --------------------------- | -------------------------------------------------------------------------- |
| `401 Unauthorized`          | Verify `PROXY_API_KEY` matches your `Authorization: Bearer` header exactly |
| `500 Internal Server Error` | Check provider key validity; enable `--enable-request-logging` for details |
| All keys on cooldown        | All keys failed recently; check `logs/detailed_logs/` for upstream errors  |
| Model not found             | Verify format is `provider/model_name` (e.g., `gemini/gemini-2.5-flash`)   |
| OAuth callback failed       | Ensure callback port (8085, 51121, 11451) isn't blocked by firewall        |
| Streaming hangs             | Increase `TIMEOUT_READ_STREAMING`; check provider status                   |

**Detailed Logs:**

When `--enable-request-logging` is enabled, check `logs/detailed_logs/` for:

- `request.json` — Exact request payload
- `final_response.json` — Complete response or error
- `streaming_chunks.jsonl` — All SSE chunks received
- `metadata.json` — Performance metrics

---

## Documentation

| Document                                        | Description                                       |
| ----------------------------------------------- | ------------------------------------------------- |
| [Technical Documentation](DOCUMENTATION.md)     | Architecture, internals, provider implementations |
| [Library README](src/rotator_library/README.md) | Using the resilience library directly             |
| [Deployment Guide](Deployment%20guide.md)       | Hosting on Render, Railway, VPS                   |
| [.env.example](.env.example)                    | Complete environment variable reference           |

---

## License

This project is dual-licensed:

- **Proxy Application** (`src/proxy_app/`) — [MIT License](src/proxy_app/LICENSE)
- **Resilience Library** (`src/rotator_library/`) — [LGPL-3.0](src/rotator_library/COPYING.LESSER)
