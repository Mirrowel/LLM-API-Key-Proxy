# Universal LLM API Proxy & Resilience Library [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/C0C0UZS4P)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Mirrowel/LLM-API-Key-Proxy) [![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Mirrowel/LLM-API-Key-Proxy)


## Detailed Setup and Features

This project provides a powerful solution for developers building complex applications, such as agentic systems, that interact with multiple Large Language Model (LLM) providers. It consists of two distinct but complementary components:

1.  **A Universal API Proxy**: A self-hosted FastAPI application that provides a single, OpenAI-compatible endpoint for all your LLM requests. Powered by `litellm`, it allows you to seamlessly switch between different providers and models without altering your application's code.
2.  **A Resilience & Key Management Library**: The core engine that powers the proxy. This reusable Python library intelligently manages a pool of API keys to ensure your application is highly available and resilient to transient provider errors or performance issues.

## Features

-   **Universal API Endpoint**: Simplifies development by providing a single, OpenAI-compatible interface for diverse LLM providers.
-   **High Availability**: The underlying library ensures your application remains operational by gracefully handling transient provider errors and API key-specific issues.
-   **Resilient Performance**: A global timeout on all requests prevents your application from hanging on unresponsive provider APIs.
-   **Advanced Concurrency Control**: A single API key can be used for multiple concurrent requests. By default, it supports concurrent requests to *different* models. With configuration (`MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>`), it can also support multiple concurrent requests to the *same* model using the same key.
-   **Intelligent Key Management**: Optimizes request distribution across your pool of keys by selecting the best available one for each call.
-   **Automated OAuth Discovery**: Automatically discovers, validates, and manages OAuth credentials from standard provider directories (e.g., `~/.gemini/`, `~/.qwen/`, `~/.iflow/`).
-   **Stateless Deployment Support**: Deploy easily to platforms like Railway, Render, or Vercel. The new export tool converts complex OAuth credentials (Gemini CLI, Qwen, iFlow) into simple environment variables, removing the need for persistent storage or file uploads.
-   **Batch Request Processing**: Efficiently aggregates multiple embedding requests into single batch API calls, improving throughput and reducing rate limit hits.
-   **New Provider Support**: Full support for **iFlow** (API Key & OAuth), **Qwen Code** (API Key & OAuth), and **NVIDIA NIM** with DeepSeek thinking support, including special handling for their API quirks (tool schema cleaning, reasoning support, dedicated logging).
-   **Duplicate Credential Detection**: Intelligently detects if multiple local credential files belong to the same user account and logs a warning, preventing redundancy in your key pool.
-   **Escalating Per-Model Cooldowns**: If a key fails for a specific model, it's placed on a temporary, escalating cooldown for that model, allowing it to be used with others.
-   **Automatic Daily Resets**: Cooldowns and usage statistics are automatically reset daily, making the system self-maintaining.
-   **Detailed Request Logging**: Enable comprehensive logging for debugging. Each request gets its own directory with full request/response details, streaming chunks, and performance metadata.
-   **Provider Agnostic**: Compatible with any provider supported by `litellm`.
-   **OpenAI-Compatible Proxy**: Offers a familiar API interface with additional endpoints for model and provider discovery.
-   **Advanced Model Filtering**: Supports both blacklists and whitelists to give you fine-grained control over which models are available through the proxy.
-   **🆕 Interactive Launcher TUI**: Beautiful, cross-platform TUI for configuration and management with an integrated settings tool for advanced configuration.


---

## 1. Quick Start

### Windows (Simplest)

1.  **Download the latest release** from the [GitHub Releases page](https://github.com/Mirrowel/LLM-API-Key-Proxy/releases/latest).
2.  Unzip the downloaded file.
3.  **Run the executable** (run without arguments). This launches the **interactive TUI launcher** which allows you to:
    -   🚀 Run the proxy server with your configured settings
    -   ⚙️ Configure proxy settings (Host, Port, PROXY_API_KEY, Request Logging)
    -   🔑 Manage credentials (add/edit API keys & OAuth credentials)
    -   📊 View provider status and advanced settings
    -   🔧 Configure advanced settings interactively (custom API bases, model definitions, concurrency limits)
    -   🔄 Reload configuration without restarting

> **Note:** The legacy `launcher.bat` is deprecated.

### macOS / Linux

**Option A: Using the Executable (Recommended)**
If you downloaded the pre-compiled binary for your platform, no Python installation is required.

1.  **Download the latest release** from the GitHub Releases page.
2.  Open a terminal and make the binary executable:
    ```bash
    chmod +x proxy_app
    ```
3.  **Run the Interactive Launcher**:
    ```bash
    ./proxy_app
    ```
    This launches the TUI where you can configure and run the proxy.

4.  **Or run directly with arguments** to bypass the launcher:
    ```bash
    ./proxy_app --host 0.0.0.0 --port 8000
    ```

**Option B: Manual Setup (Source Code)**
If you are running from source, use these commands:

**1. Install Dependencies**
```bash
# Ensure you have Python 3.10+ installed
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Launch the Interactive TUI**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/proxy_app/main.py
```

**3. Or run directly with arguments to bypass the launcher**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/proxy_app/main.py --host 0.0.0.0 --port 8000
```
*To enable logging, add `--enable-request-logging` to the command.*

---

## 2. Interactive TUI Launcher

The proxy now includes a powerful **interactive Text User Interface (TUI)** that makes configuration and management effortless.

### Features

- **🎯 Main Menu**:
  - Run proxy server with saved settings
  - Configure proxy settings (host, port, API key, logging)
  - Manage credentials (API keys & OAuth)
  - View provider & advanced settings status
  - Reload configuration
  
- **🔧 Advanced Settings Tool**:
  - Configure custom OpenAI-compatible providers
  - Define provider models (simple or advanced JSON format)
  - Set concurrency limits per provider
  - Interactive numbered menus for easy selection
  - Pending changes system with save/discard options

- **📊 Status Dashboard**:
  - Shows configured providers and credential counts
  - Displays custom providers and API bases
  - Shows active advanced settings
  - Real-time configuration status

### How to Use

**Running without arguments launches the TUI:**
```bash
# Windows
proxy_app.exe

# macOS/Linux
./proxy_app

# From source
python src/proxy_app/main.py
```

**Running with arguments bypasses the TUI:**
```bash
# Direct startup (skips TUI)
proxy_app.exe --host 0.0.0.0 --port 8000
```

### Configuration Files

The TUI manages two configuration files:
- **`launcher_config.json`**: Stores launcher-specific settings (host, port, logging preference)
- **`.env`**: Stores all credentials and advanced settings (PROXY_API_KEY, provider credentials, custom settings)

All advanced settings configured through the TUI are stored in `.env` for compatibility with manual editing and deployment platforms.

---

## 3. Detailed Setup (From Source)

This guide is for users who want to run the proxy from the source code on any operating system.

### Step 1: Clone and Install

First, clone the repository and install the required dependencies into a virtual environment.

**Linux/macOS:**
```bash
# Clone the repository
git clone https://github.com/Mirrowel/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Windows:**
```powershell
# Clone the repository
git clone https://github.com/Mirrowel/LLM-API-Key-Proxy.git
cd LLM-API-Key-Proxy

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Keys

Create a `.env` file to store your secret keys. You can do this by copying the example file.

**Linux/macOS:**
```bash
cp .env.example .env
```

**Windows:**
```powershell
copy .env.example .env
```

Now, open the new `.env` file and add your keys.

**Refer to the `.env.example` file for the correct format and a full list of supported providers.**

The proxy supports two types of credentials:

1.  **API Keys**: Standard secret keys from providers like OpenAI, Anthropic, etc.
2.  **OAuth Credentials**: For services that use OAuth 2.0, like the Gemini CLI.

#### Automated Credential Discovery (Recommended)

For many providers, **no configuration is necessary**. The proxy automatically discovers and manages credentials from their default locations:
-   **API Keys**: Scans your environment variables for keys matching the format `PROVIDER_API_KEY_1` (e.g., `GEMINI_API_KEY_1`).
-   **OAuth Credentials**: Scans default system directories (e.g., `~/.gemini/`, `~/.qwen/`, `~/.iflow/`) for all `*.json` credential files.

You only need to create a `.env` file to set your `PROXY_API_KEY` and to override or add credentials if the automatic discovery doesn't suit your needs.

#### Interactive Credential Management Tool

The proxy includes a powerful interactive CLI tool for managing all your credentials. This is the recommended way to set up credentials:

```bash
python -m rotator_library.credential_tool
```

**Or use the TUI Launcher** (recommended):
```bash
python src/proxy_app/main.py
# Then select "3. 🔑 Manage Credentials"
```

**Main Menu Features:**

1. **Add OAuth Credential** - Interactive OAuth flow for Gemini CLI, Qwen Code, and iFlow
   - Automatically opens your browser for authentication
   - Handles the entire OAuth flow including callbacks
   - Saves credentials to the local `oauth_creds/` directory
   - For Gemini CLI: Automatically discovers or creates a Google Cloud project
   - For Qwen Code: Uses Device Code flow (you'll enter a code in your browser)
   - For iFlow: Starts a local callback server on port 11451

2. **Add API Key** - Add standard API keys for any LiteLLM-supported provider
   - Interactive prompts guide you through the process
   - Automatically saves to your `.env` file
   - Supports multiple keys per provider (numbered automatically)

3. **Export Credentials to .env** - The "Stateless Deployment" feature
   - Converts file-based OAuth credentials into environment variables
   - Essential for platforms without persistent file storage
   - Generates a ready-to-paste `.env` block for each credential

**Stateless Deployment Workflow (Railway, Render, Vercel, etc.):**

If you're deploying to a platform without persistent file storage:

1. **Setup credentials locally first**:
   ```bash
   python -m rotator_library.credential_tool
   # Select "Add OAuth Credential" and complete the flow
   ```

2. **Export to environment variables**:
   ```bash
   python -m rotator_library.credential_tool
   # Select "Export Gemini CLI to .env" (or Qwen/iFlow)
   # Choose your credential file
   ```

3. **Copy the generated output**:
   - The tool creates a file like `gemini_cli_credential_1.env`
   - Contains all necessary `GEMINI_CLI_*` variables

4. **Paste into your hosting platform**:
   - Add each variable to your platform's environment settings
   - Set `SKIP_OAUTH_INIT_CHECK=true` to skip interactive validation
   - No credential files needed; everything loads from environment variables

**Local-First OAuth Management:**

The proxy uses a "local-first" approach for OAuth credentials:

- **Local Storage**: All OAuth credentials are stored in `oauth_creds/` directory
- **Automatic Discovery**: On first run, the proxy scans system paths (`~/.gemini/`, `~/.qwen/`, `~/.iflow/`) and imports found credentials
- **Deduplication**: Intelligently detects duplicate accounts (by email/user ID) and warns you
- **Priority**: Local files take priority over system-wide credentials
- **No System Pollution**: Your project's credentials are isolated from global system credentials

**Example `.env` configuration:**
```env
# A secret key for your proxy server to authenticate requests.
# This can be any secret string you choose.
PROXY_API_KEY="a-very-secret-and-unique-key"

# --- Provider API Keys (Optional) ---
# The proxy automatically finds keys in your environment variables.
# You can also define them here. Add multiple keys by numbering them (_1, _2).
GEMINI_API_KEY_1="YOUR_GEMINI_API_KEY_1"
GEMINI_API_KEY_2="YOUR_GEMINI_API_KEY_2"
OPENROUTER_API_KEY_1="YOUR_OPENROUTER_API_KEY_1"

# --- OAuth Credentials (Optional) ---
# The proxy automatically finds credentials in standard system paths.
# You can override this by specifying a path to your credential file.
GEMINI_CLI_OAUTH_1="/path/to/your/specific/gemini_creds.json"

# --- Gemini CLI: Stateless Deployment Support ---
# For hosts without file persistence (Railway, Render, etc.), you can provide
# Gemini CLI credentials directly via environment variables:
GEMINI_CLI_ACCESS_TOKEN="ya29.your-access-token"
GEMINI_CLI_REFRESH_TOKEN="1//your-refresh-token"
GEMINI_CLI_EXPIRY_DATE="1234567890000"
GEMINI_CLI_EMAIL="your-email@gmail.com"
# Optional: GEMINI_CLI_PROJECT_ID, GEMINI_CLI_CLIENT_ID, etc.
# See IMPLEMENTATION_SUMMARY.md for full list of supported variables

# --- Dual Authentication Support ---
# Some providers (qwen_code, iflow) support BOTH OAuth and direct API keys.
# You can use either method, or mix both for credential rotation:
QWEN_CODE_API_KEY_1="your-qwen-api-key"  # Direct API key
# AND/OR use OAuth: oauth_creds/qwen_code_oauth_1.json
IFLOW_API_KEY_1="sk-your-iflow-key"      # Direct API key
# AND/OR use OAuth: oauth_creds/iflow_oauth_1.json
```

### 4. Run the Proxy

You can run the proxy in two ways:

**A) Using the Compiled Executable (Recommended)**

A pre-compiled, standalone executable for Windows is available on the [latest GitHub Release](https://github.com/Mirrowel/LLM-API-Key-Proxy/releases/latest). This is the easiest way to get started as it requires no setup.

For the simplest experience, follow the **Quick Start** guide at the top of this document.

**B) Running from Source**

Start the server by running the `main.py` script

```bash
python src/proxy_app/main.py
```
This launches the interactive TUI launcher by default. To run the proxy directly, use:

```bash
python src/proxy_app/main.py --host 0.0.0.0 --port 8000
```

The proxy is now running and available at `http://127.0.0.1:8000`.

### 5. Make a Request

You can now send requests to the proxy. The endpoint is `http://127.0.0.1:8000/v1/chat/completions`.

Remember to:
1.  Set the `Authorization` header to `Bearer your-super-secret-proxy-key`.
2.  Specify the `model` in the format `provider/model_name`.

Here is an example using `curl`:
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer your-super-secret-proxy-key" \
-d '{
    "model": "gemini/gemini-2.5-flash",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

---

## Advanced Usage

### Using with the OpenAI Python Library (Recommended)

The proxy is OpenAI-compatible, so you can use it directly with the `openai` Python client.

```python
import openai

# Point the client to your local proxy
client = openai.OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="a-very-secret-and-unique-key" # Use your PROXY_API_KEY here
)

# Make a request
response = client.chat.completions.create(
    model="gemini/gemini-2.5-flash", # Specify provider and model
    messages=[
        {"role": "user", "content": "Write a short poem about space."}
    ]
)

print(response.choices[0].message.content)
```

### Using with `curl`

```bash
You can also send requests directly using tools like `curl`.

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer a-very-secret-and-unique-key" \
-d '{
    "model": "gemini/gemini-2.5-flash",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

### Available API Endpoints

-   `POST /v1/chat/completions`: The main endpoint for making chat requests.
-   `POST /v1/embeddings`: The endpoint for creating embeddings.
-   `GET /v1/models`: Returns a list of all available models from your configured providers.
-   `GET /v1/providers`: Returns a list of all configured providers.
-   `POST /v1/token-count`: Calculates the token count for a given message payload.

---

## 4. Advanced Topics

### Batch Request Processing

The proxy includes a `Batch Manager` that optimizes high-volume embedding requests.
- **Automatic Aggregation**: Multiple individual embedding requests are automatically collected into a single batch API call.
- **Configurable**: Works out of the box, but can be tuned for specific needs.
- **Benefits**: Significantly reduces the number of HTTP requests to providers, helping you stay within rate limits while improving throughput.

### How It Works

The proxy is built on a robust architecture:

1.  **Intelligent Routing**: The `UsageManager` selects the best available key from your pool. It prioritizes idle keys first, then keys that can handle concurrency, ensuring optimal load balancing.
2.  **Resilience & Deadlines**: Every request has a strict deadline (`global_timeout`). If a provider is slow or fails, the proxy retries with a different key immediately, ensuring your application never hangs.
3.  **Batching**: High-volume embedding requests are automatically aggregated into optimized batches, reducing API calls and staying within rate limits.
4.  **Deep Observability**: (Optional) Detailed logs capture every byte of the transaction, including raw streaming chunks, for precise debugging of complex agentic interactions.

### Command-Line Arguments and Scripts

The proxy server can be configured at runtime using the following command-line arguments:

-   `--host`: The IP address to bind the server to. Defaults to `0.0.0.0` (accessible from your local network).
-   `--port`: The port to run the server on. Defaults to `8000`.
-   `--enable-request-logging`: A flag to enable detailed, per-request logging. When active, the proxy creates a unique directory for each transaction in the `logs/detailed_logs/` folder, containing the full request, response, streaming chunks, and performance metadata. This is highly recommended for debugging.

### New Provider Highlights

#### **Gemini CLI (Advanced)**
A powerful provider that mimics the Google Cloud Code extension.
-   **Zero-Config Project Discovery**: Automatically finds your Google Cloud Project ID or onboards you to a free-tier project if none exists.
-   **Internal API Access**: Uses high-limit internal endpoints (`cloudcode-pa.googleapis.com`) rather than the public Vertex AI API.
-   **Smart Rate Limiting**: Automatically falls back to preview models (e.g., `gemini-2.5-pro-preview`) if the main model hits a rate limit.

#### **Qwen Code**
-   **Dual Authentication**: Use either standard API keys or OAuth 2.0 Device Flow credentials.
-   **Schema Cleaning**: Automatically removes `strict` and `additionalProperties` from tool schemas to prevent API errors.
-   **Stream Stability**: Injects a dummy `do_not_call_me` tool to prevent stream corruption issues when no tools are provided.
-   **Reasoning Support**: Parses `<think>` tags in responses and exposes them as `reasoning_content` (similar to OpenAI's o1 format).
-   **Dedicated Logging**: Optional per-request file logging to `logs/qwen_code_logs/` for debugging.
-   **Custom Models**: Define additional models via `QWEN_CODE_MODELS` environment variable (JSON array format).

#### **iFlow**
-   **Dual Authentication**: Use either standard API keys or OAuth 2.0 Authorization Code Flow.
-   **Hybrid Auth**: OAuth flow provides an access token, but actual API calls use a separate `apiKey` retrieved from user profile.
-   **Local Callback Server**: OAuth flow runs a temporary server on port 11451 to capture the redirect.
-   **Schema Cleaning**: Same as Qwen Code - removes unsupported properties from tool schemas.
-   **Stream Stability**: Injects placeholder tools to stabilize streaming for empty tool lists.
-   **Dedicated Logging**: Optional per-request file logging to `logs/iflow_logs/` for debugging proprietary API behaviors.
-   **Custom Models**: Define additional models via `IFLOW_MODELS` environment variable (JSON array format).


### Advanced Configuration

The following advanced settings can be added to your `.env` file (or configured interactively via the TUI Settings Tool):

#### OAuth and Refresh Settings

-   **`OAUTH_REFRESH_INTERVAL`**: Controls how often (in seconds) the background refresher checks for expired OAuth tokens. Default is `3600` (1 hour).
    ```env
    OAUTH_REFRESH_INTERVAL=1800  # Check every 30 minutes
    ```

-   **`SKIP_OAUTH_INIT_CHECK`**: Set to `true` to skip the interactive OAuth setup/validation check on startup. Essential for non-interactive environments like Docker containers or CI/CD pipelines.
    ```env
    SKIP_OAUTH_INIT_CHECK=true
    ```

#### Concurrency Control

-   **`MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>`**: Set the maximum number of simultaneous requests allowed per API key for a specific provider. Default is `1` (no concurrency). Useful for high-throughput providers.
    ```env
    MAX_CONCURRENT_REQUESTS_PER_KEY_OPENAI=3
    MAX_CONCURRENT_REQUESTS_PER_KEY_ANTHROPIC=2
    MAX_CONCURRENT_REQUESTS_PER_KEY_GEMINI=1
    ```

#### Custom Model Lists

For providers that support custom model definitions (Qwen Code, iFlow), you can override the default model list:

-   **`QWEN_CODE_MODELS`**: JSON array of custom Qwen Code models. These models take priority over hardcoded defaults.
    ```env
    QWEN_CODE_MODELS='["qwen3-coder-plus", "qwen3-coder-flash", "custom-model-id"]'
    ```

-   **`IFLOW_MODELS`**: JSON array of custom iFlow models. These models take priority over hardcoded defaults.
    ```env
    IFLOW_MODELS='["glm-4.6", "qwen3-coder-plus", "deepseek-v3.2"]'
    ```

#### Provider-Specific Settings

-   **`GEMINI_CLI_PROJECT_ID`**: Manually specify a Google Cloud Project ID for Gemini CLI OAuth. Only needed if automatic discovery fails.
    ```env
    GEMINI_CLI_PROJECT_ID="your-gcp-project-id"
    ```

**Example:**
```bash
python src/proxy_app/main.py --host 127.0.0.1 --port 9999 --enable-request-logging
```


#### Windows Batch Scripts

For convenience on Windows, you can use the provided `.bat` scripts in the root directory:

-   **`launcher.bat`** *(deprecated)*: Legacy launcher with manual menu system. Still functional but superseded by the new TUI.

### Troubleshooting

-   **`401 Unauthorized`**: Ensure your `PROXY_API_KEY` is set correctly in the `.env` file and included in the `Authorization: Bearer <key>` header of your request.
-   **`500 Internal Server Error`**: Check the console logs of the `uvicorn` server for detailed error messages. This could indicate an issue with one of your provider API keys (e.g., it's invalid or has been revoked) or a problem with the provider's service. If you have logging enabled (`--enable-request-logging`), inspect the `final_response.json` and `metadata.json` files in the corresponding log directory under `logs/detailed_logs/` for the specific error returned by the upstream provider.
-   **All keys on cooldown**: If you see a message that all keys are on cooldown, it means all your keys for a specific provider have recently failed. If you have logging enabled (`--enable-request-logging`), check the `logs/detailed_logs/` directory to find the logs for the failed requests and inspect the `final_response.json` to see the underlying error from the provider.

---

## Library and Technical Docs

-   **Using the Library**: For documentation on how to use the `api-key-manager` library directly in your own Python projects, please refer to its [README.md](src/rotator_library/README.md).
-   **Technical Details**: For a more in-depth technical explanation of the library's architecture, components, and internal workings, please refer to the [Technical Documentation](DOCUMENTATION.md).

### Advanced Model Filtering (Whitelists & Blacklists)

The proxy provides a powerful way to control which models are available to your applications using environment variables in your `.env` file.

#### How It Works

The filtering logic is applied in this order:

1.  **Whitelist Check**: If a provider has a whitelist defined (`WHITELIST_MODELS_<PROVIDER>`), any model on that list will **always be available**, even if it's on the blacklist.
2.  **Blacklist Check**: For any model *not* on the whitelist, the proxy checks the blacklist (`IGNORE_MODELS_<PROVIDER>`). If the model is on the blacklist, it will be hidden.
3.  **Default**: If a model is on neither list, it will be available.

This allows for two powerful patterns:

#### Use Case 1: Pure Whitelist Mode

You can expose *only* the specific models you want. To do this, set the blacklist to `*` to block all models by default, and then add the desired models to the whitelist.

**Example `.env`:**
```env
# Block all Gemini models by default
IGNORE_MODELS_GEMINI="*"

# Only allow gemini-1.5-pro and gemini-1.5-flash
WHITELIST_MODELS_GEMINI="gemini-1.5-pro-latest,gemini-1.5-flash-latest"
```

#### Use Case 2: Exemption Mode

You can block a broad category of models and then use the whitelist to make specific exceptions.

**Example `.env`:**
```env
# Block all preview models from OpenAI
IGNORE_MODELS_OPENAI="*-preview*"

# But make an exception for a specific preview model you want to test
WHITELIST_MODELS_OPENAI="gpt-4o-2024-08-06-preview"
```
