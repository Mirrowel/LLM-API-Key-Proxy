# Universal LLM API Proxy & Resilience Library [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/C0C0UZS4P)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Mirrowel/LLM-API-Key-Proxy) [![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Mirrowel/LLM-API-Key-Proxy)

## Easy Setup for Beginners (Windows)

This is the fastest way to get started.

1.  **Download the latest release** from the [GitHub Releases page](https://github.com/Mirrowel/LLM-API-Key-Proxy/releases/latest).
2.  Unzip the downloaded file.
3.  **Double-click `setup_env.bat`**. A window will open to help you add your API keys. Follow the on-screen instructions.
4.  **Double-click `proxy_app.exe`**. This will start the proxy server.

Your proxy is now running! You can now use it in your applications.

---

## Detailed Setup and Features

This project provides a powerful solution for developers building complex applications, such as agentic systems, that interact with multiple Large Language Model (LLM) providers. It consists of two distinct but complementary components:

1.  **A Universal API Proxy**: A self-hosted FastAPI application that provides a single, OpenAI-compatible endpoint for all your LLM requests. Powered by `litellm`, it allows you to seamlessly switch between different providers and models without altering your application's code.
2.  **A Resilience & Key Management Library**: The core engine that powers the proxy. This reusable Python library intelligently manages a pool of API keys to ensure your application is highly available and resilient to transient provider errors or performance issues.

## Features

-   **Universal API Endpoint**: Simplifies development by providing a single, OpenAI-compatible interface for diverse LLM providers.
-   **High Availability**: The underlying library ensures your application remains operational by gracefully handling transient provider errors and API key-specific issues.
-   **Resilient Performance**: A global timeout on all requests prevents your application from hanging on unresponsive provider APIs.
-   **Efficient Concurrency**: Maximizes throughput by allowing a single API key to handle multiple concurrent requests to different models.
-   **Intelligent Key Management**: Optimizes request distribution across your pool of keys by selecting the best available one for each call.
-   **Escalating Per-Model Cooldowns**: If a key fails for a specific model, it's placed on a temporary, escalating cooldown for that model, allowing it to be used with others.
-   **Automatic Daily Resets**: Cooldowns and usage statistics are automatically reset daily, making the system self-maintaining.
-   **Request Logging**: Optional logging of full request and response payloads for easy debugging.
-   **Provider Agnostic**: Compatible with any provider supported by `litellm`.
-   **OpenAI-Compatible Proxy**: Offers a familiar API interface with additional endpoints for model and provider discovery.

---

## 1. Quick Start (Windows Executable)

This is the fastest way to get started for most users on Windows.

1.  **Download the latest release** from the [GitHub Releases page](https://github.com/Mirrowel/LLM-API-Key-Proxy/releases/latest).
2.  Unzip the downloaded file.
3.  **Run `setup_env.bat`**. A window will open to help you add your API keys. Follow the on-screen instructions.
4.  **Run `proxy_app.exe`**. This will start the proxy server in a new terminal window.

Your proxy is now running and ready to use at `http://127.0.0.1:8000`.

---

## 2. Detailed Setup (From Source)

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

1.  **`PROXY_API_KEY`**: This is a secret key **you create**. It is used to authorize requests to *your* proxy, preventing unauthorized use.
2.  **Provider Keys**: These are the API keys you get from LLM providers (like Gemini, OpenAI, etc.). The proxy automatically finds them based on their name (e.g., `GEMINI_API_KEY_1`).

**Example `.env` configuration:**
```env
# A secret key for your proxy server to authenticate requests.
# This can be any secret string you choose.
PROXY_API_KEY="a-very-secret-and-unique-key"

# --- Provider API Keys ---
# Add your keys from various providers below.
# You can add multiple keys for one provider by numbering them (e.g., _1, _2).

GEMINI_API_KEY_1="YOUR_GEMINI_API_KEY_1"
GEMINI_API_KEY_2="YOUR_GEMINI_API_KEY_2"

OPENROUTER_API_KEY_1="YOUR_OPENROUTER_API_KEY_1"

NVIDIA_NIM_API_KEY_1="YOUR_NVIDIA_NIM_API_KEY_1"

CHUTES_API_KEY_1="YOUR_CHUTES_API_KEY_1"
```

### 3. Run the Proxy

You can run the proxy in two ways:

**A) Using the Compiled Executable (Recommended)**

A pre-compiled, standalone executable for Windows is available on the [latest GitHub Release](https://github.com/Mirrowel/LLM-API-Key-Proxy/releases/latest). This is the easiest way to get started as it requires no setup.

For the simplest experience, follow the **Easy Setup for Beginners** guide at the top of this document.

**B) Running from Source**

Start the server by running the `main.py` script directly.

```bash
python src/proxy_app/main.py
```

The proxy is now running and available at `http://127.0.0.1:8000`.

### 4. Make a Request

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

### How It Works

When a request is made to the proxy, the application uses its core resilience library to ensure the request is handled reliably:

1.  **Selects an Optimal Key**: The `UsageManager` selects the best available key from your pool. It uses a tiered locking strategy to find a healthy, available key, prioritizing those with the least recent usage. This allows for concurrent requests to different models using the same key, maximizing efficiency.
2.  **Makes the Request**: The proxy uses the acquired key to make the API call to the target provider via `litellm`.
3.  **Manages Errors Gracefully**:
    -   It uses a `classify_error` function to determine the failure type.
    -   For **transient server errors**, it retries the request with the same key using exponential backoff.
    -   For **key-specific issues (e.g., authentication or provider-side limits)**, it temporarily places that key on a cooldown for the specific model and seamlessly retries the request with the next available key from the pool.
4.  **Tracks Usage & Releases Key**: On a successful request, it records usage stats. The key is then released back into the available pool, ready for the next request.

### Command-Line Arguments and Scripts

The proxy server can be configured at runtime using the following command-line arguments:

-   `--host`: The IP address to bind the server to. Defaults to `0.0.0.0` (accessible from your local network).
-   `--port`: The port to run the server on. Defaults to `8000`.
-   `--enable-request-logging`: A flag to enable logging of full request and response payloads to the `logs/` directory. This is useful for debugging.

**Example:**
```bash
python src/proxy_app/main.py --host 127.0.0.1 --port 9999 --enable-request-logging
```

#### Windows Batch Scripts

For convenience on Windows, you can use the provided `.bat` scripts in the root directory to run the proxy with common configurations:

-   **`start_proxy.bat`**: Starts the proxy on `0.0.0.0:8000` with default settings.
-   **`start_proxy_debug_logging.bat`**: Starts the proxy and automatically enables request logging.

### Troubleshooting

-   **`401 Unauthorized`**: Ensure your `PROXY_API_KEY` is set correctly in the `.env` file and included in the `Authorization: Bearer <key>` header of your request.
-   **`500 Internal Server Error`**: Check the console logs of the `uvicorn` server for detailed error messages. This could indicate an issue with one of your provider API keys (e.g., it's invalid or has been revoked) or a problem with the provider's service.
-   **All keys on cooldown**: If you see a message that all keys are on cooldown, it means all your keys for a specific provider have recently failed. Check the `logs/` directory (if enabled) or the `key_usage.json` file for details on why the failures occurred.

---

## Library and Technical Docs

-   **Using the Library**: For documentation on how to use the `api-key-manager` library directly in your own Python projects, please refer to its [README.md](src/rotator_library/README.md).
-   **Technical Details**: For a more in-depth technical explanation of the library's architecture, components, and internal workings, please refer to the [Technical Documentation](DOCUMENTATION.md).
