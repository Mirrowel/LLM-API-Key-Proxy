# API Key Proxy with Rotating Key Library [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/C0C0UZS4P)

## Easy Setup for Beginners (Windows)

This is the fastest way to get started.

1.  **Download the latest release** from the [GitHub Releases page](https://github.com/Mirrowel/LLM-API-Key-Proxy/releases/latest).
2.  Unzip the downloaded file.
3.  **Double-click `setup_env.bat`**. A window will open to help you add your API keys. Follow the on-screen instructions.
4.  **Double-click `proxy_app.exe`**. This will start the proxy server.

Your proxy is now running! You can now use it in your applications.

---

## Detailed Setup and Features

This project provides a robust solution for managing and rotating API keys for various Large Language Model (LLM) providers. It consists of two main components:

1.  A reusable Python library (`rotating-api-key-client`) for intelligently rotating API keys.
2.  A FastAPI proxy application that uses this library to provide an OpenAI-compatible endpoint.

## Features

-   **Advanced Concurrency Control**: A single API key can handle multiple concurrent requests to different models, maximizing throughput.
-   **Smart Key Rotation**: Intelligently selects the least-used, available API key to distribute request loads evenly.
-   **Escalating Per-Model Cooldowns**: If a key fails for a specific model (e.g., due to rate limits), it's placed on a temporary, escalating cooldown for that model, allowing it to be used with others.
-   **Automatic Retries**: Automatically retries requests on transient server errors (e.g., 5xx status codes) with exponential backoff.
-   **Automatic Daily Resets**: Cooldowns and usage statistics are automatically reset daily, making the system self-maintaining.
-   **Request Logging**: Optional logging of full request and response payloads for easy debugging.
-   **Provider Agnostic**: Compatible with any provider supported by `litellm`.
-   **OpenAI-Compatible Proxy**: Offers a familiar API interface with additional endpoints for model and provider discovery.

## Quick Start Guide

This guide will get you up and running in just a few minutes.

### 1. Setup

First, clone the repository and install the required dependencies.

**For Linux/macOS:**
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

**For Windows:**
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

### 2. Configure API Keys

Next, create your `.env` file by copying the provided example. This file is where you will store all your secret keys.

**For Linux/macOS:**
```bash
cp .env.example .env
```

**For Windows:**
```powershell
copy .env.example .env
```

Now, open the new `.env` file and replace the placeholder values with your actual API keys.

**Refer to the `.env.example` file for the correct format and a full list of supported providers.**

The two main types of keys are:

1.  **`PROXY_API_KEY`**: This is a secret key *you create*. It is used to authorize requests to *your* proxy, preventing unauthorized use.
2.  **Provider Keys**: These are the API keys you get from LLM providers (like Gemini, OpenAI, etc.). The proxy automatically finds them based on their name (e.g., `GEMINI_API_KEY_1`).

**Example `.env` configuration:**
```env
# A secret key for your proxy server to authenticate requests.
# This can be any secret string you choose.
PROXY_API_KEY="YOUR_PROXY_API_KEY"

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

Start the FastAPI server with `uvicorn`. The `--reload` flag will automatically restart the server when you make code changes.

```bash
uvicorn src.proxy_app.main:app --reload
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
    "model": "gemini/gemini-2.5-flash-preview-05-20",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

---

## Advanced Usage

### Using with the OpenAI Python Library

The proxy is OpenAI-compatible, so you can use it directly with the `openai` Python client. This is the recommended way to integrate the proxy into your applications.

```python
import openai

# Point the client to your local proxy
client = openai.OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="your-super-secret-proxy-key" # Use your proxy key here
)

# Make a request
response = client.chat.completions.create(
    model="gemini/gemini-2.5-flash-preview-05-20", # Specify provider and model
    messages=[
        {"role": "user", "content": "Write a short poem about space."}
    ]
)

print(response.choices[0].message.content)
```

### Available API Endpoints

-   `POST /v1/chat/completions`: The main endpoint for making chat requests.
-   `GET /v1/models`: Returns a list of all available models from your configured providers.
-   `GET /v1/providers`: Returns a list of all configured providers.
-   `POST /v1/token-count`: Calculates the token count for a given message payload.

### Enabling Request Logging

For debugging purposes, you can log the full request and response for every API call. To enable this, start the proxy with the `--enable-request-logging` flag:

**When running from source:**
```bash
uvicorn src.proxy_app.main:app --reload -- --enable-request-logging
```

**When running the executable:**
```powershell
./proxy_app.exe --enable-request-logging
```

Logs will be saved in the `logs/` directory.

## How It Works

The core of this project is the `RotatingClient` library, which manages a pool of API keys with a sophisticated concurrency model. When a request is made, the client:

1.  **Acquires the Best Key**: It requests the best available key from the `UsageManager`. The manager uses a tiered locking strategy to find a key that is not on cooldown and preferably not in use. If a key is busy with another request for the *same model*, it waits. Otherwise, it allows concurrent use for *different models*.
2.  **Makes the Request**: It uses the acquired key to make the API call via `litellm`.
3.  **Handles Errors**:
    -   It uses a `classify_error` function to determine the failure type.
    -   For **server errors**, it retries the request with the same key using exponential backoff.
    -   For **rate-limit or auth errors**, it records the failure, applies an escalating cooldown for that specific key-model pair, and the client immediately tries the next available key.
4.  **Tracks Usage & Releases Key**: On a successful request, it records usage stats. The key's lock is then released, notifying any waiting requests that it is available.

## Troubleshooting

-   **`401 Unauthorized`**: Ensure your `PROXY_API_KEY` is set correctly in the `.env` file and included in the `Authorization` header of your request.
-   **`500 Internal Server Error`**: Check the console logs of the `uvicorn` server for detailed error messages. This could indicate an issue with one of your provider API keys or a problem with the provider's service.
-   **All keys on cooldown**: If you see a message that all keys are on cooldown, it means all your keys for a specific provider have recently failed. Check the `logs/` directory for details on why the failures occurred.

## Library and Technical Docs

-   **Using the Library**: For documentation on how to use the `rotating-api-key-client` library directly in your own Python projects, please refer to its [README.md](src/rotator_library/README.md).
-   **Technical Details**: For a more in-depth technical explanation of the library's architecture, components, and internal workings, please refer to the [Technical Documentation](DOCUMENTATION.md).
