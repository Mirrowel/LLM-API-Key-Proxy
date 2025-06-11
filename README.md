# API Key Proxy with Rotating Key Library [![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/C0C0UZS4P)

This project provides a robust solution for managing and rotating API keys for various Large Language Model (LLM) providers. It consists of two main components:

1.  A reusable Python library (`rotating-api-key-client`) for intelligently rotating API keys.
2.  A FastAPI proxy application that uses this library to provide an OpenAI-compatible endpoint.

## Features

-   **Smart Key Rotation**: Intelligently selects the least-used API key to distribute request loads evenly.
-   **Automatic Retries**: Automatically retries requests on transient server errors (e.g., 5xx status codes).
-   **Key Cooldowns**: Temporarily disables keys that encounter rate limits or authentication errors to prevent further issues.
-   **Usage Tracking**: Monitors daily and global usage for each API key.
-   **Provider Agnostic**: Compatible with any provider supported by `litellm`.
-   **OpenAI-Compatible Proxy**: Offers a familiar API interface for seamless interaction with different models.

## How It Works

The core of this project is the `RotatingClient` library, which manages a pool of API keys. When a request is made, the client:

1.  **Selects the Best Key**: It identifies the key with the lowest usage count that is not currently in a cooldown period.
2.  **Makes the Request**: It uses the selected key to make the API call via `litellm`.
3.  **Handles Errors**:
    -   If a **retriable error** (like a 500 server error) occurs, it waits and retries the request.
    -   If a **non-retriable error** (like a rate limit or invalid key error) occurs, it places the key on a temporary cooldown and selects a new key for the next attempt.
4.  **Tracks Usage**: On a successful request, it records the usage for the key.

The FastAPI proxy application exposes this functionality through an API endpoint that mimics the OpenAI API, making it easy to integrate with existing tools and applications.

## Project Structure

```
.
├── logs/                     # Logs for failed requests
├── src/
│   ├── proxy_app/            # The FastAPI proxy application
│   │   └── main.py
│   └── rotator_library/      # The rotating-api-key-client library
│       ├── __init__.py
│       ├── client.py
│       ├── error_handler.py
│       ├── failure_logger.py
│       ├── usage_manager.py
│       ├── providers/
│       └── ...
├── .env.example
├── README.md
└── requirements.txt
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    The `requirements.txt` file includes all necessary packages and installs the `rotator_library` in editable mode (`-e`), allowing for simultaneous development of the library and the proxy.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Create a `.env` file by copying the example file:
    ```bash
    cp .env.example .env
    ```
    Edit the `.env` file to add your API keys. The proxy automatically detects keys for different providers based on the naming convention `PROVIDER_API_KEY_N`.

    ```env
    # A secret key to protect your proxy from unauthorized access
    PROXY_API_KEY="your-secret-proxy-key"

    # Add API keys for each provider. They will be rotated automatically.
    GEMINI_API_KEY_1="your-gemini-api-key-1"
    GEMINI_API_KEY_2="your-gemini-api-key-2"

    OPENAI_API_KEY_1="your-openai-api-key-1"
    ```

## Running the Proxy

To start the proxy application, run the following command:
```bash
uvicorn src.proxy_app.main:app --reload
```
The proxy will be available at `http://127.0.0.1:8000`.

## Using the Proxy

You can make requests to the proxy as if it were the OpenAI API. Remember to include your `PROXY_API_KEY` in the `Authorization` header.

The `model` parameter must be specified in the format `provider/model_name` (e.g., `gemini/gemini-2.5-flash-preview-05-20`, `openai/gpt-4`).

### Example with `curl` (Non-Streaming):
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer your-secret-proxy-key" \
-d '{
    "model": "gemini/gemini-2.5-flash-preview-05-20",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}'
```

### Example with `curl` (Streaming):
```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer your-secret-proxy-key" \
-d '{
    "model": "gemini/gemini-2.5-flash-preview-05-20",
    "messages": [{"role": "user", "content": "Write a short story about a robot."}],
    "stream": true
}'
```

### Example with Python `requests`:
```python
import requests
import json

proxy_url = "http://127.0.0.1:8000/v1/chat/completions"
proxy_key = "your-secret-proxy-key"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {proxy_key}"
}

data = {
    "model": "gemini/gemini-2.5-flash-preview-05-20",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
}

response = requests.post(proxy_url, headers=headers, data=json.dumps(data))
print(response.json())
```

## Troubleshooting

-   **`401 Unauthorized`**: Ensure your `PROXY_API_KEY` is set correctly in the `.env` file and included in the `Authorization` header of your request.
-   **`500 Internal Server Error`**: Check the console logs of the `uvicorn` server for detailed error messages. This could indicate an issue with one of your provider API keys or a problem with the provider's service.
-   **All keys on cooldown**: If you see a message that all keys are on cooldown, it means all your keys for a specific provider have recently failed. Check the `logs/` directory for details on why the failures occurred.

## Using the Library in Other Projects

The `rotating-api-key-client` is a standalone library that can be integrated into any Python project. For detailed documentation on how to use it, please refer to its `README.md` file located at `src/rotator_library/README.md`.

## Detailed Documentation

For a more in-depth technical explanation of the `rotating-api-key-client` library's architecture, components, and internal workings, please refer to the [Technical Documentation](DOCUMENTATION.md).
