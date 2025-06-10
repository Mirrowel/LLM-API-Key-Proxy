# API Key Proxy with Rotating Key Library

This project provides two main components:

1.  A reusable Python library (`rotating-api-key-client`) for intelligently rotating API keys.
2.  A FastAPI proxy application that uses this library to provide an OpenAI-compatible endpoint for various LLM providers.

## Features

-   **Smart Key Rotation**: The library automatically uses the least-used key to distribute load.
-   **Automatic Retries**: Retries requests on transient server errors.
-   **Cooldowns**: Puts keys on a temporary cooldown after rate limit or authentication errors.
-   **Usage Tracking**: Tracks daily and global usage for each key.
-   **Provider Agnostic**: Works with any provider supported by `litellm`.
-   **OpenAI-Compatible Proxy**: The proxy provides a familiar API for interacting with different models.

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
│       ├── pyproject.toml
│       └── README.md
├── .env.example
├── .gitignore
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

3.  **Install the dependencies:**

    The `requirements.txt` file includes the proxy's dependencies and installs the `rotator_library` in editable mode (`-e`), so you can develop both simultaneously.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**

    Create a `.env` file by copying the `.env.example`:

    ```bash
    cp .env.example .env
    ```

    Edit the `.env` file with your API keys:

    ```
    # A secret key for your proxy to prevent unauthorized access
    PROXY_API_KEY="your-secret-proxy-key"

    # Add one or more API keys from your chosen provider (e.g., Gemini)
    # The keys will be tried in order.
    GEMINI_API_KEY_1="your-gemini-api-key-1"
    GEMINI_API_KEY_2="your-gemini-api-key-2"
    # ...and so on
    ```

## Running the Proxy

To run the proxy application:

```bash
uvicorn src.proxy_app.main:app --reload
```

The proxy will be available at `http://127.0.0.1:8000`.

## Using the Proxy

You can make requests to the proxy as if it were the OpenAI API. Make sure to include your `PROXY_API_KEY` in the `Authorization` header.

### Example with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer your-secret-proxy-key" \
-d '{
    "model": "gemini/gemini-pro",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": false
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
    "model": "gemini/gemini-pro",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "stream": False
}

response = requests.post(proxy_url, headers=headers, data=json.dumps(data))

print(response.json())
```

## Using the Library in Other Projects

The `rotating-api-key-client` library is designed to be reusable. You can find more information on how to use it in its own `README.md` file located at `src/rotator_library/README.md`.
