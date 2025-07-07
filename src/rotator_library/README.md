# Rotating API Key Client

A robust, asynchronous, and thread-safe client that intelligently rotates and retries API keys for use with `litellm`. This library is designed to make your interactions with LLM providers more resilient, concurrent, and efficient.

## Key Features

-   **Asynchronous by Design**: Built with `asyncio` and `httpx` for high-performance, non-blocking I/O.
-   **Advanced Concurrency Control**: A single API key can be used for multiple concurrent requests to *different* models, maximizing throughput while ensuring thread safety. Requests for the *same model* using the same key are queued, preventing conflicts.
-   **Smart Key Rotation**: Acquires the least-used, available key using a tiered, model-aware locking strategy to distribute load evenly.
-   **Intelligent Error Handling**:
    -   **Escalating Per-Model Cooldowns**: If a key fails, it's placed on a temporary, escalating cooldown for that specific model, allowing it to continue being used for others.
    -   **Automatic Retries**: Retries requests on transient server errors (e.g., 5xx) with exponential backoff.
    -   **Key-Level Lockouts**: If a key fails across multiple models, it's temporarily taken out of rotation entirely.
-   **Robust Streaming Support**: The client includes a wrapper for streaming responses that can reassemble fragmented JSON chunks and intelligently detect and handle errors that occur mid-stream.
-   **Detailed Usage Tracking**: Tracks daily and global usage for each key, including token counts and approximate cost, persisted to a JSON file.
-   **Automatic Daily Resets**: Automatically resets cooldowns and archives stats daily to keep the system running smoothly.
-   **Provider Agnostic**: Works with any provider supported by `litellm`.
-   **Extensible**: Easily add support for new providers through a simple plugin-based architecture.

## Installation

To install the library, you can install it directly from a local path. Using the `-e` flag installs it in "editable" mode, which is recommended for development.

```bash
pip install -e .
```

## `RotatingClient` Class

This is the main class for interacting with the library. It is designed to be a long-lived object that manages its own HTTP client and key usage state.

### Initialization

```python
from rotating_api_key_client import RotatingClient
from typing import Dict, List

# Define your API keys, grouped by provider
api_keys: Dict[str, List[str]] = {
    "gemini": ["your_gemini_key_1", "your_gemini_key_2"],
    "openai": ["your_openai_key_1"],
}

client = RotatingClient(
    api_keys=api_keys,
    max_retries=2,
    usage_file_path="key_usage.json"
)
```

-   `api_keys`: A dictionary where keys are provider names (e.g., `"openai"`, `"gemini"`) and values are lists of API keys for that provider.
-   `max_retries`: The number of times to retry a request with the *same key* if a transient server error occurs.
-   `usage_file_path`: The path to the JSON file where key usage data will be stored.

### Concurrency and Resource Management

The `RotatingClient` is asynchronous and manages an `httpx.AsyncClient` internally. It's crucial to close the client properly to release resources. The recommended way is to use an `async with` block, which handles setup and teardown automatically.

```python
import asyncio

async def main():
    async with RotatingClient(api_keys=api_keys) as client:
        # ... use the client ...
        response = await client.acompletion(
            model="gemini/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response)

asyncio.run(main())
```

### Methods

#### `async def acompletion(self, **kwargs) -> Any:`

This is the primary method for making API calls. It's a wrapper around `litellm.acompletion` that adds the core logic for key acquisition, rotation, and retries.

-   **Parameters**: Accepts the same keyword arguments as `litellm.acompletion`. The `model` parameter is required and must be a string in the format `provider/model_name`.
-   **Returns**:
    -   For non-streaming requests, it returns the `litellm` response object.
    -   For streaming requests, it returns an async generator that yields OpenAI-compatible Server-Sent Events (SSE). The wrapper ensures that key locks are released and usage is recorded only after the stream is fully consumed.

**Streaming Example:**

```python
async def stream_example():
    async with RotatingClient(api_keys=api_keys) as client:
        response_stream = await client.acompletion(
            model="gemini/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Tell me a long story."}],
            stream=True
        )
        async for chunk in response_stream:
            print(chunk)

asyncio.run(stream_example())
```

#### `async def aembedding(self, **kwargs) -> Any:`

A wrapper around `litellm.aembedding` that provides the same key rotation and retry logic for embedding requests.

#### `def token_count(self, model: str, text: str = None, messages: List[Dict[str, str]] = None) -> int:`

Calculates the token count for a given text or list of messages using `litellm.token_counter`.

#### `async def get_available_models(self, provider: str) -> List[str]:`

Fetches a list of available models for a specific provider. Results are cached in memory.

#### `async def get_all_available_models(self, grouped: bool = True) -> Union[Dict[str, List[str]], List[str]]:`

Fetches a dictionary of all available models, grouped by provider, or as a single flat list if `grouped=False`.

## Error Handling and Cooldowns

The client uses a sophisticated error handling mechanism:

-   **Error Classification**: All exceptions from `litellm` are passed through a `classify_error` function to determine their type (`rate_limit`, `authentication`, `server_error`, etc.).
-   **Server Errors**: The client will retry the request with the *same key* up to `max_retries` times, using an exponential backoff strategy.
-   **Rotation Errors (Rate Limit, Auth, etc.)**: The client records the failure in the `UsageManager`, which applies an escalating cooldown to the key for that specific model. The client then immediately acquires a new key and continues its attempt to complete the request.
-   **Key-Level Lockouts**: If a key fails on multiple different models, the `UsageManager` can apply a key-level lockout, taking it out of rotation entirely for a short period.

## Extending with Provider Plugins

The library uses a dynamic plugin system. To add support for a new provider's model list, you only need to:

1.  **Create a new provider file** in `src/rotator_library/providers/` (e.g., `my_provider.py`).
2.  **Implement the `ProviderInterface`**: Inside your new file, create a class that inherits from `ProviderInterface` and implements the `get_models` method.

```python
# src/rotator_library/providers/my_provider.py
from .provider_interface import ProviderInterface
from typing import List
import httpx

class MyProvider(ProviderInterface):
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        # Logic to fetch and return a list of model names
        # The model names should be prefixed with the provider name.
        # e.g., ["my-provider/model-1", "my-provider/model-2"]
        # Example:
        # response = await client.get("https://api.myprovider.com/models", headers={"Auth": api_key})
        # return [f"my-provider/{model['id']}" for model in response.json()]
        pass
```

The system will automatically discover and register your new provider.

## Detailed Documentation

For a more in-depth technical explanation of the library's architecture, including the `UsageManager`'s concurrency model and the error classification system, please refer to the [Technical Documentation](../../DOCUMENTATION.md).
