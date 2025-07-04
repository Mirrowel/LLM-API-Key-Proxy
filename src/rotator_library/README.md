# Rotating API Key Client

A robust, asynchronous, and thread-safe client that intelligently rotates and retries API keys for use with `litellm`. This library is designed to make your interactions with LLM providers more resilient, concurrent, and efficient.

## Features

-   **Asynchronous by Design**: Built with `asyncio` and `httpx` for high-performance, non-blocking I/O.
-   **Advanced Concurrency Control**: A single key can be used for multiple concurrent requests to *different* models, maximizing throughput while ensuring thread safety.
-   **Smart Key Rotation**: Acquires the least-used, available key using a tiered, model-aware locking strategy.
-   **Escalating Per-Model Cooldowns**: If a key fails, it's placed on a temporary, escalating cooldown for that specific model.
-   **Automatic Retries**: Retries requests on transient server errors with exponential backoff.
-   **Detailed Usage Tracking**: Tracks daily and global usage for each key, including token counts and approximate cost.
-   **Automatic Daily Resets**: Automatically resets cooldowns and archives stats daily.
-   **Provider Agnostic**: Works with any provider supported by `litellm`.
-   **Extensible**: Easily add support for new providers through a plugin-based architecture.

## Installation

To install the library, you can install it directly from a local path, which is recommended for development.

```bash
# The -e flag installs it in "editable" mode
pip install -e .
```

## `RotatingClient` Class

This is the main class for interacting with the library. It is designed to be a long-lived object that manages its own HTTP client and key usage state.

### Initialization

```python
from rotating_api_key_client import RotatingClient

client = RotatingClient(
    api_keys: Dict[str, List[str]],
    max_retries: int = 2,
    usage_file_path: str = "key_usage.json"
)
```

-   `api_keys`: A dictionary where keys are provider names (e.g., `"openai"`, `"gemini"`) and values are lists of API keys for that provider.
-   `max_retries`: The number of times to retry a request with the *same key* if a transient server error occurs.
-   `usage_file_path`: The path to the JSON file where key usage data will be stored.

### Concurrency and Resource Management

The `RotatingClient` is asynchronous and manages an `httpx.AsyncClient` internally. It's crucial to close the client properly to release resources. This can be done manually or by using an `async with` block.

**Manual Management:**
```python
client = RotatingClient(api_keys=api_keys)
# ... use the client ...
await client.close()
```

**Recommended (`async with`):**
```python
async with RotatingClient(api_keys=api_keys) as client:
    # ... use the client ...
```

### Methods

#### `async def acompletion(self, **kwargs) -> Any:`

This is the primary method for making API calls. It's a wrapper around `litellm.acompletion` that adds the core logic for key acquisition, rotation, and retries.

-   **Parameters**: Accepts the same keyword arguments as `litellm.acompletion`. The `model` parameter is required and must be a string in the format `provider/model_name`.
-   **Returns**:
    -   For non-streaming requests, it returns the `litellm` response object.
    -   For streaming requests, it returns an async generator that yields OpenAI-compatible Server-Sent Events (SSE). The wrapper ensures that key locks are released and usage is recorded only after the stream is fully consumed.

**Example:**

```python
import asyncio
from rotating_api_key_client import RotatingClient

async def main():
    api_keys = {"gemini": ["key1", "key2"]}
    async with RotatingClient(api_keys=api_keys) as client:
        response = await client.acompletion(
            model="gemini/gemini-2.5-flash-preview-05-20",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        print(response)

asyncio.run(main())
```

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
    async def get_models(self, api_key: str, http_client: httpx.AsyncClient) -> List[str]:
        # Logic to fetch and return a list of model names
        # The model names should be prefixed with the provider name.
        # e.g., ["my-provider/model-1", "my-provider/model-2"]
        pass
```

The system will automatically discover and register your new provider.

## Detailed Documentation

For a more in-depth technical explanation of the library's architecture, including the `UsageManager`'s concurrency model and the error classification system, please refer to the [Technical Documentation](../../DOCUMENTATION.md).
