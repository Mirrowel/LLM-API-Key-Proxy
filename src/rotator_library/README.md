# Rotating API Key Client

A simple, thread-safe client that intelligently rotates and retries API keys for use with `litellm`. This library is designed to make your interactions with LLM providers more resilient and efficient.

## Features

-   **Smart Key Rotation**: Automatically uses the least-used key to distribute load.
-   **Automatic Retries**: Retries requests on transient server errors.
-   **Per-Model Cooldowns**: If a key fails for a specific model (e.g., due to rate limits), it is only put on cooldown for that model, allowing it to be used with other models.
-   **Usage Tracking**: Tracks daily and global usage for each key.
-   **Provider Agnostic**: Works with any provider supported by `litellm`.
-   **Extensible**: Easily add support for new providers through a plugin-based architecture.

## Installation

To install the library, you can install it directly from a local path, which is recommended for development.

```bash
# The -e flag installs it in "editable" mode
pip install -e .
```

## `RotatingClient` Class

This is the main class for interacting with the library.

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

### Methods

#### `async def acompletion(self, **kwargs) -> Any:`

This is the primary method for making API calls. It's a wrapper around `litellm.acompletion` that adds key rotation and retry logic.

-   **Parameters**: Accepts the same keyword arguments as `litellm.acompletion` (e.g., `messages`, `stream`). The `model` parameter is required and must be a string in the format `provider/model_name` (e.g., `"gemini/gemini-2.5-flash-preview-05-20"`).
-   **Returns**:
    -   For non-streaming requests, it returns the `litellm` response object.
    -   For streaming requests, it returns an async generator that yields OpenAI-compatible Server-Sent Events (SSE).

**Example:**

```python
import asyncio
from rotating_api_key_client import RotatingClient

async def main():
    api_keys = {"gemini": ["key1", "key2"]}
    client = RotatingClient(api_keys=api_keys)

    response = await client.acompletion(
        model="gemini/gemini-2.5-flash-preview-05-20",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response)

asyncio.run(main())
```

#### `def token_count(self, model: str, text: str = None, messages: List[Dict[str, str]] = None) -> int:`

Calculates the token count for a given text or list of messages using `litellm.token_counter`.
The `model` parameter is required and must be a string in the format `provider/model_name` (e.g., `"gemini/gemini-2.5-flash-preview-05-20"`).
**Example:**

```python
count = client.token_count(
    model="gemini/gemini-2.5-flash-preview-05-20",
    messages=[{"role": "user", "content": "Count these tokens."}]
)
print(f"Token count: {count}")
```

#### `async def get_available_models(self, provider: str) -> List[str]:`

Fetches a list of available models for a specific provider. Results are cached.

#### `async def get_all_available_models(self) -> Dict[str, List[str]]:`

Fetches a dictionary of all available models, grouped by provider.

## Error Handling and Cooldowns

The client is designed to handle errors gracefully:

-   **Server Errors (`5xx`)**: The client will retry the request with the *same key* up to `max_retries` times.
-   **Rate Limit / Auth Errors**: These are considered "rotation" errors. The client will immediately place the failing key on a temporary cooldown for that specific model and retry the request with a different key. This ensures that a single model failure does not sideline a key for all other models.
-   **Unrecoverable Errors**: For critical errors, the client will fail fast and raise the exception.

Cooldowns are managed by the `UsageManager` on a per-model basis, preventing failing keys from being used repeatedly for models they have recently failed with. Upon a successful call, any existing cooldown for that key-model pair is cleared.

## Extending with Provider Plugins

You can add support for fetching model lists from new providers by creating a custom provider plugin.

1.  **Create a new provider file** in `src/rotator_library/providers/`, for example, `my_provider.py`.
2.  **Implement the `ProviderPlugin` interface**:

    ```python
    # src/rotator_library/providers/my_provider.py
    from .provider_interface import ProviderPlugin
    from typing import List

    class MyProvider(ProviderPlugin):
        async def get_models(self, api_key: str) -> List[str]:
            # Logic to fetch and return a list of model names
            # e.g., ["my-provider/model-1", "my-provider/model-2"]
            pass
    ```

3.  **Register the plugin** in `src/rotator_library/providers/__init__.py`:

    ```python
    # src/rotator_library/providers/__init__.py
    from .openai_provider import OpenAIProvider
    from .gemini_provider import GeminiProvider
    from .my_provider import MyProvider # Import your new provider

    PROVIDER_PLUGINS = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "my_provider": MyProvider, # Add it to the dictionary
    }
    ```

The `RotatingClient` will automatically use your new plugin when `get_available_models` is called for `"my_provider"`.

## Detailed Documentation

For a more in-depth technical explanation of the `rotating-api-key-client` library's architecture, components, and internal workings, please refer to the [Technical Documentation](../../DOCUMENTATION.md).
