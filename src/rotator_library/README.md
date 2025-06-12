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

-   **Parameters**: Accepts the same keyword arguments as `litellm.acompletion` (e.g., `messages`, `stream`). The `model` parameter is required and must be a string in the format `provider/model_name` (e.g., `"gemini/gemini-2.5-flash-preview-05-20"`, `"openrouter/google/gemini-flash-1.5"`, `"chutes/deepseek-ai/DeepSeek-R1-0528"`).
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

The library uses a dynamic plugin system. To add support for a new provider, you only need to do two things:

1.  **Create a new provider file** in `src/rotator_library/providers/` (e.g., `my_provider.py`). The name of the file (without `_provider.py`) will be used as the provider name (e.g., `my_provider`).
2.  **Implement the `ProviderInterface`**: Inside your new file, create a class that inherits from `ProviderInterface` and implements the `get_models` method.

```python
# src/rotator_library/providers/my_provider.py
from .provider_interface import ProviderInterface
from typing import List

class MyProvider(ProviderInterface):
    async def get_models(self, api_key: str) -> List[str]:
        # Logic to fetch and return a list of model names
        # The model names should be prefixed with the provider name.
        # e.g., ["my-provider/model-1", "my-provider/model-2"]
        pass
```

The system will automatically discover and register your new provider when the library is imported.

### Special Case: `chutes.ai`

The `chutes` provider is handled as a special case. Since `litellm` does not support it directly, the `RotatingClient` modifies the request by setting the `api_base` to `https://llm.chutes.ai/v1` and remapping the model from `chutes/model-name` to `openai/model-name`. This allows `chutes.ai` to be used as a custom OpenAI-compatible endpoint.

## Detailed Documentation

For a more in-depth technical explanation of the `rotating-api-key-client` library's architecture, components, and internal workings, please refer to the [Technical Documentation](../../DOCUMENTATION.md).
