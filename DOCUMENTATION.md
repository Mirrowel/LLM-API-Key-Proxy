# Technical Documentation: `rotating-api-key-client`

This document provides a detailed technical explanation of the `rotating-api-key-client` library, its components, and its internal workings.

## 1. `client.py` - The `RotatingClient`

The `RotatingClient` is the central component of the library, orchestrating API calls, key rotation, and error handling.

### Request Lifecycle (`acompletion`)

When `acompletion` is called, it follows these steps:

1.  **Model and Provider Validation**: It first checks that a `model` is specified and extracts the provider name from it (e.g., `"gemini"` from `"gemini/gemini-2.5-flash-preview-05-20"`). It ensures that API keys for this provider are available.

2.  **Key Selection Loop**: The client enters a loop to find a valid key and complete the request.
    a.  **Get Next Smart Key**: It calls `self.usage_manager.get_next_smart_key()` to get the least-used key for the given model that is not currently on cooldown.
    b.  **No Key Available**: If all keys for the provider are on cooldown, it waits for 5 seconds before restarting the loop.

3.  **Attempt Loop**: Once a key is selected, it enters a retry loop (`for attempt in range(self.max_retries)`):
    a.  **API Call**: It calls `litellm.acompletion` with the selected key and the user-provided arguments.
    b.  **Success**:
        -   If the call is successful and **non-streaming**, it calls `self.usage_manager.record_success()`, returns the response, and the process ends.
        -   If the call is successful and **streaming**, it returns a `_streaming_wrapper` async generator. This wrapper formats the response chunks as Server-Sent Events (SSE) and calls `self.usage_manager.record_success()` only when the stream is fully consumed.
    c.  **Failure**: If an exception occurs:
        -   The failure is logged using `log_failure()`.
        -   **Server Error**: If `is_server_error()` returns `True` and there are retries left, it waits for a moment and continues to the next attempt with the *same key*.
        -   **Unrecoverable Error**: If `is_unrecoverable_error()` returns `True`, the exception is immediately raised, terminating the process.
        -   **Other Errors (Rate Limit, Auth, etc.)**: For any other error, it's considered a "rotation" error. `self.usage_manager.record_rotation_error()` is called to put the key on cooldown, and the inner `attempt` loop is broken. The outer `while` loop then continues, fetching a new key.

## 2. `usage_manager.py` - The `UsageManager`

This class is responsible for all logic related to tracking and selecting API keys.

### Key Data Structure

Usage data is stored in a JSON file (e.g., `key_usage.json`). Here's a conceptual view of its structure:

```json
{
  "api_key_1_hash": {
    "last_used": "timestamp",
    "cooldown_until": "timestamp",
    "global_usage": 150,
    "daily_usage": {
      "YYYY-MM-DD": 100
    },
    "model_usage": {
      "gemini/gemini-2.5-flash-preview-05-20": 50
    }
  }
}
```

-   **Key Hashing**: Keys are stored by their SHA256 hash to avoid exposing sensitive keys in logs or files.
-   `cooldown_until`: If a key fails, this timestamp is set. The key will not be selected until the current time is past this timestamp.
-   `model_usage`: Tracks the usage count for each specific model, which is the primary metric for the "smart" key selection.

### Core Methods

-   `get_next_smart_key()`: This is the key selection logic. It filters out any keys that are on cooldown and then finds the key with the lowest usage count for the requested `model`.
-   `record_success()`: Increments the usage counters (`global_usage`, `daily_usage`, `model_usage`) for the given key.
-   `record_rotation_error()`: Sets the `cooldown_until` timestamp for the given key, effectively taking it out of rotation for a short period.

## 3. `error_handler.py`

This module contains functions to classify exceptions returned by `litellm`.

-   `is_server_error(e)`: Checks if the exception is a transient server-side error (typically a `5xx` status code) that is worth retrying with the same key.
-   `is_unrecoverable_error(e)`: Checks for critical errors (e.g., invalid request parameters) that should immediately stop the process. Any error that is not a server error or an unrecoverable error is treated as a "rotation" error by the client.

## 4. `failure_logger.py`

-   `log_failure()`: This function logs detailed information about a failed API request to a file in the `logs/` directory. This is crucial for debugging issues with specific keys or providers. The log includes the hashed API key, the model, the error message, and the request data.

## 5. `providers/` - Provider Plugins

The provider plugin system allows for easy extension to support model list fetching from new LLM providers.

-   **`provider_interface.py`**: Defines the abstract base class `ProviderPlugin` with a single abstract method, `get_models`. Any new provider plugin must inherit from this class and implement this method.
-   **Implementations**: Each provider (e.g., `openai_provider.py`, `gemini_provider.py`) has its own file containing a class that implements the `ProviderPlugin` interface. The `get_models` method contains the specific logic to call the provider's API and return a list of their available models.
-   **`__init__.py`**: This file contains a dynamic plugin system that automatically discovers and registers any provider implementation placed in the `providers/` directory.

### Special Provider: `chutes.ai`

The `chutes` provider is handled as a special case within the `RotatingClient`. Since `litellm` does not have native support for `chutes.ai`, the client performs the following modifications at runtime:

1.  **Sets `api_base`**: It sets the `api_base` to `https://llm.chutes.ai/v1`.
2.  **Remaps the Model**: It changes the model name from `chutes/some-model` to `openai/some-model` before passing the request to `litellm`.

This allows the system to use `chutes.ai` as if it were a custom OpenAI endpoint, while still leveraging the library's key rotation and management features.
