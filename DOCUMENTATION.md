# Technical Documentation: `rotating-api-key-client`

This document provides a detailed technical explanation of the `rotating-api-key-client` library, its components, and its internal workings. The library has evolved into a sophisticated, asynchronous client for managing LLM API keys with a strong focus on concurrency, resilience, and state management.

## 1. `client.py` - The `RotatingClient`

The `RotatingClient` is the central component, orchestrating API calls, key management, and error handling. It is designed as a long-lived, async-native object.

### Core Responsibilities
-   Managing an `httpx.AsyncClient` for non-blocking HTTP requests.
-   Interfacing with the `UsageManager` to acquire and release API keys.
-   Handling provider-specific request modifications.
-   Executing API calls via `litellm` with a robust retry and rotation strategy.
-   Providing a safe wrapper for streaming responses.

### Request Lifecycle (`acompletion`)

When `acompletion` is called, it follows these steps:

1.  **Provider and Key Validation**: It extracts the provider from the `model` name and ensures keys are configured for it.

2.  **Key Acquisition Loop**: The client enters a loop to find a valid key and complete the request. It iterates through all keys for the provider until one succeeds or all have been tried.
    a.  **Acquire Best Key**: It calls `self.usage_manager.acquire_key()`. This is a blocking call that waits until a suitable key is available, based on the manager's tiered locking strategy (see `UsageManager` section).
    b.  **Prepare Request**: It prepares the `litellm` keyword arguments. This includes:
        -   **Request Sanitization**: Calling `sanitize_request_payload()` to remove parameters that might be unsupported by the target model, preventing errors.
        -   **Provider-Specific Logic**: Applying special handling for providers like Gemini (safety settings), Gemma (system prompts), and Chutes.ai (`api_base` and model name remapping).

3.  **Retry Loop**: Once a key is acquired, it enters an inner retry loop (`for attempt in range(self.max_retries)`):
    a.  **API Call**: It calls `litellm.acompletion` with the acquired key.
    b.  **Success (Non-Streaming)**:
        -   It calls `self.usage_manager.record_success()` to update usage stats and clear any cooldowns for the key-model pair.
        -   It calls `self.usage_manager.release_key()` to release the lock on the key for this model.
        -   It returns the response, and the process ends.
    c.  **Success (Streaming)**:
        -   It returns a `_safe_streaming_wrapper` async generator. This wrapper is critical:
            -   It yields SSE-formatted chunks to the consumer.
            -   After the stream is fully consumed, its `finally` block ensures that `record_success()` and `release_key()` are called. This guarantees that the key lock is held for the entire duration of the stream and released correctly, even if the consumer abandons the stream.
    d.  **Failure**: If an exception occurs:
        -   The failure is logged in detail by `log_failure()`.
        -   The exception is passed to `classify_error()` to get a structured `ClassifiedError` object.
        -   **Server Error**: If the error type is `server_error`, it waits with exponential backoff and retries the request with the *same key*.
        -   **Rotation Error (Rate Limit, Auth, etc.)**: For any other error, it's considered a rotation trigger. `self.usage_manager.record_failure()` is called to apply an escalating cooldown, and `self.usage_manager.release_key()` releases the lock. The inner `attempt` loop is broken, and the outer `while` loop continues, acquiring a new key.

## 2. `usage_manager.py` - Stateful Concurrency & Usage Management

This class is the heart of the library's state management and concurrency control. It is a stateful, async-native service that ensures keys are used efficiently and safely across multiple concurrent requests.

### Key Concepts

-   **Asynchronous Design & Lazy Loading**: The entire class is asynchronous, using `aiofiles` for non-blocking file I/O and a `_lazy_init` pattern. The usage data from the JSON file is loaded only when the first request is made.
-   **Concurrency Primitives**:
    -   **`filelock`**: A file-level lock (`.json.lock`) prevents race conditions if multiple *processes* are running and sharing the same usage file.
    -   **`asyncio.Lock` & `asyncio.Condition`**: Each key has its own `asyncio.Lock` and `asyncio.Condition` object. This enables the fine-grained, model-aware locking strategy.

### Tiered Key Acquisition (`acquire_key`)

This method implements the core logic for selecting a key. It is a "smart" blocking call.

1.  **Filtering**: It first filters out any keys that are on a global or model-specific cooldown.
2.  **Tiering**: It categorizes the remaining, valid keys into two tiers:
    -   **Tier 1 (Ideal)**: Keys that are completely free (not being used by any model).
    -   **Tier 2 (Acceptable)**: Keys that are currently in use, but for *different models* than the one being requested.
3.  **Selection**: It attempts to acquire a lock on a key, prioritizing Tier 1 over Tier 2. Within each tier, it prioritizes the least-used key.
4.  **Waiting**: If no keys in Tier 1 or Tier 2 can be locked, it means all eligible keys are currently handling requests for the *same model*. The method then `await`s on the `asyncio.Condition` of the best available key, waiting until it is notified that the key has been released.

### Failure Handling & Cooldowns (`record_failure`)

-   **Escalating Backoff**: When a failure is recorded, it applies a cooldown that increases with the number of consecutive failures for a specific key-model pair (e.g., 10s, 30s, 60s, up to 2 hours).
-   **Authentication Errors**: These are treated more severely, applying an immediate 5-minute key-level lockout.
-   **Key-Level Lockouts**: If a single key accumulates 3 or more long-term (2-hour) cooldowns across different models, the manager assumes the key is compromised or disabled and applies a 5-minute global lockout on the key.

### Data Structure

The `key_usage.json` file has a more complex structure to store this detailed state:
```json
{
  "api_key_hash": {
    "daily": {
      "date": "YYYY-MM-DD",
      "models": {
        "gemini/gemini-1.5-pro": {
          "success_count": 10,
          "prompt_tokens": 5000,
          "completion_tokens": 10000,
          "approx_cost": 0.075
        }
      }
    },
    "global": { /* ... similar to daily, but accumulates over time ... */ },
    "model_cooldowns": {
      "gemini/gemini-1.5-flash": 1719987600.0
    },
    "failures": {
      "gemini/gemini-1.5-flash": {
        "consecutive_failures": 2
      }
    },
    "key_cooldown_until": null,
    "last_daily_reset": "YYYY-MM-DD"
  }
}
```

## 3. `error_handler.py`

This module provides a centralized function, `classify_error`, which is a significant improvement over the previous boolean checks.

-   It takes a raw exception from `litellm` and returns a `ClassifiedError` data object.
-   This object contains the `error_type` (e.g., `'rate_limit'`, `'authentication'`, `'server_error'`), the original exception, the status code, and any `retry_after` information extracted from the error message.
-   This structured classification allows the `RotatingClient` to make more intelligent decisions about whether to retry with the same key or rotate to a new one.

## 4. `request_sanitizer.py` (New Module)

-   This module's purpose is to prevent `InvalidRequestError` exceptions from `litellm` that occur when a payload contains parameters not supported by the target model (e.g., sending a `thinking` parameter to a model that doesn't support it).
-   The `sanitize_request_payload` function is called just before `litellm.acompletion` to strip out any such unsupported parameters, making the system more robust.

## 5. `providers/` - Provider Plugins

The provider plugin system remains for fetching model lists. The interface now correctly specifies that the `get_models` method receives an `httpx.AsyncClient` instance, which it should use to make its API calls. This ensures all HTTP traffic goes through the client's managed session.

## 6. `proxy_app/` - The Proxy Application

The `proxy_app` directory contains the FastAPI application that serves the rotating client.

### `main.py` - The FastAPI App

This file contains the FastAPI application that exposes the `RotatingClient` through an OpenAI-compatible API.

#### Command-Line Arguments

-   `--enable-request-logging`: This flag enables logging of all incoming requests and outgoing responses to the `logs/` directory. This is useful for debugging and monitoring the proxy's activity. By default, this is disabled.
