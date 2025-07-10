# Technical Documentation: API Key Proxy & Rotator Library

This document provides a detailed technical explanation of the API Key Proxy and the `rotating-api-key-client` library, covering their architecture, components, and internal workings.

## 1. Architecture Overview

The project is a monorepo containing two primary components:

1.  **`rotator_library`**: A standalone, reusable Python library for intelligent API key rotation and management.
2.  **`proxy_app`**: A FastAPI application that consumes the `rotator_library` and exposes its functionality through an OpenAI-compatible web API.

This architecture separates the core rotation logic from the web-serving layer, making the library portable and the proxy a clean implementation of its features.

---

## 2. `rotator_library` - The Core Engine

This library is the heart of the project, containing all the logic for key rotation, usage tracking, and provider management.

### 2.1. `client.py` - The `RotatingClient`

The `RotatingClient` is the central class that orchestrates all operations. It is designed as a long-lived, async-native object.

#### Initialization

The client is initialized with your provider API keys, retry settings, and a new `global_timeout`.

```python
client = RotatingClient(
    api_keys=api_keys,
    max_retries=2,
    global_timeout=30  # in seconds
)
```

-   `global_timeout`: A crucial new parameter that sets a hard time limit for the entire request lifecycle, from the moment `acompletion` is called until a response is returned or the timeout is exceeded.

#### Core Responsibilities

*   Managing a shared `httpx.AsyncClient` for all non-blocking HTTP requests.
*   Interfacing with the `UsageManager` to acquire and release API keys.
*   Dynamically loading and using provider-specific plugins from the `providers/` directory.
*   Executing API calls via `litellm` with a robust, **deadline-driven** retry and rotation strategy.
*   Providing a safe, stateful wrapper for handling streaming responses.

#### Request Lifecycle: A Deadline-Driven Approach

The request lifecycle has been redesigned around a single, authoritative time budget to ensure predictable performance and prevent requests from hanging indefinitely.

1.  **Deadline Establishment**: The moment `acompletion` or `aembedding` is called, a `deadline` is calculated: `time.time() + self.global_timeout`. This `deadline` is the absolute point in time by which the entire operation must complete.

2.  **Deadline-Aware Key Rotation Loop**: The main `while` loop now has a critical secondary condition: `while len(tried_keys) < len(keys_for_provider) and time.time() < deadline:`. The loop will exit immediately if the `deadline` is reached, regardless of how many keys are left to try.

3.  **Deadline-Aware Key Acquisition**: The `self.usage_manager.acquire_key()` method now accepts the `deadline`. The `UsageManager` will not wait indefinitely for a key; if it cannot acquire one before the `deadline` is met, it will raise a `NoAvailableKeysError`, causing the request to fail fast with a "busy" error.

4.  **Deadline-Aware Retries**: When a transient error occurs, the client calculates the necessary `wait_time` for an exponential backoff. It then checks if this wait time fits within the remaining budget (`deadline - time.time()`).
    -   **If it fits**: It waits (`asyncio.sleep`) and retries with the same key.
    -   **If it exceeds the budget**: It skips the wait entirely, logs a warning, and immediately rotates to the next key to avoid wasting time.

5.  **Refined Error Propagation**:
    -   **Fatal Errors**: Invalid requests or authentication errors are raised immediately to the client.
    -   **Intermittent Errors**: Rate limits, server errors, and other temporary issues are now handled internally. The error is logged, the key is rotated, but the exception is **not** propagated to the end client. This prevents the client from seeing disruptive, intermittent failures.
    -   **Final Failure**: A non-streaming request will only return `None` (indicating failure) if either a) the global `deadline` is exceeded, or b) all keys for the provider have been tried and have failed. A streaming request will yield a final `[DONE]` with an error message in the same scenarios.

### 2.2. `usage_manager.py` - Stateful Concurrency & Usage Management

This class is the stateful core of the library, managing concurrency, usage, and cooldowns.

#### Key Concepts

*   **Async-Native & Lazy-Loaded**: The class is fully asynchronous, using `aiofiles` for non-blocking file I/O. The usage data from the JSON file is loaded only when the first request is made (`_lazy_init`).
*   **Fine-Grained Locking**: Each API key is associated with its own `asyncio.Lock` and `asyncio.Condition` object. This allows for a highly granular and efficient locking strategy.

#### Tiered Key Acquisition (`acquire_key`)

This method implements the intelligent logic for selecting the best key for a job, now with deadline awareness.

1.  **Deadline Enforcement**: The entire acquisition process runs in a `while time.time() < deadline:` loop. If a key cannot be found before the deadline, the method raises `NoAvailableKeysError`.
2.  **Filtering**: It first filters out any keys that are on a global or model-specific cooldown.
3.  **Tiering**: It categorizes the remaining, valid keys into two tiers:
    -   **Tier 1 (Ideal)**: Keys that are completely free (not being used by any model).
    -   **Tier 2 (Acceptable)**: Keys that are currently in use, but for *different models* than the one being requested. This allows a single key to be used for concurrent calls to, for example, `gemini-1.5-pro` and `gemini-1.5-flash`.
4.  **Selection**: It attempts to acquire a lock on a key, prioritizing Tier 1 over Tier 2. Within each tier, it prioritizes the key with the lowest usage count.
5.  **Waiting**: If no keys in Tier 1 or Tier 2 can be locked, it means all eligible keys are currently handling requests for the *same model*. The method then `await`s on the `asyncio.Condition` of the best available key. Crucially, this wait is itself timed out by the remaining request budget, preventing indefinite waits.

#### Failure Handling & Cooldowns (`record_failure`)

*   **Escalating Backoff**: When a failure is recorded, it applies a cooldown that increases with the number of consecutive failures for that specific key-model pair (e.g., 10s, 30s, 60s, up to 2 hours).
*   **Authentication Errors**: These are treated more severely, applying an immediate 5-minute key-level lockout.
*   **Key-Level Lockouts**: If a single key accumulates 3 or more long-term (2-hour) cooldowns across different models, the manager assumes the key is compromised or disabled and applies a 5-minute global lockout on the key.

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

This module provides a centralized function, `classify_error`, which is a significant improvement over simple boolean checks.

*   It takes a raw exception from `litellm` and returns a `ClassifiedError` data object.
*   This object contains the `error_type` (e.g., `'rate_limit'`, `'authentication'`), the original exception, the status code, and any `retry_after` information extracted from the error message.
*   This structured classification allows the `RotatingClient` to make more intelligent decisions about whether to retry with the same key or rotate to a new one.

### 2.4. `providers/` - Provider Plugins

The provider plugin system allows for easy extension. The `__init__.py` file in this directory dynamically scans for all modules ending in `_provider.py`, imports the provider class from each, and registers it in the `PROVIDER_PLUGINS` dictionary. This makes adding new providers as simple as dropping a new file into the directory.

---

## 3. `proxy_app` - The FastAPI Proxy

The `proxy_app` directory contains the FastAPI application that serves the `rotator_library`.

### 3.1. `main.py` - The FastAPI App

This file defines the web server and its endpoints.

#### Lifespan Management

The application uses FastAPI's `lifespan` context manager to manage the `RotatingClient` instance. The client is initialized when the application starts and gracefully closed (releasing its `httpx` resources) when the application shuts down. This ensures that a single, stateful client instance is shared across all requests.

#### Endpoints

*   `POST /v1/chat/completions`: The main endpoint for chat requests.
*   `POST /v1/embeddings`: The endpoint for creating embeddings.
*   `GET /v1/models`: Returns a list of all available models from configured providers.
*   `GET /v1/providers`: Returns a list of all configured providers.
*   `POST /v1/token-count`: Calculates the token count for a given message payload.

#### Authentication

All endpoints are protected by the `verify_api_key` dependency, which checks for a valid `Authorization: Bearer <PROXY_API_KEY>` header.

#### Streaming Response Handling

For streaming requests, the `chat_completions` endpoint returns a `StreamingResponse` whose content is generated by the `streaming_response_wrapper` function. This wrapper serves two purposes:
1.  It passes the chunks from the `RotatingClient`'s stream directly to the user.
2.  It aggregates the full response in the background so that it can be logged completely once the stream is finished.

### 3.2. `request_logger.py`

This module provides the `log_request_response` function, which writes the request and response data to a timestamped JSON file in the `logs/` directory. It handles creating separate directories for `completions` and `embeddings`.

### 3.3. `build.py`

This is a utility script for creating a standalone executable of the proxy application using PyInstaller. It includes logic to dynamically find all provider plugins and explicitly include them as hidden imports, ensuring they are bundled into the final executable.
