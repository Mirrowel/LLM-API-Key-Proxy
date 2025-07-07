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

#### Core Responsibilities

*   Managing a shared `httpx.AsyncClient` for all non-blocking HTTP requests.
*   Interfacing with the `UsageManager` to acquire and release API keys.
*   Dynamically loading and using provider-specific plugins from the `providers/` directory.
*   Executing API calls via `litellm` with a robust retry and rotation strategy.
*   Providing a safe, stateful wrapper for handling streaming responses.

#### Request Lifecycle (`acompletion` & `aembedding`)

When `acompletion` or `aembedding` is called, it follows a sophisticated, multi-layered process:

1.  **Provider & Key Validation**: It extracts the provider from the `model` name (e.g., `"gemini/gemini-1.5-pro"` -> `"gemini"`) and ensures keys are configured for it.

2.  **Key Acquisition Loop**: The client enters a `while` loop that attempts to find a valid key and complete the request. It iterates until one key succeeds or all have been tried.
    a.  **Acquire Best Key**: It calls `self.usage_manager.acquire_key()`. This is a crucial, potentially blocking call that waits until a suitable key is available, based on the manager's tiered locking strategy (see `UsageManager` section).
    b.  **Prepare Request**: It prepares the `litellm` keyword arguments. This includes applying provider-specific logic (e.g., remapping safety settings for Gemini, handling `api_base` for Chutes.ai) and sanitizing the payload to remove unsupported parameters.

3.  **Retry Loop**: Once a key is acquired, it enters an inner `for` loop (`for attempt in range(self.max_retries)`):
    a.  **API Call**: It calls `litellm.acompletion` or `litellm.aembedding`.
    b.  **Success (Non-Streaming)**:
        -   It calls `self.usage_manager.record_success()` to update usage stats and clear any cooldowns.
        -   It calls `self.usage_manager.release_key()` to release the lock.
        -   It returns the response, and the process ends.
    c.  **Success (Streaming)**:
        -   It returns the `_safe_streaming_wrapper` async generator. This wrapper is critical:
            -   It yields SSE-formatted chunks to the consumer.
            -   It can reassemble fragmented JSON chunks and detect errors mid-stream.
            -   Its `finally` block ensures that `record_success()` and `release_key()` are called *only after the stream is fully consumed or closed*. This guarantees the key lock is held for the entire duration of the stream.
    d.  **Failure**: If an exception occurs:
        -   The exception is passed to `classify_error()` to get a structured `ClassifiedError` object.
        -   **Server Error**: If the error is temporary (e.g., 5xx), it waits with exponential backoff and retries the request with the *same key*.
        -   **Rotation Error (Rate Limit, Auth, etc.)**: For any other error, it's a trigger to rotate. `self.usage_manager.record_failure()` is called to apply a cooldown, and the lock is released. The inner `attempt` loop is broken, and the outer `while` loop continues, acquiring a new key.

### 2.2. `usage_manager.py` - Stateful Concurrency & Usage Management

This class is the stateful core of the library, managing concurrency, usage, and cooldowns.

#### Key Concepts

*   **Async-Native & Lazy-Loaded**: The class is fully asynchronous, using `aiofiles` for non-blocking file I/O. The usage data from the JSON file is loaded only when the first request is made (`_lazy_init`).
*   **Fine-Grained Locking**: Each API key is associated with its own `asyncio.Lock` and `asyncio.Condition` object. This allows for a highly granular and efficient locking strategy.

#### Tiered Key Acquisition (`acquire_key`)

This method implements the intelligent logic for selecting the best key for a job.

1.  **Filtering**: It first filters out any keys that are on a global or model-specific cooldown.
2.  **Tiering**: It categorizes the remaining, valid keys into two tiers:
    -   **Tier 1 (Ideal)**: Keys that are completely free (not being used by any model).
    -   **Tier 2 (Acceptable)**: Keys that are currently in use, but for *different models* than the one being requested. This allows a single key to be used for concurrent calls to, for example, `gemini-1.5-pro` and `gemini-1.5-flash`.
3.  **Selection**: It attempts to acquire a lock on a key, prioritizing Tier 1 over Tier 2. Within each tier, it prioritizes the key with the lowest usage count.
4.  **Waiting**: If no keys in Tier 1 or Tier 2 can be locked, it means all eligible keys are currently handling requests for the *same model*. The method then `await`s on the `asyncio.Condition` of the best available key, waiting efficiently until it is notified that a key has been released.

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
