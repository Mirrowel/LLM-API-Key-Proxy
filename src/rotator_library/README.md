# Resilience & API Key Management Library

A robust, asynchronous, and thread-safe Python library for managing a pool of API keys. It is designed to be integrated into applications (such as the Universal LLM API Proxy included in this project) to provide a powerful layer of resilience and high availability when interacting with multiple LLM providers.

## Key Features

-   **Asynchronous by Design**: Built with `asyncio` and `httpx` for high-performance, non-blocking I/O.
-   **Advanced Concurrency Control**: A single API key can be used for multiple concurrent requests. By default, it supports concurrent requests to *different* models. With configuration (`MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>`), it can also support multiple concurrent requests to the *same* model using the same key.
-   **Smart Key Management**: Selects the optimal key for each request using a tiered, model-aware locking strategy to distribute load evenly and maximize availability.
-   **Configurable Rotation Strategy**: Choose between deterministic least-used selection (perfect balance) or default weighted random selection (unpredictable, harder to fingerprint).
-   **Deadline-Driven Requests**: A global timeout ensures that no request, including all retries and key selections, exceeds a specified time limit.
-   **OAuth & API Key Support**: Built-in support for standard API keys and complex OAuth flows.
    -   **Gemini CLI**: Full OAuth 2.0 web flow with automatic project discovery, free-tier onboarding, and credential prioritization (paid vs free tier).
    -   **Antigravity**: Full OAuth 2.0 support for Gemini 3, Gemini 2.5, and Claude Sonnet 4.5 models with thought signature caching(Full support for Gemini 3 and Claude models). **First on the scene to provide full support for Gemini 3** via Antigravity with advanced features like thought signature caching and tool hallucination prevention.
    -   **Qwen Code**: Device Code flow support.
    -   **iFlow**: Authorization Code flow with local callback handling.
-   **Stateless Deployment Ready**: Can load complex OAuth credentials from environment variables, eliminating the need for physical credential files in containerized environments.
-   **Intelligent Error Handling**:
    -   **Escalating Per-Model Cooldowns**: Failed keys are placed on a temporary, escalating cooldown for specific models.
    -   **Key-Level Lockouts**: Keys failing across multiple models are temporarily removed from rotation.
    -   **Stream Recovery**: The client detects mid-stream errors (like quota limits) and gracefully handles them.
-   **Credential Prioritization**: Automatic tier detection and priority-based credential selection (e.g., paid tier credentials used first for models that require them).
-   **Advanced Model Requirements**: Support for model-tier restrictions (e.g., Gemini 3 requires paid-tier credentials).
-   **Robust Streaming Support**: Includes a wrapper for streaming responses that reassembles fragmented JSON chunks.
-   **Detailed Usage Tracking**: Tracks daily and global usage for each key, persisted to a JSON file.
-   **Automatic Daily Resets**: Automatically resets cooldowns and archives stats daily.
-   **Provider Agnostic**: Works with any provider supported by `litellm`.
-   **Extensible**: Easily add support for new providers through a simple plugin-based architecture.
-   **Temperature Override**: Global temperature=0 override to prevent tool hallucination with low-temperature settings.
-   **Shared OAuth Base**: Refactored OAuth implementation with reusable [`GoogleOAuthBase`](providers/google_oauth_base.py) for multiple providers.

## Installation

To install the library, you can install it directly from a local path. Using the `-e` flag installs it in "editable" mode, which is recommended for development.

```bash
pip install -e .
```

## `RotatingClient` Class

This is the main class for interacting with the library. It is designed to be a long-lived object that manages the state of your API key pool.

### Initialization

```python
import os
from dotenv import load_dotenv
from rotator_library import RotatingClient

# Load environment variables from .env file
load_dotenv()

# Dynamically load all provider API keys from environment variables
api_keys = {}
for key, value in os.environ.items():
    # This pattern finds keys like "GEMINI_API_KEY_1" or "OPENAI_API_KEY"
    if (key.endswith("_API_KEY") or "_API_KEY_" in key) and key != "PROXY_API_KEY":
        # Extracts "gemini" from "GEMINI_API_KEY_1"
        provider = key.split("_API_KEY")[0].lower()
        if provider not in api_keys:
            api_keys[provider] = []
        api_keys[provider].append(value)

# Initialize empty dictionary for OAuth credentials (or load from CredentialManager)
oauth_credentials = {}

client = RotatingClient(
    api_keys=api_keys,
    oauth_credentials=oauth_credentials,
    max_retries=2,
    usage_file_path="key_usage.json",
    configure_logging=True,
    global_timeout=30,
    abort_on_callback_error=True,
    litellm_provider_params={},
    ignore_models={},
    whitelist_models={},
    enable_request_logging=False,
    max_concurrent_requests_per_key={},
    rotation_tolerance=2.0  # 0.0=deterministic, 2.0=recommended random
)
```

#### Arguments

-   `api_keys` (`Optional[Dict[str, List[str]]]`): A dictionary mapping provider names (e.g., "openai", "anthropic") to a list of API keys.
-   `oauth_credentials` (`Optional[Dict[str, List[str]]]`): A dictionary mapping provider names (e.g., "gemini_cli", "qwen_code") to a list of file paths to OAuth credential JSON files.
-   `max_retries` (`int`, default: `2`): The number of times to retry a request with the *same key* if a transient server error (e.g., 500, 503) occurs.
-   `usage_file_path` (`str`, default: `"key_usage.json"`): The path to the JSON file where usage statistics (tokens, cost, success counts) are persisted.
-   `configure_logging` (`bool`, default: `True`): If `True`, configures the library's logger to propagate logs to the root logger. Set to `False` if you want to handle logging configuration manually.
-   `global_timeout` (`int`, default: `30`): A hard time limit (in seconds) for the entire request lifecycle. If the request (including all retries) takes longer than this, it is aborted.
-   `abort_on_callback_error` (`bool`, default: `True`): If `True`, any exception raised by `pre_request_callback` will abort the request. If `False`, the error is logged and the request proceeds.
-   `litellm_provider_params` (`Optional[Dict[str, Any]]`, default: `None`): A dictionary of extra parameters to pass to `litellm` for specific providers.
-   `ignore_models` (`Optional[Dict[str, List[str]]]`, default: `None`): A dictionary where keys are provider names and values are lists of model names/patterns to exclude (blacklist). Supports wildcards (e.g., `"*-preview"`).
-   `whitelist_models` (`Optional[Dict[str, List[str]]]`, default: `None`): A dictionary where keys are provider names and values are lists of model names/patterns to always include, overriding `ignore_models`.
-   `enable_request_logging` (`bool`, default: `False`): If `True`, enables detailed per-request file logging (useful for debugging complex interactions).
-   `max_concurrent_requests_per_key` (`Optional[Dict[str, int]]`, default: `None`): A dictionary defining the maximum number of concurrent requests allowed for a single API key for a specific provider. Defaults to 1 if not specified.
-   `rotation_tolerance` (`float`, default: `0.0`): Controls credential rotation strategy:
    - `0.0`: **Deterministic** - Always selects the least-used credential for perfect load balance.
    - `2.0` (default, recommended): **Weighted Random** - Randomly selects credentials with bias toward less-used ones. Provides unpredictability (harder to fingerprint) while maintaining good balance.
    - `5.0+`: **High Randomness** - Even heavily-used credentials have significant selection probability. Maximum unpredictability.
    
    The weight formula is: `weight = (max_usage - credential_usage) + tolerance + 1`
    
    **Use Cases:**
    - `0.0`: When perfect load balance is critical
    - `2.0`: When avoiding fingerprinting/rate limit detection is important
    - `5.0+`: For stress testing or maximum unpredictability

### Concurrency and Resource Management

The `RotatingClient` is asynchronous and manages an `httpx.AsyncClient` internally. It's crucial to close the client properly to release resources. The recommended way is to use an `async with` block.

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

This is the primary method for making API calls. It's a wrapper around `litellm.acompletion` that adds the core logic for key acquisition, selection, and retries.

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

A wrapper around `litellm.aembedding` that provides the same key management and retry logic for embedding requests.

#### `def token_count(self, model: str, text: str = None, messages: List[Dict[str, str]] = None) -> int:`

Calculates the token count for a given text or list of messages using `litellm.token_counter`.

#### `async def get_available_models(self, provider: str) -> List[str]:`

Fetches a list of available models for a specific provider, applying any configured whitelists or blacklists. Results are cached in memory.

#### `async def get_all_available_models(self, grouped: bool = True) -> Union[Dict[str, List[str]], List[str]]:`

Fetches a dictionary of all available models, grouped by provider, or as a single flat list if `grouped=False`.

## Credential Tool

The library includes a utility to manage credentials easily:

```bash
python -m src.rotator_library.credential_tool
```

Use this tool to:
1.  **Initialize OAuth**: Run the interactive login flows for Gemini, Qwen, and iFlow.
2.  **Export Credentials**: Generate `.env` compatible configuration blocks from your saved OAuth JSON files. This is essential for setting up stateless deployments.

## Provider Specifics

### Qwen Code
-   **Auth**: Uses OAuth 2.0 Device Flow. Requires manual entry of email/identifier if not returned by the provider.
-   **Resilience**: Injects a dummy tool (`do_not_call_me`) into requests with no tools to prevent known stream corruption issues on the API.
-   **Reasoning**: Parses `<think>` tags in the response and exposes them as `reasoning_content`.
-   **Schema Cleaning**: Recursively removes `strict` and `additionalProperties` from all tool schemas. Qwen's API has stricter validation than OpenAI's, and these properties cause `400 Bad Request` errors.

### iFlow
-   **Auth**: Uses Authorization Code Flow with a local callback server (port 11451).
-   **Key Separation**: Distinguishes between the OAuth `access_token` (used to fetch user info) and the `api_key` (used for actual chat requests).
-   **Resilience**: Similar to Qwen, injects a placeholder tool to stabilize streaming for empty tool lists.
-   **Schema Cleaning**: Recursively removes `strict` and `additionalProperties` from all tool schemas to prevent API validation errors.
-   **Custom Models**: Supports model definitions via `IFLOW_MODELS` environment variable (JSON array of model IDs or objects).

### NVIDIA NIM
-   **Discovery**: Dynamically fetches available models from the NVIDIA API.
-   **Thinking**: Automatically injects the `thinking` parameter into `extra_body` for DeepSeek models (`deepseek-v3.1`, etc.) when `reasoning_effort` is set to low/medium/high.

### Google Gemini (CLI)
-   **Auth**: Simulates the Google Cloud CLI authentication flow.
-   **Project Discovery**: Automatically discovers the default Google Cloud Project ID with enhanced onboarding flow.
-   **Credential Prioritization**: Automatic detection and prioritization of paid vs free tier credentials.
-   **Model Tier Requirements**: Gemini 3 models automatically filtered to paid-tier credentials only.
-   **Gemini 3 Support**: Full support for Gemini 3 models with:
    - `thinkingLevel` configuration (low/high)
    - Tool hallucination prevention via system instruction injection
    - ThoughtSignature caching for multi-turn conversations
    - Parameter signature injection into tool descriptions
-   **Rate Limits**: Implements smart fallback strategies (e.g., switching from `gemini-1.5-pro` to `gemini-1.5-pro-002`) when rate limits are hit.

### Antigravity
-   **Auth**: Uses OAuth 2.0 flow similar to Gemini CLI, with Antigravity-specific credentials and scopes.
-   **Credential Prioritization**: Automatic detection and prioritization of paid vs free tier credentials (paid tier resets every 5 hours, free tier resets weekly).
-   **Models**: Supports Gemini 3 Pro, Gemini 2.5 Flash/Flash Lite, Claude Sonnet 4.5 (with/without thinking), Claude Opus 4.5 (thinking only), and GPT-OSS 120B via Google's internal Antigravity API.
-   **Quota Groups**: Models that share quota are automatically grouped:
    - Claude/GPT-OSS: `claude-sonnet-4-5`, `claude-opus-4-5`, `gpt-oss-120b-medium`
    - Gemini 3 Pro: `gemini-3-pro-high`, `gemini-3-pro-low`, `gemini-3-pro-preview`
    - Gemini 2.5 Flash: `gemini-2.5-flash`, `gemini-2.5-flash-thinking`, `gemini-2.5-flash-lite`
    - All models in a group deplete the usage of the group equally. So in claude group - it is beneficial to use only Opus, and forget about Sonnet and GPT-OSS.
-   **Quota Baseline Tracking**: Background job fetches quota status from API every 5 minutes to provide accurate remaining quota estimates.
-   **Thought Signature Caching**: Server-side caching of `thoughtSignature` data for multi-turn conversations with Gemini 3 models.
-   **Tool Hallucination Prevention**: Automatic injection of system instructions and parameter signatures for Gemini 3 and Claude to prevent tool parameter hallucination.
-   **Parallel Tool Usage Instruction**: Configurable instruction injection to encourage parallel tool calls (enabled by default for Claude).
-   **Thinking Support**:
    - Gemini 3: Uses `thinkingLevel` (string: "low"/"high")
    - Gemini 2.5 Flash: Uses `-thinking` variant when `reasoning_effort` is provided
    - Claude Sonnet 4.5: Uses `thinkingBudget` (optional - supports both thinking and non-thinking modes)
    - Claude Opus 4.5: Uses `thinkingBudget` (always uses thinking variant)
-   **Base URL Fallback**: Automatic fallback between sandbox and production endpoints.

## Error Handling and Cooldowns

The client uses a sophisticated error handling mechanism:

-   **Error Classification**: All exceptions from `litellm` are passed through a `classify_error` function to determine their type (`rate_limit`, `authentication`, `server_error`, `quota`, `context_length`, etc.).
-   **Server Errors**: The client will retry the request with the *same key* up to `max_retries` times, using an exponential backoff strategy.
-   **Key-Specific Errors (Authentication, Quota, etc.)**: The client records the failure in the `UsageManager`, which applies an escalating cooldown to the key for that specific model. The client then immediately acquires a new key and continues its attempt to complete the request.
-   **Escalating Cooldown Strategy**: Consecutive failures for a key on the same model result in increasing cooldown períods:
    - 1st failure: 10 seconds
    - 2nd failure: 30 seconds
    - 3rd failure: 60 seconds
    - 4th+ failure: 120 seconds
-   **Key-Level Lockouts**: If a key fails on multiple different models (3+ distinct models), the `UsageManager` applies a global 5-minute lockout for that key, removing it from rotation entirely.
-   **Authentication Errors**: Immediate 5-minute global lockout (key is assumed revoked or invalid).

### Global Timeout and Deadline-Driven Logic

To ensure predictable performance, the client now operates on a strict time budget defined by the `global_timeout` parameter.

-   **Deadline Enforcement**: When a request starts, a `deadline` is set. The entire process, including all key rotations and retries, must complete before this deadline.
-   **Deadline-Aware Retries**: If a retry requires a wait time that would exceed the remaining budget, the wait is skipped, and the client immediately rotates to the next key.
-   **Silent Internal Errors**: Intermittent failures like provider capacity limits or temporary server errors are logged internally but are **not raised** to the caller. The client will simply rotate to the next key.

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
    async def get_models(self, credential: str, client: httpx.AsyncClient) -> List[str]:
        # Logic to fetch and return a list of model names
        # The credential argument allows using the key to fetch models
        pass
```

The system will automatically discover and register your new provider.

## Detailed Documentation

For a more in-depth technical explanation of the library's architecture, including the `UsageManager`'s concurrency model and the error classification system, please refer to the [Technical Documentation](../../DOCUMENTATION.md).

