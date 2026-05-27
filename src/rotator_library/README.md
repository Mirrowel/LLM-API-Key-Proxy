# Resilience & API Key Management Library

A robust, asynchronous, and thread-safe Python library for managing a pool of API keys. It is designed to be integrated into applications (such as the Universal LLM API Proxy included in this project) to provide a powerful layer of resilience and high availability when interacting with multiple LLM providers.

## Key Features

-   **Asynchronous by Design**: Built with `asyncio` and `httpx` for high-performance, non-blocking I/O.
-   **Anthropic API Compatibility**: Built-in translation layer (`anthropic_compat`) enables Anthropic API clients (like Claude Code) to use any supported provider.
-   **Advanced Concurrency Control**: A single API key can be used for multiple concurrent requests. `OPTIMAL_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>` is the soft spread-before-stacking target, while `MAX_CONCURRENT_REQUESTS_PER_KEY_<PROVIDER>` is the hard safety ceiling. Balanced mode defaults to optimal=1/max=unlimited; sequential defaults to sticky/unlimited. `0` or negative values mean unlimited.
-   **Smart Key Management**: Selects the optimal key for each request using a capacity-phase, model-aware locking strategy to maximize availability while preserving provider-side cache locality.
-   **Configurable Rotation Strategy**: Sequential selection is the default. Balanced selection can be enabled for providers where even distribution is preferred.
-   **Deadline-Driven Requests**: A global timeout ensures that no request, including all retries and key selections, exceeds a specified time limit.
-   **OAuth & API Key Support**: Built-in support for standard API keys and complex OAuth flows.
    -   **Gemini CLI**: Full OAuth 2.0 web flow with automatic project discovery, free-tier onboarding, and credential prioritization (paid vs free tier).
-   **Stateless Deployment Ready**: Can load complex OAuth credentials from environment variables, eliminating the need for physical credential files in containerized environments.
-   **Intelligent Error Handling**:
    -   **Escalating Per-Model Cooldowns**: Failed keys are placed on a temporary, escalating cooldown for specific models.
    -   **Key-Level Lockouts**: Keys failing across multiple models are temporarily removed from rotation.
    -   **Stream Recovery**: The client detects mid-stream errors (like quota limits) and gracefully handles them.
-   **Credential Prioritization**: Automatic tier detection and priority-based credential selection (e.g., paid tier credentials used first for models that require them).
-   **Advanced Model Requirements**: Support for model-tier restrictions (e.g., Gemini 3 requires paid-tier credentials).
-   **Robust Streaming Support**: Includes a wrapper for streaming responses that reassembles fragmented JSON chunks.
-   **Detailed Usage Tracking**: Tracks daily and global usage for each key, persisted per provider in `usage/usage_<provider>.json`.
-   **Classifier-Scoped Routing**: Isolate user-owned provider credentials by `classifier + provider` while preserving the existing global/default provider pools.
-   **Private Credential Identity**: Scoped private credentials are tracked by safe `private:<fingerprint>` identifiers so raw user API keys are not written to usage files.
-   **Automatic Daily Resets**: Automatically resets cooldowns and archives stats daily.
-   **Provider Agnostic**: Works with any provider supported by `litellm`.
-   **Extensible**: Easily add support for new providers through a simple plugin-based architecture.
-   **Temperature Override**: Global temperature=0 override to prevent tool hallucination with low-temperature settings.
-   **Shared OAuth Base**: Refactored OAuth implementation with reusable [`GoogleOAuthBase`](providers/google_oauth_base.py) for multiple providers.
-   **Fair Cycle Rotation**: Ensures each credential exhausts at least once before any can be reused within a tier. Prevents a single credential from being repeatedly used while others sit idle. Configurable per provider with tracking modes and cross-tier support.
-   **Custom Usage Caps**: Set custom limits per tier, per model/group that are more restrictive than actual API limits. Supports percentages (e.g., "80%") and multiple cooldown modes (`quota_reset`, `offset`, `fixed`). Credentials go on cooldown before hitting actual API limits.
-   **Centralized Defaults**: All tunable defaults are defined in [`config/defaults.py`](config/defaults.py) for easy customization and visibility.

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
    usage_file_path="usage.json",
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
-   `oauth_credentials` (`Optional[Dict[str, List[str]]]`): A dictionary mapping provider names (e.g., "gemini_cli") to a list of file paths to OAuth credential JSON files.
-   `max_retries` (`int`, default: `2`): The number of times to retry a request with the *same key* if a transient server error (e.g., 500, 503) occurs.
-   `usage_file_path` (`str`, optional): Base path for usage persistence (defaults to `usage/` in the data directory). The client stores per-provider files under `usage/usage_<provider>.json` next to this path.
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
-   **Scoped parameters**: Also accepts optional `classifier`, `api_keys`, `providers`, and `private` keyword arguments for classifier-scoped routing.
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

A wrapper around `litellm.aembedding` that provides the same key management and retry logic for embedding requests. It accepts the same scoped parameters as `acompletion`: `classifier`, `api_keys`, `providers`, and `private`.

#### `def token_count(self, model: str, text: str = None, messages: List[Dict[str, str]] = None) -> int:`

Calculates the token count for a given text or list of messages using `litellm.token_counter`.

#### `async def get_available_models(self, provider: str, ...) -> List[str]:`

Fetches a list of available models for a specific provider, applying any configured whitelists or blacklists. Results are cached in memory.

Scoped model discovery accepts `classifier`, `api_keys`, `providers`, `private`, `model_filters`, and `force_refresh`. Classified discovery uses only classifier/request credentials, honors request/registered provider `base_url` overrides, does not fall back to global keys, and caches by provider config, filters, and a safe fingerprint of the scoped credential set.

#### `async def get_all_available_models(self, grouped: bool = True, ...) -> Union[Dict[str, List[str]], List[str]]:`

Fetches a dictionary of all available models, grouped by provider, or as a single flat list if `grouped=False`. It accepts the same scoped discovery parameters as `get_available_models`.

#### `async def get_quota_stats(self, provider_filter: str = None, classifier: str = None) -> dict:`

Returns usage and quota stats. Without `classifier`, default stats preserve previous behavior and skip classifier-scoped managers. With `classifier`, only usage managers for that classifier are included.

### Classifier-Scoped Routing

Classifier-scoped routing lets a host application use one `RotatingClient` for both platform-owned keys and user-owned provider connections.

Default/global call:

```python
response = await client.acompletion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Use the platform pool"}],
)
```

Classified/stateless call:

```python
response = await client.acompletion(
    model="logfare/my-model",
    messages=[{"role": "user", "content": "Use my own key"}],
    classifier="user_123",
    api_keys={"logfare": ["user-logfare-key"]},
    providers={"logfare": {"base_url": "https://logfare.example/v1"}},
    private=True,
)
```

Rules:

- `classifier + provider` is the isolation boundary.
- Classified requests use only request-supplied or registered classifier credentials.
- Classified requests never inherit global/default API keys.
- Provider definitions/base URLs can be inherited or overridden without mutating global provider config.
- `private=True` stores safe `private:<fingerprint>` identifiers in usage state and hides `full_path` in stats.
- Scoped usage files are stored under `usage/classifiers/<safe_classifier>/usage_<provider>.json`.

### Registered Scope Management

Registered scopes keep classifier state in memory so callers do not need to pass provider config and keys on every request.

```python
await client.register_scope(
    "user_123",
    providers={"logfare": {"base_url": "https://logfare.example/v1"}},
    api_keys={"logfare": ["user-logfare-key"]},
    private=True,
)

response = await client.acompletion(
    model="logfare/my-model",
    messages=[{"role": "user", "content": "Use the registered scope"}],
    classifier="user_123",
)
```

Management methods:

```python
await client.register_scope(classifier, providers=None, api_keys=None, private=True)
await client.update_scope(classifier, providers=None, api_keys=None, private=None)
await client.get_scope(classifier, include_secrets=False)
await client.remove_scope(classifier)

await client.add_scope_provider(classifier, provider, config)
await client.update_scope_provider(classifier, provider, config)
await client.remove_scope_provider(classifier, provider)
await client.list_scope_providers(classifier)

await client.add_scope_credentials(classifier, provider, keys, private=True)
await client.set_scope_credentials(classifier, provider, keys, private=True)
await client.remove_scope_credentials(classifier, provider, credential_ids=None)
await client.list_scope_credentials(classifier, provider=None, include_secrets=False)
```

`get_scope()` and `list_scope_credentials()` do not return raw keys unless `include_secrets=True` is explicitly requested. Registered scopes are runtime state only; the host application remains responsible for durable encrypted secret storage, user permissions, and billing policy.

For full examples and edge cases, see [Classifier-Scoped Routing](../../docs/CLASSIFIER_SCOPED_ROUTING.md).

#### `async def anthropic_messages(self, request, raw_request=None, pre_request_callback=None) -> Any:`

Handle Anthropic Messages API requests. Accepts requests in Anthropic's format, translates them to OpenAI format internally, processes them through `acompletion`, and returns responses in Anthropic's format.

-   **Parameters**:
    -   `request`: An `AnthropicMessagesRequest` object (from `anthropic_compat.models`)
    -   `raw_request`: Optional raw request object for client disconnect checks
    -   `pre_request_callback`: Optional async callback before each API request
-   **Returns**:
    -   For non-streaming: dict in Anthropic Messages format
    -   For streaming: AsyncGenerator yielding Anthropic SSE format strings

#### `async def anthropic_count_tokens(self, request) -> dict:`

Handle Anthropic count_tokens API requests. Counts the number of tokens that would be used by a Messages API request.

-   **Parameters**: `request` - An `AnthropicCountTokensRequest` object
-   **Returns**: Dict with `input_tokens` count in Anthropic format

## Anthropic API Compatibility

The library includes a translation layer (`anthropic_compat`) that enables Anthropic API clients to use any OpenAI-compatible provider.

### Usage

```python
from rotator_library.anthropic_compat import (
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
    translate_anthropic_request,
    openai_to_anthropic_response,
    anthropic_streaming_wrapper,
)

# Create an Anthropic-format request
request = AnthropicMessagesRequest(
    model="gemini/gemini-2.5-flash",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

# Use with RotatingClient
async with RotatingClient(api_keys=api_keys) as client:
    response = await client.anthropic_messages(request)
    print(response["content"][0]["text"])
```

### Features

-   **Full Message Translation**: Converts between Anthropic and OpenAI message formats including text, images, tool_use, and tool_result blocks
-   **Extended Thinking Support**: Translates Anthropic's `thinking` configuration to `reasoning_effort` for providers that support it
-   **Streaming SSE Conversion**: Converts OpenAI streaming chunks to Anthropic's SSE event format (`message_start`, `content_block_delta`, etc.)
-   **Cache Token Handling**: Properly translates `prompt_tokens_details.cached_tokens` to Anthropic's `cache_read_input_tokens`
-   **Tool Call Support**: Full support for tool definitions and tool use/result blocks

## Credential Tool

The library includes a utility to manage credentials easily:

```bash
python -m src.rotator_library.credential_tool
```

Use this tool to:
1.  **Initialize OAuth**: Run the interactive login flow for Gemini CLI.
2.  **Export Credentials**: Generate `.env` compatible configuration blocks from your saved OAuth JSON files. This is essential for setting up stateless deployments.

## Provider Specifics

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

