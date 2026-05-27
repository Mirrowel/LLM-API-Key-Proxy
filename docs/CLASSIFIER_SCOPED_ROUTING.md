# Classifier-Scoped Routing

Classifier-scoped routing lets applications use one `RotatingClient` for both platform-owned provider pools and user-owned provider credentials.

The feature is implemented in `rotator_library`. The proxy can build on it later, but the current public API is the Python library API.

## What It Solves

Default usage pools every configured key by provider:

```text
openai -> [platform key 1, platform key 2]
logfare -> [platform key]
```

That is correct for platform/global models. It is not correct when a request must use only a specific user's provider connection or API key.

Classifier-scoped routing adds an isolation label:

```text
classifier + provider -> isolated credential pool, usage file, model cache, and provider override
```

Examples:

```text
user_123 + logfare -> only user_123's Logfare keys and Logfare base URL
user_456 + logfare -> only user_456's Logfare keys and Logfare base URL
default + openai -> existing global OpenAI pool
```

## Core Rules

- If no `classifier` and no request-scoped credentials/config are supplied, behavior stays backward compatible.
- A classified request never falls back to global/default API keys.
- A classified request can inherit global provider definitions such as env `LOGFARE_API_BASE`, but not global keys.
- Request provider overrides do not mutate global provider config or registered classifier config.
- The same provider name can be reused by many classifiers with different keys and base URLs.
- Usage, model discovery cache, quota stats, and active credential state are isolated by classifier/provider.
- Model filters hide model-listing results only. They do not block completions by default.

## Stateless Per-Call Usage

Stateless mode is the simplest integration when the host app already stores user credentials and can pass current truth on each request.

```python
from rotator_library import RotatingClient

client = RotatingClient(
    api_keys={"openai": ["platform-openai-key"]},
)

response = await client.acompletion(
    model="logfare/my-model",
    messages=[{"role": "user", "content": "Hello"}],
    classifier="user_123",
    api_keys={
        "logfare": ["user-logfare-key-1", "user-logfare-key-2"],
        "openai": ["ignored-for-this-request"],
    },
    providers={
        "logfare": {
            "base_url": "https://user-logfare.example/v1",
            "protocol": "openai_chat_completions",
        }
    },
    private=True,
)
```

For `model="logfare/my-model"`, only `api_keys["logfare"]` is considered. Keys for unrelated providers are ignored for that operation.

With the OpenAI-compatible provider override above, the LiteLLM call is converted internally to:

```text
model="openai/my-model"
api_base="https://user-logfare.example/v1"
custom_llm_provider="openai"
api_key="user-logfare-key-..."
```

The raw key is only injected at the final provider-call boundary.

## Registered Scope Usage

Registered mode keeps classifier state in the client process. It is useful for routers, model-picking UIs, and repeated requests where passing keys/config each time is inconvenient.

The host app still owns durable storage, permissions, users, billing, and encryption. Registered scope state is runtime library state.

```python
await client.register_scope(
    "user_123",
    providers={
        "logfare": {"base_url": "https://user-logfare.example/v1"},
    },
    api_keys={
        "logfare": ["user-logfare-key-1", "user-logfare-key-2"],
    },
    private=True,
)

response = await client.acompletion(
    model="logfare/my-model",
    messages=[{"role": "user", "content": "Use my provider key"}],
    classifier="user_123",
)
```

### Scope Management API

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

`get_scope()` and `list_scope_credentials()` hide raw secrets by default. Use `include_secrets=True` only inside trusted host-management code.

## Private Credentials

Pass `private=True` for user-owned keys.

Private mode changes the internal identity used for selection and persistence:

```text
raw secret -> HMAC-SHA256 fingerprint -> private:<fingerprint>
```

The HMAC key comes from `ROTATOR_LIBRARY_FINGERPRINT_KEY` when set, otherwise from the resolved data directory. This makes the same secret stable within the deployment without storing the raw secret in usage files.

Private mode guarantees:

- Usage files store the `private:<fingerprint>` accessor, not the raw API key.
- Private credentials are not written to `accessor_index`.
- Stats set `full_path` to `None` for private credentials.
- Provider calls still receive the raw key, but only at the final call boundary through in-memory request state.

Private mode does not provide encrypted durable secret storage. If registered scopes need to survive process restart, the host app should store secrets securely and re-register them on startup or pass them statelessly per request.

## Usage Files And Stats

Default/global usage remains under the existing provider files:

```text
usage/usage_openai.json
usage/usage_gemini.json
```

Classified usage is stored separately:

```text
usage/classifiers/<safe_classifier>/usage_<provider>.json
```

Classifier names are sanitized for paths. Unsafe labels are truncated and suffixed with a hash.

Stats can be queried by classifier:

```python
default_stats = await client.get_quota_stats()
user_stats = await client.get_quota_stats(classifier="user_123")
logfare_stats = await client.get_quota_stats(
    classifier="user_123",
    provider_filter="logfare",
)
```

Default stats intentionally skip classifier-scoped usage managers to preserve old behavior.

## Model Discovery

Model discovery uses the same scoped resolver as completions.

```python
models = await client.get_available_models(
    "logfare",
    classifier="user_123",
    api_keys={"logfare": ["user-logfare-key"]},
    providers={"logfare": {"base_url": "https://user-logfare.example/v1"}},
    private=True,
    model_filters={"logfare": {"blacklist": ["*/experimental*"]}},
)
```

For classified calls:

- Only request/registered classifier credentials are used.
- Global/default keys are not used.
- Request/registered provider `base_url` is honored.
- Cache keys include classifier, provider, provider config, model filters, and a safe fingerprint of the scoped credential set.
- Global model filters do not propagate by default.

All models for a classifier can be queried with:

```python
grouped = await client.get_all_available_models(classifier="user_123")
flat = await client.get_all_available_models(classifier="user_123", grouped=False)
```

When the provider config supplies `base_url` or `api_base`, model discovery calls the OpenAI-compatible `/models` endpoint directly. This avoids mutating global provider plugin registration for per-classifier provider aliases.

## Model Filters

Default no-classifier model listing continues to use the global `ignore_models` and `whitelist_models` passed to `RotatingClient`.

Classified model listing does not inherit those global filters by default. Supply classifier/request filters explicitly:

```python
models = await client.get_available_models(
    "logfare",
    classifier="user_123",
    model_filters={
        "logfare": {
            "whitelist": ["logfare/prod-*"],
            "blacklist": ["*/deprecated*"],
        }
    },
)
```

Accepted filter keys are:

```text
whitelist, whitelist_models, allow
blacklist, ignore, ignore_models, deny
```

Whitelist takes precedence over blacklist.

## Runtime Provider Overrides

Provider overrides are request-local or classifier-local. They do not change `PROVIDER_PLUGINS` or the global `ProviderConfig`.

Supported first-pass shape:

```python
providers={
    "logfare": {
        "base_url": "https://logfare.example/v1",
        "protocol": "openai_chat_completions",
    }
}
```

`protocol` is future-facing. The current implemented runtime behavior is OpenAI-compatible chat/completions routing through LiteLLM's OpenAI provider mode.

## Streaming And Embeddings

`classifier`, `api_keys`, `providers`, and `private` work for both non-streaming and streaming `acompletion()` calls.

```python
stream = await client.acompletion(
    model="logfare/my-model",
    messages=[{"role": "user", "content": "Stream"}],
    stream=True,
    classifier="user_123",
    api_keys={"logfare": ["user-key"]},
    providers={"logfare": {"base_url": "https://logfare.example/v1"}},
    private=True,
)

async for event in stream:
    print(event)
```

Embeddings accept the same scoped arguments:

```python
embedding = await client.aembedding(
    model="openai/text-embedding-3-small",
    input="hello",
    classifier="user_123",
    api_keys={"openai": ["user-openai-key"]},
    private=True,
)
```

## Limitations

- Registered scopes are in-memory runtime state. Persist them in the host application if needed.
- Scoped OAuth background refresh is not implemented yet. Existing global/default OAuth refresh behavior remains unchanged.
- Runtime provider overrides currently target OpenAI-compatible chat/completions behavior.
- The proxy HTTP API does not yet expose classifier management endpoints.
- The library does not implement user authorization, billing, or encrypted secret storage.

## Tested Behavior

The feature has a dedicated test module at `tests/test_classifier_scoped_routing.py` covering:

- Backward-compatible default/global completion.
- Stateless private scoped completion.
- Streaming scoped completion.
- No fallback from classifier scopes to global keys.
- Request overlay precedence over registered scope state.
- Registered scope add/set/remove/fetch behavior.
- Scoped model discovery, cache isolation, and filtering.
- Non-propagation of global filters to classifiers by default.
- Private stats and usage-file redaction behavior.
- UsageManager active credential resync while preserving history.
- Provider override routing without global mutation.

Targeted verification:

```bash
python -m pytest tests/test_classifier_scoped_routing.py -q
python -m compileall src tests/test_classifier_scoped_routing.py
```
