# Classifier-Scoped Routing Implementation Plan

## Goal

Add multi-user/multi-tenant isolation to `rotator_library` without rewriting the retry, rotation, cooldown, streaming, provider plugin, or quota machinery.

The library should remain general-purpose. It should not own application users, permissions, billing rules, or encrypted database storage. Instead, every operation may receive an optional `classifier` that isolates provider configuration, credentials, usage, cooldowns, model discovery, model cache entries, and stats.

If no `classifier` is supplied, existing behavior must remain compatible.

## Plain Behavior

Default/global call:

```text
model="openai/gpt-4o"
no classifier
-> behave like the current library
-> use startup/env/global provider config and credentials
-> usage goes to the existing global/default provider usage files
```

Classified/stateless call:

```text
classifier="user_123"
providers={"logfare": {"base_url": "https://.../v1", "protocol": "openai_chat_completions"}}
api_keys={"logfare": [key1, key2], "openai": [other_key]}
model="logfare/my-model"
private=True
-> use only user_123/logfare keys
-> ignore unrelated provider keys for this request
-> provider config is request-scoped and does not overwrite global/default config
-> usage/cooldowns/cache/stats are isolated under user_123/logfare
-> raw key is not persisted/logged plainly for private credentials
```

Classified call inheriting global provider config:

```text
global/default has LOGFARE_API_BASE or an equivalent runtime global provider definition
classifier="user_123"
api_keys={"logfare": [user_key]}
model="logfare/my-model"
-> use global logfare provider definition/base_url
-> do not inherit global logfare keys
-> use only user_123 supplied/registered keys
```

## Core Rules

1. `classifier` separates worlds. `user_123/logfare` and `user_456/logfare` must not share keys, usage state, cooldowns, model cache entries, or request-local provider overrides.
2. Global/default provider definitions may be inherited by classified operations by policy. Default: allowed for provider config.
3. Global/default API keys must not be inherited by classified operations unless an explicit future option enables it. Default: never inherit keys.
4. Request overlays must not mutate global/default provider config, provider plugins, model caches, or usage managers for other classifiers.
5. Stateless per-call mode is first-class. The host app can pass current provider config and keys on every call.
6. Registered mode is optional. The host app can register/update/remove scoped providers and credentials for performance, model discovery, OAuth refresh, and central management.
7. Model filters are model-list visibility filters. They should not block completions by default.
8. Global model filter propagation is configured globally per provider. Default: global whitelist propagation false, global blacklist propagation false.
9. Current support is OpenAI-compatible chat/completions. Keep the public provider config shape future-ready with `protocol`, but initially support only `openai_chat_completions` and current LiteLLM/native provider behavior.

## Public API Direction

The same optional arguments should work across completion, embeddings, model discovery, and stats where relevant:

```python
classifier: Optional[str]
api_keys: Optional[dict[str, list[str]]]
providers: Optional[dict[str, dict[str, Any]]]
private: bool = False
model_filters: Optional[dict[str, dict[str, list[str]]]]
```

`providers` entries are future-facing:

```python
{
    "logfare": {
        "base_url": "https://logfare.example/v1",
        "protocol": "openai_chat_completions",
        "inherit_global_model_filters": None,
    }
}
```

Initially supported runtime provider behavior:

```text
protocol="openai_chat_completions"
unknown/custom provider -> route through LiteLLM openai-compatible mode with api_base/custom_llm_provider
known provider with base_url override -> pass api_base override
```

Future protocols can be added later:

```text
openai_responses
anthropic_messages
litellm_native
```

## Registered Scope API

Registered state is optional and host-managed. It should support:

```python
await client.register_scope(classifier, providers=None, api_keys=None, private=True)
await client.update_scope(classifier, providers=None, api_keys=None, private=None)
await client.get_scope(classifier)
await client.remove_scope(classifier)

await client.add_scope_provider(classifier, provider, config)
await client.update_scope_provider(classifier, provider, config)
await client.remove_scope_provider(classifier, provider)
await client.list_scope_providers(classifier)

await client.add_scope_credentials(classifier, provider, keys, private=True)
await client.set_scope_credentials(classifier, provider, keys, private=True)
await client.remove_scope_credentials(classifier, provider, credential_ids=None)
await client.list_scope_credentials(classifier, provider=None)
```

These methods manage library-side runtime state only. The host app remains source of truth for users, permissions, encrypted storage, and billing.

## Internal Runtime Model

Add a resolver that computes an immutable effective scope for each operation:

```text
global/default config
+ registered classifier config
+ request overlay
= effective isolated operation scope
```

Provider config resolution for classified operations:

```text
1. request providers[provider]
2. registered classifier provider config
3. global/default provider config if inheritance is allowed
4. built-in provider behavior
```

Credential resolution for classified operations:

```text
1. request api_keys[provider]
2. registered classifier credentials for provider
3. no global/default keys unless an explicit future option enables it
```

Runtime state key:

```text
scope_key = classifier + provider
```

Usage file layout:

```text
usage/
  default/
    usage_openai.json
    usage_logfare.json
  classifiers/
    <safe_classifier>/
      usage_openai.json
      usage_logfare.json
```

The legacy global usage path can be preserved for backward compatibility if practical, but new scoped usage should use the above layout.

## Credential Privacy And Identity

For private/scoped credentials, usage identity should be a safe fingerprint, not the raw key.

Recommended fingerprint:

```text
HMAC-SHA256(secret, host/library fingerprint key)
```

Storage should keep:

```text
credential_id/fingerprint
provider
classifier
display label if supplied/generated
usage stats
```

Storage should not keep raw API keys for private credentials.

The raw key may live in memory only long enough to call the provider. If registered state stores secrets in memory, that is runtime-only unless a host-provided resolver is used later.

Future secret resolver callbacks:

```python
async def resolve_secret(classifier, provider, credential_id): ...
async def update_secret(classifier, provider, credential_id, new_secret): ...
```

## Usage Manager Strategy

Do not rewrite usage tracking. Use one existing `UsageManager` per scope/provider:

```text
default/openai -> UsageManager(provider="openai")
user_123/logfare -> UsageManager(provider="logfare")
user_456/logfare -> UsageManager(provider="logfare")
```

Required `UsageManager` changes:

1. Support syncing credentials after initialization.
2. Preserve historical stats for credentials no longer active.
3. Use externally supplied credential ids/fingerprints for private/scoped credentials.
4. Avoid persisting raw accessors for private credentials.
5. Selection candidates for a request must come only from the effective scope's allowed credentials.

## Provider Config Strategy

Do not globally mutate provider config for request overlays.

Current global env custom provider behavior should remain:

```text
LOGFARE_API_BASE -> global/default logfare provider definition
```

Request/classifier overlays should be carried on the request context and applied during LiteLLM transformation:

```text
provider="logfare"
base_url="https://user-specific/v1"
model="logfare/foo"
-> LiteLLM kwargs: model="openai/foo", api_base=base_url, custom_llm_provider="openai"
```

For classified operations that do not supply a base_url, provider config may inherit the global provider definition by default.

## Model Discovery

Model discovery must use the same effective scope as completions.

Classified model listing:

```text
provider config from request/registered/global-template
credentials from request/registered classifier only
cache isolated by classifier + provider + provider-config fingerprint + filter-policy fingerprint
```

Add/extend methods:

```python
await client.get_available_models(
    provider,
    classifier=None,
    api_keys=None,
    providers=None,
    private=False,
    model_filters=None,
    force_refresh=False,
)

await client.get_all_available_models(
    classifier=None,
    api_keys=None,
    providers=None,
    private=False,
    model_filters=None,
    grouped=True,
    force_refresh=False,
)
```

## Model Filters

Current global filters are loaded by the proxy from env and passed to the library, but filtering is executed by the library's `ModelResolver` in model discovery.

New behavior:

1. Filters hide models from model listing only by default.
2. Global filter propagation is configured globally per provider.
3. Default propagation: false for whitelist, false for blacklist.
4. Classified model listing applies classifier/request filters if supplied.
5. Classified model listing applies global filters only when global propagation for that provider says to do so.

## Runtime Mutation Safety

Use one `RotatingClient` with many isolated scope runtimes, not one client per classifier/request.

Reasons:

```text
fewer HTTP clients
less memory
easier cleanup
central stats aggregation
no background task explosion
same retry/streaming machinery
```

Need locks around:

```text
registered scope config mutation
usage manager creation/sync
model cache invalidation
provider instance cache for scoped/custom providers if added
```

Per-request overlays should become immutable effective scope objects and should not mutate shared global state.

## Background Jobs And OAuth

Existing global/default background behavior remains for existing credentials.

Scoped behavior:

```text
per-call API key or access token -> no background job
registered API key -> no OAuth refresh needed
registered OAuth credential with durable path/resolver -> scoped refresh can run later
```

Do not make scoped OAuth refresh a first-pass blocker. The first pass should preserve global OAuth behavior and make scoped OAuth/access-token calls usable per request. Registered scoped OAuth refresh can be added after resolver/update callbacks are designed.

## Implementation Slices

### Slice 1: Planning And Scaffolding

- Add this plan file.
- Add walkthrough file.
- Add effective scope dataclasses/helpers.
- Add safe classifier path helper and credential fingerprint helper.

### Slice 2: Scoped Completions

- Add optional `classifier`, `api_keys`, `providers`, `private` handling to `acompletion` and `aembedding`.
- Resolve provider from model as today.
- Resolve effective provider config and credential list.
- Create/get scoped usage manager for `classifier/provider`.
- Pass only resolved credentials into `RequestExecutor`.
- Apply provider config overlay during request transformation without global mutation.

### Slice 3: Usage Privacy

- Let `UsageManager` accept credential identity metadata/fingerprints.
- Sync runtime keys after initialization.
- Avoid writing raw private accessors to usage JSON.
- Ensure stats/logs prefer safe credential ids for private credentials.

### Slice 4: Scoped Model Discovery

- Extend `get_available_models` and `get_all_available_models` with classifier overlays.
- Use scoped credentials and provider config.
- Cache by classifier/provider/config/filter/credential-set fingerprint.
- Keep old provider-only cache behavior for default calls.

### Slice 5: Registered Scope Management

- Add register/update/fetch/remove methods.
- Add provider and credential add/set/remove/list methods.
- Ensure registered state can be externally managed and does not own app user semantics.

### Slice 6: Filters And Stats

- Add global per-provider filter propagation policy.
- Apply classifier/request filters and optional propagated global filters in model listing.
- Add classifier/provider/credential-id filters to stats.

### Slice 7: Verification

- Add focused tests or smoke scripts if no test framework exists.
- Verify old no-classifier calls still work.
- Verify two classifiers using the same custom provider name but different base URLs do not share config/cache/usage/keys.
- Verify classified calls never fall back to global keys.
- Verify private usage files do not contain raw API keys.

## Non-Goals For First Pass

- Full user management.
- Billing policy.
- Proxy HTTP management endpoints.
- Non-OpenAI-compatible custom protocols.
- Scoped OAuth background refresh without resolver/update callback design.
- Replacing the existing retry/rotation/streaming implementation.
