# Rotator Library Multi-User Requirements

This document captures the `rotator_library` changes required for ImaginAI's rewritten model/provider configuration system.

The old ImaginAI plans are historical. This document is the current requirement for making the library usable by a multi-user storytelling platform while preserving the library's existing strengths: provider abstraction, credential rotation, retries, cooldowns, usage tracking, quota handling, streaming, and model discovery.

## Product Requirement

ImaginAI must be able to configure two classes of selectable models:

1. **Global platform models** configured by an admin/host. Any permitted user can select these models. Users do not see the underlying provider key. When these models are used, `rotator_library` should use its normal pooled operation: credential rotation, load balancing, retries, cooldowns, usage tracking, and quota-aware selection.
2. **User-owned models** configured by an individual user. These models must use only that user's provider connection and credential. They must never route through another user's key or through a platform key unless explicitly configured as a global model.

ImaginAI must not talk directly to OpenAI, Gemini, Anthropic, OpenRouter, custom OpenAI-compatible servers, or any other LLM provider. All LLM communication must go through `rotator_library`.

## Current Library Constraints

The current library assumes a process-wide credential pool initialized at startup.

Important current behavior:

- `lib/proxy_app/main.py` discovers `*_API_KEY` environment variables once at import/startup and passes them into `RotatingClient`.
- `lib/rotator_library/client/rotating_client.py` stores credentials in `self.all_credentials: Dict[str, List[str]]`, keyed only by provider.
- `RotatingClient.__init__()` creates one `UsageManager` per provider and usage files like `usage_{provider}.json`.
- `RotatingClient.acompletion()` extracts the provider from `model.split("/")[0]` and builds a `RequestContext` using every credential in `self.all_credentials[provider]`.
- `RequestContext` in `lib/rotator_library/core/types.py` has no `user_id`, `credential_scope`, `provider_connection_id`, `model_config_id`, or per-request routing metadata.
- `RequestExecutor` chooses credentials from the provider-wide candidate list and injects the selected credential into either `kwargs["api_key"]` or `kwargs["credential_identifier"]`.
- `ProviderConfig` reads `*_API_BASE` from `os.environ` once and treats custom providers as environment-defined OpenAI-compatible providers.
- `PROVIDER_PLUGINS` is built once at import time. Dynamic custom providers are discovered from environment variables only.
- `get_available_models(provider)` tries credentials from the provider-wide pool and caches by provider only.

These assumptions are good for a single proxy process but are not sufficient for multi-user application routing.

## Conceptual Model Needed

The library needs to distinguish these concepts explicitly.

### Provider Connection

A provider connection describes how to reach a provider API.

Required fields:

- `id`: stable application-provided id
- `name`: display/debug name
- `provider`: provider key used by the library, e.g. `openai`, `gemini`, `openrouter`, `custom_local_llm`
- `protocol`: request protocol, e.g. `openai_chat_completions`, `openai_responses`, `anthropic_messages`, `gemini`, `litellm_native`
- `base_url`: optional, required for many custom providers
- `provider_options`: provider-specific metadata/options
- `owner_scope`: `platform` or `user`
- `owner_id`: user id when `owner_scope=user`, otherwise null
- `enabled`: bool

Provider connections must be runtime-managed. They cannot require environment variables or process restart.

### Credential

A credential describes secret material usable by a provider connection.

Required fields:

- `id`: stable application-provided id, not derived from the raw secret
- `provider_connection_id`
- `owner_scope`: `platform` or `user`
- `owner_id`: user id when user-owned
- `auth_type`: `api_key`, `oauth_file`, `oauth_env`, or future auth types
- `secret`: decrypted secret value, OAuth path, or credential reference usable by the provider implementation
- `display_name`: safe label
- `tier`: optional tier metadata
- `priority`: optional rotation priority
- `enabled`: bool

The library must never use a raw secret as the primary identity of a credential. It may receive raw secret material to call providers, but usage state, logs, caches, and selection identity should use the stable `credential.id` supplied by the host app.

### Model Config

ImaginAI model configs are application objects, but the library must accept enough routing information to execute them.

ImaginAI model configs contain:

- display name
- provider connection reference
- model id/name
- context length
- max output tokens
- temperature and other generation defaults
- optional model-specific parameters
- optional additional system prompt, owned by ImaginAI's story layer, not by provider routing
- owner scope: global/platform or user
- sort order and enabled/default flags

The library should not own story prompts. It should receive already-built messages/system fields and execute the request.

## Required Library Capabilities

### R1: Runtime Provider Connection Registry

Add first-class runtime provider registration.

Required operations:

```python
await client.register_provider_connection(connection: ProviderConnection) -> None
await client.update_provider_connection(connection: ProviderConnection) -> None
await client.remove_provider_connection(connection_id: str) -> None
await client.list_provider_connections(scope: Optional[CredentialScope] = None) -> list[ProviderConnection]
```

Requirements:

- Must not rely on `os.environ` for custom providers.
- Must support multiple custom providers with different `base_url` values simultaneously.
- Must support same protocol with different names and credentials.
- Must invalidate model-list caches when connection details change.
- Must be async-safe. Runtime mutation needs a lock around provider/credential/usage structures.

### R2: Runtime Credential Registry

Add runtime credential registration/removal without restarting the process.

Required operations:

```python
await client.add_credential(credential: CredentialDescriptor) -> None
await client.update_credential(credential: CredentialDescriptor) -> None
await client.remove_credential(credential_id: str) -> None
await client.enable_credential(credential_id: str, enabled: bool) -> None
await client.list_credentials(scope: Optional[CredentialScope] = None) -> list[CredentialDescriptor]
```

Requirements:

- Adding a credential must update selection state and usage tracking immediately.
- Removing a credential must stop future selection without corrupting historical usage stats.
- Disabling a credential must preserve stats but exclude it from request selection.
- The existing `.env` discovery path should remain supported and should import env credentials as platform-owned credentials in a reserved namespace such as `platform/env`.
- Admins/hosts must be able to create global model configs that use env-backed platform credentials without entering a new key in the ImaginAI UI.
- Existing OAuth discovery should remain supported but needs an explicit owner/scope metadata path.

### R3: Request Scope And Credential Isolation

Every completion, embedding, model-list, token-count, and future provider operation must accept a request scope.

Proposed request fields:

```python
@dataclass
class CredentialScope:
    owner_scope: Literal["platform", "user"]
    owner_id: Optional[str] = None
    provider_connection_id: Optional[str] = None

@dataclass
class RequestScope:
    owner_scope: Literal["platform", "user"]
    owner_id: Optional[str] = None
    provider_connection_id: Optional[str] = None
    model_config_id: Optional[str] = None
    credential_ids: Optional[list[str]] = None
    routing_mode: Literal["pool", "explicit_credentials"] = "pool"
```

Required behavior:

- Global/admin model configs use `owner_scope="platform"`, `routing_mode="pool"`, and the selected provider connection. The library then performs normal key rotation/load balancing across eligible platform credentials for that provider connection.
- User model configs use `owner_scope="user"`, `owner_id=<user id>`, and the selected provider connection. The library must only use credentials owned by that user for that provider connection.
- If `credential_ids` are provided, selection must be restricted to those credentials after checking ownership/scope.
- A user request must never fall back to platform credentials unless the ImaginAI model config explicitly points at a global platform model.
- Platform/global requests must never include user credentials.

This likely requires adding scope fields to `RequestContext` and moving credential candidate resolution out of `model.split("/")[0]` alone.

### R4: Provider And Model Routing Separate From Model String

The current model string convention `provider/model` is not enough.

The library should support calls like:

```python
await client.acompletion(
    scope=RequestScope(
        owner_scope="user",
        owner_id="user_123",
        provider_connection_id="conn_openrouter_user_123",
        model_config_id="model_claude_sonnet_user_123",
    ),
    model="anthropic/claude-sonnet-4",
    messages=[...],
)
```

Requirements:

- Provider routing should prefer `scope.provider_connection_id` when present.
- The model string should identify the upstream model, not be the only source of provider identity.
- The library must still support old `provider/model` calls for compatibility.
- Custom provider mapping must be based on the provider connection's `protocol` and `base_url`, not on environment variable names.
- Provider transforms should receive provider connection metadata so custom connections can apply protocol-specific behavior.

### R5: Usage Tracking Namespaced By Owner And Connection

Usage state must not be keyed only by provider and credential hash.

Required tracking dimensions:

- `owner_scope`
- `owner_id`
- `provider_connection_id`
- `credential_id`
- provider
- upstream model id
- optional `model_config_id`

Requirements:

- Platform usage and user usage must be separable in stats and persistence.
- Usage stats must survive secret rotation because `credential_id` is stable.
- Historical stats should remain after credentials are disabled or removed.
- Quota/cooldown state must be isolated so one user's exhausted key never blocks another user's key.
- Platform credential cooldowns should still be shared across all users using that platform model, because they are consuming the same platform pool.
- The library should expose filtered quota stats by owner, user, provider connection, credential, and model config.

Possible persistence approaches:

- One usage store with namespaced records.
- Separate files by namespace, e.g. `usage/platform/{connection}.json` and `usage/users/{user_id}/{connection}.json`.
- Pluggable persistence backend supplied by host app later.

### R6: Model Discovery Per Provider Connection

ImaginAI needs model-picking UI that can fetch available models from a configured provider connection.

Required operations:

```python
await client.get_available_models_for_connection(
    provider_connection_id: str,
    scope: RequestScope,
    credential_id: Optional[str] = None,
    force_refresh: bool = False,
) -> list[ModelInfo]
```

Requirements:

- Must use a credential eligible for the supplied scope and provider connection.
- User model discovery must use only that user's credential.
- Platform model discovery must use only platform credentials.
- Cache keys must include provider connection id and scope. Caching only by provider is unsafe.
- Must support OpenAI-compatible `/v1/models` fetch where possible.
- Must allow manual model id entry when provider discovery is unavailable or fails.

### R7: Protocol Support For Custom Providers

Custom providers need more than `unknown provider means OpenAI-compatible`.

At minimum, support protocol types:

- `openai_chat_completions`
- `openai_responses`
- `anthropic_messages`
- `gemini`
- `litellm_native`

Requirements:

- A provider connection's protocol determines how requests are translated and which library path is used.
- Base URL override must be per provider connection, not global per provider name.
- Provider-specific transforms must be selected by protocol/provider connection, not just by provider string.
- Existing provider plugins should continue working.
- Dynamic provider registration should not require adding a Python file for every custom OpenAI-compatible endpoint.

### R8: Explicit Credential Selection Internals

The current `UsageManager.acquire_credential(..., candidates=...)` can restrict candidates, but candidates are raw credential accessors. This needs to change.

Requirements:

- Selection candidates should be stable credential ids, not raw secrets.
- `CredentialContext` should expose `credential_id`, safe display metadata, and resolved secret/accessor separately.
- Logging should mask secrets and prefer safe credential ids/display names.
- Retry state should track credential ids rather than raw accessors.
- Provider implementations should receive the resolved secret/accessor only at the final call boundary.

### R9: Secret Handling And Logging

The host app will store secrets encrypted in its database and pass decrypted material to the library when registering/updating credentials or via a resolver callback.

Requirements:

- The library must not persist raw API keys in usage files.
- The library must not log raw API keys, raw OAuth tokens, authorization headers, or decrypted secret payloads.
- Error messages returned to callers must not include raw secrets.
- Transaction/raw request logging must keep redacting `api_key`, `credential_identifier`, `Authorization`, `api_base` if needed, and any provider-specific secret fields.
- Prefer supporting a host-provided secret resolver callback so the library can keep only credential ids in memory for long-lived apps, if practical.

### R10: Backward Compatibility Path

Existing proxy/self-host usage should continue to work.

Requirements:

- Existing `RotatingClient(api_keys={...}, oauth_credentials={...})` initialization should still work.
- Existing `client.acompletion(model="openai/gpt-4", messages=[...])` should still work.
- Existing env-based custom provider behavior should still work, but internally it should be represented as runtime provider connections and platform credentials.
- Existing proxy endpoints should continue functioning unless intentionally migrated.

## Proposed Public API Shape

This is a suggested shape, not final implementation code.

```python
from dataclasses import dataclass
from typing import Literal, Optional, Any

@dataclass
class ProviderConnection:
    id: str
    name: str
    provider: str
    protocol: Literal[
        "openai_chat_completions",
        "openai_responses",
        "anthropic_messages",
        "gemini",
        "litellm_native",
    ]
    base_url: Optional[str] = None
    owner_scope: Literal["platform", "user"] = "platform"
    owner_id: Optional[str] = None
    provider_options: dict[str, Any] | None = None
    enabled: bool = True

@dataclass
class CredentialDescriptor:
    id: str
    provider_connection_id: str
    owner_scope: Literal["platform", "user"]
    owner_id: Optional[str]
    auth_type: Literal["api_key", "oauth_file", "oauth_env"]
    secret: str
    display_name: Optional[str] = None
    tier: Optional[str] = None
    priority: int = 999
    enabled: bool = True

@dataclass
class RequestScope:
    owner_scope: Literal["platform", "user"]
    owner_id: Optional[str] = None
    provider_connection_id: Optional[str] = None
    model_config_id: Optional[str] = None
    credential_ids: Optional[list[str]] = None
    routing_mode: Literal["pool", "explicit_credentials"] = "pool"
```

Example usage for a global model:

```python
response = await client.acompletion(
    scope=RequestScope(
        owner_scope="platform",
        provider_connection_id="global_openrouter",
        model_config_id="global_sonnet",
    ),
    model="anthropic/claude-sonnet-4",
    messages=messages,
    stream=True,
)
```

Example usage for a user-owned model:

```python
response = await client.acompletion(
    scope=RequestScope(
        owner_scope="user",
        owner_id="user_123",
        provider_connection_id="user_123_openrouter",
        model_config_id="user_123_sonnet",
    ),
    model="anthropic/claude-sonnet-4",
    messages=messages,
)
```

## Integration With ImaginAI

ImaginAI should own:

- users, auth, roles, and permissions
- encrypted DB storage for secrets
- provider connection UI
- model config UI
- scenario/adventure prompts and storytelling logic
- model config selection and ordering
- admin-created global model configs
- user-created personal model configs

`rotator_library` should own:

- all provider communication
- streaming and non-streaming completions
- embeddings if used later
- provider protocol translation
- retries and failover
- credential rotation and load balancing
- cooldown and quota-aware selection
- provider model discovery
- usage and quota stats at the credential/provider/model level

## Acceptance Criteria

The library update is sufficient when these scenarios work:

1. A process starts with env credentials. They are imported as platform credentials and existing proxy-style requests still work.
2. ImaginAI registers a new user-owned OpenRouter provider connection and API key at runtime. No restart is needed.
3. ImaginAI fetches `/v1/models` for that user's OpenRouter connection using only that user's key.
4. A user selects their personal model config. The request uses only that user's credential.
5. A user selects a global model config. The request uses the platform pool and normal rotation/load balancing.
6. Two users with the same provider and same upstream model cannot use each other's credentials accidentally.
7. Disabling a credential immediately removes it from selection while preserving historical stats.
8. Removing/changing a provider connection invalidates model caches and prevents stale routing.
9. Streaming and non-streaming paths both respect request scope.
10. Quota stats can be filtered separately for platform pool, each user, each provider connection, and each credential.
11. Logs, persisted usage files, and error responses never expose raw secrets.

## Implementation Notes

The most important internal change is to stop treating `provider -> list[raw credentials]` as the core data model. The core should become:

```text
provider_connection_id -> ProviderConnection
credential_id -> CredentialDescriptor
request scope -> eligible credential ids -> selected credential id -> resolved secret/accessor -> provider call
```

This allows ImaginAI to keep a clean UI and data model while preserving the library's existing rotation machinery.
