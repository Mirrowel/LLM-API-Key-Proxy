# Provider configuration example — key reference

This directory documents the `providers` section of the optional structured
JSON config. `provider-config.example.json` is a valid sample; because JSON has
no comments, this README is the annotation layer for every key.

The config file is selected by environment variable:

| Variable | Meaning |
| --- | --- |
| `LLM_PROXY_CONFIG_FILE` | Path to the JSON config. Checked first. |
| `PROXY_CONFIG_FILE` | Fallback path. |

A path that is set but does not exist is a startup error (fail-loud). Unset
means "no config". Env vars always remain available and override the JSON for
the settings that have env equivalents.

> **Credentials are forbidden here.** The loader rejects secret-like keys
> (`api_key`, `authorization`, `token`, `credential`, `password`, ...) anywhere
> in the config. API keys stay in `<NAME>_API_KEY` env vars or provider
> credential files. See `src/rotator_library/config/experimental.py`
> (`_SECRET_KEY_PARTS`, `_reject_secret_keys`).

---

## 1. Precedence

Later layers win:

```
code class attribute  <  JSON providers.<name>  <  environment variable
```

Two documented exceptions exist:

* **Field-cache rules** merge by rule *name* with the order
  `JSON field_cache > env <NAME>_CACHE_REPLAY > class cache_replay >
  provider-declared rules`. A same-name override must not weaken isolation or
  injection behavior, or the request fails with a `configuration_error`.
* **`<PROVIDER>_API_BASE`** overrides a code provider's `default_api_base`, but
  only if that provider declared one (`ProviderInterface.get_provider_api_base`
  returns `None` when `default_api_base is None`). Dynamic providers read
  `<NAME>_API_BASE` directly.

---

## 2. Top-level sections

| Section | Status | Purpose |
| --- | --- | --- |
| `providers` | documented here | Code-free provider definitions / tuning. |
| `routing` | other runtime | Fallback groups and model routes. |
| `pricing` | other runtime | Advisory per-model pricing. |
| `streaming` | other runtime | Stream observability and timeouts. |
| `retry` | other runtime | Provider cooldown/backoff tuning. |
| `responses` | other runtime | Responses API store policy. |
| `field_cache` | other runtime | Provider-independent field-cache rules. |
| `hooks` | see §7 | Process-wide (`hooks.global`) hook names. |

Unknown top-level sections are collected with a warning and ignored.

---

## 3. `providers.<name>` keys

`<name>` must match `^[A-Za-z0-9][A-Za-z0-9_-]*$` and is case-folded for
lookup. The name is the provider identity used in `provider/model` references,
in quota groups, and in usage accounting.

| Key | Type | Default | What it does |
| --- | --- | --- | --- |
| `protocol_name` | string | unset (LiteLLM fallback) | Native dialect. Must be a registered *generative* protocol: `openai_chat`, `responses`, `anthropic_messages`, `gemini`, `ollama` (registry-derived allowlist includes sibling variants). Required when `auth_mode` is `none`. |
| `api_base` | string | unset (required for a config-defined provider) | Transport base. Must be `http(s)`, and must not contain credentials, query parameters, or fragments. A trailing `/` is stripped. |
| `endpoint_paths` | object `{operation: path}` | `{}` | Per-operation path templates appended to `api_base`. Values must start with `/`, are absolute paths on the base, and may use only `{model}`, `{operation}`, `{provider}` placeholders. Fragments and secret-like query keys are rejected; non-secret selectors such as `?alt=sse` are allowed. |
| `auth_mode` | `bearer` \| `x-api-key` \| `x-goog-api-key` \| `custom` \| `none` | `bearer` | Credential header scheme. `custom` requires `auth_header_name`; `none` requires `protocol_name` and mints the internal no-auth rotation slot. |
| `auth_header_name` | string | unset | Header name for `auth_mode: custom`. Must be a valid HTTP token. |
| `models` | array of non-empty strings | `()` | Configured model list. A dynamic provider returns these (provider-prefixed) instead of querying a listing endpoint. |
| `adapter_names` | array (or comma string) | unset → class attribute | Ordered payload adapter chain. Every name must resolve in the adapter registry. Order is significant. |
| `adapter_config` | object `{adapter: {config}}` | `{}` | Per-adapter configuration (for example `model_override: {model: ...}`). Keys must resolve in the adapter registry. |
| `native_streaming_supported` | boolean | unset → class attribute → `False` | Opt in to the native stream executor for this provider/model. |
| `field_cache` | array of rule objects | `()` | Provider state to cache and replay. Distinct schema from `cache_replay` — see §5. |
| `model_quota_groups` | object `{group: [models]}` | unset → class attribute | Models that share a quota pool and reset together. Overridable by `QUOTA_GROUPS_<PROVIDER>_<GROUP>`. |
| `hooks` | array (or comma string) of names/objects | unset → class attribute | Extra pipeline hooks appended after class hooks. See §7. |
| `transport_profiles` | object `{profile: entry}` | unset | Multi-face transport declarations. See §4. |
| `default_profile` | string | unset | Profile used for a bare `provider/model` request. Must name a declared profile. |
| `profiles` | object | unset | Alias for `transport_profiles`; `transport_profiles` wins when both are present. |
| `cache_replay` | array | unset | **Accepted by validation but not consumed** by the runtime-config loader today. Use `field_cache`, or the env/class `cache_replay` surfaces (§6). |
| `model_protocols` | object | unset | **Accepted by validation but not consumed** (reserved acceptance table). Per-model protocol acceptance is not implemented. |

`protocol_name` in the example for `myserver` is `openai_chat`, so the generic
native runtime builds a Chat Completions body and sends it to
`https://api.myserver.example/v1/chat/completions`.

---

## 4. `transport_profiles` entry schema

Each profile entry declares one transport face of the same provider identity.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `protocol` (or `protocol_name`) | string | provider `protocol_name` | Generative protocol for this face. |
| `endpoint_paths` | object | `{}` | Per-operation paths. Wins over the provider-level `endpoint_paths`. |
| `endpoint_path` | string | unset | Legacy singular path applied to any operation when `endpoint_paths` has no entry. |
| `auth_mode` | string | provider `auth_mode` | Per-profile auth override. |
| `auth_header_name` | string | provider value | Required when the profile's `auth_mode` is `custom`. |

Addressing: `provider:profile/model` pins a profile explicitly and fails loudly
if the profile is unknown. A bare `provider/model` uses `default_profile`, or
the unique profile matching the client's protocol; when several profiles match
and none is the default, the request is rejected with the candidate list rather
than guessed.

The `acme` entry shows three faces behind one identity: a native Gemini face,
an OpenAI-compatible face, and an Anthropic Messages face. Requests are
`acme:openai/acme-pro` or `acme:anthropic/acme-pro`.

---

## 5. `field_cache` rule schema (JSON)

This is the array form consumed by `providers.<name>.field_cache` (it is also
accepted directly as a list, which is treated as `{"*": [...]}`). Rules compile
to `FieldCacheRule` objects (`field_cache/types.py`,
`config/experimental.py::_field_cache_rule_from_dict`).

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | string | **required** | Filesystem-safe (no `/`, `\`, `:`). |
| `source` | string | **required** | `request`, `response`, `stream_event`, `unified_request`, `unified_response`, `unified_stream_event`. |
| `path` | string | **required** | Extraction path: dotted keys, `[n]` indexes, `*` wildcard. |
| `target_path` | string | unset | Legacy injection shorthand; builds `inject = {target: data.target or "request", path: target_path}`. |
| `mode` | string | `last` | `last`, `all`, `last_user_turn`, `last_assistant_turn`, `per_tool_call` (the latter needs `metadata.tool_call_id_path`). |
| `scope` | string or array | `provider,model,credential,session` | **Must include all four** of `provider`, `model`, `credential`, `session` in this JSON schema. |
| `inject` | object | unset | `{target, path, when_missing_only, insert, as_list}`. `target` ∈ `request`, `unified_request`, `metadata`, `response`, `unified_response`. |
| `enabled` | boolean | `true` | Disable without deleting. |
| `critical` | boolean | `false` | `true` makes a rule error fail the request; `false` contains it (log + skip). |
| `ttl_seconds` | integer | unset | Retention window. |
| `metadata` | object | `{}` | Rule metadata; may carry `compatibility`, `transform`, `tool_call_id_path`. |
| `allow_missing_session` | boolean | `false` | Permit caching when no session id is resolved. |
| `max_values` | integer | `1024` | Value-count bound. |
| `max_bytes` | integer | `4194304` | Byte bound. |
| `cache_key` | string | unset | Explicit cache key (filesystem-safe). |

EXTRACTION and INJECTION run on the **native execution path only**. A provider
using a custom `acompletion()` or the LiteLLM fallback neither caches nor
injects.

---

## 6. `cache_replay` (declarative) surfaces

`cache_replay` is a list of declarative rules that compile to the same engine.
It works as:

* a **class attribute** on a code provider
  (`ProviderInterface.cache_replay`), and
* the **`<NAME>_CACHE_REPLAY`** environment variable holding JSON text.

`providers.<name>.cache_replay` in JSON is accepted by validation but not read
by the current runtime; use `field_cache` for JSON-configured rules.

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | string | **required** | Rule name (also the override key). |
| `source` | string | `response` | Same vocabulary as §5. |
| `path` | string | **required** | Extraction path. |
| `keep` | string | `last` | `last`, `all`, `turn`, `turns:N`, `per_tool_call`. |
| `inject.path` | string | unset | Restore path. Without it the rule only caches. |
| `inject.if` | `auto` \| `always` | `auto` | `auto` injects only when absent; `always` overwrites. |
| `inject.target` | string | `request` | Restore target. |
| `inject.insert` | boolean | `false` | List-tail insertion. |
| `inject.as_list` | boolean | `false` | Treat values as a list. |
| `compatibility` | `bound` \| `portable` | `bound` | Opaque provider state vs. transformable state. |
| `transform` | string | unset | Registered transform; requires `compatibility: portable`. |
| `scope` | array | `provider,model,credential,session` | Strengthened to always include `provider` and `model`. |
| `critical` | boolean | `false` | Fail-closed escape hatch. |
| `ttl_seconds` | integer | unset | Retention window. |
| `tool_call_id_path` | string | unset | Required for `keep: per_tool_call`. |

---

## 7. `hooks`

Provider `hooks` entries are either a bare name string or an object:

| Key | Type | Default | Notes |
| --- | --- | --- | --- |
| `name` | string | **required** | Must resolve in the hook registry at startup. |
| `stages` | array | the hook's own stages | Optional override of the bound stages. |
| `priority` | integer | hook's `priority` | Smaller fires first. |
| `critical` | boolean | hook's `critical` | `true` fails the request on hook error. |

Resolution order is class hooks, then JSON hooks, then globals; ties break by
that order after `priority`. Unknown names/stages fail at startup, never per
request. Today the only hooks registered automatically are adapter bridges
named `adapter:<adapter-name>` (for example `adapter:suppress_developer_role`);
any other hook name must be registered by provider code before startup.

`hooks.global` at the top level of the config lists process-wide hook names,
applied to every request after provider declarations.

---

## 8. Adapters

Built-ins (`adapters/builtin.py`) and their configs:

| Adapter | Aliases | Config |
| --- | --- | --- |
| `noop` | `none`, `passthrough` | — |
| `model_override` | `override_model` | `{model: "upstream-id"}` |
| `suppress_developer_role` | `developer_role` | `{mode: "system" \| "user" \| "drop"}` |
| `reasoning_content` | `reasoning_rewrite` | `{output_field, source_fields}` (OpenAI-chat wire only) |
| `field_rename` | `field_copy` | `{rules: [{source_path, target_path, stage, move, as_list, when_missing_only}]}` |
| `antigravity_envelope` | — | `{request_type, user_agent, project}` — must be declared **last**. |

Ordering convention: envelope adapters wrap everything before them, so any
content-level adapter must run first.

---

## 9. Authentication modes

| `auth_mode` | Header sent | Requires |
| --- | --- | --- |
| `bearer` (default) | `Authorization: Bearer <credential>` | — |
| `x-api-key` | `x-api-key: <credential>` | — |
| `x-goog-api-key` | `x-goog-api-key: <credential>` | — |
| `custom` | `<auth_header_name>: <credential>` | `auth_header_name` |
| `none` | no header | `protocol_name`; no credential needed |

Dynamic providers with no credential env and no explicit `auth_mode` mint the
internal no-auth slot automatically. A code provider declares
`default_auth_mode = "none"` for the same behavior (the local Ollama example).

The `local-ollama` entry shows `auth_mode: none`; `gateway` shows a custom
header. When a provider is created from JSON without an `api_base`, startup
fails with `providers.<name>.api_base` required.

---

## 10. Models and quota groups

`models` (JSON) and `<NAME>_MODELS` (env) are **different things**:

* `providers.<name>.models` is the *discovery list* for a dynamic provider.
  When present, `get_models()` returns it (provider-prefixed) instead of
  querying a listing endpoint. Values are non-empty strings.
* `<NAME>_MODELS` is a *model-definitions* JSON value (`["model-a"]` or
  `{"model-a": {"id": "upstream-id", "options": {...}}}`) consumed by
  `ModelDefinitions` for id aliases and per-model default options. It does not
  feed dynamic-provider discovery.

`model_quota_groups` maps a group name to its member models. Members share
cooldowns and reset together. `QUOTA_GROUPS_<PROVIDER>_<GROUP>` (comma list)
overrides a group's members; an empty value disables the group.

There is **no** per-model modality key and **no** per-model profile-limit key
in this surface. Modality metadata is served by the model-info layer from
upstream/`/models` data, and profile selection is per-request
(`provider:profile/model`), not per-model configuration.

---

## 11. Environment knobs for providers

These are the actual env names read by code (do not invent others):

| Variable | Read by | Effect |
| --- | --- | --- |
| `<NAME>_API_BASE` | dynamic provider, `ProviderConfig`, `get_provider_api_base` | Transport base (dynamic) / override (code with a declared base). |
| `<NAME>_API_KEY`, `<NAME>_API_KEY_1`, ... | `proxy_app/main.py` env scan | Credentials; numbered keys rotate independently. |
| `<NAME>_MODELS` | `model_definitions.py` | Model definitions (id aliases, per-model options). |
| `<NAME>_CACHE_REPLAY` | `client/executor.py` | JSON cache_replay rule list. |
| `QUOTA_GROUPS_<PROVIDER>_<GROUP>` | `provider_interface.py` | Quota-group member override. |
| `ROTATION_MODE_<PROVIDER>` | provider interface | `balanced` / `sequential`. |
| `FAIR_CYCLE_*`, `CUSTOM_CAP_*`, `MAX_CONCURRENT_*`, `ROTATION_MODE_*` | usage config | Usage/rotation tuning. |
| `LLM_PROXY_CONFIG_FILE`, `PROXY_CONFIG_FILE` | `config/experimental.py` | This config file. |

**Not present anywhere in the codebase** (despite being a common assumption):
`<NAME>_PROTOCOL`, `<NAME>_AUTH_MODE`, `<NAME>_AUTH_HEADER_NAME`,
`<NAME>_ENDPOINT_CHAT`, and `<NAME>_CONFIG`. For a config-defined provider,
protocol/auth/endpoints/profiles/adapters/hooks are declared in this JSON.

---

## 12. Validation and failure modes

Startup validation (`experimental.py::_validate_provider_sections`) rejects:

* unsupported keys under a provider (anything outside the table in §3);
* unknown/non-generative `protocol_name`;
* malformed `api_base` (non-HTTP, credentials, query, fragment);
* endpoint paths that are not absolute, contain fragments, use unsupported
  placeholders, or carry secret-like query keys;
* `auth_mode: custom` without `auth_header_name`, or `auth_mode: none`
  without `protocol_name`;
* unknown adapter names, malformed model lists, bad profile names, a
  `default_profile` that is not declared, and unknown hook names/stages.

---

## 13. Walkthrough of `provider-config.example.json`

* `myserver` — single-protocol `openai_chat` provider with an explicit
  `api_base`, a configured `models` list, `model_override` adapter + config,
  a `field_cache` rule that replays the response id as a prompt-cache key, a
  quota group, and an extra hook.
* `acme` — multi-protocol provider addressed as `acme:profile/model`. Its
  `native`, `openai`, and `anthropic` profiles each declare their own
  `endpoint_paths` and `auth_mode`; `default_profile` is `native`.
* `local-ollama` — zero-credential local provider (`auth_mode: none`).
* `gateway` — custom auth header (`X-API-Token`).

To use it: set `LLM_PROXY_CONFIG_FILE=./docs/examples/provider-config.example.json`
and provide credentials such as `MYSERVER_API_KEY`, `ACME_API_KEY`, and
`GATEWAY_API_KEY` in the environment. Do not add credentials to the JSON.
