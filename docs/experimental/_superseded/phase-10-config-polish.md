# Phase 10 Plan: Config Polish

## Goal

Make the new protocol/routing/field-cache/streaming/pricing features configurable in a consistent, documented, testable way without replacing the current `.env` workflow. Phase 10 should add a small optional JSON configuration layer, keep environment variables as the final override, document all new knobs in `.env.example`, and expose validation helpers so invalid config fails clearly instead of silently changing routing or accounting behavior.

## Non-Goals

- Do not introduce SQLite or any database.
- Do not replace `.env` as the primary user workflow.
- Do not replace provider class declarations, `UsageManager`, `SessionTracker`, `SelectionEngine`, or provider quota trackers.
- Do not implement full multi-user/security config.
- Do not move secrets into JSON config.
- Do not require JSON config for existing deployments.
- Do not rewrite every direct `os.getenv()` call in the repo.
- Do not change default routing, pricing, usage, or streaming behavior when no new config is present.

## Configuration Precedence

- Built-in defaults and provider declarations are the base.
- Optional JSON config may provide structured routing/pricing/streaming/provider metadata.
- Environment variables override JSON config.
- Request-level explicit routing/provider fields still win for that request where already supported.
- Secrets remain environment/OAuth-file based; JSON config should not contain API keys or bearer tokens.

## Current Code Context

- `.env.example` documents many legacy env vars but not all Phase 6-9 additions.
- `routing/config.py` currently reads fallback groups and model routes from env only.
- Phase 9 added `ModelPricing`, `CostCalculator`, and `ProviderInterface.get_model_pricing()` but not env/JSON pricing.
- Phase 8 added stream metrics primitives but runtime TTFB/stall/heartbeat env knobs are not implemented.
- Phase 7 added provider cooldown env parsing in `retry_policy.provider_cooldown_env()`.
- Phase 3 added field-cache rule dataclasses, but no user-facing JSON config parser for cache rules.
- Provider base URLs and native provider declarations are mostly provider methods or direct env vars.
- Existing usage config env parsing is large and should not be rewritten in Phase 10.
- Reports are user-facing only and should remain uncommitted.

## Files To Add

- `src/rotator_library/config/experimental.py`
- `tests/test_experimental_config.py`
- `tests/test_config_pricing.py`
- `tests/test_config_routing_json.py`
- `tests/test_config_stream_settings.py`
- `tests/test_env_example_experimental_config.py`
- Maybe `docs/experimental/config-reference.md` if the implementation needs more detail than `.env.example`.

## Files Likely To Touch

- `src/rotator_library/routing/config.py`
- `src/rotator_library/usage/costs.py`
- `src/rotator_library/client/streaming.py`
- `src/rotator_library/client/executor.py` only if streaming env knobs need to be passed through.
- `src/rotator_library/field_cache/rules.py` or equivalent only if JSON parsing needs a small helper.
- `src/rotator_library/providers/provider_interface.py` only if pricing declarations need docstring/typing alignment.
- `.env.example`
- `docs/experimental/phase-10-config-polish.md`

## JSON Config Model

Add `ExperimentalConfig` dataclass with optional sections:

- `routing`
- `pricing`
- `streaming`
- `field_cache`
- `providers`

Add loader:

- `load_experimental_config(path=None, env=None)`
- If `path` is `None`, read from `LLM_PROXY_CONFIG_FILE` or `PROXY_CONFIG_FILE`.
- Missing path returns an empty config.
- Invalid JSON raises `ExperimentalConfigError`.
- Unknown top-level sections should be preserved in metadata or warned about, not fatal.

Add helpers:

- `as_bool()`
- `as_int()`
- `as_float()`
- `env_key(provider, model, suffix)` sanitizing provider/model names consistently.

JSON should not read or interpolate secrets.

## Routing JSON

Support JSON shape:

- `routing.fallback_groups.<name>.targets = ["codex/gpt-5.1-codex@native", "openai/gpt-5.1@litellm_fallback"]`
- `routing.fallback_groups.<name>.failover_on = ["rate_limit", "server_error"]`
- `routing.fallback_groups.<name>.stop_on = ["authentication", "validation"]`
- `routing.model_routes.<model_alias> = "group:code_chain"` or target spec.

`routing/config.py` should merge JSON first, then env:

- env `FALLBACK_GROUPS`, `FALLBACK_GROUP_*`, `MODEL_ROUTE_*` override or add entries.
- env overrides should retain current behavior for existing deployments.

Validation:

- empty groups are invalid.
- `group:` model routes must reference known groups after merge.
- invalid target specs raise `RoutingConfigError`.

Tests:

- JSON-only routing works.
- env override replaces JSON group targets.
- env model route can reference JSON group.
- invalid JSON group route fails clearly.
- existing env-only routing tests still pass.

## Pricing Config

Add env pricing support in `CostCalculator` or a small helper:

- `MODEL_PRICE_{PROVIDER}_{MODEL}_INPUT`
- `MODEL_PRICE_{PROVIDER}_{MODEL}_OUTPUT`
- `MODEL_PRICE_{PROVIDER}_{MODEL}_CACHE_READ`
- `MODEL_PRICE_{PROVIDER}_{MODEL}_CACHE_WRITE`
- `MODEL_PRICE_{PROVIDER}_{MODEL}_REASONING`

Add JSON pricing support:

- `pricing.<provider>.<model>.input`
- `pricing.<provider>.<model>.output`
- `pricing.<provider>.<model>.cache_read`
- `pricing.<provider>.<model>.cache_write`
- `pricing.<provider>.<model>.reasoning`
- optional `currency`

Precedence:

- provider explicit `get_model_pricing()` first, because provider code may know exact native pricing.
- env pricing overrides JSON pricing.
- JSON pricing applies before LiteLLM fallback.
- LiteLLM remains last fallback.

If env parsing is invalid, log warning and ignore that component.

Tests:

- JSON pricing calculates buckets.
- env pricing overrides JSON pricing.
- provider explicit pricing remains highest priority.
- missing pricing remains `unavailable`.
- `skip_cost_calculation` still wins over all pricing sources.

## Streaming Settings

Add `StreamRuntimeSettings` dataclass:

- `ttfb_timeout_seconds`
- `stall_timeout_seconds`
- `heartbeat_seconds`
- `trace_metrics`

Load from JSON `streaming` and env:

- `STREAM_TTFB_TIMEOUT_SECONDS`
- `STREAM_STALL_TIMEOUT_SECONDS`
- `STREAM_HEARTBEAT_SECONDS`
- `STREAM_TRACE_METRICS`

Phase 10 should not enforce timeouts by default.

Minimal runtime integration:

- `StreamingHandler.wrap_stream()` should read settings and conditionally emit lifecycle metrics only when trace metrics enabled, default true to preserve Phase 8 behavior.
- Do not implement heartbeat injection unless small and obviously safe. If implemented, default disabled and only emit SSE comments during controlled tests.
- Do not abort streams on TTFB/stall by default. If values are configured, trace `stream_stall_detected` when observable, but do not sever client streams unless explicitly `STREAM_STALL_ABORT=true` is introduced and tested. Prefer no abort in Phase 10.

Tests:

- JSON/env settings parse.
- env overrides JSON.
- `STREAM_TRACE_METRICS=false` suppresses lifecycle trace passes while SSE output remains unchanged.
- default trace behavior remains current.

## Field-Cache Config

If feasible, add parser for JSON field-cache rule declarations without wiring every provider:

- `field_cache.<provider>.<model or "*">[]`
- fields: `name`, `source`, `path`, `scope`, `mode`, `target_path`, `max_entries`.

Keep it as helper-only if integration is too risky:

- parse into existing `FieldCacheRule` dataclasses.
- providers/Phase 10+ can call it from `get_field_cache_rules()`.

Tests:

- valid JSON rule parses.
- invalid paths/rule modes fail clearly.
- provider/model wildcard merge order documented and tested if implemented.

If the existing field-cache dataclasses do not support all desired fields cleanly, document and defer live provider integration.

## Provider Metadata Config

- Support safe non-secret provider metadata only.
- API base URLs can remain existing `*_API_BASE` env vars for now.
- JSON may define provider protocol/adapter names for future use, but Phase 10 should not make untrusted JSON instantiate arbitrary classes.
- Allow only names that already exist in protocol/adapter registries if validation is implemented.
- Do not place API keys, OAuth tokens, or authorization headers in JSON.
- Tests can validate config rejects/ignores secret-looking keys like `api_key`, `authorization`, `access_token`.

## Config Validation And Diagnostics

Add a validation result:

- warnings for unknown sections/keys.
- errors for invalid routing target specs and invalid numeric pricing.
- no credential values in error messages.

Add a CLI-free helper test; do not add a new CLI unless tiny.

Add transform trace or startup log only if a config is loaded:

- log path, loaded sections, warning count.
- no config contents with secrets.

## `.env.example` Updates

Add section for Phase 6 routing:

- `FALLBACK_GROUPS`
- `FALLBACK_GROUP_<NAME>`
- `MODEL_ROUTE_<MODEL_ALIAS>`
- execution suffixes `@auto`, `@native`, `@custom`, `@litellm_fallback`.

Add section for Phase 7 provider cooldown:

- `PROVIDER_COOLDOWN_MIN_SECONDS`
- `PROVIDER_COOLDOWN_DEFAULT_SECONDS`
- `PROVIDER_COOLDOWN_ON_QUOTA`
- transient retry delay/jitter already in defaults but should be documented.

Add section for Phase 8 streaming:

- `STREAM_RETRY_ON_REASONING_ONLY`
- `STREAM_TRACE_METRICS`
- `STREAM_TTFB_TIMEOUT_SECONDS`
- `STREAM_STALL_TIMEOUT_SECONDS`
- `STREAM_HEARTBEAT_SECONDS`

Add section for Phase 9 pricing:

- `MODEL_PRICE_<PROVIDER>_<MODEL>_INPUT`
- `MODEL_PRICE_<PROVIDER>_<MODEL>_OUTPUT`
- cache/reasoning variants.

Add optional JSON config:

- `LLM_PROXY_CONFIG_FILE=./config/llm-proxy.json`
- note that env overrides JSON and secrets should stay in env/OAuth files.

Keep docs concise; do not document every old legacy env in Phase 10 unless already present.

## Test Modernization

- Do not take on the entire stale broad test suite unless small.
- Phase 10 can add focused tests for config parsing and maintained regression set.
- Existing stale ignored tests should not block completion unless they are tracked and part of maintained runs.

## Implementation Checkpoints

1. Add `ExperimentalConfig` loader and validation helpers with tests.
2. Integrate JSON+env routing config merge with existing routing tests.
3. Add JSON/env model pricing support to `CostCalculator` with tests.
4. Add stream runtime settings parsing and `STREAM_TRACE_METRICS` runtime integration with tests.
5. Add field-cache JSON rule parser helper if feasible with tests.
6. Update `.env.example` and optional config reference doc.
7. Run Phase 10 focused tests and Phase 1-9 maintained regressions.
8. Review with `explore` and `explore-heavy`; fix findings; write uncommitted Phase 10 report.

## Risks And Mitigations

- Config precedence confusion. Mitigation: tests for defaults, JSON, env override, provider explicit override, and request-level route behavior.
- Accidentally allowing secrets in JSON. Mitigation: validation rejects secret-looking keys in safe config sections.
- Routing merge could alter existing env behavior. Mitigation: env-only tests remain unchanged and env overrides JSON.
- Pricing could become authoritative incorrectly. Mitigation: keep costs advisory and return zero/unavailable when ambiguous.
- Streaming trace toggles could hide useful debug data unexpectedly. Mitigation: default trace metrics remains enabled.
- Over-scoping into a full config framework. Mitigation: only structured config for new experimental features; leave legacy env parsing intact.
