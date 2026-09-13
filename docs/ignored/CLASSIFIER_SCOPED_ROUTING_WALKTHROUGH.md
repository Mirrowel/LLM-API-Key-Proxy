# Classifier-Scoped Routing Walkthrough

This file tracks implementation progress and key decisions so work can resume after context compaction or interruption.

## Current Objective

Implement classifier-scoped routing and usage isolation in `rotator_library` while preserving existing no-classifier behavior.

## Decisions Locked In

- Stateless per-call mode is first-class: callers can pass `classifier`, provider configs, and keys on each operation.
- Registered scope mode is also supported later for routers, model discovery, and durable management.
- The library does not own app users, permissions, billing, or encrypted database storage.
- A classified operation may inherit global provider definitions/base URLs by default.
- A classified operation must not inherit global API keys by default.
- Request/provider overlays must not mutate global provider config or other classifiers.
- Same provider name can be reused by different classifiers with different base URLs and keys.
- Usage, cooldowns, quotas, model discovery caches, stats, and filters must be separable by classifier/provider.
- Model filters hide `/v1/models` results only by default; completions are not blocked by filters unless a future hard-enforcement option is added.
- Global model filter propagation is configured globally per provider and defaults to false for both whitelist and blacklist.
- First pass supports current OpenAI-compatible chat/completions custom provider behavior. `protocol` is future-facing.

## Architecture Notes From Exploration

- `RotatingClient` currently builds `self.all_credentials: Dict[str, List[str]]` from startup API keys and OAuth paths.
- `RotatingClient.acompletion()` currently extracts provider from `model.split("/")[0]`, uses all credentials for that provider, builds `RequestContext`, and delegates to `RequestExecutor`.
- `RequestExecutor` already handles retry, rotation, streaming, non-streaming, cooldown waits, and usage recording. It should be reused.
- `UsageManager` currently initializes once with raw credential accessors, derives stable ids from raw secrets/paths, and persists raw `accessor` plus `accessor_index`.
- Custom providers are currently startup/env-only: `*_API_BASE` creates a dynamic provider and `ProviderConfig` maps unknown providers through LiteLLM OpenAI-compatible routing.
- Model filters are loaded by the proxy from env, passed to `RotatingClient`, and applied in library model discovery through `ModelResolver.is_model_allowed()`.

## Implementation Log

- Created `CLASSIFIER_SCOPED_ROUTING_IMPLEMENTATION_PLAN.md` with the full implementation strategy.
- Created this walkthrough file.
- Added first request-scope fields to `RequestContext`: `usage_manager_key`, request-local `provider_config`, `credential_secrets`, and `classifier`.
- Extended provider transformation plumbing so a request-local provider override can reach `ProviderConfig.convert_for_litellm()` without mutating global provider config.
- `ProviderConfig.convert_for_litellm()` now accepts an optional `provider_override` and prefers `base_url`/`api_base` from that override before env/global API bases.
- Next edit is the executor/client path that will use scoped usage-manager keys and resolve private credential ids back to raw secrets only at the provider call boundary.
- `RequestExecutor` now looks up usage managers by `context.usage_manager_key` when present, passes request-local provider overrides into transforms, and resolves `context.credential_secrets` only when injecting `api_key`/`credential_identifier` into the provider call. Retry, logging, failure accumulation, and usage state continue to see the safe credential id.
- `UsageManager.initialize()` no longer returns immediately after first initialization; it can resync active credentials while preserving loaded historical state. This is needed for scoped/stateless calls where the active credential set can change per request.
- `RotatingClient` now has classifier helpers for safe directory names, HMAC private credential fingerprints, scoped usage-manager keys/files, and scoped usage-manager creation under `usage/classifiers/<safe_classifier>/usage_<provider>.json`.
- `RotatingClient.acompletion()` and `aembedding()` now pop `classifier`, `api_keys`, `providers`, `private`, and `model_filters` from call kwargs; resolve an isolated scope; pass only relevant provider credentials into `RequestContext`; and attach provider overrides/credential-secret maps.
- Added in-memory registered scope management methods: `register_scope`, `update_scope`, `get_scope`, `remove_scope`, provider add/update/remove/list helpers, and credential add/set/remove/list helpers. `get_scope()`/`list_scope_credentials()` hide raw secrets by default and support `include_secrets=True` for host-managed retrieval.
- `get_available_models()` and `get_all_available_models()` now accept `classifier`, `api_keys`, `providers`, `private`, `model_filters`, and `force_refresh`. Classified listing uses scoped credentials only, resolves registered/request provider overrides, and caches by classifier/provider/config/filter/credential-set fingerprint.
- Scoped model discovery applies request/classifier filters only. Global env/proxy filters still apply to default no-classifier listing through the existing `ModelResolver`; classified listing does not inherit those filters by default.
- Runtime provider model discovery can fetch OpenAI-compatible `/models` directly when a request/registered provider override supplies `base_url`/`api_base`, so custom provider aliases do not require mutating `PROVIDER_PLUGINS`.
- `get_quota_stats()` now accepts `classifier`; default stats skip classifier-scoped managers to preserve old behavior, while classifier stats select managers under that classifier prefix.
- Private scoped credentials are represented in usage state by `private:<hmac>` identifiers only. Raw secrets are held in `RequestContext.credential_secrets` and resolved only when injecting credentials into provider calls.
- `UsageStorage` no longer writes private credential identifiers to `accessor_index`, and serialized private credential states include `private: true`. `UsageManager.get_stats_for_endpoint()` sets `full_path` to `None` for private credentials and marks those entries with `private: true`.
- Verification run: `python -m compileall src` passes.
- Verification run: scoped registration smoke test passes. It confirmed registered scopes hide secrets by default, private credential ids start with `private:`, scoped execution resolves the raw secret only through `credential_secrets`, and classified `openai` did not fall back to global `openai` credentials.
- Verification run: scoped model discovery smoke test passes with an `httpx.MockTransport`; it queried the request/registered `base_url`, sent the raw secret in the Authorization header, returned scoped models, and applied scoped blacklist filtering.
- Added `tests/test_classifier_scoped_routing.py` as the durable test suite for this feature. Coverage includes default/global completion compatibility, stateless private scoped completion, streaming scoped completion, no fallback from classifier to global keys, request overlay precedence over registered scope state, registered scope add/set/remove/fetch behavior, scoped model discovery/cache/filtering, non-propagation of global filters to classifiers, private stats hiding `full_path`, classifier-filtered quota stats, usage-manager active credential resync, and provider override non-mutation.
- Verification run: `python -m pytest tests/test_classifier_scoped_routing.py -q` passes with 11 tests.
- Verification run: `python -m compileall src tests/test_classifier_scoped_routing.py` passes.
- Broader suite note: `python -m pytest tests -q --ignore=tests/_retired` currently fails during collection on unrelated pre-existing test/code drift: missing `rotator_library.fallback_groups`, missing `UsageStats` export, and missing `DEFAULT_QUOTA_COSTS` in `gemini_cli_quota_tracker`. A stable subset including `tests/refactor` also has unrelated legacy failures around old `UsageManager` constructor signatures, missing old `CredentialState.usage`, async tests without pytest async plugin, and old custom-cap parsing expectations. The new classifier-scoped tests are isolated and passing.
- Added user-facing documentation for classifier-scoped routing: `docs/CLASSIFIER_SCOPED_ROUTING.md` now describes the problem, core rules, stateless calls, registered scope management, private credential identity, usage file layout, scoped model discovery, model filters, runtime provider overrides, streaming/embedding support, limitations, and test coverage.
- Updated `README.md` with a high-level feature mention, a direct scoped routing example, key behavior bullets, a link to the dedicated guide, and the scoped usage file path in the configuration file table.
- Updated `src/rotator_library/README.md` with library API details for scoped `acompletion`, `aembedding`, model discovery, classifier quota stats, stateless scoped calls, registered scope management, privacy behavior, and a link to the dedicated guide.
- Updated `DOCUMENTATION.md` with technical architecture notes covering effective scope resolution, request context scoped fields, no global-key fallback, private HMAC credential identity, scoped usage storage/stats, and scoped model discovery/cache behavior.
- Review fix: `RequestExecutor._ensure_initialized()` now resyncs initialized scoped usage managers on each request while preserving the global/default fast path. This fixes stateless per-call key changes and registered `set_scope_credentials()` replacements after a scope has already been used.
- Review fix: scoped usage-manager construction now applies the client constructor's `rotation_tolerance` and provider `get_usage_reset_config()` behavior just like global/default managers.
- Review fix: scoped model discovery cache keys now include a safe fingerprint of the effective credential ids, preventing model-list reuse across different scoped/stateless keys for the same classifier/provider/base URL.
- Review fix: `.gitignore` now narrowly unignores `tests/test_classifier_scoped_routing.py` despite the local `.git/info/exclude` `tests/` rule, while keeping unrelated local tests ignored. The test file is self-contained and no longer imports untracked `tests.refactor.helpers`.
- Added regression tests for the review findings: executor-path stateless credential resync, registered `set_scope_credentials()` resync, scoped manager rotation/reset-config inheritance, and credential-sensitive model cache keys. Verification run: `python -m pytest tests/test_classifier_scoped_routing.py -q` passes with 15 tests.
- Verification run after review fixes: `python -m compileall src tests/test_classifier_scoped_routing.py` passes.
- Modularization pass: extracted classifier scope state and registered scope management into `src/rotator_library/client/scopes.py` as `ScopeManager`. `RotatingClient` keeps thin compatibility delegates for `_safe_scope_name`, normalization, fingerprinting, scope resolution, and public registered-scope methods.
- Modularization pass: extracted scoped/default model listing into `src/rotator_library/client/model_discovery.py` as `ModelDiscoveryService`. It owns model cache keys, scoped model filters, OpenAI-compatible `/models` calls, and `get_available_models`/`get_all_available_models` internals.
- Modularization pass: extracted usage-manager lifecycle and construction into `src/rotator_library/client/usage_managers.py` as `UsageManagerRegistry`. It owns global manager creation, scoped manager creation, initialization, credential metadata, concurrency settings, and provider reset-config application.
- Modularization pass: extracted completion/embedding `RequestContext` creation into `src/rotator_library/client/request_builder.py` as `RequestContextBuilder`. It owns scoped kwarg popping, scope resolution, model/provider validation, model id resolution for completions, transaction logger creation, session id inference, and context assembly.
- Modularization pass: extracted quota/stats/reload operations into `src/rotator_library/client/quota.py` as `QuotaService`. It owns classifier-aware quota stats, usage reload, and force-refresh quota behavior.
- `src/rotator_library/client/rotating_client.py` is now a slimmer facade around these services plus lifecycle, token counting, provider instance lookup, LiteLLM log sanitation, and Anthropic compatibility passthroughs. Line count after extraction: `rotating_client.py` 824 lines; new modules: `scopes.py` 351, `model_discovery.py` 273, `usage_managers.py` 233, `request_builder.py` 143, `quota.py` 200.
- Verification run after each extraction: `python -m compileall src tests/test_classifier_scoped_routing.py && python -m pytest tests/test_classifier_scoped_routing.py -q` passes with 15 tests.
- Review note follow-up: removed the constructor-order smell where `UsageManagerRegistry` received callbacks that delegated through `_scope_manager` before `_scope_manager` existed. `_scope_usage_key()` and `_scope_usage_file()` are now pure `RotatingClient` helpers using `ScopeManager.safe_scope_name()` and `_usage_base_path`, so the registry does not depend on `_scope_manager` for scoped manager path/key creation.
- Added regression test `test_usage_registry_scope_paths_do_not_depend_on_scope_manager_construction_order`, which temporarily sets `client._scope_manager = None` and verifies `UsageManagerRegistry.ensure_scoped_usage_manager()` still creates the expected scoped manager key and usage file path.
- Verification run after constructor-order fix: `python -m compileall src tests/test_classifier_scoped_routing.py && python -m pytest tests/test_classifier_scoped_routing.py -q` passes with 16 tests.

## Next Steps

1. Add proxy-facing request/API shape if the proxy needs to expose classifier management later.
2. Design scoped OAuth refresh with host-provided resolver/update callbacks if needed.
3. If more facade slimming is desired, next candidates are LiteLLM logging sanitation and provider-instance/cache helper extraction.
