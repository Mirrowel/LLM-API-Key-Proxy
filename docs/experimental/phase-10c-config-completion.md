# Phase 10c: Config Honesty, Startup Secret Safety, And Validation Completion

## Goal

Close the third-pass Phase 10 findings by making accepted configuration surfaces either live-wired and validated or rejected clearly. The proxy remains `.env`-first; JSON config stays optional, structured, and secret-free.

## Scope

- Stop printing the full `PROXY_API_KEY` at startup.
- Wire safe `providers` JSON fields into provider runtime behavior.
- Validate provider protocol/adapter names against registries.
- Reject unsupported provider JSON keys instead of silently accepting non-live surfaces.
- Keep credentials, auth headers, API keys, OAuth tokens, and endpoint secrets out of JSON.
- Make invalid pricing env values non-fatal.
- Validate direct model-route target specs at config load/startup.
- Reject generic `credential` / `credentials` key names in JSON config.
- Preserve Phase 8c Responses streaming runtime settings behavior.

## Non-Goals

- Do not introduce SQLite, a database, or a new full application settings framework.
- Do not move secrets into JSON config.
- Do not instantiate arbitrary classes from JSON; only reference already registered protocol/adapter names.
- Do not replace provider declarations; JSON is an override layer for safe metadata.
- Do not rewrite every existing `os.getenv()` call.

## Implementation Plan

1. Startup secret masking.
   - Add a display helper in `proxy_app/main.py`.
   - Show that a key is set without printing the raw key.
   - Reuse the helper in both startup banners.

2. Provider config schema.
   - Add `ProviderRuntimeConfig` in `config.experimental`.
   - Support `protocol_name`, `adapter_names`, `adapter_config`, `native_streaming_supported`, `field_cache`, and `model_quota_groups`.
   - Reject unsupported provider keys.
   - Validate protocol and adapter names via existing registries with lazy imports.

3. Runtime provider wiring.
   - Use JSON protocol/adapters/adapter config from `ProviderInterface` methods.
   - Append configured field-cache rules after provider-declared rules.
   - Let JSON opt into native streaming only when protocol support exists.
   - Merge JSON model quota groups before env `QUOTA_GROUPS_*` overrides.

4. Pricing tolerance.
   - Ignore malformed env pricing components rather than raising during requests.
   - Let JSON pricing or LiteLLM fallback still apply when env pricing is invalid.

5. Routing validation.
   - Validate direct non-`group:` model routes by parsing them during `load_routing_config_from_env()`.
   - Keep unknown group validation unchanged.

6. Secret rejection expansion.
   - Add `credential` and `credentials` variants to secret-key detection.
   - Add tests for common camelCase and plural forms.

7. Documentation and tests.
   - Update `.env.example` safe-provider-config documentation.
   - Add focused tests for startup masking, provider config wiring, pricing tolerance, routing validation, and secret rejection.
   - Re-run config/routing/pricing/provider/Responses-streaming regression slices.

## Acceptance Criteria

- Startup never prints the full `PROXY_API_KEY`.
- The `providers` JSON section is live-wired for supported safe fields and rejects unsupported keys.
- Configured protocol/adapter names are registry-validated.
- JSON quota groups affect provider grouping, with env overrides final.
- Invalid pricing env values do not fail requests.
- Direct model-route targets fail clearly at config load if malformed.
- `credential` / `credentials` keys are rejected as secret-like.
- Focused tests pass and both reviewers report no blockers/highs/mediums.
