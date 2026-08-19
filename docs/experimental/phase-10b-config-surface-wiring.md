# Phase 10b: Config Surface Wiring And Runtime Settings Completion

## Goal

Correct the Phase 10 validation finding that the config layer exists but does not wire enough runtime surfaces introduced in corrective phases. The proxy remains `.env`-first, optional JSON config is structured and secret-free, and environment variables override JSON values.

## Non-Goals

- Do not create a full application settings framework.
- Do not move secrets into JSON config.
- Do not replace provider credential discovery.
- Do not add security or multi-user config.
- Do not introduce SQLite or durable DB config.
- Do not commit user-facing reports.

## Implementation Plan

1. Add retry/cooldown runtime settings.
   - Parse provider cooldown, provider backoff, and failure-history settings from JSON and env.
   - Preserve existing env var names and defaults.
   - Env overrides JSON.

2. Wire retry/cooldown settings into retry policy.
   - Keep imports lazy to avoid startup cycles.
   - Preserve existing monkeypatch/env test behavior.

3. Add Responses store settings config.
   - Parse `RESPONSES_STORE_TTL_SECONDS`, `RESPONSES_STORE_MAX_ITEMS`, `RESPONSES_STORE_FAILED`, and `RESPONSES_STORE_IN_PROGRESS`.
   - Support JSON under `responses.store`.
   - Preserve default behavior.

4. Wire Responses store settings into proxy startup.
   - Construct `ResponsesService(store_settings=get_responses_store_settings())` in the FastAPI app path.
   - Direct tests can still inject explicit settings.

5. Harden field-cache/provider config parser coverage.
   - Cover TTL, metadata mode hints, inject insert behavior, invalid rule errors, and secret rejection in new sections.

6. Update `.env.example`.
   - Document active streaming timeout/heartbeat settings.
   - Document provider/model cooldown and backoff knobs.
   - Document Responses store policy knobs.
   - Document provider-reported cost precedence and SSE cost comments.
   - Show safe structured-config section names.

## Acceptance Criteria

- Optional JSON config can configure retry/cooldown/backoff and Responses storage policies.
- Env vars override JSON for all new settings.
- Existing env-only behavior remains compatible.
- ResponsesService at proxy startup uses configured store settings.
- Streaming/pricing/routing/field-cache config regressions remain passing.
- Secret-like JSON keys are still rejected in all new sections.
- `.env.example` documents Phase 7b-10b runtime knobs accurately.
- Focused tests and dual-agent review pass.
