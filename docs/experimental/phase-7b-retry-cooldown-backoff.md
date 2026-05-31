# Phase 7b: Retry/Cooldown Backoff And Failure History

## Goal

Correct the Phase 7 audit findings. Phase 7 made provider cooldown activation real and centralized retry/failover decisions, but it still lacks provider/model cooldown scopes, scoped backoff, and structured in-memory failure history.

## Non-Goals

- Do not replace `UsageManager`, credential cooldowns, usage windows, `classify_error()`, or retry-after parsing.
- Do not introduce SQLite or a new persistence database.
- Do not make native streaming safe; unsupported native streaming remains fail-closed.
- Do not change Phase 6b fallback hard-stop behavior.
- Do not commit user-facing reports.

## Current State

- `CooldownManager` tracks provider cooldowns only.
- `RequestExecutor._wait_for_cooldown()` waits on provider cooldown only.
- `retry_policy.decide_provider_cooldown()` returns provider-level decisions without model scope or failure-history context.
- `MODEL_CAPACITY_EXHAUSTED` is noticed in logs but still behaves like a generic provider server error.
- There is no structured provider/model failure-history ring for future observability and bounded backoff decisions.

## Implementation Plan

1. Extend `CooldownManager` with scoped cooldown methods.
   - Preserve `start_cooldown()`, `is_cooling_down()`, and `get_remaining_cooldown()`.
   - Add provider/model scoped methods and extend-only semantics per scope key.
   - Provider cooldown blocks all models; model cooldown blocks only that provider/model.

2. Update executor cooldown waiting.
   - Pass model into cooldown wait paths.
   - Wait for the max of provider and model cooldown remaining.
   - Trace waits without credentials or raw provider text.

3. Add model-capacity detection.
   - Detect `MODEL_CAPACITY_EXHAUSTED`, model capacity, and capacity-exhausted signals from exceptions and dict payloads.
   - Keep `error_type="server_error"` for compatibility, but choose model cooldown scope.

4. Extend cooldown decisions.
   - Add `scope`, `model`, and optional backoff metadata to `ProviderCooldownDecision`.
   - Large retry-after rate limits stay provider-scoped.
   - Model-capacity failures become model-scoped.
   - Quota cooldown remains disabled by default.

5. Add in-memory failure history.
   - Bounded ring with timestamp, provider, model, error type, scope, duration, and reason.
   - No disk persistence.
   - Executor records successful cooldown starts for future observability and backoff.

6. Add bounded repeated-transient backoff.
   - Track repeated transient `server_error`/`api_connection` failures within a configurable window.
   - Escalate cooldown duration conservatively up to a max.
   - Do not backoff hard-stop categories.

7. Wire scoped cooldown start in `RequestExecutor`.
   - Use scoped cooldown methods when available and fall back to provider-only fakes in tests.
   - Keep existing trace pass names with added scope/model metadata.

8. Update streaming error decisions.
   - Keep `decide_streaming_error_action()` side-effect-free.
   - Include cooldown scope/model in the decision.
   - Visible output still suppresses provider/model cooldown.

## Tests

- `tests/test_cooldown_activation.py`
- `tests/test_retry_policy.py`
- `tests/test_streaming_error_handler.py`
- Executor-focused cooldown/trace tests.
- Phase 6b routing regression subset.

## Acceptance Criteria

- Provider cooldown behavior remains backwards-compatible and extend-only.
- Model cooldowns do not block unrelated models on the same provider.
- Provider cooldowns still block all models for that provider.
- Model-capacity errors produce model-scoped cooldown/backoff.
- Repeated transient failures can produce bounded scoped backoff from in-memory history.
- Cooldown starts/waits are traceable and sanitized.
- Streaming cooldown decisions expose scope and respect visible-output blocking.
- Focused tests and dual-agent review pass.
