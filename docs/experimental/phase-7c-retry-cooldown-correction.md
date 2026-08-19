# Phase 7c: Cooldown Budget, Transient Backoff, And Attempt History

## Goal

Close all Phase 7/7b third-pass retry/cooldown findings while preserving existing retry-after parsing and credential-rotation semantics.

## Scope

- Fail fast when an active provider/model cooldown exceeds the remaining request deadline budget.
- Prevent a single generic `server_error` / `api_connection` without `retry_after` from starting provider-wide cooldown.
- Record repeated transient failures into `FailureHistory` even when the cooldown threshold has not yet been reached.
- Clear or reduce failure history on successful provider/model calls.
- Populate structured `routing_attempt_history` for live non-streaming and streaming fallback attempts.
- Add tests for cooldown-over-budget, single transient no-cooldown, repeated transient cooldown, success reset, model-scoped cooldown isolation, and stream parity.

## Non-Goals

- Do not replace the retry-after parser.
- Do not replace `UsageManager` or `SessionTracker`.
- Do not introduce durable failure-history storage.
- Do not change small retry-after same-credential behavior.
- Do not weaken fallback hard-stop policy from Phase 6c.

## Implementation Plan

1. Cooldown-over-budget fail-fast.
   - Raise a retryable routing error instead of returning when cooldown remaining exceeds the request deadline budget.
   - Trace `cooldown_wait_exceeds_budget` without exposing credentials or provider payloads.

2. Generic transient cooldown threshold.
   - No-`retry_after` `server_error` and `api_connection` should start cooldown only after `FailureHistory.backoff_for()` crosses the configured threshold.
   - Large explicit `retry_after` behavior remains unchanged.

3. Record skipped transient failures.
   - Record sanitized transient entries when cooldown is skipped because the backoff threshold is not yet met.
   - Keep history bounded and in-memory only.

4. Success reset.
   - Add a clear/reset helper on `FailureHistory`.
   - Clear matching provider/model transient entries after successful non-streaming and completed streaming calls.

5. Structured attempt history.
   - Append sanitized attempt entries for fallback failures and successes into `RequestContext.routing_attempt_history`.
   - Include error type, target identity, execution mode, output visibility, status code when available, fallback decision, and timing where cheap.

6. Model cooldown isolation.
   - Add tests proving model-scoped cooldowns block only the matching model while provider-scoped cooldowns block all provider models.

## Acceptance Criteria

- Cooldown that exceeds request budget does not silently allow execution.
- A single generic transient without retry-after does not start provider-wide cooldown.
- Repeated transient failures can still start bounded cooldown/backoff.
- Successful calls clear matching failure-history entries.
- Live routing attempt history is populated and sanitized.
- Model-scoped cooldown blocks only its model plus provider-wide cooldown still blocks all provider models.
- Focused tests pass and both reviewers report no blockers/highs/mediums.
