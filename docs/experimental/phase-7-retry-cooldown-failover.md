# Phase 7 Plan: Retry/Cooldown/Failover Cleanup

## Goal

Harden the retry and failover layer now that ordered routing exists. Phase 7 makes provider-level cooldown functional, reduces fallback policy duplication, preserves the current retry-after parser, and improves final error visibility across target chains without replacing `UsageManager`, `SessionTracker`, credential rotation, or the existing retry classifier.

## Non-Goals

- Do not replace `UsageManager`, `SelectionEngine`, `SessionTracker`, or the retry-after parser.
- Do not rewrite the entire executor.
- Do not implement the Phase 8 streaming transport/library overhaul.
- Do not implement native streaming dispatch unless a small safe seam is necessary.
- Do not add SQLite or any database.
- Do not implement full JSON config merging; Phase 10 owns config polish.
- Do not make fallback happen after visible streamed output.
- Do not silently fallback to LiteLLM unless the route target explicitly selects `litellm_fallback`.

## Current Code Context

- `CooldownManager` exists and `_wait_for_cooldown()` is called before target attempts, but `start_cooldown()` is not called by the executor, so provider cooldown is effectively dormant.
- Existing retry-after parsing lives in `error_handler.py` and is strong; keep it.
- Existing `RequestExecutor._handle_error_with_context()` handles non-streaming per-credential retries and rotations.
- Streaming has duplicated retry/rotation handling in several exception branches.
- Phase 6 added `FallbackPolicy`, `FallbackAttemptRunner`, `clone_context_for_target()`, and executor inline fallback loops.
- Phase 6 final review found no blockers but noted two useful cleanup targets: executor inline loops do not pass `FallbackGroup` overrides to `FallbackPolicy`, and `FallbackAttemptRunner` is tested but not used by live executor.
- Phase 6 response-based fallback maps exhausted structured errors to retryable categories, but final target-chain failures do not yet summarize all target failures.
- Provider-level cooldown should apply to provider-wide or IP/global throttling, not every small per-key retry-after.

## Files To Add

- `src/rotator_library/retry_policy.py`
- `tests/test_retry_policy.py`
- `tests/test_cooldown_activation.py`
- `tests/test_request_executor_fallback_error_summary.py`
- Possibly `tests/test_request_executor_group_policy.py`

## Files Likely To Touch

- `src/rotator_library/client/executor.py`
- `src/rotator_library/cooldown_manager.py`
- `src/rotator_library/routing/types.py`
- `src/rotator_library/routing/executor.py`
- `src/rotator_library/core/types.py`
- `src/rotator_library/error_handler.py` only if docstring/type comments need alignment; do not weaken parser behavior.
- Existing Phase 6 tests if group policy wiring changes expected trace order.

## Retry Policy Foundation

Add a small `retry_policy.py` module to centralize decisions that are currently duplicated:

- `classify_route_error(error, provider)`
- `should_provider_cooldown(classified_error, *, small_cooldown_threshold, provider_cooldown_threshold)`
- `provider_cooldown_duration(classified_error, default_duration)`
- `should_retry_same_credential(classified_error, small_cooldown_threshold)`
- `should_rotate_credential(classified_error)`
- `is_target_failover_eligible(error_type, group=None, stream=False, emitted_output=False)`

This module should call existing `classify_error()`, `should_retry_same_key()`, and `should_rotate_on_error()` rather than reimplementing them.

It should document why small retry-after values stay on the same credential while large/provider-level values can activate provider cooldown.

It should not own sleeping or mutation; it only returns decisions.

## Cooldown Activation

Add provider cooldown activation in non-streaming error handling:

- If `classified.retry_after` exists and is above `SMALL_COOLDOWN_RETRY_THRESHOLD`, start provider cooldown for that duration when the error is provider-wide enough.
- Candidate categories: `rate_limit`, `server_error`, and possibly provider-specific `quota_exceeded` only when retry-after suggests global reset rather than per-key quota.
- Be conservative: avoid cooling down an entire provider for every credential quota if the error looks per-credential.

Add env knobs:

- `PROVIDER_COOLDOWN_MIN_SECONDS` default perhaps same as small-cooldown threshold or a modest value.
- `PROVIDER_COOLDOWN_DEFAULT_SECONDS` for retryable provider-wide errors without retry-after.
- `PROVIDER_COOLDOWN_ON_QUOTA` default false unless evidence indicates provider-wide quota.

Existing `_wait_for_cooldown()` stays the wait path.

`CooldownManager.start_cooldown()` should extend only when the new expiry is later than the current expiry, not shorten an active cooldown.

Add trace pass:

- `provider_cooldown_started`
- metadata: provider, duration, error_type, retry_after_present, reason.

Logging failure to start cooldown should never fail the request.

## Fallback Runner And Live Loop Cleanup

Either wire `FallbackAttemptRunner` into live non-streaming and streaming wrappers or keep inline wrappers but pass a resolved group object.

Minimal preferred approach for Phase 7:

- Add `routing_group` optional field to `RequestContext` or attach group policy data to context.
- `RequestContextBuilder` should carry the `FallbackGroup` resolved from config when routing is active.
- Executor fallback wrappers pass `group=context.routing_group` to `FallbackPolicy.should_fallback()`.
- Keep inline wrappers if that avoids contorting trace behavior.

If refactoring to `FallbackAttemptRunner` stays small, do it; otherwise leave runner as tested support and make inline loops group-aware.

Add tests proving group overrides are honored in live executor wrappers for non-streaming and streaming.

## Cross-Target Error Summary

Add a target failure accumulator for fallback groups:

- target name
- provider
- model
- execution mode
- error type
- message summary
- emitted_output for streams

If all fallback targets fail, final client error should include `fallback_targets` details under `error.details`, with no credentials.

Preserve existing per-target credential error summaries from `RequestErrorAccumulator`.

Do not expose credential secrets or raw request data.

Add trace pass:

- `routing_fallback_exhausted`
- include accumulated target failures.

Tests should cover:

- all targets fail and final response contains both target failures.
- exception failure on first target and structured proxy error on second target are both summarized.
- no credentials in summary.

## Provider Cooldown And Fallback Interaction

- If a target enters provider cooldown and still fails/exhausts, fallback to next target may proceed if policy allows.
- If a provider is already cooling down, `_wait_for_cooldown()` should wait only if there is request budget; otherwise target should fail fast in a way fallback can interpret.
- Consider returning/raising a classified `provider_cooldown_budget_exceeded` or mapping to `rate_limit` for target fallback.
- Keep this minimal: do not replace capacity waiting or usage limits.

## Streaming Retry/Fallback Cleanup

- Keep Phase 6 invariant: fallback only before visible output.
- Deduplicate obvious streaming error branches only if the change is small and tests cover it.
- Add cooldown activation in streaming error handling where retry-after is available and no visible output has been emitted.
- Do not implement native streaming execution-mode dispatch in Phase 7 unless reviewers insist; Phase 8 owns streaming transport design.
- Add provider cooldown trace for streaming provider-wide throttles.

## Error-Classification Alignment

Add tests tying actual `classify_error()` output to fallback policy decisions for:

- `rate_limit`
- `quota_exceeded`
- `server_error`
- `api_connection`
- `authentication`
- `forbidden`
- `invalid_request`
- `context_window_exceeded`
- `credential_reauth_needed`
- `pre_request_callback_error`
- `cancelled`
- `unsupported_operation`

Preserve conservative `unknown` behavior unless a test shows a strong reason to fallback.

## Stale Fallback Test Cleanup

There is a stale ignored test file `tests/test_fallback_groups.py` that imports a non-existent older module.

- Do not delete it unless it is tracked and relevant to current runs.
- If it is tracked or causing collection failures in broader test runs, replace/archive it with current routing tests in a focused commit.
- If ignored/untracked, leave it alone unless it blocks CI.

## Transform Trace Requirements

Keep existing Phase 6 traces:

- `routing_decision`
- `routing_target_attempt_started`
- `routing_target_attempt_failed`
- `routing_target_attempt_succeeded`
- `routing_fallback_selected`
- `routing_fallback_exhausted`
- `routing_litellm_fallback`
- `routing_native_execution_selected`
- stream equivalents.

Add:

- `provider_cooldown_started`
- `provider_cooldown_skipped`
- `routing_group_policy_applied` if group overrides influence a decision.

Trace metadata must not include credentials or secrets.

## Testing Plan

Retry policy tests:

- existing classifier output maps to expected retry/rotate/fallback/cooldown decisions.
- small retry-after chooses same-credential retry, not provider cooldown.
- large retry-after can start provider cooldown.
- `unknown` remains conservative for target fallback.

Cooldown tests:

- `CooldownManager.start_cooldown()` extends but does not shorten cooldown.
- non-streaming large retry-after starts provider cooldown.
- small retry-after does not start provider cooldown.
- cooldown trace emitted.
- `_wait_for_cooldown()` respects deadline budget.

Fallback group policy tests:

- live non-streaming fallback honors group-specific `failover_on`.
- live non-streaming fallback honors group-specific `stop_on`.
- live streaming fallback honors group policy before output.

Error summary tests:

- all target failures are summarized in final error details.
- target summary excludes credentials.

Regression tests:

- Phase 1 protocol tests.
- Phase 2 transform logging tests.
- Phase 3 adapter/field-cache tests.
- Phase 4 Responses tests.
- Phase 5 provider/native tests.
- Phase 6 routing tests.
- `test_session_tracking.py`.
- `test_selection_engine.py`.

## Commit Checkpoints

1. Add retry policy helper module and classifier-alignment tests.
2. Harden `CooldownManager` extension semantics and tests.
3. Wire provider cooldown activation into non-streaming executor with trace tests.
4. Add group-policy awareness to live fallback wrappers with tests.
5. Add cross-target error summaries with tests.
6. Add streaming cooldown/group-policy cleanup tests if not covered earlier.
7. Run focused and regression tests.
8. Review with `explore` and `explore-heavy`; fix findings; write uncommitted Phase 7 report.

## Risks And Mitigations

- Provider cooldown could over-throttle healthy credentials. Mitigation: only activate on large retry-after/provider-wide categories, keep quota cooldown disabled by default unless configured.
- Refactoring fallback loops could break trace order. Mitigation: prefer small group-aware changes unless runner integration stays simple.
- Error summaries could leak credentials. Mitigation: summarize target/provider/model/error only; never include credentials.
- Streaming fallback could corrupt output if changed carelessly. Mitigation: preserve Phase 6 visible-output gate and add tests.
- Retry policy could drift from existing parser. Mitigation: call existing parser/classifier helpers instead of rewriting them.
