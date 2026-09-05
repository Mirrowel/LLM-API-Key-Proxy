# Phase 6c: Routing/Fallback Correctness And Stale Test Repair

## Goal

Close all Phase 6/6b third-pass routing findings while preserving the fallback group architecture and the hard-stop safety from Phase 6b.

## Scope

- Replace stale `tests/test_fallback_groups.py` with tests for the current routing package.
- Implement requested-model-in-group promotion.
- Align streaming execution-mode behavior with non-streaming behavior for `@custom`, `@native`, and `@litellm_fallback`.
- Ensure explicit native configuration errors are hard-stop configuration errors, not retryable unsupported-operation errors.
- Expand structured route-error alias normalization.
- Adjust fallback target session namespace per target.
- Add focused tests for every blocker/high/medium.

## Non-Goals

- Do not resurrect the old `FallbackGroupManager`.
- Do not reintroduce `RotatingClient(model_fallback_groups=...)`.
- Do not implement routing for embeddings unless it is trivial and needed by tests.
- Do not weaken hard-stop policy.
- Do not commit user-facing reports.

## Implementation Plan

1. Replace stale fallback tests.
   - Rewrite `tests/test_fallback_groups.py` around `routing.config`, `FallbackResolver`, `FallbackAttemptRunner`, and executor helpers.
   - Cover env/JSON group parsing, target promotion, execution suffix parsing, and fallback exhaustion behavior.

2. Requested-model promotion.
   - Promote a requested provider/model target to the first attempt when it appears in a fallback group.
   - Preserve the relative order of all other targets.
   - Apply this both for explicit `MODEL_ROUTE_* = group:name` routes and provider-prefixed requests that appear in any group.

3. Streaming execution-mode parity.
   - Share execution-mode selection logic between streaming and non-streaming paths.
   - `@litellm_fallback` must force LiteLLM.
   - `@custom` must require custom provider logic.
   - `@native` must fail closed if native streaming is unsupported.
   - `auto` remains custom first, then native streaming if explicitly supported, then LiteLLM.

4. Explicit native configuration errors.
   - Missing native declaration, unsupported operation, and missing endpoint/header helpers should raise `RoutingExecutionError(error_type="configuration_error")` for explicit native execution failures.
   - Hard-stop policy then blocks fallback.

5. Structured route-error aliases.
   - Add common status-less aliases including `invalid_api_key`, `unauthorized`, `invalid_argument`, `rate_limited`, `too_many_requests`, `resource_exhausted`, `unavailable`, `deadline_exceeded`, and context-window variants.

6. Session namespace adjustment.
   - `clone_context_for_target()` should rewrite standard session-tracking namespaces to the target provider/model instead of preserving the first target namespace.
   - Unknown/custom namespace shapes remain unchanged.

## Acceptance Criteria

- Broad test collection no longer fails on stale fallback imports.
- Requested provider-prefixed models inside a group are tried first.
- Streaming and non-streaming execution-mode selection are consistent.
- Explicit native config errors are hard-stop configuration errors.
- Common structured aliases normalize correctly.
- Fallback target contexts do not reuse the first target's provider/model session namespace.
- Focused tests pass and both reviewers report no blockers/highs/mediums.
