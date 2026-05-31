# Phase 6b: Routing/Fallback Correctness And Structured Error Safety

## Goal

Correct the Phase 6 audit findings. Phase 6 built ordered fallback groups and live executor wrappers, but fallback still needs stronger safety around hard-stop errors, structured error responses, streaming policy enforcement, and sanitized target summaries.

## Non-Goals

- Do not replace `UsageManager`, `SessionTracker`, retry-after parsing, or credential rotation behavior.
- Do not implement rich target-group selector syntax beyond ordered fallback chains.
- Do not add security or multi-user features.
- Do not make native streaming work; unsupported native streaming remains fail-closed.
- Do not include raw provider messages or credentials in cross-target summaries.
- Do not commit user-facing reports.

## Current State

- `FallbackPolicy.should_fallback()` uses group `failover_on` and `stop_on` directly, so a group can currently override auth/permanent errors into fallback eligibility.
- `_route_error_type_from_response()` only recognizes retry-like summaries and a few proxy error types; it does not robustly inspect structured status/code/details fields.
- `FallbackGroup.streaming_policy` exists but live streaming fallback does not use it.
- Streaming fallback correctly tracks visible output, but needs stronger tests for error/control frames and `never` streaming policy.
- Non-streaming fallback summaries are structural, but call sites still pass raw exception strings that should be avoided entirely.

## Implementation Plan

1. Add hard-stop route error categories.
   - Add a non-overridable hard-stop set for auth, forbidden, invalid request, context-window, credential reauth, pre-request callback, cancellation, and configuration errors.
   - Document that these are safety boundaries and group policies cannot opt into cross-target fallback for them.

2. Normalize routing policy vocabulary.
   - Add `normalize_route_error_type()` for aliases such as `auth`, `permission_denied`, `bad_request`, `validation`, `transient`, `network`, and `configuration`.
   - Use it in policy, routing runner, retry helper, and executor route classification.

3. Make hard stops win in `FallbackPolicy`.
   - Streaming visible output still blocks fallback first.
   - Hard-stop categories return false before group policy evaluation.
   - Group stop/failover sets are normalized before matching.

4. Validate routing group policy.
   - Reject configured `failover_on` entries that normalize to hard-stop categories.
   - Parse and validate `streaming_policy` from JSON routing config.
   - Preserve environment override precedence.

5. Enforce `FallbackGroup.streaming_policy` in live executor paths.
   - `never` prevents streaming fallback even before output.
   - `pre_output_only` remains the default.
   - Trace metadata should include the policy used.

6. Harden structured response classification.
   - Inspect `error.type`, `error.code`, `error.status`, `error.details.status_code`, and detail classification fields before summaries.
   - Hard-stop signals win over retryable text.
   - Keep retryable classification for 429/quota/rate-limit, 5xx/server, timeout, and connection signals.

7. Sanitize target-failure summaries.
   - Keep raw messages out of fallback details and traces.
   - Remove unnecessary raw string arguments from summary call sites.
   - Add tests proving secrets/provider text do not appear.

8. Improve stream fallback frame handling.
   - Verify error/control frames are non-visible output.
   - Test `event: error`, `type: error`, `response.failed`, `[DONE]`, comments, and heartbeat-like frames.
   - Ensure visible text/tool deltas still block fallback.

9. Add deterministic exhaustion metadata.
   - Include sanitized target summaries in routing exhaustion traces for stream and non-stream paths.
   - Avoid raw exception text in trace metadata.

10. Add execution-mode safety tests.
    - Explicit native configuration errors must be hard stops.
    - Unsupported operation behavior must be clear and tested.
    - LiteLLM fallback remains available for retryable errors.

## Tests

- `tests/test_fallback_policy.py`
- `tests/test_retry_policy.py`
- `tests/test_routing_config.py`
- `tests/test_fallback_resolver.py`
- `tests/test_routing_executor.py`
- `tests/test_streaming_fallback_policy.py`
- `tests/test_request_executor_native_routing.py`
- Phase 5b provider/native regression subset if executor routing helpers change.

## Acceptance Criteria

- Auth, forbidden, invalid request, context-window, credential reauth, pre-request callback, cancellation, and configuration errors never fallback across targets.
- Structured error responses classify deterministically with hard-stop signals taking precedence.
- Streaming fallback respects `FallbackGroup.streaming_policy`.
- Error/control stream frames before visible output do not accidentally lock routing.
- Cross-target summaries and traces remain sanitized.
- Focused tests and dual-agent review pass.
