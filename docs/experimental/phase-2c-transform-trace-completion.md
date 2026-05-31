# Phase 2c: Transform Trace Completion And Redaction Hardening

## Goal

Close the Phase 2/2b third-pass tracing findings: Anthropic compatibility coverage, native stream-event adapter tracing, Responses SSE formatting traces, transform-failure traces, final-response ordering, and camelCase secret redaction.

## Scope

- Add transform-pass traces around live Anthropic Messages compatibility request/response conversion.
- Add Anthropic streaming conversion traces and explicit upstream close on disconnect or abnormal exit.
- Ensure final client response traces represent the final post-normalization returned payload.
- Run and trace native stream-event adapter chains.
- Trace Responses final SSE formatting boundaries.
- Harden redaction for camelCase secret-bearing keys.
- Emit transform-failure traces for built-in provider transforms and Responses conversion errors.

## Non-Goals

- Do not redesign transaction log storage.
- Do not remove legacy request/response JSON logs.
- Do not enable native streaming for priority providers.
- Do not change public API response shapes.

## Implementation Plan

1. Anthropic non-stream traces.
   - Trace `anthropic_raw_request`, `anthropic_to_openai_request`, `anthropic_openai_response`, `openai_to_anthropic_response`, and `anthropic_final_response`.

2. Anthropic streaming traces and close safety.
   - Trace source OpenAI chunks and emitted Anthropic SSE frames.
   - Close upstream stream via `aclose()` / `close()` on disconnect or abnormal exit.
   - Trace stream transform errors before emitting the Anthropic error frames.

3. Provider transform failure traces.
   - Wrap built-in provider transforms and emit `transform_log_error` before re-raising.
   - Preserve provider-hook behavior but keep sanitized metadata.

4. Native stream-event adapter chain.
   - Run adapter chain for parsed native stream events.
   - Trace `after_stream_event_adapter_chain` before formatting.

5. Responses SSE formatting trace.
   - Trace each formatted `ResponsesStreamEvent` frame and terminal frame emitted by `stream_response()`.

6. Final response trace ordering.
   - Verify or adjust `final_client_response` ordering so it occurs after usage normalization/cost accounting.

7. Redaction hardening.
   - Normalize camelCase keys such as `apiKey`, `accessToken`, `refreshToken`, `clientSecret`, and `idToken`.

8. Responses transform-failure traces.
   - Emit standardized transform error traces for Responses parse/bridge/storage conversion errors without changing raised errors.

## Tests

- Anthropic non-stream and stream trace tests.
- Anthropic disconnect upstream close test.
- Built-in provider transform failure trace test.
- Native stream-event adapter mutation/trace test.
- Responses SSE formatting trace test.
- CamelCase redaction tests.
- Final response trace-ordering tests.

## Acceptance Criteria

- Anthropic compatibility has transform-pass coverage for request conversion, response conversion, and stream conversion.
- Anthropic stream disconnect closes upstream when possible.
- Built-in provider transform exceptions emit transform-failure traces.
- Native stream-event adapter chains are executed and traced.
- Responses stream SSE formatting boundaries are traced.
- Final client response traces reflect post-normalization final payloads.
- CamelCase secret-bearing keys are redacted.
- Focused tests pass and both `explore` and `explore-heavy` reviewers report no blockers/highs/mediums.
