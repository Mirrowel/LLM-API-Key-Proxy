# Phase 8 Plan: Streaming Library Upgrade

## Goal

Turn streaming into a reusable, transport-aware library layer instead of scattered executor/provider-specific logic. Phase 8 should preserve the current HTTP SSE behavior, add measured stream lifecycle events, consolidate retry/error handling where safe, expose a WebSocket-ready transport seam, and prepare native provider streaming to share the same policies. It must keep the Phase 6/7 invariants: no fallback after visible output, provider cooldown only before visible output or error-only chunks, and no replacement of `UsageManager`, `SessionTracker`, or retry-after parsing.

## Non-Goals

- Do not replace `UsageManager`, `SessionTracker`, `SelectionEngine`, or retry-after parsing.
- Do not implement a live WebSocket route yet unless it is strictly a disabled/placeholder seam.
- Do not migrate every provider to native streaming in one pass.
- Do not alter non-streaming request behavior.
- Do not add SQLite or any database.
- Do not implement full cost/quota overhaul; Phase 9 owns usage/quota/cost.
- Do not change client-visible SSE formats except to fix bugs or add compatible error/end events.
- Do not fallback to another target after visible model output.

## Current Code Context

- `RequestExecutor._execute_streaming()` owns the active streaming credential retry loop and still has four duplicated exception branches.
- `RequestExecutor._execute_streaming_with_fallback()` wraps target fallback and already tracks visible output.
- Phase 7 added `_stream_chunk_is_visible_output()` and `_can_start_stream_provider_cooldown()` safety helpers.
- `StreamingHandler.wrap_stream()` converts LiteLLM chunks to chat-completions SSE strings, records usage after completion, records response anchors, and handles client disconnect.
- `stream_retry_policy.py` owns reasoning-only retry safety for post-output stream failures.
- `ResponsesService.stream_response()` converts chat stream chunks into Responses SSE events, stores final streamed responses, and has a `ResponsesWebSocketFormatter` placeholder.
- `NativeProviderExecutor.stream()` can stream mocked native provider JSON-line chunks and trace raw/parsed/formatted events, but live executor streaming does not route through native execution modes yet.
- Transform trace already has stream pass names from Phase 2 and later phases.
- There is no central stream metrics object for TTFB/TTFT/stalls/cancellation.

## Files To Add

- `src/rotator_library/streaming/__init__.py`
- `src/rotator_library/streaming/events.py`
- `src/rotator_library/streaming/metrics.py`
- `src/rotator_library/streaming/transport.py`
- `src/rotator_library/streaming/policy.py`
- `src/rotator_library/streaming/errors.py`
- `tests/test_stream_events.py`
- `tests/test_stream_metrics.py`
- `tests/test_stream_transport.py`
- `tests/test_stream_policy.py`
- `tests/test_streaming_error_handler.py`
- `tests/test_request_executor_stream_metrics.py`
- `tests/test_native_streaming_transport_seam.py`

## Files Likely To Touch

- `src/rotator_library/client/executor.py`
- `src/rotator_library/client/streaming.py`
- `src/rotator_library/client/stream_retry_policy.py`
- `src/rotator_library/native_provider/executor.py`
- `src/rotator_library/native_provider/streaming.py`
- `src/rotator_library/responses/streaming.py`
- `src/rotator_library/responses/service.py`
- `src/proxy_app/main.py` only if route headers/cancellation handling need a small SSE-compatible fix.

## Streaming Event Model

Add `StreamEvent` dataclass:

- `event_type`: `started`, `raw_chunk`, `parsed_chunk`, `delta`, `reasoning_delta`, `tool_delta`, `usage`, `error`, `completed`, `cancelled`, `heartbeat`, `metadata`.
- `protocol`: `openai_chat`, `responses`, `anthropic_messages`, `gemini`, `native`, `litellm_fallback`, or provider-specific.
- `transport`: `sse`, `websocket`, `jsonl`, or future string.
- `data`: JSON-safe event payload.
- `raw`: optional raw chunk for trace only, sanitized/serialized.
- `metadata`: JSON-safe metadata.
- `visible_output`: bool.
- `timestamp_utc`.

Add helpers:

- `stream_event_from_sse_chunk()`
- `stream_event_to_sse()`
- `stream_event_to_websocket_message()` placeholder/seam.

Keep formatters small and override-friendly.

## Transport Abstraction

Add `StreamTransportFormatter` base/protocol:

- `format_event(event)`
- `format_error(error_event)`
- `format_done()`
- `is_terminal_event(event)`

Add `SSEStreamFormatter` for existing HTTP SSE output.

Add `WebSocketStreamFormatter` placeholder that formats JSON messages but is not wired to a route yet.

Add `JSONLineStreamFormatter` for native provider/internal stream tests if useful.

Responses API can keep `ResponsesSSEFormatter` but should share the transport interface or adapt to it.

## Metrics And Lifecycle

Add `StreamMetrics`:

- `started_at`
- `first_byte_at`
- `first_visible_output_at`
- `last_chunk_at`
- `completed_at`
- `chunk_count`
- `visible_chunk_count`
- `error_count`
- `cancelled`
- properties: `ttfb_seconds`, `ttft_seconds`, `duration_seconds`, `idle_seconds`.

Add `StreamMonitor`:

- records raw chunk, formatted chunk, visible output, errors, completion, cancellation.
- can detect stall if `time_since_last_chunk > stall_timeout`.

Env knobs:

- `STREAM_TTFB_TIMEOUT_SECONDS` optional/disabled by default.
- `STREAM_STALL_TIMEOUT_SECONDS` optional/disabled by default.
- `STREAM_HEARTBEAT_SECONDS` optional/disabled by default.

Phase 8 should add metrics and trace events first. Enforced timeouts can be opt-in to avoid behavior surprises.

## Stream Policy

- Move/re-export `can_retry_stream_after_error()` into the new streaming policy package, preserving current behavior.
- Keep `stream_retry_policy.py` as a compatibility wrapper if imports exist.
- Add visible-output detection policy:
  - Chat-completions content/tool/function deltas are visible.
  - Reasoning-only deltas are not visible for fallback only if the existing env allows reasoning-only retry.
  - Error chunks and `[DONE]` are not visible.
  - Responses `response.output_text.delta` is visible.
  - Responses `response.failed` is not visible by itself.
- Add tests for malformed chunks failing closed.

## Streaming Error Handling

Add `StreamingErrorDecision` dataclass:

- `classified`
- `action`: `retry_same`, `rotate`, `fail`, `fallback_allowed`, `fallback_blocked_after_output`
- `start_provider_cooldown`: bool
- `provider_cooldown_duration`
- `reason`

Add helper that consumes:

- exception
- provider
- last_streamed_chunk
- attempt
- max_retries
- deadline
- allow_reasoning_only_retry
- retry/cooldown env settings

It should use existing `classify_error()`, `should_retry_same_key()`, `should_rotate_on_error()`, and Phase 7 retry policy.

It should not sleep or mutate credential state; executor still owns those side effects.

Refactor `_execute_streaming()` gradually:

- First introduce helper and tests.
- Then replace duplicated branch decision logic where safe.
- Preserve exact output semantics and trace ordering.

If full deduplication is too risky, use helper for cooldown/visibility/fallback decisions but keep branch structure.

## Fallback And Cooldown Invariants

- Fallback to next target is allowed only before visible output.
- Provider cooldown can start only before visible output or after error-only chunks.
- Same-credential retry after reasoning-only chunks remains controlled by `STREAM_RETRY_ON_REASONING_ONLY` behavior from current code.
- If a stream emits visible output then errors:
  - no cross-target fallback.
  - return/raise the current upstream stream failure behavior.
  - emit `routing_stream_fallback_blocked_after_output` when in a fallback group.
- Preserve Phase 7 tests.

## Native Provider Streaming

- Add a streaming mode seam to `NativeProviderExecutor` that can return `StreamEvent`s or formatted SSE through the common formatter.
- Live `RequestExecutor` does not need to route native streaming fully unless safe, but the mode should be testable:
  - `execution=native` streaming target can call native executor stream when provider declares native streaming support.
  - if not supported, fail with `unsupported_operation` before output so fallback can choose next target.
- Add provider interface optional method only if needed:
  - `supports_native_streaming(model, operation)`
  - default false/no-op.
- This should prepare Claude Code/Codex/Copilot/Antigravity skeletons without claiming live support.

## Responses Streaming Integration

- Keep current `/v1/responses` SSE behavior.
- Share metrics/trace helpers where possible.
- Preserve stored final streamed response behavior.
- Add tests:
  - Responses stream emits visible-output metrics.
  - failed stream records error metrics.
  - WebSocket formatter seam can format equivalent event payloads but no route is exposed.
- Do not change stored response shape unless necessary.

## Transaction Trace Requirements

Add or standardize stream trace passes:

- `stream_started`
- `stream_first_byte`
- `stream_first_visible_output`
- `stream_stall_detected`
- `stream_heartbeat_sent`
- `stream_completed`
- `stream_cancelled`
- `stream_error_decision`
- `stream_metrics_final`

Existing pass names stay:

- `raw_stream_chunk`
- `parsed_stream_chunk`
- `assembled_stream_response`
- provider/native/responses stream pass names.

Trace metrics must not include raw credentials or auth headers.

Raw chunks should continue through existing redaction/serialization.

## Client Disconnect And Cancellation

- `StreamingHandler.wrap_stream()` already checks `request.is_disconnected()`.
- Add metrics event for cancellation/disconnect.
- Add trace `stream_cancelled`.
- Do not mark success on cancellation.
- Preserve current behavior for partial streams.

## Heartbeats And Stall Detection

Implement monitor primitives and tests first.

Optional runtime heartbeat support:

- if `STREAM_HEARTBEAT_SECONDS > 0`, emit SSE comment heartbeat `: keep-alive\n\n` or compatible event only when no provider chunk has arrived.
- default disabled to avoid client behavior changes.

Optional stall detection:

- if `STREAM_STALL_TIMEOUT_SECONDS > 0`, classify as transient stream failure before visible output or fail after visible output.
- default disabled.

Tests can use fake clocks.

## Testing Plan

Stream event tests:

- SSE chunk to event parsing.
- Responses event visibility.
- malformed chunk fails closed.
- event-to-SSE formatting.
- WebSocket formatter seam output shape.

Metrics tests:

- TTFB and TTFT calculations.
- chunk counts.
- cancellation.
- stall detection with fake clock.

Policy tests:

- current reasoning-only retry behavior preserved.
- visible output detection for text/tool deltas.
- error and done chunks not visible.

Error handler tests:

- large retry-after before output starts cooldown decision.
- visible output blocks fallback/cooldown.
- transient errors choose same-key retry before max retry.
- permanent errors fail.

Executor tests:

- streaming metrics trace passes emitted.
- cancellation trace emitted.
- pre-output fallback still works.
- post-output fallback remains blocked.
- provider cooldown still starts before output.

Native stream tests:

- native streaming emits common stream events/trace.
- unsupported native streaming fails before output and can fallback.

Responses streaming tests:

- existing route behavior unchanged.
- metrics final trace emitted.

Regression tests:

- Phase 1 protocol tests.
- Phase 2 transform logging tests.
- Phase 3 adapter/cache tests.
- Phase 4 Responses tests.
- Phase 5 provider/native tests.
- Phase 6 routing tests.
- Phase 7 retry/cooldown tests.
- `test_session_tracking.py`.
- `test_selection_engine.py`.

## Commit Checkpoints

1. Add streaming event/transport/metrics primitives with tests.
2. Move/re-export stream retry/visibility policy with tests.
3. Add streaming error decision helper with tests.
4. Integrate stream metrics and trace passes into `StreamingHandler` and/or executor with tests.
5. Refactor streaming error branches only as far as tests make safe.
6. Add native streaming seam/support detection tests.
7. Add Responses streaming metrics/transport seam tests.
8. Run focused and regression tests.
9. Review with `explore` and `explore-heavy`; fix findings; write uncommitted Phase 8 report.

## Risks And Mitigations

- Stream behavior is client-visible. Mitigation: keep SSE output format unchanged by default and add primitives before enforcement.
- Timeout/stall handling can break long reasoning streams. Mitigation: default stall/TTFB enforcement disabled; only metrics on by default.
- Refactoring executor streaming can regress retry semantics. Mitigation: keep branch structure unless shared helper is proven by tests.
- WebSocket support can be over-promised. Mitigation: formatter seam only, no route unless explicitly implemented later.
- Visible-output detection can be too permissive. Mitigation: fail closed on malformed/ambiguous chunks and preserve current tests.
- Metrics can leak data if raw chunks are logged. Mitigation: trace summaries and use existing redaction/serialization paths.
