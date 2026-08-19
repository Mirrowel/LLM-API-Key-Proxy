# Phase 8b: Streaming Hardening, Cancellation, Heartbeats, And Stall Policy

## Goal

Correct the Phase 8 audit findings. Phase 8 added stream events, metrics, formatters, and observability, but stream hardening still needs upstream cancellation, active TTFB/stall policy, heartbeat support, and generic native HTTP streaming support.

## Non-Goals

- Do not enable native streaming for priority providers from Phase 5b.
- Do not rewrite the entire streaming executor or change existing chat-completions SSE chunk format by default.
- Do not implement a WebSocket FastAPI route.
- Do not weaken Phase 6b fallback visible-output safety or Phase 7b cooldown/retry latch behavior.
- Do not introduce persistence.
- Do not commit user-facing reports.

## Current State

- `StreamMonitor` records TTFB, TTFT, chunk counts, errors, cancellation, and stall status, but no active timeout policy uses it.
- `StreamingHandler.wrap_stream()` detects client disconnect but does not guarantee upstream iterator closure.
- No heartbeat frames are emitted during long waits between chunks.
- `NativeHTTPTransport.stream_json_lines()` requires custom injected clients to expose `stream_json_lines()` and does not support generic `httpx.AsyncClient.stream()`.
- Existing tests cover metrics and fallback, but not upstream cancellation, heartbeats, TTFB timeout, stall timeout, or generic native stream transport.

## Implementation Plan

1. Extend stream runtime settings.
   - Add `ttfb_timeout_seconds`, `stall_timeout_seconds`, `heartbeat_interval_seconds`, and `cancel_upstream_on_disconnect`.
   - Add env overrides: `STREAM_TTFB_TIMEOUT_SECONDS`, `STREAM_STALL_TIMEOUT_SECONDS`, `STREAM_HEARTBEAT_INTERVAL_SECONDS`, `STREAM_CANCEL_UPSTREAM_ON_DISCONNECT`.
   - Keep defaults behavior-compatible: no timeout/heartbeat unless configured; upstream cancellation enabled on disconnect.

2. Add upstream stream close helper.
   - Close via `aclose()` when available, otherwise `close()`.
   - Use it on client disconnect, cancellation, or abnormal stream exit.
   - Log close failures only at debug/trace level.

3. Add heartbeat formatting.
   - Add `format_heartbeat()` to SSE/WebSocket/JSONL formatters.
   - SSE heartbeat is a comment frame such as `: heartbeat\n\n`.
   - Heartbeats must not count as visible output, session evidence, or usage.

4. Add heartbeat emission in `StreamingHandler.wrap_stream()`.
   - When configured, wait for upstream chunks with heartbeat interval timeout and yield heartbeat comments while waiting.
   - Default remains no heartbeat.

5. Add TTFB timeout policy.
   - If configured and no first byte arrives in time, raise `StreamedAPIError` with structured `api_connection` timeout payload.
   - This occurs before visible output so existing retry/fallback can apply.

6. Add stall timeout policy.
   - If configured and no chunk arrives for the configured interval after first byte, raise `StreamedAPIError` with structured `api_connection` timeout payload.
   - Phase 7b visible-output latch must still suppress retry/fallback/cooldown if output was already visible.

7. Add native `httpx` stream support.
   - Keep custom `stream_json_lines()` support first.
   - Otherwise use `client.stream("POST", ...)` with `aiter_lines()` or `aiter_bytes()` fallback.
   - Preserve `[DONE]`, ignore empty lines, and parse `data:` JSON when possible.

8. Add lifecycle trace events.
   - `stream_heartbeat`
   - `stream_ttfb_timeout`
   - `stream_stall_timeout`
   - `stream_upstream_cancelled`
   - `stream_upstream_close_failed`
   - Keep snapshots disabled and metadata sanitized.

## Tests

- Formatter heartbeat tests.
- Streaming handler disconnect/upstream close tests.
- Heartbeat interval tests.
- TTFB timeout tests.
- Stall timeout tests before and after visible output.
- Native HTTP transport tests for `httpx`-style streaming.
- Phase 7b retry/cooldown/routing regression subset.

## Acceptance Criteria

- Client disconnect closes upstream async streams when possible.
- Heartbeats are supported, disabled by default, and non-visible.
- Configured TTFB/stall timeouts produce structured stream errors.
- Visible-output latch still prevents retry/fallback/cooldown after output.
- Native HTTP transport supports generic `httpx` streaming plus custom test clients.
- Stream traces include heartbeat/timeout/cancel metadata without secrets.
- Focused tests and dual-agent review pass.
