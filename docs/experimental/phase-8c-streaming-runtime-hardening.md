# Phase 8c: Responses Stream Runtime Hardening And Anthropic Close Safety

## Goal

Close all Phase 8/8b third-pass streaming findings while preserving the transport-neutral Responses stream model and the existing Anthropic compatibility wrapper.

## Scope

- Apply Phase 8b stream runtime settings directly in `ResponsesService.stream_events()`.
- Add Responses heartbeat formatting support so SSE wrappers can emit non-visible heartbeat frames.
- Enforce Responses TTFB timeout before first upstream chunk.
- Enforce Responses stall timeout after a prior upstream chunk/visible output.
- Close upstream Responses chat streams on client disconnect, timeout, or abnormal exit.
- Keep Responses cost-comment handling for Phase 9c, but avoid making heartbeat/cost metadata visible output.
- Ensure Anthropic compatibility streaming always attempts to close upstream on disconnect and wrapper exit, including generator-style streams that only expose close on the iterator.

## Non-Goals

- Do not add WebSocket routes.
- Do not replace the Responses chat bridge with native Responses execution.
- Do not change public Responses event ordering except inserting SSE comment heartbeats when configured.
- Do not make timeout defaults active; all timeout/heartbeat knobs remain opt-in through existing config.
- Do not solve Phase 9 cost-comment propagation here except preserving metadata as non-visible.

## Implementation Plan

1. Responses heartbeat formatter.
   - Add heartbeat handling to `ResponsesSSEFormatter` and `ResponsesWebSocketFormatter`.
   - Represent heartbeat as a transport-neutral non-terminal `ResponsesStreamEvent` with `event_name="heartbeat"`.
   - SSE heartbeat output is `": heartbeat\n\n"`.

2. Responses runtime settings.
   - Load `get_stream_runtime_settings()` inside `ResponsesService.stream_events()`.
   - Use polling around upstream `__anext__()` to enforce optional TTFB and stall timeouts and emit optional heartbeats.
   - Keep all settings disabled by default.

3. Upstream close helper.
   - Close upstream iterator/stream with `aclose()` or `close()` on disconnect, timeout, or abnormal exit.
   - Trace close/close-failure events.

4. Disconnect detection.
   - Poll `request.is_disconnected()` while waiting for upstream chunks.
   - If disconnected, close upstream when configured and stop without emitting model output.

5. Anthropic close safety.
   - Track both the async iterator and original OpenAI stream as close candidates.
   - Ensure disconnect and wrapper exit close whichever object exposes `aclose()` / `close()`.

6. Tests.
   - Responses heartbeat, TTFB timeout, stall timeout after output, disconnect close, and heartbeat SSE comment tests.
   - Anthropic iterator-only upstream close test.
   - Broader streaming/Responses/Anthropic regression slice.

## Acceptance Criteria

- Responses streaming honors `STREAM_TTFB_TIMEOUT_SECONDS`, `STREAM_STALL_TIMEOUT_SECONDS`, `STREAM_HEARTBEAT_INTERVAL_SECONDS`, and `STREAM_CANCEL_UPSTREAM_ON_DISCONNECT`.
- Responses heartbeat frames are non-visible SSE comments.
- Responses upstream chat streams are closed on disconnect/timeout/abnormal exit.
- Anthropic compatibility streaming closes upstream even when close is exposed only on the async iterator.
- Focused tests pass and both reviewers report no blockers/highs/mediums.
