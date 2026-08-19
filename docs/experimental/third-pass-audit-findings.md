# Third-Pass Audit Findings

This document preserves the complete third-pass validation findings so the remediation pass cannot lose track of them.

References used by the reviewers:

- `C:\Projects\test\new 1.txt`
- `C:\Projects\test\new 2.txt`
- General docs: `00-master-plan.md`, `01-protocol-architecture.md`, `02-transform-logging.md`, `03-field-cache-rules.md`, `04-provider-roadmap.md`, `05-routing-retry-usage-roadmap.md`, `06-phase-workflow.md`, `07-detailed-phase-roadmap.md`
- Phase first-pass plans: `phase-1-protocol-core.md` through `phase-10-config-polish.md`
- Phase second-pass plans: `phase-1b-protocol-breadth-operation-model.md` through `phase-10b-config-surface-wiring.md`

## Overall Verdict

The third-pass review is not clean. Each phase received one normal `explore` review and one `explore-heavy` review. The reviewers found remaining blockers, highs, mediums, and low-risk residuals across the implemented work.

## Severity Summary

| Phase | Highest Severity | Clean? |
|---|---:|---:|
| Phase 1 / 1b Protocols | High | No |
| Phase 2 / 2b Transform Tracing | High | No |
| Phase 3 / 3b Field Cache | Medium | No |
| Phase 4 / 4b Responses | Medium | No |
| Phase 5 / 5b Providers | Blocker | No |
| Phase 6 / 6b Routing/Fallback | Blocker | No |
| Phase 7 / 7b Retry/Cooldown | High | No |
| Phase 8 / 8b Streaming | Medium | No |
| Phase 9 / 9b Usage/Cost | High | No |
| Phase 10 / 10b Config | High | No |

## Phase 1 / 1b: Protocols

### Blockers

- None reported.

### High

- Native OpenAI Chat and Responses formatting returns unified usage shape instead of protocol-native usage shape.
  - OpenAI Chat `format_response()` can emit `Usage.to_dict()` fields like `input_tokens` / `output_tokens` instead of `prompt_tokens` / `completion_tokens` / `total_tokens`.
  - Responses `format_response()` can do the same instead of Responses-compatible usage fields.
  - Native executor returns formatted protocol bodies directly to clients, so this can leak wrong usage schemas into live native responses.

### Medium

- OpenAI legacy `function_call` is preserved only in `extra`, not modeled as a unified `ToolCall`.
- Ollama lacks a `format_response()` override, so adapter-mutated unified responses can be ignored when `raw` is present.
- Operation compatibility is not enforced in live native provider execution.
  - Native execution resolves protocol and operation but does not call `supports_operation()`.
  - Misconfigured provider/protocol/operation combinations can silently fall back or drift.

### Low / Residual

- No cross-protocol conversion tests such as Anthropic -> unified -> OpenAI or Gemini -> unified -> OpenAI.
- `format_stream_event()` is not overridden by Anthropic, Gemini, Responses, or Ollama; they rely on base raw-pass-through behavior.
- Ollama declares `jsonl` support without JSONL-specific stream formatting.
- Unified types have `to_dict()` but no `from_dict()` deserialization helpers.
- Duplicate helper functions exist across protocol modules.
- No explicit `format_response()` tests for OpenAI Chat, Anthropic Messages, or Gemini.
- `CostDetails` protocol extraction is present in OpenAI Chat and Responses only.
- Registry discovery imports protocol modules without import-error isolation.
- Missing tests for formatted usage schemas, legacy `function_call`, Ollama mutated response formatting, registry failure isolation, and native unsupported operation rejection.

## Phase 2 / 2b: Transform Tracing

### Blockers

- None reported.

### High

- Anthropic compatibility path lacks transform-pass tracing for live conversion boundaries.
  - Missing explicit traces for Anthropic raw request, Anthropic -> OpenAI conversion, OpenAI -> Anthropic final response, and per-event stream conversion.
  - `AnthropicHandler` and `anthropic_streaming_wrapper` mostly rely on legacy transaction logger calls.

### Medium

- `raw_provider_stream_response` logs the stream iterator object rather than provider stream data.
- Streaming handler lifecycle traces are gated by `trace_metrics`; chunk traces remain, but lifecycle diagnostics can disappear.
- Non-streaming final response trace can be emitted before usage normalization, making `final_client_response` not truly final.
- Native streaming does not run or trace the stream-event adapter chain.
- Responses streaming does not trace final SSE formatting boundaries.
- Redaction misses common camelCase secret keys such as `apiKey`, `accessToken`, `refreshToken`, and `clientSecret`.
- Non-streaming Responses errors are not traced with standardized transform-failure shape.
- Built-in provider transform exceptions are not logged as transform failures.

### Low / Residual

- `ProviderTransforms.apply_sync()` has no trace support; currently not used by live execution.
- `AnthropicHandler.count_tokens()` has no transaction logger or trace entries.
- Provider logger trace entries cannot override per-entry `credential_id`; they rely on context correlation.
- No dedicated test for Anthropic transform tracing.
- Responses service has redundant `if transaction_logger` guards before `_trace()` calls.
- Native provider streaming does not apply adapter chains to stream events.
- Explicit LiteLLM fallback is traceable by metadata but has no pass named `litellm_fallback_request`.

## Phase 3 / 3b: Adapters And Field Cache

### Blockers

- None reported.

### High

- None reported.

### Medium

- Runtime does not execute all declared field-cache sources and targets.
  - `request`, `unified_request`, `unified_response`, and `unified_stream_event` are accepted, but live native execution only injects `request` and extracts `response` / `stream_event`.
- Adapter-chain traces can leak arbitrary configured provider-state fields.
  - Native executor applies rule-aware redaction around its own traces, but `run_adapter_chain()` logs payloads directly.
- Credential-scoped field-cache rules do not fail closed when `credential_id` is missing.
  - Missing credential can become a shared hashed `_none` bucket in direct/library native executor usage.

### Low / Residual

- Copilot provider has no field-cache rules by design.
- `ProviderCacheFieldStore.clear()` depends on injected cache exposing `clear()`.
- Turn-mode inference defaults to common `messages` shape; custom providers need explicit metadata.
- `per_tool_call` injection requires tool-call ID in context metadata or payload path.
- `FieldCacheEngine` defaults to a fresh in-memory store if used directly; `NativeProviderExecutor` correctly shares one per executor instance.
- Native streaming path extracts from stream events but does not inject into stream events, which is expected.
- `ProviderCacheFieldStore.append()` may serialize already wrapped values twice in edge cases.
- Process-local default store is per `NativeProviderExecutor` instance, not a global singleton.
- JSON parsing accepts non-positive `ttl_seconds`, which effectively disables expiry.
- Provider declarations are still conservative/mock-live for several priority providers.

## Phase 4 / 4b: Responses API

### Blockers

- None reported.

### High

- None reported.

### Medium

- Responses storage is process-local by default; durable JSON/current-state or provider-cache-backed storage is not wired into app startup.
- `previous_response_id` bridge context is lossy.
  - The bridge replays parent output only, not parent input or full lineage.
- Responses route errors are wrapped as FastAPI `detail.error` instead of top-level OpenAI-compatible `error` bodies.

### Low / Residual

- `ProviderCacheResponsesStore.delete()` returns `False` if the injected cache lacks `delete_async`.
- WebSocket formatter exists but has no FastAPI route or service-neutral integration test.
- `_record_responses_session_anchor()` no-ops if `session_tracker` or `session_id` is unavailable.
- No periodic TTL cleanup task; expired entries prune on access/write.
- Bridge output conversion handles text/tool calls but not all richer Responses item types.
- `store_in_progress` returns opt-in `in_progress` state without a formal retrieval schema.
- `ResponsesStreamState` uses fixed `msg_0` for bridge streams.
- No explicit concurrent access tests for `InMemoryResponsesStore`.
- No cancel endpoint.
- `validate_stream_request()` and `stream_events()` can load the same parent response twice.
- `stream_response(..., transport=...)` always uses SSE formatting.
- Formatted SSE frames themselves are not traced.
- Tests do not include end-to-end FastAPI + real `RotatingClient` sticky continuation or durable app-level storage wiring.

## Phase 5 / 5b: Provider-Native Integrations

### Blockers

- Native provider responses are formatted back to the provider protocol, not the originating client protocol.
  - `/v1/chat/completions` routed to Claude Code, Codex, or Antigravity can return Anthropic Messages, Responses API, or Gemini payloads instead of OpenAI Chat payloads.
  - Native context does not carry a client target protocol; native executor uses the provider protocol for `format_response()`.

### High

- Claude Code native requests can omit required Anthropic `max_tokens`.
- Antigravity model normalization is unsafe for duplicated aliases and can lose low/high thinking-level behavior.

### Medium

- `get_native_headers()` and `get_native_endpoint()` are not declared on `ProviderInterface` even though the executor requires them dynamically.
- Native streaming is disabled for all priority providers. This is safe, but it leaves the native streaming path unexercised by real priority providers.
- Claude Code auth header convention may be wrong for API-key credentials because it always uses `Authorization: Bearer` with `anthropic-version`.
- ProviderInterface native contract lacks `supports_native_operation()` / `should_use_native_protocol()` style hooks, making non-stream native auto-selection all-or-nothing by protocol declaration.
- Native streaming fail-closed behavior is duplicated and only partially fail-closed; the executor duplicate catches fewer exceptions than the safe helper.
- Antigravity quota grouping regresses retired parity by grouping models too broadly.
- Field-cache declarations exist, but thinking/signature reinjection is not proven semantically correct for protocol-native continuation requirements.

### Low / Residual

- Copilot has no field-cache rules by design.
- Codex message-to-Responses conversion is minimal and does not handle tool calls, multipart content, images, or rich system formatting.
- Antigravity intentionally restores only a conservative safe subset; many retired features remain absent.
- Antigravity has quota groups but no usage reset configs or tier priorities.
- Priority provider model discovery catches broad exceptions and silently falls back to hardcoded lists.
- Execution-mode routing priority is correct and fail-closed.
- Native streaming fail-closed behavior is tested for explicit native streaming.
- Mock-live tests cover direct native request execution for all four priority providers, but not full end-to-end credential rotation/usage manager execution.
- Some provider docstrings still describe skeleton behavior.

## Phase 6 / 6b: Routing And Fallback

### Blockers

- Stale fallback test file imports a missing module, so broad test collection can fail.
  - `tests/test_fallback_groups.py` imports `rotator_library.fallback_groups.FallbackGroupManager`, which no longer exists.
  - The same file asserts obsolete `RotatingClient(model_fallback_groups=...)` / `fallback_manager` behavior.

### High

- Requested-model-in-group promotion is not implemented.
  - If the requested model is inside a fallback group, the router should try it first and preserve remaining group order.
- Streaming execution modes do not match non-streaming execution-mode behavior.
  - Streaming can ignore explicit `@litellm_fallback` when a plugin has custom logic and can fall through differently for `@custom`.
- Explicit native non-streaming configuration errors can be treated as retryable fallback errors because some `RoutingExecutionError`s default to `unsupported_operation`.
- Structured route-error classification misses common status-less aliases such as `invalid_api_key`, `unauthorized`, `invalid_argument`, `rate_limited`, `too_many_requests`, `resource_exhausted`, and `unavailable`.

### Medium

- Fallback target session namespace is cloned from the first target instead of being re-inferred or adjusted per target, risking response anchors under the wrong provider/model namespace.

### Low / Residual

- No end-to-end test for env config -> `RequestContextBuilder` -> `RequestExecutor.execute()` full fallback scenario.
- `build_embedding_context()` does not resolve routing groups.
- No explicit test for `_execute_non_streaming_with_fallback()` when all targets return structured error responses.
- `_route_error_type_from_response()` falls back to text-summary scanning when structured fields are ambiguous.
- Fallback exhaustion summaries are sanitized and appear safe.
- Routing route aliases are lowercase keys only; there is no hyphen/space alias normalization.
- Wrapper-level streaming tests are thinner than helper-level coverage for some event frames.

## Phase 7 / 7b: Retry, Cooldown, Backoff, Failure History

### Blockers

- None reported.

### High

- Provider/model cooldown can be bypassed when the remaining cooldown exceeds the request deadline budget.
  - `_wait_for_cooldown()` logs and returns, and callers continue into credential acquisition/execution.
- Generic transient errors start provider-wide cooldown too aggressively.
  - A single `server_error` or `api_connection` without `retry_after` can start provider cooldown immediately.

### Medium

- `decide_streaming_error_action()` is implemented and tested but not used by the live executor.
- Failure history lacks success reset; successes do not clear or reduce provider/model failure history.
- Structured retry attempt history is not populated beyond traces/summaries; `routing_attempt_history` exists but is unused in live executor.

### Low / Residual

- `FallbackAttemptRunner` is tested but not used by the live executor; inline wrappers remain.
- `decide_streaming_error_action()` does not accept `failure_history`, which would matter if it is adopted live.
- No `test_request_executor_group_policy.py` file despite the plan mentioning it.
- Streaming error branches in the executor are duplicated across several handlers.
- Model-capacity detection is narrow and misses variants such as `CAPACITY_EXHAUSTED` or `capacity-exhausted`.
- Tests miss cooldown-over-budget fail-fast/fallback, single generic transient non-cooldown, success reset, live model cooldown not blocking unrelated models, and live streaming parity with the decision helper.

## Phase 8 / 8b: Streaming Hardening

### Blockers

- None reported.

### High

- None reported.

### Medium

- Responses streaming bypasses active Phase 8b hardening at the Responses layer.
  - No direct TTFB/stall/heartbeat/disconnect/close policy is applied in `ResponsesService.stream_events()`.
  - Responses formatter has no heartbeat formatter.
- Anthropic compatibility streaming can stop on disconnect without explicitly closing the upstream OpenAI stream.

### Low / Residual

- No formal `StreamTransportFormatter` protocol/ABC.
- Executor streaming error handling remains duplicated across branches.
- Responses streaming does not emit heartbeats or enforce TTFB/stall policy directly.
- Native streaming executor does not integrate with `StreamingHandler.wrap_stream()`.
- Native `_parse_stream_line()` returns raw text on JSON parse failure.
- No specific test for stall timeout after visible output.
- Normalized stream event parser comment says malformed chunks fail closed for visibility, but parser returns `visible_output=False`; routing safety is protected elsewhere.
- `decide_streaming_error_action()` is not used by the live executor.

## Phase 9 / 9b: Usage, Quota, Cost

### Blockers

- None reported.

### High

- Provider-reported top-level response costs are ignored when a `usage` object exists.
  - `extract_usage_record()` unwraps full responses to `response["usage"]`, discarding sibling `cost`, `total_cost`, `cost_details`, or `provider_reported_cost`.
- OpenAI-like cache-write tokens can be double-counted in normalized totals/costs.
  - `input_tokens = prompt_tokens - cache_read`, while `cache_write_tokens` is stored separately and also added to total.
- Responses streaming drops SSE cost comments/events unless final usage carries cost.

### Medium

- `event: cost` frames can block streaming retry/fallback despite being metadata.
- Native streaming executor does not emit `usage_accounting_summary` trace.
- Native streaming executor does not preserve provider-reported cost from raw response.
- Structured provider cost breakdowns without `total_cost` are not preserved or summed.

### Low / Residual

- Responses bridge passes cost fields through verbatim rather than structuring them, which is acceptable but should be documented.
- No explicit native streaming cost-trace test.
- Streaming cost calculator does not pass `provider_plugin`, so provider explicit pricing can be missed in streaming mode when no provider-reported cost exists.
- Quota snapshot `used` maps to request count, not token count; metadata says request/token window but field naming is ambiguous.
- No integration-level test for Responses usage with provider-reported cost.
- No quota snapshot test for combined model and group filters.
- Planned trace pass names like `usage_cost_calculated` or `quota_snapshot_built` are not implemented if considered distinct from `usage_accounting_summary`.

## Phase 10 / 10b: Config

### Blockers

- None reported.

### High

- Full `PROXY_API_KEY` is printed to console at startup.
- Provider/protocol/adapter/model/quota-checker config surfaces are still not live-wired, despite the `providers` JSON section being accepted/documented.

### Medium

- Responses streaming ignores Phase 10 streaming runtime settings.
- Pricing env parsing can fail requests instead of warning/ignoring invalid values.
- Routing config validation is not fully startup-safe; direct model-route target specs can parse later at request time.
- Secret rejection is key-name based and misses generic `credential` / `credentials` key names.

### Low / Residual

- `STREAM_RETRY_ON_REASONING_ONLY` is env-only and not part of `StreamRuntimeSettings` JSON config.
- `get_responses_store_settings()` returns `Any` rather than a precise type.
- `provider_cooldown_env()` remains a legacy tuple wrapper over richer retry settings.
- `STREAM_HEARTBEAT_SECONDS` legacy alias resolution is subtle.
- No standalone `config-reference.md` was created.
- `providers` section accepts raw dicts but does not validate protocol/adapter names; this is conservative but incomplete for future wiring.
- `.env.example` drift test does not verify the legacy `STREAM_HEARTBEAT_SECONDS=0` default line.

## Cross-Phase Remediation Order

1. Phase 5 blocker: native provider output must be converted back to the originating client protocol.
2. Phase 6 blocker: retire/update stale fallback tests and fix routing execution correctness.
3. Phase 10 high: stop printing secrets and make accepted config surfaces honest/live-wired or reject unsupported shapes.
4. Phase 1 high/medium protocol formatting and operation enforcement.
5. Phase 9 high usage/cost correctness.
6. Phase 7 high cooldown budget and transient cooldown behavior.
7. Phase 2 tracing gaps, especially Anthropic compatibility and redaction.
8. Phase 3 field-cache runtime/redaction gaps.
9. Phase 4 Responses storage/continuation/error-shape gaps.
10. Phase 8 streaming hardening gaps for Responses and Anthropic compatibility.

## Required Completion Bar

- Every blocker/high/medium listed above must be fixed or explicitly re-reviewed as no longer applicable.
- Low-risk items should be fixed when small/safe; otherwise they must be carried into phase reports as explicit residuals.
- Each remediation phase must follow the established loop: plan in conversation, write plan doc, implement focused commits, run focused tests, run `explore` and `explore-heavy`, fix findings, re-review, write user-facing report without committing the report.
