# Phase 2b: Complete Live Transform-Pass Trace Coverage

## Goal

Correct the Phase 2 audit gap. Phase 2 created the trace writer and added important request, response, stream, provider, and error trace entries, but the validation pass found that it still does not capture every meaningful real transformation boundary. Phase 2b makes transform tracing systematic across the live LiteLLM path, provider transforms, native protocol execution, Responses service/streaming, adapter chains, field-cache passes, and stream wrappers without changing request behavior.

## Non-Goals

- Do not replace LiteLLM execution in this phase.
- Do not make non-chat protocols live routes here.
- Do not implement field-cache persistence or new field-cache semantics here; only improve logging of existing cache operations.
- Do not implement WebSocket runtime support here.
- Do not add security/multi-user features beyond existing redaction behavior.
- Do not commit user-facing reports.
- Do not touch unrelated dirty `ARCHITECTURE.md`, `STRUCTURE.md`, `.opencode/`, `docs/issues/`, or old phase reports unless explicitly needed.

## Current Code State

- `TransformTraceWriter` exists with JSONL and snapshot files.
- `TransactionLogger.log_transform_pass()` and `log_transform_error()` exist and are additive.
- Existing legacy trace entries include `raw_client_request`, `prepared_provider_request`, `final_client_response`, stream chunk entries, provider request/chunk/final/error entries, routing metadata, retry/cooldown metadata, and usage summaries.
- `ProviderTransforms.apply()` mutates request kwargs through built-in transforms, provider hook transforms, model options, and LiteLLM conversion, but it does not emit per-step trace entries.
- `RequestExecutor._prepare_request_kwargs()` calls `ProviderTransforms.apply()`, then `log_transformed_request()` logs only the final prepared provider kwargs.
- Non-streaming LiteLLM/provider responses are logged only after success as `final_client_response`; there is no explicit `raw_provider_response` or `post_usage_normalization_response` distinction.
- Streaming logs `raw_stream_chunk`, `parsed_stream_chunk`, and `assembled_stream_response`, but the live stream handler has additional normalization/usage/error decision work that is not fully trace-covered.
- `NativeProviderExecutor` traces some native passes, but it skips key states: raw native input, parsed unified request, built provider request before adapters, after response adapters, field-cache extraction details, and final formatting boundaries in a complete request/response sequence.
- `run_adapter_chain()` traces `before_adapter_chain` and `after_adapter`, but not a final `after_adapter_chain` summary and not enough metadata to distinguish built-in vs provider-declared adapter chains in live reviews.
- `FieldCacheEngine` logs per-rule operations, but injection/extraction needs start/end summary trace entries so debugging can see when a cache pass happened but no rule matched.
- Responses service currently logs a service create pass, but stream output and bridge conversions are not fully represented as protocol/bridge/final boundaries.

## Trace Taxonomy

### Client Request

- `raw_client_request`
- `pre_provider_transform_request`
- `after_builtin_provider_transform`
- `after_provider_hook_transform`
- `after_provider_model_options`
- `before_litellm_conversion`
- `after_litellm_conversion`
- `prepared_provider_request`

### LiteLLM / Provider Execution

- `provider_execution_request`
- `raw_provider_response`
- `post_usage_normalization_response`
- `final_client_response`

### Native Protocol Request

- `raw_native_client_request`
- `native_protocol_selected`
- `parsed_native_unified_request`
- `built_native_provider_request`
- `before_adapter_chain`
- `after_adapter`
- `after_adapter_chain`
- `field_cache_injection_start`
- `field_cache_rule_*`
- `field_cache_injection_complete`
- `native_provider_request`

### Native Protocol Response

- `raw_native_provider_response`
- `parsed_native_unified_response`
- `formatted_native_response`
- `after_response_adapter_chain`
- `field_cache_extraction_start`
- `field_cache_rule_*`
- `field_cache_extraction_complete`
- `usage_accounting_summary`
- `final_client_response`

### Native Stream

- `native_provider_stream_request`
- `raw_native_provider_stream_chunk`
- `parsed_native_stream_event`
- `after_stream_event_adapter_chain`
- `after_field_cache_stream_extraction`
- `formatted_client_stream_event`

### Responses API

- `responses_raw_request`
- `responses_parsed_request`
- `responses_bridge_chat_request`
- `responses_bridge_chat_response`
- `responses_stored_response`
- `responses_final_response`
- stream equivalents for created, delta, completed, bridge chunk, and final states where available.

### Errors

- Continue using `transform_log_error`.
- Include failed pass name, component, provider, protocol, transport, and sanitized payload.

## Implementation Plan

1. Add small local trace helpers where they avoid repeated fragile `if transaction_logger` blocks.
   - Avoid global state.
   - Avoid dependency cycles.
   - Prefer local helpers when a shared helper would create import loops.

2. Expand `ProviderTransforms.apply()` tracing.
   - Add optional keyword-only `transaction_logger`, `credential_id`, `transport`, and `trace_metadata` parameters.
   - Trace `pre_provider_transform_request` before any mutation.
   - Trace `after_builtin_provider_transform` after each built-in transform that returns a modification string.
   - Trace `after_provider_hook_transform` when provider hooks modify or report modifications.
   - Trace provider hook errors as `transform_log_error` without changing failure behavior.
   - Trace `after_provider_model_options` when options are applied.
   - Trace `before_litellm_conversion` and `after_litellm_conversion` around `convert_for_litellm()`.

3. Pass trace context from `RequestExecutor._prepare_request_kwargs()`.
   - Provide transaction logger, provider/model/session/scope/classifier metadata, transport, and stable credential ID where available.
   - Add an optional `credential_id` argument rather than changing existing call semantics.

4. Add raw provider and normalization response trace in non-streaming executor.
   - Trace `provider_execution_request` immediately before the provider call after pre-request callback.
   - Trace `after_pre_request_callback` if callback changed kwargs.
   - Trace `raw_provider_response` immediately after provider call returns.
   - Trace `post_usage_normalization_response` after `_normalize_response_usage()` and before returning.

5. Expand streaming trace coverage.
   - Ensure streaming request path gets the same provider-transform traces as non-streaming.
   - Keep `raw_stream_chunk`, `parsed_stream_chunk`, and `assembled_stream_response`.
   - Add `stream_error_event` for error SSE payloads produced by executor fallback/error handling.
   - Add `stream_done_event` for `[DONE]` boundaries with snapshots disabled.
   - Do not change emitted SSE bytes.

6. Expand native provider trace coverage.
   - Add `raw_native_client_request` before protocol parse.
   - Trace `parsed_native_unified_request` after `protocol.parse_request()`.
   - Trace `built_native_provider_request` after `protocol.build_request()` and before adapters.
   - Trace `after_request_adapter_chain` after request adapters.
   - Trace `parsed_native_unified_response`, `formatted_native_response`, and `after_response_adapter_chain` on response path.
   - Add equivalent stream-event boundaries where applicable.

7. Expand adapter chain summary traces.
   - Keep `before_adapter_chain` and per-adapter `after_adapter`.
   - Add `after_adapter_chain` with adapter list, stage, and count even when there are no adapters.
   - Include `changed_from_previous` where feasible.
   - Disable snapshots for stream-event summaries.

8. Expand field-cache summary traces.
   - Log `field_cache_injection_start` / `field_cache_extraction_start` before the pass.
   - Log `field_cache_injection_complete` / `field_cache_extraction_complete` after completion.
   - Include rule count, matched count, changed count, source, and operation type.
   - Keep per-rule trace entries intact.

9. Expand Responses API and bridge tracing.
   - Trace raw request, parsed unified request, bridge chat request, bridge response, stored response, and final response where those boundaries exist.
   - Trace emitted streaming response events with stable pass names and snapshots disabled for per-chunk events.
   - Preserve route bodies and SSE event ordering.

10. Tests.
    - Extend `tests/test_transaction_logger_transform_trace.py` for new pass names and redaction stability.
    - Add or extend tests for `ProviderTransforms.apply()` tracing each stage without changing payload behavior.
    - Add native provider executor trace order tests for non-streaming and streaming.
    - Add adapter chain summary trace test.
    - Add field-cache start/complete trace tests, including no-rules/no-match cases.
    - Add Responses service/stream trace tests if existing fixtures can cover it without large integration cost.
    - Run protocol regressions because trace serialization touches protocol objects.

## Commit Checkpoints

1. `docs(experimental): plan transform trace coverage correction`.
2. Provider transform trace coverage and tests.
3. Executor request/response/stream trace coverage and tests.
4. Native provider/adapter/field-cache trace summaries and tests.
5. Responses trace coverage and tests.
6. Review fixes after `explore` and `explore-heavy`.
7. User-facing Phase 2b report, uncommitted.

## Risks And Mitigations

- Trace logging accidentally mutates payloads. Mitigation: pass deep-copied or read-only snapshots and rely on `sanitize_for_trace()`.
- New trace arguments break callers. Mitigation: make all new args keyword-only optional and keep old call sites valid.
- Stream trace output grows too much. Mitigation: keep per-stream-event snapshots disabled and log compact metadata.
- Import cycles. Mitigation: use private local helpers when needed.
- Brittle tests. Mitigation: assert pass presence and causal relative order only where required.
