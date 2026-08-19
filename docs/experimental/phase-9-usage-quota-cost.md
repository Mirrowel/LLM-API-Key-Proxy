# Phase 9 Plan: Usage, Quota, Cost

## Goal

Make usage, quota, and cost accounting consistent across LiteLLM fallback, native protocol adapters, Responses API, streaming, routing fallback chains, and provider-specific quota metadata while preserving the existing `UsageManager`, `SelectionEngine`, fair-cycle behavior, storage format, and provider quota reset logic. Phase 9 should formalize a normalized usage/cost layer that feeds the current usage engine rather than replacing it.

## Non-Goals

- Do not replace `UsageManager`, `TrackingEngine`, `LimitEngine`, `SelectionEngine`, `SessionTracker`, or the retry-after parser.
- Do not introduce SQLite or any database.
- Do not implement the full Phase 10 JSON/env config system.
- Do not change credential selection strategy semantics.
- Do not rewrite provider quota trackers.
- Do not require live provider credentials or live pricing API calls.
- Do not make cost enforcement mandatory; cost tracking is additive unless a provider already has limits.
- Do not change client-visible response usage fields except where normalization already happens today.

## Current Code Context

- `CredentialContext.mark_success()` already accepts prompt, completion, thinking, cache read/write tokens, approximate cost, and response headers.
- `TrackingEngine.record_usage()` stores request count, token buckets, output tokens, total tokens, and approximate cost into windows/totals.
- `RequestExecutor._extract_usage_tokens()` extracts usage from LiteLLM-like response objects, subtracting reasoning tokens from completion tokens when the provider includes reasoning inside completion.
- `RequestExecutor._calculate_cost()` uses LiteLLM cost helpers and returns `0.0` for providers that set `skip_cost_calculation`.
- `StreamingHandler.wrap_stream()` records stream usage after final chunk and currently calculates cost with `litellm.get_model_info()`.
- Phase 4 Responses streaming maps chat usage into Responses usage for storage.
- Phase 5 native protocols can produce provider-native usage shapes through protocol parse/format paths.
- Phase 6 routing can try multiple targets, but only the successful target should record usage.
- Phase 7 fallback summaries are client-safe and should not be polluted by raw usage/cost internals.
- Phase 8 stream metrics are timing/count-only and do not alter usage recording.

## Files To Add

- `src/rotator_library/usage/accounting.py`
- `src/rotator_library/usage/costs.py`
- `src/rotator_library/usage/quota.py`
- `tests/test_usage_accounting.py`
- `tests/test_usage_costs.py`
- `tests/test_usage_quota_snapshots.py`
- `tests/test_executor_usage_accounting.py`
- `tests/test_streaming_usage_accounting.py`
- `tests/test_responses_usage_accounting.py`
- `tests/test_native_usage_accounting.py`

## Files Likely To Touch

- `src/rotator_library/usage/types.py`
- `src/rotator_library/usage/manager.py`
- `src/rotator_library/usage/tracking/engine.py`
- `src/rotator_library/client/executor.py`
- `src/rotator_library/client/streaming.py`
- `src/rotator_library/responses/bridge.py`
- `src/rotator_library/responses/service.py`
- `src/rotator_library/native_provider/executor.py`
- `src/rotator_library/core/utils.py` only if current usage normalization helper needs a small adapter.
- Provider files only for optional cost/quota declarations, not broad rewrites.

## Normalized Usage Model

Add `UsageRecord` dataclass:

- `input_tokens`
- `output_tokens`
- `completion_tokens`
- `reasoning_tokens`
- `cache_read_tokens`
- `cache_write_tokens`
- `total_tokens`
- `raw_total_tokens`
- `request_count`
- `source`
- `provider`
- `model`
- `metadata`

Token semantics:

- `input_tokens` means billable non-cache-read prompt tokens when provider separates cache-read tokens.
- `cache_read_tokens` means prompt/cache tokens read from provider cache.
- `cache_write_tokens` means prompt/cache tokens written or created.
- `completion_tokens` means visible/output text/tool tokens, excluding reasoning when the provider reports reasoning separately.
- `reasoning_tokens` means hidden thinking/reasoning tokens.
- `output_tokens = completion_tokens + reasoning_tokens`.
- `total_tokens = input_tokens + cache_read_tokens + cache_write_tokens + completion_tokens + reasoning_tokens`.
- `raw_total_tokens` preserves provider-reported total before normalization for debugging.

Rationale: current code already avoids double-counting reasoning by subtracting thinking from completion when necessary. Phase 9 makes that rule explicit and reusable.

## Usage Extraction

Add `extract_usage_record(response_or_usage, provider=None, model=None, source=None)`.

Support:

- LiteLLM/OpenAI object usage attributes.
- dict usage fields.
- OpenAI `prompt_tokens_details.cached_tokens`.
- OpenAI `completion_tokens_details.reasoning_tokens`.
- Anthropic `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`.
- Gemini `usageMetadata`, `promptTokenCount`, `candidatesTokenCount`, `thoughtsTokenCount`, `cachedContentTokenCount`, `totalTokenCount`.
- Responses `input_tokens`, `output_tokens`, `output_tokens_details.reasoning_tokens`, `input_tokens_details.cached_tokens`.
- Existing stream usage dicts after `normalize_usage_for_response()`.

Unknown usage shapes return an empty `UsageRecord` with source metadata rather than raising in runtime paths.

Tests must cover dicts, objects, nested details, and double-count prevention.

## Cost Model

Add `CostBreakdown` dataclass:

- `input_cost`
- `cache_read_cost`
- `cache_write_cost`
- `output_cost`
- `reasoning_cost`
- `total_cost`
- `currency`
- `pricing_source`
- `metadata`

Add `ModelPricing` dataclass:

- per-token prices for input, cache read, cache write, output, reasoning.
- `currency`.
- `source`.

Add `CostCalculator`:

- prefers explicit provider/model pricing declarations.
- falls back to LiteLLM model info/completion_cost where applicable.
- returns zero cost for `skip_cost_calculation`.
- does not call network.

Minimal Phase 9 provider declarations:

- Optional `get_model_pricing(model)` on `ProviderInterface`, default `None`.
- Providers can later define native pricing without touching accounting code.

Env pricing can be minimal and Phase-10-ready:

- `MODEL_PRICE_{PROVIDER}_{MODEL}_INPUT`
- `MODEL_PRICE_{PROVIDER}_{MODEL}_OUTPUT`
- `MODEL_PRICE_{PROVIDER}_{MODEL}_CACHE_READ`
- `MODEL_PRICE_{PROVIDER}_{MODEL}_CACHE_WRITE`
- `MODEL_PRICE_{PROVIDER}_{MODEL}_REASONING`

If env parsing is too broad, implement programmatic pricing tests and leave env polish to Phase 10.

## UsageManager Integration

- Preserve existing `mark_success()` signature for compatibility.
- Add optional `usage_record` and `cost_breakdown` parameters only if needed, but do not force every call site to change.
- Preferred minimal integration:
  - Executor and streaming code use `UsageRecord` internally, then pass existing numeric fields to `mark_success()`.
  - `approx_cost` remains a float in current storage.
  - Add optional metadata later only if storage impact is small.
- Do not change persisted JSON shape unless additive and backward-compatible.
- Add trace pass:
  - `usage_accounting_summary`
  - Include normalized token fields, raw total, cost total, pricing source, provider/model, and source.
  - No credential secrets.

## Quota Snapshot Model

Add `QuotaSnapshot` dataclass:

- `provider`
- `model`
- `quota_group`
- `credential_id` optional stable/masked identifier.
- `window_name`
- `limit`
- `used`
- `remaining`
- `reset_at`
- `source`
- `metadata`

Add helper to build snapshots from `UsageManager` state:

- per model.
- per quota group.
- per credential where available.

Keep `WindowLimitChecker` behavior unchanged.

Tests should prove snapshots reflect group windows and model windows without affecting limits.

Optional `UsageAPI` helper can expose snapshots for future UI/TUI work if small.

## Routing/Fallback Usage Behavior

- Only successful target attempts record usage.
- Failed target attempts continue to record failures through existing paths.
- Route fallback summaries should include optional `usage_recorded: false` for failed targets only if already tracked internally; do not expose raw costs.
- Tests:
  - first target failure + second target success records usage only on second provider/credential.
  - native target success records normalized usage.
  - explicit LiteLLM fallback target records normalized usage.

## Streaming Usage

- Replace ad-hoc stream usage extraction/cost calculation with `UsageRecord` and `CostCalculator`.
- Preserve existing `mark_success()` numeric values and final `[DONE]` behavior.
- Preserve `skip_cost_calculation`.
- Add tests:
  - stream final usage with cache/read/write/reasoning records expected buckets.
  - stream with missing usage records zero usage but still marks success as today.
  - stream metrics from Phase 8 remain independent from token usage.

## Responses Usage

- Use `UsageRecord` when converting/storing Responses usage.
- Ensure `previous_response_id` storage is not affected.
- Add tests:
  - non-streaming Responses usage maps to normalized fields.
  - streaming Responses usage stores expected input/output totals.
  - reasoning details are not double counted.

## Native Provider Usage

- In `NativeProviderExecutor.execute()`, protocol-formatted provider responses should expose usage in a shape `extract_usage_record()` can understand.
- Add trace summary but do not require the isolated native executor to mutate `UsageManager`.
- Live routed native execution via `RequestExecutor` should record normalized usage because it receives the native response and passes through the same executor accounting.
- Tests:
  - native OpenAI-style response usage records normalized fields.
  - native Gemini usage metadata records thought/cached tokens.
  - native Responses usage records reasoning/details.

## Quota/Cost Reporting

- Keep existing quota viewer/API behavior unchanged unless adding optional fields is safe.
- Add cost totals to existing total/window stats already present as `approx_cost`.
- Do not implement hard cost caps unless trivial and isolated; cost caps can be a Phase 10+ config feature.
- Add clear docs/comments that `approx_cost` is advisory and depends on available pricing.

## Transform Trace Requirements

Add:

- `usage_accounting_summary`
- `usage_cost_calculated`
- `quota_snapshot_built` for explicit snapshot APIs/tests, not every request.

Metadata:

- provider, model, source, pricing source, skip-cost flag.

Data:

- normalized usage record and cost breakdown only.

Do not log credentials, raw headers, or raw provider errors.

## Testing Plan

Accounting tests:

- OpenAI/LiteLLM object usage.
- OpenAI dict usage with prompt/completion details.
- Anthropic usage.
- Gemini `usageMetadata`.
- Responses usage.
- reasoning double-count prevention.
- missing/unknown usage shape returns empty record.

Cost tests:

- explicit pricing calculates all buckets.
- provider `skip_cost_calculation` returns zero.
- LiteLLM fallback path still returns a float when LiteLLM has model info.
- missing pricing returns zero with `pricing_source="unavailable"`.

Quota snapshot tests:

- model window snapshot.
- group window snapshot.
- missing window snapshot is empty/no-op.

Executor tests:

- non-stream success uses normalized usage.
- trace emits usage accounting summary.
- fallback second target success records usage once.

Streaming tests:

- final stream usage uses normalized accounting.
- stream cost honors skip-cost provider.
- Phase 8 metrics still emitted.

Responses tests:

- stored Responses usage shape remains compatible.
- normalized trace emitted when transaction logger exists.

Native tests:

- native response usage shapes normalize.

Regression tests:

- Phase 1 protocol tests.
- Phase 2 transform logging tests.
- Phase 3 adapter/cache tests.
- Phase 4 Responses tests.
- Phase 5 provider/native tests.
- Phase 6 routing tests.
- Phase 7 retry/cooldown tests.
- Phase 8 streaming tests.
- `test_session_tracking.py`.
- `test_selection_engine.py`.

## Commit Checkpoints

1. Add `UsageRecord`, extraction helpers, and tests.
2. Add `CostBreakdown`, pricing helpers/calculator, and tests.
3. Wire executor non-streaming accounting and trace summary with tests.
4. Wire streaming accounting/cost calculation with tests.
5. Add quota snapshot helpers and tests.
6. Add Responses/native usage accounting coverage.
7. Run focused and regression tests.
8. Review with `explore` and `explore-heavy`; fix findings; write uncommitted Phase 9 report.

## Risks And Mitigations

- Token normalization could change quota accounting. Mitigation: preserve current numeric `mark_success()` buckets and test current OpenAI/LiteLLM behavior.
- Cost estimates may be wrong for providers with unknown pricing. Mitigation: return zero/unavailable rather than guessing; make `approx_cost` advisory.
- Persisted usage JSON compatibility could break. Mitigation: keep existing storage fields and only add optional helpers unless tests prove serialization is safe.
- Provider-specific usage shapes are broad. Mitigation: implement common native shapes now and leave provider overrides possible.
- Fallback chains could double-record usage. Mitigation: only successful attempt calls `mark_success()`; add tests for failure then success.
