# Phase 9c: Usage And Cost Correctness Completion

## Goal

Close all Phase 9/9b third-pass usage/cost findings while preserving `UsageManager` as the authoritative persistence/limit engine and keeping provider-reported cost precedence from Phase 9b.

## Scope

- Preserve provider-reported top-level response cost even when a `usage` object exists.
- Prevent OpenAI-like cache-write tokens from being double-counted in normalized totals/costs.
- Preserve/sum structured provider cost breakdowns that omit `total_cost`.
- Carry Responses streaming SSE cost comments/events into completed usage/cost records.
- Ensure `event: cost` frames are treated as metadata and never visible output for retry/fallback.
- Add native streaming `usage_accounting_summary` traces and preserve provider-reported stream cost where protocol events/raw chunks expose usage.

## Non-Goals

- Do not replace `UsageManager`, quota windows, or persisted usage JSON format.
- Do not invent cost totals when neither provider-reported cost nor configured/advisory pricing exists.
- Do not add billing persistence beyond existing usage state and traces.
- Do not implement admin quota/cost dashboards.
- Do not alter public token fields except to correct double-counting.

## Implementation Plan

1. Top-level provider cost preservation.
   - Merge sibling top-level cost fields into nested `usage` payloads before normalization.
   - Covered keys: `cost`, `total_cost`, `cost_details`, `provider_reported_cost`, `currency`, `costMetadata`.

2. Structured provider cost breakdowns.
   - Sum known provider-reported cost fields when no total is present.
   - Preserve currency and source metadata.

3. Cache-write double-count prevention.
   - For OpenAI-like usage, subtract both cache-read and cache-write tokens from regular input tokens.
   - Keep cache-write tokens as their own normalized bucket.

4. Responses streaming SSE cost comments/events.
   - Parse `: cost ...` comments and `event: cost` frames.
   - Merge cost into in-progress/final Responses usage without emitting model output.
   - Let final provider usage cost override earlier cost comments.

5. Metadata visibility for `event: cost`.
   - Add stream policy tests locking cost frames as non-visible and retry-safe.

6. Native streaming usage/cost trace.
   - Track stream usage/cost from unified events and raw chunks.
   - Emit `usage_accounting_summary` on stream completion.

7. Tests.
   - Cover usage normalization, cost precedence, Responses streaming costs, stream visibility policy, and native streaming cost trace.

## Acceptance Criteria

- Top-level provider-reported cost is preserved when a `usage` object exists.
- Cache-write tokens are not double-counted in OpenAI-like normalized totals/costs.
- Structured provider cost breakdowns without totals are summed and treated as provider-reported cost.
- Responses streaming carries SSE cost comments/events into final usage/cost and treats them as metadata.
- `event: cost` frames do not block retry/fallback as visible output.
- Native streaming emits `usage_accounting_summary` and preserves stream cost when available.
- Focused tests pass and both reviewers report no blockers/highs/mediums.
