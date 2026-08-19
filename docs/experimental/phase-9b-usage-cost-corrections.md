# Phase 9b: Provider-Reported Cost, SSE Cost Events, And Quota Snapshot Hardening

## Goal

Correct Phase 9 validation findings. Phase 9 normalized token usage and added advisory cost calculation, but provider-reported costs and streaming cost events are not yet first-class runtime accounting inputs.

## Non-Goals

- Do not replace `UsageManager`, JSON persistence, or window/limit engines.
- Do not introduce SQLite or any new database.
- Do not make advisory pricing authoritative when providers report actual cost.
- Do not build an admin quota UI.
- Do not commit user-facing reports.
- Do not alter stream fallback/cooldown safety from Phases 6b-8b.

## Current State

- `protocols.types.CostDetails` exists and some protocol adapters populate it.
- `UsageRecord` has token buckets but no first-class provider-reported cost fields.
- `extract_usage_record()` ignores `cost_details`, `cost`, `total_cost`, and protocol `Usage.cost`.
- `CostCalculator.calculate()` prefers advisory pricing and drops actual provider cost values.
- `StreamingHandler` does not parse SSE comment/event cost frames.
- `mark_success(approx_cost=...)` can already store an approximate cost total without changing `UsageManager` shape.

## Implementation Plan

1. Extend `UsageRecord` with provider-reported cost fields.
   - `provider_reported_cost: Optional[float]`
   - `cost_currency: str = "USD"`
   - `cost_source: Optional[str]`
   - Include them in `to_dict()`.

2. Extract provider-reported costs from provider shapes.
   - OpenAI-like `cost_details.total_cost`, `cost_details.cost`, `cost`, `total_cost`, and `provider_reported_cost`.
   - Anthropic/Gemini generic cost fields and `costMetadata`.
   - Protocol `Usage.cost` values.

3. Make provider-reported actual cost win in `CostCalculator`.
   - `skip_cost_calculation` remains the highest-priority behavior.
   - Add `provider_reported_cost` to `CostBreakdown` and make `total_cost` use it without double counting.
   - Preserve advisory pricing paths when no provider-reported cost exists.

4. Parse SSE cost comment/event frames.
   - Support `: cost {json}`, `: cost 0.001`, and `event: cost` frames.
   - Keep cost comments non-visible and out of session anchors.
   - Final provider usage cost overrides earlier cost comments.

5. Extend formatted SSE usage extraction.
   - Ensure formatted SSE chunks with `usage.cost_details` flow through `UsageRecord` and cost calculation.

6. Harden native and Responses cost traces.
   - Native usage traces should include cost breakdown.
   - Responses usage traces should show provider-reported cost when present.

7. Keep quota snapshots honest.
   - Do not invent cost totals if usage state does not store them.
   - Document/request-token-window scope explicitly.

## Tests

- Provider-reported cost extraction tests in `test_usage_accounting.py`.
- Provider-reported cost precedence tests in `test_usage_costs.py`.
- SSE cost comment/event tests in `test_streaming_usage_accounting.py`.
- Native/Responses trace cost tests.
- Quota snapshot honesty tests.
- Phase 8b stream regression subset.

## Acceptance Criteria

- Provider-reported actual cost is preserved in `UsageRecord`.
- `CostCalculator` uses provider-reported cost before advisory pricing, while skip-cost still wins.
- SSE cost comments/events update streaming cost accounting and `mark_success(approx_cost=...)`.
- Cost comments remain non-visible for fallback/retry/session purposes.
- Native and Responses usage traces include provider-reported cost when present.
- Quota snapshots remain read-only request/token window reports and do not invent unsupported cost totals.
- Focused tests and dual-agent review pass.
