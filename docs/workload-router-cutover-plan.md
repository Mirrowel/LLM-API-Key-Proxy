# Workload-Router Gated Cutover Plan

Task-board: #1070 (gate issue #1071, PR #315, both merged before this plan).

> **Direction (2026-11-22):** the target is now sidecar-in-front-of-New-API,
> with LiteLLM as the rollback path. This document still governs the
> LiteLLM-side shadow/opt-in stages used for measurement; the layering and the
> work order are in `docs/ARCHITECTURE-DECISION-2026-11-22.md`. Do not extend
> the in-process shadow into a second production router.

This plan defines how the vendored `workload-router/` component may move from
parallel/dry-run into the VPS-40 gateway path. Every stage is opt-in, reversible
with one environment flag, and gated on measured evidence. No stage carries
production traffic until its gate passes.

## Constraints (apply to every stage)

- Free-tier only; no paid provider is introduced by any stage.
- Non-destructive: `config/virtual_models.yaml`, the existing fallback chains,
  and client-visible behavior do not change unless a stage explicitly says so
  and its gate has passed.
- 1 GB RAM budget on `<VPS40_IP>`; the shadow path must stay within it.
- Back up any gateway file before editing it on the VPS.
- The legacy gateway on port 8000 remains the fallback for as long as any
  stage is active.

## Feature flags

Implemented in this change:

| Flag | Default | Effect |
|---|---|---|
| `WORKLOAD_ROUTER_DRY_RUN` | unset (off) | When `1`/`true`/`yes`/`on`, the gateway computes the workload-router decision per live request and appends it to the dry-run log. The decision is discarded: live routing is unchanged. |
| `WORKLOAD_ROUTER_DRY_RUN_LOG` | `logs/workload-router-dry-run.jsonl` | Where records are written. Prompt text is never written. |
| `WORKLOAD_ROUTER_DRY_RUN_MAX_BYTES` | `8388608` | Rotation bound; the log is rotated to `<name>.1` when exceeded. |
| `WORKLOAD_ROUTER_DRY_RUN_CATALOG` | unset | Optional path to a `deployments.json` catalog; unset uses the credential-free built-in catalog. |

Planned flags (not implemented; they must not be assumed to exist):

| Flag | Stage | Effect |
|---|---|---|
| `WORKLOAD_ROUTER_MODE` | 3 | `off` (today's default), `shadow` (today's dry run), `opt-in` (route a named alias through the workload-router). |
| `WORKLOAD_ROUTER_ALLOWLIST` | 3 | Model aliases eligible for `opt-in`; empty means none. |

## Stages

### Stage 0 - vendored and gated (complete)

- `workload-router/` is vendored, parallel, and not production-wired; README
  states so explicitly.
- Dry-run mode exists, is off by default, and logs `tier`, `deployment`, `why`
  (plus bounded metadata) per request.
- Gate evidence:
  - `cd workload-router && python3 -m pytest -q` -> `128 passed`
  - `python3 -m pytest -q tests/test_workload_router_shadow.py` -> `15 passed`
  - secret scan clean; `git diff -- config/virtual_models.yaml` empty.

### Stage 1 - shadow observation on VPS-40 (next, still no routing change)

Run the gateway with `WORKLOAD_ROUTER_DRY_RUN=1` and collect records.

Gate (all must hold before Stage 2):
- Live responses are unchanged: gateway health, model list, and a scripted
  chat completion behave identically before and after the flag is set.
- Shadow records appear at a steady rate with `why` values from the expected
  set (`chain_top`, `chain_top_warm_cache`, `warm_cache_override`,
  `chain_top_cache_below_break_even`, `continuation_sticky`,
  `no_eligible_deployment`); unexpected values are investigated, not ignored.
- RSS of the gateway process stays within the 1 GB budget (compare
  `tools/resource_smoke.py` style readings before/after).
- Log rotation triggers without error under the configured byte bound.

Rollback: unset `WORKLOAD_ROUTER_DRY_RUN` and restart the gateway service.

### Stage 2 - decision comparison report (planned)

Aggregate the dry-run JSONL against the live chain actually used
(`live_model` vs `tier`/`deployment`) into a prompt-free report: agreement
rate, disagreement buckets by phase, and `no_eligible_deployment` share.

Gate: disagreement is explained (catalog gap vs classification gap) and the
report shows no routing defect in the shadow decisions themselves.

### Stage 3 - opt-in routing for one alias (planned, code required)

Add `WORKLOAD_ROUTER_MODE=opt-in` plus `WORKLOAD_ROUTER_ALLOWLIST`, route only
listed aliases through the workload-router executor, keep everything else on
the existing chain.

Gate: a real OpenCode tool loop succeeds through the opt-in path with at least
two provider failure modes exercised, and rollback (flag off) restores the old
path with no config edit.

### Stage 4 - default cutover (planned, explicit approval required)

Only after Stages 1-3 gates hold on live data, and only with explicit user
approval. The legacy gateway stays deployed and one flag away.

## Verification commands

```bash
# component suite (provider-free)
cd workload-router && python3 -m pytest -q

# gateway dry-run coverage
python3 -m pytest -q tests/test_workload_router_shadow.py

# dry run is off unless enabled
python3 -c "from proxy_app.workload_router_shadow import is_enabled; print(is_enabled())"

# enable, inspect one bounded record, then disable again
WORKLOAD_ROUTER_DRY_RUN=1 <start gateway>
tail -n 1 logs/workload-router-dry-run.jsonl
# unset WORKLOAD_ROUTER_DRY_RUN and restart -> no new records
```

## What this plan deliberately does not do

- It does not change the live routing path, virtual model chains, or any
  provider credential.
- It does not require or perform a VPS deployment on its own.
- It does not treat a passing local suite as proof of production behavior;
  each stage's gate is measured on the target host.
