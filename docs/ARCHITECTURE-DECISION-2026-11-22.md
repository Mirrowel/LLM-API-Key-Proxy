# Gateway architecture decision: one brain, one transport

Date: 2026-11-22
Status: decision, pending merge. Treat as binding once merged; amend in place,
do not open a sixth design document.

This note exists because two roadmaps were both still being followed
(`docs/workload-router-cutover-plan.md` for the LiteLLM path and
`workload-router/docs/newapi-v1-v2-roadmap.md` for the New-API path), and
because a third router keeps appearing in a new repo every few weeks. It
answers one question: where does each piece live.

## Decision

**New-API is the transport. The workload-router sidecar is the brain.**

| Layer | Component | Owns | Must not |
|---|---|---|---|
| Transport | New-API on `:8001` | channels, provider keys, model lists, `model_mapping`, per-channel priority, request logging, discovery, backup | classify prompts, learn from outcomes, hold per-request health state |
| Brain | `workload-router/` Python sidecar | prompt classification, profile/phase, quota reservation, cooldown, cost/opportunity ordering, failover decision, outcome ledger | store provider keys, be a second admin UI, call an LLM to decide |
| Rollback | LiteLLM gateway on `:8000` | unchanged live path | be extended with new routing features |
| Optional edge | Cloudflare Worker (`worker/` in `llm-gateway`) | rewrite `model` to a group for clients that cannot send a profile | hold policy or state |

So: yes, New-API replaces LiteLLM, but it does not become the router. The
sidecar runs in front of it and calls one model route per attempt. New-API's
`RetryTimes` stays `0` when the sidecar is authoritative, because retries below
the ledger are invisible quota spend (v2 requires this and it is already
applied on the canary).

Two placement notes:

- Today's `workload_router_shadow.py` inside LiteLLM is a **transition**
  instrument for measuring decisions on live traffic. It is not the target
  architecture and must not grow into it.
- `llm-gateway/tools/router_sidecar.py` (`:8010`) is a tier-1 keyword rewriter
  with static per-group order. It is a **fallback** for when the Python sidecar
  cannot run, not a second production brain.

## Why this split

1. New-API resolves a group by static ability priority. It cannot do
   per-request health, quota admission, or "this attempt already streamed
   bytes". Those need state at request time, which is the sidecar's job.
2. The sidecar already has the parts that are hard to rebuild: quota ledger,
   cooldowns, stream failover, 400+ provider-free tests, a live canary and a
   proven reboot path.
3. Tier-1 keyword routing alone optimises for "did not 429", not "was correct".
   The correction lives in graded outcomes, not in a better keyword list, so
   the classifier must sit next to the ledger, not next to the channel table.

## Repo roles

| Repo / path | Role | Action |
|---|---|---|
| `LLM-API-Key-Proxy/workload-router` | canonical router, all new routing work lands here | keep |
| `LLM-API-Key-Proxy` (gateway) | rollback path + client surface | keep, freeze routing changes |
| `llm-gateway` | New-API inventory, channel configs, chain ranking, operator docs | keep as config/inventory source; its `SPEC.md` governs chain order, not the router brain |
| `agentic_gateway`, `Autonomous_LLM_Router`, `llm-routed-gateway` | older routers | frozen/archived; add a one-line README pointer, port nothing else |

## Work order

Do these in order. Each step names its gate; do not start the next before the
gate passes.

1. **Freeze the alternatives (10 min).** One-line pointer to this file in
   `llm-gateway/README.md` and the three frozen repos. Prevents the next
   session from re-deciding the architecture.
2. **Land `s0-cost-vector`.** 80 commits ahead of `main` (cost vector, live
   ladder, token refresh, recovery timer). Review, then merge; `main` carries
   the deploy trigger, so merging is an owner approval, not an agent step.
   Gate: demo canaries on the branch revision, `main` tests green.
3. **Resolve the `router/tools` alias collision.** It is published by channels
   3, 13, 14, 26 and 27; the strict audit requires exactly one channel per
   alias. Choose a canonical channel per alias (operator decision) and record
   it in the route config. Gate: `groq-tools` route passes the strict audit.
4. **Fix the routing review findings, #370-#389.** Highest first: #370/#371
   (post-stream failover, terminal stream failures reported as success),
   #372/#373 (`max_tokens` clamp, truncated-response replacement), #374-#377
   (reserve per physical retry, settle streamed usage, persist counters, real
   provider limits), #380 (tools and output in context admission), #384
   (cooldown before waiting). Gate: each issue's acceptance test, suite green.
5. **Wire the sidecar in front of New-API.** Run `newapi_server.py` as the
   client-facing endpoint, New-API behind it, `RetryTimes=0`. Keep the LiteLLM
   path untouched. Gate: one real OpenCode tool loop, failover in both
   directions, restart/persistence, rollback drill.
6. **Parity, then gradual cutover.** 19 legacy aliases with documented
   profile/step mapping, then opt-in allowlist, then default. Gate: no client
   regression for a week on opt-in, and rollback is one config change.
7. **Only after cutover: learn.** S2 client oracle (the missing OpenCode
   grading plugin) before S3 ladder, S4 shadow prices, S8 priors. Tier-2
   embeddings only if graded data shows tier-1 classification is the
   bottleneck.

## Guardrails for every change here

- Free-only, validated at startup: a priced route in a `free_only` process is
  a startup error, not a silent skip.
- No LLM call in the routing path; no dependency over ~64 MB RSS on VPS-40.
- Provider-free tests stay green: `cd workload-router && python3 -m pytest -q`.
- Every change updates `workload-router/STATE.md` and keeps one flag of
  rollback.
- Do not build a third router. If a design disagrees with this file, amend it
  in place with a revision line and say why.

## Owner decisions still needed

These block specific steps and are not agent decisions:

- approval to merge `s0-cost-vector` into `main` (deploy trigger);
- approval for the eventual default cutover, and the rollback window;
- `lambda_T`, the value of one hour of your time, without which cost and
  latency cannot be compared;
- which benchmark sources may seed priors, and whether paid routes stay
  visible in reports while free-only is on;
- the four held model additions (Groq `qwen/qwen3.8-27b`, four OpenRouter
  `:free` ids) from `llm-gateway/QUESTIONS.md`;
- legal review of offering the gateway to others (task-board #516).

## Evidence behind this decision

Read-only audit on 2026-11-22 of `llm-gateway` (epic #876, all children
closed, last commit 2026-11-07), `LLM-API-Key-Proxy` (`main` 2a2a39b,
`s0-cost-vector` 2383c6c, 21 open review issues, canary state in
`workload-router/docs/newapi-v1-v2-roadmap.md` "Resume here"), and the
task-board gateway issues. The canary on `:8124` has one strict route, a
durable token, a proven reboot and a rollback drill; nothing is cut over.
