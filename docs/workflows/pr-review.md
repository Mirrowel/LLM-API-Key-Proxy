# PR Review (+ PR Review Trigger stub)

Two files, one pipeline: a zero-secret stub that reacts to PR events, and the reviewer agent it dispatches.

## PR Review Trigger (the stub) — `.github/workflows/pr-review-trigger.yml`

**Triggers:** `pull_request_target` — `opened`, `synchronize`, `ready_for_review`, `reopened`, `review_requested`.
**Executes from:** the **PR's base branch** (GitHub rule for `pull_request_target`). On PRs targeting dev, dev's copy runs — keep dev merge-synced when this file changes.
**Permissions:** `statuses: write` (the pending marker), `actions: write` (the dispatch). **Zero secrets.**

What it does, in order:

1. **Posts the pending `compliance-check` status** (retried; `continue-on-error` — the Compliance Gate is the independent second poster).
2. **Decides review-wanted:**

| Event | Review? |
|---|---|
| opened (non-draft) | yes |
| ready_for_review, reopened (non-draft) | yes |
| synchronize | only with the `Agent Monitored` label |
| review_requested | only when the requested reviewer is *our own identity* (someone clicked the reviewer button on the bot) |

3. **Dispatches PR Review** — `--ref <default branch>`, inputs `prNumber`, `triggerAction`, `source=stub`. Declined events dispatch *nothing*; the decision and the dispatch are the same act.

**Pause behavior:** while `AGENT_PAUSED=true`, the stub still runs and still posts the pending marker, but suppresses the dispatch with a visible notice — pausing never makes a PR mergeable.

**When it goes red:** the dispatch step fails loudly after 3 attempts (a lost review is a real failure). Anything else red is the pending-post path failing unrecoverably — check whether the statuses API is having a day.

**Tampered-copy ceiling** (accepted platform exposure): a modified base-branch copy can choose *when* reviews fire and make status noise. It cannot change what runs — the dispatch target ref is validated downstream, and PR Review re-fetches everything from the API.

## PR Review — `.github/workflows/pr-review.yml`

**Triggers:** `workflow_dispatch` only (router, stub, or manual).
**Executes from:** the default branch, always.
**Permissions:** `contents: read`, `pull-requests: write`.

**Inputs:** `prNumber` (required at runtime), `commentId` (requested-review context), `triggerAction` (auto runs), `source` (`stub` | `router` | `manual`).

Pipeline: bot-setup → metadata fetch (fail-closed on bogus numbers before any spend) → **review kit** (`generate-review-kit.sh`: FIRST/FOLLOW-UP detection from the agent's own review footers, full + incremental diffs, head-SHA file, instruction sets, review memory) → PR-head checkout (API-recorded SHA, TOCTOU-safe) → scrub → trust context (taint line, roster) → ack comment (FIRST runs only; the very first action, before any diff reading) → **the agent session** → verify-and-repair (footer markers, attribution) → reactions → share summary.

**What the reviewer does** (prompt-level, tuned in `.github/prompts/`):

- Navigates the diff at its own pace (never mandated to ingest it whole).
- Grades findings 🔴 Critical / 🟠 Major / 🟡 Minor / 🔵 Info; criticals and majors land as inline comments, the rest in grouped summaries. Nothing found is ever silently dropped — everything is placed.
- Ends in a verdict: **changes requested** (hard no, must-fix), **comment** (advisory, still blocks via dismissal semantics), or **approve** — approval requires a *positive* repository purpose (rank-blind: an admin's pointless PR gets no approval either).
- First reviews post a living ack (progress-edited); follow-ups post only the review — no announcement noise.
- Every posted review ends with footer markers carrying the reviewed head SHA; follow-up runs and the verify step both key off them.

**Knobs:** `AGENT_MODELS_JSON["pr-review"]`, `AGENT_MODELS_JSON["review-*"]` is not a thing — mode-level routing is via the manifest; the knobs block in the workflow env carries `BOT_NAMES_JSON`, noise filters, `PREVIOUS_BOT_REVIEWS_COUNT`. The stub's label gate is the literal string `Agent Monitored`.

**Concurrency:** group `PR Review-<N>`, serialized, no cancel — concurrent reviews of one PR are structurally impossible.

**When it goes red:** metadata fetch failures (bad number — fail-closed by design), agent session non-zero exit (real), footer verify failing to *repair* a malformed review (rare; the review itself usually landed). A skipped "Signal review" step on a stub decline is correct behavior.

**Testing it:** open a junk PR (or comment `/mirrobot-review`), then push a commit with the `Agent Monitored` label to exercise the follow-up path. Watch the run's step summary for the share link + usage stats.
