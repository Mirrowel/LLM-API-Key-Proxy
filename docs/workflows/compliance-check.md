# Compliance Check

The end-of-life merge audit. Where the reviewer asks "is this code right?", compliance asks "is this PR *ready* — consistent, documented, following the house rules?" It runs once per PR, on request, and its status is the merge gate.

**Triggers:** `workflow_dispatch` only — the router dispatches it when someone comments `/mirrobot-check` on a PR.
**Executes from:** the default branch, always.
**Permissions:** `contents: read`, `pull-requests: write`, `statuses: write` (it owns the `compliance-check` status exclusively).

## What it checks

The agent works through the **compliance watch-list** (`FILE_GROUPS_JSON` in the workflow's env block — the one block that is *yours* to rewrite per repository). The stock generic version groups:

- **Agent Platform** — when workflows/actions change: referenced scripts, prompts, and artifacts still exist; steps stay consistent with the assembly pipeline.
- **Prompts and Scripts** — when prompt parts or manifests change: part references resolve, battery pins stay accurate.
- **Documentation** — README reflects workflow/setup/secrets changes.

A deployed repo typically adds its product groups (e.g. "when `src/auth/**` changes, verify the auth docs and env-var table were updated"). Authoring guide: [customization.md](../customization.md#the-compliance-watch-list).

Beyond the watch-list, the agent verifies merge-readiness generally: PR description coherence, leftover debug code, missing docstrings/comments on new public surface, breaking-change notes.

## Status semantics

The GitHub statuses API has no "neutral" state, so the mapping is deliberate:

| Audit result | Status | Meaning |
|---|---|---|
| Blocking findings | `failure` | **BLOCKED** — no merge |
| Warnings only | `success` + warning description & report link | **WARNINGS** — mergeable; agents must *read the description* before merging on someone's behalf (description-aware merging) |
| Clean | `success` | **COMPLIANT** |

The pending marker is posted by the stub and the gate (never by this workflow — it owns the *final* state only).

## First vs follow-up

Like the reviewer, compliance detects its own previous report (a footer marker with the last-audited SHA). Follow-up audits are incremental: they re-verify every previously-raised warning against the new diff rather than re-auditing from scratch — and they keep a warning open until it's actually resolved, not just mentioned.

## Knobs

`FILE_GROUPS_JSON` (the watch-list), `AGENT_MODELS_JSON["compliance-check"]`, the shared env-block knobs (identity, noise filters). The verdict icons (🔴→BLOCKED, 🟠→WARNINGS, else COMPLIANT) are prompt-level — see the `mission-compliance` parts.

## When it goes red

Real failures: the dispatch contract step (bogus PR number — fail-closed), the agent session, or the status/report POST after retries. A gray skip = paused. If the *status* never appears at all, that's the stub+gate's outage scenario — the gate goes red loudly rather than letting the PR look mergeable.

## Testing it

Comment `/mirrobot-check` on a PR with an obvious inconsistency (say, a workflow edit with no README touch) and watch the WARNINGS path — then fix it and re-check for the incremental follow-up behavior.
