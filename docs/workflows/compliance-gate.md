# Compliance Gate

Pending-marker insurance. The smallest workflow in the platform, and the reason a GitHub API outage can never make a PR silently mergeable.

**Triggers:** `pull_request` — `opened`, `synchronize`, `ready_for_review`, `reopened`.
**Executes from:** the PR's base branch (GitHub rule for `pull_request`) — same caveat as the stub: keep dev merge-synced when this file changes.
**Permissions:** `statuses: write`. **Zero secrets, no checkout.**

## Why it exists

The `compliance-check` pending status is what branch protection keys on. Normally the PR Review Trigger stub posts it — but on 2026-08-17 a real API outage swallowed that POST *silently* (the step was `continue-on-error` by design, so everything looked green while the required check simply didn't exist). A PR with no compliance-check context at all can merge. This workflow is the second, independent poster:

- Retries with backoff.
- If it still cannot post, **fails loudly** — a red `Compliance Gate` check on the PR. Loud-and-red beats silent-and-absent: the required check stays "Expected" (merge blocked) either way, but now a human knows.

Fork PRs skip the post (read-only token there); the stub covers forks.

## When it goes red

Almost always: the statuses API is refusing writes (outage) — in which case the red is the feature. If it's red with no outage, look at the run log's final error; it names the exact API response.

## Testing it

Temporarily point branch protection at requiring `Compliance Gate` too if you want to see it in the checks list; otherwise it's a quiet insurance policy that only appears in the Actions tab.
