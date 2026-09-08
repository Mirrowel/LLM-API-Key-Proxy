# Workflows

One page per workflow. Each page covers: what triggers it, where it executes from, what it can do, every knob it reads, what its failures mean, and how to test it.

| Workflow | One-liner | Page |
|---|---|---|
| Agent Router | The single comment entrypoint — parses, dispatches exactly one target per match | [agent-router.md](agent-router.md) |
| PR Review Trigger | Zero-secret stub: decides review-wanted, posts the pending marker, dispatches | [pr-review-trigger.md → merged into pr-review.md](pr-review.md) |
| PR Review | The reviewer agent — FIRST/FOLLOW-UP protocols, severity, verdicts | [pr-review.md](pr-review.md) |
| Compliance Check | The end-of-life merge audit (`/mirrobot-check`) | [compliance-check.md](compliance-check.md) |
| Compliance Gate | Redundant pending-marker poster — insurance against silent API outages | [compliance-gate.md](compliance-gate.md) |
| Issue Analysis | Issue triage on open (duplicates, root cause, labels) | [issue-comment.md](issue-comment.md) |
| Bot Reply on Mention | The general agent — conversations, investigations, contributions, guest mode | [bot-reply.md](bot-reply.md) |
| Mention Poller | Cross-repo ears: the account's notifications → guest sessions | [mention-poller.md](mention-poller.md) |
| Agent Bootstrap | One-dispatch setup: seeds every variable, prints the secrets checklist | [agent-bootstrap.md](agent-bootstrap.md) |
| Scrub Fixture Suite | The batteries — CI enforcement of every contract in these docs | [scrub-fixtures.md](scrub-fixtures.md) |

## The shared lifecycle

Several mechanics repeat across the agent workflows; documented once here, referenced everywhere:

- **Bot setup** (composite action `.github/actions/bot-setup/`) resolves the identity (account PAT or App token), configures git, writes the OpenCode config, applies per-agent model overrides, materializes plugin files, registers credential-leaf masks, and installs opencode.
- **Trusted artifacts** — prompts, scripts, the security brief — are copied to `/tmp` *before* any PR-head checkout. After a checkout, the workspace belongs to the PR; `/tmp` stays main's.
- **The scrub** runs after every checkout (see [architecture.md](../architecture.md#the-trust-model-split-trust)).
- **Share-link capture** — every agent session runs with `--share` and its output is piped through `share-filter.sh`: the raw URL is masked and re-published encrypted. See [security.md](../security.md#encrypted-share-links).
- **Config lifecycle** — config + plugins are deleted seconds after opencode boots; an `if: always()` step guarantees removal on every exit path. Each run's step summary carries a bare `opencode stats` usage block.
- **Reactions** — comments get 👀 → 🚀/😕 (three-stage); PR/issue bodies get 👀 only. The agent may add its own tasteful reactions.
- **Pause switch** — the `AGENT_PAUSED` variable makes every agent workflow skip visibly (gray). The stub and gate keep running so merges stay blocked while paused.
