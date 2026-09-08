# Agent Router

The single comment entrypoint. Every `issue_comment[created]` event in the repository hits this workflow and nothing else.

**Triggers:** `issue_comment [created]`.
**Executes from:** the default branch, always.
**Permissions:** `contents: read` (one sparse checkout of the shared routing script), `actions: write` (dispatching targets).

## Why it exists

Before the router, every comment event fired *three* agent workflows, of which at most one proceeded — the rest skipped, but each still produced a visible run. One comment = three runs of noise. The router parses once and dispatches exactly one target per match.

## The decision matrix

Parsed by the shared script `.github/scripts/route-comment.sh` (the same script the workflows re-run for validation — routing semantics cannot drift between copies):

| Comment contains | Dispatches |
|---|---|
| `/mirrobot-review` or `/mirrobot_review` (on a PR) | **PR Review** |
| `/mirrobot-check` or `/mirrobot_check` (on a PR) | **Compliance Check** |
| `@mirrobot` or `@mirrobot-agent` (loose match — `@mirrobotics.com` also matches, deliberately, same as the original guards) | **Bot Reply** |
| none of these | nothing |

Compound comments dispatch **all** matches — "review this, then run compliance" does both.

Guards: comments authored by any `[bot]` or by the agent's own identities are ignored (bot-loop guard, case-insensitive, both known casings enumerated because Actions expressions have no case-insensitive compare). The pause switch (`AGENT_PAUSED=true`) gates the whole job.

Each dispatch passes **only a comment id**; the target re-fetches the comment, thread, and context from the API itself. Nothing about the event payload is trusted downstream — and the trust lines the target prints ("requested by ...") are derived from that re-fetch.

## When it goes red

Dispatch failures after retries (target workflow renamed/disabled — the matrix above and the workflow filenames must stay in sync), or the script checkout failing. A skipped run = not a routed comment or paused — expected and quiet by design (the router deliberately doesn't run for non-matching comments; the `if` fails before the job starts).

## Testing it

Comment `@mirrobot hi` and `/mirrobot-review` in one comment on a PR and watch two dispatches fire; comment on an issue and watch only reply fire; check the run-name of each dispatched target — it names the comment that caused it.
