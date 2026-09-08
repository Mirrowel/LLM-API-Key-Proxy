# Mention Poller (cross-repo guest mode)

The platform's ears outside its home. Lets the agent answer genuine @mentions of its **user account** in public repositories it is *not* installed in.

**Triggers:** `repository_dispatch [foreign-mention]` (from the Cloudflare worker — the normal path) and manual `workflow_dispatch` (with optional relay payload, for testing).
**Executes from:** the default branch, always.
**Permissions:** `contents: read` (shared scripts), `actions: write` (dispatching bot-reply).

## Requirements — all four, or the feature stays off

1. **Account mode identity** (`ACCOUNT_GH_TOKEN` secret) — guest mode runs as the user account; App-only installations can't do cross-repo.
2. The PAT carries the `notifications` scope (plus `public_repo`) and `actions:read`.
3. Variables: `FOREIGN_MENTIONS_ENABLED=true` (the master switch — absent/false = off) and `FOREIGN_MENTIONS_USERS` (comma logins allowed to summon; unioned with home collaborators, owner included).
4. **The mention worker deployed** — see [tools/mention-worker/README.md](../../tools/mention-worker/README.md). Worker-first is the default: a self-scheduling Durable Object polls the account's notifications every 30s (conditional requests, ETag-cached) and relays qualifying mentions as `repository_dispatch`. An in-repo cron fallback exists but is deliberately unset.

## The gauntlet (why random strangers can't summon your agent)

A mention notification reaching the workflow is only the *start* of the vetting — `handle-mentions.sh` re-verifies everything, because the worker is untrusted relay logic by design:

1. Reason filter — only `mention`, `review_requested`, `subscribed`, `comment` reasons survive (measured taxonomy; everything else is noise).
2. Skip matrix — repos that *have* the platform installed handle their own mentions (the poller never double-serves the home repo); the account owner's repos without the platform are handled; strangers' repos are handled as guest sessions under guest rules.
3. Summoner allowlist — collaborators ∪ `FOREIGN_MENTIONS_USERS`, re-fetched live; the worker's say-so alone is worth nothing.
4. Genuine-mention token check — the content actually contains a real `@account` mention.
5. Only then: dispatch bot-reply in guest mode.

Declines are acknowledged (mark-read) and never wake Actions. Handled threads are **unsubscribed-on-engage**: the worker mutes the notification thread so plain follow-up chatter in a thread it already answered delivers nothing further — real mentions always break through.

## When it goes red

The gauntlet failing *loudly* (API errors during re-verification — fail-open only for the optional allowlist variable), or the dispatch failing after retries. Everything else is quiet declines by design. If nothing fires at all: check the worker first (`__tick` + status in its control surface), then the variables, then whether the summoner is on the allowlist.

## Testing it

From a repo the bot is *not* installed in, mention the account (as an allowlisted user). The run name tells the story: "Automated mention relay from <owner>/<repo> (mention)". End-to-end latency is dominated by the worker's 30s poll cycle; measured production: ~80–120s mention→reply.

## Cost note

Guest sessions are full agent sessions against your provider. The allowlist is the throttle — keep `FOREIGN_MENTIONS_USERS` tight.
