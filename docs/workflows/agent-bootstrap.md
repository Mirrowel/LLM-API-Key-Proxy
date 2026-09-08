# Agent Bootstrap

One-dispatch setup: makes the platform's control panel *exist*.

**Triggers:** `workflow_dispatch` only — summoning requires write access; arbitrary users cannot invoke it.
**Executes from:** the default branch. **Permissions: `{}`** — it grants GITHUB_TOKEN nothing.

## What it does

Creates every documented **variable** with its safe-off default or prefilled template:

`AGENT_PAUSED=false`, the empty `AGENT_MODELS_JSON` template, `CONTEXT_FILTER_PATTERNS_JSON=[]`, `FOREIGN_MENTIONS_ENABLED=false`, `OPENCODE_PLUGINS_JSON={}`, `PREVIOUS_BOT_REVIEWS_COUNT=1`.

Then it writes a **static checklist** into the run summary: every variable with its meaning, every secret with where-to-get-it, and copy-paste `gh variable set` commands as the manual fallback.

Deliberately **not** created: the three empty-default variables (`CONTEXT_IGNORE_AUTHORS`, `TRUSTED_AGENT_USERS`, `FOREIGN_MENTIONS_USERS`) — GitHub variables cannot hold empty values (the API 422s; live-verified) and absence already means empty everywhere. Set them only when you have content.

## The token reality

GITHUB_TOKEN **cannot** reach the Actions variables API — even with `actions: write`, it 403s ("Resource not accessible by integration"; the endpoint is reserved for user/installation tokens — live-verified twice). So seeding authenticates with the repo's existing **bot identity**: `ACCOUNT_GH_TOKEN` when present, else the App installation token (the App needs its *Variables: write* permission granted). With neither — or when the token lacks access — the workflow still succeeds and prints the manual commands instead.

## State-silence

Logs and summary never reveal which variables or secrets exist, are missing, or hold what. Per-variable outcomes are never printed; the summary is static, identical on every repo. (It also *cannot* probe: the secrets API needs a user token no workflow here holds.) Re-running is always safe: existing values are left byte-for-byte intact — only absent ones are created.

## When it goes red

The seeding step failed with the bot token (usual cause: App without Variables:write). No values were modified; the summary carries the manual path.

## Testing it

Run it twice. First run: variables appear. Second run: instant success, zero changes — the never-overwrite contract proving itself.
