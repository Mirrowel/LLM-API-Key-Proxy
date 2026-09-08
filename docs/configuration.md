# Configuration reference

Every knob the platform reads, with syntax, examples, and the reasoning. Two shelves: **variables** (non-sensitive tuning, editable in repo settings without re-encrypting anything) and **secrets** (credentials and anything sensitive).

## Variables

Set under `Settings → Secrets and variables → Actions → Variables`:

```bash
gh variable set AGENT_PAUSED -R <owner>/<repo> --body "true"
```

### `AGENT_PAUSED`
**Default:** `false` (seeded). **Type:** kill switch.

`true` pauses the agent's brain: the router dispatches nothing, the mention poller stays silent, and all four agent workflows skip with a visible gray "skipped" — manual dispatches included. What deliberately keeps running: the PR Review Trigger stub and Compliance Gate, so open PRs keep their pending merge-blocker status — **pausing never makes a PR mergeable**.

*When to use it:* incidents, migrations, "the bot is being weird and I want it quiet while I look." Flip back to `false` (or delete the variable) to resume.

### `AGENT_MODELS_JSON`
**Default:** empty template (seeded). **Type:** per-agent model overrides.

```json
{
  "pr-review":        { "model": "anthropic/claude-sonnet-4", "fast": "anthropic/claude-haiku-4" },
  "bot-reply":        { "model": "openai/gpt-5" },
  "compliance-check": { "model": "anthropic/claude-sonnet-4" },
  "issue-comment":    { "model": "anthropic/claude-haiku-4" }
}
```

Rules:
- Empty string = not set → falls back to the global `OPENCODE_MODEL` secret. Missing keys: same fallback. Fill what you want, leave the rest.
- `"fast"` (optional) sets that agent's `small_model` (what opencode uses for its own sub-agents).
- Malformed JSON **fails the run loudly** — never a silent global fallback.
- Model names never appear in any log; the run only says the override is active.
- Format is `provider/model`, and the provider must exist in your `OPENCODE_CONFIG_JSON`.

*When to use it:* one heavyweight reviewer + cheap triage is the classic split; also cost control after watching the per-run `opencode stats` summaries.

### `OPENCODE_PLUGINS_JSON` (+ `_1` … `_5`)
**Default:** `{}` (seeded). **Type:** plugin files, without committing them.

```json
{ "myplugin/myplugin.js": "<the file's entire content, JSON-escaped>" }
```

Build the value without hand-escaping:

```bash
python -c "import json,pathlib;print(json.dumps({'myplugin/myplugin.js': pathlib.Path('myplugin.js').read_text()}))" \
  | gh variable set OPENCODE_PLUGINS_JSON -R <owner>/<repo> --body-file -
```

Then reference the materialized path in your `OPENCODE_CONFIG_JSON` secret's `plugin` array:

```json
"plugin": ["/home/runner/.mirrobot-plugins/myplugin/myplugin.js"]
```

Each variable may hold one plugin or several entries; `_1`..`_5` hold more. All merge; the same path in two variables is a hard error (a config mistake, always). Files land `chmod 600` outside the workspace, are permission-denied to the agent, and are **deleted seconds after opencode boots** — same lifecycle as the config. Relative paths only (no `/`, `..`, or spaces — unsafe paths fail the run). Remove the `plugin` entry when you remove a plugin: referencing a path nothing materializes is a boot error.

*When to use it:* private provider plugins (the router that must not be committed), anything opencode loads from a path.

### `TRUSTED_AGENT_USERS`
**Default:** absent (= empty). **Type:** comma-separated logins.

People — beyond collaborators — whose asks carry friendly context in the agent's judgment. A trust *signal*: the agent still evaluates risk on its own; this never bypasses its judgment.

```bash
gh variable set TRUSTED_AGENT_USERS -R <owner>/<repo> --body "alice,bob"
```

### `CONTEXT_IGNORE_AUTHORS`
**Default:** absent (= empty). **Type:** comma-separated logins.

Posts from these logins never enter the agent's thread context at all — perfect for another bot you never want the agent reading. `[bot]` suffixes work.

### `CONTEXT_FILTER_PATTERNS_JSON`
**Default:** baked AI-reviewer noise defaults. **Type:** JSON array of case-insensitive regexes.

Any match on a post's **body** drops that post from thread context. Setting the variable **replaces** the baked defaults (which target only known noise classes: CodeRabbit rate-limit/skip/too-many-files posts, "No actionable comments" notices, Greptile's status channel — substantive reviews from those same tools survive).

```json
["skip rationale I never want", "^<!-- deploy-status -->", "stale-bot marker [0-9]+"]
```

Regex metacharacters work; backslashes double as JSON escapes. Malformed JSON falls back to the defaults with a workflow warning.

### `PREVIOUS_BOT_REVIEWS_COUNT`
**Default:** `1` (seeded). **Type:** integer.

How many of the agent's own newest PR reviews are elevated (unfiltered) into its review context. Raise to `3` for long-lived PRs with many rounds — deeper memory of its own prior findings.

### `FOREIGN_MENTIONS_ENABLED`
**Default:** `false` (seeded). **Type:** cross-repo master switch.

Must be exactly `true` to arm the mention poller. Everything guest-mode: [workflows/mention-poller.md](workflows/mention-poller.md).

### `FOREIGN_MENTIONS_USERS`
**Default:** absent (= empty). **Type:** comma-separated logins.

Who may summon the agent *cross-repo* (unioned with home collaborators, owner included). Deliberately separate from `TRUSTED_AGENT_USERS`: cross-repo summoning is its own, stricter privilege — home trust does not travel.

## Secrets

`Settings → Secrets and variables → Actions → Secrets`. Minimal set: `OPENCODE_MODEL` + `OPENCODE_CONFIG_JSON` + one identity pair.

### `OPENCODE_MODEL`
Main model, `provider/model` (e.g. `anthropic/claude-sonnet-4`). Overrides any `model` key in your config — per-agent overrides (above) beat this.

### `OPENCODE_CONFIG_JSON`
Your complete [OpenCode config](https://opencode.ai/docs/config), minified to one line: providers (with models and API keys), `small_model`, agents, MCP servers, the `plugin` array, and the `permission` profile. Start from the committed template `.github/actions/bot-setup/permissions.example.json` — it's a full-config example with a placeholder provider, an MCP entry, a plugin reference, and the recommended permission block.

```bash
python minify_json_secret.py my-config.json   # RFC 8259-strict minifier
gh secret set OPENCODE_CONFIG_JSON -R <owner>/<repo> < my-config.min.json
```

Runtime lifecycle: written `chmod 600`, every credential leaf inside it (provider API keys, MCP header values, credential-bearing URLs) registered with `::add-mask::` so no later step can echo one, then **deleted seconds after opencode boots** (empirically verified: sessions complete with the file deleted mid-run). bot-setup warns when the secret's permission block drifts from the committed example — informational; the secret always wins.

### `OPENCODE_API_KEY` (optional)
API key for the main model's provider — injected as `provider.<main>.options.apiKey`. Redundant when the key lives inside your config (recommended).

### `OPENCODE_FAST_MODEL` (optional)
Global `small_model`. Per-agent `"fast"` overrides beat this.

### Identity pair — exactly one of:

| | Secret(s) | Notes |
|---|---|---|
| **Account mode** (recommended) | `ACCOUNT_GH_TOKEN` | Classic PAT of the bot account, `public_repo` scope **only**. The `workflow` scope is hard-rejected (it would remove GitHub's workflow-push backstop); missing `public_repo` fails; a broken token fails fast, never silently falls back. Add `notifications` + `actions:read` when enabling cross-repo guest mode. |
| **App mode** | `BOT_APP_ID` + `BOT_PRIVATE_KEY` | Full PEM contents including BEGIN/END lines. App permissions: Contents read-only, Issues read/write, Pull requests read/write; Variables: write if you want Bootstrap to seed via the App. |

Present account token → account mode; else App pair → app mode; else workflows fail with a clear error.

### `SHARE_LINK_PUBKEY` (optional)
RSA **public** key (PEM) for encrypted session share links. Without it, share URLs are still captured and masked but not recoverable. One-command setup: `python decrypt_share_link.py setup` (generates the pair, sets the secret, keeps the private key locally). See [security.md](security.md#encrypted-share-links).

## Interaction cheatsheet

| Want | Set |
|---|---|
| Different model per agent | `AGENT_MODELS_JSON` |
| Quieter/cheaper triage | `issue-comment` entry in the same variable |
| Mute another bot entirely | `CONTEXT_IGNORE_AUTHORS` |
| Drop a recurring noise post | `CONTEXT_FILTER_PATTERNS_JSON` (remember: replaces defaults) |
| Stop everything safely | `AGENT_PAUSED=true` |
| Private plugin | `OPENCODE_PLUGINS_JSON` + `plugin` path in the config |
| Agent follows me into other repos | `FOREIGN_MENTIONS_ENABLED` + `FOREIGN_MENTIONS_USERS` + PAT scopes + the worker |
