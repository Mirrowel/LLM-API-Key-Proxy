# Customization

Everything that's *yours* — behavior prose, watch-lists, permissions, identities, trigger words — and how to change each without breaking the machinery around it.

## Prompts

All agent behavior prose lives in `.github/prompts/parts/*.md` — there is no prompt text in the workflows. A **manifest** (`.github/prompts/manifests/<mode>.txt`) is an ordered list of part names, and a mode's full prompt is exactly those parts concatenated, then `envsubst`ed with the run's context variables.

```
parts/security-brief.md        ← every session starts here
parts/on-topic-guardrail.md    ← off-topic deflection
parts/severity.md              ← the 🔴🟠🟡🔵 ladder (every mode)
parts/mission-review.md        ← the reviewer's identity and judgment rules
manifests/pr-review-first.txt  ← = the FIRST-review assembly
```

Workflows call `bash /tmp/assemble-prompt.sh <mode> | envsubst "$VARS"`. The assembler is fail-closed (a manifest naming a missing part kills the run) and `--verify`-able (no orphan parts, no duplicate headings).

### What to edit

| You want to change | Edit |
|---|---|
| Review judgment, thresholds, verdict wording | `parts/mission-review.md` + the `protocol-*` parts |
| The agent's personality in conversations | `parts/mission-agent.md`, `parts/communication.md` |
| Compliance checklist scope (beyond FILE_GROUPS) | `parts/mission-compliance.md` |
| Security posture / refusals | `parts/security-brief.md` (carefully — it's read first, every session) |
| Severity ladder wording | `parts/severity.md` |

### Rules of the road

- **Shared parts are shared byte-for-byte.** If two modes need different rules, that's *two parts*, not one part with mode-dependent prose. This is what keeps duplicated guidance from drifting apart.
- **Battery pins cover the prompt rules.** After editing parts, run `bash .github/scripts/prompt-rule-fixtures.sh`. If a pin fails, you changed pinned behavior — update the pin in the same commit *because you decided to*, not to make it shut up.
- **Render any mode's full prompt yourself** (`bash .github/scripts/assemble-prompt.sh <mode>`) to read exactly what the agent will read.
- Envsubst variables are per-mode whitelists in the workflow (`VARS='...'`) — a new `$SOMETHING` in a part needs the variable added to the mode's list, or it ships literally.

## The compliance watch-list

`FILE_GROUPS_JSON` in `compliance-check.yml`'s env block — the platform's most repo-specific knob. Each group: a name, a **description of what the agent should verify**, and file globs:

```json
[
  {
    "name": "Auth surface",
    "description": "When auth code changes, verify docs/auth.md and the .env.example var table were updated together, and no new endpoint skips the middleware list.",
    "files": ["src/auth/**", "middleware/auth*.py"]
  }
]
```

The description is the actual instruction — write it like you'd brief a new maintainer. Globs are matched against the PR's changed files; each matching group becomes a checklist item in the audit.

## The permission profile

Lives inside your `OPENCODE_CONFIG_JSON` (the committed `permissions.example.json` is the template — a full-config example whose `permission` block you paste in). Shape: a `bash` deny-catch-all with ordered allows, targeted denies last (OpenCode is **last-match-wins** — an allowlist without the catch-all is cosmetic), plus `read`/`edit`/`webfetch`/`websearch`/`skill` rules. The example's comments explain each deny. Two invariants worth knowing:

- `webfetch: deny` on purpose (agents use `websearch` + their judgment; webfetch burns tokens).
- The `~/.config` and `~/.mirrobot-plugins` read-denies are load-bearing — they're what makes the config/plugin lifecycle airtight. If you loosen them, you're reopening the window the deletion timer closes.

bot-setup warns when your secret's permission block drifts from the example — refresh consciously.

## Renaming the agent

If "mirrobot" isn't your bot's name, change exactly these:

1. **`BOT_NAMES_JSON`** in every workflow's env block — the two identity strings: your account login and your App bot login (`["mybot", "mybot[bot]"]`). Case-insensitively matched everywhere downstream.
2. **Mention routing** — `.github/scripts/route-comment.sh` matches `@mirrobot(-agent)?`; change to your name(s).
3. **The worker** (if guest mode) — its mention-regex and allowlist docs: [tools/mention-worker/](../tools/mention-worker/README.md).
4. **Prompt prose** — parts refer to the agent by name; a sweep of `mirrobot` → your name in `parts/` keeps the voice consistent. The *name vs identity* distinction (a name is a name; identities are the two logins) survives any rename as long as `BOT_NAMES_JSON` holds the real logins.
5. Commit footer attribution (git identity) is derived from the account/App automatically — nothing to edit.

The battery pins reference `mirrobot-agent` — update `prompt-rule-fixtures.sh`/`scrub-fixtures.sh` pins in the same commit (they exist to catch exactly this kind of missed sweep).

## Trigger words

Routing lives in one place — `.github/scripts/route-comment.sh` — and the stub's label gate is the literal `Agent Monitored` in `pr-review-trigger.yml`. Changing `/mirrobot-review` to `/review` is a one-file edit plus its fixture pins (the routing matrix is pinned). The `@name` mention matching is deliberately loose (substring, same as the original guards) — tighten it in `route-comment.sh` if you'd rather.

## Adding a whole new mode

If you need a genuinely new agent kind (say, a release-notes writer): add a `mission-*` part, a manifest naming it plus the shared parts, a workflow modeled on the existing dispatch-only ones (bot-setup → context fetch → assembly → session → verify), a router matrix row in `route-comment.sh`, and fixture pins for the new manifest. The stub/gate/router need no changes — dispatch is generic.

## What NOT to customize per-repo

The scrub, the router mechanics, the share filter, the fixture suite, bot-setup. These are consistent across repos on purpose — they're the security boundary, and per-repo forks of a security boundary rot. If something in them doesn't fit your repo, that's a platform change: make it here, carry it everywhere.
