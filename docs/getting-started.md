# Getting started

This guide takes you from "I just found this repo" to "the agent is working in my repository" — and tells you exactly which parts are yours to change and which to leave alone. Budget about 10 minutes of clicking for the setup itself.

If you want to understand *how* the platform works before installing it, read [architecture.md](architecture.md) first. If you know what you're doing and just want the knob reference, that's [configuration.md](configuration.md).

## What you end up with

After setup, your repository has an agent that:

- reviews every pull request automatically (and re-reviews on push, if you label the PR `Agent Monitored`),
- answers questions and takes instructions when mentioned in any issue or PR (`@mirrobot`),
- runs a final compliance audit before merge (`/mirrobot-check`) and blocks merging until it passes,
- triages newly opened issues with duplicate detection and labels,
- optionally follows you into *other* repositories when you mention it there (cross-repo guest mode — opt-in, see [workflows/mention-poller.md](workflows/mention-poller.md)).

## Step 0 — copy the platform in

Copy this repo's `.github/` directory, `decrypt_share_link.py`, and `tools/mention-worker/` into your repository, on the default branch. That is the whole platform: workflows, prompts, scripts, and the composite action that assembles everything per run.

> Forking the whole repository works too — but don't carry this repo's docs and plans into yours; you only need the paths above.

## Step 1 — pick an identity

The agent speaks to GitHub as exactly one of two shapes. Decide now, because it decides which secrets you'll add:

| | Bot account (recommended) | GitHub App |
|---|---|---|
| What it is | A normal user account you create for the bot | A GitHub App registration |
| Posts as | `your-bot-name` | `your-bot-name[bot]` |
| Token | One classic PAT, `public_repo` scope only | Per-run installation tokens minted from the App key |
| Cross-repo mentions | Works (add `notifications` scope when you enable guest mode) | Not supported for guest mode |
| Effort | Create account, invite as collaborator | Register App, set ~4 permissions, install on the repo |

Either way, the account/App needs **Write** on the repository (direct collaborator invite works; the poller worker setup covers the edge cases).

If you rename anything, read [customization.md](customization.md#renaming-the-agent) — identity is matched case-insensitively in a couple of places you'll want to update.

## Step 2 — add secrets (three entries minimum)

`Settings → Secrets and variables → Actions → Secrets`:

| Secret | What |
|---|---|
| `OPENCODE_MODEL` | Main model in `provider/model` format, e.g. `anthropic/claude-sonnet-4` |
| `OPENCODE_CONFIG_JSON` | Your complete [OpenCode](https://opencode.ai/docs/config) config — providers, API keys, permissions — minified to one line. Start from the committed template: `.github/actions/bot-setup/permissions.example.json` |
| `ACCOUNT_GH_TOKEN` *or* `BOT_APP_ID` + `BOT_PRIVATE_KEY` | The identity pair you picked in step 1 |

```bash
gh secret set OPENCODE_MODEL       -R <owner>/<repo> --body "anthropic/claude-sonnet-4"
gh secret set OPENCODE_CONFIG_JSON -R <owner>/<repo> < config.min.json
gh secret set ACCOUNT_GH_TOKEN     -R <owner>/<repo>   # paste when prompted
```

Every optional secret — fast model, share-link key, and the full identity rules (PAT scope requirements, why the `workflow` scope is forbidden) — is documented in [configuration.md](configuration.md#secrets).

**Or skip the next step's lookup entirely:** run **Agent Bootstrap** once (`Actions → Agent Bootstrap → Run workflow` — admin only). It creates every tuning variable with a safe default and prints the full secrets checklist into the run summary. It never overwrites anything, and its logs can't reveal what exists (see [workflows/agent-bootstrap.md](workflows/agent-bootstrap.md)).

## Step 3 — gate merges (recommended)

For the compliance audit to actually *block* merges, make it a required status check:

1. `Settings → Branches → Branch protection rule` for your default branch.
2. Require status checks to pass → search for `compliance-check` → check it.
3. While you're there: require a pull request before merging (the agent's reviews read better as suggestions on PRs), and **do not** add a "Restrict updates" rule with admin-only bypass — it deadlocks bot merges entirely (learned the hard way).

## Step 4 — say hello

Open an issue or PR and write:

> `@mirrobot what does this repository do?`

Within about a minute you should see a 👀 reaction, then a reply. On a PR, the review chain fires on its own — you'll see `PR Review Trigger` decide, `PR Review` run, and a pending `compliance-check` status appear.

If nothing happens, check `Actions` in order: **Agent Router** ran (comments) or **PR Review Trigger** ran (PR events) → it dispatched a target → the target ran. Each workflow doc has a "when it goes red" section that names what its failures mean.

## What's yours vs. what's machinery

| Yours to edit | Where | Notes |
|---|---|---|
| Compliance watch-list | `FILE_GROUPS_JSON` in `compliance-check.yml` | Which file groups the final audit checks — this is repo-specific by design. See [customization.md](customization.md#the-compliance-watch-list) |
| Agent behavior / personality / rules | `.github/prompts/parts/*.md` | All prompt prose lives here. See [customization.md](customization.md#prompts) |
| Permission profile | inside your `OPENCODE_CONFIG_JSON` secret | What the agent may execute. Template in `permissions.example.json` |
| Models per agent | `AGENT_MODELS_JSON` variable | [configuration.md](configuration.md#variables) |
| Identity strings | `BOT_NAMES_JSON` in each workflow env block | Only when renaming. [customization.md](customization.md#renaming-the-agent) |
| Everything else | — | Machinery. It's consistent across repos on purpose — resist forking it per-repo unless you mean to maintain that fork. |

## Cost expectations

Every agent session is one `opencode run` against your provider. Typical events cost: an issue triage or Q&A reply is a short session; a PR review scales with diff size (the reviewer navigates the diff rather than ingesting it whole); compliance audits are cheap. Each run prints its own token usage (`opencode stats`) into the Actions step summary — check it after a few days and tune `AGENT_MODELS_JSON` if one agent dominates your bill.

## Where to go next

- [architecture.md](architecture.md) — what executes where, the life of a PR, the trust model
- [configuration.md](configuration.md) — every variable and secret, with examples
- [security.md](security.md) — the threat model this platform is built against (worth reading before you open it to strangers)
- [workflows/](workflows/) — one page per workflow
