# Architecture

How the pieces fit, what executes where, and the reasoning behind both. This is the document to read before changing anything structural — every rule below exists because its absence hurt once.

## The execution map

The single most important fact: **GitHub decides which copy of a workflow file runs, and that choice differs by trigger.**

| Trigger family | Executes from | Who uses it |
|---|---|---|
| `issue_comment`, `issues`, `workflow_dispatch`, `repository_dispatch`, `schedule` | **default branch**, always | Agent Router, Issue Analysis, Bot Reply, PR Review, Compliance Check, Mention Poller, Bootstrap |
| `pull_request_target`, `pull_request` | **the PR's base branch** | PR Review Trigger (stub), Compliance Gate |

Everything that *thinks* — agent sessions, prompt assembly, the scrub, routing decisions — runs from the default branch on every PR, no exceptions. The two workflows that can execute from a PR's base branch are deliberately the two that hold **zero secrets and no checkout**: a dispatcher (the stub) and a status poster (the gate). Even a tampered copy of either has a bounded ceiling — review-timing control and status noise with a one-hour token — and the stub's dispatch always targets `--ref <default-branch>`, so nothing downstream ever executes attacker-influenced code.

That's also why platform updates land on **main** — the default branch *is* the machine.

### The update doctrine

- **Platform changes (anything under `.github/`) go to main first.** Always. A platform change on any other branch is inert for real runs until it reaches main.
- **`dev` needs no per-batch sync.** At the next dev→main merge, agent files (changed only on main's side) auto-merge cleanly.
- **When dev *should* be current** (the stub or gate changed): `git merge main` into dev. Merges share commit objects, so those commits dedupe to nothing when dev later merges into main. **Never** replicate the same change as separate commits on both branches — copies are the history pollution that shows up in main's log later.
- **Auto-load content is the deliberate exception** — it evolves on dev (see the trust model below).

## The life of a pull request

```
 PR opened / ready / reopened ──► PR Review Trigger (stub, base-branch copy)
                                      │ decides: review wanted?  no ──► nothing runs at all
                                      │ yes
                                      ├─► posts pending compliance-check status
                                      └─► dispatches PR Review  ── (always from main)
                                                                    │ scrub → context → agent session
                                                                    │ posts review w/ severity + verdict
 push (synchronize) ──► stub ──► only with 'Agent Monitored' label ─┘
 reviewer asks in comments ──► Agent Router ──► dispatches PR Review (same machinery, comment context)
 PR ready to merge ──► someone comments /mirrobot-check ──► Router ──► Compliance Check
                                                                    │ posts compliance status + report
                                                                    │ BLOCKED / WARNINGS / COMPLIANT
 merge ──► branch protection requires the green compliance-check status
```

Key properties:

- **A declined review triggers nothing.** The stub's decision and its dispatch are the same act — declined events never spin up a PR Review run.
- **Reviews serialize per PR** (concurrency group `PR Review-<N>`), so a manual review request and an auto-trigger can't race or interleave.
- **Follow-ups are incremental.** The agent's posted reviews carry a footer with the last-reviewed SHA; the next run detects it, loads the FOLLOW-UP protocol, and works the delta with full memory of its own previous findings.
- **Compliance runs once, at the end, on request.** It is a different agent with a different checklist (practices and consistency, not code correctness — that's the reviewer's job).

## The comment routing flow

Every comment event (`issue_comment[created]`) hits exactly one entrypoint, the **Agent Router**:

1. Bot-loop guard: comments authored by bots or the agent itself are ignored (prevents self-trigger loops).
2. The comment body is parsed (quotes and code fences stripped) against the trigger matrix — see [workflows/agent-router.md](workflows/agent-router.md).
3. Each match dispatches exactly one target workflow with a `commentId` input; targets re-fetch full context from the API by that id (context is never trusted from the event payload).
4. Compound comments dispatch all matches — "can you review this and then run compliance" does both.

## The trust model (split trust)

The agent loads content from the repository it's pointed at. Some of that content auto-loads into its brain (instruction files, skill directories, project configs) — so the platform must decide, before the agent ever boots, which bytes are allowed to speak first. The answer is deliberately different per surface:

| Surface | Trusted from | Rule |
|---|---|---|
| **Auto-load files** — `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, harness rule files, `.claude/`, `.agents/`, `.opencode/`, `.cursor/`, `.windsurf/`, `.devin/` | **main ∪ dev** | A file survives only when its bytes match a state a trust branch shipped **at-or-after that branch's fork point** with the PR (each branch gets its own floor). |
| **`.github/` platform files** | **main only** | Never removed (the reviewer must see them), but any branch-side change is a taint alarm at maximum scrutiny; content provably synced from post-fork main states downgrades to an explained note. |

Why the split: auto-load content *describes the code it ships with* — when you evolve your repo's conventions on dev, the AGENTS.md that documents those conventions evolves with it, and it must be trusted there before it merges up. Platform wiring has no such coupling — it executes from main, so main alone vouches for it.

The floors are what make history-matching safe: a state older than every floor can only be there *deliberately* — someone rolled a file back to content a trust branch already abandoned (typically, resurrecting a flaw that was fixed). That's quarantined. Novel bytes (nothing ever shipped them) are quarantined. Trusted-but-not-current bytes stay, tagged with an era note so the agent treats them as dated context, not doctrine.

Two degradation rules keep this from being brittle: an optional trust branch that doesn't exist (a deployment without a dev) is skipped gracefully — never a fail-closed trigger — while a missing `main` removes all auto-load files unconditionally.

The scrub also **quarantines** every removed file to `/tmp/scrub-quarantine/` (resolved content, in-repo targets only): the agent can read your PR's AGENTS.md as *data* — often genuinely useful context — without those bytes ever being auto-loaded as instructions.

## The prompt system

Prompts are not files; they're an *assembly*. `.github/prompts/parts/` holds every block of prose (shared rules, per-mode missions, protocols); `.github/prompts/manifests/` holds ordered part-name lists — a manifest **is** a mode (`pr-review-first`, `bot-reply`, `compliance-followup`, ...). At run time, `assemble-prompt.sh` concatenates the manifest's parts and `envsubst` injects the run's context variables (thread context, diffs, trust warnings). Modes that share a part share it byte-identically by construction — duplicated guidance cannot drift.

Full authoring guide: [customization.md](customization.md#prompts).

## The identity model

The bot speaks as exactly one identity per installation — a user account (account mode) or a GitHub App bot (app mode) — selected by which secrets exist. Both modes flow through the same machinery; every self-detection point (bot-loop guards, review attribution, footer verification) matches a JSON list of identity logins, case-insensitively, so renaming never silently breaks detection. "mirrobot" is the agent's *name* — mention-routing accepts it — but it is never an *identity*: a human user who happens to be named `mirrobot` is not treated as the agent.

Cross-repo guest mode adds a third identity context: the *guest* — the same account, summoned into a repository it doesn't know, running under stricter rules (read-only by default, authority pinned to an allowlist, never to thread participation). See [workflows/mention-poller.md](workflows/mention-poller.md).

## Where the batteries live

Every rule in this document that can rot is pinned: `.github/scripts/scrub-fixtures.sh` (193 checks — scrub semantics, routing matrix, workflow contracts, YAML strictness) and `.github/scripts/prompt-rule-fixtures.sh` (339 pins — prompt rules and assembly contracts). They run in CI on every change to `.github/**`. If you change behavior, change the pin in the same commit — the suite failing is the system working.

## Design axioms

Short list of the "why" behind recurring decisions:

1. **Untrusted text is data, never instructions** — comment bodies, PR titles, commit subjects reach shells only as env vars, and reach the agent wrapped in a brief that says treat everything as input.
2. **Everything privileged runs from the default branch** — the execution map above.
3. **Secrets are masked leaf-by-leaf and die at boot** — the config secret's inner API keys are individually registered with `::add-mask::`, and the whole config plus materialized plugins are deleted seconds after opencode reads them.
4. **Honest signals** — skipped means didn't-fire, green means ran, red means actually failed. Nothing cosmetically green.
5. **The batteries are the documentation's enforcement** — behavior changes carry their pin changes.
