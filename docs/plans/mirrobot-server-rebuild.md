# Mirrobot Server v2 — Rebuild Plan

**Date:** 2026-09-04
**Status:** Approved plan (this document is the build contract)
**Supersedes:** everything in `reference/` (the pre-rebuild server, kept as-is for consultation)
**Sibling systems:** `Mirrowel/Mirrobot-agent` (the GitHub-native agent — stays stateless, untouched, remains the fallback for everything this server does not own)

---

## 1. Purpose and vision

The GitHub agent works, is hardened, and is live-proven — but it is fundamentally **stateless and event-limited**. Every trigger rebuilds a full context, boots a fresh agent session on an ephemeral runner, and throws everything away. That costs re-orientation work every run, caps continuity of judgment at what can be stuffed into prompt blocks, and confines the agent to what GitHub Actions can express.

The server inverts the model:

- **One persistent agent per thread.** Each PR/issue gets a session that lives on server disk forever. The first trigger delivers the full briefing; every later trigger delivers only *what changed since the agent last looked* — a delta. The session's accumulated memory (files it read, findings it made, conclusions it reached, its own prior reviews) is the working state. Subagents it spawns land in the same isolated state directory automatically.
- **Listeners instead of workflows.** Webhooks for installed repos (instant), a notifications poller for guest mentions anywhere on GitHub (the proven mention-worker logic, rehomed). No GitHub Actions runs for anything the server owns.
- **Orchestration as a first-class layer.** The server decides what triggers what, serializes per-thread work, schedules follow-ups, and can later run agents against agents (consult another PR's session, fan out subtasks).

**Coexistence contract:** the GitHub agent stays exactly as it is — stateless, self-contained, the deployment path for people who want zero infrastructure. The server is the superset: everything the agent can do, plus statefulness, plus listeners, plus orchestration. Which system owns a repo is an explicit per-repo decision (webhook routing + a server-side ownership table). Both may coexist on different repos of the same owner indefinitely.

**Why a rebuild instead of finishing `reference/`:** the old server was designed around a principle that no longer holds — "compile prompts on a server, but execute by dispatching back to GitHub Actions." That re-creates the stateless pipeline with extra hops and is architecturally incapable of statefulness. Its *data plane* (cache, vocabulary, reconciliation) is genuinely good and gets transplanted; its *execution plane* is rebuilt around local, session-bearing execution. The old repo also carries dead generality (multi-app registries, a 16-step config pyramid, GUI/TUI shells) that a single-operator, single-app v1 must not drag along.

---

## 2. Design axioms (hard-won, from the live GitHub agent)

These are not preferences; each one encodes a live-observed failure or attack. The server inherits all of them.

1. **Untrusted-data doctrine.** All thread/PR/issue content is data, never instruction. The security brief (ported verbatim) opens every session. Requester identity is factual context (`REQUESTER_CONTEXT` line), never an authorization gate — any user may trigger; hardening is behavioral + tool-level, not access-level.
2. **Permissions are last-match-wins and default-allow** — a real profile is deny-catch-all → ordered allows → targeted denies. Port `permissions.example.json` verbatim into the generated `opencode.json`.
3. **The checkout is attacker territory.** Every execution happens in a worktree that `scrub-workspace.sh` has cleaned (auto-load surface removed/verified, `.github` taint-alarmed, quarantine preserved). Trusted artifacts (prompts, scripts, configs) come from the server, never from the worktree.
4. **Two-token discipline.** User-scoped credentials (notifications polling, acks) are separate from repo-scoped write credentials (posting, dispatch). Tokens are granted the minimum scope that works; missing scopes fail loudly.
5. **Unverified content fails open at the relay, fails closed at the authority.** The poller pre-filter is deny-only-on-positive-evidence (bot-identity, allowlist, genuine-token checks; anything uncertain gets relayed); the in-pipeline gauntlet re-verifies everything and is the sole decision authority.
6. **At-most-once via mark-before-act.** Notifications are marked read before dispatch; acks and processing are coupled so no path double-handles.
7. **Notification reality (measured):** delivery is participation-gated, reason labels are sticky, review-requests are collaborator-gated (a 422 for strangers — @mention is the only pure-guest summon), one notification per thread (same id reactivates). After handling a thread: **unsub-on-engage** (plain follow-ups stop delivering; real mentions always break through; the reply re-subscribes; the next engagement re-silences).
8. **Rate discipline.** Conditional requests (ETag/`If-None-Match`) make idle polls free; adaptive cadence 304→30s, 200→header allowance (30–120s clamp), error→120s; `x-ratelimit-remaining` logged on every call. Junk notifications are acked, never left to zombify the unread window.
9. **Prompts are parts + manifests, battery-pinned.** Prose lives once; modes are assembly lists; every load-bearing rule has a fixture pin so drift cannot ship silently.
10. **Anti-noise is a feature, not cosmetics.** Hidden (minimized) content is excluded everywhere; AI-reviewer noise is filtered by author + pattern; reviews are the deliverable (no announcement comments); acks are living and progress-bearing.
11. **Severity and verdicts are the agent's vocabulary** (🔴🟠🟡🔵; APPROVE requires a positive repository purpose, rank-blind; "approve-but-fix" is incoherent and banned; COMMENT must state what's missing for approval).
12. **Share links are secrets.** Session share URLs are captured from the output stream, masked, RSA-OAEP-encrypted (MRB1 format), delivered via annotation/summary/log; recovery is admin-only (`decrypt_share_link.py`).
13. **Identity is dual-mode everywhere.** App installation tokens and the account PAT are interchangeable behind one interface; identity matching is case-insensitive against exactly two identities; "mirrobot" is a name, not an identity.
14. **Signals are honest.** Runs that decline skip visibly; runs that fire succeed; errors fail. Workflow/run naming states the actual source ("Automated mention relay from X (mention)", "Manual poll by @Y").

---

## 3. Architecture overview

```
┌─ INGEST ──────────────────────────────────────────────────────────┐
│  webhook listener (installed repos, instant)                      │
│  notifications poller (guest mentions anywhere; worker port)      │
│  reconciliation sweep (high-water-mark catch-up after downtime)   │
└──────────────┬────────────────────────────────────────────────────┘
               ▼  normalized events (dedup'd, HMAC'd / gauntlet'd)
┌─ STATE ───────────────────────────────────────────────────────────┐
│  omniscient cache (SQLite: issues, PRs, comments, reviews,        │
│    threads, reactions, commits, labels, cross-refs)               │
│  session store (data/sessions/<repo-slug>/<thread>/ — one         │
│    persistent opencode state dir per thread; forever)             │
│  session registry (thread-key → session id, cursor: last SHA,     │
│    last comment id, last event watermark)                         │
└──────────────┬────────────────────────────────────────────────────┘
               ▼
┌─ COMPILE ─────────────────────────────────────────────────────────┐
│  vocabulary builder (cache-first, live-API fallback; incremental  │
│    diffs; filtered discussion; noise suppression)                 │
│  prompt assembler (parts + manifests, ported from the agent repo) │
│  delta composer (first turn = full brief; later = "since you      │
│    last looked: …")                                               │
└──────────────┬────────────────────────────────────────────────────┘
               ▼
┌─ EXECUTE ─────────────────────────────────────────────────────────┐
│  orchestrator (rules → queue → per-thread serialization → run)    │
│  local runner: worktree at event SHA → scrub → generated          │
│    opencode.json → opencode run [--session id] in isolated        │
│    XDG_DATA_HOME → stream filter (share-link capture)             │
└──────────────┬────────────────────────────────────────────────────┘
               ▼
┌─ POST ────────────────────────────────────────────────────────────┐
│  posting identity (App installation token / account PAT)          │
│  reviews + inline comments (severity, verdict semantics),         │
│  comments, reactions (lifecycle + discretion), living acks,       │
│  compliance statuses (WARNINGS→success+description,               │
│    BLOCKED→failure), unsub-on-engage                              │
└───────────────────────────────────────────────────────────────────┘
```

---

## 4. Subsystems — what, how, why

### 4.1 Ingest

**Webhook listener.** FastAPI endpoint receiving GitHub webhooks for repos where the App is installed. Per-app HMAC secret validation, event id dedup (the `processed_events` table pattern from `reference/`), payload ingestion into the cache, then rule evaluation. Webhooks are the instant path for installed repos — the whole reason to prefer them over polling is latency (seconds vs the poller's ~30–60s floor) and zero rate-limit spend.

**Notifications poller (guest path).** Webhooks cannot fire where the App is not installed — so guest mentions (the account @mentioned in any public repo) arrive only through the account's notifications. The poller is a direct port of the live-proven mention-worker loop into the server's asyncio scheduler: conditional ETag polling, adaptive cadence, the deny-only pre-filter gauntlet (bot-identity → allowlist [collaborators ∪ `FOREIGN_MENTIONS_USERS`, variable read is optional: 404=silent-absent, 403=fail-open] → genuine-token check), mark-read-before-relay, unsub-on-engage after every handled thread, junk acking. *Why port rather than keep the Worker:* once the server owns the pipeline, a second execution of the same logic in a second runtime is a drift factory; the worker remains the documented fallback for server downtime (they share the at-most-once discipline, so running both is safe — whichever handles first wins, the other no-ops).

**Reconciliation sweep.** Periodic + on-wakeup high-water-mark sync per installed repo: comments, reviews, review threads (GraphQL), commits, labels, reactions, cross-references. Covers webhook misses during server downtime — the catch-up engine from `reference/`, kept nearly as-is. *Why it matters:* the server's authority is its cache; the cache must converge on truth without webhooks.

### 4.2 State

**Omniscient cache.** SQLite (aiosqlite), the 14-table schema + repo classes + staleness columns from `reference/` (`detail_synced_at`, `comments_synced_at`), transplanted with its tests. *Why SQLite and not something grander:* one operator, one disk, zero-ops, and the proven read patterns (cache-first with live fallback) already exist against it.

**Session store — the heart of statefulness.** One directory per (repo, thread): `data/sessions/<owner>__<repo>/<pr-N|issue-N|disc-N>/`. Every agent run for that thread executes with `XDG_DATA_HOME=<that dir>/xdg` — opencode writes its *entire* data dir there: the main session, every subagent session (they are ordinary sessions with a `parentID`), diffs, todos, whatever storage layout the installed opencode version uses (SQLite or JSON trees — the redirection is version-agnostic). *Why this mechanism specifically:*
- It captures the **whole agent**, not just one session's files — subagents, artifacts, everything, with zero extraction logic to break on opencode upgrades.
- It **isolates** threads from each other (no shared global db, no cross-thread leakage).
- **Fork = copy the directory** (and git history of the store gives time-travel for free if the store is itself a git repo — optional, later).
- Sessions persist **forever** on server disk — no cache TTLs, no eviction; the 7-day ceiling that killed the Actions-cache approach does not exist here.

**Session registry.** A `sessions` table: thread-key → session id, cursor state (last seen head SHA, last processed comment id, last event watermark), created/updated timestamps, turn count. The cursor is the delta composer's ground truth. A cache/registry miss degrades gracefully to a fresh full-brief run — identical to today's stateless behavior.

### 4.3 Compile

**Vocabulary builder.** The crown jewel of `reference/` (681 LOC, 23 tests): cache-first context objects with live-API fallback — PR/issue/thread/repo/event contexts, filtered discussion (hidden=minimized excluded, noise patterns applied), incremental diffs since the agent's last look (cursor-driven, replacing the marker-scraping the GitHub agent does). Transplanted and extended with: reason-matrix-aware notification context, roster/allowlist resolution, and the same identity-filtering rules the agent's `fetch-pr-discussion.sh` implements (three-block structure: elevated own-latest, filtered own-history, correlated thread).

**Prompt system — ports, not rewrites.** The agent repo's `parts/ + manifests/ + assemble-prompt.sh` are the source of truth (19 parts, 13 manifests, fail-closed assembly). The server vendors them (a sync script copies the directory at build time; the agent repo remains canonical until the server is the primary). Modes map to manifests exactly as on GitHub. *Why verbatim:* the prose is battery-pinned and live-proven against real attacks; divergence would forfeit that. New server-only parts: a `stateful-delta-turn` part (resumed-session framing: "this is an update since you last looked; act on the delta; trust freshly fetched context over memory"), and an `orchestration` note describing the tools the server provides.

**Delta composer.** First trigger on a thread → full brief (assembled prompt, exactly today's shape). Later triggers → short message: "Since you last looked: +N commits (incremental patch), new comments [filtered], new reviews/events, label changes." Generated from the session cursor + vocabulary. *The honest trade-off, stated once:* the model API re-sends session history every turn regardless — the wins are (1) the agent stops re-orienting (no re-reading files, re-running commands, re-deriving conclusions), (2) per-turn new content is a delta instead of a full re-fetch, (3) judgment continuity. Compaction (opencode's built-in) bounds the riding history; per-run token usage is logged so the curve is observed, not assumed.

### 4.4 Execute

**Orchestrator.** The rules engine (event → matching rule/mission, conditions on labels/type/author) survives conceptually from `reference/` but is rebuilt on a **single rule model** (no v0.1/v0.2 duality): a mission registry mapping (thread-type, trigger) → manifest + execution policy (stateless | stateful, posting identity, session key). Per-thread serialization: a queue keyed by (repo, thread) so a PR never has two concurrent agent turns (the concurrency-group discipline from the workflows, but in-process and exact).

**Local runner.** For each run:
1. `ensure_bare_clone` (the `reference/` git-manager, now with token-authenticated cloning) → `create_worktree(event_sha, tmp)`.
2. Write trusted artifacts into the worktree/scratch: assembled prompt, generated `opencode.json` (the permission profile + model config — server-side secrets, injected as env; the worktree contains **no** credentials beyond the run's scoped token), `scrub-workspace.sh` + kit scripts.
3. Run scrub in the worktree (trusted-anchor compare against the platform repo's main; foreign repos → `--foreign` mode: remove-all auto-load with quarantine).
4. `opencode run [--session <id>] --share -` in the worktree cwd, `XDG_DATA_HOME` pointed at the thread's session dir, `2>&1` through the **stream filter** (share-link capture/mask/encrypt — the MRB1 pipeline; the decryptor stays admin-side).
5. Parse structured output (`--format json` events): session id (upsert into the registry), token usage, artifacts, the agent's posting intents.
6. *Why worktrees instead of a persistent checkout:* event SHAs must be reproducible and isolated; bare clones give cheap diff generation (vocabulary) and exact-SHA worktrees give execution fidelity.

**Posting (also under POST, listed here for flow):** see 4.5.

**Subagents & orchestration.** The agent's own task-tool subagents run inside the same XDG dir → automatically persisted with the thread. Server-level orchestration (v2+, explicitly deferred): a `consult-session` tool letting one thread's agent ask a question of another thread's session (a one-shot `opencode run --session <other>` whose answer returns as a tool result); fan-out task graphs. *Why deferred:* v1 must prove single-session statefulness first; cross-session coupling adds interference modes (a slow consult blocking a turn) that deserve their own design pass.

### 4.5 Post

**Identity.** Dual-mode behind one interface, exactly like the agent: App installation tokens for installed repos (also powering webhooks), the account PAT for guest repos (posting as a plain public user — the containment model: collaborator nowhere except home). Scope validation at mint time, fail-loud.

**Actions.** Reviews with inline comments (severity prefixes, verdict-line + justification, footer markers for review-type detection — ported semantics), comments (body-file discipline for formatting fidelity), reaction lifecycle (workflow-owned eyes/rocket/confused + agent-discretion reactions), living acks (progress-editing), compliance statuses (`WARNINGS → success + warning description + report link; BLOCKED → failure` — the neutral-422 lesson), unsub-on-engage, cc-on-requested-review only, zero-extra-comments in review mode.

**Signals.** Every run is recorded in the DB with its trigger source, decision (fired/declined + rule), latency milestones (the measured-milestone discipline: trigger→detect→session-start→reply), token usage, and cost ticks where available.

### 4.6 Security model summary

| Layer | Mechanism |
|---|---|
| Input | HMAC webhook validation; poller gauntlet (deny-only, fail-open); untrusted-data doctrine in every prompt |
| Execution | Permission-profiled opencode; scrubbed worktrees; trusted artifacts server-side only; XDG session isolation |
| Credentials | Two-token discipline; generated per-run tokens; no secrets in worktrees; secrets in server config (`secrets/`, never committed) |
| Posting | Scope-validated identities; rate discipline; at-most-once acks |
| Recovery | Share-link encryption; full run logs; quarantine of removed auto-load files |

---

## 5. What `reference/` contributes (transplant manifest)

| Component | Disposition |
|---|---|
| `db/` (models, repos, database) + tests | **Transplant** as-is |
| `sdk/vocabulary.py` + tests | **Transplant** + extend (reason-aware contexts, cursor-driven incremental) |
| `sdk/compiler.py`, `sdk/filters.py` | **Transplant** (Jinja2 stays for *vocabulary rendering*; the parts/manifests system handles mission prose) |
| `github/webhooks.py` router/validator + tests | **Transplant**, re-wired to the new orchestrator |
| `github/reconciliation.py` | **Transplant** + add mocked-API bulk tests (the map flagged this gap) |
| `github/auth.py`, `client.py` | **Transplant** (collapse multi-app registries to single-app; keep the interface for future multi-app) |
| `git/manager.py` diff half + tests | **Transplant** + token-authenticated cloning |
| `git/manager.py` worktree half | **Adopt** as the local runner's primitives (it was built for exactly this and never wired) |
| `scheduler/` | **Transplant** (asyncio loop gains: poller task, wake-from-sleep catch-up trigger) |
| `modules/pr-review` prompt DNA | **Reference only** — superseded by the agent repo's parts (full fidelity) |
| Config pyramid (16-step, type/app modifiers), `modules/` rule duality, GUI, TUI, multi-app registries, apscheduler/alembic deps | **Drop** — dead generality; v2 uses: settings.yaml + per-repo overrides + the mission registry |
| The "dispatch to GitHub Actions" execution path | **Drop** — replaced by the local runner; a `dispatch` execution target may return later as an overflow mode |

---

## 6. Build phases and gates

**Phase 0 — Scaffold.** Fresh package layout (`src/mirrobot/`), pyproject, settings schema (single app, secrets paths, repos table, tunnel URL), structlog logging, CLI (`mirrobot check|serve|sync`), transplant DB layer. *Gate:* transplanted tests green.

**Phase 1 — Ingest.** Webhook listener (HMAC, dedup, ingestion) + reconciliation sweep + event normalization. *Gate:* a real webhook from the pilot repo lands in the cache; a forced full sync reproduces repo state.

**Phase 2 — Compile.** Vocabulary + git manager + prompt vendor-sync (script copying the agent repo's parts/manifests + scrub + kit scripts) + assembler + delta composer (stateless path first). *Gate:* compiled prompt for a pilot PR byte-matches the GitHub agent's equivalent mode (modulo context-freshness deltas).

**Phase 3 — Execute (stateless parity).** Local runner end-to-end: worktree, scrub, profile, `opencode run`, stream filter, output parse, posting (review/comment/reaction). *Gate:* `/mirrobot-review` on the pilot repo produces a real posted review with zero GitHub Actions involvement; latency milestones recorded.

**Phase 4 — Statefulness.** Session store + registry + `--session` resume + delta turns + token accounting. *Gate:* two consecutive triggers on one PR demonstrably resume the same session (agent references its own prior findings without re-fetching); kill-and-restore of the session dir works; fork-by-copy works.

**Phase 5 — Guest mode + full missions.** Poller port with the complete gauntlet + unsub-on-engage; bot-reply/compliance/issue missions at full fidelity; compliance gate semantics. *Gate:* a guest mention on a foreign repo answered end-to-end by the server; compliance status posts correctly.

**Phase 6 — Hardening & operations.** Batteries ported (prompt-rule fixtures, permission-profile checks, scrub fixtures run against the vendored scripts); rate-limit dashboarding (the `[poll]` log discipline); backup story for `data/` (sessions + cache); run-naming equivalents in the server's own run records; documentation.

---

## 7. Deployment

- **Host:** the operator's PC initially; webhooks via a Cloudflare Tunnel (`cloudflared` → localhost:8080, free, stable HTTPS). The poller path needs no tunnel. Downtime is covered by reconciliation + the at-most-once acks; the mention-worker remains a documented fallback during outages.
- **Secrets:** `secrets/` (gitignored): App private key, account PAT, dispatch/variables PAT, TEST_TOKEN, share-link keypair. The share-link private key stays on the admin machine (already there).
- **Pilot:** `Mirrowel/mirrobot-guest-test` (already the live-test lab: issues, PRs, collaborator invites proven).
- **Ownership table:** explicit per-repo `owner: server | github-agent | both(poll-only)` — the router of last resort when both listen.

---

## 8. Testing strategy

- Unit tests per subsystem (the transplanted suites + new).
- **Battery philosophy ported:** fixture scripts asserting prompt contracts (every pinned rule survives assembly), permission profile shape, scrub behaviors — runnable in CI-equivalent on every change, like the agent repo's `scrub-fixtures.sh` + `prompt-rule-fixtures.sh`.
- Scenario tests against the pilot repo (real webhooks/mentions, disposable threads).
- A dedicated **statefulness test suite**: resume, delta correctness (cursor manipulated to known states), session isolation (no cross-thread file access), fork, restore.

## 9. Risks and open questions

- **opencode resume reliability** (community reports of `--continue` flakiness): Phase 4 gate tests `--session` specifically on the pinned version; version pinning is part of the scaffold.
- **Session growth / cost curve:** compaction bounds it; token logging makes it observable; escape hatch = per-thread fresh-restart policy after N turns.
- **Home-host availability:** accepted for v1 (reconciliation covers gaps); VPS migration is a config change, not a redesign.
- **Cross-session orchestration:** deferred by design (see 4.4).
- **Universal role consolidation** (one proactive agent per thread replacing per-mission roles): the natural end-state once statefulness proves out — a prompt-system redesign owned by this plan's follow-up, not blocking v1.
