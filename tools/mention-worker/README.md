# mention-worker

The cross-repo "ears" of the Mirrobot platform, hosted OFF GitHub Actions:
a Cloudflare Worker (free tier) that polls the bot account's notifications
with conditional requests at an adaptive 30–60s cadence, pre-filters what
it relays, silences noisy threads, and wakes GitHub Actions only for
genuine, allowlisted summons.

## Architecture

One JavaScript file (`worker.js`) + `wrangler.toml`, scheduled by a
**self-rescheduling Durable Object alarm** (the `Scheduler` class — not
Cron Triggers): each wake-up commits the NEXT alarm first, then polls — no
error can ever kill the loop. Singleton via `idFromName("only")`. Why not
crons: on this account the platform scheduler registered crons but never
dispatched them once (2026-09-04, zero attempts in observability, matching
community reports 922869/928936; docs admit crons run "on underutilized
machines"). DO alarms bypass that machinery entirely and are free-tier
(SQLite-backed). No cron is attached (`crons = []`, explicit); the
one-line backup option is a comment in `wrangler.toml`.

**Conditional + adaptive polling** (measured against the endpoint's own
`X-Poll-Interval`, which reads 60 even on a quiet account): every real
response's `ETag` is persisted and sent back as `If-None-Match`, so idle
polls return **free 304s** ("leaving your rate limit untouched" — GitHub
docs). Cadence: **304 → 30s** (unless that response's header strains
>60s — honored, capped 120s); **200 → the header's allowance** (60s
typical, clamped 30–120s); **error/403/429 → 120s backoff**. Live-lesson:
GitHub omits `Last-Modified` on the EMPTY notifications response — the
ETag is always present and is the primary conditional.
`x-ratelimit-remaining` is logged on every response, so quota health is
permanently visible on the dashboard (observability enabled — every run
logs one structured `[poll] src=...` line, plus RELAYED/DECLINED/SILENCED/
JUNK audit lines, in Workers → Logs).

**Pre-filter gauntlet (deny-only, fail-open):** before relaying anything,
the worker fetches the triggering content (1 call) and checks bot-own
identity, the author/requester allowlist, and the genuine
`@mirrobot-agent` token. Declines are ACKED and **never wake Actions**
(strangers' mentions cost 2–3 API calls instead of a full run + runner
minutes). Any uncertainty (fetch error, roster unavailable) fails OPEN —
relay anyway; the in-repo gauntlet (`.github/scripts/handle-mentions.sh`,
battery-tested) re-verifies everything and stays the sole authority. A
fully compromised worker gains nothing: it could already choose not to
relay, and it still cannot make the agent act.

**Unsub-on-engage (thread noise killed at the source):** the subscription
endpoint is keyed by **NOTIFICATION thread id**
(`/notifications/threads/{id}/subscription` — NOT repo/issue number; the
wrong path 404s for everything and led to a false "no lever exists"
conclusion during testing). Implicit participation IS a subscription row
(`subscribed: true`, reason inherited from the mention). Live-verified
semantics: after unsubscribing, **plain follow-up comments deliver nothing
at all**, while **real @mentions ALWAYS break through** ("muted until you
comment or get @-mentioned once more"). The worker silences every thread
it handles — declined or relayed — one call per engagement; the bot's own
reply re-subscribes it, and the next engagement re-silences. The token
pre-filter remains as defense-in-depth for the seconds before the silence
lands.

**Junk hygiene:** every non-qualifying notification (repository
invitations, `ci_activity`, etc.) is ACKED — otherwise zombie unread
entries accumulate and eventually push real mentions out of the
30-per-page window.

## Notification reasons — measured, not assumed (2026-09-04 matrix)

Empirically verified on the account's own notifications (worker paused,
events fired one by one, delivery observed via `op=peek`):

| Event | Delivered? | Reason label |
|---|---|---|
| @mention anywhere (fresh thread, body or comment) | yes | `mention` |
| @mention in a thread the bot already replied in | yes | `mention` |
| Plain follow-up comment, thread the bot REPLIED in | yes (until silenced) | `mention` (sticky label) |
| Plain follow-up comment, mentioned-but-never-replied thread | **no** | — |
| Review-request button | yes — **collaborator-gated** (API 422 otherwise) | `review_requested` |
| Collaborator invitation | yes | `subscribed` (type RepositoryInvitation) |

Structural findings: delivery is **participation-gated** and labels are
**sticky** (follow-ups in replied threads arrive labeled `mention`, so no
reason filter can separate them — the body token check is the only
discriminator, and unsub-on-engage removes the noise entirely);
**review-requests are collaborator-gated** — in pure guest repos the
review button cannot reach the account at all, the @mention flow is the
only summon path there; **one notification per thread** (new activity
reactivates the same id).

## Secrets (wrangler secret put ...)

- `BOT_PAT` — the bot account's classic PAT (`public_repo` +
  `notifications` scopes). Polls, content fetches, roster, acks, silences.
- `DISPATCH_PAT` — fine-grained PAT on the platform repo ONLY:
  **Contents: Read-and-write** (repository_dispatch) +
  **Variables: Read-only** (reads the `FOREIGN_MENTIONS_USERS` repo
  variable; note: fine-grained *Actions: read* does NOT cover Actions
  variables — Variables is its own permission entry).
- `TEST_TOKEN` — random hex (`openssl rand -hex 24`); gates `/__tick`
  and every maintenance op.

Env (`wrangler.toml` vars): `PLATFORM_REPO` — "owner/name" of the repo
running `mention-poller.yml`.

## Setup (~10 minutes, once)

Prerequisites: a free [Cloudflare account](https://dash.cloudflare.com)
(no card) and Node.js LTS.

1. **Dispatch token** (GitHub → Settings → Developer settings →
   Fine-grained tokens → Generate): repository access = **only the
   platform repo**; permissions = **Contents: Read and write** +
   **Variables: Read-only**.

2. **Deploy**:
   ```
   cd tools/mention-worker
   npx wrangler login            # opens a browser, authorize Cloudflare
   npx wrangler secret put BOT_PAT        # classic PAT (public_repo + notifications)
   npx wrangler secret put DISPATCH_PAT   # the fine-grained token from step 1
   npx wrangler secret put TEST_TOKEN     # openssl rand -hex 24
   npx wrangler deploy
   curl "https://<worker>.workers.dev/__tick?key=$TEST_TOKEN"   # arms + first poll
   ```

3. **Verify**: `npx wrangler tail` shows `[poll] src=alarm idle-304 ...`
   every ~30s; the dashboard (Workers → Logs) accumulates the same lines
   with invocation context.

4. **Enable the platform side**: repo variable `FOREIGN_MENTIONS_ENABLED=true`
   (+ optionally `FOREIGN_MENTIONS_USERS`).

## Maintenance control (all gated by TEST_TOKEN)

```
W="https://<worker>/__tick?key=$TEST_TOKEN"
curl "$W"                          # one poll now + arm + diagnostics
curl "$W&cmd=1&op=status"          # paused? + when the alarm next fires
curl "$W&cmd=1&op=arm"             # ensure the alarm loop is running
curl "$W&cmd=1&op=disarm"          # cancel the alarm entirely (no ticks)
curl "$W&cmd=1&op=pause"           # maintenance: alarm ticks, never polls
curl "$W&cmd=1&op=resume"          # end maintenance mode
curl "$W&cmd=1&op=snooze&sec=300"  # ONE-TIME delay of the next poll
curl "$W&cmd=1&op=peek"            # read-only notifications dump (no side effects)
curl "$W&cmd=1&op=ack&id=N"        # mark one notification thread read (notification id)
curl "$W&cmd=1&op=unsub&tid=N"     # delete the account's thread subscription (notification id)
curl "$W&cmd=1&op=sub&tid=N"       # inspect a thread subscription row
curl "$W&cmd=1&op=setsub&tid=N&sub=true|false&ignored=true|false"
```

Semantics: **pause** keeps ticking (the loop survives, logs `PAUSED`,
costs nothing but the tick) until resumed — open-ended maintenance.
**snooze** pushes just the NEXT poll N seconds out, then the normal
adaptive cadence resumes on its own. **disarm** stops the loop completely;
**arm** bootstraps it again. Ops compose: a snooze while paused simply
moves the alarm; pause still suppresses the poll.

## Limits & costs (free tier)

- DO requests: 100,000/day included; this worker uses ~2,880 alarm wakes
  + a few subrequests each — far inside every limit.
- GitHub quota: idle polls are free (304); active cycles cost the real
  request + per-item fetch/ack; even 100 noisy comments/hour before
  silencing would be ~4% of the PAT's 5,000/hr.
- `tools/cron-probe/` (optional): a canary worker with a cron attached —
  if it ever starts firing, the platform's cron machinery healed.

## Secret rotation

If any token is rotated, re-run the corresponding
`npx wrangler secret put <NAME>` — no redeploy needed (secrets hot-swap).
Note `BOT_PAT` lives outside GitHub (a Cloudflare secret) — the
deliberate zero-infra trade-off of this feature; rotation is one command.
