// mention-worker — cross-repo mention relay for the Mirrobot platform.
//
// WHY: the in-repo Mention Poller schedule costs a GitHub Actions run every
// few minutes even when nothing happened. This worker moves the polling off
// GitHub Actions: Actions then only run when a QUALIFYING, PRE-FILTERED
// notification actually exists.
//
// SCHEDULING: a self-rescheduling Durable Object alarm, NOT Cron Triggers
// (live-lesson 2026-09-04: the account's crons were registered but never
// dispatched; community reports 922869/928936; docs admit crons run on
// "underutilized machines"). The alarm loop cadence (measured against the
// endpoint's own X-Poll-Interval, which reads 60 even on a quiet account):
//
//   304 Not Modified (free - "leaving your rate limit untouched" per docs):
//       repoll in 30s (unless that response's header strains >60s, then
//       honor it, capped at 120s)
//   200 (real change):   repoll in clamp(header || 60, 30..120) - the
//       endpoint's stated allowance right after it did real work
//   error / 403 / 429:   back off to 120s and log loudly
//
// CONDITIONAL POLLING: each 200's Last-Modified is persisted in DO storage
// and passed back verbatim as If-Modified-Since, so idle polls are free.
//
// PRE-FILTER GAUNTLET (deny-only, fail-open): before relaying anything, the
// worker fetches the triggering content (1 call) and checks: bot-own
// identity, author/requester allowlist (platform-repo collaborators +
// FOREIGN_MENTIONS_USERS repo variable), and the genuine @mirrobot-agent
// token. Declines are ACKED (mark-read) and NEVER wake Actions. Any
// uncertainty (fetch error, roster/variable unavailable) fails OPEN: relay
// anyway - the in-repo gauntlet (.github/scripts/handle-mentions.sh)
// re-verifies everything and stays the sole authority. A compromised worker
// gains nothing: it could already choose not to relay, and it still cannot
// make the agent act.
//
// ALL trust decisions live in the repo. This relay pre-filters only to
// avoid burning Actions runs on junk (strangers mentioning the bot
// anywhere).
//
// GET /__tick?key=<TEST_TOKEN> runs one poll on demand (through the DO, so
// conditional state + caches apply), arms the loop, returns diagnostics.
//   &cmd=1&op=<arm|disarm|pause|resume|snooze&sec=N|status|peek|ack&id=N|
//    unsub&repo=O/R&n=N>  - the maintenance surface (see the DO /cmd docs)
//
// Secrets (wrangler secret put ...):
//   BOT_PAT      — bot account classic PAT (public_repo + notifications);
//                  polls, content fetches, roster, decline acks
//   DISPATCH_PAT — fine-grained PAT on the platform repo ONLY:
//                  contents:write (repository_dispatch) + actions:read
//                  (reads the FOREIGN_MENTIONS_USERS repo variable)
//   TEST_TOKEN   — random hex string gating /__tick
// Env (wrangler.toml vars):
//   PLATFORM_REPO — "owner/name" of the repo running mention-poller.yml

const GH = "https://api.github.com";
const BOT_IDENTITIES = ["mirrobot-agent", "mirrobot-agent[bot]"];
const REASONS = new Set(["mention", "review_requested", "subscribed", "comment"]);
const ROSTER_TTL_MS = 10 * 60_000;
const VAR_TTL_MS = 5 * 60_000;
const POLL_FLOOR_MS = 30_000;
const POLL_CAP_MS = 120_000;

function clampIntervalMs(headerSec, fallbackSec) {
  const sec = Number(headerSec) > 0 ? Number(headerSec) : fallbackSec;
  return Math.min(POLL_CAP_MS, Math.max(POLL_FLOOR_MS, sec * 1000));
}

function ghHeaders(token) {
  return {
    Authorization: `Bearer ${token}`,
    Accept: "application/vnd.github+json",
    "User-Agent": "mirrobot-mention-worker",
  };
}

async function gh(path, token, init = {}) {
  return fetch(path.startsWith("http") ? path : GH + path, {
    ...init,
    headers: { ...ghHeaders(token), ...(init.headers || {}) },
  });
}

function threadRef(n) {
  const m = (n.subject?.url || "").match(/\/(?:issues|pulls)\/(\d+)$/);
  return `${n.repository?.full_name || "?"}#${m?.[1] || "?"}`;
}

// ---- roster: collaborators (INCLUDES the repo owner) ∪ the optional
// FOREIGN_MENTIONS_USERS variable (additive). The variable is OPTIONAL:
// 404 (absent) is a legitimate configuration -> silent, roster stays
// collaborators-only and authoritative. 403/5xx (unreadable) -> roster
// possibly incomplete -> allowlist rule suspended for the cycle
// (fail-open relays; the in-repo gauntlet re-verifies authoritatively).
// A collaborators fetch failure also fails the roster open.
async function getRoster(env, state) {
  const cached = (await state.storage.get("rosterCache")) || {};
  if (cached.names && Date.now() - cached.ts < ROSTER_TTL_MS) {
    return { names: new Set(cached.names), ok: true, cached: true };
  }
  const names = [];
  let ok = true;
  try {
    const res = await gh(
      `/repos/${env.PLATFORM_REPO}/collaborators?affiliation=direct&per_page=100`,
      env.BOT_PAT,
    );
    if (!res.ok) throw new Error(`collaborators ${res.status}`);
    for (const c of await res.json()) names.push(String(c.login).toLowerCase());
  } catch (e) {
    console.log(`[poll] roster collaborators failed (fail-open): ${e}`);
    ok = false;
  }
  try {
    const res = await gh(
      `/repos/${env.PLATFORM_REPO}/actions/variables/FOREIGN_MENTIONS_USERS`,
      env.DISPATCH_PAT,
    );
    if (res.status === 404) {
      // variable not set: legitimate optional-absent config, not an error
    } else if (!res.ok) {
      throw new Error(`variable ${res.status}`);
    } else {
      const val = (await res.json()).value || "";
      for (const n of val.split(/[,;\s]+/)) if (n) names.push(n.toLowerCase());
    }
  } catch (e) {
    console.log(`[poll] roster variable unreadable (fail-open): ${e}`);
    ok = false;
  }
  if (ok) await state.storage.put("rosterCache", { names, ts: Date.now() });
  return { names: new Set(names), ok };
}

// ---- pre-filter for ONE notification. Deny-only, fail-open.
async function prefilter(env, state, n, roster) {
  try {
    // review_requested: authority = WHO clicked the button (timeline actor)
    if (n.reason === "review_requested") {
      const prUrl = n.subject?.url || "";
      const tl = await gh(`${prUrl}/timeline?per_page=100`, env.BOT_PAT);
      if (!tl.ok) return { pass: true }; // fail-open
      const events = await tl.json();
      let actor = null;
      for (const ev of events) if (ev.event === "review_requested") actor = ev.actor?.login;
      if (!actor) return { pass: true };
      const a = String(actor).toLowerCase();
      if (BOT_IDENTITIES.includes(a)) return { decline: "self" };
      if (roster.ok && !roster.names.has(a)) return { decline: "allowlist" };
      return { pass: true };
    }

    // mention/subscribed/comment: fetch the triggering content (last
    // comment when there is one, else the issue/PR body itself)
    const target = n.subject?.latest_comment_url || n.subject?.url;
    if (!target) return { pass: true };
    const res = await gh(target, env.BOT_PAT);
    if (!res.ok) return { pass: true }; // fail-open
    const content = await res.json();
    const author = String(content.user?.login || "").toLowerCase();
    const body = String(content.body || "").toLowerCase();

    if (BOT_IDENTITIES.includes(author)) return { decline: "self" };
    if (!body.includes("@mirrobot-agent")) return { decline: "token" };
    if (roster.ok && !roster.names.has(author)) return { decline: "allowlist" };
    return { pass: true };
  } catch (e) {
    console.log(`[poll] prefilter error (fail-open): ${e}`);
    return { pass: true };
  }
}

// Silence an engaged thread (unsub-on-engage). Live-verified semantics
// (2026-09-04 matrix, CORRECT endpoint /notifications/threads/{id}/subscription):
// implicit participation IS a subscription row (subscribed:true via the
// mention); deleting it stops delivery of plain follow-up comments in the
// thread ENTIRELY, while real @mentions ALWAYS break through afterwards
// ("muted until you comment or get @mentioned once more" - the bot's own
// reply re-subscribes it, and the next engagement re-silences). One call
// per handled thread per engagement cycle; kills the sticky-mention noise
// at the source. The token pre-filter stays as defense-in-depth for the
// window before the silence lands.
async function silenceThread(env, src, n, diag) {
  try {
    const res = await gh(`/notifications/threads/${n.id}/subscription`, env.BOT_PAT, { method: "DELETE" });
    (diag.silenced = diag.silenced || []).push(threadRef(n));
    console.log(`[poll] src=${src} SILENCED ${threadRef(n)} (${res.status})`);
  } catch (e) {
    console.log(`[poll] src=${src} SILENCE-FAILED ${threadRef(n)}: ${e}`);
  }
}

// ---- one poll cycle with conditional requests + adaptive cadence.
async function pollOnce(env, state, src) {
  const t0 = Date.now();
  const diag = { src, pollStatus: null, idle: false, unread: 0, qualifying: 0, passed: 0, declined: [], dispatched: [], dispatchStatus: null, nextDelayMs: POLL_FLOOR_MS };

  const lastModified = await state.storage.get("lastModified");
  const etag = await state.storage.get("etag");
  const headers = {};
  // Conditional polling: ETag FIRST (GitHub omits Last-Modified on the
  // empty-notifications response - live-lesson; the etag is always present
  // and If-None-Match 304s identically). Both sent when available.
  if (etag) headers["If-None-Match"] = etag;
  if (lastModified) headers["If-Modified-Since"] = lastModified;

  const res = await gh("/notifications?all=false&per_page=30", env.BOT_PAT, { headers });
  diag.pollStatus = res.status;
  const intervalSec = res.headers.get("X-Poll-Interval");
  const rl = res.headers.get("X-RateLimit-Remaining");
  if (src === "tick") {
    diag.lmSent = lastModified || null;
    diag.lmGot = res.headers.get("Last-Modified");
    diag.etagSent = etag || null;
    diag.etagGot = res.headers.get("ETag");
    diag.hdrs = undefined;
  }

  if (res.status === 304) {
    diag.idle = true;
    // free poll: 30s floor, unless the header signals real strain (>60)
    diag.nextDelayMs = Number(intervalSec) > 60 ? clampIntervalMs(intervalSec, 60) : POLL_FLOOR_MS;
    console.log(`[poll] src=${src} idle-304 rl=${rl} next=${Math.round(diag.nextDelayMs / 1000)}s (${Date.now() - t0}ms)`);
    return diag;
  }
  if (!res.ok) {
    const body = (await res.text()).slice(0, 300);
    diag.pollError = body;
    diag.nextDelayMs = POLL_CAP_MS;
    console.log(`[poll] src=${src} ERROR status=${res.status} rl=${rl} body=${body} backing off ${POLL_CAP_MS / 1000}s`);
    return diag;
  }

  const lm = res.headers.get("Last-Modified");
  if (lm) await state.storage.put("lastModified", lm);
  const et = res.headers.get("ETag");
  if (et) await state.storage.put("etag", et);
  diag.nextDelayMs = clampIntervalMs(intervalSec, 60); // real response: the stated allowance

  const all = await res.json();
  diag.unread = all.length;
  const qualifying = all.filter((n) => REASONS.has(n.reason));
  const junk = all.filter((n) => !REASONS.has(n.reason));
  // Junk hygiene (live-lesson: non-qualifying notifications accumulate as
  // zombie unread and eventually push real mentions out of the 30/page
  // window): ack everything we will never act on. One call each.
  for (const n of junk) {
    try {
      const ack = await gh(`/notifications/threads/${n.id}`, env.BOT_PAT, { method: "PATCH" });
      console.log(`[poll] src=${src} JUNK ${threadRef(n)} reason=${n.reason} ack=${ack.status}`);
    } catch (e) {
      console.log(`[poll] src=${src} JUNK-ACK-FAILED ${threadRef(n)}: ${e}`);
    }
  }
  diag.qualifying = qualifying.length;
  if (qualifying.length === 0 && junk.length === 0) {
    console.log(`[poll] src=${src} idle-200 unread=${all.length} rl=${rl} next=${Math.round(diag.nextDelayMs / 1000)}s (${Date.now() - t0}ms)`);
    return diag;
  }
  if (qualifying.length === 0) {
    console.log(`[poll] src=${src} idle-200 unread=${all.length} junk=${junk.length} rl=${rl} next=${Math.round(diag.nextDelayMs / 1000)}s (${Date.now() - t0}ms)`);
    return diag;
  }

  const roster = await getRoster(env, state);
  const relay = [];
  for (const n of qualifying) {
    const verdict = await prefilter(env, state, n, roster);
    const ref = threadRef(n);
    if (verdict.decline) {
      // positive evidence: ack + never wake Actions
      try {
        const ack = await gh(`/notifications/threads/${n.id}`, env.BOT_PAT, { method: "PATCH" });
        diag.declined.push(`${ref} rule=${verdict.decline}`);
        console.log(`[poll] src=${src} DECLINED ${ref} reason=${n.reason} rule=${verdict.decline} ack=${ack.status}`);
      } catch (e) {
        console.log(`[poll] src=${src} DECLINE-ACK-FAILED ${ref} rule=${verdict.decline}: ${e}`);
      }
      await silenceThread(env, src, n, diag);
    } else {
      relay.push(n);
      console.log(`[poll] src=${src} RELAYED ${ref} reason=${n.reason}`);
      // engaged threads are silenced too: the pipeline will handle this
      // mention and mark read; plain follow-ups afterwards must not
      // deliver at all (verified: mentions always break through)
      await silenceThread(env, src, n, diag);
    }
  }

  if (relay.length > 0) {
    const dispatch = await fetch(`${GH}/repos/${env.PLATFORM_REPO}/dispatches`, {
      method: "POST",
      headers: ghHeaders(env.DISPATCH_PAT),
      body: JSON.stringify({
        event_type: "foreign-mention",
        client_payload: { notifications: relay },
      }),
    });
    diag.dispatchStatus = dispatch.status;
    diag.dispatched = relay.map(threadRef);
    if (!dispatch.ok) diag.dispatchBody = (await dispatch.text()).slice(0, 300);
    console.log(`[poll] src=${src} dispatched=${relay.length} status=${dispatch.status}${diag.dispatchBody ? ` body=${diag.dispatchBody}` : ""} (${Date.now() - t0}ms)`);
  }
  diag.passed = relay.length;
  return diag;
}

// Self-rescheduling scheduler: a default alarm is armed FIRST (the loop can
// never die), then corrected to the adaptive interval the poll recommends.
export class Scheduler {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }

  async ensureAlarm() {
    const current = await this.state.storage.getAlarm();
    if (current === null) {
      await this.state.storage.setAlarm(Date.now() + POLL_FLOOR_MS);
      return "bootstrapped";
    }
    return "armed";
  }

  async alarm() {
    await this.state.storage.setAlarm(Date.now() + POLL_FLOOR_MS);
    if (await this.state.storage.get("paused")) {
      console.log("[poll] src=alarm PAUSED (maintenance mode)");
      return;
    }
    try {
      const diag = await pollOnce(this.env, this.state, "alarm");
      if (diag.nextDelayMs !== POLL_FLOOR_MS) {
        await this.state.storage.setAlarm(Date.now() + diag.nextDelayMs);
      }
    } catch (err) {
      console.log(`[poll] src=alarm EXCEPTION ${err} (default alarm stays armed)`);
    }
  }

  // Maintenance command surface (gated by the worker-level __tick secret):
  //   op=arm               - ensure the alarm loop is running (bootstrap)
  //   op=disarm            - cancel the alarm entirely (no ticks at all)
  //   op=pause / resume    - maintenance mode: alarm keeps ticking but
  //                          never polls (indefinite until resume)
  //   op=snooze&sec=N      - one-time delay: push the NEXT poll N seconds
  //                          out, then normal adaptive cadence resumes
  //   op=status            - paused? + when the alarm next fires
  //   op=peek              - READ-ONLY unconditional notifications dump
  //   op=ack&id=N          - mark one thread read
  //   op=unsub&repo=O/R&n=N - delete the account's thread subscription
  async fetch(request) {
    const url = new URL(request.url);
    const path = url.pathname;
    const op = url.searchParams.get("op");
    if (path === "/ensure") {
      return Response.json({ alarm: await this.ensureAlarm() });
    }
    if (path === "/tick") {
      const diag = await pollOnce(this.env, this.state, "tick");
      diag.alarm = await this.ensureAlarm();
      return Response.json(diag);
    }
    if (path === "/cmd") {
      const out = { op };
      if (op === "pause") {
        await this.state.storage.put("paused", true);
        out.paused = true;
      } else if (op === "resume") {
        await this.state.storage.delete("paused");
        out.paused = false;
      } else if (op === "arm") {
        out.alarm = await this.ensureAlarm();
      } else if (op === "disarm") {
        await this.state.storage.deleteAlarm();
        out.alarm = "disarmed";
      } else if (op === "snooze") {
        const sec = Math.max(1, Number(url.searchParams.get("sec")) || 60);
        await this.state.storage.setAlarm(Date.now() + sec * 1000);
        out.snoozedSec = sec;
      } else if (op === "status") {
        out.paused = !!(await this.state.storage.get("paused"));
        const at = await this.state.storage.getAlarm();
        out.alarmAt = at === null ? null : new Date(at).toISOString();
        out.now = new Date().toISOString();
      } else if (op === "peek") {
        const res = await gh("/notifications?all=false&per_page=30", this.env.BOT_PAT);
        out.status = res.status;
        out.items = res.ok
          ? (await res.json()).map((n) => ({
              id: n.id,
              reason: n.reason,
              updated_at: n.updated_at,
              repo: n.repository?.full_name,
              title: n.subject?.title?.slice(0, 60),
              type: n.subject?.type,
              url: n.subject?.url,
            }))
          : await res.text();
      } else if (op === "ack") {
        const id = url.searchParams.get("id");
        const res = await gh(`/notifications/threads/${id}`, this.env.BOT_PAT, { method: "PATCH" });
        out.status = res.status;
      } else if (op === "sub") {
        const tid = url.searchParams.get("tid");
        const res = await gh(`/notifications/threads/${tid}/subscription`, this.env.BOT_PAT);
        out.status = res.status;
        out.subscription = res.ok ? await res.json() : await res.text();
      } else if (op === "setsub") {
        const tid = url.searchParams.get("tid");
        const sub = url.searchParams.get("sub") === "true";
        const ign = url.searchParams.get("ignored") === "true";
        const res = await gh(`/notifications/threads/${tid}/subscription`, this.env.BOT_PAT, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ subscribed: sub, ignored: ign }),
        });
        out.status = res.status;
        out.result = res.ok ? await res.json() : await res.text();
      } else if (op === "unsub") {
        const tid = url.searchParams.get("tid");
        const res = await gh(`/notifications/threads/${tid}/subscription`, this.env.BOT_PAT, { method: "DELETE" });
        out.status = res.status;
      } else {
        return new Response("unknown op\n", { status: 400 });
      }
      return Response.json(out);
    }
    return new Response("not found\n", { status: 404 });
  }
}

export default {
  // Cron path kept only for manual remote tests (wrangler dev --test-
  // scheduled). No cron is attached in production — see wrangler.toml.
  // Simple unconditional poll+relay; the gauntlet lives in the repo.
  async scheduled(event, env) {
    const res = await gh("/notifications?all=false&per_page=30", env.BOT_PAT);
    const diag = { pollStatus: res.status };
    if (res.ok) {
      const all = await res.json();
      const qualifying = all.filter((n) => REASONS.has(n.reason));
      diag.unread = all.length;
      diag.qualifying = qualifying.length;
      if (qualifying.length > 0) {
        const d = await fetch(`${GH}/repos/${env.PLATFORM_REPO}/dispatches`, {
          method: "POST",
          headers: ghHeaders(env.DISPATCH_PAT),
          body: JSON.stringify({ event_type: "foreign-mention", client_payload: { notifications: qualifying } }),
        });
        diag.dispatchStatus = d.status;
      }
    }
    console.log(`[poll] src=cron ${JSON.stringify(diag)}`);
  },

  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname !== "/__tick") {
      return new Response("not found\n", { status: 404 });
    }
    const key = url.searchParams.get("key") || "";
    if (!env.TEST_TOKEN || key !== env.TEST_TOKEN) {
      return new Response("forbidden\n", { status: 403 });
    }
    const id = env.SCHEDULER.idFromName("only");
    if (url.searchParams.get("cmd")) {
      const cmdUrl = "https://do/cmd?" + url.searchParams.toString();
      return env.SCHEDULER.get(id).fetch(cmdUrl);
    }
    return env.SCHEDULER.get(id).fetch("https://do/tick");
  },
}
