# Plan: Discussions Support (bot-reply extension) + Ask-Mirrobot Landing Page

Status: approved design, not yet implemented.
Scope: Mirrobot-agent repo first, then mirror to proxy main + dev/experimental snapshots.

## Research verdicts (verified)

- **Native triggers exist**: `discussion` and `discussion_comment` workflow events (GA since 2021, official changelog). Default-branch workflow file required — same main-only guarantee as `issue_comment`.
- **Proxy repo**: discussions enabled. **Mirrobot-agent**: disabled — enable via one API call (`PATCH /repos/Mirrowel/Mirrobot-agent -f has_discussions=true`).
- **Avatar**: `https://avatars.githubusercontent.com/u/317937646?v=4` — base64-embed into badge SVG (zread pattern; `<img>`-embedded SVGs cannot load external images, so snapshot embedding is the correct mechanism; refresh via one-line script when the avatar changes).
- **API**: GraphQL is the guaranteed path (`addDiscussionComment`, `updateDiscussionComment`, `addReaction`, `authorAssociation`, `isMinimized` — account PAT already GraphQL-proven in live batteries). Probe for REST discussion-comment endpoints at build time; prefer REST where it exists.

## Part 1 — bot-reply discussion mode

### Triggers (both paths required)

1. `discussion_comment [created]` **with mention** → sibling of `issue_comment` routing; trigger message = the comment.
2. `discussion [created]` **with mention in the body** → required so landing-page users (who paste the template into the discussion body) are not met with silence; trigger message = the discussion body.

Bot-loop guard (actor check) on both.

### Router

- `agent-router.yml`: two new events; same mention regex as issue comments.
- `route-comment.sh`: dispatch gains `threadType` input: `issue | pr | discussion` (comment-trigger) and `discussion-new` (body-trigger).
- Body-trigger passes the discussion number as `threadNumber`; `commentId` validation relaxed only for `discussion-new` (validator branch, not schema-level).

### bot-reply resolve step

Branch on `threadType`:
- GraphQL fetch: discussion title, body, **category**, `isAnswered` + accepted answer, comments (`databaseId`, author, `authorAssociation`, `isMinimized`).
- Locate the triggering comment; export trigger message + author trust line.
- Export `REACTION_SUBJECT_NODE` (comment node for comment-trigger, discussion node for body-trigger).
- Re-validate mention + bot-loop from fetched data (defense in depth, same pattern as issues).

### Context block

Inline flat context (issue-mode sibling): discussion body + category + answer state + recent comments.
Noise + hidden filtering reuse the existing machinery: `CONTEXT_IGNORE_AUTHORS`, `CONTEXT_FILTER_PATTERNS_JSON`, `isMinimized` — same env, same semantics.

### Posting + reactions

- `posting.md` + `mission-agent.md`: GraphQL mutation with `-F body=@/tmp/comment-body.md` (file-based mandate preserved).
- Living ack: `updateDiscussionComment` (edit by node ID replaces `--edit-last`).
- Final answer: reply comment to the discussion.
- `react.sh`: GraphQL branch (`addReaction` / list / `removeReaction` on subject node) — 👀→🚀/😕 lifecycle works on discussions; PR/issue paths untouched.

### Unchanged by design

Scrub, trust context, roster, requester-context, concurrency (`bot-reply-<number>` serializes per-discussion), compliance, pr-review.

### Permissions

- Workflow token: no changes.
- **App mode** needs the *Discussions: Read/Write* grant in GitHub App settings (user-side checkbox) — OPEN QUESTION (see below).
- Account PAT (`public_repo`): already covers discussions.

## Part 2 — Badge + landing page (Phase 1)

- `docs/ask-badge.svg` — shields-proportioned, avatar embedded base64, sits next to Zread/DeepWiki badges.
- `docs/index.html` — self-contained, no build step, no external deps; dark/light via `prefers-color-scheme`.
  - `?repo=OWNER/REPO` → hero "Mirrobot is active in OWNER/REPO"; no param → generic landing + adopter instructions.
  - **Ask button → `github.com/{repo}/discussions`** with an on-page instruction box: "Start a discussion, mention `@mirrobot-agent`, get your answer in-thread" + copy-paste question template.
  - Client-side repo fetch reads `has_discussions`; fallback for repos without discussions: prefilled issue creation (OPEN QUESTION).
  - Live-activity strip: client-side fetch of the bot's latest public review/comment (`api.github.com`, CORS-open, unauthenticated).
  - Honest architecture section + badge copy-paste snippet.
- Pages: enable via API (source: main / `docs`) + `.nojekyll`.
- Badge placement: proxy README (next to Zread/DeepWiki pair, `?repo=Mirrowel/LLM-API-Key-Proxy`), Mirrobot-agent README.

## Tests + ship order

1. New fixtures: router discussion matrix (comment-mention, body-mention, bot-loop, non-mention), react.sh GraphQL transitions (mock-gh), prompt pins (GraphQL posting block, discussion context, category/answer-state wording), trigger-inventory update.
2. Batteries + strict YAML.
3. Ship: mirrobot-agent → proxy main → dev + experimental snapshots.
4. Enable discussions on Mirrobot-agent; Pages live; badges in READMEs.
5. Live chain test on the proxy repo: real discussion, template mention, full eyes→answer→rocket.

## Open questions

1. App-mode Discussions grant — add to the Mirrobot App settings now, or account-mode-only initially?
2. Ask button fallback for repos without discussions — prefilled issue, or discussions-only?
