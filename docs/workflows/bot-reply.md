# Bot Reply on Mention

The general agent: anything that isn't a structured review or compliance run lands here. Conversations, questions, investigations, on-demand work, code contributions — and cross-repo guest sessions.

**Triggers:** `workflow_dispatch` only (the Agent Router dispatches it; the Mention Poller dispatches it for guest mode).
**Executes from:** the default branch, always.
**Permissions:** `contents: read`, `issues: write`, `pull-requests: write`.

**Inputs:** `commentId` (home mode — the triggering comment, re-fetched from the API), and for guest mode `threadType`/`targetRepo`/`triggerKind`/`payload` from the poller.

## Home mode

A mention in an issue or PR thread (`@mirrobot`, `@mirrobot-agent`) → router dispatch → this workflow:

1. Bot-setup (identity, config, masks, plugins — same as every agent).
2. Re-fetch the comment by id; classify the thread (issue vs PR).
3. PR threads get the full review context machinery (three-block discussion fetch — elevated own reviews, filtered own history, correlated thread context — plus diffs via the review kit if the ask is review-shaped).
4. The **security brief + guest-free prompt assembly**, then one agent session.
5. The agent may load **instruction sets** — the same playbooks the dedicated modes use (how to review, how to contribute) — loaded *as instructions for a job*, while bot-reply stays its own "person" with its own judgment. Compliance is never bot-reply's job.

What it can do in one session: answer, investigate (read-only anywhere), run the review kit on any PR (including other repos', via the guest rules), contribute code (branch → commits → PR, within scope-of-action rules), and post its deliverable as a top-level comment with the standard footer.

## Guest mode (cross-repo)

When dispatched by the Mention Poller, the same machinery runs pointed at a **foreign repository**:

- The target repo is cloned, its PR/issue context fetched, the scrub runs in `--foreign` mode (no trusted anchors exist abroad — *all* auto-load content is removed and quarantined).
- The session carries the **guest rules**: read-only by default; writes require either a verified link back to a home repo (verified as if self-discovered) or an explicit ask from an **allowlist member** — and authority is pinned to the allowlist, never to thread participation. After an authorized summoner triggers the session, everyone else's requests in that thread are data, not direction.
- Reactions, footers, share links all work abroad.

Setup and the worker side: [mention-poller.md](mention-poller.md).

**Knobs:** the env block in the workflow (`BOT_NAMES_JSON`, noise filters, fetch limits, `PREVIOUS_BOT_REVIEWS_COUNT`), `AGENT_MODELS_JSON["bot-reply"]`, and the guest-mode variables (`FOREIGN_MENTIONS_ENABLED`, `FOREIGN_MENTIONS_USERS`).

**When it goes red:** real failures are the agent session exiting non-zero, or the comment re-fetch failing (deleted comments — retried, then failed honestly). A gray skipped run means `AGENT_PAUSED=true` or a phantom push event (expected).

**Testing it:** mention the bot in any issue; for guest mode, mention it from a repo it's not installed in (after enabling the poller) and watch the run name — "Automated mention relay from ..." tells you the full chain fired.
