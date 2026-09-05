#!/usr/bin/env bash
# handle-mentions.sh — cross-repo mention & review-request pipeline.
#
# Gives the agent "ears" outside its home repositories: the bot ACCOUNT is
# @mentionable and review-requestable in ANY public repo; GitHub turns those
# into account notifications. This script consumes notifications, applies the
# skip/trust/verification gauntlet, and dispatches bot-reply in GUEST mode.
#
# Two entry modes (same pipeline, same trust logic — in ONE place, battery-
# tested; nothing downstream trusts either source):
#   poll  (default): reads /notifications with the account PAT
#                    (needs the `notifications` scope — the scope gate
#                    tolerates public_repo+notifications).
#   relay (--payload '<json array of notification objects>'): the external
#         mention-worker (see tools/mention-worker/) forwards RAW
#         notifications via repository_dispatch. The payload is UNTRUSTED
#         input: every field is re-fetched from the GitHub API before any
#         decision. The relay holds no logic, no trust, and marks nothing
#         read — this script is the single writer of read-state.
#
# Pipeline per notification:
#   1. reason filter      — only `mention` and `review_requested`
#   2. skip matrix        — repo owned by the HOME owner AND carrying the
#                           platform -> no-op (that repo's own instance owns
#                           it; mark read). Home-owner repo WITHOUT the
#                           platform -> handle (nothing local would answer).
#                           Anyone else's repo -> handle (a fork of this
#                           platform abroad serves THEIR bot identity, not
#                           this account).
#   3. mark read          — BEFORE dispatch (at-most-once; skipped/declined
#                           notifications are also acked so they never
#                           rescan)
#   4. subject re-fetch   — comment / issue body / PR via the GitHub API
#                           (never the relay's or notification's word)
#   5. bot-loop guard     — our own identities never trigger us
#   6. summoner trust     — actor ∈ (home-repo direct collaborators ∪
#                           FOREIGN_MENTIONS_USERS). Deliberately NOT
#                           TRUSTED_AGENT_USERS: cross-repo summoning is a
#                           separate, stricter privilege.
#   7. genuine-mention    — mention mode: the fetched body must contain a
#        verification      real @<account-login> token (case-insensitive).
#                           review mode: the PR timeline's review_requested
#                           ACTOR (who clicked the button) must pass the
#                           trust gate — not merely the PR author.
#   8. dispatch           — bot-reply guest mode (capped per run).
#
# Contract:
#   env: GH_TOKEN (account PAT with `notifications` scope for poll mode;
#        any token able to read the subject APIs for relay mode),
#        DISPATCH_GH_TOKEN (optional: a token with actions:write on the
#        platform repo for the bot-reply dispatch - pass the workflow's
#        ephemeral github.token; the ACCOUNT PAT usually cannot dispatch
#        (no write access, live-observed) and notifications are
#        user-scoped so the ephemeral token cannot poll - hence two tokens).
#        Falls back to GH_TOKEN when unset.
#        GITHUB_REPOSITORY (the HOST repo = this agent's home),
#        HOME_OWNER (login of the host repo's owner — home repos are
#        "repos owned by the same owner as the poller repo"; derived by
#        the workflow, not hardcoded),
#        FOREIGN_MENTIONS_USERS (optional comma logins — the separate
#        cross-repo allowlist variable),
#        BOT_NAMES_JSON (this agent's identities, case-insensitive),
#        MAX_DISPATCH (optional, default 3 — per-run cost bound).
#   uses: fetch-roster.sh --print from the same directory (trusted /tmp copy).
#   exit: 0 always (a clean decline is a normal outcome); dispatch failures
#         are logged loudly but never fail the run.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_DISPATCH="${MAX_DISPATCH:-3}"
dispatched=0
declined=0
acked=0

log() { echo "mentions: $*"; }

# ---------------------------------------------------------------- roster ---
# Summoner allowlist: home-repo direct collaborators ∪ FOREIGN_MENTIONS_USERS.
# Never TRUSTED_AGENT_USERS (separate privilege by design). On roster fetch
# failure: FAIL CLOSED (no cross-repo summons this run) — a fabricated or
# partial roster is worse than a skipped cycle.
if ROSTER=$(EXTRA_TRUSTED_USERS="${FOREIGN_MENTIONS_USERS:-}" \
             bash "$SCRIPT_DIR/fetch-roster.sh" --print 2>/dev/null) && [ -n "$ROSTER" ]; then
  log "summoner roster loaded ($(echo "$ROSTER" | tr ',' '\n' | grep -c .) entries)"
else
  log "roster unavailable/empty - failing closed this run (no cross-repo summons)."
  ROSTER=""
fi

in_roster() { # $1 = login (case-insensitive membership in comma list)
  [ -n "$ROSTER" ] || return 1
  [ -n "${1:-}" ] || return 1
  printf ' %s ' "$ROSTER" | tr ',' ' ' | grep -qi " $1 "
}

is_bot() { # $1 = login — this agent's identity family, case-insensitive
  printf '%s' "${1:-}" | tr 'A-Z' 'a-z' | jq -Rn --argjson bots "${BOT_NAMES_JSON:-[\"mirrobot-agent\",\"mirrobot-agent[bot]\"]}" \
    '$bots | index(input | ascii_downcase) != null'
}

has_mention_token() { # $1 = body — contains @<user-identity> (not the [bot]
  # app form, not bare name text), case-insensitive. Command substitution +
  # herestring (NOT process substitution: inside the main read-loop a
  # proc-sub feed proved unreliable on this runner family - silent EOF).
  local b_lc="$1" name names
  b_lc=$(printf '%s' "$b_lc" | tr 'A-Z' 'a-z')
  names=$(printf '%s' "${BOT_NAMES_JSON:-[]}" | jq -r '.[] | select(endswith("[bot]") | not)' 2>/dev/null)
  while read -r name; do
    [ -n "$name" ] || continue
    if printf '%s' "$b_lc" | grep -qi "@$name"; then
      return 0
    fi
  done <<< "$names"
  return 1
}

# ------------------------------------------------------------ notifications --
if [ "${1:-}" = "--payload" ]; then
  RAW="${2:?--payload requires a json array argument}"
  NOTIFS=$(printf '%s' "$RAW" | jq -c 'if type=="array" then . else [.] end' 2>/dev/null) || {
    log "relay payload is not valid json - ignoring."; exit 0; }
  log "relay mode: $(printf '%s' "$NOTIFS" | jq 'length') notification(s) forwarded"
else
  if ! NOTIFS=$(gh api "/notifications?all=false&per_page=30" 2>/dev/null); then
    log "poll failed (403 usually = PAT lacks the 'notifications' scope; regenerate ACCOUNT_GH_TOKEN with public_repo + notifications)."
    exit 0
  fi
  log "poll mode: $(printf '%s' "$NOTIFS" | jq 'length') unread notification(s)"
fi

DEFAULT_BRANCH=$(gh api "/repos/${GITHUB_REPOSITORY}" --jq .default_branch 2>/dev/null || echo main)
PLATFORM_CHECK_CACHE=""

platform_present() { # $1 = full repo name -> "true"/"false" (cached)
  [ -n "$PLATFORM_CHECK_CACHE" ] || PLATFORM_CHECK_CACHE=" "
  case "$PLATFORM_CHECK_CACHE" in
    *"|$1=true|"*) echo true; return ;;
    *"|$1=false|"*) echo false; return ;;
  esac
  if gh api "/repos/$1/contents/.github/workflows/bot-reply.yml" >/dev/null 2>&1; then
    v=true
  else
    v=false
  fi
  PLATFORM_CHECK_CACHE="${PLATFORM_CHECK_CACHE}|$1=$v|"
  echo "$v"
}

mark_read() { # $1 = notification thread id
  gh api --method PATCH "/notifications/threads/$1" >/dev/null 2>&1 || true
  acked=$((acked + 1))
}

# ------------------------------------------------------------------ main ----
while read -r n; do
  [ -n "$n" ] || continue
  id=$(printf '%s' "$n" | jq -r '.id // empty')
  reason=$(printf '%s' "$n" | jq -r '.reason // empty')
  repo=$(printf '%s' "$n" | jq -r '.repository.full_name // empty')
  owner=$(printf '%s' "$n" | jq -r '.repository.owner.login // empty' | tr 'A-Z' 'a-z')
  subject_type=$(printf '%s' "$n" | jq -r '.subject.type // empty')
  subject_url=$(printf '%s' "$n" | jq -r '.subject.url // empty')
  latest_url=$(printf '%s' "$n" | jq -r '.subject.latest_comment_url // empty')
  [ -n "$id" ] && [ -n "$repo" ] || continue

  # 1. reason filter. mention/review_requested are the direct summons;
  #    subscribed/comment are FOLLOW-UPS in threads whose first mention
  #    already subscribed the account (GitHub's reason taxonomy: once
  #    subscribed, later @mentions in the same thread classify as
  #    "subscribed" - live-observed). They are treated as mention-kind and
  #    gated by the genuine-mention CONTENT check below, which is the real
  #    filter (non-mention comments decline there). Everything else is ack.
  case "$reason" in
    mention|review_requested|subscribed|comment) ;;
    *) mark_read "$id"; declined=$((declined + 1)); continue ;;
  esac

  # 2. skip matrix — home-owner repos WITH the platform are handled locally
  if [ "$owner" = "${HOME_OWNER,,}" ]; then
    if [ "$(platform_present "$repo")" = true ]; then
      log "skip $reason in $repo (home-owner repo with the platform - local instance owns it)"
      mark_read "$id"; declined=$((declined + 1)); continue
    fi
    log "home-owner repo $repo has NO platform - handling here"
  fi

  # 3. mark read BEFORE any dispatch (at-most-once)
  mark_read "$id"

  # 4. subject re-fetch (never trust the notification/relay content fields)
  # Thread number ALWAYS from the subject (issue) URL; the comment id (when
  # the trigger is a comment) separately from latest_comment_url — the
  # combined form ".../issues/comments/N" defeats a single digit pattern.
  number=$(printf '%s' "$subject_url" | sed -n 's:.*/issues/\([0-9][0-9]*\)$:\1:p')
  [ -n "$number" ] || number=$(printf '%s' "$latest_url" | sed -n 's:.*/issues/\([0-9][0-9]*\)$:\1:p')
  if [ -z "$number" ]; then
    log "decline: could not derive thread number for $repo ($subject_type)"
    declined=$((declined + 1)); continue
  fi
  rel_comment_id=$(printf '%s' "${latest_url:-}" | sed -n 's:.*/issues/comments/\([0-9][0-9]*\)$:\1:p')

  if [ "$reason" = "review_requested" ]; then
    # Trust the ACTOR of the review_requested timeline event (who clicked
    # the button), not the PR author.
    actor=$(gh api "/repos/$repo/issues/$number/timeline?per_page=100" 2>/dev/null \
      | jq -r '[.[] | select(.event == "review_requested") | .actor.login] | last // empty')
    if [ -z "$actor" ]; then
      log "decline: no review_requested event found in $repo#$number"
      declined=$((declined + 1)); continue
    fi
    if [ "$(is_bot "$actor")" = true ]; then
      declined=$((declined + 1)); continue
    fi
    if ! in_roster "$actor"; then
      log "decline: review requester @$actor not on the cross-repo allowlist ($repo#$number)"
      declined=$((declined + 1)); continue
    fi
    kind=review-request
    trigger_author="$actor"
  else
    # Mention: the triggering content is the comment (or the issue/PR body
    # when the mention lives there — latest_comment_url then points at the
    # issue itself, without a /comments/ segment).
    if [ -n "$rel_comment_id" ]; then
      comment_id="$rel_comment_id"
      body_json=$(gh api "/repos/$repo/issues/comments/$comment_id" 2>/dev/null) || body_json=""
      trigger_author=$(printf '%s' "$body_json" | jq -r '.user.login // empty')
      body=$(printf '%s' "$body_json" | jq -r '.body // empty')
    else
      comment_id=""
      body_json=$(gh api "/repos/$repo/issues/$number" 2>/dev/null) || body_json=""
      trigger_author=$(printf '%s' "$body_json" | jq -r '.user.login // empty')
      body=$(printf '%s' "$body_json" | jq -r '.body // empty')
    fi
    if [ -z "$trigger_author" ] || [ -z "$body" ]; then
      log "decline: could not fetch trigger content for $repo#$number"
      declined=$((declined + 1)); continue
    fi
    if [ "$(is_bot "$trigger_author")" = true ]; then
      declined=$((declined + 1)); continue
    fi
    if ! in_roster "$trigger_author"; then
      log "decline: summoner @$trigger_author not on the cross-repo allowlist ($repo#$number)"
      declined=$((declined + 1)); continue
    fi
    # Genuine-mention verification: a real @<login> token for one of our
    # USER identities (case-insensitive). Commands like /mirrobot-review and
    # bare "mirrobot" text never create mention notifications, but the
    # re-check holds the line if a relay payload lies.
    if ! has_mention_token "$body"; then
      log "decline: no genuine @mention token in fetched body ($repo#$number)"
      declined=$((declined + 1)); continue
    fi
    kind=mention
  fi

  # 8. dispatch (cap). Uses DISPATCH_GH_TOKEN when provided (the workflow's
  # ephemeral token with actions:write - the account PAT cannot dispatch:
  # notifications are user-scoped, dispatches are repo-write-scoped).
  if [ "$dispatched" -ge "$MAX_DISPATCH" ]; then
    log "cap reached ($MAX_DISPATCH) - @$trigger_author's $reason in $repo#$number skipped this run (already acked; re-mention to retry)."
    continue
  fi
  if GH_TOKEN="${DISPATCH_GH_TOKEN:-$GH_TOKEN}" gh workflow run bot-reply.yml --repo "$GITHUB_REPOSITORY" --ref "$DEFAULT_BRANCH" \
       -f "targetRepo=$repo" -f "threadNumber=$number" \
       -f "commentId=${comment_id:-}" -f "triggerKind=$kind"; then
    dispatched=$((dispatched + 1))
    log "DISPATCHED guest bot-reply: @$trigger_author $kind in $repo#$number"
  else
    log "dispatch FAILED for $repo#$number (acked already - re-mention to retry)"
  fi
done < <(printf '%s' "$NOTIFS" | jq -c '.[]?' 2>/dev/null)

log "summary: dispatched=$dispatched declined=$declined acked=$acked"
