#!/usr/bin/env bash
# generate-review-kit.sh — self-serve review context kit for ANY PR in this repo.
#
# Called by the bot-reply workflow (pre-session, for the thread's PR) AND by
# the agent itself (any PR number, from any thread - "review PR #42" asked in
# an unrelated issue works identically). Generates everything a review needs:
#   - review type (FIRST / FOLLOW-UP) from THIS agent's last review marker
#   - full diff (+ incremental diff on FOLLOW-UP) as navigable files
#   - head-SHA file for the target PR (kit-scoped, never clobbers the
#     thread's own /tmp/head_sha.txt)
#   - the review instruction sets + review memory, assembled from the SAME
#     trusted parts the pr-review workflow uses, values baked in
#   - the three discussion blocks (previous reviews / history / thread)
#
# Contract:
#   $1      : PR number (required)
#   env in  : GH_TOKEN (required), GITHUB_REPOSITORY (required),
#             BOT_NAMES_JSON (optional; default = this agent's identities),
#             PREVIOUS_BOT_REVIEWS_COUNT (optional, default 1)
#   files   : /tmp/kit/<pr>/... (diffs, head_sha.txt, context.env),
#             /tmp/instructions/review-{first,followup}.md + review-memory.md
#             (kit writes these for the TARGET pr - re-run the kit to switch)
#   stdout  : KIT RESULT summary (type + paths + sizes) - the caller reads
#             this to decide which instruction file to load
#   exit    : 0 success; 1 usage/fetch failure (message on stdout)
set -uo pipefail

PR="${1:-}"
REPO="${GITHUB_REPOSITORY:-}"
if [ -z "$PR" ] || [ -z "$REPO" ] || [ -z "${GH_TOKEN:-}" ]; then
  echo "KIT ERROR: usage: generate-review-kit.sh <pr_number> (needs GH_TOKEN and GITHUB_REPOSITORY in env)"
  exit 1
fi
case "$PR" in ''|*[!0-9]*) echo "KIT ERROR: PR number must be numeric, got '$PR'"; exit 1;; esac

BOT_NAMES_JSON="${BOT_NAMES_JSON:-[\"mirrobot-agent[bot]\",\"mirrobot\",\"mirrobot-agent\"]}"
KIT_DIR="/tmp/kit/$PR"
mkdir -p "$KIT_DIR" /tmp/instructions

pr_json=$(gh api "/repos/$REPO/pulls/$PR" 2>/dev/null) || {
  echo "KIT ERROR: cannot fetch PR #$PR in $REPO (not found, no access, or API failure)"
  exit 1
}

PR_HEAD_SHA=$(printf '%s' "$pr_json" | jq -r .headRefOid)
PR_BASE=$(printf '%s' "$pr_json" | jq -r '.base.ref')
PR_AUTHOR=$(printf '%s' "$pr_json" | jq -r '.user.login')
PR_TITLE=$(printf '%s' "$pr_json" | jq -r .title)
printf '%s' "$PR_HEAD_SHA" > "$KIT_DIR/head_sha.txt"

# --- review type: latest review by THIS agent carrying the marker ----------
# (single retry: GitHub intermittently 500s this endpoint; a failed fetch
# degrades to FIRST = full-diff review, which is safe but wasteful)
reviews_json=""
for _attempt in 1 2; do
  reviews_json=$(gh api "/repos/$REPO/pulls/$PR/reviews" --paginate 2>/dev/null) && break
  sleep 2
done
reviews_json="${reviews_json:-[]}"
LAST_REVIEWED_SHA=$(printf '%s' "$reviews_json" | jq -r \
  --argjson bots "$BOT_NAMES_JSON" '
  map(select(.user.login as $u | $bots | index($u)))
  | sort_by(.submitted_at)
  | map(.body // "" | scan("last_reviewed_sha:[a-f0-9]+"))
  | flatten
  | last // ""
  | ltrimstr("last_reviewed_sha:")')
if [ -n "$LAST_REVIEWED_SHA" ]; then
  REVIEW_TYPE="FOLLOW-UP"
else
  REVIEW_TYPE="FIRST"
fi

# --- diffs (fetch the PR head AND the base ref; three-dot full diff) -------
git fetch origin "$PR_BASE" >/dev/null 2>&1 || true
git fetch origin "pull/$PR/head" >/dev/null 2>&1 || {
  echo "KIT ERROR: cannot fetch pull/$PR/head"
  exit 1
}
FULL_DIFF="$KIT_DIR/full_diff.patch"
git diff "origin/$PR_BASE...FETCH_HEAD" > "$FULL_DIFF" 2>/dev/null || git diff "origin/$PR_BASE..FETCH_HEAD" > "$FULL_DIFF"
INCREMENTAL_DIFF=""
if [ "$REVIEW_TYPE" = "FOLLOW-UP" ]; then
  INCREMENTAL_DIFF="$KIT_DIR/incremental_diff.patch"
  if git diff "$LAST_REVIEWED_SHA..FETCH_HEAD" > "$INCREMENTAL_DIFF" 2>/dev/null; then
    :
  else
    # marker no longer resolvable (force-push orphaned it): fall back to full
    cp "$FULL_DIFF" "$INCREMENTAL_DIFF"
    echo "KIT NOTE: last-reviewed SHA $LAST_REVIEWED_SHA unresolvable - incremental diff fell back to the full diff"
    LAST_REVIEWED_SHA=""
  fi
fi

# --- discussion blocks (shared machinery, GITHUB_ENV redirected to a file) --
# Parse the GITHUB_ENV-style file with awk - NEVER source it: heredoc content
# can contain arbitrary markdown (parens, backticks, quotes) that a shell
# `source` would execute.
extract_env_var() { # $1 file, $2 var name -> prints heredoc value
  awk -v v="$2" '
    index($0, v "<<") == 1 {
      delim = substr($0, length(v) + 3)
      while ((getline line) > 0) {
        if (line == delim) exit
        print line
      }
      exit
    }
  ' "$1"
}
if [ -f /tmp/fetch-pr-discussion.sh ] && [ -f "$KIT_DIR/context.env" ]; then :; elif [ -f /tmp/fetch-pr-discussion.sh ]; then
  GITHUB_ENV="$KIT_DIR/context.env" BOT_NAMES_JSON="$BOT_NAMES_JSON" \
    PREVIOUS_BOT_REVIEWS_COUNT="${PREVIOUS_BOT_REVIEWS_COUNT:-1}" \
    bash /tmp/fetch-pr-discussion.sh "$PR" >/dev/null 2>&1 || echo "KIT NOTE: discussion fetch degraded - review memory may be empty"
fi
if [ -f "$KIT_DIR/context.env" ]; then
  PREVIOUS_BOT_REVIEWS=$(extract_env_var "$KIT_DIR/context.env" PREVIOUS_BOT_REVIEWS)
  AGENT_REVIEW_HISTORY=$(extract_env_var "$KIT_DIR/context.env" AGENT_REVIEW_HISTORY)
fi
PREVIOUS_BOT_REVIEWS="${PREVIOUS_BOT_REVIEWS:-none}"
AGENT_REVIEW_HISTORY="${AGENT_REVIEW_HISTORY:-none}"

# --- instruction sets (trusted parts; same source as pr-review) ------------
PULL_REQUEST_CONTEXT="PR #$PR: $PR_TITLE by $PR_AUTHOR (base: $PR_BASE, head: $PR_HEAD_SHA)"
if [ -n "$(printf '%s' "$pr_json" | jq -r '.body // empty')" ]; then
  PULL_REQUEST_CONTEXT="$PULL_REQUEST_CONTEXT

$(printf '%s' "$pr_json" | jq -r '.body')"
fi
RVARS='${DIFF_FILE_PATH} ${INCREMENTAL_DIFF_PATH} ${LAST_REVIEWED_SHA} ${PR_HEAD_SHA} ${PREVIOUS_BOT_REVIEWS} ${AGENT_REVIEW_HISTORY} ${PR_NUMBER} ${GITHUB_REPOSITORY} ${THREAD_NUMBER} ${PR_AUTHOR} ${REVIEW_TYPE} ${PULL_REQUEST_CONTEXT}'
export THREAD_NUMBER="$PR"
if [ -f /tmp/assemble-prompt.sh ]; then
  bash /tmp/assemble-prompt.sh review-memory-instructions \
    | PREVIOUS_BOT_REVIEWS="$PREVIOUS_BOT_REVIEWS" AGENT_REVIEW_HISTORY="$AGENT_REVIEW_HISTORY" \
      PR_NUMBER="$PR" GITHUB_REPOSITORY="$REPO" envsubst "$RVARS" > /tmp/instructions/review-memory.md
  bash /tmp/assemble-prompt.sh review-first-instructions \
    | REVIEW_TYPE=FIRST DIFF_FILE_PATH="$FULL_DIFF" INCREMENTAL_DIFF_PATH="" LAST_REVIEWED_SHA="" \
      PR_HEAD_SHA="$PR_HEAD_SHA" PREVIOUS_BOT_REVIEWS="$PREVIOUS_BOT_REVIEWS" AGENT_REVIEW_HISTORY="$AGENT_REVIEW_HISTORY" \
      PR_NUMBER="$PR" GITHUB_REPOSITORY="$REPO" PR_AUTHOR="$PR_AUTHOR" PULL_REQUEST_CONTEXT="$PULL_REQUEST_CONTEXT" \
      envsubst "$RVARS" > /tmp/instructions/review-first.md
  bash /tmp/assemble-prompt.sh review-followup-instructions \
    | REVIEW_TYPE="FOLLOW-UP" DIFF_FILE_PATH="${INCREMENTAL_DIFF:-$FULL_DIFF}" INCREMENTAL_DIFF_PATH="${INCREMENTAL_DIFF:-}" LAST_REVIEWED_SHA="${LAST_REVIEWED_SHA:-}" \
      PR_HEAD_SHA="$PR_HEAD_SHA" PREVIOUS_BOT_REVIEWS="$PREVIOUS_BOT_REVIEWS" AGENT_REVIEW_HISTORY="$AGENT_REVIEW_HISTORY" \
      PR_NUMBER="$PR" GITHUB_REPOSITORY="$REPO" PR_AUTHOR="$PR_AUTHOR" PULL_REQUEST_CONTEXT="$PULL_REQUEST_CONTEXT" \
      envsubst "$RVARS" > /tmp/instructions/review-followup.md
else
  echo "KIT ERROR: /tmp/assemble-prompt.sh not present - trusted prompt artifacts missing"
  exit 1
fi

if [ "$REVIEW_TYPE" = "FOLLOW-UP" ]; then
  INSTR_NAME="review-followup"
else
  INSTR_NAME="review-first"
fi

# --- result -----------------------------------------------------------------
echo "KIT RESULT (PR #$PR):"
echo "  Review type:      $REVIEW_TYPE$([ -n "${LAST_REVIEWED_SHA:-}" ] && echo " (last reviewed: ${LAST_REVIEWED_SHA:0:12})")"
echo "  Instructions:     /tmp/instructions/$INSTR_NAME.md"
echo "  Review memory:    /tmp/instructions/review-memory.md"
echo "  Full diff:        $FULL_DIFF ($(wc -l < "$FULL_DIFF") lines)"
[ -n "$INCREMENTAL_DIFF" ] && echo "  Incremental diff: $INCREMENTAL_DIFF ($(wc -l < "$INCREMENTAL_DIFF") lines)"
echo "  Head SHA file:    $KIT_DIR/head_sha.txt"
echo "  Kit files are for PR #$PR - re-run the kit to switch PRs."
