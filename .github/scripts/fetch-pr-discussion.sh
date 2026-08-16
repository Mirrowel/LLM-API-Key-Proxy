#!/usr/bin/env bash
# fetch-pr-discussion.sh — shared PR discussion context for agent workflows.
#
# Single source of truth for the THREE-BLOCK context separation used by both
# PR Review and Compliance Check (identical machinery, identical semantics):
#   1. PREVIOUS_BOT_REVIEWS  - newest N reviews by THIS agent (BOT_NAMES_JSON,
#      NOT bots in general) with ALL their inline comments, bypassing the
#      resolved/outdated/hidden filter (markers shown instead - the latest
#      review deserves the full picture).
#   2. AGENT_REVIEW_HISTORY  - older reviews by this agent, fully filtered
#      (resolved/outdated/hidden threads and comments omitted to save context).
#   3. THREAD_CONTEXT        - everything else: issue comments (bot's
#      non-review comments included, except EXCLUDE_COMMENT_IDS), human/other
#      reviews with their inline comments CORRELATED under each review
#      (filtered), plus the filtering summary.
#
# Contract:
#   $1      : PR number
#   env in  : GH_TOKEN, GITHUB_REPOSITORY, BOT_NAMES_JSON, IGNORE_BOT_NAMES_JSON,
#             COMMENT_FETCH_LIMIT, REVIEW_FETCH_LIMIT, REVIEW_THREAD_FETCH_LIMIT,
#             THREAD_COMMENT_FETCH_LIMIT, PREVIOUS_BOT_REVIEWS_COUNT (default 1),
#             EXCLUDE_COMMENT_IDS (optional comma-separated databaseIds to drop
#             from THREAD_CONTEXT - used to dedup elevated injections),
#             PREFIX_TEXT (optional text prepended verbatim to THREAD_CONTEXT -
#             callers use it for PR metadata/body framing)
#   env out : appends THREAD_CONTEXT / PREVIOUS_BOT_REVIEWS / AGENT_REVIEW_HISTORY
#             to $GITHUB_ENV with unguessable random delimiters; THREAD_CONTEXT
#             ends with a one-line filtering summary
#   exit    : 0 on success (empty-but-valid blocks on no data); 1 on fetch
#             failure (callers treat as degraded context, not fatal)
#
# NOTE: per-thread comment fetch uses `last:` (newest N of each thread); a
# thread longer than THREAD_COMMENT_FETCH_LIMIT drops its OLDEST replies
# first (the thread ROOT survives unless the whole thread exceeds the
# limit - keep the limit >= 10 for the elevated block's fidelity).
# TODO(later): add a time-based or total-count cap for very long threads
# (e.g. PRs with 200+ comments - fetch/keep only the last X). Out of scope
# for now; the fetch limits bound the payload.
set -uo pipefail

PR_NUMBER="${1:?usage: fetch-pr-discussion.sh <pr_number>}"
: "${GH_TOKEN:?}"
: "${GITHUB_REPOSITORY:?}"

BOT_NAMES_JSON="${BOT_NAMES_JSON:-[\"mirrobot\", \"mirrobot-agent\", \"mirrobot-agent[bot]\"]}"
IGNORE_BOT_NAMES_JSON="${IGNORE_BOT_NAMES_JSON:-[\"ellipsis-dev\"]}"
COMMENT_FETCH_LIMIT="${COMMENT_FETCH_LIMIT:-20}"
REVIEW_FETCH_LIMIT="${REVIEW_FETCH_LIMIT:-30}"
REVIEW_THREAD_FETCH_LIMIT="${REVIEW_THREAD_FETCH_LIMIT:-30}"
THREAD_COMMENT_FETCH_LIMIT="${THREAD_COMMENT_FETCH_LIMIT:-10}"
ELEVATED_COUNT="${PREVIOUS_BOT_REVIEWS_COUNT:-1}"
EXCLUDE_COMMENT_IDS="${EXCLUDE_COMMENT_IDS:-}"
PREFIX_TEXT="${PREFIX_TEXT:-}"

repo_owner="${GITHUB_REPOSITORY%/*}"
repo_name="${GITHUB_REPOSITORY#*/}"

GRAPHQL_QUERY='query($owner:String!, $name:String!, $number:Int!, $commentLimit:Int!, $reviewLimit:Int!, $threadLimit:Int!, $threadCommentLimit:Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      comments(last: $commentLimit) {
        nodes {
          databaseId
          author { login }
          body
          createdAt
          isMinimized
          minimizedReason
        }
      }
      reviews(last: $reviewLimit) {
        nodes {
          databaseId
          author { login }
          body
          state
          submittedAt
          url
          isMinimized
          minimizedReason
        }
      }
      reviewThreads(last: $threadLimit) {
        nodes {
          id
          isResolved
          isOutdated
          comments(last: $threadCommentLimit) {
            nodes {
              databaseId
              author { login }
              body
              createdAt
              path
              line
              originalLine
              url
              isMinimized
              minimizedReason
              pullRequestReview {
                databaseId
                isMinimized
                minimizedReason
              }
            }
          }
        }
      }
    }
  }
}'

if ! discussion_data=$(gh api graphql \
  -F owner="$repo_owner" \
  -F name="$repo_name" \
  -F number="$PR_NUMBER" \
  -F commentLimit="$COMMENT_FETCH_LIMIT" \
  -F reviewLimit="$REVIEW_FETCH_LIMIT" \
  -F threadLimit="$REVIEW_THREAD_FETCH_LIMIT" \
  -F threadCommentLimit="$THREAD_COMMENT_FETCH_LIMIT" \
  -f query="$GRAPHQL_QUERY"); then
  echo "::warning::Discussion GraphQL fetch failed for PR #$PR_NUMBER"
  exit 1
fi

# ---- Thread context: issue comments (filtered, optional id exclusion) ----
thread_context=$(printf '%s' "$discussion_data" | jq -r \
  --argjson ignored "$IGNORE_BOT_NAMES_JSON" \
  --arg exclude_ids "$EXCLUDE_COMMENT_IDS" '
  (.data.repository.pullRequest.comments.nodes // [])
  | map(select(
      (.isMinimized != true)
      and (((.author.login? // "unknown") as $login | $ignored | index($login)) | not)
      and (((.databaseId | tostring) as $id | ($exclude_ids | split(",") | map(select(length > 0)) | index($id)) | not))
    ))
  | if length > 0 then
      map("- " + (.author.login? // "unknown") + " at " + (.createdAt // "N/A") + ":\n" + ((.body // "") | tostring) + "\n")
      | join("")
    else "No general comments."
    end
')

# ---- Reviews + correlated inline comments (three-block separation) ----
if ! agent_blocks=$(printf '%s' "$discussion_data" | jq -r \
  --argjson agentbots "$BOT_NAMES_JSON" \
  --argjson ignored "$IGNORE_BOT_NAMES_JSON" \
  --argjson count "$ELEVATED_COUNT" '
  (.data.repository.pullRequest) as $pr |
  (($pr.reviewThreads.nodes // [])
    | map(. as $th | (.comments.nodes // []) | map(. + {thResolved: ($th.isResolved == true), thOutdated: ($th.isOutdated == true)}))
    | flatten) as $allc |
  def thread_ok: (.thResolved != true) and (.thOutdated != true);
  def cmt_ok: (.isMinimized != true) and ((.pullRequestReview.isMinimized // false) != true);
  def agent_cmt: ((.author.login? // "") as $l | $agentbots | index($l)) != null;
  def markers:
    (if .thResolved then " [resolved]" else "" end)
    + (if .thOutdated then " [outdated]" else "" end)
    + (if .isMinimized then " [hidden]" else "" end);
  def fmt_c: ("- " + (.path // "Unknown file") + ":" + (((.line // .originalLine // "N/A")) | tostring) + " (" + (.createdAt // "N/A") + ") by " + (.author.login? // "unknown") + " - " + ((.body // "") | tostring) + markers + " <" + (.url // "") + ">");
  def fmt_c_kept: (if (thread_ok and cmt_ok) then fmt_c else empty end);
  def fmt_c_all: fmt_c;
  def review_block($skipfilter):
    . as $r |
    [ ($allc | map(select((.pullRequestReview.databaseId? // null) == $r.databaseId)) | .[] | (if $skipfilter then fmt_c_all else fmt_c_kept end)) ] as $lines |
    "## " + (.submittedAt // "N/A") + " - " + (.state // "UNKNOWN") + " - " + (.author.login? // "unknown") + " <" + (.url // "") + ">\n"
    + ((.body // "(No summary comment)") | tostring) + "\n"
    + (if ($lines | length) > 0 then "Inline comments:\n" + ($lines | join("\n")) + "\n" else "Inline comments: (none)\n" end);
  # THIS agent reviews, newest first
  (($pr.reviews.nodes // []) | sort_by(.submittedAt) | reverse
    | map(select((.author.login? // "") as $l | $agentbots | index($l)))) as $agent_reviews |
  # Human/other reviews with correlated active comments (agent reviews excluded)
  [ ($pr.reviews.nodes // [])[]?
    | select(
        ((.author.login? // "unknown") as $login | $ignored | index($login) | not)
        and (((.author.login? // "unknown") as $abot | $agentbots | index($abot)) | not)
        and (.isMinimized != true))
    | . as $r
    | [ ($allc | map(select((.pullRequestReview.databaseId? // null) == $r.databaseId)) | .[] | fmt_c_kept) ] as $lines
    | "- " + (.author.login? // "unknown") + " at " + (.submittedAt // "N/A") + " - " + (.state // "UNKNOWN") + " <" + (.url // "") + ">\n"
      + (if ($lines | length) > 0 then "  Inline comments:\n" + ($lines | map("  " + .) | join("\n")) + "\n" else "  (no active inline comments)\n" end)
  ] | join("") as $othertext |
  # Standalone (unlinked) active comments not by this agent
  [ ($allc | .[] | select((.pullRequestReview == null) and thread_ok and cmt_ok and (agent_cmt | not)) | fmt_c) ] as $unlinked |
  ((if ($othertext | length) > 0 then $othertext else "No formal reviews." end)
   + (if ($unlinked | length) > 0 then "\nStandalone inline comments:\n" + ($unlinked | join("\n")) + "\n" else "" end)) as $threadreviews |
  ([ $allc[] | select((thread_ok and cmt_ok) | not) ] | length) as $n_filtered |
  {
    elevated: (([$agent_reviews[0:$count][] | review_block(true)] | join("\n")) | if length > 0 then . else "(No previous reviews by this agent yet.)" end),
    history: (([$agent_reviews[$count:][] | review_block(false)] | join("\n")) | if length > 0 then . else "(No older reviews by this agent.)" end),
    threadreviews: $threadreviews,
    filter_summary: ("<filtering_summary>Context filtering applied: " + ($n_filtered | tostring) + " inline comment(s) excluded (resolved/outdated/hidden threads, or minimized comments in active threads). The elevated block bypasses the filter on purpose.</filtering_summary>")
  }
'); then
  echo "::warning::Discussion block formatting failed for PR #$PR_NUMBER"
  exit 1
fi

elevated=$(printf '%s' "$agent_blocks" | jq -r '.elevated')
history=$(printf '%s' "$agent_blocks" | jq -r '.history')
threadreviews=$(printf '%s' "$agent_blocks" | jq -r '.threadreviews')

# One-line filtering summary (what the caller's prompt used to show).
filter_summary=$(printf '%s' "$agent_blocks" | jq -r '.filter_summary')

full_thread="${PREFIX_TEXT}
${thread_context}
${threadreviews}
${filter_summary}"

# ---- Export with unguessable delimiters ----
TC_DELIMITER="GH_THREAD_CONTEXT_$(openssl rand -hex 8)"
{
  printf 'THREAD_CONTEXT<<%s\n' "$TC_DELIMITER"
  printf '%s\n' "$full_thread"
  printf '%s\n' "$TC_DELIMITER"
} >> "$GITHUB_ENV"

EL_DELIMITER="GH_ELEVATED_$(openssl rand -hex 8)"
{
  printf 'PREVIOUS_BOT_REVIEWS<<%s\n' "$EL_DELIMITER"
  printf '%s\n' "$elevated"
  printf '%s\n' "$EL_DELIMITER"
} >> "$GITHUB_ENV"

HIST_DELIMITER="GH_AGENT_HISTORY_$(openssl rand -hex 8)"
{
  printf 'AGENT_REVIEW_HISTORY<<%s\n' "$HIST_DELIMITER"
  printf '%s\n' "$history"
  printf '%s\n' "$HIST_DELIMITER"
} >> "$GITHUB_ENV"

echo "Discussion context built for PR #$PR_NUMBER: elevated=${ELEVATED_COUNT} agent review(s), thread context ready."
exit 0
