#!/usr/bin/env bash
# fetch-pr-discussion.sh — shared PR discussion context for agent workflows.
#
# Single source of truth for the THREE-BLOCK context separation used by both
# PR Review and Compliance Check (identical machinery, identical semantics):
#   1. PREVIOUS_BOT_REVIEWS  - newest N reviews by THIS agent (BOT_NAMES_JSON,
#      NOT bots in general) with ALL their inline comments, bypassing the
#      resolved/outdated filter (markers shown instead - the latest review
#      deserves the full picture). Minimized (hidden) content is the ONE
#      exception: hidden stays hidden everywhere, own content included.
#   2. AGENT_REVIEW_HISTORY  - older reviews by this agent, fully filtered
#      (resolved/outdated/hidden threads and comments omitted to save context).
#   3. THREAD_CONTEXT        - everything else: issue comments (bot's
#      non-review comments included, except EXCLUDE_COMMENT_IDS), human/other
#      reviews with their inline comments CORRELATED under each review
#      (filtered), plus the filtering summary.
#
# NOISE FILTERING (other-people content only; never the agent's own blocks):
#   CONTEXT_IGNORE_AUTHORS   - comma-separated logins whose posts are dropped
#                              outright (repo variable). Default: empty.
#   CONTEXT_FILTER_PATTERNS_JSON - JSON array of regex snippets (repo
#                              variable). Any case-insensitive match on a
#                              post's body drops that post. JSON array means
#                              patterns may contain commas/semicolons/pipes;
#                              backslashes double as JSON escapes ("\\.").
#                              Setting the variable REPLACES the baked-in
#                              defaults (defaults: known AI-reviewer noise -
#                              rate-limit notices, review-skipped/Too-many-
#                              files posts, greptile status channel). Malformed
#                              JSON falls back to defaults with a warning.
#
# Contract:
#   $1      : PR number
#   env in  : GH_TOKEN, GITHUB_REPOSITORY, BOT_NAMES_JSON, COMMENT_FETCH_LIMIT,
#             REVIEW_FETCH_LIMIT, REVIEW_THREAD_FETCH_LIMIT,
#             THREAD_COMMENT_FETCH_LIMIT, PREVIOUS_BOT_REVIEWS_COUNT (default 1),
#             EXCLUDE_COMMENT_IDS (optional comma-separated databaseIds to drop
#             from THREAD_CONTEXT - used to dedup elevated injections),
#             PREFIX_TEXT (optional text prepended verbatim to THREAD_CONTEXT -
#             callers use it for PR metadata/body framing),
#             CONTEXT_IGNORE_AUTHORS (optional), CONTEXT_FILTER_PATTERNS_JSON (optional)
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

BOT_NAMES_JSON="${BOT_NAMES_JSON:-[\"mirrobot-agent\", \"mirrobot-agent[bot]\"]}"
COMMENT_FETCH_LIMIT="${COMMENT_FETCH_LIMIT:-20}"
REVIEW_FETCH_LIMIT="${REVIEW_FETCH_LIMIT:-30}"
REVIEW_THREAD_FETCH_LIMIT="${REVIEW_THREAD_FETCH_LIMIT:-30}"
THREAD_COMMENT_FETCH_LIMIT="${THREAD_COMMENT_FETCH_LIMIT:-10}"
ELEVATED_COUNT="${PREVIOUS_BOT_REVIEWS_COUNT:-1}"
EXCLUDE_COMMENT_IDS="${EXCLUDE_COMMENT_IDS:-}"
PREFIX_TEXT="${PREFIX_TEXT:-}"

# Noise-filter configuration (repo variables; see header).
CONTEXT_IGNORE_AUTHORS="${CONTEXT_IGNORE_AUTHORS:-}"
# Baked defaults: the demonstrated noise classes of AI reviewers (coderabbit
# rate-limit/skip posts, greptile status/skip posts). Substantive reviews -
# walkthroughs, overviews, inline findings - NEVER match these.
DEFAULT_FILTER_PATTERNS_JSON='["rate limited by coderabbit\\.ai","No actionable comments were generated","Review skipped","Too many files","<!-- greptile-status -->","Too many files changed for review"]'
if [ -n "${CONTEXT_FILTER_PATTERNS_JSON:-}" ]; then
  if printf '%s' "$CONTEXT_FILTER_PATTERNS_JSON" | jq -e 'type == "array"' >/dev/null 2>&1; then
    FILTER_PATTERNS_JSON="$CONTEXT_FILTER_PATTERNS_JSON"
  else
    echo "::warning::CONTEXT_FILTER_PATTERNS_JSON is not a valid JSON array; using built-in defaults."
    FILTER_PATTERNS_JSON="$DEFAULT_FILTER_PATTERNS_JSON"
  fi
else
  FILTER_PATTERNS_JSON="$DEFAULT_FILTER_PATTERNS_JSON"
fi

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
  --arg ignore_authors "$(printf '%s' "$CONTEXT_IGNORE_AUTHORS" | tr '[:upper:]' '[:lower:]')" \
  --argjson patterns "$FILTER_PATTERNS_JSON" \
  --arg exclude_ids "$EXCLUDE_COMMENT_IDS" '
  ($ignore_authors | split(",") | map(select(length > 0))) as $ignored |
  def noisy: ((.body // "") as $b | [ $patterns[] | . as $p | select($b | test("(?i)" + $p)) ] | length > 0);
  (.data.repository.pullRequest.comments.nodes // [])
  | map(select(
      (.isMinimized != true)
      and (((.author.login? // "unknown") | ascii_downcase) as $login | $ignored | index($login) | not)
      and (((.databaseId | tostring) as $id | ($exclude_ids | split(",") | map(select(length > 0)) | index($id)) | not))
      and (noisy | not)
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
  --arg ignore_authors "$(printf '%s' "$CONTEXT_IGNORE_AUTHORS" | tr '[:upper:]' '[:lower:]')" \
  --argjson patterns "$FILTER_PATTERNS_JSON" \
  --argjson count "$ELEVATED_COUNT" '
  ($ignore_authors | split(",") | map(select(length > 0))) as $ignored |
  def noisy: ((.body // "") as $b | [ $patterns[] | . as $p | select($b | test("(?i)" + $p)) ] | length > 0);
  (.data.repository.pullRequest) as $pr |
  (($pr.reviewThreads.nodes // [])
    | map(. as $th | (.comments.nodes // []) | map(. + {thResolved: ($th.isResolved == true), thOutdated: ($th.isOutdated == true)}))
    | flatten) as $allc |
  def thread_ok: (.thResolved != true) and (.thOutdated != true);
  def cmt_ok: (.isMinimized != true) and ((.pullRequestReview.isMinimized // false) != true);
  def agent_cmt: ((.author.login? // "" | ascii_downcase) as $l | $agentbots | index($l)) != null;
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
  # (agent-reviews selection note: minimized stays hidden for the agent own
  # reviews too - hidden is hidden from ANYONE; only resolved/outdated
  # markers are bypassed, in the elevated block, on purpose)
  (($pr.reviews.nodes // []) | sort_by(.submittedAt) | reverse
    | map(select(((.author.login? // "" | ascii_downcase) as $l | $agentbots | index($l)) and (.isMinimized != true)))) as $agent_reviews |
  # Section-level clarification: GitHub auto-dismisses APPROVED reviews on new
  # pushes (CHANGES_REQUESTED/COMMENT survive); the state field loses the
  # original verdict. One note per section, only when a DISMISSED review is in it.
  def dismissed_note: if any(.[]?; .state == "DISMISSED") then "\nNote: DISMISSED here usually means an APPROVED review auto-cleared by a later push - the Verdict line in the body holds the original verdict. Treat it as re-review-the-delta, not a wrong review.\n" else "" end;
  # Human/other reviews with correlated active comments (agent reviews excluded)
  (($pr.reviews.nodes // [])
    | map(select(
        (((.author.login? // "unknown" | ascii_downcase) as $login | $ignored | index($login)) | not)
        and (((.author.login? // "unknown" | ascii_downcase) as $abot | $agentbots | index($abot)) | not)
        and (.isMinimized != true)
        and (noisy | not)))) as $other_reviews |
  # ($other_reviews already carries these filters - never duplicate the
  # predicate list; duplicated predicates drift apart on later edits.)
  [ $other_reviews[]?
    | . as $r
    | [ ($allc | map(select((.pullRequestReview.databaseId? // null) == $r.databaseId)) | .[] | fmt_c_kept) ] as $lines
    | "- " + (.author.login? // "unknown") + " at " + (.submittedAt // "N/A") + " - " + (.state // "UNKNOWN") + " <" + (.url // "") + ">\n"
      + ((.body // "") | tostring | if length > 0 then "  " + . + "\n" else "" end)
      + (if ($lines | length) > 0 then "  Inline comments:\n" + ($lines | map("  " + .) | join("\n")) + "\n" else "  (no active inline comments)\n" end)
  ] | join("") as $othertext |
  # Standalone (unlinked) active comments not by this agent
  [ ($allc | .[] | select((.pullRequestReview == null) and thread_ok and cmt_ok and (agent_cmt | not))) | fmt_c ] as $unlinked |
  ((if ($othertext | length) > 0 then $othertext else "No formal reviews." end)
   + (if ($unlinked | length) > 0 then "\nStandalone inline comments:\n" + ($unlinked | join("\n")) + "\n" else "" end)) as $threadreviews |
  ([ $allc[] | select((thread_ok and cmt_ok) | not) ] | length) as $n_filtered |
  {
    elevated: ((([$agent_reviews[0:$count][] | review_block(true)] | join("\n")) | if length > 0 then . else "(No previous reviews by this agent yet.)" end) + ($agent_reviews[0:$count] | dismissed_note)),
    history: ((([$agent_reviews[$count:][] | review_block(false)] | join("\n")) | if length > 0 then . else "(No older reviews by this agent.)" end) + ($agent_reviews[$count:] | dismissed_note)),
    threadreviews: ($threadreviews + ($other_reviews | dismissed_note)),
    filter_summary: ("<filtering_summary>Context filtering applied: " + ($n_filtered | tostring) + " inline comment(s) excluded (resolved/outdated/hidden threads, or minimized comments in active threads); hidden (minimized) content excluded everywhere, own reviews included; AI-reviewer noise posts (rate-limit/skip notices) and ignored authors dropped. The elevated block bypasses only the resolved/outdated filter, on purpose.</filtering_summary>")
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
