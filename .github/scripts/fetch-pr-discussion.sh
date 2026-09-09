#!/usr/bin/env bash
# fetch-pr-discussion.sh — shared PR discussion context for agent workflows.
#
# Single source of truth for the THREE-BLOCK context separation used by both
# PR Review and Compliance Check (identical machinery, identical semantics):
#   1. PREVIOUS_BOT_REVIEWS  - newest N reviews by THIS agent (BOT_NAMES_JSON,
#      NOT bots in general) with their inline comments, bypassing the
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
# CONTEXT BUDGET (CONTEXT_LIMITS_JSON repo variable; per-key defaults):
#   {
#     "comments": 30,               // conversation comments, newest first
#     "reviews": 15,                // review objects, newest first
#     "own-reviews": 5,             // SAFEGUARD: this agent's own newest
#                                    //   reviews are ALWAYS included, even
#                                    //   beyond the reviews window
#     "threads-per-review": 25,     // inline threads per selected review
#     "thread-comments": 10,        // replies per review-linked thread
#     "orphan-threads": 20,         // review-less threads ("Add single
#                                    //   comment"), newest first
#     "orphan-thread-comments": 10, // replies per orphaned thread
#     "body-chars": 3000            // per-post body cap (comments AND review
#                                    //   summaries); longer bodies are cut
#                                    //   with a visible [body truncated]
#                                    //   marker - count caps alone cannot
#                                    //   bound a 50KB single comment
#   }
#   Everything is "up to", newest first, deduped. FILTER BEFORE CAP, in every
#   window: hidden (minimized) content never consumes a slot; resolved/outdated
#   content never does outside the elevated block (the elevated block alone
#   bypasses resolved/outdated - with markers - because it IS the agent's
#   memory of its own findings). Windows OVERFILL 3x at fetch time and a
#   cursor catch-up page runs when noise truncates one while slots stay
#   unfilled, so budget numbers count content shown, never content wasted
#   (thread-reply windows overfill but do not paginate per-thread - rate
#   frugality, documented bound). Per-comment `outdated` gives mixed
#   threads precise treatment in filtered blocks: a fresh reply on a moved
#   anchor survives; the stale anchor itself drops.
#   Fetch ceilings (implementation bounds, not tunables): reviews window 64,
#   threads window 100 - allocation happens inside them; on truly enormous
#   PRs older content beyond a window is unreachable (documented ceiling).
#   Malformed JSON: warning + per-key defaults (a broken knob must be
#   visible, not fatal - unlike AGENT_MODELS_JSON it cannot brick identity).
#
# NOISE FILTERING (other-people content only; never the agent's own blocks):
#   CONTEXT_IGNORE_AUTHORS   - comma-separated logins whose posts are dropped
#                              outright (repo variable). Default: empty.
#   CONTEXT_FILTER_PATTERNS_JSON - JSON array of regex snippets (repo
#                              variable). Any case-insensitive match on a
#                              post's body drops that post. Setting the
#                              variable REPLACES the baked-in defaults
#                              (defaults: known AI-reviewer noise). Malformed
#                              JSON falls back to defaults with a warning.
#
# Contract:
#   $1      : PR number
#   env in  : GH_TOKEN, GITHUB_REPOSITORY, BOT_NAMES_JSON (resolved by
#             bot-config.sh), CONTEXT_LIMITS_JSON (optional; per-key defaults),
#             PREVIOUS_BOT_REVIEWS_COUNT (default 1 - the ELEVATED count,
#             distinct from the own-reviews fetch safeguard),
#             EXCLUDE_COMMENT_IDS (optional comma-separated databaseIds to drop
#             from THREAD_CONTEXT - used to dedup elevated injections),
#             PREFIX_TEXT (optional text prepended verbatim to THREAD_CONTEXT),
#             CONTEXT_IGNORE_AUTHORS (optional), CONTEXT_FILTER_PATTERNS_JSON (optional)
#   env out : appends THREAD_CONTEXT / PREVIOUS_BOT_REVIEWS / AGENT_REVIEW_HISTORY
#             to $GITHUB_ENV with unguessable random delimiters; THREAD_CONTEXT
#             ends with a one-line filtering summary
#   exit    : 0 on success (empty-but-valid blocks on no data); 1 on fetch
#             failure (callers treat as degraded context, not fatal)
set -uo pipefail

PR_NUMBER="${1:?usage: fetch-pr-discussion.sh <pr_number>}"
: "${GH_TOKEN:?}"
: "${GITHUB_REPOSITORY:?}"

BOT_NAMES_JSON="${BOT_NAMES_JSON:-[\"mirrobot-agent\", \"mirrobot-agent[bot]\"]}"
ELEVATED_COUNT="${PREVIOUS_BOT_REVIEWS_COUNT:-1}"
EXCLUDE_COMMENT_IDS="${EXCLUDE_COMMENT_IDS:-}"
PREFIX_TEXT="${PREFIX_TEXT:-}"

# ---- Context budget: CONTEXT_LIMITS_JSON over per-key defaults -------------
LIM_COMMENTS=30; LIM_REVIEWS=15; LIM_OWN=5
LIM_THREADS_PER_REVIEW=25; LIM_THREAD_COMMENTS=10
LIM_ORPHAN_THREADS=20; LIM_ORPHAN_THREAD_COMMENTS=10
LIM_BODY_CHARS=3000
if [ -n "${CONTEXT_LIMITS_JSON:-}" ]; then
  if printf '%s' "$CONTEXT_LIMITS_JSON" | jq -e 'type == "object"' >/dev/null 2>&1; then
    LIM_COMMENTS=$(printf '%s' "$CONTEXT_LIMITS_JSON" | jq -r '.comments // 30')
    LIM_REVIEWS=$(printf '%s' "$CONTEXT_LIMITS_JSON" | jq -r '.reviews // 15')
    LIM_OWN=$(printf '%s' "$CONTEXT_LIMITS_JSON" | jq -r '."own-reviews" // 5')
    LIM_THREADS_PER_REVIEW=$(printf '%s' "$CONTEXT_LIMITS_JSON" | jq -r '."threads-per-review" // 25')
    LIM_THREAD_COMMENTS=$(printf '%s' "$CONTEXT_LIMITS_JSON" | jq -r '."thread-comments" // 10')
    LIM_ORPHAN_THREADS=$(printf '%s' "$CONTEXT_LIMITS_JSON" | jq -r '."orphan-threads" // 20')
    LIM_ORPHAN_THREAD_COMMENTS=$(printf '%s' "$CONTEXT_LIMITS_JSON" | jq -r '."orphan-thread-comments" // 10')
    LIM_BODY_CHARS=$(printf '%s' "$CONTEXT_LIMITS_JSON" | jq -r '."body-chars" // 3000')
  else
    echo "::warning::CONTEXT_LIMITS_JSON is not a JSON object; using per-key defaults."
  fi
fi
# Implementation fetch ceilings (see header).
REVIEW_WINDOW=64

# ---- Overfetch + fill loop --------------------------------------------------
# Budget slots are CONTENT SHOWN, never content fetched: a hidden or noisy
# post inside the raw window must not silently consume a context slot.
# GraphQL charges 1 point per REQUEST (node count is free), so the cheap
# strategy is: overfetch 3x the window in the same single query, filter,
# take the newest N. Only when a window was truncated by the fetch limit
# AND the filtered survivors still under-fill it does ONE cursor catch-up
# page run (max 1 extra request per flat window; the common clean case
# costs exactly one request total, same as before). Thread-reply windows
# share the 3x overfetch but do not paginate per-thread (rate frugality;
# documented bound).
CMT_FETCH=$(( LIM_COMMENTS * 3 ));   [ "$CMT_FETCH" -gt 100 ] && CMT_FETCH=100
REV_FETCH=$(( REVIEW_WINDOW * 2 ));  [ "$REV_FETCH" -gt 100 ] && REV_FETCH=100
THREAD_WINDOW=100
THREAD_COMMENT_FETCH=$(( LIM_THREAD_COMMENTS > LIM_ORPHAN_THREAD_COMMENTS ? LIM_THREAD_COMMENTS : LIM_ORPHAN_THREAD_COMMENTS ))
THREAD_COMMENT_FETCH=$(( THREAD_COMMENT_FETCH * 3 )); [ "$THREAD_COMMENT_FETCH" -gt 100 ] && THREAD_COMMENT_FETCH=100

# Noise-filter configuration (repo variables; see header).
CONTEXT_IGNORE_AUTHORS="${CONTEXT_IGNORE_AUTHORS:-}"
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
        pageInfo { hasPreviousPage startCursor }
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
        pageInfo { hasPreviousPage startCursor }
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
              outdated
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
  -F commentLimit="$CMT_FETCH" \
  -F reviewLimit="$REV_FETCH" \
  -F threadLimit="$THREAD_WINDOW" \
  -F threadCommentLimit="$THREAD_COMMENT_FETCH" \
  -f query="$GRAPHQL_QUERY"); then
  echo "::warning::Discussion GraphQL fetch failed for PR #$PR_NUMBER"
  exit 1
fi

# ---- Fill loop catch-up (at most ONE extra request per flat window) --------
# Runs only when a window was truncated AND filtering left it under-filled:
# fetch the next-older page and prepend its nodes (chronological order is
# ascending; older nodes go first). Steady state: zero extra requests.
catch_up() { # $1 = connection (comments|reviews), $2 = cursor
  local conn="$1" cursor="$2" page extra fields
  # Per-connection field sets: GraphQL validates the selection against the
  # node type, so a shared query must not request review fields on comments.
  case "$conn" in
    comments) fields='databaseId author { login } body createdAt isMinimized minimizedReason' ;;
    reviews)  fields='databaseId author { login } body createdAt isMinimized minimizedReason state submittedAt url' ;;
  esac
  page=$(gh api graphql \
    -F owner="$repo_owner" -F name="$repo_name" -F number="$PR_NUMBER" \
    -F limit=100 -f cursor="$cursor" -f query="
      query(\$owner:String!, \$name:String!, \$number:Int!, \$limit:Int!, \$cursor:String!) {
        repository(owner: \$owner, name: \$name) { pullRequest(number: \$number) {
          ${conn}(last: \$limit, before: \$cursor) {
            nodes { ${fields} }
            pageInfo { hasPreviousPage startCursor }
          }
        } }
      }") || return 0   # transient failure: proceed with what we have
  extra=$(printf '%s' "$page" | jq -c --arg conn "$conn" '.data.repository.pullRequest[$conn].nodes // []')
  discussion_data=$(printf '%s' "$discussion_data" | jq -c --arg conn "$conn" --argjson extra "$extra" \
    '.data.repository.pullRequest[$conn].nodes = ($extra + .data.repository.pullRequest[$conn].nodes)')
}

pr_node=$(printf '%s' "$discussion_data" | jq -c '.data.repository.pullRequest')

# Flat comments: truncated + under-filled after the same filters the
# renderer applies (hidden, ignored authors, noise patterns, exclusions)?
cmt_survivors=$(printf '%s' "$pr_node" | jq --arg ignore "$(printf '%s' "$CONTEXT_IGNORE_AUTHORS" | tr '[:upper:]' '[:lower:]')" --argjson patterns "$FILTER_PATTERNS_JSON" --arg exclude "$EXCLUDE_COMMENT_IDS" -r '
  ($ignore | split(",") | map(select(length > 0))) as $ig |
  def noisy: ((.body // "") as $b | [ $patterns[] | . as $p | select($b | test("(?i)" + $p)) ] | length > 0);
  [.comments.nodes[] | select((.isMinimized != true) and ((((.author.login? // "unknown") | ascii_downcase) as $l | $ig | index($l)) | not) and (((.databaseId|tostring) as $id | ($exclude|split(",")|map(select(length>0))|index($id))|not)) and (noisy|not))] | length')
if [ "$cmt_survivors" -lt "$LIM_COMMENTS" ] \
   && [ "$(printf '%s' "$pr_node" | jq '.comments.nodes | length')" -ge "$CMT_FETCH" ] \
   && [ "$(printf '%s' "$pr_node" | jq '.comments.pageInfo.hasPreviousPage')" = "true" ]; then
  catch_up comments "$(printf '%s' "$pr_node" | jq -r '.comments.pageInfo.startCursor')"
fi

# Reviews: truncated + own-newest or other-newest selection still short?
pr_node=$(printf '%s' "$discussion_data" | jq -c '.data.repository.pullRequest')
rv_short=$(printf '%s' "$pr_node" | jq --argjson agentbots "$BOT_NAMES_JSON" --argjson limOwn "$LIM_OWN" --argjson limReviews "$LIM_REVIEWS" --arg ignore "$(printf '%s' "$CONTEXT_IGNORE_AUTHORS" | tr '[:upper:]' '[:lower:]')" --argjson patterns "$FILTER_PATTERNS_JSON" -r '
  ($ignore | split(",") | map(select(length > 0))) as $ig |
  def is_own: ((.author.login? // "" | ascii_downcase) as $l | $agentbots | index($l)) != null;
  def noisy: ((.body // "") as $b | [ $patterns[] | . as $p | select($b | test("(?i)" + $p)) ] | length > 0);
  ([(.reviews.nodes // [])[] | select(is_own and (.isMinimized != true))] | length) as $own |
  ([(.reviews.nodes // [])[] | select((is_own|not) and (.isMinimized != true) and (noisy|not) and ((((.author.login? // "unknown")|ascii_downcase) as $l | $ig | index($l))|not))] | length) as $other |
  if ($own < $limOwn or $other < $limReviews) then "short" else "ok" end')
if [ "$rv_short" = "short" ] \
   && [ "$(printf '%s' "$pr_node" | jq '.reviews.nodes | length')" -ge "$REV_FETCH" ] \
   && [ "$(printf '%s' "$pr_node" | jq '.reviews.pageInfo.hasPreviousPage')" = "true" ]; then
  catch_up reviews "$(printf '%s' "$pr_node" | jq -r '.reviews.pageInfo.startCursor')"
fi
# The big downstream program re-filters and re-selects from the (possibly
# extended) node arrays; overfetched surplus is dropped there by the caps.

# ---- Thread context: issue comments (filtered, optional id exclusion) ----
# The upstream fetch overfills this window on purpose (filter-before-cap:
# hidden/noisy posts never consume a slot), so the cap lives HERE now:
# newest $limComments survivors of the filtered array (chronological
# ascending, hence the negative slice).
thread_context=$(printf '%s' "$discussion_data" | jq -r \
  --arg ignore_authors "$(printf '%s' "$CONTEXT_IGNORE_AUTHORS" | tr '[:upper:]' '[:lower:]')" \
  --argjson patterns "$FILTER_PATTERNS_JSON" \
  --argjson limComments "$LIM_COMMENTS" \
  --argjson bodyChars "$LIM_BODY_CHARS" \
  --arg exclude_ids "$EXCLUDE_COMMENT_IDS" '
  ($ignore_authors | split(",") | map(select(length > 0))) as $ignored |
  def clip($s): if ($s | length) > $bodyChars then ($s[0:$bodyChars] + "\n[body truncated]") else $s end;
  def noisy: ((.body // "") as $b | [ $patterns[] | . as $p | select($b | test("(?i)" + $p)) ] | length > 0);
  (.data.repository.pullRequest.comments.nodes // [])
  | map(select(
      (.isMinimized != true)
      and (((.author.login? // "unknown") | ascii_downcase) as $login | $ignored | index($login) | not)
      and (((.databaseId | tostring) as $id | ($exclude_ids | split(",") | map(select(length > 0)) | index($id)) | not))
      and (noisy | not)
    ))
  | if length > $limComments then .[($limComments * -1):] else . end
  | if length > 0 then
      map("- " + (.author.login? // "unknown") + " at " + (.createdAt // "N/A") + ":\n" + clip((.body // "") | tostring) + "\n")
      | join("")
    else "No general comments."
    end
')

# ---- Reviews + allocated inline comments (three-block separation) ---------
# Selection: newest LIM_REVIEWS reviews overall, PLUS this agent's own newest
# LIM_OWN always included (the safeguard). Allocation: per selected review,
# its newest LIM_THREADS_PER_REVIEW threads, each capped at LIM_THREAD_COMMENTS
# replies; orphan threads (review-less) newest LIM_ORPHAN_THREADS with
# LIM_ORPHAN_THREAD_COMMENTS replies. FILTER BEFORE CAP (hidden never counts;
# resolved/outdated never counts outside the elevated block; per-comment
# `outdated` drops stale anchors while fresh replies survive).
if ! agent_blocks=$(printf '%s' "$discussion_data" | jq -r \
  --argjson agentbots "$BOT_NAMES_JSON" \
  --arg ignore_authors "$(printf '%s' "$CONTEXT_IGNORE_AUTHORS" | tr '[:upper:]' '[:lower:]')" \
  --argjson patterns "$FILTER_PATTERNS_JSON" \
  --argjson count "$ELEVATED_COUNT" \
  --argjson limReviews "$LIM_REVIEWS" \
  --argjson limOwn "$LIM_OWN" \
  --argjson limThreadsPerReview "$LIM_THREADS_PER_REVIEW" \
  --argjson limThreadComments "$LIM_THREAD_COMMENTS" \
  --argjson limOrphanThreads "$LIM_ORPHAN_THREADS" \
  --argjson limOrphanThreadComments "$LIM_ORPHAN_THREAD_COMMENTS" \
  --argjson bodyChars "$LIM_BODY_CHARS" '
  ($ignore_authors | split(",") | map(select(length > 0))) as $ignored |
  def clip($s): if ($s | length) > $bodyChars then ($s[0:$bodyChars] + "\n[body truncated]") else $s end;
  def noisy: ((.body // "") as $b | [ $patterns[] | . as $p | select($b | test("(?i)" + $p)) ] | length > 0);
  (.data.repository.pullRequest) as $pr |
  def is_own: ((.author.login? // "" | ascii_downcase) as $l | $agentbots | index($l)) != null;
  # ---- flat comments with their thread context attached --------------------
  (($pr.reviewThreads.nodes // [])
    | to_entries
    | map(.key as $thId | .value as $th | (.value.comments.nodes // [])
        | map(. + {thId: $thId, thResolved: ($th.isResolved == true), thOutdated: ($th.isOutdated == true)}))
    | flatten) as $allc |
  def cmt_ok: (.isMinimized != true) and ((.pullRequestReview.isMinimized // false) != true);
  def thread_ok: (.thResolved != true) and (.thOutdated != true);
  def cmt_fresh: ((.outdated // false) != true);
  def agent_cmt: ((.author.login? // "" | ascii_downcase) as $l | $agentbots | index($l)) != null;
  def markers:
    (if .thResolved then " [resolved]" else "" end)
    + (if .thOutdated then " [outdated]" else "" end)
    + (if .isMinimized then " [hidden]" else "" end);
  def fmt_c: ("- " + (.path // "Unknown file") + ":" + (((.line // .originalLine // "N/A")) | tostring) + " (" + (.createdAt // "N/A") + ") by " + (.author.login? // "unknown") + " - " + clip((.body // "") | tostring) + markers + " <" + (.url // "") + ">");
  # ---- review selection: newest limReviews + own-newest limOwn (safeguard) -
  (($pr.reviews.nodes // []) | sort_by(.submittedAt) | reverse) as $reviews_new |
  ([ $reviews_new[] | select(is_own and (.isMinimized != true)) ] | .[0:$limOwn]) as $own_sel |
  ([ $reviews_new[] | select((is_own | not) and (.isMinimized != true) and (noisy | not) and (((.author.login? // "unknown" | ascii_downcase) as $login | $ignored | index($login)) | not)) ] | .[0:$limReviews]) as $other_sel |
  (($own_sel + $other_sel) | unique_by(.databaseId) | sort_by(.submittedAt) | reverse) as $selected |
  ([ $selected[] | select(is_own) ]) as $agent_reviews |
  # ---- per-review thread allocation (newest threads, capped) ---------------
  # FILTER BEFORE CAP, strictly: inside each thread, filter comments FIRST
  # (hidden never counts; resolved/outdated/stale-anchors never count outside
  # the elevated block - the elevated block alone bypasses resolved/outdated,
  # with markers, but never the hidden drop), drop threads left with zero
  # survivors so they consume no slot, THEN take the newest
  # threads-per-review threads, and only then the newest thread-comments
  # replies per surviving thread.
  def rv_alloc($rid; $skipfilter):
    [ $allc[] | select((.pullRequestReview.databaseId? // null) == $rid) ]
    | group_by(.thId)
    | map(sort_by(.createdAt) | reverse
        | (if $skipfilter then map(select(cmt_ok)) else map(select(cmt_ok and thread_ok and cmt_fresh)) end))
    | map(select(length > 0))
    | sort_by((.[0].createdAt // "0")) | reverse
    | .[0:$limThreadsPerReview]
    | map(.[0:$limThreadComments])
    | flatten;
  def review_block($skipfilter):
    . as $r |
    (rv_alloc($r.databaseId; $skipfilter)) as $lines |
    "## " + (.submittedAt // "N/A") + " - " + (.state // "UNKNOWN") + " - " + (.author.login? // "unknown") + " <" + (.url // "") + ">\n"
    + clip((.body // "(No summary comment)") | tostring) + "\n"
    + (if ($lines | length) > 0 then "Inline comments:\n" + ($lines | map(fmt_c) | join("\n")) + "\n" else "Inline comments: (none)\n" end);
  def dismissed_note: if any(.[]?; .state == "DISMISSED") then "\nNote: DISMISSED here usually means an APPROVED review auto-cleared by a later push - the Verdict line in the body holds the original verdict. Treat it as re-review-the-delta, not a wrong review.\n" else "" end;
  # ---- orphan threads: review-less, filtered, newest, capped ----------------
  # Same filter-before-cap discipline: filter the comments of each thread
  # first, drop emptied threads (no slot consumed), then cap the newest
  # orphan threads and their replies.
  ([ $allc | group_by(.thId)[]
      | select([.[] | select(.pullRequestReview != null)] | length == 0) ]
    | map(sort_by(.createdAt) | reverse
        | map(select(cmt_ok and thread_ok and cmt_fresh)))
    | map(select(length > 0))
    | sort_by((.[0].createdAt // "0")) | reverse
    | .[0:$limOrphanThreads]
    | map(.[0:$limOrphanThreadComments])
    | flatten) as $orphan_cmts |
  [ $orphan_cmts[] | select(agent_cmt | not) | fmt_c ] as $unlinked |
  # ---- other reviews render (selected only, filtered comments) -------------
  [ $selected[]
    | select(is_own | not)
    | . as $r
    | (rv_alloc($r.databaseId; false)) as $lines
    | "- " + (.author.login? // "unknown") + " at " + (.submittedAt // "N/A") + " - " + (.state // "UNKNOWN") + " <" + (.url // "") + ">\n"
      + (clip((.body // "") | tostring) | if length > 0 then "  " + . + "\n" else "" end)
      + (if ($lines | length) > 0 then "  Inline comments:\n" + ($lines | map(fmt_c) | join("\n")) + "\n" else "  (no active inline comments)\n" end)
  ] | join("") as $othertext |
  ([ $allc[] | select((cmt_ok and (thread_ok and cmt_fresh)) | not) ] | length) as $n_filtered |
  {
    elevated: ((([$agent_reviews[0:$count][] | review_block(true)] | join("\n")) | if length > 0 then . else "(No previous reviews by this agent yet.)" end) + ($agent_reviews[0:$count] | dismissed_note)),
    history: ((([$agent_reviews[$count:][] | review_block(false)] | join("\n")) | if length > 0 then . else "(No older reviews by this agent.)" end) + ($agent_reviews[$count:] | dismissed_note)),
    threadreviews: ((if ($othertext | length) > 0 then $othertext else "No formal reviews." end)
   + (if ($unlinked | length) > 0 then "\nStandalone inline comments (no review):\n" + ($unlinked | join("\n")) + "\n" else "" end)
   + (($selected | map(select(is_own | not))) | dismissed_note)),
    filter_summary: ("<filtering_summary>Context filtering applied: " + ($n_filtered | tostring) + " inline comment(s) excluded (resolved/outdated/hidden threads, stale anchors, or minimized comments in active threads); hidden (minimized) content excluded everywhere, own reviews included; AI-reviewer noise posts (rate-limit/skip notices) and ignored authors dropped. The elevated block bypasses only the resolved/outdated filter, on purpose. Context budget (newest-first, filter-before-cap): " + ($limReviews | tostring) + " reviews + " + ($limOwn | tostring) + " own safeguard, " + ($limThreadsPerReview | tostring) + " threads/review, " + ($limThreadComments | tostring) + " replies/thread, " + ($limOrphanThreads | tostring) + " orphan threads.</filtering_summary>")
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

echo "Discussion context built for PR #$PR_NUMBER: budget reviews=$LIM_REVIEWS+own$LIM_OWN threads/review=$LIM_THREADS_PER_REVIEW thread-comments=$LIM_THREAD_COMMENTS orphans=${LIM_ORPHAN_THREADS}x${LIM_ORPHAN_THREAD_COMMENTS}; elevated=${ELEVATED_COUNT}."
exit 0
