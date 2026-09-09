#!/usr/bin/env bash
# minimized-nodes.sh - minimized (hidden) review + comment node ids for a PR
# ============================================================================
# WHY: GitHub does NOT propagate a review-level hide to its inline comments,
#      and REST endpoints expose no minimization at all - only GraphQL does.
#      Hiding a review means the operator wanted it GONE: it must not count
#      as reviewed coverage (marker anchoring, follow-up selection, report
#      detection) anywhere. This script is the single source of "what is
#      hidden" for every consumer (pr-review/bot-reply detection steps,
#      generate-review-kit.sh marker hunt, compliance report detection).
#
# usage  : minimized-nodes.sh <pr_number>
#   env in : GH_TOKEN (required), GITHUB_REPOSITORY (required),
#            QUERY_REPO (optional OWNER/NAME override - bot-reply's PR mode
#            detects against the TARGET repo, not the home repo)
#   env out: none (stdout only)
#   stdout : one JSON object {"reviews":[<node-id>,...],"comments":[<node-id>,...]}
#            - reviews: node ids of MINIMIZED pull request reviews
#            - comments: node ids of MINIMIZED issue comments on the PR
#            (node ids join directly against REST .node_id and
#             gh pr view --json id fields; databaseIds do NOT)
#   exit  : 0 on success (empty arrays when nothing is hidden);
#           1 on fetch failure - callers FAIL CLOSED by treating the fetch
#           as "cannot verify hidden state" and saying so, never by silently
#           trusting unfiltered content.
set -uo pipefail

PR_NUMBER="${1:?usage: minimized-nodes.sh <pr_number>}"
: "${GH_TOKEN:?}"
REPO_TARGET="${QUERY_REPO:-${GITHUB_REPOSITORY:-}}"
: "${REPO_TARGET:?}"

# Cursor walk, capped at 3 pages per connection (300 items - the hidden set
# of even the messiest PR stays far below that; the cap bounds cost). Empty
# cursors use the page-1 shape (after:"" is an invalid cursor); cursors are
# base64url (A-Za-z0-9+/=), safe to embed as GraphQL string literals, and
# each connection carries its own clause so a mixed one-page/many-page state
# works.
Q_HEAD='query($owner:String!,$name:String!,$n:Int!){
  repository(owner:$owner,name:$name){
    pullRequest(number:$n){
      reviews(first:100AFTER_RC){
        pageInfo{ hasNextPage endCursor }
        nodes{ id isMinimized }
      }
      comments(first:100AFTER_CC){
        pageInfo{ hasNextPage endCursor }
        nodes{ id isMinimized }
      }
    }
  }
}'

reviews_ids='[]'
comments_ids='[]'
rc=""
cc=""
for _page in 1 2 3; do
  rclause=""; [ -n "$rc" ] && rclause=",after:\"$rc\""
  cclause=""; [ -n "$cc" ] && cclause=",after:\"$cc\""
  q="${Q_HEAD/AFTER_RC/$rclause}"
  q="${q/AFTER_CC/$cclause}"
  payload=$(gh api graphql -f query="$q" \
    -f owner="${REPO_TARGET%%/*}" -f name="${REPO_TARGET##*/}" -F n="$PR_NUMBER" 2>/dev/null)
  [ -n "$payload" ] || {
    echo "::error::minimized-nodes: GraphQL fetch failed for PR #$PR_NUMBER - hidden state unverifiable." >&2
    exit 1
  }
  pr_node=$(printf '%s' "$payload" | jq -c '.data.repository.pullRequest // empty')
  [ -n "$pr_node" ] || break
  reviews_ids=$(printf '%s' "$pr_node" | jq -c --argjson acc "$reviews_ids" \
    '$acc + [.reviews.nodes[]? | select(.isMinimized == true) | .id]')
  comments_ids=$(printf '%s' "$pr_node" | jq -c --argjson acc "$comments_ids" \
    '$acc + [.comments.nodes[]? | select(.isMinimized == true) | .id]')
  rc=$(printf '%s' "$pr_node" | jq -r 'if .reviews.pageInfo.hasNextPage then .reviews.pageInfo.endCursor else "" end')
  cc=$(printf '%s' "$pr_node" | jq -r 'if .comments.pageInfo.hasNextPage then .comments.pageInfo.endCursor else "" end')
  [ -z "$rc" ] && [ -z "$cc" ] && break
done

jq -cn --argjson r "$reviews_ids" --argjson c "$comments_ids" '{reviews: $r, comments: $c}'
