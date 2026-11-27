#!/usr/bin/env bash
# Reply in a GitHub Discussion (GraphQL only). Anchors to the triggering
# comment by default, which is the lane's reply-to-the-summoner rule.

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

usage() {
  cat <<'EOF'
usage: post-discussion.sh <body.md> [options]
  --discussion NODE   discussion node id (default DISCUSSION_NODE_ID)
  --reply-to NODE     comment to reply under (default DISCUSSION_REPLY_TO_NODE)
  --no-footer         omit the AI-attribution signature
  --dry-run           print the command and body path, post nothing
EOF
}

D=""; R=""; FOOTER=1; DRY=0; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --discussion) D=${2:?}; shift 2 ;;
    --reply-to) R=${2:?}; shift 2 ;;
    --no-footer) FOOTER=0; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) POS+=("$1"); shift ;;
  esac
done
[ ${#POS[@]} -eq 1 ] || { usage >&2; exit 2; }
BODY=${POS[0]}
require_file "$BODY"
D=${D:-${DISCUSSION_NODE_ID:-}}
R=${R:-${DISCUSSION_REPLY_TO_NODE:-}}
[ -n "$D" ] || die "no discussion node id: pass --discussion or set DISCUSSION_NODE_ID"

OUT=$(mktemp)
trim_blank_tail "$BODY" "$OUT"
if [ "$FOOTER" = 1 ] && ! has_literal "$BODY" "$COMMENT_SIGNATURE"; then
  printf '\n\n%s\n' "$COMMENT_SIGNATURE" >> "$OUT"
fi

if [ -n "$R" ]; then
  Q='mutation($d: ID!, $b: String!, $r: ID!) { addDiscussionComment(input: {discussionId: $d, body: $b, replyToId: $r}) { comment { id url } } }'
  CMD=(gh api graphql -f "query=$Q" -f "d=$D" -F "b=@$OUT" -f "r=$R")
else
  Q='mutation($d: ID!, $b: String!) { addDiscussionComment(input: {discussionId: $d, body: $b}) { comment { id url } } }'
  CMD=(gh api graphql -f "query=$Q" -f "d=$D" -F "b=@$OUT")
fi

if [ "$DRY" = 1 ]; then
  printf 'body=%s reply_to=%s\n' "$OUT" "${R:-none}"
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

note "Posting discussion reply (${R:+under $R}${R:-top level})"
"${CMD[@]}" | jq -r '"comment_id=\(.data.addDiscussionComment.comment.id) url=\(.data.addDiscussionComment.comment.url)"'
