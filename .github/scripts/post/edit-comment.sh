#!/usr/bin/env bash
# Rewrite an existing comment by id (the living ack, a final ack edit).
# Default is no signature: acks are progress posts; pass --footer when the
# edit turns the comment into the deliverable.

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

usage() {
  cat <<'EOF'
usage: edit-comment.sh <comment-id> <body.md> [options]
  --repo OWNER/REPO   default GH_REPO/GITHUB_REPOSITORY
  --footer            append the AI-attribution signature
  --dry-run           print the body path and command, patch nothing
EOF
}

REPO=""; FOOTER=0; DRY=0; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO=${2:?}; shift 2 ;;
    --footer) FOOTER=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) POS+=("$1"); shift ;;
  esac
done
[ ${#POS[@]} -eq 2 ] || { usage >&2; exit 2; }
ID=$(num_or_die "${POS[0]}" comment-id)
BODY=${POS[1]}
require_file "$BODY"
REPO=${REPO:-$(repo_or_die)}

OUT=$(mktemp)
trim_blank_tail "$BODY" "$OUT"
if [ "$FOOTER" = 1 ]; then
  if ! has_literal "$BODY" "$COMMENT_SIGNATURE"; then
    printf '\n\n%s\n' "$COMMENT_SIGNATURE" >> "$OUT"
  fi
fi

if [ "$DRY" = 1 ]; then
  printf 'body=%s\ngh api --method PATCH /repos/%s/issues/comments/%s -F body=@%s\n' "$OUT" "$REPO" "$ID" "$OUT"
  exit 0
fi

gh api --method PATCH "repos/$REPO/issues/comments/$ID" -F body=@"$OUT" >/dev/null
printf 'edited comment_id=%s\n' "$ID"
