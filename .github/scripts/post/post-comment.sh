#!/usr/bin/env bash
# Post a comment on an issue or pull request number.
# Body always travels as a file; the attribution signature is appended unless
# --no-footer (acks and progress posts).

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

usage() {
  cat <<'EOF'
usage: post-comment.sh <number> <body.md> [options]
  --repo OWNER/REPO   default GH_REPO/GITHUB_REPOSITORY
  --no-footer         omit the AI-attribution signature (acks, progress)
  --dry-run           print the command and body path, post nothing
EOF
}

REPO=""; FOOTER=1; DRY=0; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO=${2:?}; shift 2 ;;
    --no-footer) FOOTER=0; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) POS+=("$1"); shift ;;
  esac
done
[ ${#POS[@]} -eq 2 ] || { usage >&2; exit 2; }
NUM=$(num_or_die "${POS[0]}" number)
BODY=${POS[1]}
require_file "$BODY"
REPO=${REPO:-$(repo_or_die)}

OUT=$(mktemp)
trim_blank_tail "$BODY" "$OUT"
if [ "$FOOTER" = 1 ]; then
  printf '\n\n%s\n' "$COMMENT_SIGNATURE" >> "$OUT"
fi

if [ "$DRY" = 1 ]; then
  printf 'gh issue comment %s --repo %s --body-file %s\n' "$NUM" "$REPO" "$OUT"
  exit 0
fi

gh issue comment "$NUM" --repo "$REPO" --body-file "$OUT"
