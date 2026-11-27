#!/usr/bin/env bash
# Open an issue with a title file and a body file (no inline quoting).
# Labels are passed through: they must already exist in the repository.

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

usage() {
  cat <<'EOF'
usage: create-issue.sh <title-file> <body.md> [options]
  --label NAME        label to add (repeatable; must exist)
  --repo OWNER/REPO   default GH_REPO/GITHUB_REPOSITORY
  --dry-run
EOF
}

REPO=""; DRY=0; POS=(); LABELS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABELS+=("${2:?}"); shift 2 ;;
    --repo) REPO=${2:?}; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) POS+=("$1"); shift ;;
  esac
done
[ ${#POS[@]} -eq 2 ] || { usage >&2; exit 2; }
TITLE_FILE=${POS[0]}; BODY=${POS[1]}
require_file "$TITLE_FILE"
require_file "$BODY"
REPO=${REPO:-$(repo_or_die)}
TITLE=$(tr -d '\r' < "$TITLE_FILE" | sed -n '1p' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
[ -n "$TITLE" ] || die "title file's first line is empty: $TITLE_FILE"

CMD=(gh issue create --repo "$REPO" --title "$TITLE" --body-file "$BODY")
for l in "${LABELS[@]:-}"; do
  [ -n "$l" ] && CMD+=(--label "$l")
done

if [ "$DRY" = 1 ]; then
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

"${CMD[@]}"
