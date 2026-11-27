#!/usr/bin/env bash
# Open a pull request with a title file and a body file (no inline quoting).

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

usage() {
  cat <<'EOF'
usage: create-pr.sh <title-file> <body.md> [options]
  --base BRANCH       target branch
  --head BRANCH       source branch (owner:branch for a fork)
  --draft             open as draft
  --repo OWNER/REPO   default GH_REPO/GITHUB_REPOSITORY
  --dry-run
EOF
}

REPO=""; BASE=""; HEAD=""; DRAFT=0; DRY=0; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --base) BASE=${2:?}; shift 2 ;;
    --head) HEAD=${2:?}; shift 2 ;;
    --draft) DRAFT=1; shift ;;
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

CMD=(gh pr create --repo "$REPO" --title "$TITLE" --body-file "$BODY")
[ -n "$BASE" ] && CMD+=(--base "$BASE")
[ -n "$HEAD" ] && CMD+=(--head "$HEAD")
[ "$DRAFT" = 1 ] && CMD+=(--draft)

if [ "$DRY" = 1 ]; then
  printf '%q ' "${CMD[@]}"; printf '\n'
  exit 0
fi

"${CMD[@]}"
