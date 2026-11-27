#!/usr/bin/env bash
# Set the `compliance-check` commit status. Validates the state enum and the
# description length before the API can 422, and prints the status URL.

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

usage() {
  cat <<'EOF'
usage: set-status.sh <state> <description> [options]
  state: success | failure | error | pending
  --sha SHA           commit (default /tmp/head_sha.txt, else git HEAD)
  --context NAME      status context (default compliance-check)
  --target-url URL    link the status to the report
  --repo OWNER/REPO   default GH_REPO/GITHUB_REPOSITORY
  --allow-missing-sha
  --dry-run
EOF
}

REPO=""; SHA_ARG=""; CTX="compliance-check"; TARGET=""; DRY=0; ALLOW=0; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --sha) SHA_ARG=${2:?}; shift 2 ;;
    --context) CTX=${2:?}; shift 2 ;;
    --target-url) TARGET=${2:?}; shift 2 ;;
    --repo) REPO=${2:?}; shift 2 ;;
    --allow-missing-sha) ALLOW=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) POS+=("$1"); shift ;;
  esac
done
[ ${#POS[@]} -eq 2 ] || { usage >&2; exit 2; }
STATE=${POS[0]}; DESC=${POS[1]}
case "$STATE" in success|failure|error|pending) ;; *) die "invalid state: $STATE (success|failure|error|pending)" ;; esac
[ -n "$DESC" ] || die "description is required"
[ "${#DESC}" -le 140 ] || die "description is ${#DESC} chars; the status API limit is 140"
REPO=${REPO:-$(repo_or_die)}
SHA=$(sha_resolve "$SHA_ARG" "$ALLOW")

if [ "$DRY" = 1 ]; then
  printf 'POST /repos/%s/statuses/%s state=%s context=%s description=%s' "$REPO" "$SHA" "$STATE" "$CTX" "$DESC"
  [ -n "$TARGET" ] && printf ' target_url=%s' "$TARGET"
  printf '\n'
  exit 0
fi

ARGS=(--method POST "repos/$REPO/statuses/$SHA" -f state="$STATE" -f context="$CTX" -f description="$DESC")
[ -n "$TARGET" ] && ARGS+=(-f target_url="$TARGET")
resp=$(gh api "${ARGS[@]}")
printf 'status=%s url=%s\n' "$STATE" "$(printf '%s' "$resp" | jq -r '.url // empty')"
