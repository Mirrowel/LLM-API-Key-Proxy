#!/usr/bin/env bash
# Print the canonical footer block for a hand-rolled artifact.
# The posting scripts append this themselves; this exists for the cases where
# the agent deliberately bypasses a script (a new artifact kind, an edge case).

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

usage() {
  cat <<'EOF'
usage: artifact-footer.sh <review|compliance|comment> [sha] [options]
  --pr N   pull request number (compliance marker; default PR_NUMBER)
  --repo   unused, accepted for symmetry
EOF
}

KIND=""; SHA_ARG=""; PR=""; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --pr) PR=${2:?}; shift 2 ;;
    --repo) shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) POS+=("$1"); shift ;;
  esac
done
[ ${#POS[@]} -ge 1 ] || { usage >&2; exit 2; }
KIND=${POS[0]}; SHA_ARG=${POS[1]:-}

case "$KIND" in
  review)
    SHA=$(sha_resolve "$SHA_ARG" 1)
    printf '%s\n%s\n%s\n' "$REVIEW_TAG" "$REVIEW_SIGNATURE" "<!-- last_reviewed_sha:${SHA} -->"
    ;;
  compliance)
    SHA=$(sha_resolve "$SHA_ARG" 1)
    PR=$(num_or_die "${PR:-$(pr_default)}" PR)
    printf '%s\n%s\n%s\n' "$COMPLIANCE_TAG" "$COMPLIANCE_SIGNATURE" "<!-- compliance-check-id: ${PR}-${SHA} -->"
    ;;
  comment)
    printf '%s\n' "$COMMENT_SIGNATURE"
    ;;
  *) die "unknown kind: $KIND (review|compliance|comment)" ;;
esac
