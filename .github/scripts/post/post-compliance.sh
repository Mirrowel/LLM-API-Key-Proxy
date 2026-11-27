#!/usr/bin/env bash
# Post the compliance report: the identity tag, signature and
# compliance-check-id marker are appended here, so the report can never be
# posted untagged.

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

usage() {
  cat <<'EOF'
usage: post-compliance.sh <report.md> [options]
  --pr N              pull request number (default PR_NUMBER)
  --sha SHA           audited commit (default /tmp/head_sha.txt, else git HEAD)
  --repo OWNER/REPO   default GH_REPO/GITHUB_REPOSITORY
  --allow-missing-sha
  --dry-run
EOF
}

REPO=""; SHA_ARG=""; PR=""; DRY=0; ALLOW=0; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --pr) PR=${2:?}; shift 2 ;;
    --sha) SHA_ARG=${2:?}; shift 2 ;;
    --repo) REPO=${2:?}; shift 2 ;;
    --allow-missing-sha) ALLOW=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) POS+=("$1"); shift ;;
  esac
done
[ ${#POS[@]} -eq 1 ] || { usage >&2; exit 2; }
REPORT=${POS[0]}
require_file "$REPORT"
REPO=${REPO:-$(repo_or_die)}
PR=$(num_or_die "${PR:-$(pr_default)}" PR)
SHA=$(sha_resolve "$SHA_ARG" "$ALLOW")

OUT=$(mktemp)
prepare_compliance_body "$REPORT" "$OUT" "$PR" "$SHA"

if [ "$DRY" = 1 ]; then
  printf 'body=%s\ngh issue comment %s --repo %s --body-file %s\n' "$OUT" "$PR" "$REPO" "$OUT"
  exit 0
fi

note "Posting compliance report on $REPO#$PR at $SHA"
gh issue comment "$PR" --repo "$REPO" --body-file "$OUT"
