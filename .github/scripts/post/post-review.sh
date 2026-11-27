#!/usr/bin/env bash
# Post a pull-request review: body file, inline comments JSON, event.
#
# Prefills what the API needs and appends the canonical footers (identity tag,
# signature, last_reviewed_sha marker) so they cannot be lost. Defaults are
# overridable on purpose: pass --sha to review a different (real) commit,
# --diff to validate against an explicit patch, --no-validate to skip the hunk
# check when no diff is available.

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

usage() {
  cat <<'EOF'
usage: post-review.sh <body.md> <comments.json> --event COMMENT|APPROVE|REQUEST_CHANGES [options]
  --sha SHA              reviewed commit (default /tmp/head_sha.txt, else git HEAD)
  --pr N                 pull request number (default PR_NUMBER/THREAD_NUMBER)
  --repo OWNER/REPO      default GH_REPO/GITHUB_REPOSITORY
  --diff FILE            patch used for inline validation (default: discovery)
  --no-validate          skip the inline hunk check entirely
  --allow-missing-sha    permit a SHA absent from the local repository
  --dry-run              print the payload, post nothing
EOF
}

EVENT=""; SHA_ARG=""; PR=""; REPO=""; DIFF_PATH=""; DRY=0; ALLOW=0; VALIDATE=1
POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --event) EVENT=${2:?}; shift 2 ;;
    --sha) SHA_ARG=${2:?}; shift 2 ;;
    --pr) PR=${2:?}; shift 2 ;;
    --repo) REPO=${2:?}; shift 2 ;;
    --diff) DIFF_PATH=${2:?}; shift 2 ;;
    --no-validate) VALIDATE=0; shift ;;
    --allow-missing-sha) ALLOW=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) POS+=("$1"); shift ;;
  esac
done
[ ${#POS[@]} -eq 2 ] || { usage >&2; exit 2; }
BODY=${POS[0]}; COMMENTS=${POS[1]}

[ -n "$EVENT" ] || die "--event COMMENT|APPROVE|REQUEST_CHANGES is required"
case "$EVENT" in COMMENT|APPROVE|REQUEST_CHANGES) ;; *) die "invalid --event: $EVENT" ;; esac
require_file "$BODY"
require_file "$COMMENTS"
REPO=${REPO:-$(repo_or_die)}
PR=$(num_or_die "${PR:-$(pr_default)}" PR)
SHA=$(sha_resolve "$SHA_ARG" "$ALLOW")

jq -e 'type == "array"' "$COMMENTS" >/dev/null 2>&1 \
  || die "comments file must be a JSON array (JSONL is not accepted): $COMMENTS"

if [ "$VALIDATE" = 1 ]; then
  validate_comment_json "$COMMENTS" || die "inline comments failed validation; nothing was posted"
fi

OUT_BODY=$(mktemp)
prepare_review_body "$BODY" "$OUT_BODY" "$SHA"
PAYLOAD=$(mktemp)
jq -n --arg event "$EVENT" --arg sha "$SHA" --rawfile body "$OUT_BODY" --slurpfile comments "$COMMENTS" \
  '{event: $event, commit_id: $sha, body: $body, comments: $comments[0]}' > "$PAYLOAD"

COUNT=$(jq 'length' "$COMMENTS")
if [ "$DRY" = 1 ]; then
  note "payload: $PAYLOAD"
  cat "$PAYLOAD"
  exit 0
fi

note "Posting $EVENT review on $REPO#$PR at $SHA ($COUNT inline comment(s))"
resp=$(gh api --method POST -H "Accept: application/vnd.github+json" \
  "repos/$REPO/pulls/$PR/reviews" --input "$PAYLOAD")
printf 'review_id=%s url=%s\n' \
  "$(printf '%s' "$resp" | jq -r '.id')" \
  "$(printf '%s' "$resp" | jq -r '.html_url // empty')"
