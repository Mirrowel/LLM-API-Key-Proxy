#!/usr/bin/env bash
# Post one standalone inline review comment (no review object).
# Validates the path/line against the PR diff, or takes --file-level for a
# whole-file remark. Line hints on failure, so the agent fixes it first try.

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

usage() {
  cat <<'EOF'
usage: post-inline.sh <path> <line> <body.md> [options]
       post-inline.sh <path> --file-level <body.md> [options]
  --start-line N     multi-line range start (line is the range end)
  --side RIGHT|LEFT  diff side (default RIGHT)
  --sha SHA          commit id (default /tmp/head_sha.txt, else git HEAD)
  --pr N             pull request number (default PR_NUMBER/THREAD_NUMBER)
  --repo OWNER/REPO  default GH_REPO/GITHUB_REPOSITORY
  --diff FILE        patch used for validation (default: discovery)
  --no-validate      skip the hunk check
  --allow-missing-sha
  --dry-run
EOF
}

LINE=""; START=""; SIDE="RIGHT"; SHA_ARG=""; PR=""; REPO=""; DIFF_PATH=""
DRY=0; ALLOW=0; VALIDATE=1; FILE_LEVEL=0
POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --start-line) START=${2:?}; shift 2 ;;
    --start-side) SIDE_START=${2:?}; shift 2 ;;
    --side) SIDE=${2:?}; shift 2 ;;
    --sha) SHA_ARG=${2:?}; shift 2 ;;
    --pr) PR=${2:?}; shift 2 ;;
    --repo) REPO=${2:?}; shift 2 ;;
    --diff) DIFF_PATH=${2:?}; shift 2 ;;
    --no-validate) VALIDATE=0; shift ;;
    --allow-missing-sha) ALLOW=1; shift ;;
    --file-level) FILE_LEVEL=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) POS+=("$1"); shift ;;
  esac
done

if [ "$FILE_LEVEL" = 1 ]; then
  [ ${#POS[@]} -eq 2 ] || { usage >&2; exit 2; }
  PATH_ARG=${POS[0]}; BODY=${POS[1]}
  LINE=0
else
  [ ${#POS[@]} -eq 3 ] || { usage >&2; exit 2; }
  PATH_ARG=${POS[0]}; LINE=${POS[1]}; BODY=${POS[2]}
  case "$LINE" in *[!0-9]*) die "line must be numeric: $LINE" ;; esac
  [ -n "$START" ] && { case "$START" in *[!0-9]*) die "--start-line must be numeric: $START" ;; esac; }
fi
case "$SIDE" in RIGHT|LEFT) ;; *) die "--side must be RIGHT or LEFT: $SIDE" ;; esac
require_file "$BODY"
[ -n "$PATH_ARG" ] || die "path is required"
REPO=${REPO:-$(repo_or_die)}
PR=$(num_or_die "${PR:-$(pr_default)}" PR)
SHA=$(sha_resolve "$SHA_ARG" "$ALLOW")

if [ "$VALIDATE" = 1 ]; then
  DIFF=$(find_diff "$DIFF_PATH")
  if [ -z "$DIFF" ]; then
    note "warning: no diff found; skipping hunk validation"
  elif [ "$FILE_LEVEL" = 1 ]; then
    validate_comment "$DIFF" "$PATH_ARG" 0 "$SIDE" 0 1 \
      || die "path is not in the PR diff; nothing was posted"
  else
    validate_comment "$DIFF" "$PATH_ARG" "$LINE" "$SIDE" "${START:-0}" 0 \
      || die "fix the comment location; nothing was posted"
  fi
fi

PAYLOAD=$(mktemp)
if [ "$FILE_LEVEL" = 1 ]; then
  jq -n --arg sha "$SHA" --arg path "$PATH_ARG" --rawfile body "$BODY" \
    '{commit_id: $sha, path: $path, subject_type: "file", body: $body}' > "$PAYLOAD"
else
  jq -n --arg sha "$SHA" --arg path "$PATH_ARG" --argjson line "$LINE" --arg side "$SIDE" \
    --argjson start "${START:-null}" --rawfile body "$BODY" \
    '{commit_id: $sha, path: $path, line: $line, side: $side, body: $body}
     + (if $start != null then {start_line: $start, start_side: $side} else {} end)' > "$PAYLOAD"
fi

if [ "$DRY" = 1 ]; then
  note "payload: $PAYLOAD"
  cat "$PAYLOAD"
  exit 0
fi

note "Posting inline comment on $REPO#$PR ($PATH_ARG${LINE:+:$LINE})"
resp=$(gh api --method POST -H "Accept: application/vnd.github+json" \
  "repos/$REPO/pulls/$PR/comments" --input "$PAYLOAD")
printf 'comment_id=%s url=%s\n' \
  "$(printf '%s' "$resp" | jq -r '.id')" \
  "$(printf '%s' "$resp" | jq -r '.html_url // empty')"
