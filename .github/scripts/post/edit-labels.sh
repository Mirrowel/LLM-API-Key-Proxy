#!/usr/bin/env bash
# Add/remove labels on an issue or pull request, validated against the labels
# that actually exist. Unknown names are refused with suggestions (a typo must
# not half-apply), and maintainer-only labels need --allow-maintainer.

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_lib.sh"

usage() {
  cat <<'EOF'
usage: edit-labels.sh <number> [--add a,b] [--remove c,d] [options]
  --repo OWNER/REPO      default GH_REPO/GITHUB_REPOSITORY
  --allow-maintainer     permit maintainer-class labels (accepted, Priority)
  --dry-run              print the plan, change nothing
EOF
}

NUM=""; ADD=""; RM=""; REPO=""; ALLOW_MAINT=0; DRY=0; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --add) ADD=${2:?}; shift 2 ;;
    --remove) RM=${2:?}; shift 2 ;;
    --repo) REPO=${2:?}; shift 2 ;;
    --allow-maintainer) ALLOW_MAINT=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) POS+=("$1"); shift ;;
  esac
done
[ ${#POS[@]} -eq 1 ] || { usage >&2; exit 2; }
NUM=$(num_or_die "${POS[0]}" number)
REPO=${REPO:-$(repo_or_die)}
[ -n "$ADD" ] || [ -n "$RM" ] || die "nothing to do: pass --add and/or --remove"

LIVE=$(mktemp)
MAINT=$(mktemp)
gh label list --repo "$REPO" --limit 200 --json name --jq '.[].name' > "$LIVE" \
  || die "could not read the repository's labels"
if [ -f .github/labels.txt ]; then
  awk -F'|' '$3 ~ /maintainer/ { print $1 }' .github/labels.txt > "$MAINT"
fi

resolve_one() { # $1 requested -> prints canonical name or empty
  awk -v want="$1" 'tolower($0) == tolower(want) { print; exit }' "$LIVE"
}
suggest_one() { # $1 requested -> up to 3 candidates
  awk -v want="$1" '
    BEGIN { w = tolower(want) }
    { n = tolower($0); if (index(n, w) > 0 || index(w, n) > 0) { print "  did you mean: " $0; c++ } if (c >= 3) exit }
  ' "$LIVE"
}
is_maintainer() { # $1 canonical name
  awk -v want="$1" 'tolower($0) == tolower(want) { found = 1; exit } END { exit(found ? 0 : 1) }' "$MAINT"
}

ADD_CANON=(); RM_CANON=()
fail=0
check_side() { # $1 comma list, $2 label for messages, $3 out array name is handled by caller
  local list="$1" side="$2" item canon
  [ -n "$list" ] || return 0
  local IFS=','
  for item in $list; do
    item=$(printf '%s' "$item" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    [ -n "$item" ] || continue
    canon=$(resolve_one "$item" || true)
    if [ -z "$canon" ]; then
      note "unknown label ($side): $item"
      suggest_one "$item" >&2
      fail=1
      continue
    fi
    if is_maintainer "$canon" && [ "$ALLOW_MAINT" = 0 ]; then
      note "refusing maintainer-class label ($side): $canon (pass --allow-maintainer if a maintainer asked)"
      fail=1
      continue
    fi
    if [ "$side" = add ]; then ADD_CANON+=("$canon"); else RM_CANON+=("$canon"); fi
  done
  unset IFS
}
check_side "$ADD" add
check_side "$RM" remove
[ "$fail" = 0 ] || die "fix the label names above; nothing was changed"

if [ ${#ADD_CANON[@]} -eq 0 ] && [ ${#RM_CANON[@]} -eq 0 ]; then
  note "no labels to add or remove"
  exit 0
fi

PLAN=$(mktemp)
if [ ${#ADD_CANON[@]} -gt 0 ]; then
  printf '%s\n' "${ADD_CANON[@]}" | jq -Rsc 'split("\n") | map(select(length > 0)) | {labels: .}' > "$PLAN"
  if [ "$DRY" = 1 ]; then
    printf 'POST /repos/%s/issues/%s/labels\n' "$REPO" "$NUM"
    cat "$PLAN"
  else
    gh api --method POST "repos/$REPO/issues/$NUM/labels" --input "$PLAN" >/dev/null
  fi
fi
for name in "${RM_CANON[@]}"; do
  enc=$(jq -rn --arg s "$name" '$s | @uri')
  if [ "$DRY" = 1 ]; then
    printf 'DELETE /repos/%s/issues/%s/labels/%s\n' "$REPO" "$NUM" "$enc"
  else
    gh api --method DELETE "repos/$REPO/issues/$NUM/labels/$enc" >/dev/null
  fi
done

if [ "$DRY" = 1 ]; then exit 0; fi
printf 'labels on #%s: ' "$NUM"
gh api "repos/$REPO/issues/$NUM/labels" --jq '[.[].name] | join(", ")'
