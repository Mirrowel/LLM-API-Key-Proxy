#!/usr/bin/env bash
# Prompt STRUCTURAL battery: machine contracts only.
#
# Doctrine (operator-set): prompt prose is fully editable end to end - no
# content pins, not even security rules. Rule deletions belong to diff
# review and the review agent, not to CI. What remains machine-testable:
#   1. every manifest assembles (fail-closed assembler) and renders clean
#   2. every ${VAR} a mode's parts use is in the list its RENDERER passes
#      to envsubst - extracted from the renderers themselves (single
#      source: the workflows and the review kit), never duplicated here
#   3. the workflow<->prompt marker coupling: pr-review's verify step
#      greps posted reviews for a marker token; the assembled review
#      modes must teach that exact token (extracted from the workflow,
#      so renaming it in both places stays green)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PASS=0; FAIL=0
check() { if [ "$2" = "$3" ]; then echo "PASS: $1"; PASS=$((PASS+1)); else echo "FAIL: $1"; echo "  want=[$2]"; echo "  got =[$3]"; FAIL=$((FAIL+1)); fi; }

# ---- assembly cache: renders are deterministic per (parts+manifests+
# assembler) content; cache under .fixture-cache (gitignored, machine-local,
# same discipline as the scrub fixture template). Cold runs build, warm
# runs read - the 13 manifest assemblies are this battery's cost floor.
FIXROOT="${FIXTURE_CACHE:-$(cd "$SCRIPT_DIR/../.." && pwd)/.fixture-cache}"
CACHEKEY="$(cat .github/prompts/parts/*.md .github/prompts/manifests/*.manifest .github/scripts/assemble-prompt.sh 2>/dev/null | sha256sum | cut -c1-16)"
PROMPT_CACHE="$FIXROOT/prompts-$CACHEKEY"

# ---- dummy env: every var any renderer lists must resolve non-empty -----
# These are the battery's stand-in values; check D asserts list completeness
# against them so a renderer's new var fails loudly here instead of as an
# empty prompt field in production.
export PR_AUTHOR=octocat PR_NUMBER=42 GITHUB_REPOSITORY=Own/repo PR_HEAD_SHA=abc123
export PULL_REQUEST_CONTEXT='<ctx>' DIFF_FILE_PATH=/tmp/d.txt
export THREAD_CONTEXT='<tc>' NEW_COMMENT_AUTHOR=someone NEW_COMMENT_BODY='<b>'
export THREAD_NUMBER=42 THREAD_AUTHOR=octo IS_FIRST_REVIEW=true
export FULL_DIFF_PATH=/tmp/f.txt INCREMENTAL_DIFF_PATH=/tmp/i.txt LAST_REVIEWED_SHA=abc123
export ISSUE_CONTEXT='<ic>' ISSUE_NUMBER=7 ISSUE_AUTHOR=octo REVIEW_TYPE=FIRST
export DISCUSSION_NODE_ID=D_kwDO_123 DISCUSSION_TITLE='Disc Title' DISCUSSION_REPLY_TO_NODE=''
export PR_TITLE=t PR_BODY=b PR_LABELS=l FILE_GROUPS=g REPORT_TEMPLATE=r DIFF_PATH=/tmp/c.txt CHANGED_FILES=c CHANGED_FILES_JSON=cj
export TRIGGER_MESSAGE='<tm>' PREVIOUS_BOT_REVIEWS='<pbr>' AGENT_REVIEW_HISTORY='<arh>' PREVIOUS_COMPLIANCE_REPORT='<pcr>'
export REBASE_CONTEXT='<rc>'
export REVIEW_KIT_SUMMARY='PR #42: Review type: FOLLOW-UP | Instructions: /tmp/instructions/review-followup.md | Incremental diff: /tmp/kit/42/incremental_diff.patch (120 lines)'
export BOT_IDENTITY_LIST='mirrobot-agent,mirrobot-agent[bot]' BOT_IDENTITY_PRIMARY='mirrobot-agent'

# ---- renderer lists, extracted from their single sources ----------------
# list_of <file> <VARNAME> -> sorted unique bare names on one line
list_of() {
  awk -v v="$2" -v q="'" '$0 ~ "^ *" v "=" q { print; exit }' "$1" \
    | sed -e "s/^ *//" -e "s/^[A-Z]*=//" \
    | tr -d "'\${}" \
    | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' '
}
R_PR=$(list_of  .github/workflows/pr-review.yml          VARS)
R_KIT=$(list_of .github/scripts/generate-review-kit.sh   RVARS)
R_BOTI=$(list_of .github/workflows/bot-reply.yml         IVARS)
R_BOT=$(list_of .github/workflows/bot-reply.yml          VARS)
R_IC=$(list_of  .github/workflows/issue-comment.yml      VARS)
R_CC=$(list_of  .github/workflows/compliance-check.yml   VARS)

# ---- mode -> renderer-list map (the render sites, not copies) -----------
mode_list() { case "$1" in
  pr-review-first|pr-review-followup)                          printf '%s' "$R_PR" ;;
  review-first-instructions|review-followup-instructions|review-memory-instructions) printf '%s' "$R_KIT" ;;
  agentlib-*)                                                  printf '%s' "$R_BOTI" ;;
  bot-reply)                                                   printf '%s' "$R_BOT" ;;
  issue-comment)                                               printf '%s' "$R_IC" ;;
  compliance-first|compliance-followup)                        printf '%s' "$R_CC" ;;
  *)                                                           printf 'UNMAPPED' ;;
esac; }

# ---- 1. assembler self-verify (orphans, duplicate H2, fail-closed) ------
check "assembler --verify (no orphan parts, no duplicate H2)" ok \
  "$(bash .github/scripts/assemble-prompt.sh --verify >/dev/null 2>&1 && echo ok || echo broken)"

# ---- 2. per-manifest: assemble, placeholder-list check, clean render ----
# One pass per manifest, in parallel (background); raw assemblies are
# cached to $TMP for the marker section. Aggregation is grep-based - this
# battery emits no cosmetic PASS-prefixed lines.
run_mode() { # mode-name
  local mode="$1" raw lst used only_used missing v llist rendered leftover
  if [ -s "$PROMPT_CACHE/raw-$mode" ]; then
    raw="$(cat "$PROMPT_CACHE/raw-$mode")"
  else
    raw="$(bash .github/scripts/assemble-prompt.sh "$mode" 2>/dev/null)"
    mkdir -p "$PROMPT_CACHE" 2>/dev/null && printf '%s' "$raw" > "$PROMPT_CACHE/raw-$mode"
  fi
  printf '%s' "$raw" > "$TMP/raw-$mode"
  [ -n "$raw" ] && check "$mode: assembles non-empty" yes yes || check "$mode: assembles non-empty" yes no

  lst="$(mode_list "$mode")"
  if [ "$lst" = "UNMAPPED" ]; then check "$mode: renderer list mapped" yes no; return 0; fi
  check "$mode: renderer list mapped" yes yes

  # Contract (one-directional by design): every BRACED ${X} placeholder in
  # the assembled prose must be in the renderer's envsubst list - an
  # unlisted placeholder reaches the agent as a literal "${X}". Bare $X is
  # deliberately not counted: envsubst substitutes listed bare names, but
  # prose also legitimately contains shell snippets ($b, $KEY in examples)
  # that are NOT placeholders. Listed-but-unused is fine too - renderers
  # pass union lists shared across mode variants.
  used="$(printf '%s' "$raw" | grep -o '\${[A-Za-z_][A-Za-z0-9_]*}' | tr -d '${}' | sort -u | tr '\n' ' ')"

  only_used=""
  for v in $used; do case " $lst " in *" $v "*) ;; *) only_used="$only_used $v" ;; esac; done
  if [ -z "$only_used" ]; then
    check "$mode: every \${VAR} placeholder is in the renderer list" ok ok
  else
    check "$mode: every \${VAR} placeholder is in the renderer list (unlisted:${only_used})" ok mismatch
  fi

  # every listed var must be defined in the dummy env (onboarding aid; a
  # renderer's new var fails here instead of rendering empty in production)
  missing=""
  for v in $lst; do [ "${!v+set}" = set ] || missing="$missing $v"; done
  check "$mode: every listed var is defined in the dummy env" none "${missing:-none}"

  # render with the mode's own list -> no unresolved ${...} may remain
  llist=""
  for v in $lst; do llist="$llist"'\${'"$v}"; done
  rendered="$(printf '%s' "$raw" | envsubst "$llist")"
  leftover="$(printf '%s' "$rendered" | grep -o '\${[A-Za-z_][A-Za-z0-9_]*}' | head -3 | tr '\n' ' ')"
  check "$mode: renders with no unresolved \${VAR}" none "${leftover:-none}"
}
for man in .github/prompts/manifests/*.manifest; do
  mode="$(basename "$man" .manifest)"
  run_mode "$mode" >> "$TMP/log-$mode" 2>&1 &
done
wait
cat "$TMP"/log-* 2>/dev/null | sort
PASS=$(cat "$TMP"/log-* 2>/dev/null | grep -c '^PASS:')
FAIL=$(cat "$TMP"/log-* 2>/dev/null | grep -c '^FAIL:')

# ---- 3. workflow<->prompt marker couplings -------------------------------
# Tokens the WORKFLOWS grep out of agent-posted content (review detection,
# verify, attribution) must be taught by the assembled review modes. Each
# token is extracted FROM the workflow file - renaming a marker in workflow
# + parts together stays green; teaching a token the verifier does not
# expect turns red.
coupling() { # token label modes...
  local tok="$1" label="$2"; shift 2
  local mode raw
  check "workflow still carries marker: $label" yes "$([ -n "$tok" ] && echo yes || echo no)"
  [ -z "$tok" ] && return 0
  for mode in "$@"; do
    raw="$(cat "$TMP/raw-$mode" 2>/dev/null)"
    printf '%s' "$raw" | grep -qF -- "$tok" \
      && check "$mode: teaches the workflow's $label marker" yes yes \
      || check "$mode: teaches the workflow's $label marker" no no
  done
}
sha_tok="$(grep -o 'last_reviewed_sha' .github/workflows/pr-review.yml | head -1)"
sig_tok="$(grep -o 'This review was generated by an AI assistant' .github/workflows/pr-review.yml | head -1)"
coupling "$sha_tok" "reviewed-SHA footer" pr-review-first pr-review-followup review-first-instructions review-followup-instructions
coupling "$sig_tok" "AI-attribution signature" pr-review-first pr-review-followup review-first-instructions review-followup-instructions

# prune superseded prompt-cache generations (keep only the current key)
ls -dt "$FIXROOT"/prompts-* 2>/dev/null | awk 'NR>1' | xargs -r rm -rf

echo "----"; echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
