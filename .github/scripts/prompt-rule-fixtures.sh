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
CACHEKEY="$(cat .github/prompts/parts/*.md .github/prompts/manifests/*.manifest .github/prompts/skills/*/*.md .github/prompts/skills/*/meta.yml .github/scripts/assemble-prompt.sh 2>/dev/null | sha256sum | cut -c1-16)"
PROMPT_CACHE="$FIXROOT/prompts-$CACHEKEY"

# ---- dummy env: every var any renderer lists must resolve non-empty -----
# These are the battery's stand-in values; check D asserts list completeness
# against them so a renderer's new var fails loudly here instead of as an
# empty prompt field in production.
export PR_AUTHOR=octocat PR_NUMBER=42 GITHUB_REPOSITORY=Own/repo PR_HEAD_SHA=abc123
export PULL_REQUEST_CONTEXT='<ctx>' DIFF_FILE_PATH=/tmp/d.txt
export THREAD_CONTEXT='<tc>' NEW_COMMENT_AUTHOR=someone NEW_COMMENT_BODY='<b>'
export GUEST_SUMMONER_LINE='<summoner>' SCRUB_REMOVALS_SUMMARY='<scrub>' TARGET_REPO=Other/x
export THREAD_NUMBER=42 THREAD_AUTHOR=octo IS_FIRST_REVIEW=true
export FULL_DIFF_PATH=/tmp/f.txt INCREMENTAL_DIFF_PATH=/tmp/i.txt LAST_REVIEWED_SHA=abc123
export ISSUE_CONTEXT='<ic>' ISSUE_NUMBER=7 ISSUE_AUTHOR=octo REVIEW_TYPE=FIRST
export DISCUSSION_NODE_ID=D_kwDO_123 DISCUSSION_TITLE='Disc Title' DISCUSSION_REPLY_TO_NODE=''
export PR_TITLE=t PR_BODY=b PR_LABELS=l FILE_GROUPS=g REPORT_TEMPLATE=r DIFF_PATH=/tmp/c.txt CHANGED_FILES=c CHANGED_FILES_JSON=cj
export TRIGGER_MESSAGE='<tm>' PREVIOUS_BOT_REVIEWS='<pbr>' AGENT_REVIEW_HISTORY='<arh>' PREVIOUS_COMPLIANCE_REPORT='<pcr>'
export REBASE_CONTEXT='<rc>'
export REVIEW_KIT_SUMMARY='PR #42: Review type: FOLLOW-UP | Instructions: /tmp/instructions/review-followup.md | Incremental diff: /tmp/kit/42/incremental_diff.patch (120 lines)'
export BOT_IDENTITY_LIST='mirrobot-agent,mirrobot-agent[bot]' BOT_IDENTITY_PRIMARY='mirrobot-agent'
export LABEL_PALETTE='<lp>'
export THREAD_IDENTITY='PR #42 "a title" by @octocat in Own/repo, head abc123.'
# Assemblies install the skills a manifest declares: keep them out of the
# operator's real ~/.config and note required ones under the scratch dir.
export SKILLS_DIR="$TMP/skills-install" SKILLS_REQUIRED_FILE="$TMP/skills-required.txt"

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
R_GUEST=$(list_of .github/workflows/bot-reply-guest.yml  GVARS)
R_IC=$(list_of  .github/workflows/issue-comment.yml      VARS)
R_CC=$(list_of  .github/workflows/compliance-check.yml   VARS)

# ---- mode -> renderer-list map (the render sites, not copies) -----------
mode_list() { case "$1" in
  pr-review-first|pr-review-followup)                          printf '%s' "$R_PR" ;;
  review-first-instructions|review-followup-instructions|review-memory-instructions) printf '%s' "$R_KIT" ;;
  agentlib-*)                                                  printf '%s' "$R_BOTI" ;;
  bot-reply)                                                   printf '%s' "$R_BOT" ;;
  bot-reply-guest)                                             printf '%s' "$R_GUEST" ;;
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
    # One install dir per mode: the manifest assemblies run in parallel and
    # each install wipes stale mirrobot-* dirs, so a shared dir would race.
    raw="$(SKILLS_DIR="$TMP/skills-install-$mode" bash .github/scripts/assemble-prompt.sh "$mode" 2>/dev/null)"
    if [ -n "$raw" ]; then
      mkdir -p "$PROMPT_CACHE" 2>/dev/null && printf '%s' "$raw" > "$PROMPT_CACHE/raw-$mode"
    fi
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
# The footers moved from the prompt prose into the posting toolkit: the
# workflow still greps the token, _lib.sh is the single place that writes it,
# and every review mode submits through post-review.sh.
lib=".github/scripts/post/_lib.sh"
check "workflow still carries marker: reviewed-SHA footer" yes "$([ -n "$sha_tok" ] && echo yes || echo no)"
check "workflow still carries marker: AI-attribution signature" yes "$([ -n "$sig_tok" ] && echo yes || echo no)"
check "toolkit owns the reviewed-SHA marker" yes "$(grep -qF 'last_reviewed_sha:' "$lib" && echo yes || echo no)"
check "toolkit owns the AI-attribution signature" yes "$(grep -qF 'This review was generated by an AI assistant' "$lib" && echo yes || echo no)"
for mode in pr-review-first pr-review-followup review-first-instructions review-followup-instructions; do
  raw="$(cat "$TMP/raw-$mode" 2>/dev/null)"
  printf '%s' "$raw" | grep -qF '/tmp/post/post-review.sh' \
    && check "$mode: submits through the posting toolkit" yes yes \
    || check "$mode: submits through the posting toolkit" no no
done

# ---- 4. skills: inline delivery, install, gating, fail-closed ------------
# One body per skill, two deliveries: inline in the lanes that always need it
# (issue analysis, first review, the manage set) and an installed skill in the
# lanes that need it only sometimes. Everything here is structural.
SKILLS_SRC=".github/prompts/skills"
check "skills: the labels skill is declared inline where grades are always owed" "4" \
  "$(grep -l '^inline:labels$' .github/prompts/manifests/*.manifest | wc -l | tr -d ' ')"
check "skills: the labels skill is installed where grades come up sometimes" "3" \
  "$(grep -l '^skill:labels$' .github/prompts/manifests/*.manifest | wc -l | tr -d ' ')"
check "skills: neither compliance lane carries label machinery" "0" \
  "$(grep -l 'labels' .github/prompts/manifests/compliance-*.manifest 2>/dev/null | wc -l | tr -d ' ')"
check "skills: the guest lane declares no label skill" no \
  "$(grep -q 'labels' .github/prompts/manifests/bot-reply-guest.manifest && echo yes || echo no)"
check "skills: the guest lane declares the Discussion skill" yes \
  "$(grep -q '^skill:discussions$' .github/prompts/manifests/bot-reply-guest.manifest && echo yes || echo no)"

# every ${VAR} in any skill must be in the renderer lists of the lanes that
# carry it (labels is the only placeholder today)
skill_vars="$(grep -rhoE '\$\{[A-Za-z_][A-Za-z0-9_]*\}' "$SKILLS_SRC" | tr -d '${}' | sort -u | tr '\n' ' ')"
check "skills: placeholders are known" "LABEL_PALETTE " "$skill_vars"
for pair in "pr-review-first:R_PR" "bot-reply:R_BOT" "issue-comment:R_IC" "agentlib-manage:R_BOTI" "review-first-instructions:R_KIT"; do
  mode=${pair%%:*}; var=${pair##*:}; lst="${!var}"
  check "skills: $mode renderer can resolve the label palette" yes "$(case " $lst " in *' LABEL_PALETTE '*) echo yes ;; *) echo no ;; esac)"
done

# inline delivery: content present, no install, no routing block
mkdir -p "$TMP/inline"
LABEL_PALETTE='<lp>' SKILLS_DIR="$TMP/inline/skills" SKILLS_ENVSUBST='${LABEL_PALETTE}' \
  bash .github/scripts/assemble-prompt.sh pr-review-first > "$TMP/inline/prompt.txt" 2>"$TMP/inline/err"
check "skills: inline delivery renders the palette" yes "$(grep -qF '<lp>' "$TMP/inline/prompt.txt" && echo yes || echo no)"
check "skills: inline delivery resolves the palette placeholder" no "$(grep -qF '${LABEL_PALETTE}' "$TMP/inline/prompt.txt" && echo yes || echo no)"
check "skills: inline delivery installs nothing" "0" "$(find "$TMP/inline/skills" -maxdepth 1 -type d -name 'mirrobot-*' 2>/dev/null | wc -l | tr -d ' ')"
check "skills: no routing block is injected (opencode lists skills itself)" no "$(grep -qF '[SKILLS FOR THIS RUN]' "$TMP/inline/prompt.txt" && echo yes || echo no)"

# installation is opt-in: a bare assembly (a local preview) touches no disk
mkdir -p "$TMP/bare"
LABEL_PALETTE='<lp>' SKILLS_DIR="$TMP/bare/skills" SKILLS_ENVSUBST='${LABEL_PALETTE}' \
  bash .github/scripts/assemble-prompt.sh bot-reply >/dev/null 2>&1
check "skills: a bare assembly installs nothing" "0" "$(find "$TMP/bare/skills" -maxdepth 1 -type d -name 'mirrobot-*' 2>/dev/null | wc -l | tr -d ' ')"

# skill delivery: installed, namespaced, rendered; skip honored
mkdir -p "$TMP/inst/skills/mirrobot-stale"
INST="$TMP/inst/skills"
LABEL_PALETTE='<lp>' SKILLS_DIR="$INST" SKILLS_REQUIRED_FILE="$TMP/inst/req.txt" \
  SKILLS_ENVSUBST='${LABEL_PALETTE}' SKILL_INSTALL_SKIP='discussions' SKILLS_INSTALL=1 \
  bash .github/scripts/assemble-prompt.sh bot-reply > "$TMP/inst/prompt.txt" 2>"$TMP/inst/err"
check "skills: skill delivery removes stale mirrobot dirs" "0" "$([ -d "$INST/mirrobot-stale" ] && echo 1 || echo 0)"
check "skills: the declared skill is installed" yes "$([ -f "$INST/mirrobot-labels/SKILL.md" ] && echo yes || echo no)"
check "skills: a skipped skill is not installed" no "$([ -e "$INST/mirrobot-discussions" ] && echo yes || echo no)"
check "skills: SKILL.md frontmatter names the skill" yes "$(head -1 "$INST/mirrobot-labels/SKILL.md" | grep -qx -- '---' && sed -n '2p' "$INST/mirrobot-labels/SKILL.md" | grep -qx 'name: mirrobot-labels' && echo yes || echo no)"
check "skills: the description is a quoted single line" yes "$(sed -n '3p' "$INST/mirrobot-labels/SKILL.md" | grep -qE '^description: ".*"$' && echo yes || echo no)"
check "skills: the installed description is the meta trigger" yes "$(sed -n '3p' "$INST/mirrobot-labels/SKILL.md" | grep -qF 'Use before stating a grade' && echo yes || echo no)"
check "skills: the installed body is rendered" yes "$(grep -qF '<lp>' "$INST/mirrobot-labels/SKILL.md" && echo yes || echo no)"
check "skills: the installed body has no unresolved placeholder" no "$(grep -qE '\$\{[A-Za-z_]+\}' "$INST/mirrobot-labels/SKILL.md" && echo yes || echo no)"
check "skills: skill delivery injects no routing block" no "$(grep -qF '[SKILLS FOR THIS RUN]' "$TMP/inst/prompt.txt" && echo yes || echo no)"
check "skills: the Discussion skill installs only when not skipped" yes "$(SKILLS_DIR="$TMP/inst/skills2" SKILL_INSTALL_SKIP='' SKILLS_ENVSUBST='${LABEL_PALETTE}' LABEL_PALETTE='<lp>' SKILLS_INSTALL=1 bash .github/scripts/assemble-prompt.sh bot-reply 2>/dev/null >/dev/null && [ -f "$TMP/inst/skills2/mirrobot-discussions/SKILL.md" ] && echo yes || echo no)"

# fail-closed: a broken skill must stop the run, not ship a short prompt
mkdir -p "$TMP/sb"
cp -r .github/prompts/parts .github/prompts/manifests "$TMP/sb/"
cp -r "$SKILLS_SRC" "$TMP/sb/skills"
cp .github/scripts/assemble-prompt.sh "$TMP/sb/"
printf 'skill:labels\n' > "$TMP/sb/manifests/zz-skill.manifest"
printf 'name: mirrobot-labels\n' > "$TMP/sb/skills/labels/meta.yml"
SKILLS_DIR="$TMP/sb/skills-out" SKILLS_INSTALL=1 bash "$TMP/sb/assemble-prompt.sh" zz-skill >/dev/null 2>"$TMP/sb/err"
check "skills: an invalid meta fails closed" "1" "$?"
check "skills: the failure names the skill" yes "$(grep -q "skill 'labels'" "$TMP/sb/err" && echo yes || echo no)"
printf 'name: mirrobot-labels\ndescription: test skill\n' > "$TMP/sb/skills/labels/meta.yml"
printf 'Unresolved ${NOPE} placeholder.\n' > "$TMP/sb/skills/labels/body.md"
SKILLS_DIR="$TMP/sb/skills-out" SKILLS_ENVSUBST='${LABEL_PALETTE}' LABEL_PALETTE=x SKILLS_INSTALL=1 \
  bash "$TMP/sb/assemble-prompt.sh" zz-skill >/dev/null 2>"$TMP/sb/err2"
check "skills: an unresolved placeholder fails closed" "1" "$?"
check "skills: the failure names the placeholder" yes "$(grep -q 'NOPE' "$TMP/sb/err2" && echo yes || echo no)"
SKILLS_DIR="$TMP/sb/skills-out" SKILLS_INSTALL=1 bash "$TMP/sb/assemble-prompt.sh" zz-skill >/dev/null 2>"$TMP/sb/err3"
check "skills: an install without a render list fails on placeholders" "1" "$?"
check "skills: that failure names the lane requirement" yes "$(grep -q 'SKILLS_ENVSUBST' "$TMP/sb/err3" && echo yes || echo no)"
printf 'skill:nope\n' > "$TMP/sb/manifests/zz-skill.manifest"
SKILLS_DIR="$TMP/sb/skills-out" SKILLS_INSTALL=1 bash "$TMP/sb/assemble-prompt.sh" zz-skill >/dev/null 2>&1
check "skills: a missing skill fails closed" "1" "$?"
rm -rf "$TMP/sb"

# prune superseded prompt-cache generations (keep only the current key)
ls -dt "$FIXROOT"/prompts-* 2>/dev/null | awk 'NR>1' | xargs -r rm -rf

echo "----"; echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
