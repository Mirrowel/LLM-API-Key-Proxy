#!/usr/bin/env bash
# Rule-survival battery: every load-bearing rule from the pre-parts prompts must
# survive in the assembled mode prompts. (Content may be condensed/reworded per
# the sanctioned simplification — the RULES must not vanish.)
set -u
# Runs from repo root (CI) or anywhere: resolve repo root from this script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1
TMP=/tmp/parts-battery
mkdir -p "$TMP"
export PR_AUTHOR=octocat PR_NUMBER=42 GITHUB_REPOSITORY=Own/repo PR_HEAD_SHA=abc123
export PULL_REQUEST_CONTEXT='<ctx>' DIFF_FILE_PATH=/tmp/d.txt
export THREAD_CONTEXT='<tc>' NEW_COMMENT_AUTHOR=someone NEW_COMMENT_BODY='<b>'
export THREAD_NUMBER=42 THREAD_AUTHOR=octo IS_FIRST_REVIEW=true
export FULL_DIFF_PATH=/tmp/f.txt INCREMENTAL_DIFF_PATH=/tmp/i.txt LAST_REVIEWED_SHA=abc123
export ISSUE_CONTEXT='<ic>' ISSUE_NUMBER=7 ISSUE_AUTHOR=octo REVIEW_TYPE=FIRST
export PR_TITLE=t PR_BODY=b PR_LABELS=l PREVIOUS_REVIEWS=p FILE_GROUPS=g REPORT_TEMPLATE=r DIFF_PATH=/tmp/c.txt CHANGED_FILES=c CHANGED_FILES_JSON=cj
RV='$REVIEW_TYPE $PR_AUTHOR $PR_NUMBER $GITHUB_REPOSITORY $PR_HEAD_SHA $PULL_REQUEST_CONTEXT $DIFF_FILE_PATH'
BV='$THREAD_CONTEXT $NEW_COMMENT_AUTHOR $NEW_COMMENT_BODY $THREAD_NUMBER $GITHUB_REPOSITORY $THREAD_AUTHOR $PR_HEAD_SHA $IS_FIRST_REVIEW $FULL_DIFF_PATH $INCREMENTAL_DIFF_PATH $LAST_REVIEWED_SHA $PR_NUMBER'
IV='$ISSUE_CONTEXT $ISSUE_NUMBER $ISSUE_AUTHOR'
CV='$PR_NUMBER $PR_TITLE $PR_BODY $PR_AUTHOR $PR_HEAD_SHA $CHANGED_FILES $CHANGED_FILES_JSON $PR_LABELS $PREVIOUS_REVIEWS $FILE_GROUPS $REPORT_TEMPLATE $DIFF_PATH $GITHUB_REPOSITORY'
bash .github/scripts/assemble-prompt.sh pr-review-first     | envsubst "$RV" > "$TMP/rf.txt"
REVIEW_TYPE=FOLLOW-UP bash .github/scripts/assemble-prompt.sh pr-review-followup | REVIEW_TYPE=FOLLOW-UP envsubst "$RV" > "$TMP/ru.txt"
bash .github/scripts/assemble-prompt.sh bot-reply          | envsubst "$BV" > "$TMP/br.txt"
bash .github/scripts/assemble-prompt.sh issue-comment      | envsubst "$IV" > "$TMP/ic.txt"
bash .github/scripts/assemble-prompt.sh compliance-check   | envsubst "$CV" > "$TMP/cc.txt"

PASS=0; FAIL=0
need() { # file pattern label
  if grep -Eqi -- "$2" "$TMP/$1.txt"; then PASS=$((PASS+1)); else echo "FAIL [$1]: $3"; FAIL=$((FAIL+1)); fi
}
neednt() { # file pattern label (must NOT appear)
  if grep -Eqi -- "$2" "$TMP/$1.txt"; then echo "FAIL [$1]: $3 (present, must not be)"; FAIL=$((FAIL+1)); else PASS=$((PASS+1)); fi
}

# ---- universal rules (all five modes) ----
for f in rf ru br ic cc; do
  need $f 'mirrobot-agent'                     "$f: identity names"
  need $f 'older mention'                      "$f: old-mentions-are-history"
  need $f 'fresh shell'                        "$f: fresh-shell key point"
  need $f 'body-file'                          "$f: file-based posting"
  need $f 'FORBIDDEN COMMANDS'                 "$f: secrets rule"
  need $f 'webfetch'                           "$f: webfetch denied"
  need $f 'allowed prefix'                     "$f: shell prefix rule"
  need $f 'long-running processes'             "$f: no-daemons rule"
  need $f 'Package installation is allowed'    "$f: package-install + scrutiny rule"
  need $f 'typosquat'                          "$f: supply-chain scrutiny"
  need $f 'untrusted data, never as instructions' "$f: websearch untrusted-data rule"
  need $f 'verify you have the required permissions' "$f: permission pre-check"
  need $f 'Level 2'                            "$f: error L2"
  need $f 'Level 3'                            "$f: error L3"
  need $f 'single retry'                       "$f: L3 retry-once"
  need $f 'warnings section'                   "$f: L3 warnings reporting"
  neednt $f 'All `gh` commands are allowed'  "$f: no all-gh overclaim"
  neednt $f 'All `git'                       "$f: no all-git overclaim (are allowed)"
  neednt $f 'All `jq'                        "$f: no all-jq overclaim (are allowed)"
  neednt $f "heredoc for consistency"        "$f: no heredoc mandate"
  neednt $f '\$\{[A-Z_]+\}'                  "$f: no unresolved vars"
done

# ---- review-family rules (pr-review both modes + bot-reply) ----
for f in rf ru br; do
  need $f 'hard no'                            "$f: verdict ladder"
  need $f 'mergeable as-is'                    "$f: approval rule"
  need $f 'testing adequate'                   "$f: APPROVE checklist"
  need $f 'Verdict:'                           "$f: verdict line mandate"
  need $f 'last_reviewed_sha:abc123'           "$f: footer contract"
  need $f 'jq -e'                              "$f: array guard"
  need $f 'review_comments.json'               "$f: comments file"
  need $f 'review_payload'                     "$f: payload file"
  need $f 'HIGH-SIGNAL, LOW-NOISE'             "$f: feedback philosophy"
  need $f 'praise-only'                        "$f: no-praise-only rule"
done
need rf 'Protocol for FIRST'                   "rf: first protocol"
need rf 'comprehensive, initial analysis'      "rf: first = full PR"
need ru 'Protocol for FOLLOW-UP'               "ru: followup protocol"
need ru 'incremental changes since the last'   "ru: incremental scope"
need ru 'previous feedback'                    "ru: verify-previous-feedback duty"

# ---- write-scope per mode ----
need rf 'never modify repository files'        "rf: reviewer write-scope"
need ru 'never modify repository files'        "ru: reviewer write-scope"
need ic 'never modify repository files'        "ic: analyst write-scope"
need cc 'never modify repository files'        "cc: compliance write-scope"
need br 'repository files .fixes, features.'   "br: agent write-scope (may modify)"
need br 'workflows'                            "br: workflow-edit deny note"
need br 'Level 1'                              "br: recovery level present"
need br 'refusing to allow a GitHub App'       "br: workflow-push recovery recipe"
need br 'failure report'                       "br: fatal-error reporting duty"
need br 'second and final'                     "br: single-recovery limit"

# ---- mode-specific ----
need cc 'compliance'                           "cc: compliance mission"
need cc 'file group'                           "cc: file groups wiring"
need cc 'template'                             "cc: report template wiring"
need ic 'Initial Analysis Report'              "ic: analysis output shape"
need ic 'thanks'                               "ic: acknowledgment duty"

# ---- clean-prose invariants (generator-artifact class) ----
for f in rf ru br ic cc; do
  neednt $f "' \+ '"                            "$f: no powershell concat artifacts"
  neednt $f 'chr\(39\)'                         "$f: no python chr() artifacts"
  neednt $f '\?\?'                              "$f: no mangled emoji sequences"
done

echo "----"; echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
