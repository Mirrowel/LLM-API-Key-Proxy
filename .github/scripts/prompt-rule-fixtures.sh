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
export PR_TITLE=t PR_BODY=b PR_LABELS=l FILE_GROUPS=g REPORT_TEMPLATE=r DIFF_PATH=/tmp/c.txt CHANGED_FILES=c CHANGED_FILES_JSON=cj
export TRIGGER_MESSAGE='<tm>' PREVIOUS_BOT_REVIEWS='<pbr>' AGENT_REVIEW_HISTORY='<arh>' PREVIOUS_COMPLIANCE_REPORT='<pcr>'
export REVIEW_KIT_SUMMARY='PR #42: Review type: FOLLOW-UP | Instructions: /tmp/instructions/review-followup.md | Incremental diff: /tmp/kit/42/incremental_diff.patch (120 lines)'
RV='$REVIEW_TYPE $PR_AUTHOR $PR_NUMBER $GITHUB_REPOSITORY $PR_HEAD_SHA $PULL_REQUEST_CONTEXT $DIFF_FILE_PATH $TRIGGER_MESSAGE $PREVIOUS_BOT_REVIEWS $AGENT_REVIEW_HISTORY $THREAD_CONTEXT'
BV='$THREAD_CONTEXT $NEW_COMMENT_AUTHOR $NEW_COMMENT_BODY $TRIGGER_MESSAGE $THREAD_NUMBER $GITHUB_REPOSITORY $THREAD_AUTHOR $PR_HEAD_SHA $IS_FIRST_REVIEW $FULL_DIFF_PATH $INCREMENTAL_DIFF_PATH $LAST_REVIEWED_SHA $PR_NUMBER $PREVIOUS_BOT_REVIEWS $AGENT_REVIEW_HISTORY $REVIEW_KIT_SUMMARY'
IV='$ISSUE_CONTEXT $ISSUE_NUMBER $ISSUE_AUTHOR $TRIGGER_MESSAGE $GITHUB_REPOSITORY'
CV='$PR_NUMBER $PR_TITLE $PR_BODY $PR_AUTHOR $PR_HEAD_SHA $CHANGED_FILES $CHANGED_FILES_JSON $PR_LABELS $PREVIOUS_COMPLIANCE_REPORT $TRIGGER_MESSAGE $THREAD_CONTEXT $PREVIOUS_BOT_REVIEWS $AGENT_REVIEW_HISTORY $DIFF_PATH $INCREMENTAL_DIFF_PATH $FILE_GROUPS $REPORT_TEMPLATE $GITHUB_REPOSITORY'
bash .github/scripts/assemble-prompt.sh pr-review-first     | envsubst "$RV" > "$TMP/rf.txt"
REVIEW_TYPE=FOLLOW-UP bash .github/scripts/assemble-prompt.sh pr-review-followup | REVIEW_TYPE=FOLLOW-UP envsubst "$RV" > "$TMP/ru.txt"
bash .github/scripts/assemble-prompt.sh bot-reply          | envsubst "$BV" > "$TMP/br.txt"
bash .github/scripts/assemble-prompt.sh issue-comment      | envsubst "$IV" > "$TMP/ic.txt"
bash .github/scripts/assemble-prompt.sh compliance-first    | envsubst "$CV" > "$TMP/cc.txt"
bash .github/scripts/assemble-prompt.sh compliance-followup | envsubst "$CV" > "$TMP/cf.txt"

PASS=0; FAIL=0
need() { # file pattern label
  if grep -Eqi -- "$2" "$TMP/$1.txt"; then PASS=$((PASS+1)); else echo "FAIL [$1]: $3"; FAIL=$((FAIL+1)); fi
}
neednt() { # file pattern label (must NOT appear)
  if grep -Eqi -- "$2" "$TMP/$1.txt"; then echo "FAIL [$1]: $3 (present, must not be)"; FAIL=$((FAIL+1)); else PASS=$((PASS+1)); fi
}

# ---- universal rules (all modes) ----
for f in rf ru br ic cc cf; do
  need $f 'mirrobot-agent'                     "$f: identity names"
  need $f 'ON-TOPIC GUARDRAIL'                 "$f: on-topic guardrail present"
  need $f 'Off-topic .refuse it.'              "$f: refusal duty"
  need $f 'You may react to comments'          "$f: dynamic reactions part"
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

# ---- review-family rules (pr-review both modes; machinery lives in the
# instruction sets for bot-reply, which pins its loading instead) ----
for f in rf ru; do
  need $f 'hard no'                            "$f: verdict ladder"
  need $f 'mergeable as-is'                    "$f: approval rule"
  need $f 'testing adequate'                   "$f: APPROVE checklist"
  need $f 'Verdict:'                           "$f: verdict line mandate"
  need $f 'last_reviewed_sha:'                 "$f: footer contract"
  need $f 'head_sha.txt'                       "$f: mechanical SHA file"
  need $f 'jq -e'                              "$f: array guard"
  need $f 'review_comments.json'               "$f: comments file"
  need $f 'review_payload'                     "$f: payload file"
  need $f 'HIGH-SIGNAL, LOW-NOISE'             "$f: feedback philosophy"
  need $f 'praise-only'                        "$f: no-praise-only rule"
  need $f 'Severity System'                    "$f: severity part"
  need $f 'critical., .major., .minor., .info.|"severity"' "$f: scratchpad severity field"
  need $f 'grouped under the severity headings' "$f: severity-grouped summary"
  need $f 'harmless is not the bar'             "$f: merge-worthiness gate"
  need $f 'ladder binds for maintainers and admins' "$f: rank-blind verdicts"
done
# bot-reply base: loads the machinery instead of embedding it
need br 'STRATEGY INDEX'                       "br: strategy index"
need br 'generate-review-kit.sh'               "br: review kit loading"
need br 'run the kit yourself'                 "br: cross-thread kit path"
need br 'head_sha.txt'                         "br: footer safety line"
need br 'THREAD CONTEXT'                       "br: thread context block"
need br 'Severity System'                      "br: severity universal"
neednt br '## Verdict Levels'                  "br: verdict machinery extracted"
neednt br '## Review Submission Flow'          "br: submission machinery extracted"
neednt br 'HIGH-SIGNAL, LOW-NOISE'             "br: philosophy machinery extracted"
need rf 'Protocol for FIRST'                   "rf: first protocol"
need rf 'comprehensive, initial analysis'      "rf: first = full PR"
need ru 'Protocol for FOLLOW-UP'               "ru: followup protocol"
need ru 'incremental changes since the last'   "ru: incremental scope"
need ru 'previous feedback'                    "ru: verify-previous-feedback duty"

# ---- write-scope per mode ----
need rf '/tmp scratch files ONLY'             "rf: reviewer write-scope"
need ru '/tmp scratch files ONLY'             "ru: reviewer write-scope"
need ic 'never modify repository files'        "ic: analyst write-scope"
need cc 'never modify repository files'        "cc: compliance write-scope"
need br 'repository files .fixes, features.'   "br: agent write-scope (may modify)"
need br 'workflows'                            "br: workflow-edit deny note"
need br 'Level 1'                              "br: recovery level present"
need br 'refusing to allow'                 "br: workflow-push recovery recipe"
need br 'wording varies by token type'     "br: recovery grep is wording-agnostic"
need br 'failure report'                       "br: fatal-error reporting duty"
need br 'second and final'                     "br: single-recovery limit"

# ---- workflow contracts (grep the raw workflow files) ----
for w in .github/workflows/bot-reply.yml .github/workflows/pr-review.yml .github/workflows/compliance-check.yml; do
  if grep -q '\["mirrobot",' "$w" 2>/dev/null && grep -q 'BOT_NAMES_JSON' "$w"; then
    echo "FAIL: $w: BOT_NAMES_JSON contains bare mirrobot (name != identity)"; FAILED=1
  fi
  grep -q 'mirrobot-agent\[bot\]' "$w" || { echo "FAIL: $w: BOT_NAMES_JSON missing app identity"; FAILED=1; }
done
for w in .github/workflows/bot-reply.yml .github/workflows/pr-review.yml .github/workflows/compliance-check.yml .github/workflows/issue-comment.yml; do
  grep -q 'CONTEXT_IGNORE_AUTHORS' "$w" || { echo "FAIL: $w: CONTEXT_IGNORE_AUTHORS unwired"; FAILED=1; }
  grep -q 'CONTEXT_FILTER_PATTERNS_JSON' "$w" || { echo "FAIL: $w: CONTEXT_FILTER_PATTERNS_JSON unwired"; FAILED=1; }
  grep -q 'ellipsis' "$w" && { echo "FAIL: $w: hardcoded ellipsis survives"; FAILED=1; }
done
grep -q 'head -1 /tmp/scrub-taint.txt' .github/workflows/pr-review.yml || { echo 'FAIL: pr-review taint summary not head -1'; FAILED=1; }
grep -q 'head -1 /tmp/scrub-taint.txt' .github/workflows/compliance-check.yml || { echo 'FAIL: compliance taint summary not head -1'; FAILED=1; }
grep -q 'head -1 /tmp/scrub-taint.txt' .github/workflows/bot-reply.yml || { echo 'FAIL: bot-reply taint summary not head -1'; FAILED=1; }
grep -q 'cut -c1-600' .github/workflows/*.yml && { echo 'FAIL: flatten-cut taint pattern survives'; FAILED=1; }

# ---- mode-specific ----
need cc 'compliance'                           "cc: compliance mission"
need cc 'file group'                           "cc: file groups wiring"
need cc 'template'                             "cc: report template wiring"
need ic 'Initial Analysis Report'              "ic: analysis output shape"
need ic 'thanks'                               "ic: acknowledgment duty"

# ---- context noise filtering + identity (2026-08 rework) ----
for f in rf ru cc cf; do
  need $f 'Other AI reviewers'                   "$f: AI-reviewer input-not-authority stance"
  need $f 'never authority'                      "$f: never defer clause"
  need $f 'noise posts .rate-limit.skip notices.' "$f: filtering explained"
done
need ru 'your NAME, not an identity'           "ru: name-vs-identity rule"
need br 'your name, not an identity'           "br: name-vs-identity rule"
# ---- severity universal + inline format (agent trait, every mode) ----
for f in rf ru br ic cc cf; do
  need $f 'Severity System'                      "$f: severity part present"
  need $f '🔴 **Critical**'                      "$f: inline severity format"
done

# ---- scope of action (universal; agent trait) ----
for f in rf ru br ic cc cf; do
  need $f 'SCOPE OF ACTION'                      "$f: scope part present"
  need $f 'A request is not authority'           "$f: request-not-authority rule"
  need $f 'Leads, not authority'                 "$f: leads rule"
  need $f 'Attribution'                          "$f: attribution rule"
done
neednt rf 'explicitly state your curation logic' "rf: curation ceremony removed"
neednt ru 'explicitly state your curation logic' "ru: curation ceremony removed"
need rf 'Place the Findings'                     "rf: placement step"
need ru 'Place the Findings'                     "ru: placement step"
need rf 'your choice: jq batch appends'          "rf: tool-agnostic scratchpad"

# ---- trigger-message + three-block context (WS4/WS2+3) ----
for f in rf ru br ic cc cf; do
  need $f 'THE REQUEST THAT TRIGGERED YOU'    "$f: trigger-message block present"
  need $f 'This is the request to answer'     "$f: trigger primacy rule"
done
for f in rf ru cc cf; do
  need $f 'YOUR PREVIOUS REVIEWS'             "$f: elevated agent reviews block"
  need $f 'YOUR OLDER REVIEWS'                "$f: agent review history block"
  need $f 'THREAD CONTEXT'                    "$f: thread context block"
  need $f 'untrusted data'                    "$f: thread context untrusted rule"
done

# ---- compliance status semantics (WS1) + protocol split ----
need cc 'error., .failure., .pending., .success.|accepts ONLY these states' "cc: status enum documented"
neednt cc "state='neutral'"                   "cc: neutral state banned"
need cc 'Passed with warnings - see report'   "cc: warnings status string"
need cc 'Blocking issues - see report'        "cc: blocking status string"
need cc 'target_url'                          "cc: status links to report"
need cc 'Post the report BEFORE the status'   "cc: post-report-first order"
need cc 'good-practices|GOOD PRACTICES'      "cc: practices-audit framing"
need cc 'Hex SHA discipline'                "cc: sha-file rule"
need cc 'deliberately audit a different commit' "cc: honest-declaration carve-out"
need cc 'cat /tmp/head_sha.txt'              "cc: status URL sources sha file"
need cc 'Severity Groups'                    "cc: severity-grouped report"
need cc '#### ✅ \[Group Name\] - COMPLIANT'   "cc: compliance group icons"
need cc 'Incomplete README'                  "cc: blocked example bullet"
need rf 'Nothing relevant is off-limits'     "rf: reviewer scope not fenced"
need rf 'review-mode discipline'             "rf: review/contribute hand-off"
for f in rf ru; do
  need $f 'the review IS the deliverable'     "$f: no post-review comment rule"
  need $f 'No follow-up comment announcing|no announcement after' "$f: closing rule explicit"
done
need cc 'Protocol for FIRST Compliance Check'  "cc: FIRST protocol present"
neednt cc 'Protocol for FOLLOW-UP Compliance Check' "cc: no stray FOLLOW-UP protocol"
need cf 'Protocol for FOLLOW-UP Compliance Check' "cf: FOLLOW-UP protocol present"
neednt cf 'Protocol for FIRST Compliance Check'   "cf: no stray FIRST protocol"
need cf 'Resolved'                            "cf: re-verify previous findings"
need cf 'Carry unresolved findings forward'   "cf: carry-forward rule"
need cf 'primary scope is the changes since the last checked commit' "cf: incremental diff wiring"

# ---- CI status discipline (WS5, all modes via tool-restrictions) ----
for f in rf ru br ic cc cf; do
  need $f 'compliance-check. status is owned'  "$f: status exclusivity rule"
done

# ---- clean-prose invariants (generator-artifact class) ----
for f in rf ru br ic cc cf; do
  neednt $f "' \+ '"                            "$f: no powershell concat artifacts"
  neednt $f 'chr\(39\)'                         "$f: no python chr() artifacts"
  neednt $f '\?\?'                              "$f: no mangled emoji sequences"
done

echo "----"; echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
