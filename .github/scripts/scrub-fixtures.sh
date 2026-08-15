#!/usr/bin/env bash
# Regression fixtures for scrub-workspace.sh taint logic + fetch-roster.sh
# roster transforms + the permission profile's jq-env deny patterns.
#
# Run:  bash .github/scripts/scrub-fixtures.sh
# Requires: git, jq, bash. Exits non-zero on any failure. Creates no files
# outside a mktemp -d directory (cleaned up on exit) and /tmp/scrub-*.log.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRUB="$SCRIPT_DIR/scrub-workspace.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
PASS=0; FAIL=0

check() { if [ "$2" = "$3" ]; then echo "PASS: $1"; PASS=$((PASS+1)); else echo "FAIL: $1"; echo "  want=[$2]"; echo "  got =[$3]"; FAIL=$((FAIL+1)); fi; }

# ---- fixture repo ----------------------------------------------------------
SRC="$WORK/src"; mkdir -p "$SRC/.github/workflows"; cd "$SRC" || exit 1
git init -q -b main .; git config user.email t@t; git config user.name t
printf 'wf: v1\n' > .github/workflows/main.yml; printf 'doc one\n' > DOC.md
git add -A; git commit -qm A
git branch stale main; git branch evil main; git branch mergebase main
printf 'wf: v2 hardened\n' > .github/workflows/main.yml; printf 'new\n' > .github/workflows/new.yml
printf 'doc two\n' > DOC.md
git add -A; git commit -qm 'C: main hardens .github (conflicts with stale DOC)'
git checkout -q stale;  printf 'typo fix\n' > DOC.md; git add -A; git commit -qm 'stale: docs only (based before C)'
git checkout -q evil;   printf 'wf: MALICIOUS\n' > .github/workflows/main.yml; git add -A; git commit -qm 'evil: modify workflow'
git checkout -q -b revert-hide main
printf 'wf: MALICIOUS\n' > .github/workflows/main.yml; git add -A; git commit -qm 'sneak: modify workflow'
printf 'wf: v2 hardened\n' > .github/workflows/main.yml; git add -A; git commit -qm 'sneak: revert (net tree identical)'
# evil-merge: branch with NO .github commits merges main, resolution smuggles
# a .github edit into the merge commit (no per-commit file lines in git log).
git checkout -q mergebase; printf 'doc three\n' > DOC.md; git add -A; git commit -qm 'mb: docs change (will conflict)'
git merge -q --no-commit main >/dev/null 2>&1 || true
printf 'doc merged\n' > DOC.md; printf 'wf: EVIL MERGE\n' > .github/workflows/main.yml
git add -A; git commit -qm 'evil merge: .github edit hidden in merge resolution'
git checkout -q main

cd "$WORK" && git clone -q "$SRC" work && cd work || exit 1
git fetch -q origin '+refs/heads/*:refs/remotes/origin/*'

run_scrub() { # branch -> ALARM|INFO|CLEAN
  git checkout -q --detach "origin/$1"
  rm -f /tmp/scrub-taint.txt
  bash "$SCRUB" --anchor main >/tmp/scrub-fix.log 2>&1
  if [ -s /tmp/scrub-taint.txt ] && grep -q "TAINT ALERT" /tmp/scrub-taint.txt; then echo ALARM
  elif [ -s /tmp/scrub-taint.txt ] && grep -q "EXPLAINED" /tmp/scrub-taint.txt; then echo INFO
  elif [ ! -s /tmp/scrub-taint.txt ]; then echo CLEAN
  else echo UNKNOWN; fi
}

# ---- taint matrix ----------------------------------------------------------
check "syntax scrub-workspace" OK "$(bash -n "$SCRUB" && echo OK)"
check "syntax fetch-roster"    OK "$(bash -n "$SCRIPT_DIR/fetch-roster.sh" && echo OK)"
check "stale-base docs branch -> INFO (informed, not alarmed)"  INFO  "$(run_scrub stale)"
check "direct .github modify -> ALARM"                          ALARM "$(run_scrub evil)"
check "modify+revert identical tree -> ALARM"                   ALARM "$(run_scrub revert-hide)"
check "evil merge (no per-commit .github lines) -> ALARM"       ALARM "$(run_scrub mergebase)"
check "anchor tip itself -> CLEAN"                              CLEAN "$(run_scrub main)"

# ---- channel hygiene -------------------------------------------------------
git checkout -q --detach origin/evil; rm -f /tmp/scrub-taint.txt; bash "$SCRUB" --anchor main >/dev/null 2>&1
flat=$(tr '\n' ' ' < /tmp/scrub-taint.txt | tr -s ' ' | cut -c1-600)
check "scrutiny instruction survives 600-char flatten+cut" yes "$(echo "$flat" | grep -q 'MAXIMUM SCRUTINY' && echo yes || echo no)"
check "attacker commit subjects never enter the alert"     no  "$(grep -q 'evil: modify workflow' /tmp/scrub-taint.txt && echo yes || echo no)"

# ---- roster transforms -----------------------------------------------------
pages='[{"login":"Mirrowel"},{"login":"contributor1"}]
[{"login":"contributor2"}]'
got=$(printf '%s\n' "$pages" | jq -sr --arg extra "Trusted-Ghost; contributor1 , ,x" \
  '[.[][].login] + ($extra | split("[,; \t\n]+"; null) | map(select(length > 0))) | map(ascii_downcase) | sort | unique | join(", ")')
check "roster: union + semicolon + downcase-dedupe + empty-skip" "contributor1, contributor2, mirrowel, trusted-ghost, x" "$got"
got2=$(printf '[{"login":"Mirrowel"}]\n' | jq -sr --arg extra "" \
  '[.[][].login] + ($extra | split("[,; \t\n]+"; null) | map(select(length > 0))) | map(ascii_downcase) | sort | unique | join(", ")')
check "roster: empty extras" "mirrowel" "$got2"

# ---- requester-context trusted-user compare (case-insensitive parity) -----
rc_match() { # login trusted_list -> 1 if listed (replicates action.yml loop)
  local login="$1" list="$2" trusted=0 login_lc cand_lc cand
  login_lc=$(printf '%s' "$login" | tr '[:upper:]' '[:lower:]')
  for cand in $(printf '%s' "$list" | tr ',;' '  '); do
    [ -n "$cand" ] || continue
    cand_lc=$(printf '%s' "$cand" | tr '[:upper:]' '[:lower:]')
    [ "$cand_lc" = "$login_lc" ] && trusted=1
  done
  echo "$trusted"
}
check "requester-context: non-canonical case entry matches" 1 "$(rc_match SomeUser 'other, SOMEUSER, x')"
check "requester-context: exact entry matches"              1 "$(rc_match octocat 'octocat')"
check "requester-context: different user does not match"    0 "$(rc_match octocat 'someoneelse')"
check "requester-context: semicolon-separated matches"      1 "$(rc_match octocat 'a; OCTOCAT')"

# ---- permission pattern matrix (fnmatch semantics, as opencode uses) -------
Q="'"
deny_rules=("jq -n env*" "jq -n ${Q}env*" "jq -n \"env*" "jq -n \$ENV*" "jq -n ${Q}\$ENV*" "jq *\$ENV*")
allowed_tests=("jq -n --arg event REQUEST_CHANGES {x: \$event}" "jq --rawfile body /tmp/b.md ." "jq -c . /tmp/x.json")
denied_tests=("jq -n env" "jq -n ${Q}env" "jq -n \"env" "jq -n ${Q}env.GITHUB_TOKEN" "jq -n \$ENV" "jq -n ${Q}\$ENV" "jq .a \$ENV")
pt=0; for t in "${allowed_tests[@]}"; do for r in "${deny_rules[@]}"; do [[ $t == $r ]] && pt=1; done; done
check "permission: legit jq flows unaffected" 0 "$pt"
pt=0; for t in "${denied_tests[@]}"; do hit=0; for r in "${deny_rules[@]}"; do [[ $t == $r ]] && hit=1; done; [ $hit -eq 0 ] && pt=1; done
check "permission: all env-dump forms denied" 0 "$pt"

# ---- agent-router decision matrix (exercises the REAL route-comment.sh) ----
route() { # body is_pr -> flags or "none" — delegates to the shared script
  # SCRIPT_DIR is the absolute path computed at script start (line 9); do NOT
  # re-derive it here — the fixture sections above change CWD, so a relative
  # re-derivation would resolve against the fixture repo and break in CI.
  printf '%s' "$1" | bash "$SCRIPT_DIR/route-comment.sh" "$2"
}
check "router: plain mention (PR)"          "reply"                  "$(route 'hey @mirrobot look at this' true)"
check "router: plain mention (issue)"       "reply"                  "$(route 'hey @mirrobot look at this' false)"
check "router: review command (PR)"         "review"                 "$(route 'please /mirrobot-review' true)"
check "router: review command (issue)"      "none"                   "$(route 'please /mirrobot-review' false)"
check "router: check underscore (PR)"       "compliance"             "$(route '/mirrobot_check' true)"
check "router: compound comment (PR)"       "review compliance reply" "$(route '@mirrobot run /mirrobot-review then /mirrobot-check' true)"
check "router: mention in code fence"       "none"                   "$(route 'look:
````
@mirrobot
````
done' false)"
check "router: mention inline code"         "none"                   "$(route 'the `@mirrobot` token' false)"
check "router: mention quoted"              "none"                   "$(route '> @mirrobot said that' false)"
check "router: review cmd quoted"           "none"                   "$(route '> /mirrobot-review' true)"
check "router: mention agent suffix"        "reply"                  "$(route '@mirrobot-agent ping' false)"
check "router: substring (matches - original semantics were substring too)" "reply" "$(route 'email support@mirrobotics.com' false)"
check "router: quoted cmd + real mention"   "reply"                  "$(route '> /mirrobot-review
@mirrobot hi' true)"

echo "----"; echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
