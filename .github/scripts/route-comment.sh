#!/usr/bin/env bash
# route-comment.sh — shared comment-trigger decision logic.
#
# The single source of truth for "does this comment trigger which agent":
# agent-router.yml parses every incoming comment with it, bot-reply.yml
# re-validates routed comments with it (defense in depth), and
# scrub-fixtures.sh exercises IT (not a copy) — so routing semantics cannot
# drift between the three.
#
# Contract:
#   stdin : the raw comment body
#   $1    : "true" | "false" — is the thread a pull request?
#   stdout: zero or more of  review | compliance | reply  (space-separated)
#   exit  : 0 always (a no-trigger comment is a normal outcome)
#
# Semantics (must replicate the ORIGINAL per-workflow guards exactly):
#   - /mirrobot-review or /mirrobot_review  -> review      (PRs only)
#   - /mirrobot-check  or /mirrobot_check   -> compliance  (PRs only)
#   - @mirrobot or @mirrobot-agent mention  -> reply
#   - compound comments yield ALL matches (parallel agents, as before)
#   - trigger words count ONLY in actual content: fenced code blocks
#     (```), inline code (`...`), and quoted lines (> ...) are stripped first
#   - NOTE (preserved loose matching): @mirrobot matches as a substring,
#     identical to the original contains() guards — e.g. @mirrobotics.com
#     triggers; tightened only if the original semantics are deliberately
#     changed some day.
set -u
is_pr="${1:-false}"

clean=$(awk '
  /^```/ { in_code = !in_code; next }
  !in_code { print }
' | sed 's/`[^`]*`//g' | grep -v '^[[:space:]]*>' || true)

routes=""
if printf '%s' "$clean" | grep -qE '/mirrobot[-_]review'; then routes="$routes review"; fi
if printf '%s' "$clean" | grep -qE '/mirrobot[-_]check';  then routes="$routes compliance"; fi
if printf '%s' "$clean" | grep -qE '@mirrobot(-agent)?';  then routes="$routes reply"; fi

# Review/compliance commands only apply to pull requests.
if [ "$is_pr" != "true" ]; then
  routes=$(printf '%s' "$routes" | sed 's/ review//; s/ compliance//')
fi

routes="${routes# }"   # trim the single leading space, if any
printf '%s\n' "${routes:-none}"
exit 0
