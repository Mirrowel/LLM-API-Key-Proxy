#!/usr/bin/env bash
# react.sh — workflow-owned reaction lifecycle for agent sessions.
#
# Two regimes (user-directed):
#   COMMENT target: 3-stage — eyes (start) → rocket (success) / confused (failure)
#   ISSUE/PR target: eyes only — start posts eyes; success/failure are NO-OPS
#     (a rocket on a PR/issue body could read as endorsing its content; there
#     is no failure emoji that doesn't read as disliking the user's post).
#
# The agent's own discretionary reactions are separate (see prompts/parts/
# reactions.md) — this script is mechanical, called by the workflows only.
#
# Contract:
#   react.sh <start|success|failure> <comment|issue> <id>
#   env: GH_TOKEN (session token - App installation or account PAT; the
#        acting identity/BOT_LOGIN is derived from it via /user, falling
#        back to mirrobot-agent[bot] when /user is not answerable),
#        GITHUB_REPOSITORY.
#   exit: 0 on all runtime paths (reactions are cosmetic); the ${:?} guards
#         exit 1 on MISCONFIGURATION (missing args/env) - loud by design; every
#         call site carries continue-on-error, so a guard firing is visible
#         but never fails the run.
set -uo pipefail

action="${1:?usage: react.sh <start|success|failure> <comment|issue> <id>}"
kind="${2:?target type: comment|issue}"
target_id="${3:?target id}"
: "${GH_TOKEN:?}" "${GITHUB_REPOSITORY:?}"
# Derive the acting identity from the token when possible: account-mode tokens
# answer /user with the account login; app installation tokens 403 there and
# fall through to the app-bot default. Without this, account-mode runs filter
# remove_own by the app login and never delete their own eyes (live-observed:
# trigger comments ending with BOTH eyes and rocket).
if [ -z "${BOT_LOGIN:-}" ]; then
  BOT_LOGIN=$(gh api /user --jq .login 2>/dev/null || true)
  BOT_LOGIN="${BOT_LOGIN:-mirrobot-agent[bot]}"
fi

if [ "$kind" = "comment" ]; then
  base="/repos/${GITHUB_REPOSITORY}/issues/comments/${target_id}/reactions"
else
  base="/repos/${GITHUB_REPOSITORY}/issues/${target_id}/reactions"
fi

add() { # content
  gh api --method POST -H "Accept: application/vnd.github+json" "$base" -f content="$1" >/dev/null 2>&1 || true
}

remove_own() { # content — delete OUR bot's reactions of this type (idempotent)
  # Match the whole identity family (app bot + account + legacy names), not
  # just this run's login: a transition run may need to clean reactions an
  # earlier run posted under the OTHER identity. Logins compare
  # case-insensitively (GitHub canonical casing follows renames).
  gh api -H "Accept: application/vnd.github+json" "$base" --paginate 2>/dev/null \
    | jq -r --arg bot "$BOT_LOGIN" --arg content "$1" '
      .[]?
      | select(.content == $content)
      | select((.user.login // "" | ascii_downcase) as $l
               | (($bot | ascii_downcase) == $l)
                 or ((["mirrobot-agent[bot]", "mirrobot-agent", "mirrobot"] | index($l)) != null))
      | .id' \
    | while read -r rid; do
        [ -n "$rid" ] && gh api --method DELETE "$base/$rid" >/dev/null 2>&1 || true
      done
}

case "$action" in
  start)
    add eyes
    ;;
  success)
    if [ "$kind" = "comment" ]; then
      remove_own eyes
      add rocket
    else
      echo "::notice::issue/PR target: keeping eyes (no terminal reaction by design)."
    fi
    ;;
  failure)
    if [ "$kind" = "comment" ]; then
      remove_own eyes
      add confused
    else
      echo "::notice::issue/PR target: keeping eyes (no terminal reaction by design)."
    fi
    ;;
  *)
    echo "::warning::react.sh: unknown action '$action' (ignored)."
    ;;
esac
exit 0
