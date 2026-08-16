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
#   env: GH_TOKEN (App token — reactions post as the bot), GITHUB_REPOSITORY,
#        BOT_LOGIN (default mirrobot-agent[bot])
#   exit: 0 on all runtime paths (reactions are cosmetic); the ${:?} guards
#         exit 1 on MISCONFIGURATION (missing args/env) - loud by design; every
#         call site carries continue-on-error, so a guard firing is visible
#         but never fails the run.
set -uo pipefail

action="${1:?usage: react.sh <start|success|failure> <comment|issue> <id>}"
kind="${2:?target type: comment|issue}"
target_id="${3:?target id}"
: "${GH_TOKEN:?}" "${GITHUB_REPOSITORY:?}"
BOT_LOGIN="${BOT_LOGIN:-mirrobot-agent[bot]}"

if [ "$kind" = "comment" ]; then
  base="/repos/${GITHUB_REPOSITORY}/issues/comments/${target_id}/reactions"
else
  base="/repos/${GITHUB_REPOSITORY}/issues/${target_id}/reactions"
fi

add() { # content
  gh api --method POST -H "Accept: application/vnd.github+json" "$base" -f content="$1" >/dev/null 2>&1 || true
}

remove_own() { # content — delete OUR bot's reactions of this type (idempotent)
  gh api -H "Accept: application/vnd.github+json" "$base" --paginate 2>/dev/null \
    | jq -r --arg bot "$BOT_LOGIN" --arg content "$1" '.[]? | select(.user.login == $bot and .content == $content) | .id' \
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
