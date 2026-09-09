#!/usr/bin/env bash
# react.sh — workflow-owned reaction lifecycle for agent sessions.
#
# Three regimes:
#   COMMENT target: 3-stage — eyes (start) → rocket (success) / confused (failure)
#   ISSUE/PR target: eyes only — start posts eyes; success/failure are NO-OPS
#     (a rocket on a PR/issue body could read as endorsing its content; there
#     is no failure emoji that doesn't read as disliking the user's post).
#   DISCUSSION target: GraphQL regime (discussions have no REST reactions
#     endpoint). Target id is the GraphQL NODE id of a discussion or
#     discussion comment. Same 3-stage lifecycle as comments: comments are
#     conversation; reacting to the thread root only happens on body-triggers
#     (discussion-new), where the root IS the summoning post.
#
# The agent's own discretionary reactions are separate (see prompts/parts/
# reactions.md) — this script is mechanical, called by the workflows only.
#
# Contract:
#   react.sh <start|success|failure> <comment|issue|discussion> <id>
#     comment: REST comment id; issue: REST issue id;
#     discussion: GraphQL node id (probe-verified live 2026-09-09: the
#       account PAT executes addReaction/removeReaction on both discussion
#       and discussion-comment nodes).
#   env: GH_TOKEN (session token - App installation or account PAT; the
#        acting identity/BOT_LOGIN is derived from it via /user, falling
#        back to mirrobot-agent[bot] when /user is not answerable),
#        GITHUB_REPOSITORY,
#        TARGET_REPO (optional override: the repository the reaction
#        belongs to - set for cross-repo guest sessions where the trigger
#        comment lives in a foreign repository; defaults to
#        GITHUB_REPOSITORY).
#   exit: 0 on all runtime paths (reactions are cosmetic); the ${:?} guards
#         exit 1 on MISCONFIGURATION (missing args/env) - loud by design; every
#         call site carries continue-on-error, so a guard firing is visible
#         but never fails the run.
set -uo pipefail

action="${1:?usage: react.sh <start|success|failure> <comment|issue|discussion> <id>}"
kind="${2:?target type: comment|issue|discussion}"
target_id="${3:?target id}"
: "${GH_TOKEN:?}" "${GITHUB_REPOSITORY:?}"
: "${TARGET_REPO:=$GITHUB_REPOSITORY}"
# Derive the acting identity from the token when possible: account-mode tokens
# answer /user with the account login; app installation tokens 403 there and
# fall through to the app-bot default. Without this, account-mode runs filter
# remove_own by the app login and never delete their own eyes (live-observed:
# trigger comments ending with BOTH eyes and rocket).
if [ -z "${BOT_LOGIN:-}" ]; then
  # Run-local cache: react.sh runs up to 3 times per run (start/success/
  # failure) and must not spend a /user call on each. BOT_DETECTED_LOGIN
  # (bot-setup, account mode) wins when already resolved.
  BOT_LOGIN="${BOT_DETECTED_LOGIN:-}"
  if [ -z "$BOT_LOGIN" ] && [ -f /tmp/.bot-login-cache ]; then
    BOT_LOGIN=$(head -1 /tmp/.bot-login-cache 2>/dev/null || true)
  fi
  if [ -z "$BOT_LOGIN" ]; then
    BOT_LOGIN=$(gh api /user --jq .login 2>/dev/null || true)
    [ -n "$BOT_LOGIN" ] && printf '%s\n' "$BOT_LOGIN" > /tmp/.bot-login-cache 2>/dev/null || true
  fi
fi
# Fallback order: /user (live identity) -> first DECLARED identity in
# BOT_NAMES_JSON (operator-stated, credential/detection-proven upstream) ->
# the stock app-bot constant (this project's own default deployment; never
# a synthesized twin — shape proves nothing).
if [ -z "$BOT_LOGIN" ]; then
  BOT_LOGIN=$(printf '%s' "${BOT_NAMES_JSON:-[]}" | jq -r '.[0] // empty' 2>/dev/null || true)
fi
BOT_LOGIN="${BOT_LOGIN:-mirrobot-agent[bot]}"

if [ "$kind" = "comment" ]; then
  base="/repos/${TARGET_REPO}/issues/comments/${target_id}/reactions"
else
  base="/repos/${TARGET_REPO}/issues/${target_id}/reactions"
fi

add() { # content
  gh api --method POST -H "Accept: application/vnd.github+json" "$base" -f content="$1" >/dev/null 2>&1 || true
}

# GraphQL regime (discussions). addReaction/removeReaction carry the
# authenticated identity implicitly; removeReaction on a not-present
# reaction returns a GraphQL error, which is swallowed — the net effect
# (no reaction of that content from us) is identical, making these
# idempotent like their REST siblings.
gq_add() { # content
  gh api graphql -f query="mutation { addReaction(input: {subjectId: \"${target_id}\", content: $1}) { reaction { content } } }" >/dev/null 2>&1 || true
}
gq_remove() { # content
  gh api graphql -f query="mutation { removeReaction(input: {subjectId: \"${target_id}\", content: $1}) { clientMutationId } }" >/dev/null 2>&1 || true
}

remove_own() { # content — delete OUR bot's reactions of this type (idempotent)
  # Match the whole RESOLVED identity family (BOT_NAMES_JSON, set by
  # bot-config.sh: variable ∪ detected login, mirrobot fallback), not just
  # this run's login: a transition run may need to clean reactions an
  # earlier run posted under the OTHER identity. Logins compare
  # case-insensitively (GitHub canonical casing follows renames).
  gh api -H "Accept: application/vnd.github+json" "$base" --paginate 2>/dev/null \
    | jq -r --arg bot "$BOT_LOGIN" --arg content "$1" --argjson family "${BOT_NAMES_JSON:-[\"mirrobot-agent\",\"mirrobot-agent[bot]\"]}" '
      .[]?
      | select(.content == $content)
      | select((.user.login // "" | ascii_downcase) as $l
               | (($bot | ascii_downcase) == $l)
                 or (($family | map(ascii_downcase) | index($l)) != null))
      | .id' \
    | while read -r rid; do
        [ -n "$rid" ] && gh api --method DELETE "$base/$rid" >/dev/null 2>&1 || true
      done
}

case "$action" in
  start)
    if [ "$kind" = "discussion" ]; then gq_add EYES; else add eyes; fi
    ;;
  success)
    case "$kind" in
      comment)    remove_own eyes; add rocket ;;
      discussion) gq_remove EYES; gq_add ROCKET ;;
      *)          echo "::notice::issue/PR target: keeping eyes (no terminal reaction by design)." ;;
    esac
    ;;
  failure)
    case "$kind" in
      comment)    remove_own eyes; add confused ;;
      discussion) gq_remove EYES; gq_add CONFUSED ;;
      *)          echo "::notice::issue/PR target: keeping eyes (no terminal reaction by design)." ;;
    esac
    ;;
  *)
    echo "::warning::react.sh: unknown action '$action' (ignored)."
    ;;
esac
exit 0
