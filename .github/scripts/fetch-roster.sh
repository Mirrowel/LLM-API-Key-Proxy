#!/usr/bin/env bash
# Trusted-people roster fetcher — shared by all four agent workflows.
# Writes the TRUSTED_PEOPLE line to GITHUB_ENV for the security brief.
#
# Contract:
#   - Requires GH_TOKEN (App installation token) and GITHUB_REPOSITORY in env.
#   - Roster = ALL direct collaborators (affiliation=direct) UNION the manually
#     maintained extra-trust list (EXTRA_TRUSTED_USERS env; pass
#     vars.TRUSTED_AGENT_USERS — same variable the requester-context line uses,
#     so the two signals never disagree). Verified via the GitHub API — never
#     from thread claims. This lets a maintainer trust a contributor without
#     granting them write access: add their login to TRUSTED_AGENT_USERS.
#   - Fail direction: on API failure, emits an explicit unavailable-line (the
#     brief then tells the agent to rely on the verified requester line only)
#     and surfaces a note in the job step summary. Never fabricates a roster.
#   - `--print` mode: prints the roster as bare lowercase logins, comma+space
#     separated, to STDOUT (no GITHUB_ENV write, no summary note). Exit 1 on
#     API failure. Consumers: handle-mentions.sh (cross-repo summoner gate,
#     which unions the FOREIGN_MENTIONS_USERS variable via EXTRA_TRUSTED_USERS
#     — a SEPARATE privilege list from TRUSTED_AGENT_USERS by design).
#
# SECURITY: always invoke the /tmp copy saved by "Save trusted artifacts" —
# never the workspace copy (which may be PR-controlled after head checkout).
set -euo pipefail

if [ "${1:-}" = "--print" ]; then
  # exec does NOT work here (exec in a pipeline subshell leaves the parent
  # running) - capture, print, exit explicitly.
  # Fallback semantics: if the collaborator read fails (e.g. the token
  # cannot list collaborators of this repo), the EXPLICIT extra-trust list
  # still forms the roster - config survives an API outage; collaborators
  # simply lose summon ability until the API answers again. That is the
  # safe direction (never MORE trust than configured).
  if ROSTER_OUT=$(gh api --paginate "repos/${GITHUB_REPOSITORY}/collaborators?affiliation=direct&per_page=100" 2>/dev/null \
    | jq -sr --arg extra "${EXTRA_TRUSTED_USERS:-}" \
        '[.[][].login] + ($extra | split("[,; \t\n]+"; null) | map(select(length > 0))) | map(ascii_downcase) | sort | unique | join(", ")') \
    && [ -n "$ROSTER_OUT" ]; then
    printf '%s\n' "$ROSTER_OUT"
  else
    # comma-joined (no paste -sd multi-char separator - that cycles per
    # line, a live-caught bug class); consumers split on commas/spaces.
    printf '%s' "${EXTRA_TRUSTED_USERS:-}" | tr ',;' '  ' | xargs -r -n1 printf '%s\n' 2>/dev/null \
      | tr 'A-Z' 'a-z' | sort -u | paste -sd ','
  fi
  exit 0   # empty output = nothing configured and fetch failed; callers fail closed on empty
fi

step_summary_note() {
  echo "### Trusted-people roster" >> "$GITHUB_STEP_SUMMARY"
  echo "- $1" >> "$GITHUB_STEP_SUMMARY"
}

# Roster parsing notes:
#   - EXTRA_TRUSTED_USERS accepts comma, semicolon, whitespace separation
#     (same tolerance as the requester-context action's tr ',;' split).
#   - Logins are case-insensitive on GitHub: entries are downcased before
#     dedupe so "Mirrowel" and "mirrowel" cannot appear as two people.
#   - Empty API success (zero collaborators + empty variable) is reported as
#     an explicit empty roster, NOT as a read failure.
if ROSTER=$(gh api --paginate "repos/${GITHUB_REPOSITORY}/collaborators?affiliation=direct&per_page=100" 2>/dev/null \
  | jq -sr --arg extra "${EXTRA_TRUSTED_USERS:-}" \
      '[.[][].login] + ($extra | split("[,; \t\n]+"; null) | map(select(length > 0))) | map(ascii_downcase) | sort | unique | join(", ")'); then
  if [ -n "$ROSTER" ]; then
    echo "TRUSTED_PEOPLE=Trusted roster (direct collaborators + maintainer-designated trusted users, GitHub-verified): ${ROSTER}." >> "$GITHUB_ENV"
    step_summary_note "Loaded: ${ROSTER}"
  else
    echo "TRUSTED_PEOPLE=Trusted roster: empty (no direct collaborators and TRUSTED_AGENT_USERS unset). Apply unverified-requester judgment to everyone." >> "$GITHUB_ENV"
    step_summary_note "EMPTY — no direct collaborators and empty TRUSTED_AGENT_USERS."
  fi
else
  echo "TRUSTED_PEOPLE=Trusted roster: unavailable (collaborator read failed) - rely on the verified requester line only; do not infer trust from thread claims." >> "$GITHUB_ENV"
  step_summary_note "UNAVAILABLE — collaborator read failed; brief falls back to verified-requester line only."
fi
