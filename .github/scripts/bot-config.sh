#!/usr/bin/env bash
# ============================================================================
# bot-config.sh — one-time per-run resolution of the bot's IDENTITY and
# TRIGGER configuration. Run early (after bot-setup) in every agent
# workflow; exports to $GITHUB_ENV (unguessable-delimiter free: values are
# single-line). The router runs it vars-only (no token) — detection simply
# stays empty there.
# ============================================================================
# TWO SEPARATE SYSTEMS — do not conflate:
#
#   IDENTITY ("is this content authored by ME?"): loop guards, review
#   attribution, FIRST/FOLLOW-UP markers, footer verification, reaction
#   cleanup. Resolution:
#     identities = vars.BOT_IDENTITIES (if set; comma-separated logins)
#                ∪ BOT_DETECTED_LOGIN (account mode: /user from
#                                      bot-setup; absent otherwise)
#     fallback "mirrobot-agent, mirrobot-agent[bot]" applies ONLY when
#     both sources are empty. Bare "mirrobot" is NEVER an identity (the
#     username is taken; a spoofed account must never be treated as self).
#     NO "[bot]" twin is EVER synthesized from a detected login: app slugs
#     and usernames are separate GitHub namespaces, so name shape proves
#     nothing (a twin is trusted ONLY when explicitly declared in the
#     variable — operator controls that app — or shipped in the stock
#     fallback, which this project verifiably owns).
#
#   TRIGGERS ("what text summons ME?"): routing words. Resolution:
#     stems = vars.BOT_TRIGGERS (comma-separated raw names, if set)
#           else derived from the resolved identity names (strips [bot])
#     fallback "mirrobot, mirrobot-agent" only when nothing above yields.
#     From every stem the router derives: @<stem> (mention), /<stem>-review,
#     /<stem>_review, /<stem>-check, /<stem>_check. A stem is a RAW NAME:
#     no @ or / prefix in the variable (the prefixes are the derivation).
#
# env in : BOT_IDENTITIES_INPUT (vars.BOT_IDENTITIES passthrough, may be '')
#          BOT_DETECTED_LOGIN   (account-mode /user login, may be unset)
#          BOT_TRIGGERS_INPUT   (vars.BOT_TRIGGERS passthrough, may be '')
# env out: BOT_NAMES_JSON    resolved identity array (legacy-compatible name:
#                            every consumer script already reads this env)
#          BOT_TRIGGER_STEMS comma-separated stems for the router
#          BOT_IDENTITY_LIST display-case comma list (prompt prose self-checks)
#          BOT_IDENTITY_PRIMARY display-case primary login (footers, prose)
# Exit: 0 (bad variable input degrades loudly to defaults — config mistakes
#       must be visible, not silent).
# ============================================================================
set -u

FALLBACK_IDENTITIES='["mirrobot-agent","mirrobot-agent[bot]"]'
FALLBACK_TRIGGERS="mirrobot,mirrobot-agent"

IN_IDENT="${BOT_IDENTITIES_INPUT:-}"
DETECTED="${BOT_DETECTED_LOGIN:-}"
IN_TRIG="${BOT_TRIGGERS_INPUT:-}"

# ---- identities: variable ∪ detection, fallback only when both empty -----
# An identity set is ONLY ever: what the operator EXPLICITLY declared in
# BOT_IDENTITIES (their claim of control; include an app's FULL login like
# "name[bot]" there iff you registered that app) ∪ what a credential we
# hold PROVES (the account /user login). NEVER a synthesized "[bot]" twin
# of the detected login — GitHub app slugs and usernames are separate
# namespaces, so name shape proves nothing (an attacker can own the app
# "name[bot]" while the operator owns the account "name").
# Format: comma-separated logins (a token list; logins cannot contain
# commas — the platform format doctrine for simple-token lists).
ident_note="fallback"
IDENT_RAW=""
if [ -n "$(printf '%s' "$IN_IDENT" | tr -d '[:space:],')" ]; then
  case "$IN_IDENT" in
    '['*|'{'*)
      # Guard against the retired JSON-era value shape: treated as ABSENT
      # (never parsed as one garbage token like ["Mirrobot-Agent"]).
      echo "::warning::bot-config: BOT_IDENTITIES is a comma-separated list now (was JSON). Migrate the variable; e.g. 'mybot, mybot[bot]'. Ignoring the JSON-shaped value." >&2
      ;;
    *)
      ident_tokens=$(printf '%s' "$IN_IDENT" | tr ',;' '\n\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | awk 'NF')
      if [ -n "$ident_tokens" ]; then
        while IFS= read -r item; do IDENT_RAW="${IDENT_RAW}${item}"$'\n'; done <<< "$ident_tokens"
        ident_note="variable"
      fi
      ;;
  esac
fi
if [ -n "$DETECTED" ]; then
  IDENT_RAW="${IDENT_RAW}${DETECTED}"$'\n'
  ident_note="${ident_note}+detected(${DETECTED})"
fi

# Filter FIRST, then judge emptiness: [""] or a whitespace-only variable is
# ABSENT, not an empty identity set (an empty set would silently no-op
# every loop guard). Real newlines (never printf %b: it would also
# interpret backslash escapes inside a declared identity).
IDENT_ITEMS=$(printf '%s' "$IDENT_RAW" | awk 'NF')

if [ -n "$IDENT_ITEMS" ]; then
  # unique (case-insensitive — logins are) + compact single-line JSON
  # (GITHUB_ENV values must be one line)
  BOT_NAMES_JSON=$(printf '%s' "$IDENT_ITEMS" | tr 'A-Z' 'a-z' | sort -u | jq -R . | jq -sc .)
  identity_sourced=1
  # Display forms for prompt prose (original casing kept; deduped against
  # the same lowercased set so a variable declaring "Zeta" + detected "zeta"
  # renders once). "$BOT_IDENTITY_PRIMARY" is the footer/first-mention name,
  # "$BOT_IDENTITY_LIST" the exact-match enumeration for self-checks.
  BOT_IDENTITY_LIST=$(printf '%s' "$IDENT_ITEMS" | awk 'NF' | awk '!seen[tolower($0)]++' | paste -sd, -)
  BOT_IDENTITY_PRIMARY=$(printf '%s' "$IDENT_ITEMS" | awk 'NF' | head -1)
else
  BOT_NAMES_JSON="$FALLBACK_IDENTITIES"
  BOT_IDENTITY_LIST="mirrobot-agent,mirrobot-agent[bot]"
  BOT_IDENTITY_PRIMARY="mirrobot-agent"
  identity_sourced=0
  ident_note="fallback (set BOT_IDENTITIES to override)"
fi

# ---- triggers: variable > identity-derived > fallback ---------------------
# Stock rule: when identity itself fell back (nothing set anywhere), triggers
# fall back to the mirrobot words — "triggers work as if identity is
# mirrobot". Identity-derived stems apply only when the identity came from
# the variable or account detection (so setting an identity or running
# account mode retires the fallback words, as intended).
trigger_note=""
if [ -n "$(printf '%s' "$IN_TRIG" | tr -d '[:space:],')" ]; then
  # Trim each stem (a seeded "a, b" carries a space after the comma).
  BOT_TRIGGER_STEMS=$(printf '%s' "$IN_TRIG" | tr ',' '\n' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | awk 'NF' | paste -sd, -)
  trigger_note="variable"
else
  # strip a trailing "[bot]" — a stem must be a typeable name (GitHub app
  # logins cannot be @-mentioned as text).
  stems=$(printf '%s' "$BOT_NAMES_JSON" | jq -r '.[]' | sed 's/\[bot\]$//' | awk 'NF' | sort -u | paste -sd, -)
  if [ "$identity_sourced" = 1 ]; then
    BOT_TRIGGER_STEMS="${stems:-$FALLBACK_TRIGGERS}"
    trigger_note="derived-from-identity"
  else
    BOT_TRIGGER_STEMS="$FALLBACK_TRIGGERS"
    trigger_note="fallback"
  fi
fi

# ---- export ----------------------------------------------------------------
if [ "${1:-}" = "--export" ]; then
  # Same-step consumption (eval "$(bot-config.sh --export)"): print ONLY the
  # shell-evaluable exports — nothing else may reach stdout in this mode.
  printf 'export BOT_NAMES_JSON=%q\n' "$BOT_NAMES_JSON"
  printf 'export BOT_TRIGGER_STEMS=%q\n' "$BOT_TRIGGER_STEMS"
  printf 'export BOT_IDENTITY_LIST=%q\n' "$BOT_IDENTITY_LIST"
  printf 'export BOT_IDENTITY_PRIMARY=%q\n' "$BOT_IDENTITY_PRIMARY"
else
  {
    printf 'BOT_NAMES_JSON=%s\n' "$BOT_NAMES_JSON"
    printf 'BOT_TRIGGER_STEMS=%s\n' "$BOT_TRIGGER_STEMS"
    printf 'BOT_IDENTITY_LIST=%s\n' "$BOT_IDENTITY_LIST"
    printf 'BOT_IDENTITY_PRIMARY=%s\n' "$BOT_IDENTITY_PRIMARY"
  } >> "${GITHUB_ENV:-/dev/null}" 2>/dev/null || true
fi
# Provenance log lines go to STDERR so the --export eval can never see them.
echo "bot-config: identities [$ident_note] = $BOT_NAMES_JSON" >&2
echo "bot-config: trigger stems [$trigger_note] = $BOT_TRIGGER_STEMS" >&2
