#!/usr/bin/env bash
# live-share-announce.sh - surface the live share URL mid-run, encrypted.
#
# Background companion of the agent session step: polls the OpenShare
# urlFile; the moment the plugin creates the share (seconds into the
# session), the URL is RSA-OAEP-encrypted (MRB1 wire form - the legacy
# share-link shape decrypt_share_link.py natively opens) with
# SESSION_ARTIFACT_PUBKEY and published as a notice annotation plus a
# step-summary block. Ciphertext only: the raw URL never reaches the
# public log. The operator decrypts mid-run and watches the session
# live. One announcement per run (a run has one main share); the script
# exits right after, and the caller kills it if the session ends first.
#
# Usage (caller starts it in the background before the opencode run):
#   bash /tmp/live-share-announce.sh & ANNOUNCER=$!
#   ... opencode run ... ; kill $ANNOUNCER 2>/dev/null || true
#
# Env:
#   SESSION_ARTIFACT_PUBKEY  repo secret (absent = passive-only run,
#                            nothing to announce - exit immediately)
#   OPENSHARE_URL_FILE       the plugin's urlFile (arg 1 overrides)
# Arg 2: max wait seconds (default 3600).
#
# Output is the ::notice:: annotation ONLY (visible on the run page the
# moment it fires). The end-of-run summary step owns the single canonical
# summary block - no duplicate share blobs in the summary.
set -euo pipefail

URL_FILE="${1:-${OPENSHARE_URL_FILE:-}}"
MAX_WAIT="${2:-3600}"
PUBKEY="${SESSION_ARTIFACT_PUBKEY:-}"
[ -n "$URL_FILE" ] || exit 0
[ -n "$PUBKEY" ] || exit 0

deadline=$(( $(date +%s) + MAX_WAIT ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ -s "$URL_FILE" ]; then
    url="$(head -n1 "$URL_FILE" | jq -r 'if type == "object" then .url // empty else empty end' 2>/dev/null || true)"
    if [ -z "$url" ]; then
      # legacy bare-URL line format
      cand="$(head -n1 "$URL_FILE")"
      case "$cand" in http*) url="$cand" ;; esac
    fi
    if [ -n "$url" ]; then
      pubfile="$(mktemp)"
      printf '%s\n' "$PUBKEY" > "$pubfile"
      blob=""
      # retry loop: fresh files can be briefly unreadable (AV scanners,
      # slow network filesystems) - up to ~1.5s total
      for _ in 1 2 3 4 5; do
        blob="$(printf '%s' "$url" \
          | openssl pkeyutl -encrypt -pubin -inkey "$pubfile" \
              -pkeyopt rsa_padding_mode:oaep -pkeyopt rsa_oaep_md:sha256 \
              2>/dev/null \
          | openssl base64 -A)" || blob=""
        [ -n "$blob" ] && break
        sleep 0.3
      done
      rm -f "$pubfile"
      if [ -n "$blob" ]; then
        printf '::notice::title=Live share (encrypted - decrypt with decrypt_share_link.py)::MRB1.%s\n' "$blob"
      fi
      exit 0
    fi
  fi
  sleep 2
done
