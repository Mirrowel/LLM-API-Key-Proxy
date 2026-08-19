#!/usr/bin/env bash
# share-filter.sh - stream filter for `opencode run --share` invocations.
#
# Captures the session share link (https://opncd.ai/share/...) from the
# output stream WITHOUT letting the raw URL reach the public Actions log,
# then re-publishes it RSA-OAEP-encrypted so an admin holding the private
# key can recover it (decrypt_share_link.py in the repo root).
#
# Architecture note: the awk stage NEVER spawns subprocesses (it only
# captures/masks/writes files) - encryption happens in this bash script
# after the stream ends. Spawning openssl from inside awk is fragile on
# MSYS/Windows (argument conversion mangles "-pkeyopt name:value") and
# gains nothing: the encrypted block lands at the end of the stream,
# plus a ::notice:: annotation and the step-summary files.
#
# Usage:
#   opencode run --share - < prompt.txt | bash /tmp/share-filter.sh
#   (the calling step must set `set -o pipefail` to preserve the exit code)
#
# Env:
#   SHARE_LINK_PUBKEY   PEM public key (repo secret SHARE_LINK_PUBKEY).
#                       Optional - when unset/invalid the URL is still
#                      captured and masked, just not encrypted.
#   SHARE_CTX_THREAD    e.g. "PR #162" / "Issue #9"           (optional)
#   SHARE_CTX_HEAD      commit SHA this run reviewed           (optional)
#   SHARE_CTX_DETAIL    e.g. "FOLLOW-UP (last reviewed abc..)" (optional)
#   GITHUB_REPOSITORY / GITHUB_RUN_ID / GITHUB_ACTOR - default CI env.
#
# Outputs:
#   stdout: passthrough log; the link line is replaced by a capture notice,
#           the encrypted block is emitted at end of stream.
#   ::add-mask::<url>  prevents any later accidental re-echo
#   ::notice::         prominent annotation carrying the blob
#   $TMP/share-link.enc / $TMP/share-link.ctx  - for the step summary
set -euo pipefail

TMP="${RUNNER_TEMP:-/tmp}"
ENC_OUT="$TMP/share-link.enc"
CTX_OUT="$TMP/share-link.ctx"
URL_OUT="$TMP/share-link.url"

# ---- stage 1: stream filter (pure awk, no subprocesses) -------------------
awk -v url_out="$URL_OUT" -v ctx_out="$CTX_OUT" \
    -v repo="${GITHUB_REPOSITORY:-unknown}" -v run="${GITHUB_RUN_ID:-unknown}" \
    -v actor="${GITHUB_ACTOR:-unknown}" \
    -v thread="${SHARE_CTX_THREAD:-}" -v head="${SHARE_CTX_HEAD:-}" \
    -v detail="${SHARE_CTX_DETAIL:-}" '
BEGIN { captured = 0 }
{
  if ($0 ~ /opncd\.ai\/share\//) {
    line = $0
    esc = sprintf("%c", 27)
    gsub(esc "\\[[0-9;]*m", "", line)              # strip ANSI color codes
    if (match(line, /https:\/\/opncd\.ai\/share\/[A-Za-z0-9_-]+/)) {
      url = substr(line, RSTART, RLENGTH)
      printf "::add-mask::%s\n", url
      if (captured++) {
        print "~ [share link repeated - already captured above]"
        next
      }
      print "~ [share link captured - encrypted block at end of stream]"
      printf "%s\n", url > url_out
      ctx = "context: " repo
      if (thread != "")  ctx = ctx " | " thread
      if (head != "")    ctx = ctx " | head " head
      if (detail != "")  ctx = ctx " | " detail
      ctx = ctx " | run " run " | by " actor
      printf "%s\n", ctx > ctx_out
      next
    }
  }
  print
}
'

# ---- stage 2: encrypt (bash-side, retry-tolerant) -------------------------
if [ ! -s "$URL_OUT" ]; then
  exit 0                                            # nothing captured
fi

CTX="$(cat "$CTX_OUT" 2>/dev/null || true)"

PUBKEY_FILE=""
if [ -n "${SHARE_LINK_PUBKEY:-}" ]; then
  PUBKEY_FILE="$TMP/share-link.pub.pem"
  printf '%s\n' "$SHARE_LINK_PUBKEY" > "$PUBKEY_FILE"
fi

encrypt_url() {  # $1=url $2=pubkey-file -> base64 blob on stdout
  printf '%s' "$1" \
    | openssl pkeyutl -encrypt -pubin -inkey "$2" \
        -pkeyopt rsa_padding_mode:oaep -pkeyopt rsa_oaep_md:sha256 \
        2>/dev/null \
    | openssl base64 -A
  echo                                              # terminate for $( )
}

URL="$(cat "$URL_OUT")"
BLOB=""
if [ -n "$PUBKEY_FILE" ]; then
  # retry loop: fresh files can be briefly unreadable (AV scanners, slow
  # network filesystems) - up to ~3s total
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    BLOB="$(encrypt_url "$URL" "$PUBKEY_FILE")"
    [ -n "$BLOB" ] && break
    sleep 0.3
  done
fi

if [ -n "$BLOB" ]; then
  printf '~ [share link encrypted] MRB1.%s\n' "$BLOB"
  printf '%s\n' "$CTX"
  printf '::notice::title=Mirrobot share link (encrypted)::MRB1.%s\n' "$BLOB"
  printf 'MRB1.%s\n' "$BLOB" > "$ENC_OUT"
else
  echo "~ [share link captured - NOT encrypted: no valid SHARE_LINK_PUBKEY secret]"
  printf '%s\n' "$CTX"
  echo "::notice::title=Mirrobot share link::captured and masked; set the SHARE_LINK_PUBKEY secret to enable encryption"
  printf '(masked)\n' > "$ENC_OUT"
fi
printf '%s\n' "$CTX" > "$CTX_OUT"
rm -f "$URL_OUT"
