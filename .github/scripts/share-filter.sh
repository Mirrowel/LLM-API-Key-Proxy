#!/usr/bin/env bash
# share-filter.sh - stream filter for `opencode run` invocations.
#
# Keeps the public Actions log clean: masks/drops the model-header banner
# (the model identifier is config-derived and must not surface) and
# redacts credential-shaped content that no registered mask anticipated
# (token prefixes, config/auth JSON leaves). Also touches the boot
# sentinel so the calling workflow's cleanup waiter can delete the
# sensitive config/plugin files shortly after (config lifecycle).
#
# Architecture note: the awk stage NEVER spawns subprocesses (it only
# masks/redacts/writes the sentinel) - keep it that way. Spawning openssl
# from inside awk is fragile on MSYS/Windows (argument conversion mangles
# "-pkeyopt name:value").
#
# Live session sharing is OpenShare's business (urls land in the
# runner-temp urlFile, never in this stream); the encrypted session
# artifact is produced by export-session.py post-run. This filter is log
# hygiene only.
#
# Usage:
#   opencode run - < prompt.txt 2>&1 | bash /tmp/share-filter.sh
#   (the calling step must set `set -o pipefail` to preserve the exit code)
#
# Env:
#   GITHUB_REPOSITORY / GITHUB_RUN_ID / GITHUB_ACTOR - default CI env.
#
# Outputs:
#   stdout: passthrough log with redactions applied.
#   ::add-mask::<token>  prevents any later accidental re-echo
#   /tmp/.oc-booted      sentinel consumed by the config-cleanup waiter
set -euo pipefail

BOOT_OUT="/tmp/.oc-booted"

awk -v boot_out="$BOOT_OUT" '
BEGIN { booted = 0 }
{
  if (!booted) { printf "" > boot_out; booted = 1 }
  # OpenShare plugin boot banner - pure diagnostics in the public log.
  # Dropped here; alarm/error [openshare] lines pass through (gate
  # assurance rides on them).
  if ($0 ~ /^\[openshare\] share mode: /) { next }
  # Model-header line ("> build <middot> <model>" / "> plan ...", possibly
  # ANSI-colored): the model identifier is config-derived and must not
  # surface in public logs. Mask the token wherever it appears, drop the
  # header line itself entirely.
  if (substr($0, 1, 2) == "> ") {
    line = $0
    esc = sprintf("%c", 27)
    gsub(esc "\\[[0-9;]*m", "", line)
    sub(/[\r ]+$/, "", line)
    sep = " · "   # literal UTF-8 middle dot: byte-stable on mawk/MSYS, char-stable on gawk
    si = index(line, sep)
    if (si > 0) {
      mode = substr(line, 3, si - 3)
      model = substr(line, si + length(sep))
      if ((mode == "build" || mode == "plan") && model != "") {
        if (length(model) >= 8) {
          printf "::add-mask::%s\n", model
        }
        next
      }
    }
    line = $0
  }
  # Credential-shaped content redaction (defense in depth for the log:
  # registered masks cover known values; these SHAPE rules catch anything
  # credential-like that no mask anticipated - config/auth JSON leaves,
  # token prefixes, oauth material). awk regex, no subprocesses.
  line = $0
  changed = 0
  # (a) GitHub token family + known provider prefixes, anywhere in the line.
  while (match(line, /(gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,}|x-access-token:[A-Za-z0-9_]{16,}|cr_[A-Za-z0-9]{16,}|sk-ant-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{16,}|xai-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{16,})/)) {
    tok = substr(line, RSTART, RLENGTH)
    if (length(tok) >= 8) printf "::add-mask::%s\n", tok
    line = substr(line, 1, RSTART - 1) "[REDACTED]" substr(line, RSTART + RLENGTH)
    changed = 1
  }
  # (b) Credential key/value pairs from echoed JSON: "apiKey": "cr_...",
  # "key": "...", "refresh"/"access"/"token"/"secret"/"password": "...".
  # Value must be >= 12 chars of token-ish material to avoid prose false
  # positives.
  while (match(line, /"[Aa]pi[_-]?[Kk]ey"|[Tt]oken"|[Ss]ecret"|[Pp]assword"|[Aa]uthorization"|[Cc]ookie"|[Rr]efresh"|[Cc]redential"/)) {
    kstart = RSTART; kend = RSTART + RLENGTH
    tail = substr(line, kend)
    if (match(tail, /^[ \t]*:[ \t]*"[^"]{12,}"/)) {
      val = substr(tail, RSTART, RLENGTH)
      gsub(/^.*"|"$/, "", val)
      if (length(val) >= 8) printf "::add-mask::%s\n", val
      line = substr(line, 1, kend - 1) substr(line, kend, 1) "[REDACTED]" substr(tail, RSTART + RLENGTH)
      changed = 1
    } else break
  }
  if (changed) { print line; next }
  print
}
'
