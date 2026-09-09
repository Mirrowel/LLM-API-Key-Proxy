#!/usr/bin/env bash
# ============================================================================
# opencode-cleanup.sh — config/plugin lifecycle cleanup (shared trusted
# artifact; the workflows run the /tmp copy, never a workspace copy).
# ============================================================================
# OpenCode reads its config ONCE at boot (empirically verified: sessions
# complete correctly with the file deleted mid-run). The share filter drops
# the /tmp/.oc-booted sentinel on the first output line, proving boot
# finished — so:
#
#   waiter   (run in background BEFORE the agent session):
#            wait for the sentinel (max 180s), then give opencode 2s to
#            finish the config read, then delete config + plugins. This is
#            the primary deletion path: nothing sensitive remains on disk
#            for the rest of the run.
#   now      (run inline after the session exits, and from the if:always()
#            post-run step): immediate belt for fast-exit/failure paths.
#
# Env: none. Exit: 0 always (cleanup must never fail a run).
set -u

delete_sensitive() {
  rm -f /tmp/.oc-booted "$HOME/.config/opencode/opencode.json" 2>/dev/null || true
  rm -rf "$HOME/.mirrobot-plugins" 2>/dev/null || true
}

case "${1:-}" in
  waiter)
    for _ in $(seq 1 180); do
      [ -f /tmp/.oc-booted ] && break
      sleep 1
    done
    sleep 2   # config read completes at boot; give it a moment
    delete_sensitive
    ;;
  now)
    delete_sensitive
    ;;
  *)
    echo "usage: opencode-cleanup.sh waiter|now" >&2
    exit 1
    ;;
esac
exit 0
