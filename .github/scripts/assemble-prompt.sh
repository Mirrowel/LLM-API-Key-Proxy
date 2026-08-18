#!/usr/bin/env bash
# assemble-prompt.sh — assemble an agent prompt from its manifest parts.
#
# Usage:
#   assemble-prompt.sh <manifest-name>     print the assembled prompt to stdout
#   assemble-prompt.sh --list              list manifests and parts
#   assemble-prompt.sh --verify            check every manifest part resolves,
#                                          no orphan parts, no duplicate headings
#
# Contract:
#   - <manifest-name> matches .github/prompts/manifests/<name>.manifest
#     (with or without the .manifest suffix); the manifest lists part names,
#     one per line; blank lines and #-comments ignored.
#   - Parts resolve relative to THIS SCRIPT'S directory (../prompts/parts),
#     so the script works from the repo root AND from /tmp copies (the
#     workflows copy parts/, manifests/ and this script together).
#   - Output: parts concatenated in manifest order. ${VARS} are left RAW —
#     the caller pipes through envsubst with its own variable list.
#   - Missing part or manifest: exit 1 naming it (fail-closed — a silently
#     short prompt must never reach an agent).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# parts/ and manifests/ sit next to this script when copied to /tmp, or under
# .github/prompts/ in the repo. Support both layouts.
if [ -d "$SCRIPT_DIR/parts" ]; then
  PARTS_DIR="$SCRIPT_DIR/parts"
  MANIFESTS_DIR="$SCRIPT_DIR/manifests"
else
  PARTS_DIR="$SCRIPT_DIR/../prompts/parts"
  MANIFESTS_DIR="$SCRIPT_DIR/../prompts/manifests"
fi

list() {
  echo "Manifests:"
  for m in "$MANIFESTS_DIR"/*.manifest; do
    [ -e "$m" ] || { echo "  (none found in $MANIFESTS_DIR)"; return 0; }
    echo "  $(basename "$m" .manifest)"
  done
  echo "Parts:"
  for p in "$PARTS_DIR"/*.md; do
    [ -e "$p" ] || { echo "  (none found in $PARTS_DIR)"; return 0; }
    echo "  $(basename "$p" .md)"
  done
}

manifest_parts() { # manifest-file -> part names, one per line
  grep -v '^\s*$' "$1" | grep -v '^\s*#' | sed 's/\r$//' | sed 's/\.md$//'
}

real_headings() { # file -> markdown headings OUTSIDE fenced code blocks
  # bash-comment examples inside ``` fences (e.g. "# Then post it:") are not
  # headings; toggle fencing on ``` lines before matching. Trailing whitespace
  # is stripped so a heading cannot evade duplicate detection by padding.
  awk '/^[[:space:]]*```/ { fence = !fence; next } !fence && /^#{1,2} / { sub(/[[:space:]]+$/, ""); print }' "$1"
}

verify() {
  FAIL=0
  used_all=""
  for m in "$MANIFESTS_DIR"/*.manifest; do
    [ -e "$m" ] || { echo "no manifests found"; exit 1; }
    name=$(basename "$m" .manifest)
    dupes=$(manifest_parts "$m" | sort | uniq -d)
    [ -z "$dupes" ] || { echo "FAIL [$name]: duplicate parts: $dupes"; FAIL=1; }
    headings=$(for pn in $(manifest_parts "$m"); do
      real_headings "$PARTS_DIR/$pn.md" 2>/dev/null || true
    done | sort | uniq -d)
    [ -z "$headings" ] || { echo "FAIL [$name]: duplicate headings:"; echo "$headings"; FAIL=1; }
    while IFS= read -r pn; do
      [ -n "$pn" ] || continue
      if [ ! -f "$PARTS_DIR/$pn.md" ]; then
        echo "FAIL [$name]: missing part: $pn"
        FAIL=1
      fi
      used_all="$used_all $pn "
    done < <(manifest_parts "$m")
  done
  for p in "$PARTS_DIR"/*.md; do
    [ -e "$p" ] || break
    pn=$(basename "$p" .md)
    case "$used_all" in
      *" $pn "*) ;;
      *) echo "WARN: orphan part (in no manifest): $pn"; ;;
    esac
  done
  [ "$FAIL" -eq 0 ] && echo "verify: OK" || exit 1
}

case "${1:-}" in
  --list) list; exit 0 ;;
  --verify) verify; exit 0 ;;
  "") echo "usage: $0 <manifest-name> | --list | --verify" >&2; exit 2 ;;
esac

NAME="${1%.manifest}"
MANIFEST="$MANIFESTS_DIR/$NAME.manifest"
if [ ! -f "$MANIFEST" ]; then
  echo "assemble-prompt: manifest not found: $NAME (looked in $MANIFESTS_DIR)" >&2
  exit 1
fi

first=1
while IFS= read -r pn; do
  [ -n "$pn" ] || continue
  P="$PARTS_DIR/$pn.md"
  if [ ! -f "$P" ]; then
    echo "assemble-prompt: part not found: $pn (required by $NAME)" >&2
    exit 1
  fi
  if [ "$first" -eq 1 ]; then first=0; else printf '\n---\n\n'; fi
  cat "$P"
done < <(manifest_parts "$MANIFEST")
