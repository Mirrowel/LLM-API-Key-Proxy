#!/usr/bin/env bash
# split-diff.sh — split an oversized diff file into navigable parts; never truncate.
#
# CONTRACT
#   split-diff.sh <diff-file> [max-bytes]
# Env in : none (args only). max-bytes defaults to 1000000.
# Out    : prints the effective diff path on stdout (ALWAYS the input path).
#           Under the threshold: file untouched, no-op.
#           Over it: the input path is REPLACED by an index file; the content
#           moves to "<path>.part01", "<path>.part02", ... alongside it.
#           Every byte of the original diff survives — nothing is truncated.
#           Cuts land on '^diff --git' section boundaries when possible; a
#           single section larger than max-bytes is split at line boundaries
#           inside that section (marked in the index).
#           Exit 0 = split or no-op; exit 1 = usage/unreadable input.
# Index  : line 1-2 are a [DIFF SPLIT ...] header marking this as an index;
#          each following line: part name, bytes, lines, file list.
# Used by: pr-review.yml, bot-reply.yml, compliance-check.yml (runner-side)
#          and generate-review-kit.sh (agent-side, sibling in its own dir).
set -euo pipefail

FILE="${1:-}"
MAX="${2:-1000000}"

if [ -z "$FILE" ] || [ ! -f "$FILE" ]; then
  echo "split-diff.sh: usage: split-diff.sh <diff-file> [max-bytes]" >&2
  exit 1
fi
case "$MAX" in *[!0-9]*) echo "split-diff.sh: max-bytes must be numeric" >&2; exit 1;; esac
[ "$MAX" -ge 1000 ] || { echo "split-diff.sh: max-bytes floor is 1000" >&2; exit 1; }

SIZE=$(wc -c < "$FILE")
LINES=$(wc -l < "$FILE")

if [ "$SIZE" -le "$MAX" ] || [ "$LINES" -eq 0 ]; then
  printf '%s\n' "$FILE"; exit 0
fi
if head -c 12 "$FILE" | grep -q '^\[DIFF SPLIT'; then
  printf '%s\n' "$FILE"; exit 0  # already an index (idempotent)
fi

DIR=$(dirname "$FILE")
BASE=$(basename "$FILE")

# One awk pass: map sections (byte offset + line + first file), greedy-pack
# them into parts <= MAX bytes, emit a part plan on stdout.
# Plan format: line 1 = part count; then "startLine endLine fileList" per part.
# An oversized section becomes its own part (flagged: file list gets
# "(OVERSIZED: line-split)" so the shell side sub-splits it).
PLAN=$(awk -v max="$MAX" -v total_bytes="$SIZE" '
  function span_flush() {
    if (s_start) { n++; st[n]=s_start; en[n]=at_end ? NR : NR-1; by[n]=boff-s_off; fl[n]=s_file }
    s_start=0
  }
  function part_push(startL, endL, files) { p++; P_s[p]=startL; P_e[p]=endL; P_f[p]=files }
  /^diff --git / {
    span_flush()
    s_start=NR; s_off=boff; s_file=$0
    sub(/^diff --git a\//, "", s_file); sub(/ b\/.*$/, "", s_file)
    if (s_file ~ /^diff --git/) s_file="(unparsed)"
  }
  { boff += length($0) + 1 }
  END {
    at_end=1
    span_flush()
    # prelude (lines before the first section) as its own tiny span
    if (n > 0 && st[1] > 1) { n++; for (i=n; i>1; i--) { st[i]=st[i-1]; en[i]=en[i-1]; by[i]=by[i-1]; fl[i]=fl[i-1] }; st[1]=1; en[1]=st[2]-1; by[1]=0; fl[1]="(prelude)" }
    p=0; c_bytes=0; c_start=0; c_end=0; c_files=""; c_nfiles=0
    for (i=1; i<=n; i++) {
      if (c_start == 0) { c_start=st[i]; c_bytes=0; c_files=""; c_nfiles=0 }
      else if (c_bytes > 0 && by[i] > max) {
        # next section alone is oversized: close current part first
        part_push(c_start, c_end, c_files); c_start=st[i]; c_bytes=0; c_files=""; c_nfiles=0
      } else if (c_bytes + by[i] > max && c_nfiles > 0) {
        part_push(c_start, c_end, c_files); c_start=st[i]; c_bytes=0; c_files=""; c_nfiles=0
      }
      c_bytes += by[i]; c_end=en[i]; c_nfiles++
      if (c_nfiles <= 8) c_files = (c_files == "") ? fl[i] : c_files "," fl[i]
      if (by[i] > max) {  # oversized section: its own part, line-split later
        part_push(c_start, c_end, c_files " (OVERSIZED: line-split)")
        c_start=0; c_bytes=0; c_files=""; c_nfiles=0
      }
    }
    if (c_start > 0 && c_start <= en[n]) part_push(c_start, c_end, c_files)
    print p
    for (i=1; i<=p; i++) print P_s[i], P_e[i], P_f[i]
  }
' "$FILE")

mapfile -t PLANL <<< "$PLAN"
NP=${PLANL[0]}

part=0
tmp_index="$DIR/.split-index.$$"
: > "$tmp_index"

emit() { # $1 start line, $2 end line, $3 file list
  part=$((part+1))
  pname=$(printf '%s.part%02d' "$BASE" "$part")
  sed -n "${1},${2}p" "$FILE" > "$DIR/$pname"
  printf '%s  %s bytes  %s lines  files: %s\n' "$pname" "$(wc -c < "$DIR/$pname")" "$(wc -l < "$DIR/$pname")" "$3" >> "$tmp_index"
}

if [ "${NP:-0}" -ge 1 ]; then
  for ((i=1; i<=NP; i++)); do
    sl=${PLANL[i]%% *}
    rest=${PLANL[i]#* }
    el=${rest%% *}
    fdesc=${rest#* }
    case "$fdesc" in
      *"(OVERSIZED: line-split)"*)
        # split this single section at line boundaries (~max/80 lines per chunk)
        chunk=$(( MAX / 80 )); [ "$chunk" -lt 200 ] && chunk=200
        cs=$sl
        while [ "$cs" -le "$el" ]; do
          ce=$(( cs + chunk - 1 )); [ "$ce" -gt "$el" ] && ce=$el
          emit "$cs" "$ce" "$fdesc (lines $cs-$ce)"
          cs=$(( ce + 1 ))
        done
        ;;
      *) emit "$sl" "$el" "$fdesc" ;;
    esac
  done
else
  # No '^diff --git' sections at all: plain line-chunk split.
  lines_per_part=$(( LINES / ((SIZE / MAX) + 1) )); [ "$lines_per_part" -lt 1 ] && lines_per_part=1
  cs=1
  while [ "$cs" -le "$LINES" ]; do
    ce=$(( cs + lines_per_part - 1 )); [ "$ce" -gt "$LINES" ] && ce=$LINES
    emit "$cs" "$ce" "(line range $cs-$ce)"
    cs=$(( ce + 1 ))
  done
fi

{
  printf '[DIFF SPLIT - the full diff is %s bytes in %d parts. THIS FILE IS ONLY THE INDEX.\n' "$SIZE" "$part"
  printf 'Read parts selectively: start with the ones touching files you care about. You never need all parts at once. Part files sit alongside this file.]\n'
  cat "$tmp_index"
} > "$FILE"
rm -f "$tmp_index"

printf '%s\n' "$FILE"
