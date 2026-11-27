#!/usr/bin/env bash
# ============================================================================
# render-labels.sh — the labels the agent may apply, as one rendered block
# (LABEL_PALETTE) for the prompt. Two inputs:
#
#   .github/labels.txt    the committed palette: ONE source for what Agent
#                         Bootstrap seeds, and for how each label is
#                         classified (who applies it, which group it is in)
#   live-labels.json      `gh label list --json name,description,color`:
#                         what the repository actually has right now
#
# With both, the rendered block lists exactly the repository's live labels,
# grouped by the palette (labels the palette does not know are listed as
# "other"). The agent never lists labels itself and can never see a label
# that does not exist. Without a live file the palette is rendered as is.
#
# Prints the block to stdout. On a runner (GITHUB_ENV set) it exports
# LABEL_PALETTE instead. No palette file = labeling is off (a placeholder
# is rendered); a malformed palette or live file fails the run loudly.
#
#   usage: render-labels.sh [palette-file] [live-labels.json]
# ============================================================================
set -euo pipefail

FILE="${1:-.github/labels.txt}"
LIVE="${2:-}"
OFF="(this repository has no label palette: do not label anything)"

group_labels() { # haslive palette.tsv [live.txt] -> the rendered block
  awk -F'|' -v haslive="$1" '
    BEGIN {
      n = split("kind flag severity value risk ai waiting resolution decision contribution automation", order, " ")
      title["kind"] = "what it is"
      title["flag"] = "flags"
      title["severity"] = "severity (how bad)"
      title["value"] = "value (how good if done)"
      title["risk"] = "risk (of doing it)"
      title["ai"] = "your own conclusions"
      title["waiting"] = "waiting on"
      title["resolution"] = "fixed, or not a bug"
      p = 0; m = 0; o = 0; nlive = 0
    }
    function append(list, name) { return list == "" ? name : list ", " name }
    FNR == NR {
      p++
      if ($1 == "maintainer") maint[++m] = $3
      else cls[$2, ++cnt[$2]] = $3
      known[$3] = 1
      next
    }
    $0 == "" { next }
    {
      nlive++
      islive[$0] = 1
      if (!($0 in known)) other[++o] = $0
    }
    END {
      if (haslive) print "Labels this repository has right now (nothing else exists; never create one):"
      else print "Palette labels (not checked against the repository):"
      shown = 0
      for (i = 1; i <= n; i++) {
        c = order[i]
        list = ""
        for (j = 1; j <= cnt[c]; j++) {
          name = cls[c, j]
          if (!haslive || (name in islive)) list = append(list, name)
        }
        if (list != "") { print "- " (title[c] == "" ? c : title[c]) ": " list; shown++ }
      }
      if (o > 0) {
        list = ""
        for (j = 1; j <= o && j <= 40; j++) list = append(list, other[j])
        more = o > 40 ? " ... (" o - 40 " more, list them with: gh label list --limit 500 --json name)" : ""
        print "- labels of this repository that no group above describes (use one only when it clearly carries a grade or a conclusion): " list more
        shown++
      }
      list = ""
      for (j = 1; j <= m; j++) {
        name = maint[j]
        if (!haslive || (name in islive)) list = append(list, name)
      }
      if (list != "") print "Maintainer-only (never apply or remove): " list
      if (haslive && shown == 0) print "(this repository has no labels)"
      if (haslive && nlive >= 200) print "(the repository has 200 labels or more; this list stops at 200)"
    }
  ' "${@:2}"
}

palette_tsv() { # palette file -> who|class|name per label; fails on a malformed line
  tr -d '\r' < "$1" | awk -F'|' '
    BEGIN {
      n = split("kind flag severity value risk ai waiting resolution decision contribution automation", order, " ")
      for (i = 1; i <= n; i++) valid[order[i]] = 1
      bad = 0
    }
    function problem(line, why) {
      printf "labels: %s: %s\n", why, line > "/dev/stderr"
      bad = 1
    }
    /^[ \t]*#/ || /^[ \t]*$/ { next }
    {
      if (NF != 5) { problem($0, "want name|color|class|who|description"); next }
      if ($1 ~ /^[ \t]/ || $1 ~ /[ \t]$/) { problem($0, "name has leading or trailing whitespace"); next }
      if (seen[$1]++) { problem($0, "duplicate name"); next }
      if ($2 !~ /^[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]$/) { problem($0, "color must be six hex digits"); next }
      if (!($3 in valid)) { problem($0, "unknown class"); next }
      if ($4 != "agent" && $4 != "maintainer") { problem($0, "who must be agent or maintainer"); next }
      is_m = ($3 == "decision" || $3 == "contribution" || $3 == "automation")
      if (is_m != ($4 == "maintainer")) { problem($0, "only decision, contribution and automation labels belong to the maintainer"); next }
      if ($5 == "") { problem($0, "empty description"); next }
      if (length($5) > 100) { problem($0, "description over 100 characters"); next }
      print $4 "|" $3 "|" $1
    }
    END { exit bad }
  '
}

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [ ! -f "$FILE" ]; then
  text="$OFF"
else
  palette_tsv "$FILE" > "$tmp/palette.tsv" || exit 1
  if [ ! -s "$tmp/palette.tsv" ]; then
    text="$OFF"
  elif [ -n "$LIVE" ]; then
    # Label names are repository metadata reaching a prompt: control
    # characters never make it through.
    jq -r '.[] | .name' "$LIVE" | tr -d '\r' | tr -d '\000-\010\013\014\016-\037' | LC_ALL=C sort -u > "$tmp/live.txt"
    text="$(group_labels 1 "$tmp/palette.tsv" "$tmp/live.txt")"
  else
    text="$(group_labels 0 "$tmp/palette.tsv")"
  fi
fi

if [ -n "${GITHUB_ENV:-}" ]; then
  # A multi-line value needs the delimiter form. The delimiter is random so
  # no label name can ever terminate the value early.
  delim="LABEL_PALETTE_EOF_$(od -An -N6 -tx1 /dev/urandom | tr -d ' \n')"
  {
    echo "LABEL_PALETTE<<$delim"
    printf '%s\n' "$text"
    echo "$delim"
  } >> "$GITHUB_ENV"
else
  printf '%s\n' "$text"
fi
