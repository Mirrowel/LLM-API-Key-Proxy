#!/usr/bin/env bash
# ============================================================================
# scrub-workspace.sh — canonical agent workspace scrub
# ============================================================================
# Trust rule: agent-auto-loaded files may exist in the workspace only when
# byte-identical to a maintained branch. Anything a PR/head added or changed
# relative to the anchor is removed from the working tree. Ref-based diffs
# (what reviews read) are unaffected by working-tree removal, so scrubbed
# content stays fully visible in the review — it just stops auto-loading
# into the agent.
#
# Auto-load surface (what opencode ingests at startup / directory entry):
#   AGENTS.md, CLAUDE.md (any depth), .claude/, .opencode/,
#   opencode.json, opencode.jsonc (any depth — project configs merge
#   additively over the global config and can re-allow denied permissions,
#   define MCP servers, or pull remote instruction URLs).
# Plus: .mirrobot_files/ (workflow scratch space) — always wiped; the
# workflow diff steps regenerate its contents from scratch.
#
# Anchor selection: --anchor <branch> is honored ONLY when it is a
# maintained branch (ALLOWED_BRANCHES below — edit on the default branch
# only). Any other value — including the base of an unmaintained branch a
# PR happens to target — falls back to DEFAULT_ANCHOR. This list
# intentionally mirrors the MAINTAINED_BASE_BRANCHES job env in the agent
# workflows; both live in default-branch-controlled files only.
#
# Invocation contexts:
#   workflow step: after EVERY checkout, before any agent/opencode work.
#   in-agent:      when the bot checks out a ref it did not create, it runs
#                  `bash /tmp/scrub-workspace.sh --anchor <pr-base>` —
#                  NEVER the workspace copy under .github/scripts/, which
#                  belongs to the (possibly untrusted) checked-out tree.
#
# Fail-closed: if the anchor ref cannot be resolved, ALL auto-load files
# are removed. Removals are printed and appended to /tmp/scrub-removals.txt
# so the agent can surface them (path + reason) in its final summary.
# Exit code: 0 on success (including fail-closed scrub), 1 only when the
# workspace is not a git repository.
set -u

ALLOWED_BRANCHES="main dev"
DEFAULT_ANCHOR="main"
REMOVALS_FILE="${SCRUB_REMOVALS_FILE:-/tmp/scrub-removals.txt}"

cd "$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "::error::scrub-workspace: not a git repository."
  exit 1
}

anchor="$DEFAULT_ANCHOR"
if [ "${1:-}" = "--anchor" ] && [ -n "${2:-}" ]; then
  requested="${2:-}"
  case " $ALLOWED_BRANCHES " in
    *" $requested "*) anchor="$requested" ;;
    *)
      echo "::notice::Requested anchor '$requested' is not a maintained branch ($ALLOWED_BRANCHES); using '$anchor'."
      ;;
  esac
fi

if ! git rev-parse --verify --quiet "refs/remotes/origin/$anchor^{commit}" >/dev/null 2>&1; then
  echo "Anchor branch '$anchor' not present locally; fetching..."
  git fetch --quiet origin "$anchor:refs/remotes/origin/$anchor" 2>/dev/null || true
fi
if git rev-parse --verify --quiet "refs/remotes/origin/$anchor^{commit}" >/dev/null 2>&1; then
  ANCHOR="refs/remotes/origin/$anchor"
else
  echo "::warning::scrub-workspace: cannot resolve anchor '$anchor'; removing ALL auto-load files (fail closed)."
  ANCHOR=""
fi

removed_this_run=0
note_removal() {
  echo "scrub: removed $1 ($2)"
  echo "scrub: removed $1 ($2)" >> "$REMOVALS_FILE"
  removed_this_run=$((removed_this_run + 1))
}

# normalize_path <base-dir> <target> — POSIX-ish relative path normalization
# (no external deps; handles ., .., and absolute targets).
normalize_path() {
  local base="$1" t="$2" seg
  case "$t" in
    /*) base=""; t="${t#/}" ;;
  esac
  local stack=()
  if [ -n "$base" ] && [ "$base" != "." ]; then
    IFS='/' read -ra stack <<< "${base#/}"
  fi
  local IFS='/'
  read -ra segs <<< "$t"
  for seg in "${segs[@]}"; do
    case "$seg" in
      ""|".") ;;
      "..") [ ${#stack[@]} -gt 0 ] && unset 'stack[${#stack[@]}-1]' ;;
      *) stack+=("$seg") ;;
    esac
  done
  local IFS='/'
  printf '%s' "${stack[*]}"
}

# resolved_blob <rev> <path> — prints the CONTENT the agent would load from
# <path> at <rev>, following git-tracked symlink chains (max depth 8).
# Fails (return 1) if the chain is unresolvable (missing target, absolute or
# out-of-repo target, loops) — callers treat that as "remove".
resolved_blob() {
  local rev="$1" path="${2#./}" mode target depth=0
  while :; do
    mode=$(git ls-tree "$rev" -- "$path" 2>/dev/null | awk '{print $1}')
    [ -n "$mode" ] || return 1
    if [ "$mode" != "120000" ]; then
      git show "$rev:$path" 2>/dev/null && return 0
      return 1
    fi
    depth=$((depth+1)); [ "$depth" -gt 8 ] && return 1
    target=$(git show "$rev:$path" 2>/dev/null) || return 1
    path=$(normalize_path "$(dirname "$path")" "$target")
  done
}

# keep_if_identical <path>
# Keep the path only when it is tracked in HEAD, present in the anchor
# commit, byte-identical between anchor and HEAD, and the working tree
# matches HEAD. Otherwise remove it and record why.
# Symlinked files are compared by their RESOLVED content (git compares link
# strings, but the agent loads the target's bytes — an unchanged link with a
# mutated target must count as modified).
keep_if_identical() {
  local path="$1" reason=""
  { [ -e "$path" ] || [ -L "$path" ]; } || return 0

  if ! git ls-files --error-unmatch -- "$path" >/dev/null 2>&1; then
    reason="untracked in HEAD"
  elif [ -z "$ANCHOR" ]; then
    reason="anchor unresolvable; fail-closed"
  elif ! git cat-file -e "$ANCHOR:$path" 2>/dev/null; then
    reason="not present in $anchor (added by this head)"
  elif [ -L "$path" ]; then
    head_resolved=$(resolved_blob HEAD "${path#./}") || reason="symlink chain unresolvable at HEAD"
    if [ -z "$reason" ]; then
      anchor_resolved=$(resolved_blob "$ANCHOR" "${path#./}") || reason="symlink chain unresolvable at $anchor"
      if [ -z "$reason" ] && [ "$head_resolved" != "$anchor_resolved" ]; then
        reason="resolved target differs from $anchor"
      fi
    fi
    [ -z "$reason" ] && return 0 # resolved content identical to the maintained branch — trusted, keep
  elif ! git diff --quiet "$ANCHOR" HEAD -- "$path" 2>/dev/null; then
    reason="differs from $anchor"
  elif ! git diff --quiet HEAD -- "$path" 2>/dev/null; then
    reason="working tree differs from HEAD"
  else
    return 0 # byte-identical to the maintained branch — trusted, keep
  fi

  note_removal "$path" "$reason"
  rm -rf -- "$path"
}

# --- Auto-load files at any depth (files AND symlinks) ----------------------
# Symlinks are enumerated too: a PR can commit AGENTS.md/opencode.json as a
# symlink (git mode 120000) — opencode follows it when loading, so it must be
# compared and removed exactly like a regular file.
while IFS= read -r -d '' f; do
  keep_if_identical "$f"
done < <(find . -name .git -prune -o \( -type f -o -type l \) \
  \( -name AGENTS.md -o -name CLAUDE.md -o -name opencode.json -o -name opencode.jsonc \) \
  -print0)

# --- Auto-load directories: per-file comparison ------------------------------
# Files inside .claude/ and .opencode/ are compared individually so an added
# malicious file never causes removal of its identical maintainer-approved
# siblings. POLICY: a .claude/ or .opencode/ that is itself a SYMLINK is
# removed unconditionally — a directory link with an unchanged link string
# but mutated target contents is indistinguishable cheaply from an approved
# one, and a symlinked config dir has no legitimate use to protect. Removal
# deletes the link only, never its target. Directories left empty afterwards
# are pruned.
for d in ./.claude ./.opencode; do
  { [ -e "$d" ] || [ -L "$d" ]; } || continue
  if [ -L "$d" ]; then
    note_removal "$d" "symlinked auto-load directory (policy: always removed)"
    rm -f -- "$d"
  else
    while IFS= read -r -d '' f; do
      keep_if_identical "$f"
    done < <(find "$d" \( -type f -o -type l \) -print0)
  fi
done
find ./.claude ./.opencode -type d -empty -delete 2>/dev/null || true

# --- Workflow scratch space: always wiped -----------------------------------
if [ -e .mirrobot_files ]; then
  rm -rf -- .mirrobot_files
  echo "scrub: reset .mirrobot_files (workflow scratch space)"
  echo "scrub: reset .mirrobot_files (workflow scratch space)" >> "$REMOVALS_FILE"
fi

# --- .github taint check: detect, NEVER remove ------------------------------
# Files under .github/ are NOT auto-loaded by the agent, so they are never
# removed (the reviewer must see them in the diff). But any disagreement with
# the maintained branches is a red flag — the PR is modifying the agent
# system's own configuration. Findings go to /tmp/scrub-taint.txt (consumed by
# the workflows' trust-context line) and to the job log as a loud alarm.
TAINT_FILE="${SCRUB_TAINT_FILE:-/tmp/scrub-taint.txt}"
rm -f "$TAINT_FILE"
if [ -n "$ANCHOR" ]; then
  if ! taint=$(git diff --name-status "$ANCHOR" HEAD -- .github 2>/dev/null); then
    # Anchor resolvable but the diff itself failed: fail-closed — report as
    # tainted rather than silently clean.
    taint="U	(diff unavailable - fail-closed)"
  fi
else
  # Anchor unresolvable: fail-closed — treat every .github file as tainted.
  taint=$(git ls-files -- .github 2>/dev/null | sed 's/^/U /' || true)
  [ -n "$taint" ] && taint="$taint
(anchor unresolvable - full fail-closed listing)"
fi
if [ -n "$taint" ]; then
  {
    echo "⚠ TAINT ALERT: this tree modifies the agent system's own configuration (.github/) relative to '${anchor}':"
    printf '%s\n' "$taint" | sed 's/^/  /'
  } | tee -a "$TAINT_FILE"
  echo "scrub: TAINT - .github/ differs from ${anchor} (see ${TAINT_FILE}); NOT removed — flagging for maximum-scrutiny review."
else
  echo "scrub: .github/ clean vs ${anchor}."
fi

echo "scrub: complete (anchor: ${anchor}; removed: $removed_this_run auto-load item(s); log: $REMOVALS_FILE; taint: $TAINT_FILE)"
