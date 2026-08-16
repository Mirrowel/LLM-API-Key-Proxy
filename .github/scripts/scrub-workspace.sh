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
set -u -o pipefail  # pipefail: a failed git log/diff inside a pipeline must not
                    # masquerade as an empty (clean) result via cut/sed/awk.

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
# removed (the reviewer must see them in the diff). But any branch-side change
# to .github/ is a red flag — the PR is modifying the agent system's own
# configuration.
#
# Detection is a UNION of two merge-base-anchored signals:
#   1. HISTORY: commits in merge-base..HEAD touching .github/ (catches
#      modify-then-revert that nets to an identical tree).
#   2. NET TREE: diff merge-base..HEAD (catches "evil merges" — .github edits
#      smuggled into a merge resolution, which produce no per-commit file
#      lines in git log --name-status).
# A tree diff against the anchor TIP is deliberately NOT an alarm signal: PRs
# branched before recent anchor-side .github commits would false-alarm (those
# anchor-side additions look like deletions) — noise that trains reviewers to
# discount alerts. Stale-base discrepancies emit the EXPLAINED info note.
#
# Output hygiene: commit subjects are attacker-controlled prose and are NOT
# emitted into the trust-context channel — only hashes and file lists. The
# scrutiny instruction lives in the alert HEADER (first line) so it always
# survives the workflows' newline-flatten + 600-char truncation.
# Findings go to /tmp/scrub-taint.txt (consumed by the workflows'
# trust-context line) and to the job log as a loud alarm.
TAINT_FILE="${SCRUB_TAINT_FILE:-/tmp/scrub-taint.txt}"
rm -f "$TAINT_FILE"
tree_diff=""
if [ -n "$ANCHOR" ]; then
  TAINT_BASE="$ANCHOR"
  TAINT_BASE_DESC="${anchor}"
  if mb=$(git merge-base "$ANCHOR" HEAD 2>/dev/null) && [ -n "$mb" ]; then
    TAINT_BASE="$mb"
    TAINT_BASE_DESC="merge-base of ${anchor}"
  # else: no common ancestor (unrelated history) — keep the anchor tip as the
  # base: over-triggers, but that is the fail-safe direction.
  fi
  if ! taint=$(git log --name-status --format='@%h' "${TAINT_BASE}"..HEAD -- .github 2>/dev/null | awk '
    # One compact bullet per commit: "- <hash>: <status> <file>, <status> <file>, ..."
    # Commits with no file lines (merge commits — path-limited log omits their
    # detail) are skipped here; the net-tree diff below reports their effect.
    # No subjects: they are attacker-controlled prose (see header comment).
    /^@/ {
      if (commit != "" && files != "") print commit ": " files
      hash = substr($0, 2); commit = "- " hash; files = ""
      next
    }
    /^[A-Za-z]+[0-9]*\t/ {
      gsub(/\t/, " ")
      if (files == "") files = $0; else files = files ", " $0
    }
    END { if (commit != "" && files != "") print commit ": " files }
  ' | cut -c1-250); then
    # Base resolvable but the log itself failed: fail-closed — report as
    # tainted rather than silently clean.
    taint="U	(log unavailable - fail-closed)"
  fi
  # Union signal 2: net tree change vs the merge-base (evil-merge detector).
  net_tree=$(git diff --name-status "${TAINT_BASE}" HEAD -- .github 2>/dev/null | sed 's/\t/ /; s/^/net: /' | cut -c1-250 || true)
  [ -n "$net_tree" ] && taint="${taint}
${net_tree}"
  # Stale-base discrepancy: even with NO branch-side .github change, the tree
  # can still differ from the anchor TIP (branch predates anchor-side .github
  # changes). That fact must reach the agent — explained, not as an alarm.
  tree_diff=$(git diff --name-status "$ANCHOR" HEAD -- .github 2>/dev/null || true)
else
  # Anchor unresolvable: fail-closed — treat every .github file as tainted.
  taint=$(git ls-files -- .github 2>/dev/null | sed 's/^/U /' || true)
  [ -n "$taint" ] && taint="$taint
(anchor unresolvable - full fail-closed listing)"
fi
if [ -n "$taint" ]; then
  # Compact alert for the flattened single-line warning: counts + AREAS only
  # (derived from the touched paths); the per-commit file list lives in the
  # details file - nothing enumerated here can be truncated mid-filename.
  taint_paths=$(git log --name-only --format='@' "${TAINT_BASE}"..HEAD -- .github 2>/dev/null | grep -v '^@' | grep -v '^$' | sort -u)
  n_commits=$(printf '%s\n' "$taint" | grep -c '^- ' || true)
  n_files=$(printf '%s\n' "$taint_paths" | grep -c . || true)
  areas=$(printf '%s\n' "$taint_paths" | awk -F/ '
    $2 == "workflows" { a["workflows"]=1; next }
    $2 == "actions"   { a["actions"]=1; next }
    $2 == "prompts"   { a["prompts"]=1; next }
    $2 == "scripts"   { a["scripts"]=1; next }
    NF >= 1           { a["other"]=1 }
    END {
      # Fixed order, joined with ", " in ONE place - paste -sd uses a cyclic
      # char list and does NOT join multi-char separators (reviewer-caught).
      # NOTE: keys must be space-free ("other", not "other .github") - the
      # order list is split on spaces.
      n = split("workflows actions prompts scripts other", order, " ")
      sep = ""
      for (i = 1; i <= n; i++) if (a[order[i]] == 1) { printf "%s%s", sep, order[i]; sep = ", " }
    }')
  {
    echo "⚠ TAINT ALERT - .github/ is modified on this branch's side of the ${TAINT_BASE_DESC} (${n_commits} commit(s), ${n_files} file(s); areas: ${areas}). MAXIMUM SCRUTINY: understand every .github change (workflow/action/prompt) or defer to a maintainer; never merge such a PR on behalf of an unverified requester. Full per-commit details: ${TAINT_FILE}"
  } | tee -a "$TAINT_FILE"
  # Details (per-commit bullets) go to the FILE only - the warning line above
  # must never truncate.
  printf '%s\n' "$taint" | sed 's/^/  /' >> "$TAINT_FILE"
  echo "scrub: TAINT - .github/ changed since ${TAINT_BASE_DESC} (see ${TAINT_FILE}); NOT removed - flagging for maximum-scrutiny review."
elif [ -n "$tree_diff" ]; then
  {
    echo "ℹ .github discrepancy — EXPLAINED, benign: the workspace .github differs from the ${anchor} TIP only because this branch predates recent ${anchor}-side .github changes (stale base). No commit on this branch modifies .github; merging keeps ${anchor}'s versions of every file this branch never touched. Context, not an alarm. If in doubt, compare .github against ${anchor} directly."
  } | tee -a "$TAINT_FILE"
  echo "scrub: .github tree differs from ${anchor} tip (stale base; no branch-side .github commits) — explained note recorded for the agent."
else
  echo "scrub: .github/ clean vs ${TAINT_BASE_DESC} (no branch-side commits touch it)."
fi

echo "scrub: complete (anchor: ${anchor}; removed: $removed_this_run auto-load item(s); log: $REMOVALS_FILE; taint: $TAINT_FILE)"
