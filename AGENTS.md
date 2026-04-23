---
description:
alwaysApply: true
---

# LLM-API-Key-Proxy — Agent Instructions

## ⚠️ MANDATORY: Read Before Any Code Change

This repository is a **fork** maintained as a linear commit stack on top of `upstream/dev`.
**You MUST follow the workflow below for every change you make, no exceptions.**

---

## How the Fork Works

```
upstream/dev
  ├── feat(anthropic): ...        ← one clean commit per feature area
  ├── feat(chutes): ...
  ├── feat(codex): ...
  ├── ... (15 more) ...
  └── feat: add health endpoints  ← HEAD (dev)
```

- `dev` is a **linear stack** of squashed, self-contained commits on `upstream/dev`
- Each commit has a **topic prefix**: `feat(codex):`, `fix(core):`, `feat(tui):`, etc.
- There are **no merge commits** — the history is always flat and linear
- Per-feature change history is tracked in repo-tracked `.fork/` metadata so
  every contributor and developer workspace sees the same ledger (see
  **Feature Tracking Ledger** below)

### Release Notes

The automated build workflow (`build.yml`) generates release changelogs from
commit messages. It works by comparing topic prefixes between builds — each
topic prefix is treated as a stable feature identifier.

- **New topics** appear in the "What's New" section of the release
- **Renamed topics** show as both "removed" (old name) and "new" (new name) — avoid unless intentional
- **Upstream syncs** are detected and reported when `upstream/dev` advances

### Feature Tracking Ledger

Because `dev` is rewritten with autosquash, git history on `dev` only shows the
current clean feature stack. It does **not** preserve the incremental commits,
rationale, verification notes, or deployment observations that happened while a
feature evolved.

To preserve that history across contributors and developer workspaces, the
canonical feature ledger is committed to this repository under:

```text
.fork/
  stack.yml                  # shared feature IDs, stack order, file ownership
  check-stack.py             # validation before autosquash/push
  features/
    <feature-key>.md         # append-only shared feature history
```

Local workspace state under paths such as
`/opt/data/workspace/developer/state/llm-api-key-proxy/` is useful for scratch
notes, run logs, reviews, and temporary artifacts, but it is **not canonical**.
Do not rely on local state as the only record of a durable feature change.

`<feature-key>` should match the stable topic prefix when possible:

| Commit Prefix | Feature Key |
|---------------|-------------|
| `feat(xai):` | `xai` |
| `feat(codex):` | `codex` |
| `feat(webui):` | `webui` |
| `feat(core):` | `core` |
| `feat(model-routing):` | `model-routing` |
| `feat(tests):` | `tests` |

Each `.fork/features/<feature-key>.md` entry should record:

- date and short title
- target commit/topic prefix
- files changed
- temporary/fixup commit hashes, if any existed before autosquash
- final rewritten stack commit hash after autosquash, if known
- verification commands and outcomes
- rationale, compatibility notes, and follow-up risks

Example:

```markdown
## 2026-06-18 — Add xAI device-code OAuth controls

Target: `feat(xai): add xAI Grok OAuth provider with PKCE and Device Code flows`
Files:
- `src/llm_api_key_proxy/providers/xai_provider.py`
- `webui/src/...`

Working commits before autosquash:
- `abc1234 fixup! feat(xai): ...`

Final stack commit after autosquash:
- `02f7470 feat(xai): ...`

Verification:
- `uv run python3 -m py_compile ...` — passed
- `cd webui && npm run build` — passed

Notes:
- Preserves `% used + reset` quota display for undocumented xAI units.
```

If the feature file does not exist yet, create it before pushing the code
change. Keep local state for bulky logs and review artifacts; summarize the
durable outcome in `.fork/features/<feature-key>.md`.

#### Feature Registry

`.fork/stack.yml` is the machine-readable source of truth for feature ownership,
stack order, allowed historical exceptions, and file ownership. It replaces the
old local-only `feature-registry.yml` idea.

When `.fork/stack.yml` and the manual table below disagree, stop and reconcile
them before editing. Do not silently choose one.
---

## Making a Change

### Step 1: Identify which commit owns the files you're changing

```bash
git log --oneline upstream/dev..HEAD
```

Match files to commits:

| File Pattern | Owning Commit Prefix |
|-------------|---------------------|
| `providers/<name>_provider.py` | `feat(<name>):` |
| `providers/utilities/<name>_*` | `feat(<name>):` |
| `providers/copilot_*` | `feat(copilot):` |
| `client/rotating_client.py` | `feat(core):` |
| `client/executor.py`, `streaming.py`, `errors.py` | `feat(core):` |
| `client/transforms.py` | `feat(core):` |
| `proxy_app/main.py` | `feat(core):` |
| `proxy_app/quota_viewer.py` | `feat(tui):` |
| `proxy_app/log_viewer.py` | `feat(tui):` |
| `model_alias_registry.py`, `cross_provider_executor.py` | `feat(model-routing):` |
| `error_handler.py`, `error_tracker.py` | `feat(core):` |
| `credential_manager.py`, `credential_tool.py` | `feat(core):` |
| `tests/*` | `feat: add local test suite` |

### Step 2: Lint all changed Python files before staging

**MANDATORY — do not skip this step.** Run the following on every `.py` file you touched:

```bash
# Syntax check (stdlib — zero deps)
uv run python3 -m py_compile src/path/to/file.py

# Undefined names / missing imports / unused imports
uv run ruff check src/path/to/file.py --select F401,F811,F821,E9
```

> This project uses `uv` for environment management. Always prefix `python3` and
> `ruff` commands with `uv run` rather than relying on system-level installations.

The pre-commit hook (`.git/hooks/pre-commit`) also runs these automatically when
you `git commit`, but running them manually first gives faster feedback.

Common things to verify after a change:
- Every name used in the file is either defined locally or imported.
- No import statements were accidentally deleted while editing.
- `py_compile` exits 0.


### Step 3: Commit with the `fixup!` prefix

Stage only the files that belong to the change. Do **not** use `git add -A` in
this repository: `worktrees/` is intentionally untracked for local git worktrees,
and `.dev` symlinks or other workspace artifacts may also exist locally.

```bash
# Edit files...
git add src/path/to/file.py tests/path/to/test_file.py
# or, for documentation-only changes:
# git add AGENTS.md
git commit -m "fixup! feat(codex): Responses API rewrite, dynamic model discovery, and OAuth exports"
```

> **CRITICAL:** The text after `fixup!` must **exactly match** the first line of the
> target commit. Copy it from `git log --oneline`.

### Step 4: Update the feature ledger

After creating the fixup commit, but before autosquashing or pushing, update the
repo-tracked per-feature ledger for the owning feature under:

```text
.fork/features/<feature-key>.md
```

For small documentation-only changes, the ledger entry may be brief. For code,
behavior, release, quota, auth, provider, or WebUI changes, include the files
changed, verification commands, and the temporary/fixup commit hash that will
disappear after autosquash.

If this is a new feature area:

1. Add the feature to `.fork/stack.yml` with its stable ID, commit subject,
   stack order, and file ownership globs.
2. Create `.fork/features/<feature-key>.md` with the shared change history.
3. Keep bulky logs/reviews in local workspace state if useful, but summarize the
   durable outcome in `.fork/features/<feature-key>.md`.

Stage the `.fork/` metadata with the code/docs change so other contributors see
it after the rewritten `dev` branch is pushed.

### Step 5: Fold it into the correct commit

```bash
GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash upstream/dev
```

This automatically moves your fixup commit next to its target and squashes them.

### Step 6: Verify the rebased stack

After autosquash/rebase, rerun the checks that cover the changed files before
pushing the rewritten branch. At minimum, rerun the targeted Python checks from
Step 2 for touched Python files. If the change touches WebUI or release tooling,
also run the relevant build/test command for that area.

Examples:

```bash
uv run python3 -m py_compile src/path/to/file.py
uv run ruff check src/path/to/file.py --select F401,F811,F821,E9
cd webui && npm run build
```

Record the post-rebase verification outcome in the relevant
`.fork/features/<feature-key>.md` entry. Then run the shared stack validator:

```bash
uv run python .fork/check-stack.py
```

### Step 7: Record final stack hash and push

After verification passes, record the final rewritten commit hash in the relevant
`.fork/features/<feature-key>.md` entry if it is known:

```bash
git log --oneline upstream/dev..HEAD --grep='feat(xai)'
```

Then push:

```bash
git push origin dev --force-with-lease
```

---

## Adding an Entirely New Feature

```bash
# Just commit at the tip with a new prefix:
git add src/path/to/new_feature.py tests/path/to/test_new_feature.py
git commit -m "feat(newprovider): add SomeProvider with quota tracking"

# Update the new feature's shared ledger before pushing
$EDITOR .fork/features/<feature-key>.md

# Verify the committed stack before pushing
uv run python3 -m py_compile src/path/to/new_feature.py
uv run ruff check src/path/to/new_feature.py --select F401,F811,F821,E9
uv run python .fork/check-stack.py

# Record the new stack commit hash in the ledger
git log --oneline upstream/dev..HEAD --grep='feat(newprovider)'

# Push
git push origin dev --force-with-lease
```

No fixup needed — new features go at the end of the stack naturally, but the
feature ledger is still required before pushing.

---

## Upstream Sync

When the upstream repository updates:

```bash
git fetch upstream
git rebase upstream/dev
# Resolve any conflicts in the specific commit that breaks
git push origin dev --force-with-lease
```

Each commit is replayed one at a time. Conflicts are localized to the specific
commit that touched the affected lines — resolve it there and continue.

---

## Rules

1. **NEVER add raw commits** without a topic prefix. Every commit must be
   `feat(<area>):`, `fix(<area>):`, or `fixup! <exact target commit message>`.

2. **NEVER merge branches into dev.** Dev is a linear rebase-only branch.

3. **Always use `--force-with-lease`** when pushing dev (it's a rewritten branch).

4. **One commit per feature area.** If you're fixing something in an existing
   area, use `fixup!` + autosquash to fold it back in.

5. **Keep the stack ordered.** Independent providers come first, shared
   infrastructure (`core`) in the middle, cross-cutting features (`tui`,
   `model-routing`, `copilot`) at the end.

6. **When a rebase conflict occurs during autosquash**, stop and resolve it
   carefully. You can always compare with the current file content using
   `git stash` to save your work and inspect.

7. **Always lint Python files before committing.** Run `uv run python3 -m py_compile
   <file>` and `uv run ruff check <file> --select F401,F811,F821,E9` on every file
   you changed. The pre-commit hook enforces this automatically, but treat it
   as a manual checklist item too — catching errors before `git add` is faster
   than fixing a broken deployment.

8. **Keep topic prefixes stable.** The automated release changelog uses commit
   messages as feature identifiers. Renaming a topic prefix (e.g.
   `feat(codex):` → `feat(openai-codex):`) causes the release notes to show
   both a "removed" entry and a "new" entry. If a rename is intentional, do it
   in a single rebase so the changelog shows both sides cleanly.

9. **Update the repo-tracked feature ledger for every durable change.** Because
   autosquash rewrites away incremental commits, `.fork/features/<feature>.md`
   is the durable shared history of how a feature evolved. Do not autosquash or
   push without recording the change there.

10. **Treat local workspace state as non-canonical.** Local state directories are
   useful for bulky logs, reviews, and scratch notes, but `.fork/stack.yml` and
   `.fork/features/*.md` are the shared records that must travel with the repo.

---

## Quick Reference

```bash
# See the full fork stack
git log --oneline upstream/dev..HEAD

# Find which commit owns a file
git log --oneline upstream/dev..HEAD -- path/to/file.py

# Lint changed Python files (run BEFORE git add)
uv run python3 -m py_compile src/path/to/file.py
uv run ruff check src/path/to/file.py --select F401,F811,F821,E9

# Stage only files that belong to this change, then make a fixup commit
git add src/path/to/file.py tests/path/to/test_file.py
git commit -m "fixup! <exact commit message from git log>"

# Update shared feature ledger before autosquash/push
$EDITOR .fork/features/<feature-key>.md

# Fold it in
GIT_SEQUENCE_EDITOR=: git rebase -i --autosquash upstream/dev

# Sync with upstream before recording final hashes
git fetch upstream && git rebase upstream/dev

# Verify the rebased stack before pushing
uv run python3 -m py_compile src/path/to/file.py
uv run ruff check src/path/to/file.py --select F401,F811,F821,E9
uv run python .fork/check-stack.py

# Record final rewritten stack hash in the ledger
git log --oneline upstream/dev..HEAD --grep='<feature-prefix>'

# Push
git push origin dev --force-with-lease
```

## Additional References

- **Local Docker testing** (container info, hot-patching, remote folder structure): `.private/README.md`
