# Fork tooling and workflow

Canonical feature ID: `tooling`
Stack subject: `feat(tooling): add AGENTS.md and .agent/ config for linear stack workflow`
Manifest: `.fork/stack.yml`

This file is the shared, repo-tracked history for fork workflow/tooling changes.
Local workspace state under `/opt/data/workspace/developer/state/llm-api-key-proxy/`
may contain run logs and scratch notes, but it is not canonical.

## 2026-06-19 — Add shared fork workflow metadata

Target: `feat(tooling): add AGENTS.md and .agent/ config for linear stack workflow`
Files:
- `AGENTS.md`
- `.fork/stack.yml`
- `.fork/features/tooling.md`
- `.fork/features/gemini-cli.md`
- `.fork/check-stack.py`

Working commits before autosquash:
- pending

Final stack commit after autosquash:
- pending

Verification:
- pending

Notes:
- Replaces local-only workspace ledgers as the canonical feature history with repo-tracked `.fork/` metadata.
- Adds `.fork/stack.yml` as the shared source of truth for feature IDs, stack order, file ownership, and allowed historical exceptions.
- Adds `.fork/check-stack.py` to catch duplicate release-note sections, executable `git add -A` examples, missing feature metadata, and unexpected duplicate feature commits.
