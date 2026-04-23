# Gemini CLI provider

Canonical feature ID: `gemini-cli`
Stack subject: `feat(gemini-cli): expose gemini-3.5-flash, unify quota groups, fix token counts`
Manifest: `.fork/stack.yml`

This file is the shared, repo-tracked history for gemini-cli feature changes.
Local workspace state may contain run logs and scratch notes, but this file is
canonical across contributors and developer workspaces.

## 2026-06-19 — Fold quota display and tier fixes into gemini-cli feature commit

Target: `feat(gemini-cli): expose gemini-3.5-flash, unify quota groups, fix token counts`
Files:
- `src/proxy_app/api/config.py`
- `src/rotator_library/providers/gemini_auth_base.py`
- `src/rotator_library/providers/utilities/base_quota_tracker.py`
- `src/rotator_library/providers/utilities/gemini_cli_quota_tracker.py`
- `src/rotator_library/providers/utilities/gemini_credential_manager.py`
- `src/rotator_library/providers/utilities/gemini_shared_utils.py`
- `src/rotator_library/usage/manager.py`

Working commits before autosquash:
- `6412368 fix(gemini-cli): stabilize quota group display order and credential tier badges`
- `b541acc fix(gemini-cli): correct quota limits per upstream tier documentation`
- `869e4f3 fix(gemini-cli): preserve raw API tier names instead of normalizing on persist`

Final stack commit after autosquash:
- `a5f0170 feat(gemini-cli): expose gemini-3.5-flash, unify quota groups, fix token counts`

Verification:
- `uv run python3 -m py_compile src/proxy_app/api/config.py src/rotator_library/providers/gemini_auth_base.py src/rotator_library/providers/utilities/base_quota_tracker.py src/rotator_library/providers/utilities/gemini_cli_quota_tracker.py src/rotator_library/providers/utilities/gemini_credential_manager.py src/rotator_library/providers/utilities/gemini_shared_utils.py src/rotator_library/usage/manager.py` — passed.
- `uv run ruff check <same files> --select F401,F811,F821,E9` — passed.

Notes:
- The three standalone `fix(gemini-cli)` commits were folded into the owning feature commit so the `dev` stack preserves one clean feature commit for the main gemini-cli feature area.
- The earlier historical `fix(gemini-cli): fast-fail on non-rotatable errors and pro quota handling` remains documented as an explicit stack exception in `.fork/stack.yml`.
