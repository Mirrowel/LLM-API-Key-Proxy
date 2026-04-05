## 2026-06-19 — Fix codex exhausted credential never cleared on quota recovery

Target: `feat(codex): Responses API rewrite, dynamic model discovery, and OAuth exports`
Files:
- `src/rotator_library/providers/codex_provider.py`
- `src/rotator_library/providers/utilities/codex_quota_tracker.py`

Working commits before autosquash:
- `cc599b6d fixup! feat(codex): ...`

Final stack commit after autosquash:
- `26977cec feat(codex): ...`

Verification:
- `uv run python3 -m py_compile src/rotator_library/providers/codex_provider.py` — passed
- `uv run python3 -m py_compile src/rotator_library/providers/utilities/codex_quota_tracker.py` — passed
- `uv run ruff check src/rotator_library/providers/codex_provider.py --select F401,F811,F821,E9` — passed
- `uv run ruff check src/rotator_library/providers/utilities/codex_quota_tracker.py --select F401,F811,F821,E9` — passed
- Hotpatched to docker-test and verified live: 28/28 credentials active, 0 exhausted, 0 cooldown

Notes:
- Bug: Codex quota tracker had zero calls to `clear_cooldown_if_exists`, while `base_quota_tracker.py` used it on every recovery. Once exhausted, credentials stayed blocked until the original `cooldown.until` timestamp expired, even if the API reported quota recovery (used_percent < 100).
- Fix: Added `clear_cooldown_if_exists` to all three paths that push quota data to UsageManager:
  - `_push_quota_to_usage_manager` (header/response path) — clears per-tier cooldowns on every API response
  - `_store_baselines_to_usage_manager` (initial fetch) — clears stale cooldowns during startup
  - `run_background_job` (periodic 300s refresh) — fetches ALL credentials, evaluates exhaustion waterfall, and clears recovered cooldowns
- Also pre-registered tier quota groups (`5h-limit`, `weekly-limit`, `monthly-limit`) in deterministic ascending window-size order for consistent UI display regardless of credential fetch order.
