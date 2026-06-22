# Umans feature ledger

## 2026-06-22 — Add Umans provider with request-based quota tracking

Target: `feat(umans): add Umans provider with request-based quota tracking`
Files:
- `src/rotator_library/providers/umans_provider.py`
- `src/rotator_library/providers/utilities/umans_quota_tracker.py`
- `tests/test_umans_quota_tracker.py`
- `.fork/stack.yml`
- `.fork/features/umans.md` (this file)

Working commits before autosquash:
- (new feature, no fixup)

Final stack commit:
- `feat(umans): add Umans provider with request-based quota tracking`

Verification:
- `uv run python3 -m py_compile src/rotator_library/providers/umans_provider.py` — passed
- `uv run python3 -m py_compile src/rotator_library/providers/utilities/umans_quota_tracker.py` — passed
- `uv run ruff check src/rotator_library/providers/umans_provider.py --select F401,F811,F821,E9` — passed
- `uv run ruff check src/rotator_library/providers/utilities/umans_quota_tracker.py --select F401,F811,F821,E9` — passed
- `uv run --no-project python3 .fork/check-stack.py` — passed
- `uv run pytest tests/test_umans_quota_tracker.py -v` — passed
- Full test suite (`uv run pytest tests/ -q`) — passed

Notes:
- Authentication uses `Authorization: Bearer` against `https://api.code.umans.ai`.
- Two plans are detected from the `/v1/usage` response:
  - `code_pro`: 200 req / 5h soft limit, 400 hard cap, 3 concurrent sessions.
  - `max`: no request limit, 4 concurrent sessions.
- `UMANS_QUOTA_LIMIT` only overrides the soft request limit for `code_pro` keys.
- Request-quota tracking is **display-only** (`apply_exhaustion=False`) until the
  burst-ceiling enforcement behavior is observed. A 429 response will still put
  the credential on cooldown via the generic error handler.
- Concurrency tracking is display-only for all plans.
- The class-level `default_max_concurrent_per_key = 3` is the safe default;
  `get_credential_concurrency_limit()` returns 4 for detected max-plan keys.
- LiteLLM has no Umans pricing, so `skip_cost_calculation = True`.

Risks / follow-ups:
- Burst ceiling behavior is not yet confirmed. Once observed, consider switching
  `apply_exhaustion=True` at the soft limit or using `UMANS_QUOTA_LIMIT` to
  target the hard cap.
- The `/v1/messages` Anthropic-compatible endpoint is intentionally left to
  the standard OpenAI-compatible path in this change.
