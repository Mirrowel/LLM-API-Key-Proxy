# Scrub Fixture Suite

The batteries. CI enforcement of every contract the rest of these docs describe — if a doc says "X always happens," there is a pin here that fails when X stops happening.

**Triggers:** `push`/`pull_request` touching `.github/scripts|prompts|actions|workflows/**`, plus manual dispatch.
**Executes from:** wherever it runs (CI context) — it tests the *checkout's* copy, which is the point on PRs.

## What runs

1. **`scrub-fixtures.sh`** (193 checks) —
   - a real fixture repository built from scratch (branches: stale, evil, evil-merge, sync-content, sync-rollback, dev, autoload variants) with the actual scrub script run against each;
   - the full taint matrix: direct modify → alarm, modify-then-revert → alarm, evil merge → alarm, stale base → explained note, platform-sync content → explained, rollback-below-fork → alarm;
   - the split-trust autoload matrix: dev-tip doctrine kept silently, dev-intermediate kept with era note, pre-fork rollback removed + quarantined, hostile skills/configs removed, trusted siblings kept, out-of-repo symlink targets never staged into quarantine;
   - graceful degradation: a bare clone with `dev` stripped simulates a deployment without a dev — notice, never fail-closed;
   - workflow contracts: pipefail, pause gates, config-lifecycle waiters, stats discipline (no `--models`), plugin wiring, drift-check scope, bootstrap state-silence, share-filter sentinel, the YAML `#`-comment trap class, the stub-dispatch tripwires;
   - routing matrix, roster transforms, permission jq-env deny patterns;
   - strict YAML (duplicate-key-rejecting loader) over every workflow and action file.
2. **`prompt-rule-fixtures.sh`** (339 pins) — every behavioral rule in the prompt parts, pinned as greps over the *assembled* prompts; assembly contracts (every part referenced by a manifest, no orphans); the envsubst variable sets per mode.
3. **`assemble-prompt.sh --verify`** — manifest integrity.

## The rule

Behavior changes carry their pin changes in the same commit. A red suite after your edit means the suite disagrees with your change — that's the system working; either your change is wrong or the pin needs updating *because you decided the behavior should differ*. Both are fine. Silently deleting a pin is not.

## When it goes red

Read the failing check's name first — they're written as sentences ("rollback below fork era -> ALARM"). The `want=`/`got=` lines below each failure show the divergence. Local run: `bash .github/scripts/scrub-fixtures.sh` (creates nothing outside mktemp + /tmp logs).

## Testing it

It *is* the test. Its own CI run is on every `.github/**` push — including this suite's own changes, so a broken checker gets caught by itself.
