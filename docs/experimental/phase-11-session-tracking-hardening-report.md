# Phase 11: Session Tracking And Compaction Hardening Report

## Result

Phase 11 is complete. The captured false-positive families are now rejected, real compaction requires structural replacement of most known history, minority/middle-only replacement remains ordinary continuity, and optional persistence has a strict restart-safe schema and substantially broader failure/concurrency coverage.

Implementation and report changes remain uncommitted.

## Incident Findings

- The July Mistral report was caused by a downstream location-classifier prompt quoting one freshly generated response, not by the identical client regeneration immediately before it.
- The July DeepSeek events had the same one-response classifier shape.
- The June 12 DeepSeek events were ordinary growing histories whose early assistant messages were treated as size-only probes.
- The June 25 events combined ordinary history, a context contraction without summary replacement, exact resends, and follow-ups. They repeatedly produced descendants from one parent.
- The tracker logged 21 compacted-descendant events across these families.
- Persistence was disabled in the captured run and did not cause the incidents.

## Compaction Decision Contract

- Only early user/system/developer messages can be compaction probes.
- Assistant, tool, and function-result history cannot become size-only probes.
- A marker is evidence, not a decision; matched parent evidence is mandatory.
- Each session tracks a content-free high-water request-history profile.
- Retaining at least half of the parent high-water history is ordinary continuity.
- Replacing more than half is structural compaction only when parent evidence is sufficient.
- Unmarked compaction requires overlap with at least two distinct completed response events.
- Duplicate content from one response remains one response event.
- One quoted response cannot create compaction lineage or ordinary sticky response bridging from a user/system/developer role.
- Exact replay of a validated compacted payload reuses the existing child.
- Changing non-probe history changes the replay key and does not reuse the old child.
- Strong trusted, provider, or tool identity can keep a compacted request on the existing live session.
- All evidence remains isolated by usage scope, provider, and model/session scope.

## Tracking And Persistence Hardening

- Successful responses receive deterministic response-event provenance.
- Response provenance survives when assistant content returns in later request history.
- Shared content anchors retain their first live owner instead of being stolen by an auxiliary session.
- Per-session/global trimming and TTL pruning maintain bidirectional session-anchor ownership.
- Late responses cannot resurrect an expired session.
- Weak/ordinary evidence is evicted before strong replay/tool identity.
- Persistence schema 2 stores hashed high-water history, scoped anchor metadata, response-event groups, and replay bindings without raw content.
- Loading rejects malformed containers, invalid/non-finite timestamps, unsupported schemas, expired sessions, orphan anchors, namespace mismatches, invalid strengths, and malformed history signatures.
- Session anchor sets are rebuilt from validated anchor records instead of trusting duplicated serialized ownership.
- Loaded state enforces both per-session and global caps without orphaning either side.
- Failed and exceptional writes retain dirty state for retry.
- Stale delayed generations cannot overwrite a newer persisted snapshot.
- Disk writes are serialized outside the main tracker lock.

## Streaming Evidence Hardening

- Streaming response identity is recorded only after an explicit provider completion signal.
- Bare iterator EOF may remain a transport success for provider compatibility but cannot establish response identity.
- `[DONE]`, processed usage-backed final chunks, and usage-backed raw SSE final frames establish completion.
- A raw intermediate `finish_reason` without usage is insufficient because some providers populate it on every chunk.
- Event-prefixed SSE and `data:` frames with or without a space are parsed.
- Non-object JSON, malformed choices/deltas, and non-list tool-call payloads are ignored safely.
- Duplicate streamed tool-call IDs are deduplicated.

## Test Coverage

Compaction regressions cover:

- Ordinary long growing history.
- Identical regeneration and resend shapes.
- One-response classifier prompts.
- Full marked replacement.
- Most-but-not-all marked and unmarked replacement.
- Exactly-half retention.
- Majority retention.
- Middle-only replacement.
- Context contraction without summary replacement.
- Two distinct response events versus duplicate response content.
- Trusted explicit and tool identity.
- Cross-scope isolation.
- Exact replay and changed-history non-replay.
- Competing parents with equal scores but different response-event diversity.

Persistence/state tests cover:

- Schema-2 metadata round trip.
- Compaction and replay across two restarts.
- Malformed, unsupported, expired, orphaned, and cross-namespace state.
- Per-session/global caps during runtime and load.
- TTL equality and late responses.
- Shared-anchor ownership.
- Failed writes, exceptional writes, throttled writes, forced flushes, and stale generations.
- Blocked disk I/O without holding the state lock.
- Concurrent inference, response recording, and flushing.
- Deterministic affinity across tracker instances and restart.

Streaming tests cover:

- Completed response evidence.
- EOF without completion evidence.
- Early raw finish reason without usage.
- `[DONE]` and usage-backed completion.
- Event-framed/no-space SSE.
- Non-object/malformed SSE and tool-call payloads.

## Verification

- `python -m pytest tests/test_session_tracking.py -q`: 66 passed, 15 subtests passed.
- Session/request/selection integration slice: 81 passed, 15 subtests passed.
- Streaming regression slice: 70 passed.
- Python compilation passed for both changed runtime modules.
- `git diff --check` passed for all Phase 11 files.
- Explore review: no blocker, high, or medium findings after re-review.
- Explore-heavy review: no blocker, high, or medium findings after final completion-gate re-review.

The unrestricted test run is not currently a clean repository-wide signal. Initial collection was blocked by 13 unrelated retired/Gemini import errors. After excluding those known collections, the run reached 724 passing tests but retained 71 unrelated failures and 7 tool-test errors from existing refactor drift, missing async-test plugins, environment-sensitive tests, and provider changes. Phase 11 focused and integration slices are clean.

## Files

- `src/rotator_library/session_tracking.py`
- `src/rotator_library/client/streaming.py`
- `tests/test_session_tracking.py`
- `DOCUMENTATION.md`
- `docs/experimental/phase-11-session-tracking-hardening.md`
- `docs/experimental/phase-11-session-tracking-hardening-report.md`

## Review Disposition

The first reviews found valid stream-evidence parsing/completion gaps and several missing boundary tests. Those were fixed and re-reviewed. Two reported anchor-ownership defects were independently rejected after direct inspection: shared anchors are skipped before entering a second session set, and global trimming already removes the owning session-set entry. Dedicated load-cap and ownership tests now guard both invariants.
