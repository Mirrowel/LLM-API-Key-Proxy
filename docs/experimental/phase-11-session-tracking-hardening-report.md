# Phase 11: Session Tracking And Compaction Hardening Report

## Result

Phase 11 is complete. The captured false-positive families are now rejected, real compaction requires structural replacement of most known history, minority/middle-only replacement remains ordinary continuity, and optional persistence has a strict restart-safe schema and substantially broader failure/concurrency coverage.

The core hardening is committed; later global-domain, binding, Responses, and
closed-tool-event refinements remain in the staged/working-tree change set.

Every request now emits a temporary warning-level lineage decision. The log identifies new roots, continuations, compacted children, exact compaction replays, untracked requests, and sessions restored from persistence.
Accepted continuations use `matched_session_id`; weak evidence rejected while creating a new root is reported separately as `candidate_session_id`.

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
- Unmarked compaction requires overlap with at least two distinct completed response events and request-side parent evidence.
- Duplicate content from one response remains one response event.
- One quoted response cannot create compaction lineage or ordinary sticky response bridging from a user/system/developer role.
- Exact replay of a validated compacted payload reuses the existing child.
- Changed or extended tails retaining the validated summary continue the same child through an evidence-bearing context anchor.
- Context anchors are minted only from probe groups that actually matched parent response evidence; shared long system/user harnesses cannot become bindings by position alone.
- Strong trusted or provider identity takes precedence over replay/context bindings and suppresses unrelated compaction lineage.
- Raw tool-call IDs are supporting evidence, not authoritative identity, because deterministic IDs can be reused across sessions.
- Each one-to-one closed assistant-call/tool-result event contributes one medium
  evidence group keyed by hashed ID, function name, and canonical arguments.
- Closure is request-local: persisted medium evidence cannot upgrade a later
  unpaired call, duplicate results count once, and one result closes one call.
- Generic evidence and logical IDs cross providers/models only inside one strict
  public, classifier, or ad hoc credential-bundle domain. Provider-native
  evidence remains provider/session-scope qualified.

## Tracking And Persistence Hardening

- Successful responses receive deterministic response-event provenance.
- Response provenance survives when assistant content returns in later request history.
- Shared content anchors retain their first live owner instead of being stolen by an auxiliary session.
- Per-session/global trimming and TTL pruning maintain bidirectional session-anchor ownership.
- Late responses cannot resurrect an expired session.
- Weak/ordinary evidence is evicted before strong replay/context identity with deterministic tie-breaking.
- Fallback response callbacks stay in the logical domain while source-provider affinity is cleared before cross-provider selection.
- Persistence schema 3 stores hashed high-water history, scoped anchor metadata, response-event groups, and replay bindings without raw content or external IDs.
- Loading rejects malformed containers, invalid/non-finite timestamps, unsupported schemas, expired sessions, orphan anchors, namespace mismatches, invalid strengths/sources, malformed history signatures, oversized files, and excess sessions/strings.
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
- Streamed function/argument fragments are reconstructed by provider choice and
  tool index; cumulative snapshots replace prefixes instead of being appended.

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
- Trusted explicit/provider identity and raw tool-ID non-authority.
- Sparse tool loops with one closed event plus independent message evidence or
  two distinct closed events, including provider switches and restart.
- Unpaired calls, duplicate results, duplicate IDs, changed function/arguments,
  nameless calls, malformed arguments, and cross-domain events.
- Trusted/provider precedence over existing replay/context bindings.
- Raw short and entropy-looking tool-ID collision resistance.
- Cross-scope isolation.
- Exact replay plus marked/unmarked changed-tail child continuation.
- Shared system/user harness rejection for context binding.
- Two-response aggregation rejection without request-side evidence.
- Competing parents with equal scores but different response-event diversity.

Persistence/state tests cover:

- Schema-3 metadata round trip and explicit schema-2 rejection.
- Compaction and replay across two restarts.
- Malformed, unsupported, expired, orphaned, and cross-namespace state.
- Per-session/global caps during runtime and load.
- TTL equality and late responses.
- Shared-anchor ownership.
- Failed writes, exceptional writes, throttled writes, forced flushes, and stale generations.
- Blocked disk I/O without holding the state lock.
- Concurrent inference, response recording, and flushing.
- Deterministic affinity across tracker instances and restart.
- Context/replay TTL, supported cap pressure, restart, and scope isolation.

Stateful simulations cover:

- Six ordered agentic tool rounds switching DeepSeek/Anthropic/OpenAI, with response recording, tool results, two mid-loop persistence restarts, cross-provider compaction, exact replay after restart, child continuation, and parent continuation.
- Eight long ordinary turns rotating providers/models, with repeated response content, two persistence restarts, per-turn high-water growth, and no false compaction.
- Long roleplay with same/cross-provider exact redo, edited regeneration, a cross-provider middle response rewrite while later turns remain, rollback, resumed branching, and no false lineage.
- First-turn reroll ambiguity, shared harness exclusion, trusted global identity, provider-native collision resistance, independent provider binding clocks, and strict public/classifier/private-bundle isolation.

Responses/isolation tests cover:

- Scoped routing reaches RequestContextBuilder without storing or tracing credential containers.
- Every `previous_response_id` ancestor must belong to the requesting domain.
- Storage ownership is composite by `(domain, response_id)`, including IDs that would collide under path sanitization.
- Non-public retrieval/deletion/input-items require the opaque domain header returned at creation.
- Ad hoc credential bundles use distinct usage managers; named classifier identity remains stable across credential rotation.
- External anchor IDs are hashed and raw scope strings cannot impersonate internally derived domain markers.

Streaming tests cover:

- Completed response evidence.
- EOF without completion evidence.
- Early raw finish reason without usage.
- `[DONE]` and usage-backed completion.
- Event-framed/no-space SSE.
- Non-object/malformed SSE and tool-call payloads.
- Incremental/cumulative arguments, interleaved calls, shared tool indexes across
  choices, and choices arriving in separate frames.

## Verification

- `python -m pytest tests/test_session_tracking.py -q`: 119 passed, 18 subtests passed.
- Final session/request/selection/routing/classifier/executor slice: 173 passed, 18 subtests passed.
- Final Responses bridge/service/store/routes/streaming/accounting slice: 76 passed.
- Python compilation passed for all changed runtime modules.
- `git diff --check` passed for all Phase 11 files.
- Explore scenario review: no blocker, high, or medium findings after final re-review.
- Explore-heavy isolation/persistence review: no blocker, high, or medium findings after the final domain-marker, composite-storage, and zero-TTL fixes.
- Explore-heavy closed-tool-event review: no blocker, high, or medium findings
  after request-local closure, one-to-one pairing, and stable streamed choice
  indexing were verified.

The unrestricted test run is not currently a clean repository-wide signal. Initial collection is blocked by 13 unrelated retired/Gemini import errors. After excluding only those known collections, the active run reached 791 passing tests plus 16 subtests, with 68 unrelated failures from existing refactor drift, missing async-test plugins, legacy constructor assumptions, and order-dependent provider-global test state. Phase 11 focused and integration slices are clean.

## Files

- `src/rotator_library/session_tracking.py`
- `src/rotator_library/client/rotating_client.py`
- `src/rotator_library/client/request_builder.py`
- `src/rotator_library/client/scopes.py`
- `src/rotator_library/client/streaming.py`
- `src/rotator_library/client/executor.py`
- `src/rotator_library/core/types.py`
- `src/rotator_library/providers/provider_interface.py`
- `src/rotator_library/routing/attempts.py`
- `src/rotator_library/usage/config.py`
- `src/rotator_library/usage/selection/strategies/sequential.py`
- `src/rotator_library/responses/bridge.py`
- `src/rotator_library/responses/service.py`
- `src/rotator_library/responses/store.py`
- `src/proxy_app/main.py`
- `tests/test_session_tracking.py`
- `tests/test_request_builder_routing.py`
- `tests/test_selection_engine.py`
- `tests/test_routing_attempts.py`
- `tests/test_classifier_scoped_routing.py`
- `tests/test_responses_bridge.py`
- `tests/test_responses_service.py`
- `tests/test_responses_store.py`
- `tests/test_responses_routes.py`
- `.env.example`
- `README.md`
- `DOCUMENTATION.md`
- `docs/experimental/phase-11-session-tracking-hardening.md`
- `docs/experimental/phase-11-session-tracking-hardening-report.md`

## Review Disposition

The first reviews found valid stream-evidence parsing/completion gaps and several missing boundary tests. The stateful-scenario reviews then found changed-tail child forking, context precedence, shared-harness binding, namespace migration, and raw tool-ID collision risks. The closed-tool-event review found persisted-strength promotion, many-to-one result pairing, and streamed choice-index collisions. These were reproduced, fixed, and covered by dedicated restart/isolation tests. Two reported anchor-ownership defects were independently rejected after direct inspection: shared anchors are skipped before entering a second session set, and global trimming already removes the owning session-set entry.
