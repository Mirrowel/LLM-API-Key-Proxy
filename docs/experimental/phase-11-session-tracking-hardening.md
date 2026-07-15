# Phase 11: Session Tracking And Compaction Hardening

## Goal

Make session inference conservative, deterministic, restart-safe, and resistant to false compaction detection. Compaction is a structural replacement of all or most of a known conversation with summary context. Replacing only a minority of history, including a section in the middle, remains ordinary continuity.

## Evidence

- `logs/proxy.log` contains 21 compacted-descendant events across four clusters.
- Normal growing histories were classified as compaction because a long early assistant message matched its response-derived anchor.
- Side-channel classifier requests were classified as compaction because one long user instruction quoted a single immediately preceding response.
- The reported Mistral sequence included a genuine client-level regeneration, but the regeneration itself was not classified. The downstream classifier request after each generation was classified.
- One DeepSeek sequence removed recent turns, but it did not replace the conversation with a summary. The detector had already classified earlier full-history requests and continued creating descendants for resends and follow-ups.
- False compaction changes routing behavior: probe indexes are removed from normal continuity, a new session is created, and sequential selection may choose another credential.
- Persistence was disabled for the captured run and did not cause the incidents, but its loader, ownership, failure, TTL, and concurrency contracts need broader coverage.

## Behavioral Contract

1. A compaction candidate must appear in the first two messages and have an eligible role.
2. Size alone never proves compaction.
3. Ordinary assistant, tool, and function-result history is not compaction-probe material.
4. An explicit marker is evidence, not a decision; a known parent and structural history replacement are still required.
5. Unmarked compaction requires evidence from at least two distinct prior response events.
6. The candidate must replace most of the parent session's high-water request history.
7. Retaining at least half of the high-water history is ordinary continuity.
8. Replacing a minority section in the middle is ordinary continuity even if that section contains summary-like text.
9. Exact replay of a validated compacted request reuses its existing child session.
10. A failed API attempt or partial stream never contributes response identity.
11. Session, provider, model, and credential-scope isolation remains mandatory.

## Scope

- Role-aware and structure-aware compaction decisions.
- High-water request-history profiles on live session state.
- Distinct response-event provenance for response anchors.
- Preservation of response provenance when responses return in request history.
- Replay binding for validated compacted children.
- Session/anchor ownership and pruning invariants.
- Strict, schema-versioned persistence loading and restart behavior.
- Save throttling, failure recovery, stale generations, and lock handoff.
- Complete and partial streaming response-anchor integration.
- Payload-free decision diagnostics.
- Extensive unit, regression, persistence, integration, and concurrency tests.

## Non-Goals

- Do not infer compaction semantically with an LLM.
- Do not inspect or persist raw prompt/response content in tracker state.
- Do not make one quoted response sufficient for unmarked compaction.
- Do not bind repeated standalone prompts that lack trustworthy session evidence.
- Do not change balanced/sequential selection policy beyond corrected session evidence.
- Do not migrate unsupported historical persistence schemas.

## Implementation Plan

1. Request-history profiles.
   - Build deterministic per-message signatures from normalized role, content, and tool identity.
   - Store the largest observed request profile as each session's high-water history.
   - Update equal-sized profiles to follow normal fixed-window conversations.
   - Compare candidate requests against the matched parent's high-water profile using multiset overlap.
   - Treat removal of more than half the parent profile as structural replacement.

2. Compaction candidate selection.
   - Restrict size-only probes to early user/system/developer messages.
   - Keep explicit marker recognition limited to early system/developer context.
   - Validate a parent match before setting `possible_compaction` or suppressing continuity anchors.
   - Reject partial/middle replacement when at least half of parent history remains.

3. Response-event provenance.
   - Derive a deterministic group from each normalized successful response event.
   - Assign all medium response content anchors from one response to that event group.
   - Count distinct response groups rather than matching chunks.
   - Preserve response source/group metadata when the same content is later observed as request history in the same session.
   - Require two distinct response groups for unmarked size-only compaction.

4. Compaction replay binding.
   - Derive an opaque replay anchor from the validated probe payload.
   - Store it only on a confirmed compacted child.
   - Keep parent identity in anchor metadata.
   - Resolve an exact replay before creating another child.

5. State integrity.
   - Preserve the first live owner of shared content anchors instead of letting an auxiliary request steal them.
   - Keep pruning and both trim paths bidirectionally consistent.
   - Preserve stronger/provenance-rich metadata when refreshing the same anchor in one session.
   - Verify every stored anchor has one live owner and every session anchor points back to that owner.

6. Persistence hardening.
   - Bump the persistence schema for high-water history metadata.
   - Treat the anchor mapping as authoritative and rebuild session anchor sets during load.
   - Reject malformed containers, timestamps, strengths, namespaces, orphan anchors, expired records, and unsupported schemas without failing startup.
   - Clamp anchor expiry to its owning session and enforce configured limits after load.
   - Preserve dirty state after failed writes and reject stale delayed snapshots.
   - Keep disk I/O outside the main state lock and serialize it under the save-I/O lock.

7. Diagnostics and documentation.
   - Log confirmed lineage with marker/size decision kind, retained-history ratio, and response-event count.
   - Keep rejected candidate detail at debug level.
   - Never log prompt text or raw anchor values.
   - Update session-stickiness documentation with the structural decision contract.

## Test Matrix

### False-Positive Regressions

- Full ordinary `[user, assistant, user]` continuation with a recorded response.
- Long user plus long assistant history from the June request shape.
- One-message classifier quoting one complete response.
- One-message classifier sharing only response chunks.
- Identical client regeneration before the previous response is appended.
- Context branch contraction without a summary.
- Exact resend of contracted ordinary history.
- Shared large system/developer harness prompts.
- Marker text with no matched parent.
- Oversized assistant/tool messages.
- Sanitized structural fixtures matching every observed incident family.

### Replacement Boundary

- Entire history replaced by a marked summary.
- Most history replaced by a marked summary.
- Most history replaced by an unmarked summary spanning multiple response events.
- Exactly half retained remains ordinary continuity.
- More than half retained remains ordinary continuity.
- One middle turn replaced remains ordinary continuity.
- Several middle turns replaced while a majority remains is ordinary continuity.
- Summary-like text outside the first two messages remains ordinary continuity.
- Parent high-water history is not reduced by a shorter normal request.

### Positive And Replay Cases

- Marked system and developer summaries identify a parent.
- One-response unmarked summary is rejected conservatively.
- Two-response unmarked summary identifies a parent.
- Duplicate copies of one response count as one response event.
- Exact compacted-request replay reuses one child.
- Child follow-up continues after a completed response.
- Strong trusted/provider identity survives a compaction candidate.
- Namespace, provider, model, and credential scope stay isolated.

### Persistence And State

- Schema round trip preserves history profiles and all anchor metadata.
- Restart preserves normal continuation, compaction decisions, and replay binding.
- Missing, invalid, non-object, unsupported, malformed, expired, orphaned, and namespace-mismatched state is ignored safely.
- TTL equality expires state.
- Late response recording after expiry does not resurrect a session.
- Per-session/global trimming preserves ownership invariants.
- Shared anchors keep one live owner and never create stale secondary ownership.
- Affinity is deterministic across fresh trackers and restart.
- Flush throttling, forced flush, writer failure/recovery, stale generation ordering, and mutation during a blocked write are deterministic.
- Concurrent infer/record/flush operations preserve invariants and do not hold the main lock during disk I/O.

### Integration

- Request building performs one inference per logical request.
- Internal credential retries reuse the existing request context/session.
- Successful non-streaming responses record response provenance.
- Completed streams record response provenance.
- Clean EOF without an explicit provider completion signal does not record response provenance.
- Failed, partial, and disconnected streams do not record response provenance.
- Corrected session and affinity values reach sequential selection.

## Acceptance Criteria

- All 21 captured event shapes are rejected as compaction under sanitized replay tests.
- Full/most-history summary replacement is detected when parent evidence is sufficient.
- Middle-only/minority replacement does not create a compacted child.
- One quoted response never proves unmarked compaction.
- Exact replay of a validated compacted request returns the same child.
- Restart preserves all decisions and state invariants.
- Persistence corruption or write failure never fails request handling.
- Main tracker lock is not held during disk I/O.
- Targeted and full test suites pass.
- Independent explore and explore-heavy reviews report no unresolved blocker, high, or medium findings.
