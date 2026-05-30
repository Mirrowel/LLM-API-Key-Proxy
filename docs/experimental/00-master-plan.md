# Experimental Native Protocol Roadmap

This branch is for a long-running experimental rewrite that makes native protocol support the first-class extension point of `rotator_library`, while preserving the existing credential rotation, quota, fair-cycle, session tracking, and provider plugin strengths.

## Operating Rules

- Work only on the `experimental` branch.
- Keep all repository work inside `C:\Projects\test\LLM-API-Key-Proxy` and child paths.
- Treat commits as checkpoints. A phase may contain many commits.
- Commit messages must include a body describing what changed, why, tests run, and follow-up considerations.
- Do not commit phase reports written for the user unless explicitly requested. Planning docs under `docs/experimental/` are committed.
- Before each phase implementation, first produce a fresh exhaustive phase plan in conversation text, based on the current code state. Only after that plan is settled should it be written to `docs/experimental/phase-N-*.md`.
- After each phase implementation, call both `explore` and `explore-heavy` agents to review the work against the phase plan, external reference areas, and current proxy behavior. Fix findings and re-review as needed.
- Keep LiteLLM as a fallback path for protocols/providers that are not natively covered yet. Native protocol support should be preferred when available.

## Strategic Goal

The target architecture is:

```text
client API request
  -> protocol parse into unified representation
  -> field-cache injection
  -> adapter chain
  -> provider override hooks
  -> provider-native request build
  -> provider execution and credential rotation
  -> provider-native response/stream parse
  -> field-cache extraction
  -> adapter chain
  -> protocol formatting for the client
  -> transaction logging for every transform state
```

Providers should be able to declare an existing protocol and only override the parts that are genuinely provider-specific. A custom provider should usually be configurable through protocol choice, adapters, field-cache rules, auth strategy, and model options rather than requiring a large bespoke provider implementation.

## Priority Order

1. Native protocol foundations, unified types, transformers, adapters, and field-cache rules.
2. OpenAI Responses API support, including future WebSocket extension points.
3. Provider work following the protocol layer: Claude Code, Codex, Copilot, Antigravity, and Gemini CLI parity review.
4. Routing and fallback groups, with optional target-group selectors later.
5. Retry, provider/model cooldown, and failover cleanup.
6. Protocol-aware quota, usage, and cost normalization.
7. Streaming library hardening: SSE now, WebSocket-ready later.
8. Config polish using `.env` and optional JSON. No SQLite dependency for now.
9. Extensive staged tests and review-agent verification.

## Non-Goals For This Branch

- Do not make the proxy a full multi-user admin product yet.
- Do not require SQLite or Postgres for the main feature set.
- Do not remove LiteLLM before native coverage exists.
- Do not replace the existing `UsageManager`, fair-cycle, custom caps, or evidence-based `SessionTracker`.
- Do not port frontend/UI work from the external reference gateway.

## Current Strengths To Preserve

- Credential-level rotation and priority-aware selection.
- Fair cycle and custom caps.
- Windowed quota tracking and quota groups.
- Evidence-based session tracking with compaction handling.
- Provider plugin discovery.
- Gemini CLI provider behavior unless a reviewed change is clearly better.
- Resilient file/JSON state writing.
- Dynamic OpenAI-compatible provider discovery.

## Reference Gateway Ideas To Import Carefully

- Unified protocol/transformer style.
- Adapter registry and configurable provider/model adapters.
- Target groups and direct routing syntax, adapted into fallback-first routing.
- Responses API transformer and storage concepts.
- Stream TTFB/stall detection concepts, implemented with Python-native async primitives.
- Provider/model cooldown and retry-history concepts.
- Usage/cost normalization and provider-reported cost extraction.
- Broader provider support patterns for Claude Code, Codex, Copilot, and Antigravity.

## Phase Index

1. Protocol Core.
2. Transform Pass Logging.
3. Adapter and Field Cache System.
4. Responses API and WebSocket-Ready Transport Shape.
5. Provider Protocol Overhaul.
6. Routing and Fallback Groups.
7. Retry/Cooldown/Failover Cleanup.
8. Streaming Library Upgrade.
9. Usage, Quota, and Cost Accuracy.
10. Config Polish.

Each phase may be subdivided if implementation scope becomes too large.

## Completeness Matrix

This matrix exists so the branch does not lose any requested scope while phases evolve. The phase plans are still refreshed before implementation, but every item below must remain accounted for.

| Requested area | Planned coverage |
| --- | --- |
| Protocols are priority #1 | Phases 1 and 4 create native protocol foundations and Responses support before provider work. |
| Protocols are bases, not gospel | Phase 1 requires override-friendly protocol methods, subclassing, copy/mutate registration, and provider-specific overrides. |
| Move away from LiteLLM | Phase 1 adds a `litellm_fallback` protocol path; later providers should prefer native protocols and use LiteLLM only for unsupported coverage. |
| Add protocols automatically like providers | Phase 1 adds protocol auto-discovery and registry behavior modeled after provider discovery. |
| Cover current providers and reference providers | Phase 1 protocols must cover shapes used by current providers; Phase 5 covers Claude Code, Codex, Copilot, Antigravity, and Gemini CLI parity. |
| Responses API is very needed | Phase 4 is dedicated to Responses, `previous_response_id`, storage, SSE, and WebSocket-ready transport shape. |
| WebSocket support later | Phases 1, 4, and 8 require transport separation so WebSocket can be added without rewriting protocol logic. |
| Adapters/transformers tied to protocols | Phases 1, 2, and 3 define protocol parse/build plus transform tracing, adapter registry, and field-cache rules. |
| Cache and return provider fields | Phase 3 implements configurable extraction/injection rules for request, response, and stream fields with scope and mode controls. |
| Reasoning content and similar fields | Phase 3 explicitly covers reasoning content, thinking signatures, prompt cache keys, response IDs, and provider session IDs. |
| Return all possible or last user/assistant use | Phase 3 modes include `last`, `all`, `last_user_turn`, `last_assistant_turn`, and `per_tool_call`. |
| Per-model custom provider behavior | Phases 3, 5, and 10 cover provider/model field cache rules, adapters, model options, and optional JSON config. |
| Transaction logging after every transform | Phase 2 adds ordered request, response, and stream transform trace passes and integrates them with transaction logging. |
| Comments, docstrings, and key decisions | All implementation phases require docstrings for public abstractions and comments for non-obvious transform, protocol, and future-extension decisions. |
| Providers are priority #2 | Phase 5 follows protocol foundations with Claude Code, Codex, Copilot, Antigravity, and Gemini CLI parity review. |
| Antigravity comparison | Phase 5 explicitly compares the reference Antigravity behavior against `src/rotator_library/providers/_retired/`. |
| Routing is interesting | Phase 6 implements fallback chains first, with target-group selectors later if useful. |
| Fallback groups preferred over target groups | Phase 6 starts with ordered fallback groups and only adds target-group-style selectors after that base works. |
| Retry/cooldown/failover cleanup | Phase 7 makes provider/model cooldown real, adds retry history, backoff, retry-after precedence, and success reset. |
| Quota/usage/cost improvements | Phase 9 adds protocol-aware normalizers, provider-reported cost extraction, structured cost fields, and checker abstractions while keeping existing usage engines. |
| Streaming as library capability | Phase 8 hardens streaming below the proxy route layer with TTFB, TTFT, stall detection, cancellation, and transport-aware stream events. |
| Config via env/json, no SQLite | Phase 10 adds optional JSON config with env overrides and validation. SQLite remains out of scope. |
| Multi-user proxy later | The branch keeps multi-user/admin features as a future expansion and only preserves extension points where natural. |
| Exhaustive tests in stages | Every phase requires tests alongside implementation and phase-end review by both `explore` and `explore-heavy`. |
| Reports are for the user, not git | `06-phase-workflow.md` says planning docs are committed, but phase reports are not committed by default. |

## Code Quality Expectations

- Public protocol, adapter, transport, field-cache, and provider-extension classes must have docstrings that explain intent, override points, and future expansion hooks.
- Non-obvious transformations must have comments explaining why data is changed, preserved, reordered, or intentionally dropped.
- Lossy protocol conversions must be documented at the conversion site.
- Future WebSocket, target-group, and multi-user extension seams should be noted in comments where they affect today's design.
- Tests should prefer golden fixtures for protocol shapes and focused unit tests for transform edge cases.
