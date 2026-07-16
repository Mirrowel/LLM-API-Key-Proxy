# Protocol Interoperability Review And Implementation State

**Status:** Active correction phase
**Started:** 2026-07-16
**Branch:** `experimental`
**Purpose:** Durable review, implementation contract, task breakdown, and live state for the protocol interoperability correction.

## Product Contract

The proxy must treat client protocols and provider protocols as independent choices.

For every supported generative protocol:

```text
client payload
  -> client protocol parser
  -> canonical request
  -> provider protocol builder
  -> provider
  -> provider protocol parser
  -> canonical response or stream events
  -> selected output protocol formatter
  -> client payload
```

The required first-class generative protocols are:

- OpenAI Chat Completions
- OpenAI Responses
- Anthropic Messages
- Gemini generateContent / streamGenerateContent

The input protocol, provider protocol, and output protocol must be independently selectable. The default output protocol is the originating client protocol, but callers and configured integrations may choose another supported output protocol.

Examples that must be valid:

- Chat client -> Anthropic provider -> Chat response
- Responses client -> Gemini provider -> Responses response
- Gemini client -> OpenAI-compatible provider -> Gemini response
- Gemini client -> provider-native protocol -> Anthropic response
- Anthropic client -> Responses provider -> Chat response when explicitly selected

Providers must not need to know which protocol the client used. A provider declares only the protocol and operation it expects, plus provider-specific endpoint, authentication, envelope, model alias, cache, quota, and adapter behavior.

Custom providers configured at runtime must be able to declare a supported provider protocol. Custom integrations must also be able to select an output protocol rather than being limited to OpenAI-compatible output.

## Review Verdict

The existing branch contains useful protocol primitives but does not implement the product contract end to end.

The current live architecture uses OpenAI Chat as the effective internal language:

- Chat requests enter the completion executor directly.
- Anthropic requests are converted to Chat by the legacy compatibility handler before execution.
- Responses requests are converted to Chat by `ResponsesBridge` before execution.
- No public Gemini client route exists.
- Native request execution asks the provider protocol to parse an already Chat-shaped request.
- Provider preparation hooks perform partial Chat-to-provider reshaping.
- Native responses are always formatted as OpenAI Chat before optional outer wrappers convert them again.

This works for basic text demonstrations but not for general protocol interoperability.

## Confirmed Architectural Findings

### Client protocol identity is not authoritative

The originating protocol is not carried through the common request context. Native execution currently fixes the client response protocol to OpenAI Chat.

### Native request conversion uses the wrong source parser

The provider protocol currently parses the incoming request. Correct conversion requires the client protocol to parse the incoming request and the provider protocol to build the outgoing provider request.

### Public protocol routes bypass the common protocol core

Anthropic and Responses use dedicated Chat bridges. Gemini is not available as a client-facing protocol.

### Provider hooks contain protocol translation

Provider preparation currently performs partial message conversion for some native providers. This creates provider-specific translation islands and handles mostly plain text. Provider hooks must instead run after provider-protocol formatting and be limited to provider quirks.

### Raw and extension preservation is unsafe across protocols

The existing `raw` and `extra` fields are valuable for same-protocol fidelity and tracing. Destination formatters currently replay some source-native fields into different protocols, producing hybrid payloads.

### The unified model is only partly canonical

Several fields still preserve source-protocol concepts rather than common meaning:

- content block type names
- system and developer instruction placement
- generation parameter dictionaries
- tool placement and tool configuration
- stop reasons and statuses
- output item layout
- stream event names and lifecycle
- provider-native extensions

### Streaming is not protocol-independent

All active native providers disable native streaming by default. Only the OpenAI Chat protocol has a complete cross-protocol stream formatter. Responses streaming is operationally strong but text-only, and Anthropic streaming still uses the legacy Chat wrapper.

### Existing tests prove adapters more than interoperability

Most protocol tests verify same-protocol parse/build or parse/format behavior. Native provider tests verify simple text requests and responses. There is no complete client-protocol x provider-protocol x output-protocol contract suite.

## Directly Reproduced Conversion Failures

Read-only runtime probes against the current protocol classes reproduced the following:

- Chat system messages remained conversation messages when written as Anthropic instead of becoming the Anthropic system field.
- Chat images written as Anthropic lost the image source.
- Chat assistant tool calls disappeared from Anthropic output.
- Chat tool results remained an unsupported `tool` role in Anthropic output.
- Anthropic image, thinking, and tool-use blocks leaked into Chat content while malformed duplicate Chat tool calls were also emitted.
- Chat system messages written as Gemini retained an invalid system content role.
- Chat image and tool objects leaked into Gemini parts instead of being converted.
- Chat tool calls written as Responses stayed embedded in message objects instead of becoming Responses function-call items.
- Anthropic and Gemini stop reasons were returned as raw Chat finish reasons.
- Responses reasoning, message, and function-call output items became separate Chat choices instead of one semantically correct Chat response.

## Preservation Policy

### Meaning-changing fields

The proxy must not silently discard or corrupt fields that alter the meaning or required behavior of a request.

These include:

- system and developer instructions
- user and assistant content
- tool definitions
- tool choice requirements
- tool calls and tool results
- required images, files, audio, or other modalities
- structured output and schema requirements
- continuation identity
- required reasoning behavior
- explicit response modalities

If the selected provider protocol cannot represent a required meaning safely, conversion must fail before contacting the provider with an error formatted for the selected output protocol. The error must identify the unsupported capability without exposing secrets or internal provider state.

### Optional tuning fields

Advisory controls may be omitted when no safe equivalent exists, provided the omission is recorded in a concise conversion summary.

Examples include provider-unsupported sampling hints, log-probability requests, service tiers, or optional vendor tuning fields. A future strict mode may promote selected warnings to errors.

### Unknown and provider-extension fields

Unknown fields are protocol-owned extensions, not universal fields.

- Same input and provider protocol: preserve safe unknown fields and replay them unless explicitly blocked.
- Different input and provider protocol: do not forward unknown fields without an explicit mapping.
- Provider response to the same output protocol: preserve safe provider-native extensions.
- Provider response to a different output protocol: keep unknown provider fields internal unless the output protocol has an explicit extension mapping.
- Secret, credential, cache signature, and provider-state fields are never generic passthrough fields.

This preserves same-protocol compatibility without creating invalid cross-protocol hybrid payloads.

## Provider-State Policy

Opaque provider state includes thought signatures, reasoning signatures, provider response IDs, prompt-cache IDs, and similar values that a client may never receive or return.

Policy:

- Extract opaque state from provider responses and completed stream events.
- Cache it even when it is not exposed in the selected output protocol.
- Scope it at least by provider, model family, credential, and logical session when provider validity requires those dimensions.
- Inject it only when returning to a compatible provider context.
- Never move opaque signatures from one provider family to another.
- Never assume reasoning text and opaque reasoning signatures are interchangeable.
- Permit client-visible reasoning text to translate only when the destination protocol supports it and policy allows exposure.
- Completion-gate stream extraction so partial or failed streams do not establish false continuation state.

The existing field-cache system is the intended mechanism. Its declarations, extraction sources, injection timing, scopes, persistence behavior, and trace redaction must be verified during this correction.

## Canonical Model Requirements

### Request

The canonical generative request must represent:

- logical operation independent of provider endpoint operation
- model selection
- ordered system and developer instructions
- user, assistant, and tool turns
- text content
- image content with URL/data/file identity, MIME type, and detail metadata
- document/file content
- audio/video content through extensible media descriptors
- tool definitions and full input schemas
- provider-native built-in tool categories where meaning can be normalized
- tool choice and function-calling policy
- assistant tool calls
- tool results with correlation identity and error state
- reasoning policy, effort, budget, summary, and visibility
- output token limit
- temperature, top-p, top-k, penalties, seed, and stop sequences
- structured output requirements and JSON schema
- response modalities
- continuation and cache routing identity
- metadata
- source-protocol extensions isolated by protocol
- original payload for tracing and same-protocol preservation

### Response

The canonical generative response must represent:

- response identity and provider-native identity
- model
- ordered output items
- assistant content
- reasoning items and summaries
- tool calls
- annotations and citations
- canonical completion status
- canonical stop reason plus source-native reason
- usage and cost
- errors and incomplete status
- provider-protocol extensions isolated by protocol
- original payload for tracing and same-protocol preservation

### Stream

The canonical stream must model lifecycle rather than source event names:

- response started
- output item started
- text started, delta, and completed
- reasoning started, delta, and completed
- tool call started
- tool argument delta
- tool call completed
- annotations and media events where supported
- usage
- response completed
- response incomplete
- response failed
- transport heartbeat as transport metadata rather than model output

Every protocol formatter must be able to construct its own valid stream lifecycle from these events.

## Runtime Architecture Requirements

### Protocol selection

Each execution carries three independent protocol identities:

- input protocol
- provider protocol
- output protocol

The output protocol defaults to the input protocol.

### Operation selection

Logical operation and provider wire operation are separate.

Chat Completions, Anthropic Messages, Responses, and Gemini generateContent all fulfill the same logical generative operation while using different wire operation names and endpoints.

### Provider preparation

Provider-specific preparation runs after the provider protocol has built a valid provider-native payload.

Allowed responsibilities:

- provider envelope
- model aliasing
- provider-required defaults
- provider-specific headers outside the payload
- endpoint selection
- narrow schema quirks

Disallowed responsibility:

- generic client-protocol to provider-protocol translation

### Same-protocol fast path

A same-protocol path may start from the original payload to preserve unknown fields, then overlay canonical changes. This optimization must be source/target gated and must not bypass credential, routing, cache, usage, error, or trace behavior.

### LiteLLM

LiteLLM remains an explicit execution fallback for providers that have not opted into native transport. It must not remain the implicit canonical representation for all public client protocols.

## Public API Requirements

### Existing routes

- `/v1/chat/completions` defaults input/output to OpenAI Chat.
- `/v1/messages` defaults input/output to Anthropic Messages.
- `/v1/responses` defaults input/output to OpenAI Responses and retains storage/continuation behavior.

### Gemini routes

Add first-class Gemini request support for generateContent, streamGenerateContent, and countTokens-compatible operations. Route aliases and URL model extraction should match practical Gemini clients while preserving the proxy's provider/model routing convention.

### Output protocol override

Library callers and configured integrations must be able to select a supported output protocol independently of input protocol. The public HTTP override mechanism must be explicit, authenticated by the existing proxy boundary, and excluded from provider payloads.

## Configurable Provider Requirements

Runtime provider configuration must support:

- provider protocol name
- endpoint or API base behavior
- authentication/header mode using existing secret handling
- native operation mapping
- streaming capability
- ordered adapters
- field-cache rules
- default output protocol for integrations that require one

Configuration must validate protocol names, operations, transports, and incompatible combinations before requests reach the provider.

Provider-specific quota, tier, priority, reset-window, and concurrency data remains defined by provider classes. Protocol selection does not centralize provider-variable policy.

## Test Contract

The generative protocol contract suite must cover every input/provider/output combination for the four primary protocols.

For each meaningful combination, verify:

- plain text
- ordered system and developer instructions
- multimodal text and image input
- tool definitions
- required/automatic/specific tool choice
- assistant tool calls
- tool results
- reasoning text
- opaque reasoning signatures where applicable
- structured output
- generation controls
- canonical stop/status mapping
- usage details
- safe unknown-field behavior
- errors
- non-streaming output
- streaming text
- streaming reasoning
- streaming tool calls and argument deltas
- completion and failure lifecycle

Same-protocol fidelity and cross-protocol semantic correctness are separate test classes.

Tests must assert destination payload validity, not merely the presence of a response.

## Implementation Phases

### Phase A: Durable review and contract

State: **completed**

- Persist this review and implementation state.
- Establish product and preservation policies.
- Record completion criteria and phased work.

### Phase B: Canonical model and protocol contract

State: **completed**

- Add canonical instruction, media, tool policy, generation, status, output-item, error, and stream lifecycle semantics.
- Preserve non-generative operation compatibility.
- Add source-scoped extension handling.
- Separate logical and wire operations.

### Phase C: Protocol conversions

State: **completed**

- Rework Chat, Anthropic, Responses, and Gemini readers and writers around canonical meaning.
- Implement stop/status and generation-control mappings.
- Add destination capability validation.
- Add complete non-streaming cross-protocol golden tests.

### Phase D: Runtime wiring

State: **completed**

- Carry input/provider/output protocol identities through the common request context.
- Parse with the input protocol and build with the provider protocol.
- Parse with the provider protocol and format with the selected output protocol.
- Move provider preparation after provider-protocol formatting.
- Replace hardcoded Chat output selection.
- Keep legacy routes available only as controlled fallback during migration.

### Phase E: Public Gemini and configurable providers

State: **completed for non-streaming; streamGenerateContent moves with Phase F**

- Add Gemini generateContent and countTokens-compatible routes.
- Reserve streamGenerateContent for the canonical streaming checkpoint so the
  public route never exposes an unconverted or falsely advertised stream.
- Add explicit output protocol selection.
- Extend custom provider configuration to supported native protocols and output defaults.
- Validate configurations before execution.

### Phase F: Canonical streaming

State: **pending**

- Implement canonical stream lifecycle.
- Implement all four input/output stream formatters.
- Preserve text, reasoning, tool calls, argument deltas, usage, completion, and errors.
- Integrate existing timeout, heartbeat, cancellation, retry visibility, and completion-gated state behavior.

### Phase G: Provider state and integration validation

State: **pending**

- Verify hidden signature and provider continuation caching.
- Verify provider/credential/model/session isolation.
- Verify response protocol output does not affect provider-state extraction.
- Run mocked provider contract tests for every native provider.
- Run full local test suite.

### Phase H: Independent review and correction

State: **pending**

- Run an `explore` mapping review after implementation appears complete.
- Run an `explore-heavy` reasoning review after implementation appears complete.
- Reproduce and fix all confirmed blocker, high, and medium findings.
- Repeat both reviews after substantial fixes.
- Update this document with final evidence and remaining intentional limitations.

## Commit Checkpoints

Commits are allowed for this correction and should be made after meaningful verified work.

Planned checkpoints:

1. Persist review and implementation contract.
2. Canonical model and base protocol contract.
3. Complete non-streaming protocol conversions and contract matrix.
4. Runtime input/provider/output protocol wiring.
5. Public Gemini and configurable provider protocol support.
6. Canonical streaming and stream contract matrix.
7. Provider-state verification and integration hardening.
8. Independent review corrections and final state update.

Before each commit:

- inspect status and diff
- run focused tests for the checkpoint
- stage only intended files
- use a concise repository-style commit message

## Completion Criteria

This correction is complete only when:

- all four primary protocols can be used as input protocols
- all four primary protocols can be used as provider protocols
- all four primary protocols can be selected as output protocols
- the default output matches the input protocol
- providers do not translate client protocols themselves
- custom configured providers can declare supported provider protocols
- required semantics are preserved or rejected explicitly before transport
- same-protocol extensions are preserved safely
- cross-protocol extensions do not leak blindly
- non-streaming tools, results, reasoning, media, instructions, and structured output work across supported pairings
- streaming text, reasoning, tools, usage, completion, and failure work across supported pairings
- hidden provider state remains cached and correctly isolated regardless of selected output protocol
- focused and full local tests pass
- final `explore` and `explore-heavy` reviews have no unresolved blocker, high, or medium findings
- this document records final verification evidence and intentional limitations

## Live State Log

### 2026-07-16: Initial protocol readiness review

- Mapped the protocol registry, canonical types, four primary protocol adapters, native executor, public routes, provider preparation hooks, tests, and the in-repo reference gateway.
- Confirmed the protocol product contract was not implemented end to end.
- Reproduced invalid cross-protocol payloads directly.
- Confirmed simple text is the primary working cross-protocol case.
- Confirmed all current public protocol routes converge on OpenAI Chat before provider execution.
- Confirmed native output is hardcoded to OpenAI Chat.
- Confirmed no public Gemini client route exists.
- Confirmed native streaming remains disabled for all four active native providers.
- Product owner confirmed Gemini must be a first-class client protocol.
- Product owner delegated unsupported-field and extension policy decisions to engineering judgment.
- Product owner confirmed opaque provider state should remain cached even when clients do not receive or return it.
- Selected strict preservation for meaning-changing semantics, warning-based best effort for optional tuning fields, same-protocol extension passthrough, and explicit-only cross-protocol extension mapping.

### 2026-07-16: Canonical non-streaming conversion checkpoint

- Added independent input/provider/output protocol identities and logical-operation identity to the protocol context and unified records.
- Added canonical source ownership, conversion warnings, media identity, ordered output items, normalized tool arguments/results, and canonical completion reasons.
- Reworked Chat, Anthropic, Responses, and Gemini request/response readers and writers around canonical instruction, content, reasoning, tool, media, generation, status, and usage semantics.
- Added source-aware passthrough: same-protocol extensions remain available while foreign raw fields and reasoning extensions are not replayed into another protocol.
- Added destination validation for required content, media identity, tool identity, tool choice, response modalities, provider-bound continuation/background controls, and failed-response envelopes.
- Added explicit warning behavior for unsupported optional controls and strengthened schema behavior.
- Added provider-state compatibility gating for hidden reasoning/thought signatures. Protocol equality alone is not sufficient to emit opaque state.
- Added a four-by-four non-streaming request and response semantic matrix with destination-wire assertions and focused regressions for all review findings.
- Corrected tool-result behavior so application JSON containing an `error` field is never reclassified as transport failure. Native Anthropic `is_error` is encoded explicitly for protocols without a native error flag without relying on ambiguous reverse inference.
- Ran four iterative `explore` and `explore-heavy` review passes. Both reviewers signed off with no unresolved blocker, high, or medium findings for the isolated canonical non-streaming checkpoint.
- Verification: 172 focused protocol/native-provider tests passed; `git diff --check` passed except repository line-ending notices.
- A raw `pytest -q tests` collection remains blocked by pre-existing retired Antigravity tests and two stale quota tests importing removed modules. This is recorded for the later full-local-verification phase and was not caused by this checkpoint.
- Runtime execution still uses provider-side parsing of Chat-shaped requests and hardcodes Chat output. This is intentionally not claimed as fixed until Phase D.
- Canonical streaming remains intentionally deferred to Phase F.

### 2026-07-16: Live non-streaming runtime checkpoint

- Added `RotatingClient.agenerate()` as the protocol-independent generative entry point while preserving `acompletion()` as the Chat-compatible facade.
- The request builder now retains the untouched client payload, parses it with the selected input protocol, and carries canonical meaning plus independent input/output identities through retries and fallback targets.
- Native execution now follows the required sequence: client protocol reader, provider protocol writer, narrow provider preparation, provider transport, provider protocol reader, selected output protocol writer.
- Removed provider-owned Chat-to-Responses and Chat-to-Gemini translation. Codex and Antigravity preparation hooks now receive already-valid provider-native payloads.
- Native provider session hooks and validators receive provider-native request shapes. Native attempts bypass the legacy Chat/LiteLLM provider-transform pipeline; callbacks alone retain a Chat compatibility view.
- Callback compatibility views are deep copies. Nested message/tool edits are merged back into canonical meaning field by field without mutating the immutable baseline or discarding source-native metadata such as Anthropic cache controls.
- Custom and LiteLLM non-streaming responses now pass through the canonical response reader/writer when output differs from Chat.
- Anthropic Messages and Responses non-streaming services now execute through the common protocol path. Responses retains local storage, capability, continuation expansion, and session behavior without converting through Chat.
- Local Responses lineage disables provider continuation-cache injection, preventing locally expanded history from also receiving an upstream `previous_response_id`.
- Opaque thought signatures are extracted from raw provider responses and cached before client formatting. They are never returned to clients, including same-protocol output, and are injected only from provider/credential/session-scoped cache state.
- Centralized top-level provider error normalization handles object, string, and numeric-status envelopes before success parsing. Structured status survives credential rotation, fallback policy, Responses service handling, and protocol-specific HTTP formatting.
- Fixed the public Chat route so successful native dictionaries are returned as successes instead of being mistaken for generic 429 error dictionaries. Raw response logging supports both dictionaries and SDK models.
- Added a 64-case live runtime matrix covering all input/provider/output combinations, a real `agenerate` builder-to-native-executor handoff, real nested callback behavior, two-credential structured-error exhaustion, provider-native hook/validation shape, continuation suppression, response-adapter ordering, and hidden-signature caching/suppression.
- Required iterative `explore` and `explore-heavy` reviews completed. Every confirmed blocker, high, and medium finding was corrected and both reviewers issued final safe-to-commit verdicts.
- Verification: 269 focused protocol/runtime/provider tests passed; all 68 currently tracked test files passed with 763 tests and 18 subtests; compile checks and `git diff --check` passed apart from repository line-ending notices.
- Canonical streaming, public Gemini endpoints, and configuration-defined provider protocols remain explicitly deferred to Phases E and F.

### 2026-07-16: Public protocol and configurable-provider checkpoint

- Added first-class Gemini `generateContent` and `countTokens` client routes for
  both `/v1beta/models/...` and `/v1/models/...`. Bare Gemini model IDs route to
  the Gemini provider unless an explicit model-route alias exists.
- Added independent non-streaming output selection. Precedence is explicit
  library argument, `X-Proxy-Output-Protocol`, configured provider default, then
  the input protocol. Aliases are normalized case-insensitively and unknown
  values fail as client errors.
- Chat, Anthropic, Responses, and Gemini ingress now share that selector.
  Responses always executes and stores a native Responses object before an
  optional return-format conversion, preserving retrieval and continuation.
- Added config-defined custom providers with startup-snapshotted API base,
  provider protocol, endpoint templates, auth-header mode, model list, adapters,
  field-cache rules, native-stream capability, and default output protocol.
- Limited configurable provider and output protocols to the four supported
  generative protocols. Broader registered operation adapters such as embeddings,
  audio, and images are not falsely advertised as interchangeable generation
  protocols.
- Kept credentials outside structured config. API bases reject userinfo, query,
  and fragments; operation paths remain on the configured origin, reject secret-
  bearing query keys and fragments, and permit protocol selectors such as
  Gemini's `?alt=sse`.
- Native HTTP transport now treats only 2xx as success and preserves non-success
  status plus structured provider bodies for retry, cooldown, and selected-
  protocol error formatting.
- Aggregate credential exhaustion now raises a protocol-formatable error without
  regressing ordered fallback: structured failure details determine retryability,
  and final errors retain all target and attempt summaries.
- Cross-protocol streams are rejected centrally from both `agenerate()` and
  direct `acompletion()` entry paths. Gemini `generateContent` also rejects
  `stream=true`; the real `streamGenerateContent` route is intentionally coupled
  to Phase F's canonical event conversion.
- Iterative `explore` and fresh `explore-heavy` audits found and drove fixes for
  header normalization, stream bypasses, endpoint credential safety, native HTTP
  errors, Responses error formatting, fallback exhaustion, config snapshots,
  non-generative declarations, redirects, and public test tracking.
- Final verification: the complete git-tracked local suite passes 889 tests and
  18 subtests; the widened protocol/public/provider subset passes 390 tests;
  `compileall` and staged `git diff --check` pass apart from repository line-
  ending notices.
- Final `explore` and `explore-heavy` indexed-state reviews both issued explicit
  safe-to-commit verdicts with no unresolved blocker, high, or medium findings.
- Intentional boundaries: custom-provider transport config is snapshotted at
  process startup; Gemini `countTokens` is locally estimated; cross-protocol
  streaming and the public `streamGenerateContent` route remain Phase F work.

### Current next action

Finish Phase E sign-off and commit, then implement Phase F: canonical streaming
across Chat, Anthropic, Responses, and Gemini, including the public
`streamGenerateContent` surface.
