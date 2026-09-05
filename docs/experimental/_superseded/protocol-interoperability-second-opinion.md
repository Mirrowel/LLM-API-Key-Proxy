# Protocol Interoperability Review — Second Opinion

**Primary implementation ledger:** `docs/experimental/protocol-interoperability-review.md`  
**Purpose:** Durable independent review of the Phase A-H protocol correction. This file is owned by the review session and should be updated as implementation work returns with source changes, tests, and primary evidence.

> **Recovery note (2026-09-05):** The original file was lost from the working tree (it was never committed). This copy was reconstructed by the review session from its own verbatim records: the original second-opinion verdict and the full grilling decision log through Decision 9 plus grilling closure. Content is faithful to what was written; only this note is new. The branch advanced after the original review (`9530632` canonical cross-protocol streaming, `fcae83f` provider state isolation, `5499ad1` selected error boundaries, `f4ac60a` agent platform sync), so the defect references below reflect the 2026-07 state of `experimental` and must be re-verified against current HEAD before the correction plan is executed or closed.

> **Decision-log precedence:** The grilling decision log in this document records the user's latest product decisions. Where it conflicts with earlier findings or correction-plan wording below, the decision log takes precedence. After the grilling is complete, this review and the existing implementation/phase documents will be synthesized into one new comprehensive plan, and that new plan will be the authoritative implementation contract.

## Second-Opinion Verdict

I agree with the original pre-correction diagnosis, but I disagree with the later claim that Phases A-H made the protocol core beta-ready.

The seven correction commits fixed the central routing architecture. They did not finish semantic fidelity. The current system is a credible interoperability foundation, but it can still reorder, discard, merge, or malformedly reconstruct required information.

The status in `docs/experimental/protocol-interoperability-review.md:675-727` should be reopened. "No protocol implementation work remains" is incorrect.

## What Is Fixed

- Chat, Responses, Anthropic, and Gemini are now explicit input protocols.
- Input, provider, and output protocol identities are carried separately.
- All four public generation routes enter the shared runtime.
- Native providers build from canonical requests rather than receiving arbitrary client payloads.
- Output protocol selection is independent and defaults sensibly.
- Config-defined providers can declare one of the four generative protocols.
- Same-protocol versus cross-protocol extension policy is directionally correct.
- Provider errors are generally normalized before success formatting.
- Generic field-cache and streaming-formatting foundations exist.
- GitHub CI is intentionally out of scope. I agree with the user's correction on that.

## Confirmed Defects

1. **Critical: Responses continuation loses encrypted reasoning.**  
   A Responses reasoning item containing `encrypted_content` is rebuilt with only its summary. Proxy-expanded history therefore breaks stateless reasoning continuity even for Responses to Responses execution.  
   References: `src/rotator_library/protocols/responses.py:324-355`, `src/rotator_library/responses/service.py:1286-1304`.

2. **Critical: Streaming is still normalized through public Chat SSE.**  
   Native provider events are parsed, formatted as Chat SSE, processed operationally, parsed again, and finally formatted for the selected client. Chat SSE cannot faithfully carry Responses item identities, multiple candidates, refusal blocks, annotations, or complete reasoning lifecycle data. The canonical event should remain authoritative through retries, accounting, and session handling.  
   References: `src/rotator_library/client/executor.py:760-765`, `src/rotator_library/native_provider/executor.py:228-280`, `src/rotator_library/protocols/streaming.py`.

3. **High: Alternatives are merged into one answer.**  
   Chat choices and Gemini candidates have no canonical candidate identity. Cross-protocol output concatenates alternatives. In a direct probe, `first` and `second` became one Anthropic response and both received the last choice's stop reason. Chat `n` and Gemini `candidateCount` are also dropped instead of mapped.  
   References: `src/rotator_library/protocols/types.py:376-416`, `src/rotator_library/protocols/canonical.py:539-549`.

4. **High: Instruction order changes.**  
   System and developer turns are extracted and moved before conversation turns. A direct same-protocol probe changed `user A -> developer B -> user C` into `developer B -> user A -> user C`. The callback merge path repeats this behavior.  
   References: `src/rotator_library/protocols/canonical.py:169-195`, `src/rotator_library/client/executor.py:810-843`.

5. **High: Required semantics can become empty successes.**  
   Responses built-in tool output, refusal blocks, annotations, and citations are not represented canonically. A web-search output converted to Chat became an empty assistant success. A Responses refusal also became an empty successful Chat response.  
   References: `src/rotator_library/protocols/responses.py:382-541`, `src/rotator_library/protocols/validation.py:155-165`.

6. **High: Same-protocol payloads are not reliably lossless.**  
   A standard nested Chat file part was rebuilt as `{"type": "file"}` with its file identity and filename removed. Responses built-in tools gain invented `name` and `parameters` fields. Same-protocol preservation therefore does not yet meet its own policy.

7. **High: Required reasoning controls are silently dropped.**  
   An Anthropic reasoning budget converted to Chat disappeared without even producing a conversion warning. Other reasoning enablement and visibility controls have similar gaps. Optional warning objects remain internal and are not recorded in a usable conversion summary.  
   References: `src/rotator_library/protocols/anthropic_messages.py:433-463`, `src/rotator_library/protocols/canonical.py:198-240`.

8. **High: Streaming block ordering is corrupted.**  
   Every text block uses `text:0` and every reasoning block uses `reasoning:0`. A direct `text A -> tool -> text B` probe produced one `AB` message item before the tool. Official `response.reasoning_summary_text.delta` events are parsed without any canonical delta.  
   References: `src/rotator_library/protocols/streaming.py:359-378`, `src/rotator_library/protocols/responses.py:216-246`.

9. **High: Client-supplied opaque state can be treated as provider-owned.**  
   `RequestContextBuilder` assigns the selected provider as `input_provider`, although the payload came from an external client. On a same-protocol first target, that identity can authorize client-supplied signatures as compatible provider state. Provider-state trust needs field-level provenance from the cache, not inference from routing order.  
   References: `src/rotator_library/client/request_builder.py:267-372`, `src/rotator_library/native_provider/executor.py:52-73`, `src/rotator_library/protocols/canonical.py:145-166`.

10. **High: Responses-service streaming failures can abort without a protocol error.**  
    Errors occurring after `StreamingResponse` begins escape the route's exception handling. Clients can receive a truncated connection instead of a Chat, Anthropic, Responses, or Gemini terminal error frame.  
    References: `src/rotator_library/responses/service.py:390-507`, `src/proxy_app/main.py:1187-1210`.

11. **Medium: Response adapters run on the client payload.**  
    Provider adapters are applied after output-protocol formatting, despite remaining configured in provider-protocol context. Provider-response adapters should run before provider response parsing; client-output adapters must be a separate explicit stage.  
    Reference: `src/rotator_library/native_provider/executor.py:93-118`.

12. **Medium: Configuration and shell boundaries need hardening.**  
    Field-cache replacement can duplicate names or disable protected behavior. The protocol correction also added 339 lines to `main.py`, mostly repeated selection and error behavior that belongs in `rotator_library`, contrary to the thin-shell rule.

## Out Of Scope Providers

Codex, Claude Code, Antigravity, Copilot, Gemini CLI, and every other concrete provider implementation are out of scope and are treated as unverified garbage scaffolding for this review.

- They are not readiness evidence.
- Their mocked "real provider contract" tests are not real provider validation.
- No provider file should be repaired or extended during protocol-core work.
- No new provider should be generated from assumptions like these were.
- Each provider must later be implemented manually from captured requests, responses, streaming frames, authentication behavior, and upstream documentation.
- Generic runtime tests must use synthetic providers.
- Gemini wire-protocol support remains in scope; the Gemini CLI provider does not.

The prior correction improperly changed four provider implementations and several provider-specific tests. I will not count any of that toward completion or touch those provider files in this packet.

## Where I Disagree With Reviewers

- I rejected the claim that Gemini `FileData.displayName` must be preserved. Official `FileData` contains `mimeType` and `fileUri`; `displayName` belongs to the separate File resource.
- I did not confirm the claimed in-memory append race in the supported single-event-loop path. There is no suspension between its read and write. Cross-thread use would be a separate contract question.
- I disagree with one heavy reviewer's dismissal of the `input_provider` problem. The source and direct signature-emission probe confirm that routing identity is currently being used as source provenance.
- Plexus is useful as a boundary reference, but not as the target design. It also collapses multiple choices and loses developer/system distinctions.

## Meaning Of "Any To Any"

The defensible product promise is:

> Any supported client protocol can target any supported provider protocol and request any supported output protocol. Every required meaning is either translated correctly or rejected explicitly. Nothing required is silently discarded or converted into an empty success.

Some wire features genuinely have no equivalent. For example:

- A required image or tool result must never be erased. Reject before provider transport if the provider language cannot express it.
- A provider-generated output item that the selected client language cannot express must produce a protocol-formatted conversion failure, not an empty response.
- Optional tuning hints such as an unsupported sampling control may be omitted, but only with one concise recorded conversion summary.

> Superseded by Decision 1 and Decision 7 below: the client response protocol equals the client request protocol, and conversion must exhaust the ordered best-effort hierarchy before any rejection.

- Same-protocol safe extensions should survive unchanged.
- Cross-protocol extensions must remain internal unless an explicit mapping exists.

## Grilling Decision Log

### Decision 1: Client request and response use one protocol

**Status:** Confirmed by the user.

The protocol used by the client to send a request is also the protocol the proxy uses to return the response:

```text
client request protocol = client response protocol
```

This is a hard contract for the current protocol runtime, not merely a default. Therefore:

- Remove `X-Proxy-Output-Protocol`.
- Remove independent `output_protocol_name` state.
- Remove provider `default_output_protocol` behavior and configuration.
- HTTP routes determine the client protocol in both directions.
- Direct library callers must explicitly declare their client protocol rather than inheriting the selected provider's protocol.
- Independent client-response protocol conversion is outside the current architecture. It may only return later as a separately designed advanced feature.

The core protocol model now has three layers:

```text
client protocol <-> neutral canonical <-> upstream protocol
```

The client protocol is bidirectional. The upstream protocol is also normally bidirectional. The neutral canonical representation is the interoperability substrate between them.

This decision supersedes the earlier three-independent-protocol framing in this document, including:

- The statement that a client may request any independently selected output protocol.
- The `4 input x 4 provider x 4 output` test-matrix requirement.
- Any correction-plan item that treats client output as independent from client input.

The replacement acceptance matrix is provisionally `4 client protocols x 4 upstream protocols`, with each cell covering the complete request and response round trip. Its final form will be settled by the remaining grilling decisions.

### Decision 2: One upstream protocol per target/profile

**Status:** Confirmed by the user.

Every configured upstream target is bound to exactly one upstream protocol at startup. Any supported client protocol may use that target, but every request sent to that target uses the target's declared upstream protocol.

Example:

```text
target:
  provider identity: example
  base URL: https://api.example.com
  upstream protocol: responses
  endpoint: /v1/responses
  credentials: example credential pool
```

The protocol is a property of the target's transport contract, not a per-request choice made by an ordinary caller. This keeps request finalization, response unwrapping, endpoint validation, and protocol conformance deterministic.

An ordinary provider or target therefore handles one upstream wire shape. It does not branch over Chat, Responses, Anthropic, and Gemini based on the originating client.

Future multi-protocol upstreams are represented as multiple transport profiles, for example:

```text
example/chat
example/responses
```

Those profiles may share provider-level identity and state, including credentials, quota, cooldowns, model discovery, and accounting. Each profile still declares exactly one upstream protocol, endpoint contract, and protocol-specific finalization path. Multi-profile selection is deferred until that feature is deliberately designed.

For configurable generic upstreams, the initial contract is likewise one selected protocol per configured target together with its base URL, endpoint mapping, authentication configuration, models, and other transport metadata.

### Decision 3: Each provider target declares the representation it receives

**Status:** Confirmed by the user.

Each provider target explicitly declares the representation its execution boundary expects to receive. The provider is never required to accept arbitrary client protocols.

The normal contract is a fixed upstream wire protocol:

```text
provider input representation = declared upstream protocol
```

For example, a target that declares `responses` receives a valid Responses payload regardless of whether the client used Chat, Anthropic, Gemini, or Responses:

```text
any client protocol
  -> client parser
  -> neutral canonical
  -> Responses builder
  -> provider receives Responses
```

The provider may then perform ordered operations that remain within its declared contract, including wire adaptation, envelope construction, authentication preparation, endpoint selection, and transport execution.

A provider may instead explicitly request the neutral canonical representation:

```text
provider input representation = neutral canonical
```

This supports unusual executors, SDKs, subprocess/RPC integrations, proprietary transports, or providers that deliberately want to own additional conversion steps. A neutral-input provider must return a neutral response, or use another explicitly declared and validated return boundary, so the core can still format the original client protocol correctly.

The representation choice is fixed and validated for the target/profile rather than inferred from the originating client. The default and recommended choice for an ordinary HTTP target is its one declared upstream wire protocol. Neutral input is an explicit provider execution contract, not an implicit fallback.

#### How Plexus does it

Plexus's active normal path parses the client payload into `UnifiedChatRequest`, selects the target API type, uses that target protocol's transformer to build the provider payload, and only then runs provider adapters and native/OAuth transport preparation. Its provider transport therefore normally receives the selected upstream wire protocol rather than the unified representation. Same-protocol targets may preserve the original raw payload.

Plexus previously contained internal-context/custom execution paths, but its active native provider path favors protocol-shaped payloads. This is the reference default, not a restriction on this proxy: this design preserves the same simple normal path while explicitly allowing a provider target to request Neutral when that is the better execution boundary.

### Review method: Plexus is the reference, not the target

For every remaining grilling or architecture question where Plexus has an applicable implementation, the review must include a `How Plexus does it` comparison before recommending a decision. Plexus is evidence that a boundary can work and a source of implementation tradeoffs; it is not authoritative, and its lossy or overly Chat-centric choices must not be copied automatically.

### Scope clarification: Dynamic generic providers extend the existing provider system

**Status:** Confirmed by source inspection and corrected by the user before a decision was recorded.

Generic upstream support does not introduce a new provider-storage, credential, or rotation architecture. It expands the existing dynamic provider mechanism represented by `src/rotator_library/providers/openai_compatible_provider.py` and `src/rotator_library/providers/__init__.py`.

The existing lifecycle is:

```text
<NAME>_API_BASE
  -> dynamically generated provider class registered as <name>
  -> ordinary provider discovery, routing, usage, cooldown, and rotation

<NAME>_API_KEY[_N]
  -> existing credential discovery
  -> ordinary credential pool for <name>
  -> per-attempt credential selection
```

The provider object does not own credential storage. It receives the selected credential through the same execution path used by premade providers. Multiple matching keys continue to rotate through the existing usage and credential-selection system.

The intended expansion is to let that dynamically generated provider additionally declare its upstream execution contract, including the selected protocol or Neutral representation, base URL, endpoint paths, authentication-header behavior, models, and applicable adapters/finalizers. It then behaves like any premade provider.

The following are explicitly outside this protocol work:

- redesigning credential discovery or storage
- replacing the existing `<NAME>_API_KEY[_N]` convention
- introducing a separate credential database or provider-record system
- changing usage, rotation, cooldown, or credential-pool semantics

Experimental already moved in this direction by allowing a config-named dynamic provider to select a protocol and native transport metadata while retaining the existing credentials. The independent client-output settings added alongside it are superseded by Decision 1.

#### How Plexus does it

Plexus stores provider records and encrypted credentials in its database. That storage model is not applicable here. The useful reference behavior is narrower: a Plexus target declares an API type, the generic transformer builds that protocol, and provider transport receives the resulting wire payload. This proxy retains its existing dynamic class generation and credential rotation instead of copying Plexus's provider-record model.

### Decision 4: Raw-preserving same-protocol fast path

**Status:** Confirmed by the user.

When the client protocol and a target's declared provider-input/upstream wire protocol match, the runtime preserves the original wire payload as the basis of the upstream request rather than reconstructing it from Neutral by default.

```text
matching client and upstream protocol
  -> parse a Neutral sidecar for proxy concerns
  -> preserve the original request payload
  -> apply required provider finalization
  -> send in the same protocol
```

The Neutral sidecar still supports routing, credential selection, session inference, validation, accounting, tracing, callbacks, and provider-state handling. The fast path skips loss-prone wire reconstruction; it does not make the request invisible to the proxy.

The runtime may intentionally strip or replace fields when required by an explicit rule, including security policy, target capability, proxy-owned continuation expansion, a canonical semantic edit, or a declared provider adapter/finalizer. Such changes must be deliberate and traceable. Unknown or unmodeled fields must not disappear merely because the runtime parsed and rebuilt a same-protocol payload.

If a semantic edit cannot be safely overlaid on the raw payload, the runtime may rebuild the affected structure from Neutral, but it must preserve unrelated source-native fields and prove the resulting same-protocol behavior through fidelity tests.

The corresponding same-protocol response path should preserve the provider's raw response when the client expects that same protocol and no explicit response adapter or proxy-owned semantic change requires reconstruction. Usage, errors, session evidence, and provider state may still be observed or extracted without changing the client payload.

This fast path applies to targets that request a wire-protocol representation. A target that explicitly requests Neutral has chosen a different execution boundary and does not receive raw protocol passthrough.

#### How Plexus does it

Plexus forwards the original request body when incoming and target API types match, subject to specific carve-outs. It applies routing metadata and target adapters without passing through its unified request builder. On return, it forwards the raw provider response or stream when the client and target formats match. This design adopts the fidelity benefit while requiring intentional stripping and canonical overlays to remain explicit and testable.

### Decision 5: Symmetric provider execution representation for v1

**Status:** Confirmed by the user.

Each target/profile uses one symmetric provider execution representation for both request and response boundaries in the v1 protocol runtime:

```text
provider receives Responses -> provider returns Responses
provider receives Anthropic -> provider returns Anthropic
provider receives Chat -> provider returns Chat
provider receives Gemini -> provider returns Gemini
provider receives Neutral -> provider returns Neutral
```

Provider-specific envelopes, wire adapters, authentication, SDK calls, or internal conversion steps may exist inside execution, but the provider must return to its declared representation at the runtime boundary. For a declared wire protocol, the generic runtime owns that protocol's request builder and response parser. For a declared Neutral representation, the provider consumes and returns neutral types.

Request and response representations are not independently configurable in v1. This keeps parsing, formatting, validation, tracing, field-cache handling, streaming, and same-protocol passthrough aligned around one target contract.

Asymmetric execution is explicitly planned as a post-v1 nice-to-have:

```text
request representation != response representation
```

It is not part of this correction plan or its acceptance criteria. It may be designed after v1 only when a verified integration demonstrates a real need. The current symmetric contract must not grow speculative compatibility branches for it.

#### How Plexus does it

Plexus uses one target API type symmetrically: its target transformer builds that protocol's request and parses that protocol's response. Provider envelopes and OAuth preparation wrap the selected API type rather than creating independently declared request and response formats. This design follows that simple boundary for v1 while documenting asymmetry as a deliberate future extension rather than claiming it can never exist.

### Derived rule: Streaming follows the same conversion boundary

**Status:** Confirmed by the user as a direct consequence of the existing decisions, not a separate product choice.

Streaming follows the same rule as non-streaming traffic:

```text
conversion required
  -> provider stream parser
  -> neutral canonical stream events
  -> client stream formatter

matching wire protocols and no required transformation
  -> raw-preserving same-protocol stream path
```

Neutral is the safe interoperability medium whenever conversion is required. The operational stream layer must not introduce an intermediate public protocol such as Chat SSE between the provider parser and client formatter.

Retry visibility, structured errors, usage, cost, session completion, field-cache extraction, cancellation, and timeout handling must observe the Neutral event flow without forcing an additional wire-format round trip. A Neutral-input provider returns Neutral stream events directly.

The Neutral stream model must be rich enough to preserve candidate, item, block, tool, reasoning, refusal, citation, usage, completion, error, and opaque-state lifecycles. This is an implementation-completeness requirement derived from the lossless-conversion contract.

#### How Plexus does it

Plexus uses the same topology for cross-protocol streams: provider stream transformer to unified chunks to client stream formatter, with raw bypass for matching protocols. Its unified stream type is too Chat-centric for the full fidelity required here, so Plexus is the structural reference while this runtime requires a richer Neutral event model.

### Decision 7: Ordered best-effort conversion hierarchy

**Status:** Confirmed by the user.

When protocols express the same intent differently, conversion must exhaust an ordered hierarchy before any rejection:

```text
1. Exact field/block mapping
2. Equivalent destination construct
3. Deterministic inference or coercion
4. Proxy-managed emulation
5. Honest protocol-valid fallback representation
6. Explicit error only if required meaning still cannot be represented
```

Examples:

| Source meaning | Possible destination handling |
|---|---|
| Responses `instructions` | Anthropic `system`, Gemini `systemInstruction`, Chat system/developer message |
| Reasoning budget | Destination effort level or thinking budget using a documented conversion table |
| Web-search tool | Responses web search, Anthropic server tool, Gemini Google Search |
| Citations | Destination annotations, citation blocks, grounding metadata, or a faithful text citation fallback |
| Refusal | Native refusal block, safety status, or explicit refusal content plus normalized stop reason |
| `previous_response_id` | Proxy expands stored continuation into canonical history |
| Provider thought signature | Cache internally and restore only to the compatible provider |
| Unsupported optional sampling hint | Omit with one trace summary |
| Required modality with no representation | Return a protocol-formatted error before transport |

Every convertible semantic must be classified as exactly one of:

- **Required:** must convert, emulate, or fail explicitly. It must never be silently dropped or become an empty success.
- **Optional tuning:** may be omitted, but only with one concise recorded conversion summary.
- **Opaque provider state:** kept internal with provider/credential/session provenance and restored only to a compatible target.
- **Unknown extension:** preserved only on a safe same-protocol path; remains internal on cross-protocol routes unless an explicit mapping exists.

Rejection is the final outcome only for genuinely unrepresentable required meaning. Invented semantics, fabricated fields, and silent loss are never acceptable at any level of the hierarchy.

#### How Plexus does it

Plexus already mixes exact transformer mappings, inference and coercion, provider/model auto-compatibility, web-search adapters, and same-protocol passthrough. But when its unified model lacks a field, Plexus often silently drops it — multiple choices, refusals, some sampling controls, audio/video, and some citation directions. That silent-loss behavior must not be copied: this hierarchy makes best-effort conversion mandatory and silent loss a defect.

### Decision 8: Opaque provider-state provenance

**Status:** Confirmed by the user.

Opaque provider state — thinking signatures, encrypted reasoning, thought signatures, cache handles, provider response IDs — is exclusive to the provider that produced it in almost all cases. Forwarding it to a different provider produces invalid requests, not interoperability. Same-protocol transport is therefore the correct and best-case path for this class of data.

Two-tier rule:

```text
Runtime-injected opaque state:
  sourced only from the scoped field cache, with recorded
  origin provider + credential + session provenance,
  and restored only when returning to a compatible provider context

Client-supplied opaque state:
  forwarded verbatim only on the same-protocol raw path (Decision 4),
  never rewritten, never injected into a rebuilt cross-protocol payload,
  never treated as compatible with a different provider
```

Routing identity is never a valid provenance source. The current `input_provider` behavior, which labels client input as provider-originated based on selection order, is a defect and is void under this decision.

On cross-protocol routes where an opaque field has no destination representation, the field is not translated into foreign payload content. The runtime may extract it into the scoped field cache (with provenance) so a later request routed back to the compatible provider can restore it; otherwise it simply does not cross the conversion boundary. Per Decision 7, opaque provider state is its own semantic class, never a silent-loss defect.

#### How Plexus does it

Plexus keeps signatures alive only through same-protocol raw passthrough and drops them on cross-protocol conversion — no cache, no reinjection, no cross-provider leakage, but also no continuity after a protocol switch. This design matches the passthrough-first behavior (Decision 4) and adds what Plexus lacks: provenance-scoped caching so continuity survives rotation and later returns to the compatible provider.

### Decision 9: Multi-candidate policy — first candidate wins

**Status:** Confirmed by the user; revised 2026-09-05 from concatenation to first-candidate-wins.

Multi-candidate requests (`n` in OpenAI Chat, `candidateCount` in Gemini — the same prompt answered N independent times in one call) are a rarely used feature, mainly associated with code-completion/FIM workflows. Defaults are 1 and most models support only 1. This policy exists so the rare case degrades deterministically:

```text
Both sides support multiplicity
  -> direct mapping (Chat n <-> Gemini candidateCount), full fidelity

Request asks for multiplicity the upstream cannot produce
  -> send the maximum the upstream supports,
     return the response it actually produced,
     record one conversion summary per Decision 7

Upstream returns multiple candidates, client protocol carries one
  -> return the FIRST candidate, drop the rest,
     record one conversion summary
```

Concatenation of alternatives was considered and explicitly rejected by the user. Dropping beyond-first candidates is not silent loss within this policy: it is a mandated deterministic degradation with a recorded summary. The Neutral model still tracks candidate identity and per-candidate stop status internally so the first candidate is selected deterministically and its own stop status is reported.

#### How Plexus does it

Plexus keeps only `choices[0]` and drops the remaining alternatives silently; `n > 1` is lost with no summary. This policy matches the first-wins shape but adds the recorded summary and preserves multiplicity through direct `n`/`candidateCount` mapping where both sides support it.

### Grilling closure (2026-09-05)

The grilling decision tree is complete. Decisions 1-9 above are the authoritative product contract. Remaining items (adapter staging details, error framing, instruction ordering, Neutral model richness) are implementation requirements derived from these decisions, not open product choices.

**Acceptance bar:** the 4 client x 4 upstream round-trip matrix. Each cell: non-streaming and streaming, the same-protocol raw fast path where client protocol equals upstream protocol, required semantics preserved or explicitly rejected per Decision 7, multi-candidate degradation per Decision 9, and errors protocol-formatted at every stage including mid-stream. All cells driven by synthetic providers only.

The matrix is tests, not architecture: 16 test scenarios over 4 single-file protocol adapters plus one neutral core. Each protocol lives in one adapter file containing both directions (`parse_request`/`format_response` for its client role, `build_request`/`parse_response` for its upstream role). "Client protocol" and "upstream protocol" are roles at request time, not split implementations — nobody implements half a protocol, and there is exactly one file to inspect when a protocol misbehaves.

After closure, this review and the phase documents are synthesized into the single authoritative final plan (`docs/experimental/00-final-plan.md`), which supersedes them all; superseded documents move to `docs/experimental/_superseded/`.

## Correction Plan

1. **Reopen the review ledger.**  
   Change the durable protocol review from beta-ready to reopened, record these reproductions, remove concrete providers from evidence, and preserve the no-CI decision.

2. **Make the canonical model genuinely lossless.**  
   Add ordered instruction placement, candidate identity, per-candidate stop status, refusal, annotations/citations, built-in tool categories, output modality identity, and opaque-state provenance. Keep provider state separate from visible reasoning.

3. **Repair same-protocol fidelity first.**  
   Implement raw-payload overlays gated strictly to the same protocol. Preserve Responses encrypted reasoning, standard Chat files, native tools, statuses, and extension fields exactly while allowing canonical edits to win.

4. **Complete cross-protocol non-streaming conversion.**  
   Map known equivalents including `n` and `candidateCount`, stop/refusal status, tools, media, structured output, and reasoning controls. Add request and response capability validation. Reject unrepresentable required semantics.

5. **Correct runtime trust and adapter boundaries.**  
   Stop labeling external input as provider-originated. Give cached opaque fields explicit trusted provenance. Run provider-response adapters before provider parsing, keep client-output adapters separate, preserve callback ordering, and harden field-cache override rules.

6. **Replace the Chat streaming seam.**  
   Normalize each provider frame once into canonical events. Run retry visibility, usage, cost, timeout, cancellation, session completion, and fallback logic over those events. Format the selected output only afterward.

7. **Finish the stream lifecycle.**  
   Track candidate, output-item, content-block, and tool identities independently. Parse official reasoning, refusal, annotation, built-in-tool, usage, completion, incomplete, and failure events. Ensure every post-start failure emits a valid selected-protocol terminal frame.

8. **Restore the thin shell.**  
   Move repeated protocol selection, error classification, and terminal stream behavior out of `src/proxy_app/main.py`. Route handlers should authenticate, parse, delegate, and construct the HTTP response only.

9. **Replace shallow evidence with contract tests.**  
   Add complete same-protocol fidelity fixtures and rich runtime matrices. Every cell should assert either valid semantic preservation or the expected explicit rejection. Add an equivalent streaming matrix using synthetic providers only.

10. **Verify and review incrementally.**  
    Commit after each meaningful verified group. Run focused tests, the complete local suite, compile/import checks, and `git diff --check`. Then reuse the base `explore` review for mapping and the existing heavy reviewer only for the final critical judgment. No blocker, high, or medium finding may remain before restoring the beta-ready verdict.

> Superseded by the decision log: correction-plan items must be re-stated in the synthesized comprehensive plan under the clarified three-layer model (`client protocol <-> neutral canonical <-> upstream protocol`), the one-upstream-protocol-per-target contract, the target-declared execution representation, the raw-preserving same-protocol fast path, symmetric v1 execution, and the ordered conversion hierarchy of Decision 7. The independent-output aspects of steps 6, 7, and 9 are void per Decision 1; the `4 x 4 x 4` matrix becomes `4 client x 4 upstream`.

## Verification State (2026-07-17 review pass)

No files were modified during this second-opinion pass. I ran direct read-only protocol probes and `git diff --check`; I did not rerun the complete pytest suite in plan mode. The existing dirty `ARCHITECTURE.md`, `STRUCTURE.md`, and session artifacts were untouched.
