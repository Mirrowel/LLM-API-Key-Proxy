# Provider Roadmap

Providers follow protocols. The protocol layer must land first so provider work can be small, testable, and declarative where possible.

## Provider Declaration Target

A provider should eventually be expressible as:

```text
provider name
  + protocol(s)
  + auth strategy
  + model definitions/options
  + adapter chain
  + field cache rules
  + quota checker/parser
  + optional protocol/provider overrides
```

Providers can still have custom Python code. The point is to make custom code narrow.

## Priority Providers

1. Claude Code.
2. Codex.
3. Copilot.
4. Antigravity.
5. Gemini CLI review and parity improvements.

## Claude Code

Review the external reference gateway for:

- OAuth/token handling.
- request/response protocol shape.
- tool filtering or tool proxy behavior.
- quota checks.
- stream behavior.
- Claude Code-specific headers and model naming.

Expected implementation direction:

- use Anthropic Messages or Responses where applicable.
- add provider-specific adapters for tool behavior.
- field cache rules for thinking/signatures if needed.

## Codex

Review the external reference gateway for:

- Responses API route usage.
- Codex-specific user-agent/version behavior.
- OAuth/account handling.
- cooldown parsing for Codex usage limits.
- stream events.

Expected implementation direction:

- build on the Responses protocol.
- add Codex provider auth and headers.
- include version/user-agent support if needed.

## Copilot

Review the external reference gateway for:

- GitHub Copilot OAuth flows.
- endpoint selection.
- model naming.
- quota checker behavior.
- provider-specific request filtering.

Expected implementation direction:

- use protocol adapters where possible.
- add provider-specific auth/token refresh only where necessary.

## Antigravity

Required comparison:

- current retired implementation under `src/rotator_library/providers/_retired/`.
- external reference Antigravity provider/checker/OAuth behavior.

Expected implementation direction:

- restore only what is still valid.
- reuse protocol and field cache rules.
- avoid resurrecting obsolete device-profile behavior unless clearly required.

## Gemini CLI

The current Gemini CLI provider is already deep. Review the external reference gateway only for missed behavior:

- quota checker details.
- thought signature handling.
- stream transform differences.
- OAuth edge cases.
- Gemini 3 tool behavior.
- request headers and endpoint details.

Do not rewrite Gemini CLI just for architectural purity.

## Provider Tests

Each provider should have tests for:

- config/load/registration.
- auth header or token acquisition.
- request translation.
- response translation.
- stream translation.
- quota parser/checker behavior.
- field cache extraction/injection.
- LiteLLM fallback not used when native path should apply.
