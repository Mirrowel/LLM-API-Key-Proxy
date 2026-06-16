# Routing, Retry, Cooldown, Usage, And Cost Roadmap

These systems should be layered around the protocol/provider work without replacing the existing credential engine.

## Routing Direction

The preferred first routing model is fallback chains, not a full target-group router from the external reference gateway.

Example:

```json
{
  "fallback_groups": [
    ["gemini_cli/gemini-2.5-pro", "openrouter/google/gemini-2.5-pro", "openai/gpt-4.1"]
  ]
}
```

Behavior:

- If the requested model is in a group, start with the requested model and then continue through the rest of the group.
- Retryable provider/model failures can move to the next candidate.
- Non-retryable request errors stop unless explicitly configured otherwise.
- Each candidate delegates to the current credential rotation engine.
- Session tracking and classifier scopes must remain isolated.

## Target Groups Later

Reference target groups are still useful later for selectors:

- `in_order`
- `random`
- `usage`
- `cost`
- `latency`
- `performance`

But the first implementation should solve ordered fallback groups and the missing/stale fallback module before adding selector complexity.

## Retry/Cooldown Direction

Current provider cooldown is effectively inert because `CooldownManager.start_cooldown()` is not called. The real active cooldowns are per-credential in `UsageManager`.

Upgrade direction:

- keep per-credential cooldowns in `UsageManager`.
- add provider/model cooldown for provider-wide or model-wide failures.
- add consecutive failure tracking.
- add exponential backoff.
- `retry_after` should override computed backoff.
- success should clear provider/model failure count.
- credential quota exhaustion should not globally cool down a provider unless evidence says the provider/model is globally exhausted.

## Retry History

Add structured attempt records:

- candidate provider/model.
- credential stable ID or masked identity.
- protocol path used.
- status: success, failed, skipped.
- error type.
- retryable decision.
- cooldown decision.
- timing.

These records should be available to transaction logging and optionally client-facing debug output later.

## Usage And Cost Direction

Keep existing windowing, fair cycle, custom caps, quota groups, and usage persistence. Add protocol-aware normalization before usage is recorded.

Needed normalizers:

- OpenAI Chat.
- OpenAI Responses.
- Anthropic Messages.
- Gemini.
- OAuth/custom providers.

Fields to preserve:

- input tokens.
- output tokens.
- reasoning/thinking tokens.
- cache read tokens.
- cache write tokens.
- total tokens.
- provider-reported cost.
- estimated cost.
- cost source and metadata.

## Cost Direction

Provider-reported cost should win when available. Sources include:

- `usage.cost_details`.
- provider-specific response fields.
- SSE comment lines such as `: cost { ... }`.

Estimated cost should remain as fallback.

## Tests

Required tests:

- fallback chain success after first failure.
- non-retryable error stops fallback.
- streaming fallback only before visible output.
- provider/model cooldown scope.
- backoff escalation.
- success reset.
- usage normalization for each protocol.
- provider-reported cost precedence.
