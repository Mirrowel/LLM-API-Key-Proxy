# Issue: Hardcoded Background Model Probes in Claude Code

## Description
When using **Claude Code** (v2.1.2+) with the LLM-API-Key-Proxy, the client performs background "probes" for specific model names that are not explicitly requested by the user in the `--model` flag.

These probes appear in the proxy logs as requests for models such as:
- `claude-haiku-4-5-20251001`
- `claude-opus-4-5-20251101`
- `claude-3-5-sonnet-20241022`

## Root Cause
These model names are hardcoded into the Claude Code binary for two primary purposes:
1. **Capability Discovery**: Checking if the endpoint supports specific beta features (e.g., `interleaved-thinking`, `computer-use`).
2. **Task Scaling**: Attempting to use a cheaper "Haiku-class" model for background indexing or file scanning tasks to optimize performance and cost.

Because these requests are generated internally by the client, they lack the `antigravity/` provider prefix and use versioned model strings that may not exist in the proxy's default mapping tables.

## Impact (Prior to Fix)
- **Endless Thinking**: Claude Code may hang or show a "thinking" spinner indefinitely while it retries these failed background probes.
- **Log Noise**: Proxy logs are cluttered with 400/404 errors for models that are not configured.
- **Session Instability**: Critical background tasks like code indexing may fail to complete.

## Resolution
The proxy now includes built-in pattern matching to handle these probes automatically:

1. **Provider Fallback** (`client.py`): The `_resolve_provider_fallback()` helper automatically routes requests containing "claude" or "gemini" (without a provider prefix) to the `antigravity` provider.

2. **Pattern-Based Model Mapping** (`antigravity_provider.py`): The `_alias_to_internal()` method uses substring matching to map versioned model names to internal equivalents:
   - Models containing `opus` → `claude-opus-4-5`
   - Models containing `sonnet` → `claude-sonnet-4-5`
   - Models containing `haiku` → `gemini-3-flash`
   - Models containing `gemini-3` → appropriate Gemini variant

## Manual Override (Optional)
For edge cases not covered by the built-in patterns, you can add explicit mappings to your `.env` file:

```env
# Map specific model probes to known models
ANTIGRAVITY_MODELS='{
  "claude-3-5-sonnet-20241022": {"id": "claude-sonnet-4-5"}
}'
```

## Status
- **Client**: Claude Code v2.1.2+
- **Proxy Version**: PR #47 (Anthropic Compatibility)
- **Status**: Resolved via built-in pattern matching; manual config only needed for edge cases.
