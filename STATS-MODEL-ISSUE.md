# Issue: Hardcoded Background Model Probes in Claude Code

## Description
When using **Claude Code** (v2.1.2+) with the LLM-API-Key-Proxy, the client performs background "probes" for specific model names that are not explicitly requested by the user in the `--model` flag.

These probes appear in the proxy logs as requests for models such as:
- `claude-haiku-4-5-20251001`
- `gemini-claude-opus-4-5-thinking`
- `claude-3-5-sonnet-20241022`

## Root Cause
These model names are hardcoded into the Claude Code binary for two primary purposes:
1. **Capability Discovery**: Checking if the endpoint supports specific beta features (e.g., `interleaved-thinking`, `computer-use`).
2. **Task Scaling**: Attempting to use a cheaper "Haiku-class" model for background indexing or file scanning tasks to optimize performance and cost.

Because these requests are generated internally by the client, they lack the `antigravity/` provider prefix and use futuristic version strings that do not exist in the proxy's default mapping tables, resulting in `404 Not Found` errors from the backend.

## Impact
- **Endless Thinking**: Claude Code may hang or show a "thinking" spinner indefinitely while it retries these failed background probes.
- **Log Noise**: Proxy logs are cluttered with 400/404 errors for models that are not configured.
- **Session Instability**: Critical background tasks like code indexing may fail to complete.

## Recommended Workaround
Since this behavior is hardcoded in the client and cannot be disabled via `--no-stats` or environment variables, the recommended solution is to provide a "safety net" mapping in the Proxy's configuration.

Add the following to your `LLM-API-Key-Proxy/.env` file to funnel these probes into the models you actually have credentials for:

```env
# Map prefix-less background probes to the Antigravity provider
ANTIGRAVITY_MODELS='{
  "claude-haiku-4-5-20251001": {"id": "claude-sonnet-4-5"},
  "gemini-claude-opus-4-5-thinking": {"id": "claude-opus-4-5"},
  "claude-3-5-sonnet-20241022": {"id": "claude-sonnet-4-5"}
}'
```

## Status
- **Client**: Claude Code v2.1.2+
- **Proxy Version**: PR #47 (Anthropic Compatibility)
- **Status**: Identified as client-side behavior; requires configuration-level workaround for production stability.