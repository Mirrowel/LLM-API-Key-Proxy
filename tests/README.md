# SPDX-License-Identifier: MIT
# Copyright (c) 2026 b3nw
#
# LLM-API-Key-Proxy Test Suite

## Design Philosophy

This test suite is designed to be **fully runnable locally without any real API keys or provider connections**. We achieve this through:

1. **Mocked HTTP layer** — `httpx.AsyncClient` is mocked so no real outbound requests are made
2. **Synthetic credentials** — In-memory credential files and env vars, never touching real keys
3. **FastAPI TestClient** — Full proxy app is tested end-to-end using `httpx.AsyncClient` against the ASGI app
4. **Deterministic fixtures** — All test data is self-contained in `conftest.py`

### Cost-Safe Guarantees

- ✅ **Zero cost** — No queries are ever sent to real LLM providers
- ✅ **No API keys needed** — Tests use fake/synthetic credentials
- ✅ **No OAuth flows** — OAuth token refresh is mocked; no browser interaction needed
- ✅ **No network access** — All HTTP calls are intercepted at the `httpx`/`litellm` level
- ✅ **Runs in <30s** — Fast enough to run before every commit

## Test Categories

### 1. Unit Tests — Pure Logic (no I/O, no network)

| Test Module | What It Covers | Why It Matters |
|---|---|---|
| `test_anthropic_translator.py` | Anthropic↔OpenAI format translation | Breakage here silently corrupts all Claude Code requests |
| `test_anthropic_streaming.py` | SSE format conversion (OpenAI→Anthropic events) | Streaming breakage is hard to detect in production |
| `test_error_handler.py` | Error classification, duration parsing | Misclassified errors cause wrong retry/rotation behavior |
| `test_request_sanitizer.py` | Parameter stripping (dimensions, thinking) | Invalid params cause 400s from providers |
| `test_provider_transforms.py` | Per-provider request mutations | Transform regressions silently break specific providers |
| `test_model_filters.py` | Whitelist/blacklist model filtering | Wrong filter = missing or extra models exposed |
| `test_usage_tracking.py` | Window tracking, quota groups, custom caps | Usage bugs cause over/under-use of credentials |
| `test_credential_filter.py` | Tier-based credential filtering | Wrong tier = requests sent to incompatible credentials |
| `test_model_alias.py` | MODEL_ALIAS env parsing, alias resolution | Alias breakage = cross-provider routing fails |
| `test_model_latest_registry.py` | Glob matching, semver sorting, suffix stripping | "latest" alias sends to wrong model version |

### 2. Integration Tests — Component Interaction (mocked HTTP)

| Test Module | What It Covers | Why It Matters |
|---|---|---|
| `test_rotating_client.py` | Key acquisition, rotation, retry, cooldown | The core orchestration — re-organization broke this before |
| `test_cross_provider.py` | Multi-provider failover via aliases | Cross-provider routing is a complex new feature |
| `test_proxy_endpoints.py` | FastAPI endpoint routing & auth | Endpoint breakage = 404/401 for all clients |
| `test_credential_manager.py` | Discovery, dedup, env-based creds | Credential loading bugs = zero providers available |
| `test_background_refresher.py` | OAuth refresh scheduling | Stale tokens = auth failures in production |
| `test_provider_singleton.py` | Singleton metaclass for providers | Multiple instances = split caches, inconsistent state |

### 3. Branch-Specific Regression Tests

| Test Module | What It Covers | Why It Matters |
|---|---|---|
| `test_anthropic_compat_e2e.py` | Full Anthropic Messages API round-trip | Each branch modifies translator/streaming differently |
| `test_provider_plugins.py` | All provider plugin registration & init | Branch-specific providers can fail to register |
| `test_usage_window_modes.py` | per_model vs credential vs daily reset modes | Different branches alter usage tracking config |

## Running

```bash
# All tests
pytest tests/ -v

# Just unit tests (fast, <5s)
pytest tests/ -v -m unit

# Just integration tests
pytest tests/ -v -m integration

# Specific module
pytest tests/ test_anthropic_translator.py -v
```
