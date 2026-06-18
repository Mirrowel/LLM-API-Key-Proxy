# xAI provider

Canonical feature ID: `xai`
Stack subjects:
- `feat(xai): add xAI Grok OAuth provider with PKCE and Device Code flows`
- `feat(xai): enable xAI Grok device-code OAuth in admin WebUI`
Manifest: `.fork/stack.yml`

This file is the shared, repo-tracked history for xAI feature changes.
Local workspace state may contain run logs and scratch notes, but this file is
canonical across contributors and developer workspaces.

## 2026-06-19 — Fix OAuth credential hot-load, replacement, and cleanup on delete

Target: `feat(xai): enable xAI Grok device-code OAuth in admin WebUI`
Files:
- `src/proxy_app/api/oauth.py`
- `src/proxy_app/api/config.py`
- `src/proxy_app/main.py`

Working commits before autosquash:
- `eab0cebe fixup! feat(xai): enable xAI Grok device-code OAuth in admin WebUI`

Final stack commit after autosquash:
- `e10f4229 feat(xai): enable xAI Grok device-code OAuth in admin WebUI`

Verification:
- `uv run python3 -m py_compile src/proxy_app/api/oauth.py src/proxy_app/api/config.py src/proxy_app/main.py` — passed
- `uv run ruff check src/proxy_app/api/oauth.py src/proxy_app/api/config.py src/proxy_app/main.py --select F401,F811,F821,E9` — passed
- Hot-patched to live `llm-proxy` container on `docker-test`, restarted via Komodo, verified healthy with x-ai provider active (10 models, 1 credential)
- Browser smoke-tested credentials page at `https://llm-proxy.ext.ben.io/ui/credentials`

Notes:
- Three issues fixed in this changeset:
  1. **Bug: no hot-load on OAuth add** — `_save_credential_file()` wrote the JSON
     but never updated the running `RotatingClient`. New credentials only became
     active after a full restart. Added `_hot_load_credential()` to register the
     credential in `all_credentials`, `oauth_credentials`, `oauth_providers`, and
     re-initialize the usage manager immediately.
  2. **Bug: empty provider key after last credential deleted** — The delete endpoint
     filtered credentials out but left `all_credentials["x-ai"] = []`, which caused
     `model_discovery` to return `[]` and `background_refresher` to skip the
     provider. Now `del`s the empty key and discards from `oauth_providers`.
  3. **Feature: credential replacement** — Previously, adding a credential for the
     same account always created `_N+1.json`. Now `_find_existing_credential()`
     matches by email/login/account_id and overwrites in-place (with timestamped
     `.bak` backup), preventing duplicate files for the same identity.
- `set_app_ref(app)` added to `main.py` lifespan so background OAuth poll tasks
  (which lack a `Request` object) can access the `RotatingClient` for hot-loading.
