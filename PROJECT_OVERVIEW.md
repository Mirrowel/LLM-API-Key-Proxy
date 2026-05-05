# LLM API Key Proxy - Project Overview

## What this project is trying to accomplish

LLM API Key Proxy is a self-hosted gateway that gives teams a single, OpenAI-compatible endpoint for many model providers.

Instead of wiring every client directly to OpenAI, Anthropic, Gemini, OpenRouter, and others, apps call this proxy at `/v1/*`.
The proxy then routes requests to the right upstream provider while keeping a consistent API surface.

The current direction is to make this useful for real multi-user teams, not just single-user local setups.

## Core goals

1. **Unified API surface**
   - Expose familiar OpenAI-style endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`, etc.).
   - Support Anthropic-compatible endpoints (`/v1/messages`, `/v1/messages/count_tokens`).

2. **Provider abstraction + routing**
   - Route `provider/model` requests to the correct backend.
   - Keep client-side integrations simple while supporting multiple providers behind the scenes.

3. **Multi-user access control**
   - Allow admin-managed users (no self-signup).
   - Let users create/revoke personal API keys.
   - Support transition modes via `AUTH_MODE=users|legacy|both` for migration from a single legacy proxy key.

4. **Usage accounting and visibility**
   - Attribute request usage to user/API key.
   - Provide user and admin usage summaries, breakdowns, and dashboards.

5. **Safe deployment defaults**
   - Secure session cookies and CSRF protection for UI forms.
   - Hash-only API token storage.
   - Stricter CORS and secret handling for production mode.
   - SQLite reliability hardening and usage retention controls.

## Who this is for

- Teams that want centralized control of LLM access.
- Developers who need one endpoint for many providers.
- Admins who need per-user key lifecycle management and usage reporting.

## What success looks like

- Clients can switch providers without rewriting auth/routing logic.
- Admins can onboard users safely and audit usage.
- Existing OpenAI/Anthropic client compatibility remains stable.
- The proxy can run reliably in development and production with clear security expectations.
