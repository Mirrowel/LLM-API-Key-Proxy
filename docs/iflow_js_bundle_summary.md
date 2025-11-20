**iFlow JS Bundle Summary**

Path: `external/iflow-cli-tarball/package/bundle/iflow.js`

Purpose
- A single-file bundled/minified Node/Browser CLI distribution for the iFlow CLI.
- Contains the CLI runtime, UI strings, crypto helpers, networking logic and an OAuth-based authentication flow used by iFlow's tooling.

Bundle metadata
- File size (approx): ~10.2 MB (minified/bundled, many libraries included).
- Other files in same folder: `iflow-cli-vscode-ide-companion-*.vsix`, `sandbox-*.sb` files (IDE companions / sandbox artifacts).

High-level contents and useful targets
- Crypto helpers
  - Implements both browser and Node crypto adapters (named `BrowserCrypto` and `NodeCrypto`).
  - Exposed helpers include: `sha256DigestHex`, `sha256DigestBase64`, random bytes generator, HMAC signing helpers, and base64 helpers.
  - These utilities are used to compute PKCE code_challenge values (S256) and other digests.

- PKCE evidence
  - The bundle includes code that computes SHA-256 digests and derives hex/base64 strings. This is strong evidence PKCE is used by the CLI.
  - Keywords to look for: `sha256DigestHex`, `code_challenge`, `code_verifier`, `code_challenge_method`, `PKCE`, `S256`.

- OAuth + Browser flow
  - The CLI appears to use a browser-based authorization flow:
    - It constructs an authorization URL (authUrl) and attempts to open it in the user's browser.
    - It starts or expects a local callback path (the pattern `/oauth2callback` appears in the bundle).
    - After the browser redirects to the local callback, the bundle's callback handler extracts `code` and `state` and resolves them (pattern seen as resolving an object like `{code: c, state: u}`).
  - User-facing strings found in the bundle include messages and instructions such as:
    - "Attempting to open authentication page in your browser."
    - "Navigate to the URL below to sign in" (or similar localized messages).

- Token exchange and API-key retrieval
  - The bundle contains logic that exchanges an authorization code for tokens and then retrieves user info to extract provider-specific `apiKey` (iFlow uses a separate API key for API calls).
  - Token exchange patterns include POSTing to token endpoints and handling success/error JSON responses.
  - Note: In some CLI flows, both PKCE and client-secret based exchanges may be present; inspect the exact token-request payload to confirm which parameters are sent.

- Additional artifacts bundled
  - Localization strings, telemetry/logging code, UI wrappers, and other third-party libraries are bundled together, making raw inspection noisy.

Practical guidance for other agents
- Why it's tricky
  - `iflow.js` is minified/compiled and contains many third-party modules concatenated together. This makes it hard to read directly—searching for specific keywords is the fastest approach.

- Key search terms (recommended)
  - `sha256`, `sha256DigestHex`, `code_challenge`, `code_verifier`, `code_challenge_method`, `authUrl`, `oauth2callback`, `Attempting to open`, `authUrl`, `authorize`, `token`, `getUserInfo`, `apiKey`, `redirect`, `state`.

- Useful PowerShell commands (run at repo root) to inspect the bundle quickly
  - Search for keywords:
```powershell
Select-String -Path external\iflow-cli-tarball\package\bundle\iflow.js -Pattern "sha256|code_challenge|authUrl|oauth2callback|getUserInfo|apiKey|Attempting to open" -SimpleMatch
```
  - View a nearby window of lines (example indices found via Select-String):
```powershell
# read lines 940..1005 (example offsets)
(Get-Content -Path 'external\iflow-cli-tarball\package\bundle\iflow.js')[940..1005] -join "`n"
```

- How to extract provider-specific values
  1. Search for `client_id`, `authorize`, `oauth` and `token` strings to find endpoint constants and client identifiers.
  2. Look for `authUrl` construction code to see which query parameters are included (e.g., `redirect`, `state`, `client_id`, `code_challenge`).
  3. Inspect token exchange code to see whether it sends `client_secret` or `code_verifier` (or both).

- Repro steps for someone implementing the auth flow in another language (concise)
  1. Locate the `authUrl` construction and confirm required query params (client_id, redirect, state, any `loginMethod`/`type` parameters).
 2. Confirm whether PKCE is required by checking for `code_challenge`/`code_challenge_method` in the auth request and `code_verifier` in the token request.
 3. Confirm redirect URI path (`/oauth2callback` pattern) and port if the CLI starts a local callback server.
 4. Confirm token exchange method and auth method (Basic auth with client_id:client_secret in Authorization header OR an exchange that uses client_secret in body, or an exchange that uses `code_verifier` and omits client_secret).
 5. Confirm post-token user info fetch that returns `apiKey` separately—recreate that call to obtain the API key.

- Recommendations / next actions for an agent
  - If you need to replicate exactly, extract the authUrl and token request payloads verbatim from the bundle.
  - If the bundle uses PKCE, add PKCE (code_verifier + code_challenge=S256) to your client implementation. If it uses Basic client_secret authentication instead, ensure you include the client_secret in exchanges.
  - Use a modular approach: implement the browser + callback pattern, but make the PKCE piece optional/conditional so it can work with servers that require or permit it.

- References inside this repo (helpful snippets to reuse)
  - Example PKCE helper exists in `src/rotator_library/providers/qwen_auth_base.py` — you can reuse code-verifier / code-challenge generation from there.

Appendix: quick checklist for auditors
- [ ] Confirm `authUrl` query parameters.
- [ ] Confirm presence of `code_challenge` in auth request.
- [ ] Confirm presence of `code_verifier` in token request.
- [ ] Confirm token exchange auth method (Basic, client_secret body param, or neither).
- [ ] Confirm local callback path and default port.
- [ ] Confirm user-info endpoint that returns `apiKey`.

-- End of summary
