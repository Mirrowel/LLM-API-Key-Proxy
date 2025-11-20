**iFlow OAuth Flow – Extracted From Bundle (`iflow.js`)**

Purpose
- Provide a precise, implementation-ready description of the iFlow authentication flow as embedded in the bundled CLI (`external/iflow-cli-tarball/package/bundle/iflow.js`).
- Enable faithful replication in other tooling without re-reading the minified bundle.

Scope
- Focuses ONLY on iFlow-specific authorization (NOT generic Google / other OAuth client code also present in the bundle).
- Omits internal unrelated libraries (telemetry, localization, unrelated OAuth handlers for other issuers).

Overview of Sequence
1. Determine callback port: Either `OAUTH_CALLBACK_PORT` env var, or dynamically allocate an ephemeral free port.
2. Build local redirect URI: `http://localhost:{port}/oauth2callback` (host override via `OAUTH_CALLBACK_HOST`, default `localhost`).
3. Generate anti-CSRF `state`: 32 random hex bytes.
4. Construct authorization URL:
   - Base: `https://iflow.cn/oauth`
   - Query parameters:
     - `loginMethod=phone`
     - `type=phone`
     - `redirect=<encoded redirect URL>&state=<state>` (note: the bundle concatenates and encodes the redirect first, then appends `&state=` and the state value; effectively `redirect` holds both the actual callback URL and embeds the state parameter)
     - `client_id=10009311001`
5. Display URL & attempt to open it in the system browser.
6. Local HTTP server listens for requests to `/oauth2callback`.
7. On callback:
   - Reject if path not `/oauth2callback`.
   - If `error` param present: redirect to `https://iflow.cn/oauth/error` and fail.
   - If `state` mismatch: respond with plain text error (“State mismatch. Possible CSRF attack”).
   - If `code` is present and `state` matches: exchange code for tokens.
8. Token exchange: POST `https://iflow.cn/oauth/token`.
9. On success: persist credentials (JSON file via internal helper), then fetch user info to obtain `apiKey`.
10. Redirect browser to success page: `https://iflow.cn/oauth/success`.

Authorization URL Details
- Final composed form (illustrative – values substituted):
  ```
  https://iflow.cn/oauth?loginMethod=phone&type=phone&redirect={ENCODED_REDIRECT_AND_STATE}&client_id=10009311001
  ```
- Important nuance: The bundle embeds the state inside the `redirect` query value (pattern: `redirect=<encodeURIComponent(callbackUrl)> &state=<state>` BEFORE concatenation). Replication MAY instead send state as a separate top-level query parameter if the server tolerates it; however to remain faithful, mimic the bundle pattern: keep `state` appended inside the `redirect` value.
  - Conservative replication: EXACT concatenation used by bundle:
    - Let `callback = http://localhost:{port}/oauth2callback`
    - Let `state = <random-hex>`
    - Let `redirectParamValue = encodeURIComponent(callback) + "&state=" + state`
    - Query param: `redirect=redirectParamValue`
    - Full URL: `https://iflow.cn/oauth?loginMethod=phone&type=phone&redirect=${redirectParamValue}&client_id=10009311001`

Callback Handling Logic
- Expected parameters (in parsed search params of original request URL):
  - `code`: authorization code (required for success)
  - `state`: must equal originally generated random hex string
  - `error`: signals authentication failure
- Error conditions:
  - Missing `code`: treated as failure ("noCodeFound").
  - Missing or mismatched `state`: potential CSRF → failure.
  - Any `error` param → failure.
- On success: code progresses to token exchange.

Token Exchange (Authorization Code Grant)
- Endpoint: `https://iflow.cn/oauth/token`
- Method: `POST`
- Headers:
  - `Content-Type: application/x-www-form-urlencoded`
  - `Authorization: Basic <base64(client_id:client_secret)>`
- Body parameters (URL-encoded form):
  - `grant_type=authorization_code`
  - `code=<authorization_code>`
  - `redirect_uri=http://localhost:{port}/oauth2callback` (MUST match original callback URI before encoding inside redirect param)
  - `client_id=10009311001`
  - `client_secret=4Z3YjXycVsQvyGF1etiNlIBB4RsqSDtW`
- Response JSON fields consumed:
  - `access_token`
  - `refresh_token`
  - `expires_in` (converted to absolute expiry timestamp)
  - `token_type`
  - `scope`

User Info / API Key Retrieval
- Endpoint: `https://iflow.cn/api/oauth/getUserInfo?accessToken=<access_token>`
- Method: `GET`
- Response expected shape (simplified):
  ```json
  {
    "success": true,
    "data": { "apiKey": "...", "email": "..." }
  }
  ```
- On success, the extracted `apiKey` is appended to stored credentials and cached.
- Retries: The bundle performs up to 3 retries with backoff for transient (5xx / 408 / 429) failures.

Persisted Credential Structure (conceptual)
- Keys stored: `access_token`, `refresh_token`, `expiry_date`, `token_type`, `scope`, optionally `apiKey`.
- File path: derived via internal helper (not exposed here); replication may choose its own path (e.g. `~/.iflow/credentials.json`).
  - Permissions set mode `0600` (decimal 384) when writing.

Refresh Flow
- Trigger conditions:
  - Token nearing expiry (< 24 hours) or explicit refresh check in background tasks.
- Refresh request:
  - POST `https://iflow.cn/oauth/token`
  - Headers: same as authorization-code exchange.
  - Body params:
    - `grant_type=refresh_token`
    - `refresh_token=<stored_refresh_token>`
    - `client_id=10009311001`
    - `client_secret=4Z3YjXycVsQvyGF1etiNlIBB4RsqSDtW`
- Response handling mirrors initial token response (update expiry, access/refresh tokens, scope, token type). May re-fetch user info to ensure `apiKey` is current.

PKCE Status
- The bundle contains a generic OAuth library with PKCE helpers (code_verifier / code_challenge generation) for other issuers.
- The iFlow-specific code path DOES NOT include `code_challenge` or `code_verifier` in its authorization or token requests.
- Conclusion: PKCE is NOT required for the current iFlow CLI flow; replication should omit PKCE unless the provider later enforces it.

Environment Variables Influencing Flow
- `OAUTH_CALLBACK_PORT`: If set and valid (1–65535), use as callback port instead of ephemeral port.
- `OAUTH_CALLBACK_HOST`: Hostname for listening; default `localhost`.
- (Potential others exist for generic flows, but only these are critical for iFlow-specific path.)

Security Considerations
- State validation prevents CSRF / unsolicited redirects.
- The unusual embedding of state inside the `redirect` parameter means the state also appears server-side only if parsed out; exact replication should preserve this pattern unless confirmed safe to normalize.
- Basic auth header exposes client_secret; always use HTTPS (the CLI does). Do not log raw Authorization headers.
- Persisted credentials should be stored with restrictive file permissions.
- Refresh token invalidation must trigger re-authentication (bundle clears credentials on failure).

Error Handling Patterns (Simplified)
- Authorization errors: Redirect user to error page `https://iflow.cn/oauth/error`.
- Token request failure: Inspect HTTP status & status text; raise user-facing error.
- Refresh failure or expiry: Clear credential file and require re-login.
- User info failures: Retry transient errors (5xx / 408 / 429), otherwise log and continue without apiKey.

Replication Checklist
1. Generate random hex `state` (32 bytes recommended).  
2. Select callback port: env override or ephemeral free port.  
3. Build callback URL `http://localhost:{port}/oauth2callback`.  
4. Compose authorization URL exactly as bundle does (embedded `&state=` inside `redirect` value).  
5. Launch local HTTP server; validate `state`, extract `code`.  
6. Exchange code using Basic auth + form body (include both client_id/client_secret).  
7. Store credentials; compute `expiry_date = now + expires_in*1000`.  
8. Fetch user info to get `apiKey`; attach to stored credentials.  
9. Implement refresh POST (grant_type=refresh_token) with same auth style.  
10. Implement retry & expiry logic; clear credentials if permanently invalid.  
11. Never include PKCE unless provider changes requirements.  
12. Optionally open browser automatically and present fallback URL if open fails.  

Minimal Pseudocode Outline (Language-Agnostic)
```text
state = randomHex(32)
port = env.OAUTH_CALLBACK_PORT || findFreePort()
callback = "http://localhost:" + port + "/oauth2callback"
redirectParamValue = urlencode(callback) + "&state=" + state
authUrl = "https://iflow.cn/oauth?loginMethod=phone&type=phone&redirect=" + redirectParamValue + "&client_id=10009311001"

openBrowser(authUrl)
code = await waitForCallback(callback, state)

tokenResp = POST https://iflow.cn/oauth/token
  Headers: Content-Type, Authorization: Basic base64(client_id:client_secret)
  Body (form): grant_type=authorization_code, code, redirect_uri=callback, client_id, client_secret

creds = {access_token, refresh_token, expiry_date=now+expires_in*1000, token_type, scope}
userInfo = GET https://iflow.cn/api/oauth/getUserInfo?accessToken=creds.access_token
if userInfo.success: creds.apiKey = userInfo.data.apiKey
store(creds)

// Refresh flow
if nearing expiry:
  refreshResp = POST https://iflow.cn/oauth/token (grant_type=refresh_token,...)
  update creds, optionally re-fetch apiKey
```

Testing Recommendations
- Simulate callback manually with crafted URL containing correct `code` & `state` to test server path.
- Validate rejection on mismatched state.
- Confirm `apiKey` presence after user info call.
- Force refresh by adjusting expiry_date to near past and invoking refresh routine.

Future-Proofing
- Monitor provider announcements for PKCE enforcement; if required, switch to `code_challenge` / `code_verifier` standard pattern.
- Consider switching state embedding to separate query parameter if provider standardizes (requires test).

Summary
- The iFlow CLI uses a straightforward OAuth2 Authorization Code Grant with Basic client authentication and NO PKCE.
- The distinctive element is embedding `state` within the `redirect` parameter value.
- Replication requires careful reconstruction of the authorization URL, robust callback validation, token + refresh exchanges, and an additional user info call to retrieve the operational `apiKey`.

-- End of extracted flow documentation
