# src/rotator_library/providers/anthropic_oauth_base.py
"""
Anthropic OAuth Base Class

Base class for Anthropic OAuth2 authentication (Claude Pro/Max subscriptions).
Handles PKCE flow, token refresh, and credential management.

OAuth Configuration:
- Client ID: 9d1c250a-e61b-44d9-88ed-5944d1962f5e
- Auth URL: https://claude.ai/oauth/authorize
- Token URL: https://console.anthropic.com/v1/oauth/token
- Redirect URI: https://console.anthropic.com/oauth/code/callback
- Scopes: org:create_api_key user:profile user:inference
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import webbrowser
from urllib.parse import urlencode, urlparse, parse_qs

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt as RichPrompt
from rich.text import Text
from rich.markup import escape as rich_escape

from ..utils.headless_detection import is_headless_environment
from ..utils.reauth_coordinator import get_reauth_coordinator
from ..utils.resilient_io import safe_write_json
from ..error_handler import CredentialNeedsReauthError
from ..proxy_config import ProxyConfig

lib_logger = logging.getLogger("rotator_library")
console = Console()


@dataclass
class CredentialSetupResult:
    """Standardized result structure for credential setup operations."""
    success: bool
    file_path: Optional[str] = None
    email: Optional[str] = None
    tier: Optional[str] = None
    account_id: Optional[str] = None
    is_update: bool = False
    error: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = field(default=None, repr=False)

# =============================================================================
# OAUTH CONFIGURATION
# =============================================================================

ANTHROPIC_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
ANTHROPIC_AUTH_URL = "https://claude.ai/oauth/authorize"
ANTHROPIC_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
ANTHROPIC_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
ANTHROPIC_OAUTH_SCOPES = ["org:create_api_key", "user:profile", "user:inference"]

# Token refresh buffer in seconds (refresh tokens this far before expiry)
DEFAULT_REFRESH_EXPIRY_BUFFER: int = 5 * 60  # 5 minutes before expiry


def _generate_pkce() -> Tuple[str, str]:
    """Generate PKCE code verifier and challenge (S256)."""
    code_verifier = secrets.token_urlsafe(32)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode().rstrip("=")
    return code_verifier, code_challenge


class AnthropicOAuthBase:
    """
    Base class for Anthropic OAuth2 authentication.

    Handles:
    - Loading credentials from copied ~/.claude/.credentials.json files
      (nested claudeAiOauth format)
    - Loading credentials from env vars (ANTHROPIC_OAUTH_N_ACCESS_TOKEN)
    - Token refresh via JSON POST to Anthropic token endpoint
    - Interactive PKCE OAuth flow (manual code paste)
    - Queue-based refresh coordination
    """

    CLIENT_ID: str = ANTHROPIC_CLIENT_ID
    AUTH_URL: str = ANTHROPIC_AUTH_URL
    TOKEN_URL: str = ANTHROPIC_TOKEN_URL
    REDIRECT_URI: str = ANTHROPIC_REDIRECT_URI
    OAUTH_SCOPES: List[str] = ANTHROPIC_OAUTH_SCOPES
    ENV_PREFIX: str = "ANTHROPIC_OAUTH"
    REFRESH_EXPIRY_BUFFER_SECONDS: int = DEFAULT_REFRESH_EXPIRY_BUFFER

    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()

        # Backoff tracking
        self._refresh_failures: Dict[str, int] = {}
        self._next_refresh_after: Dict[str, float] = {}

        # Queue system for refresh and reauth
        self._refresh_queue: asyncio.Queue = asyncio.Queue()
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._reauth_queue: asyncio.Queue = asyncio.Queue()
        self._reauth_processor_task: Optional[asyncio.Task] = None

        # Tracking sets
        self._queued_credentials: set = set()
        self._unavailable_credentials: Dict[str, float] = {}
        self._unavailable_ttl_seconds: int = 360
        self._queue_tracking_lock = asyncio.Lock()
        self._queue_retry_count: Dict[str, int] = {}

        # Configuration
        self._refresh_timeout_seconds: int = 15
        self._refresh_interval_seconds: int = 30
        self._refresh_max_retries: int = 3
        self._reauth_timeout_seconds: int = 300

        # Tier cache: credential_path -> tier info
        self._tier_cache: Dict[str, Dict[str, Any]] = {}

        # Proxy configuration (injected by RotatingClient after construction)
        self._proxy_config: Optional[ProxyConfig] = None

    # =========================================================================
    # CREDENTIAL LOADING
    # =========================================================================

    def _parse_env_credential_path(self, path: str) -> Optional[str]:
        """Parse a virtual env:// path and return the credential index."""
        if not path.startswith("env://"):
            return None
        parts = path[6:].split("/")
        if len(parts) >= 2:
            return parts[1]
        return "0"

    def _load_from_env(self, credential_index: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load Anthropic OAuth credentials from environment variables.

        Expected variables for numbered format (index N):
        - ANTHROPIC_OAUTH_N_ACCESS_TOKEN
        - ANTHROPIC_OAUTH_N_REFRESH_TOKEN
        """
        if credential_index and credential_index != "0":
            prefix = f"{self.ENV_PREFIX}_{credential_index}"
            default_email = f"env-user-{credential_index}"
        else:
            prefix = self.ENV_PREFIX
            default_email = "env-user"

        access_token = os.getenv(f"{prefix}_ACCESS_TOKEN")
        refresh_token = os.getenv(f"{prefix}_REFRESH_TOKEN")

        if not access_token:
            return None

        lib_logger.debug(f"Loading {prefix} credentials from environment variables")

        expiry_str = os.getenv(f"{prefix}_EXPIRY_DATE", "0")
        try:
            expiry_date = float(expiry_str)
        except ValueError:
            expiry_date = 0

        creds = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": expiry_date,
            "_proxy_metadata": {
                "email": os.getenv(f"{prefix}_EMAIL", default_email),
                "last_check_timestamp": time.time(),
                "loaded_from_env": True,
                "env_credential_index": credential_index or "0",
            },
        }

        return creds

    def _parse_claude_credentials_file(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a Claude CLI .credentials.json file.

        The file has a nested structure:
        {
            "claudeAiOauth": {
                "accessToken": "sk-ant-oat01-...",
                "refreshToken": "sk-ant-ort01-...",
                "expiresAt": 1700000000000,  // milliseconds
                "scopes": [...],
                ...
            }
        }

        Normalizes to our internal flat format.
        """
        oauth_data = raw_data.get("claudeAiOauth", {})
        if not oauth_data:
            # Maybe it's already in flat format (from our own save)
            if raw_data.get("access_token"):
                return raw_data
            raise ValueError("No 'claudeAiOauth' key found in credentials file")

        access_token = oauth_data.get("accessToken", "")
        refresh_token = oauth_data.get("refreshToken", "")
        expires_at = oauth_data.get("expiresAt", 0)

        # expiresAt may be in milliseconds — normalise to seconds
        expiry_date = self._normalize_expiry(expires_at)

        creds = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": expiry_date,
            "_proxy_metadata": {
                "last_check_timestamp": time.time(),
                "subscription_type": oauth_data.get("subscriptionType"),
                "rate_limit_tier": oauth_data.get("rateLimitTier"),
                "email": oauth_data.get("email", ""),
            },
        }

        return creds

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        """Load credentials from file or environment."""
        if path in self._credentials_cache:
            return self._credentials_cache[path]

        async with await self._get_lock(path):
            if path in self._credentials_cache:
                return self._credentials_cache[path]

            # Check if this is a virtual env:// path
            credential_index = self._parse_env_credential_path(path)
            if credential_index is not None:
                env_creds = self._load_from_env(credential_index)
                if env_creds:
                    self._credentials_cache[path] = env_creds
                    return env_creds
                else:
                    raise IOError(
                        f"Environment variables for {self.ENV_PREFIX} credential index {credential_index} not found"
                    )

            # Try file-based loading
            try:
                lib_logger.debug(f"Loading Anthropic OAuth credentials from file: {path}")
                with open(path, "r") as f:
                    raw_data = json.load(f)
                creds = self._parse_claude_credentials_file(raw_data)
                self._credentials_cache[path] = creds

                # Cache tier info
                metadata = creds.get("_proxy_metadata", {})
                if metadata.get("subscription_type") or metadata.get("rate_limit_tier"):
                    self._tier_cache[path] = {
                        "subscription_type": metadata.get("subscription_type"),
                        "rate_limit_tier": metadata.get("rate_limit_tier"),
                    }

                return creds
            except FileNotFoundError:
                env_creds = self._load_from_env()
                if env_creds:
                    lib_logger.info(
                        f"File '{path}' not found, using Anthropic OAuth credentials from environment variables"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                raise IOError(
                    f"Anthropic OAuth credential file not found at '{path}'"
                )
            except Exception as e:
                raise IOError(
                    f"Failed to load Anthropic OAuth credentials from '{path}': {e}"
                )

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
        """Save credentials with in-memory fallback if disk unavailable."""
        self._credentials_cache[path] = creds

        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            lib_logger.debug("Credentials loaded from env, skipping file save")
            return

        if safe_write_json(
            path, creds, lib_logger, secure_permissions=True, buffer_on_failure=True
        ):
            lib_logger.debug(f"Saved updated Anthropic OAuth credentials to '{path}'.")
        else:
            lib_logger.warning(
                f"Anthropic OAuth credentials cached in memory only (buffered for retry)."
            )

    # =========================================================================
    # TOKEN EXPIRY CHECKS
    # =========================================================================

    def _normalize_expiry(self, raw: Any) -> float:
        """Normalize an expiry value to a Unix timestamp in seconds.

        Handles string coercion and millisecond timestamps (values > 1e12).
        Returns 0.0 on invalid input so callers treat the token as expired.
        """
        if isinstance(raw, str):
            try:
                raw = float(raw)
            except ValueError:
                return 0.0
        try:
            ts = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if ts > 1e12:
            ts /= 1000
        return ts

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        """Check if access token is expired or near expiry."""
        expiry_timestamp = self._normalize_expiry(creds.get("expiry_date", 0))
        return expiry_timestamp < time.time() + self.REFRESH_EXPIRY_BUFFER_SECONDS

    def _is_token_truly_expired(self, creds: Dict[str, Any]) -> bool:
        """Check if token is TRULY expired (past actual expiry)."""
        expiry_timestamp = self._normalize_expiry(creds.get("expiry_date", 0))
        return expiry_timestamp < time.time()

    # =========================================================================
    # TOKEN REFRESH
    # =========================================================================

    def _get_credential_stable_id(self, path: str) -> str:
        """Derive the stable_id for a credential path from cached metadata."""
        creds = self._credentials_cache.get(path)
        if creds:
            metadata = creds.get("_proxy_metadata", {})
            login = metadata.get("login")
            email = metadata.get("email")
            stable = login or email
            if stable:
                return stable
        return ""

    def _build_proxy_client_kwargs(self, path: str, provider: str = "") -> Dict[str, Any]:
        """Build httpx.AsyncClient kwargs with proxy routing for a credential.

        Uses the same proxy resolution as API requests so that token refresh
        traffic egresses from the same IP as normal requests.
        """
        kwargs: Dict[str, Any] = {}
        if not self._proxy_config or not self._proxy_config.has_any_proxy:
            return kwargs

        stable_id = self._get_credential_stable_id(path)
        provider = provider or self.ENV_PREFIX.lower()
        spec = self._proxy_config.resolve(provider, path, stable_id)
        if spec:
            kwargs["proxy"] = spec.url
            lib_logger.debug(
                f"Token refresh for '{Path(path).name}' will use proxy {spec.url}"
            )
        return kwargs

    async def _refresh_token(
        self, path: str, creds: Dict[str, Any], force: bool = False
    ) -> Dict[str, Any]:
        """Refresh access token using refresh token via JSON POST."""
        async with await self._get_lock(path):
            if not force and not self._is_token_expired(
                self._credentials_cache.get(path, creds)
            ):
                return self._credentials_cache.get(path, creds)

            lib_logger.debug(
                f"Refreshing Anthropic OAuth token for '{Path(path).name}' (forced: {force})..."
            )

            refresh_token = creds.get("refresh_token")
            if not refresh_token:
                raise ValueError("No refresh_token found in Anthropic credentials.")

            max_retries = self._refresh_max_retries
            new_token_data = None
            last_error = None

            proxy_kwargs = self._build_proxy_client_kwargs(path)
            async with httpx.AsyncClient(**proxy_kwargs) as client:
                for attempt in range(max_retries):
                    try:
                        # Anthropic uses JSON body for token refresh (not form-encoded)
                        response = await client.post(
                            self.TOKEN_URL,
                            json={
                                "grant_type": "refresh_token",
                                "refresh_token": refresh_token,
                                "client_id": self.CLIENT_ID,
                            },
                            headers={"Content-Type": "application/json"},
                            timeout=self._refresh_timeout_seconds,
                        )
                        response.raise_for_status()
                        new_token_data = response.json()
                        break

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        status_code = e.response.status_code
                        error_body = e.response.text

                        _err_type = ""
                        try:
                            _err_type = json.loads(error_body).get("error", "")
                        except Exception:
                            _err_type = error_body.lower()
                        if status_code == 400 and _err_type == "invalid_grant":
                            lib_logger.info(
                                f"Anthropic credential '{Path(path).name}' needs re-auth (HTTP 400: invalid_grant)."
                            )
                            asyncio.create_task(
                                self._queue_refresh(path, force=True, needs_reauth=True)
                            )
                            raise CredentialNeedsReauthError(
                                credential_path=path,
                                message=f"Anthropic refresh token invalid for '{Path(path).name}'. Re-auth queued.",
                            )

                        elif status_code in (401, 403):
                            lib_logger.info(
                                f"Anthropic credential '{Path(path).name}' needs re-auth (HTTP {status_code})."
                            )
                            asyncio.create_task(
                                self._queue_refresh(path, force=True, needs_reauth=True)
                            )
                            raise CredentialNeedsReauthError(
                                credential_path=path,
                                message=f"Anthropic token invalid for '{Path(path).name}' (HTTP {status_code}). Re-auth queued.",
                            )

                        elif status_code == 429:
                            retry_after = int(e.response.headers.get("Retry-After", 60))
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_after)
                                continue
                            raise

                        elif status_code >= 500:
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            raise

                        else:
                            raise

                    except (httpx.RequestError, httpx.TimeoutException) as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise

            if new_token_data is None:
                raise last_error or Exception("Token refresh failed after all retries")

            # Update credentials
            creds["access_token"] = new_token_data["access_token"]
            expiry_timestamp = time.time() + new_token_data.get("expires_in", 3600)
            creds["expiry_date"] = expiry_timestamp

            if "refresh_token" in new_token_data:
                creds["refresh_token"] = new_token_data["refresh_token"]

            # Update metadata
            if "_proxy_metadata" not in creds:
                creds["_proxy_metadata"] = {}
            creds["_proxy_metadata"]["last_check_timestamp"] = time.time()

            await self._save_credentials(path, creds)
            lib_logger.debug(
                f"Successfully refreshed Anthropic OAuth token for '{Path(path).name}'."
            )
            return creds

    # =========================================================================
    # LOCK & AVAILABILITY
    # =========================================================================

    async def _get_lock(self, path: str) -> asyncio.Lock:
        """Get or create a lock for a credential path."""
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

    def is_credential_available(self, path: str) -> bool:
        """Check if a credential is available for rotation."""
        if path in self._unavailable_credentials:
            marked_time = self._unavailable_credentials.get(path)
            if marked_time is not None:
                now = time.time()
                if now - marked_time > self._unavailable_ttl_seconds:
                    self._unavailable_credentials.pop(path, None)
                    self._queued_credentials.discard(path)
                else:
                    return False

        creds = self._credentials_cache.get(path)
        if creds and self._is_token_truly_expired(creds):
            if path not in self._queued_credentials:
                asyncio.create_task(
                    self._queue_refresh(path, force=True, needs_reauth=False)
                )
            return False

        return True

    # =========================================================================
    # QUEUE MANAGEMENT
    # =========================================================================

    async def _queue_refresh(
        self, path: str, force: bool = False, needs_reauth: bool = False
    ):
        """Add a credential to the appropriate refresh queue."""
        if not needs_reauth:
            now = time.time()
            if path in self._next_refresh_after:
                if now < self._next_refresh_after[path]:
                    return

        async with self._queue_tracking_lock:
            if path not in self._queued_credentials:
                self._queued_credentials.add(path)

                if needs_reauth:
                    self._unavailable_credentials[path] = time.time()
                    await self._reauth_queue.put(path)
                    await self._ensure_reauth_processor_running()
                else:
                    await self._refresh_queue.put((path, force))
                    await self._ensure_queue_processor_running()

    async def _ensure_queue_processor_running(self):
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(
                self._process_refresh_queue()
            )

    async def _ensure_reauth_processor_running(self):
        if self._reauth_processor_task is None or self._reauth_processor_task.done():
            self._reauth_processor_task = asyncio.create_task(
                self._process_reauth_queue()
            )

    async def _process_refresh_queue(self):
        """Background worker that processes normal refresh requests."""
        while True:
            path = None
            try:
                try:
                    path, force = await asyncio.wait_for(
                        self._refresh_queue.get(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    async with self._queue_tracking_lock:
                        self._queue_retry_count.clear()
                    self._queue_processor_task = None
                    return

                try:
                    creds = self._credentials_cache.get(path)
                    if creds and not self._is_token_expired(creds):
                        self._queue_retry_count.pop(path, None)
                        continue

                    if not creds:
                        creds = await self._load_credentials(path)

                    try:
                        async with asyncio.timeout(self._refresh_timeout_seconds):
                            await self._refresh_token(path, creds, force=force)
                        self._queue_retry_count.pop(path, None)

                    except asyncio.TimeoutError:
                        lib_logger.warning(f"Refresh timeout for '{Path(path).name}'")
                        await self._handle_refresh_failure(path, force, "timeout")

                    except httpx.HTTPStatusError as e:
                        if e.response.status_code in (401, 403):
                            self._queue_retry_count.pop(path, None)
                            async with self._queue_tracking_lock:
                                self._queued_credentials.discard(path)
                            await self._queue_refresh(path, force=True, needs_reauth=True)
                        else:
                            await self._handle_refresh_failure(
                                path, force, f"HTTP {e.response.status_code}"
                            )

                    except Exception as e:
                        await self._handle_refresh_failure(path, force, str(e))

                finally:
                    async with self._queue_tracking_lock:
                        if (
                            path in self._queued_credentials
                            and self._queue_retry_count.get(path, 0) == 0
                        ):
                            self._queued_credentials.discard(path)
                    self._refresh_queue.task_done()

                await asyncio.sleep(self._refresh_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                lib_logger.error(f"Error in Anthropic refresh queue processor: {e}")
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)

    async def _handle_refresh_failure(self, path: str, force: bool, error: str):
        """Handle a refresh failure with back-of-line retry logic."""
        retry_count = self._queue_retry_count.get(path, 0) + 1
        self._queue_retry_count[path] = retry_count

        if retry_count >= self._refresh_max_retries:
            lib_logger.error(
                f"Max retries reached for Anthropic '{Path(path).name}' (last error: {error})."
            )
            self._queue_retry_count.pop(path, None)
            async with self._queue_tracking_lock:
                self._queued_credentials.discard(path)
            return

        lib_logger.warning(
            f"Anthropic refresh failed for '{Path(path).name}' ({error}). "
            f"Retry {retry_count}/{self._refresh_max_retries}."
        )
        await self._refresh_queue.put((path, force))

    async def _process_reauth_queue(self):
        """Background worker that processes re-auth requests."""
        while True:
            path = None
            try:
                try:
                    path = await asyncio.wait_for(
                        self._reauth_queue.get(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    self._reauth_processor_task = None
                    return

                try:
                    lib_logger.info(f"Starting Anthropic re-auth for '{Path(path).name}'...")
                    await self.initialize_token(path, force_interactive=True)
                    lib_logger.info(f"Anthropic re-auth SUCCESS for '{Path(path).name}'")
                except Exception as e:
                    lib_logger.error(f"Anthropic re-auth FAILED for '{Path(path).name}': {e}")
                finally:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)
                    self._reauth_queue.task_done()

            except asyncio.CancelledError:
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)
                break
            except Exception as e:
                lib_logger.error(f"Error in Anthropic re-auth queue processor: {e}")
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)

    # =========================================================================
    # INTERACTIVE OAUTH FLOW
    # =========================================================================

    async def _perform_interactive_oauth(
        self, path: str, creds: Dict[str, Any], display_name: str
    ) -> Dict[str, Any]:
        """
        Perform interactive OAuth flow for Anthropic.

        Since Anthropic uses a fixed redirect URI (not localhost), the user must:
        1. Open the auth URL in a browser
        2. Complete login
        3. Copy the authorization code from the redirect page
        4. Paste it back into the terminal
        """
        code_verifier, code_challenge = _generate_pkce()
        # Anthropic uses the PKCE verifier as the state value (per opencode-anthropic-auth plugin)
        state = code_verifier

        auth_params = {
            "code": "true",  # Required by Anthropic OAuth
            "client_id": self.CLIENT_ID,
            "response_type": "code",
            "redirect_uri": self.REDIRECT_URI,
            "scope": " ".join(self.OAUTH_SCOPES),
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        }

        auth_url = f"{self.AUTH_URL}?" + urlencode(auth_params)

        is_headless = is_headless_environment()

        if is_headless:
            auth_panel_text = Text.from_markup(
                "Running in headless environment (no GUI detected).\n"
                "Please open the URL below in a browser on another machine to authorize:\n"
            )
        else:
            auth_panel_text = Text.from_markup(
                "1. Open the URL below in your browser to log in and authorize.\n"
                "2. After authorizing, you'll be redirected. Copy the authorization code.\n"
                "3. Paste the code back here."
            )

        console.print(
            Panel(
                auth_panel_text,
                title=f"Anthropic OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
                style="bold blue",
            )
        )

        escaped_url = rich_escape(auth_url)
        console.print(f"[bold]URL:[/bold] [link={auth_url}]{escaped_url}[/link]\n")

        if not is_headless:
            try:
                webbrowser.open(auth_url)
                lib_logger.info("Browser opened successfully for Anthropic OAuth flow")
            except Exception as e:
                lib_logger.warning(
                    f"Failed to open browser automatically: {e}. Please open the URL manually."
                )

        # Wait for user to paste the redirect URL or authorization code
        console.print(
            "[bold green]After authorizing, paste the full redirect URL "
            "(or just the code) here:[/bold green]\n"
            "[dim]The redirect URL looks like: "
            "https://console.anthropic.com/oauth/code/callback?code=BGDi...&state=...[/dim]"
        )

        # Use asyncio-compatible input
        loop = asyncio.get_running_loop()
        pasted_input = await loop.run_in_executor(None, input, "> ")
        pasted_input = pasted_input.strip()

        if not pasted_input:
            raise Exception("No authorization code provided.")

        # Parse the code from whatever the user pasted:
        # 1. Full redirect URL: extract ?code= query param
        # 2. code#state fragment format (as shown on Anthropic callback page)
        # 3. Bare code
        auth_code = pasted_input
        if "?" in pasted_input or pasted_input.startswith("http"):
            parsed = urlparse(pasted_input)
            qs = parse_qs(parsed.query)
            if "code" in qs:
                auth_code = qs["code"][0]
            else:
                # Fallback: treat everything before # as the code
                auth_code = pasted_input.split("#")[0].split("?")[-1]
        elif "#" in pasted_input:
            # code#state format — take only the part before #
            auth_code = pasted_input.split("#")[0]

        auth_code = auth_code.strip()
        if not auth_code:
            raise Exception("Could not extract authorization code from input.")

        # Extract state from the user's input to echo back in the token exchange.
        # - Full URL: state is in the query string (?state=...)
        # - code#state format: state follows the '#'
        # - Bare code: use the locally-generated state (code_verifier)
        if "?" in pasted_input or pasted_input.startswith("http"):
            _qs = parse_qs(urlparse(pasted_input).query)
            raw_state = _qs.get("state", [state])[0] or state
        elif "#" in pasted_input:
            _parts = pasted_input.split("#", 1)
            raw_state = _parts[1].strip() if len(_parts) > 1 and _parts[1].strip() else state
        else:
            raw_state = state

        lib_logger.info("Exchanging authorization code for tokens...")

        proxy_kwargs = self._build_proxy_client_kwargs(path) if path else {}
        async with httpx.AsyncClient(**proxy_kwargs) as client:
            response = await client.post(
                self.TOKEN_URL,
                json={
                    "grant_type": "authorization_code",
                    "code": auth_code,
                    "state": raw_state,
                    "client_id": self.CLIENT_ID,
                    "code_verifier": code_verifier,
                    "redirect_uri": self.REDIRECT_URI,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            token_data = response.json()

            new_creds = {
                "access_token": token_data.get("access_token"),
                "refresh_token": token_data.get("refresh_token"),
                "expiry_date": time.time() + token_data.get("expires_in", 3600),
                "_proxy_metadata": {
                    "last_check_timestamp": time.time(),
                },
            }

            # Prompt for an identifier — Anthropic's token response contains no email
            try:
                identifier = RichPrompt.ask(
                    "\n[bold]Enter an identifier for this credential "
                    "(e.g. your email or a label like 'pro-account')[/bold]"
                )
                identifier = identifier.strip()
            except (EOFError, KeyboardInterrupt):
                identifier = ""

            if not identifier:
                console.print(
                    "[bold yellow]No identifier provided. "
                    "Deduplication will not be possible.[/bold yellow]"
                )

            new_creds["_proxy_metadata"]["email"] = identifier or None

            if path:
                await self._save_credentials(path, new_creds)

            lib_logger.info(
                f"Anthropic OAuth initialized successfully for '{display_name}'."
            )

            return new_creds

    # =========================================================================
    # TOKEN INITIALIZATION
    # =========================================================================

    async def initialize_token(
        self,
        creds_or_path: Union[Dict[str, Any], str],
        force_interactive: bool = False,
    ) -> Dict[str, Any]:
        """Initialize OAuth token, triggering interactive OAuth flow if needed."""
        path = creds_or_path if isinstance(creds_or_path, str) else None

        if isinstance(creds_or_path, dict):
            display_name = creds_or_path.get("_proxy_metadata", {}).get(
                "display_name", "in-memory object"
            )
        else:
            display_name = Path(path).name if path else "in-memory object"

        lib_logger.debug(f"Initializing Anthropic token for '{display_name}'...")

        try:
            creds = (
                await self._load_credentials(creds_or_path) if path else creds_or_path
            )
            reason = ""

            if force_interactive:
                reason = "re-authentication was explicitly requested"
            elif not creds.get("refresh_token"):
                reason = "refresh token is missing"
            elif self._is_token_expired(creds):
                reason = "token is expired"

            if reason:
                if reason == "token is expired" and creds.get("refresh_token"):
                    try:
                        return await self._refresh_token(path, creds)
                    except Exception as e:
                        lib_logger.warning(
                            f"Automatic token refresh for '{display_name}' failed: {e}. Proceeding to interactive login."
                        )

                lib_logger.warning(
                    f"Anthropic OAuth token for '{display_name}' needs setup: {reason}."
                )

                coordinator = get_reauth_coordinator()

                async def _do_interactive_oauth():
                    return await self._perform_interactive_oauth(path, creds, display_name)

                return await coordinator.execute_reauth(
                    credential_path=path or display_name,
                    provider_name="ANTHROPIC_OAUTH",
                    reauth_func=_do_interactive_oauth,
                    timeout=300.0,
                )

            lib_logger.info(f"Anthropic OAuth token at '{display_name}' is valid.")
            return creds

        except Exception as e:
            raise ValueError(
                f"Failed to initialize Anthropic OAuth for '{path}': {e}"
            )

    # =========================================================================
    # AUTH HEADER
    # =========================================================================

    async def get_anthropic_auth_header(self, credential_path: str) -> Dict[str, str]:
        """
        Get auth header for Anthropic OAuth requests.

        Returns Bearer token header for use with Anthropic Messages API.
        """
        try:
            creds = await self._load_credentials(credential_path)

            if self._is_token_expired(creds):
                try:
                    creds = await self._refresh_token(credential_path, creds)
                except Exception as e:
                    cached = self._credentials_cache.get(credential_path)
                    if cached and cached.get("access_token"):
                        lib_logger.warning(
                            f"Token refresh failed for {Path(credential_path).name}: {e}. "
                            "Using cached token."
                        )
                        creds = cached
                    else:
                        raise

            token = creds.get("access_token")
            return {"Authorization": f"Bearer {token}"}

        except Exception as e:
            cached = self._credentials_cache.get(credential_path)
            if cached and cached.get("access_token"):
                lib_logger.error(
                    f"Credential load failed for {credential_path}: {e}. Using stale cached token."
                )
                token = cached.get("access_token")
                return {"Authorization": f"Bearer {token}"}
            raise

    async def proactively_refresh(self, credential_path: str):
        """Proactively refresh a credential by queueing it for refresh."""
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            await self._queue_refresh(credential_path, force=False, needs_reauth=False)

    def get_credential_tier_info(self, credential_path: str) -> Optional[Dict[str, Any]]:
        """Get cached tier info for a credential."""
        return self._tier_cache.get(credential_path)

    # =========================================================================
    # CREDENTIAL MANAGEMENT METHODS
    # =========================================================================

    def _get_provider_file_prefix(self) -> str:
        return "anthropic"

    def _get_oauth_base_dir(self) -> Path:
        return Path.cwd() / "oauth_creds"

    def _find_existing_credential_by_email(
        self, email: str, base_dir: Optional[Path] = None
    ) -> Optional[Path]:
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        for cred_file in glob(pattern):
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)
                existing_email = creds.get("_proxy_metadata", {}).get("email")
                if existing_email == email:
                    return Path(cred_file)
            except Exception:
                continue

        return None

    def _get_next_credential_number(self, base_dir: Optional[Path] = None) -> int:
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        existing_numbers = []
        for cred_file in glob(pattern):
            match = re.search(r"_oauth_(\d+)\.json$", cred_file)
            if match:
                existing_numbers.append(int(match.group(1)))

        if not existing_numbers:
            return 1
        return max(existing_numbers) + 1

    def _build_credential_path(
        self, base_dir: Optional[Path] = None, number: Optional[int] = None
    ) -> Path:
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        if number is None:
            number = self._get_next_credential_number(base_dir)

        prefix = self._get_provider_file_prefix()
        filename = f"{prefix}_oauth_{number}.json"
        return base_dir / filename

    async def setup_credential(
        self, base_dir: Optional[Path] = None
    ) -> CredentialSetupResult:
        """Complete credential setup flow: interactive OAuth → save → return result."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        base_dir.mkdir(parents=True, exist_ok=True)

        try:
            temp_creds: Dict[str, Any] = {
                "_proxy_metadata": {"display_name": "new Anthropic OAuth credential"}
            }
            new_creds = await self._perform_interactive_oauth(
                path=None, creds=temp_creds, display_name="Anthropic / Claude Code"
            )

            email = new_creds.get("_proxy_metadata", {}).get("email", "")
            subscription_type = new_creds.get("_proxy_metadata", {}).get("subscription_type")

            existing_path = self._find_existing_credential_by_email(email, base_dir) if email else None
            is_update = existing_path is not None

            file_path = existing_path if is_update else self._build_credential_path(base_dir)

            await self._save_credentials(str(file_path), new_creds)

            return CredentialSetupResult(
                success=True,
                file_path=str(file_path),
                email=email or None,
                tier=subscription_type,
                is_update=is_update,
                credentials=new_creds,
            )

        except Exception as e:
            lib_logger.error(f"Anthropic credential setup failed: {e}")
            return CredentialSetupResult(success=False, error=str(e))

    def list_credentials(self, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        credentials = []
        for cred_file in sorted(glob(pattern)):
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)

                # Parse Claude-format credentials if needed
                parsed = self._parse_claude_credentials_file(creds)
                metadata = parsed.get("_proxy_metadata", {})

                match = re.search(r"_oauth_(\d+)\.json$", cred_file)
                number = int(match.group(1)) if match else 0

                credentials.append({
                    "file_path": cred_file,
                    "email": metadata.get("email") or "unknown",
                    "subscription_type": metadata.get("subscription_type"),
                    "rate_limit_tier": metadata.get("rate_limit_tier"),
                    "number": number,
                })
            except Exception:
                continue

        return credentials

    def build_env_lines(self, creds: Dict[str, Any], cred_number: int) -> List[str]:
        """
        Generate .env file lines for an Anthropic OAuth credential.

        Handles both the raw Claude CLI format (claudeAiOauth nested)
        and the already-parsed flat format.

        Args:
            creds: Credential dictionary loaded from JSON
            cred_number: Credential number (1, 2, 3, etc.)

        Returns:
            List of .env file lines
        """
        # Normalize to flat format if needed
        try:
            parsed = self._parse_claude_credentials_file(creds)
        except Exception:
            parsed = creds

        email = parsed.get("_proxy_metadata", {}).get("email", "unknown")
        prefix = f"{self.ENV_PREFIX}_{cred_number}"

        lines = [
            f"# {self.ENV_PREFIX} Credential #{cred_number} for: {email}",
            f"# Exported from: {self._get_provider_file_prefix()}_oauth_{cred_number}.json",
            f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "#",
            "# To combine multiple credentials into one .env file, copy these lines",
            "# and ensure each credential has a unique number (1, 2, 3, etc.)",
            "",
            f"{prefix}_ACCESS_TOKEN={parsed.get('access_token', '')}",
            f"{prefix}_REFRESH_TOKEN={parsed.get('refresh_token', '')}",
            f"{prefix}_EXPIRY_DATE={parsed.get('expiry_date', 0)}",
            f"{prefix}_EMAIL={email}",
        ]

        return lines

    def export_credential_to_env(
        self, credential_path: str, output_dir: Optional[Path] = None
    ) -> Optional[str]:
        """
        Export a credential file to .env format.

        Args:
            credential_path: Path to the credential JSON file
            output_dir: Directory for output .env file (defaults to same as credential)

        Returns:
            Path to the exported .env file, or None on error
        """
        try:
            cred_path = Path(credential_path)

            # Load credential
            with open(cred_path, "r") as f:
                creds = json.load(f)

            # Parse to flat format for email extraction
            try:
                parsed = self._parse_claude_credentials_file(creds)
            except Exception:
                parsed = creds

            email = parsed.get("_proxy_metadata", {}).get("email", "unknown")

            # Get credential number from filename
            match = re.search(r"_oauth_(\d+)\.json$", cred_path.name)
            cred_number = int(match.group(1)) if match else 1

            # Build output path
            if output_dir is None:
                output_dir = cred_path.parent

            safe_email = email.replace("@", "_at_").replace(".", "_")
            prefix = self._get_provider_file_prefix()
            env_filename = f"{prefix}_{cred_number}_{safe_email}.env"
            env_path = output_dir / env_filename

            # Build and write content
            env_lines = self.build_env_lines(creds, cred_number)
            with open(env_path, "w") as f:
                f.write("\n".join(env_lines))

            lib_logger.info(f"Exported Anthropic credential to {env_path}")
            return str(env_path)

        except Exception as e:
            lib_logger.error(f"Failed to export Anthropic credential: {e}")
            return None

    def delete_credential(self, credential_path: str) -> bool:
        """Delete a credential file and remove it from cache."""
        try:
            cred_path = Path(credential_path)

            prefix = self._get_provider_file_prefix()
            if not cred_path.name.startswith(f"{prefix}_oauth_"):
                lib_logger.error(
                    f"File {cred_path.name} does not appear to be an Anthropic credential"
                )
                return False

            if not cred_path.exists():
                lib_logger.warning(f"Credential file does not exist: {credential_path}")
                return False

            self._credentials_cache.pop(credential_path, None)
            cred_path.unlink()
            lib_logger.info(f"Deleted Anthropic credential: {credential_path}")
            return True

        except Exception as e:
            lib_logger.error(f"Failed to delete Anthropic credential: {e}")
            return False
