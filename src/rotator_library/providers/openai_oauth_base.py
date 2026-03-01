# src/rotator_library/providers/openai_oauth_base.py
"""
OpenAI OAuth Base Class

Base class for OpenAI OAuth2 authentication providers (Codex).
Handles PKCE flow, token refresh, and API key exchange.

OAuth Configuration:
- Client ID: app_EMoamEEZ73f0CkXaXp7hrann
- Authorization URL: https://auth.openai.com/oauth/authorize
- Token URL: https://auth.openai.com/oauth/token
- Redirect URI: http://localhost:1455/auth/callback
- Scopes: openid profile email offline_access
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
import webbrowser
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape as rich_escape

from ..utils.headless_detection import is_headless_environment
from ..utils.reauth_coordinator import get_reauth_coordinator
from ..utils.resilient_io import safe_write_json
from ..error_handler import CredentialNeedsReauthError

lib_logger = logging.getLogger("rotator_library")
console = Console()

# =============================================================================
# OAUTH CONFIGURATION
# =============================================================================

# OpenAI OAuth endpoints
OPENAI_AUTH_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"

# Default OAuth callback port for local redirect server
DEFAULT_OAUTH_CALLBACK_PORT: int = 1455

# Default OAuth callback path
DEFAULT_OAUTH_CALLBACK_PATH: str = "/auth/callback"

# Token refresh buffer in seconds (refresh tokens this far before expiry)
DEFAULT_REFRESH_EXPIRY_BUFFER: int = 5 * 60  # 5 minutes before expiry


@dataclass
class CredentialSetupResult:
    """
    Standardized result structure for credential setup operations.
    """
    success: bool
    file_path: Optional[str] = None
    email: Optional[str] = None
    tier: Optional[str] = None
    account_id: Optional[str] = None
    is_update: bool = False
    error: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = field(default=None, repr=False)


def _generate_pkce() -> Tuple[str, str]:
    """
    Generate PKCE code verifier and challenge.

    Returns:
        Tuple of (code_verifier, code_challenge)
    """
    # Generate random code verifier (43-128 characters)
    code_verifier = secrets.token_urlsafe(32)

    # Create code challenge using S256 method
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).decode().rstrip("=")

    return code_verifier, code_challenge


def _parse_jwt_claims(token: str) -> Optional[Dict[str, Any]]:
    """
    Parse JWT token and extract claims from payload.

    Args:
        token: JWT token string

    Returns:
        Decoded payload as dict, or None if invalid
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        payload = parts[1]
        # Add padding if needed
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding

        decoded = base64.urlsafe_b64decode(payload).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        return None


class OpenAIOAuthBase:
    """
    Base class for OpenAI OAuth2 authentication providers.

    Subclasses must override:
        - CLIENT_ID: OAuth client ID
        - OAUTH_SCOPES: List of OAuth scopes
        - ENV_PREFIX: Prefix for environment variables (e.g., "CODEX")

    Subclasses may optionally override:
        - CALLBACK_PORT: Local OAuth callback server port (default: 1455)
        - CALLBACK_PATH: OAuth callback path (default: "/auth/callback")
        - REFRESH_EXPIRY_BUFFER_SECONDS: Time buffer before token expiry
    """

    # Subclasses MUST override these
    CLIENT_ID: str = "app_EMoamEEZ73f0CkXaXp7hrann"
    OAUTH_SCOPES: List[str] = ["openid", "profile", "email", "offline_access"]
    ENV_PREFIX: str = "CODEX"

    # Subclasses MAY override these
    AUTH_URL: str = OPENAI_AUTH_URL
    TOKEN_URL: str = OPENAI_TOKEN_URL
    CALLBACK_PORT: int = DEFAULT_OAUTH_CALLBACK_PORT
    CALLBACK_PATH: str = DEFAULT_OAUTH_CALLBACK_PATH
    REFRESH_EXPIRY_BUFFER_SECONDS: int = DEFAULT_REFRESH_EXPIRY_BUFFER

    @property
    def callback_port(self) -> int:
        """
        Get the OAuth callback port, checking environment variable first.
        """
        env_var = f"{self.ENV_PREFIX}_OAUTH_PORT"
        env_value = os.getenv(env_var)
        if env_value:
            try:
                return int(env_value)
            except ValueError:
                lib_logger.warning(
                    f"Invalid {env_var} value: {env_value}, using default {self.CALLBACK_PORT}"
                )
        return self.CALLBACK_PORT

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
        Load OAuth credentials from environment variables.

        Expected variables for numbered format (index N):
        - {ENV_PREFIX}_{N}_API_KEY (the exchanged API key)
        - {ENV_PREFIX}_{N}_ACCESS_TOKEN
        - {ENV_PREFIX}_{N}_REFRESH_TOKEN
        - {ENV_PREFIX}_{N}_ID_TOKEN
        - {ENV_PREFIX}_{N}_ACCOUNT_ID
        - {ENV_PREFIX}_{N}_EXPIRY_DATE
        - {ENV_PREFIX}_{N}_EMAIL
        """
        if credential_index and credential_index != "0":
            prefix = f"{self.ENV_PREFIX}_{credential_index}"
            default_email = f"env-user-{credential_index}"
        else:
            prefix = self.ENV_PREFIX
            default_email = "env-user"

        # Check for API key or access token
        api_key = os.getenv(f"{prefix}_API_KEY")
        access_token = os.getenv(f"{prefix}_ACCESS_TOKEN")
        refresh_token = os.getenv(f"{prefix}_REFRESH_TOKEN")

        if not (api_key or access_token):
            return None

        lib_logger.debug(f"Loading {prefix} credentials from environment variables")

        expiry_str = os.getenv(f"{prefix}_EXPIRY_DATE", "0")
        try:
            expiry_date = float(expiry_str)
        except ValueError:
            expiry_date = 0

        creds = {
            "api_key": api_key,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "id_token": os.getenv(f"{prefix}_ID_TOKEN"),
            "account_id": os.getenv(f"{prefix}_ACCOUNT_ID"),
            "expiry_date": expiry_date,
            "_proxy_metadata": {
                "email": os.getenv(f"{prefix}_EMAIL", default_email),
                "last_check_timestamp": time.time(),
                "loaded_from_env": True,
                "env_credential_index": credential_index or "0",
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
                lib_logger.debug(f"Loading {self.ENV_PREFIX} credentials from file: {path}")
                with open(path, "r") as f:
                    creds = json.load(f)
                self._credentials_cache[path] = creds
                return creds
            except FileNotFoundError:
                env_creds = self._load_from_env()
                if env_creds:
                    lib_logger.info(
                        f"File '{path}' not found, using {self.ENV_PREFIX} credentials from environment variables"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                raise IOError(
                    f"{self.ENV_PREFIX} OAuth credential file not found at '{path}'"
                )
            except Exception as e:
                raise IOError(
                    f"Failed to load {self.ENV_PREFIX} OAuth credentials from '{path}': {e}"
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
            lib_logger.debug(f"Saved updated {self.ENV_PREFIX} OAuth credentials to '{path}'.")
        else:
            lib_logger.warning(
                f"Credentials for {self.ENV_PREFIX} cached in memory only (buffered for retry)."
            )

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        """Check if access token is expired or near expiry."""
        expiry_timestamp = creds.get("expiry_date", 0)
        if isinstance(expiry_timestamp, str):
            try:
                expiry_timestamp = float(expiry_timestamp)
            except ValueError:
                expiry_timestamp = 0

        # Handle milliseconds vs seconds
        if expiry_timestamp > 1e12:
            expiry_timestamp = expiry_timestamp / 1000

        return expiry_timestamp < time.time() + self.REFRESH_EXPIRY_BUFFER_SECONDS

    def _is_token_truly_expired(self, creds: Dict[str, Any]) -> bool:
        """Check if token is TRULY expired (past actual expiry)."""
        expiry_timestamp = creds.get("expiry_date", 0)
        if isinstance(expiry_timestamp, str):
            try:
                expiry_timestamp = float(expiry_timestamp)
            except ValueError:
                expiry_timestamp = 0

        if expiry_timestamp > 1e12:
            expiry_timestamp = expiry_timestamp / 1000

        return expiry_timestamp < time.time()

    async def _refresh_token(
        self, path: str, creds: Dict[str, Any], force: bool = False
    ) -> Dict[str, Any]:
        """Refresh access token using refresh token."""
        async with await self._get_lock(path):
            if not force and not self._is_token_expired(
                self._credentials_cache.get(path, creds)
            ):
                return self._credentials_cache.get(path, creds)

            lib_logger.debug(
                f"Refreshing {self.ENV_PREFIX} OAuth token for '{Path(path).name}' (forced: {force})..."
            )

            refresh_token = creds.get("refresh_token")
            if not refresh_token:
                raise ValueError("No refresh_token found in credentials file.")

            max_retries = 3
            new_token_data = None
            last_error = None

            async with httpx.AsyncClient() as client:
                for attempt in range(max_retries):
                    try:
                        response = await client.post(
                            self.TOKEN_URL,
                            data={
                                "grant_type": "refresh_token",
                                "refresh_token": refresh_token,
                                "client_id": self.CLIENT_ID,
                            },
                            headers={"Content-Type": "application/x-www-form-urlencoded"},
                            timeout=30.0,
                        )
                        response.raise_for_status()
                        new_token_data = response.json()
                        break

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        status_code = e.response.status_code
                        error_body = e.response.text

                        if status_code == 400 and "invalid_grant" in error_body.lower():
                            lib_logger.info(
                                f"Credential '{Path(path).name}' needs re-auth (HTTP 400: invalid_grant)."
                            )
                            asyncio.create_task(
                                self._queue_refresh(path, force=True, needs_reauth=True)
                            )
                            raise CredentialNeedsReauthError(
                                credential_path=path,
                                message=f"Refresh token invalid for '{Path(path).name}'. Re-auth queued.",
                            )

                        elif status_code in (401, 403):
                            lib_logger.info(
                                f"Credential '{Path(path).name}' needs re-auth (HTTP {status_code})."
                            )
                            asyncio.create_task(
                                self._queue_refresh(path, force=True, needs_reauth=True)
                            )
                            raise CredentialNeedsReauthError(
                                credential_path=path,
                                message=f"Token invalid for '{Path(path).name}' (HTTP {status_code}). Re-auth queued.",
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

            if "id_token" in new_token_data:
                creds["id_token"] = new_token_data["id_token"]

            # Update metadata
            if "_proxy_metadata" not in creds:
                creds["_proxy_metadata"] = {}
            creds["_proxy_metadata"]["last_check_timestamp"] = time.time()

            await self._save_credentials(path, creds)
            lib_logger.debug(
                f"Successfully refreshed {self.ENV_PREFIX} OAuth token for '{Path(path).name}'."
            )
            return creds

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
        """Lazily starts the queue processor if not already running."""
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(
                self._process_refresh_queue()
            )

    async def _ensure_reauth_processor_running(self):
        """Lazily starts the re-auth queue processor if not already running."""
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
                        lib_logger.warning(
                            f"Refresh timeout for '{Path(path).name}'"
                        )
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
                lib_logger.error(f"Error in refresh queue processor: {e}")
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)

    async def _handle_refresh_failure(self, path: str, force: bool, error: str):
        """Handle a refresh failure with back-of-line retry logic."""
        retry_count = self._queue_retry_count.get(path, 0) + 1
        self._queue_retry_count[path] = retry_count

        if retry_count >= self._refresh_max_retries:
            lib_logger.error(
                f"Max retries reached for '{Path(path).name}' (last error: {error})."
            )
            self._queue_retry_count.pop(path, None)
            async with self._queue_tracking_lock:
                self._queued_credentials.discard(path)
            return

        lib_logger.warning(
            f"Refresh failed for '{Path(path).name}' ({error}). "
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
                    lib_logger.info(f"Starting re-auth for '{Path(path).name}'...")
                    await self.initialize_token(path, force_interactive=True)
                    lib_logger.info(f"Re-auth SUCCESS for '{Path(path).name}'")
                except Exception as e:
                    lib_logger.error(f"Re-auth FAILED for '{Path(path).name}': {e}")
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
                lib_logger.error(f"Error in re-auth queue processor: {e}")
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)

    async def _perform_interactive_oauth(
        self, path: str, creds: Dict[str, Any], display_name: str
    ) -> Dict[str, Any]:
        """
        Perform interactive OAuth flow (browser-based authentication).
        Uses PKCE flow for OpenAI.
        """
        is_headless = is_headless_environment()

        # Generate PKCE codes
        code_verifier, code_challenge = _generate_pkce()
        state = secrets.token_hex(32)

        auth_code_future = asyncio.get_event_loop().create_future()
        server = None

        async def handle_callback(reader, writer):
            try:
                request_line_bytes = await reader.readline()
                if not request_line_bytes:
                    return
                path_str = request_line_bytes.decode("utf-8").strip().split(" ")[1]
                while await reader.readline() != b"\r\n":
                    pass

                from urllib.parse import urlparse, parse_qs
                query_params = parse_qs(urlparse(path_str).query)

                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n")

                if "code" in query_params:
                    received_state = query_params.get("state", [None])[0]
                    if received_state != state:
                        if not auth_code_future.done():
                            auth_code_future.set_exception(
                                Exception("OAuth state mismatch")
                            )
                        writer.write(
                            b"<html><body><h1>State Mismatch</h1><p>Security error. Please try again.</p></body></html>"
                        )
                    elif not auth_code_future.done():
                        auth_code_future.set_result(query_params["code"][0])
                        writer.write(
                            b"<html><body><h1>Authentication successful!</h1><p>You can close this window.</p></body></html>"
                        )
                else:
                    error = query_params.get("error", ["Unknown error"])[0]
                    if not auth_code_future.done():
                        auth_code_future.set_exception(Exception(f"OAuth failed: {error}"))
                    writer.write(
                        f"<html><body><h1>Authentication Failed</h1><p>Error: {error}</p></body></html>".encode()
                    )

                await writer.drain()
            except Exception as e:
                lib_logger.error(f"Error in OAuth callback handler: {e}")
            finally:
                writer.close()

        try:
            server = await asyncio.start_server(
                handle_callback, "127.0.0.1", self.callback_port
            )

            from urllib.parse import urlencode

            redirect_uri = f"http://localhost:{self.callback_port}{self.CALLBACK_PATH}"

            auth_params = {
                "response_type": "code",
                "client_id": self.CLIENT_ID,
                "redirect_uri": redirect_uri,
                "scope": " ".join(self.OAUTH_SCOPES),
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
                "state": state,
                "id_token_add_organizations": "true",
                "codex_cli_simplified_flow": "true",
            }

            auth_url = f"{self.AUTH_URL}?" + urlencode(auth_params)

            if is_headless:
                auth_panel_text = Text.from_markup(
                    "Running in headless environment (no GUI detected).\n"
                    "Please open the URL below in a browser on another machine to authorize:\n"
                )
            else:
                auth_panel_text = Text.from_markup(
                    "1. Your browser will now open to log in and authorize the application.\n"
                    "2. If it doesn't open automatically, please open the URL below manually."
                )

            console.print(
                Panel(
                    auth_panel_text,
                    title=f"{self.ENV_PREFIX} OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
                    style="bold blue",
                )
            )

            escaped_url = rich_escape(auth_url)
            console.print(f"[bold]URL:[/bold] [link={auth_url}]{escaped_url}[/link]\n")

            if not is_headless:
                try:
                    webbrowser.open(auth_url)
                    lib_logger.info("Browser opened successfully for OAuth flow")
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to open browser automatically: {e}. Please open the URL manually."
                    )

            with console.status(
                "[bold green]Waiting for you to complete authentication in the browser...[/bold green]",
                spinner="dots",
            ):
                auth_code = await asyncio.wait_for(auth_code_future, timeout=310)

        except asyncio.TimeoutError:
            raise Exception("OAuth flow timed out. Please try again.")
        finally:
            if server:
                server.close()
                await server.wait_closed()

        lib_logger.info("Exchanging authorization code for tokens...")

        async with httpx.AsyncClient() as client:
            redirect_uri = f"http://localhost:{self.callback_port}{self.CALLBACK_PATH}"

            response = await client.post(
                self.TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": auth_code.strip(),
                    "client_id": self.CLIENT_ID,
                    "code_verifier": code_verifier,
                    "redirect_uri": redirect_uri,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            token_data = response.json()

            # Build credentials
            new_creds = {
                "access_token": token_data.get("access_token"),
                "refresh_token": token_data.get("refresh_token"),
                "id_token": token_data.get("id_token"),
                "expiry_date": time.time() + token_data.get("expires_in", 3600),
            }

            # Parse ID token for claims
            id_token_claims = _parse_jwt_claims(token_data.get("id_token", "")) or {}
            access_token_claims = _parse_jwt_claims(token_data.get("access_token", "")) or {}

            # Extract account ID and email
            auth_claims = id_token_claims.get("https://api.openai.com/auth", {})
            account_id = auth_claims.get("chatgpt_account_id", "")
            org_id = id_token_claims.get("organization_id")
            project_id = id_token_claims.get("project_id")

            email = id_token_claims.get("email", "")
            plan_type = access_token_claims.get("chatgpt_plan_type", "")

            new_creds["account_id"] = account_id

            # Try to exchange for API key if we have org and project
            api_key = None
            if org_id and project_id:
                try:
                    api_key = await self._exchange_for_api_key(
                        client, token_data.get("id_token", "")
                    )
                    new_creds["api_key"] = api_key
                except Exception as e:
                    lib_logger.warning(f"API key exchange failed: {e}")

            new_creds["_proxy_metadata"] = {
                "email": email,
                "account_id": account_id,
                "org_id": org_id,
                "project_id": project_id,
                "plan_type": plan_type,
                "last_check_timestamp": time.time(),
            }

            if path:
                await self._save_credentials(path, new_creds)

            lib_logger.info(
                f"{self.ENV_PREFIX} OAuth initialized successfully for '{display_name}'."
            )

            return new_creds

    async def _exchange_for_api_key(
        self, client: httpx.AsyncClient, id_token: str
    ) -> Optional[str]:
        """
        Exchange ID token for an OpenAI API key.

        Uses the token exchange grant type to get a persistent API key.
        """
        import datetime

        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

        response = await client.post(
            self.TOKEN_URL,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                "client_id": self.CLIENT_ID,
                "requested_token": "openai-api-key",
                "subject_token": id_token,
                "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
                "name": f"LLM-API-Key-Proxy [auto-generated] ({today})",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        exchange_data = response.json()

        return exchange_data.get("access_token")

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

        lib_logger.debug(f"Initializing {self.ENV_PREFIX} token for '{display_name}'...")

        try:
            creds = (
                await self._load_credentials(creds_or_path) if path else creds_or_path
            )
            reason = ""

            if force_interactive:
                reason = "re-authentication was explicitly requested"
            elif not creds.get("refresh_token") and not creds.get("api_key"):
                reason = "refresh token and API key are missing"
            elif self._is_token_expired(creds) and not creds.get("api_key"):
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
                    f"{self.ENV_PREFIX} OAuth token for '{display_name}' needs setup: {reason}."
                )

                coordinator = get_reauth_coordinator()

                async def _do_interactive_oauth():
                    return await self._perform_interactive_oauth(path, creds, display_name)

                return await coordinator.execute_reauth(
                    credential_path=path or display_name,
                    provider_name=self.ENV_PREFIX,
                    reauth_func=_do_interactive_oauth,
                    timeout=300.0,
                )

            lib_logger.info(f"{self.ENV_PREFIX} OAuth token at '{display_name}' is valid.")
            return creds

        except Exception as e:
            raise ValueError(
                f"Failed to initialize {self.ENV_PREFIX} OAuth for '{path}': {e}"
            )

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        """Get auth header with graceful degradation if refresh fails."""
        try:
            creds = await self._load_credentials(credential_path)

            # Prefer API key if available
            if creds.get("api_key"):
                return {"Authorization": f"Bearer {creds['api_key']}"}

            # Fall back to access token
            if self._is_token_expired(creds):
                try:
                    creds = await self._refresh_token(credential_path, creds)
                except Exception as e:
                    cached = self._credentials_cache.get(credential_path)
                    if cached and (cached.get("access_token") or cached.get("api_key")):
                        lib_logger.warning(
                            f"Token refresh failed for {Path(credential_path).name}: {e}. "
                            "Using cached token."
                        )
                        creds = cached
                    else:
                        raise

            token = creds.get("api_key") or creds.get("access_token")
            return {"Authorization": f"Bearer {token}"}

        except Exception as e:
            cached = self._credentials_cache.get(credential_path)
            if cached and (cached.get("access_token") or cached.get("api_key")):
                lib_logger.error(
                    f"Credential load failed for {credential_path}: {e}. Using stale cached token."
                )
                token = cached.get("api_key") or cached.get("access_token")
                return {"Authorization": f"Bearer {token}"}
            raise

    async def get_account_id(self, credential_path: str) -> Optional[str]:
        """Get the ChatGPT account ID for a credential."""
        creds = await self._load_credentials(credential_path)
        return creds.get("account_id") or creds.get("_proxy_metadata", {}).get("account_id")

    async def proactively_refresh(self, credential_path: str):
        """Proactively refresh a credential by queueing it for refresh."""
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds) and not creds.get("api_key"):
            await self._queue_refresh(credential_path, force=False, needs_reauth=False)

    # =========================================================================
    # CREDENTIAL MANAGEMENT METHODS
    # =========================================================================

    def _get_provider_file_prefix(self) -> str:
        """Get the file name prefix for this provider's credential files."""
        return self.ENV_PREFIX.lower()

    def _get_oauth_base_dir(self) -> Path:
        """Get the base directory for OAuth credential files."""
        return Path.cwd() / "oauth_creds"

    def _find_existing_credential_by_email(
        self, email: str, base_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """Find an existing credential file for the given email."""
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
        """Get the next available credential number."""
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
        """Build a path for a new credential file."""
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
        """Complete credential setup flow: OAuth -> save -> discovery."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        base_dir.mkdir(exist_ok=True)

        try:
            temp_creds = {
                "_proxy_metadata": {"display_name": f"new {self.ENV_PREFIX} credential"}
            }
            new_creds = await self.initialize_token(temp_creds)

            email = new_creds.get("_proxy_metadata", {}).get("email")

            if not email:
                return CredentialSetupResult(
                    success=False, error="Could not retrieve email from OAuth response"
                )

            existing_path = self._find_existing_credential_by_email(email, base_dir)
            is_update = existing_path is not None

            if is_update:
                file_path = existing_path
            else:
                file_path = self._build_credential_path(base_dir)

            await self._save_credentials(str(file_path), new_creds)

            account_id = new_creds.get("account_id") or new_creds.get(
                "_proxy_metadata", {}
            ).get("account_id")

            return CredentialSetupResult(
                success=True,
                file_path=str(file_path),
                email=email,
                account_id=account_id,
                is_update=is_update,
                credentials=new_creds,
            )

        except Exception as e:
            lib_logger.error(f"Credential setup failed: {e}")
            return CredentialSetupResult(success=False, error=str(e))

    def list_credentials(self, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """List all credential files for this provider."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        credentials = []
        for cred_file in sorted(glob(pattern)):
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)

                metadata = creds.get("_proxy_metadata", {})

                match = re.search(r"_oauth_(\d+)\.json$", cred_file)
                number = int(match.group(1)) if match else 0

                credentials.append({
                    "file_path": cred_file,
                    "email": metadata.get("email", "unknown"),
                    "account_id": creds.get("account_id") or metadata.get("account_id"),
                    "number": number,
                })
            except Exception:
                continue

        return credentials
