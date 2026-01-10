# src/rotator_library/providers/codex_auth_base.py

import secrets
import hashlib
import base64
import json
import time
import asyncio
import logging
import webbrowser
import os
import re
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass, field
from pathlib import Path
from glob import glob
from typing import Dict, Any, Union, Optional, List

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.markup import escape as rich_escape

from ..utils.headless_detection import is_headless_environment
from ..utils.reauth_coordinator import get_reauth_coordinator
from ..utils.resilient_io import safe_write_json
from ..error_handler import CredentialNeedsReauthError

lib_logger = logging.getLogger("rotator_library")

# OAuth constants (from openai/codex and opencode plugin)
# Note: CLIENT_ID is public and safe to expose (used for native/mobile apps).
# OAuth endpoints are OpenAI's official OAuth endpoints for Codex CLI authentication.
CLIENT_ID = os.getenv("CODEX_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann")
AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
TOKEN_URL = "https://auth.openai.com/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
SCOPE = "openid profile email offline_access"

REFRESH_EXPIRY_BUFFER_SECONDS = 3 * 60 * 60  # 3 hours buffer before expiry

console = Console()


@dataclass
class CodexCredentialSetupResult:
    """
    Standardized result structure for Codex credential setup operations.
    """

    success: bool
    file_path: Optional[str] = None
    email: Optional[str] = None
    is_update: bool = False
    error: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = field(default=None, repr=False)


class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OAuth callback on localhost:1455."""

    def log_message(self, format, *args):
        """Suppress default HTTP logging."""
        pass

    def do_GET(self):
        """Handle OAuth callback."""
        try:
            from urllib.parse import urlparse, parse_qs

            url = urlparse(self.path)
            if url.path != "/auth/callback":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")
                return

            query = parse_qs(url.query)
            state = query.get("state", [None])[0]
            code = query.get("code", [None])[0]

            if not code:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing authorization code")
                return

            # Store result on server instance
            self.server.auth_result = (code, state)

            # Send success response
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><head><title>Authentication Successful</title></head>"
                b"<body style='font-family: Arial, sans-serif; text-align: center; padding: 50px;'>"
                b"<h1 style='color: #10a37f;'>Authentication Successful!</h1>"
                b"<p>You can close this window and return to the application.</p>"
                b"</body></html>"
            )
        except Exception as e:
            lib_logger.error(f"Error in OAuth callback handler: {e}")
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Internal error")


class CodexAuthBase:
    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = (
            asyncio.Lock()
        )  # Protects the locks dict from race conditions

        # [BACKOFF TRACKING] Track consecutive failures per credential
        self._refresh_failures: Dict[str, int] = {}
        self._next_refresh_after: Dict[str, float] = {}

        # [QUEUE SYSTEM] Sequential refresh processing with two separate queues
        self._refresh_queue: asyncio.Queue = asyncio.Queue()
        self._queue_processor_task: Optional[asyncio.Task] = None

        # Re-auth queue: for invalid refresh tokens (requires user interaction)
        self._reauth_queue: asyncio.Queue = asyncio.Queue()
        self._reauth_processor_task: Optional[asyncio.Task] = None

        # Tracking sets/dicts
        self._queued_credentials: set = set()
        self._unavailable_credentials: Dict[str, float] = {}
        self._unavailable_ttl_seconds: int = 360  # 6 minutes TTL
        self._queue_tracking_lock = asyncio.Lock()

        # Retry tracking for normal refresh queue
        self._queue_retry_count: Dict[str, int] = {}

        # Configuration constants
        self._refresh_timeout_seconds: int = 15
        self._refresh_interval_seconds: int = 30
        self._refresh_max_retries: int = 3
        self._reauth_timeout_seconds: int = 300

    def _parse_env_credential_path(self, path: str) -> Optional[str]:
        """
        Parse a virtual env:// path and return the credential index.

        Supported formats:
        - "env://codex/0" - Legacy single credential
        - "env://codex/1" - First numbered credential
        """
        if not path.startswith("env://"):
            return None

        parts = path[6:].split("/")
        if len(parts) >= 2:
            return parts[1]
        return "0"

    def _load_from_env(
        self, credential_index: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load OAuth credentials from environment variables.

        Expected environment variables (for numbered format with index N):
        - CODEX_{N}_ACCESS_TOKEN (required)
        - CODEX_{N}_REFRESH_TOKEN (required)
        - CODEX_{N}_EXPIRY_DATE (optional, defaults to 0)
        - CODEX_{N}_ACCOUNT_ID (optional)
        - CODEX_{N}_EMAIL (optional)
        """
        if credential_index and credential_index != "0":
            prefix = f"CODEX_{credential_index}"
            default_email = f"env-user-{credential_index}"
        else:
            prefix = "CODEX"
            default_email = "env-user"

        access_token = os.getenv(f"{prefix}_ACCESS_TOKEN")
        refresh_token = os.getenv(f"{prefix}_REFRESH_TOKEN")

        if not (access_token and refresh_token):
            return None

        lib_logger.debug(
            f"Loading Codex CLI credentials from environment variables (prefix: {prefix})"
        )

        expiry_str = os.getenv(f"{prefix}_EXPIRY_DATE", "0")
        try:
            expiry_date = float(expiry_str)
        except ValueError:
            lib_logger.warning(
                f"Invalid {prefix}_EXPIRY_DATE value: {expiry_str}, using 0"
            )
            expiry_date = 0

        creds = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": expiry_date,
            "account_id": os.getenv(f"{prefix}_ACCOUNT_ID", ""),
            "_proxy_metadata": {
                "email": os.getenv(f"{prefix}_EMAIL", default_email),
                "last_check_timestamp": time.time(),
                "loaded_from_env": True,
                "env_credential_index": credential_index or "0",
            },
        }

        return creds

    async def _read_creds_from_file(self, path: str) -> Dict[str, Any]:
        """Reads credentials from file and populates the cache."""
        try:
            lib_logger.debug(f"Reading Codex credentials from file: {path}")
            with open(path, "r") as f:
                creds = json.load(f)
            self._credentials_cache[path] = creds
            return creds
        except FileNotFoundError:
            raise IOError(f"Codex OAuth credential file not found at '{path}'")
        except Exception as e:
            raise IOError(f"Failed to load Codex OAuth credentials from '{path}': {e}")

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        """Loads credentials from cache, environment variables, or file."""
        if path in self._credentials_cache:
            return self._credentials_cache[path]

        async with await self._get_lock(path):
            # Re-check cache after acquiring lock
            if path in self._credentials_cache:
                return self._credentials_cache[path]

            # Check if this is a virtual env:// path
            credential_index = self._parse_env_credential_path(path)
            if credential_index is not None:
                env_creds = self._load_from_env(credential_index)
                if env_creds:
                    lib_logger.info(
                        f"Using Codex CLI credentials from environment variables (index: {credential_index})"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                else:
                    raise IOError(
                        f"Environment variables for Codex CLI credential index {credential_index} not found"
                    )

            # Try file-based loading
            try:
                return await self._read_creds_from_file(path)
            except IOError:
                # Fall back to legacy env vars
                env_creds = self._load_from_env()
                if env_creds:
                    lib_logger.info(
                        f"File '{path}' not found, using Codex CLI credentials from environment variables"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                raise

    async def _save_credentials(self, path: str, creds: Dict[str, Any]) -> bool:
        """Save credentials to disk, then update cache."""
        # Don't save to file if credentials were loaded from environment
        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            self._credentials_cache[path] = creds
            lib_logger.debug("Credentials loaded from env, skipping file save")
            return True

        if not safe_write_json(
            path, creds, lib_logger, secure_permissions=True, buffer_on_failure=False
        ):
            lib_logger.error(
                f"Failed to write Codex credentials to disk for '{Path(path).name}'. "
                f"Cache NOT updated to maintain parity with disk."
            )
            return False

        self._credentials_cache[path] = creds
        lib_logger.debug(
            f"Saved updated Codex OAuth credentials to '{Path(path).name}'."
        )
        return True

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        expiry_timestamp = creds.get("expiry_date", 0) / 1000
        return expiry_timestamp < time.time() + REFRESH_EXPIRY_BUFFER_SECONDS

    def _is_token_truly_expired(self, creds: Dict[str, Any]) -> bool:
        """Check if token is TRULY expired (past actual expiry)."""
        expiry_timestamp = creds.get("expiry_date", 0) / 1000
        return expiry_timestamp < time.time()

    @staticmethod
    def _extract_account_id_from_jwt(access_token: str) -> str:
        """
        Extract account_id from JWT's custom claim.

        JWT format: header.payload.signature
        Account ID is at: https://api.openai.com/auth -> chatgpt_account_id
        """
        try:
            parts = access_token.split(".")
            if len(parts) != 3:
                raise ValueError("Invalid JWT format")

            # Decode payload (base64url)
            payload = parts[1]
            # Add padding if needed
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding

            decoded = base64.urlsafe_b64decode(payload)
            claims = json.loads(decoded)

            # Navigate to custom claim
            # The account_id is stored as "chatgpt_account_id" under the custom claim
            auth_claim = claims.get("https://api.openai.com/auth", {})
            account_id = auth_claim.get("chatgpt_account_id") or auth_claim.get("account_id")

            if not account_id:
                raise ValueError("account_id not found in JWT")

            return account_id
        except Exception as e:
            lib_logger.error(f"Failed to extract account_id from JWT: {e}")
            raise ValueError(f"Could not extract account_id from access token: {e}")

    async def _refresh_token(self, path: str, force: bool = False) -> Dict[str, Any]:
        async with await self._get_lock(path):
            cached_creds = self._credentials_cache.get(path)
            if not force and cached_creds and not self._is_token_expired(cached_creds):
                return cached_creds

            credential_index = self._parse_env_credential_path(path)
            if credential_index is not None:
                env_creds = cached_creds if cached_creds else self._load_from_env(credential_index)
                if not env_creds:
                    raise IOError(
                        f"Environment variables for Codex CLI credential index {credential_index} not found"
                    )
                self._credentials_cache[path] = env_creds
                creds_from_file = env_creds
            else:
                # Always read fresh from disk before refresh
                await self._read_creds_from_file(path)
                creds_from_file = self._credentials_cache[path]

            lib_logger.debug(f"Refreshing Codex OAuth token for '{Path(path).name}'...")
            refresh_token = creds_from_file.get("refresh_token")
            if not refresh_token:
                lib_logger.error(f"No refresh_token found in '{Path(path).name}'")
                raise ValueError("No refresh_token found in Codex credentials file.")

            # Retry logic with exponential backoff
            max_retries = 3
            new_token_data = None
            last_error = None

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }

            async with httpx.AsyncClient() as client:
                for attempt in range(max_retries):
                    try:
                        response = await client.post(
                            TOKEN_URL,
                            headers=headers,
                            data={
                                "grant_type": "refresh_token",
                                "refresh_token": refresh_token,
                                "client_id": CLIENT_ID,
                            },
                            timeout=30.0,
                        )
                        response.raise_for_status()
                        new_token_data = response.json()
                        break

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        status_code = e.response.status_code
                        error_body = e.response.text
                        lib_logger.error(
                            f"HTTP {status_code} for '{Path(path).name}': {error_body}"
                        )

                        if status_code == 400:
                            try:
                                error_data = e.response.json()
                                error_type = error_data.get("error", "")
                                error_desc = error_data.get("error_description", "")
                            except Exception:
                                error_type = ""
                                error_desc = error_body

                            if "invalid" in error_desc.lower() or error_type == "invalid_request":
                                lib_logger.info(
                                    f"Credential '{Path(path).name}' needs re-auth (HTTP 400). "
                                    f"Queued for re-authentication."
                                )
                                asyncio.create_task(
                                    self._queue_refresh(
                                        path, force=True, needs_reauth=True
                                    )
                                )
                                raise CredentialNeedsReauthError(
                                    credential_path=path,
                                    message=f"Refresh token invalid for '{Path(path).name}'.",
                                )
                            else:
                                raise

                        elif status_code in (401, 403):
                            lib_logger.info(
                                f"Credential '{Path(path).name}' needs re-auth (HTTP {status_code}). "
                                f"Queued for re-authentication."
                            )
                            asyncio.create_task(
                                self._queue_refresh(path, force=True, needs_reauth=True)
                            )
                            raise CredentialNeedsReauthError(
                                credential_path=path,
                                message=f"Token invalid for '{Path(path).name}' (HTTP {status_code}).",
                            )

                        elif status_code == 429:
                            retry_after = int(e.response.headers.get("Retry-After", 60))
                            lib_logger.warning(f"Rate limited, retry after {retry_after}s")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_after)
                                continue
                            raise

                        elif 500 <= status_code < 600:
                            if attempt < max_retries - 1:
                                wait_time = 2**attempt
                                lib_logger.warning(f"Server error, retry {attempt + 1}/{max_retries}")
                                await asyncio.sleep(wait_time)
                                continue
                            raise

                        else:
                            raise

                    except (httpx.RequestError, httpx.TimeoutException) as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            wait_time = 2**attempt
                            await asyncio.sleep(wait_time)
                            continue
                        raise

            if new_token_data is None:
                self._refresh_failures[path] = self._refresh_failures.get(path, 0) + 1
                backoff_seconds = min(300, 30 * (2 ** self._refresh_failures[path]))
                self._next_refresh_after[path] = time.time() + backoff_seconds
                raise last_error or Exception("Token refresh failed")

            # Update credentials with new tokens
            creds_from_file["access_token"] = new_token_data["access_token"]
            creds_from_file["refresh_token"] = new_token_data.get(
                "refresh_token", creds_from_file["refresh_token"]
            )
            creds_from_file["expiry_date"] = (
                time.time() + new_token_data["expires_in"]
            ) * 1000

            # Extract and update account_id from new access token
            try:
                creds_from_file["account_id"] = self._extract_account_id_from_jwt(
                    new_token_data["access_token"]
                )
            except Exception as e:
                lib_logger.warning(f"Failed to extract account_id: {e}")

            # Update metadata
            if "_proxy_metadata" not in creds_from_file:
                creds_from_file["_proxy_metadata"] = {}
            creds_from_file["_proxy_metadata"]["last_check_timestamp"] = time.time()

            # Clear failure count
            self._refresh_failures.pop(path, None)
            self._next_refresh_after.pop(path, None)

            # Save credentials
            if not await self._save_credentials(path, creds_from_file):
                raise IOError(
                    f"Failed to persist refreshed credentials for '{Path(path).name}'."
                )

            lib_logger.debug(
                f"Successfully refreshed Codex OAuth token for '{Path(path).name}'."
            )
            return self._credentials_cache[path]

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        """Returns Authorization header for API requests."""
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_path)
        return {"Authorization": f"Bearer {creds['access_token']}"}

    async def proactively_refresh(self, credential_identifier: str):
        """Proactively refreshes tokens if they're close to expiry."""
        try:
            creds = await self._load_credentials(credential_identifier)
        except IOError:
            return

        if self._is_token_expired(creds):
            await self._queue_refresh(
                credential_identifier, force=False, needs_reauth=False
            )

    # =========================================================================
    # Queue and lock management (same as QwenAuthBase)
    # =========================================================================

    async def _get_lock(self, path: str) -> asyncio.Lock:
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

    async def _queue_refresh(
        self, path: str, force: bool = False, needs_reauth: bool = False
    ):
        """Add a credential to the appropriate refresh queue."""
        if not needs_reauth:
            now = time.time()
            if path in self._next_refresh_after:
                backoff_until = self._next_refresh_after[path]
                if now < backoff_until:
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

                    try:
                        async with asyncio.timeout(self._refresh_timeout_seconds):
                            await self._refresh_token(path, force=force)
                        self._queue_retry_count.pop(path, None)

                    except asyncio.TimeoutError:
                        await self._handle_refresh_failure(path, force, "timeout")

                    except httpx.HTTPStatusError as e:
                        status_code = e.response.status_code
                        needs_reauth = False

                        if status_code == 400:
                            try:
                                error_data = e.response.json()
                                error_desc = error_data.get("error_description", "")
                            except Exception:
                                error_desc = str(e)

                            if "invalid" in error_desc.lower():
                                needs_reauth = True

                        elif status_code in (401, 403):
                            needs_reauth = True

                        if needs_reauth:
                            self._queue_retry_count.pop(path, None)
                            async with self._queue_tracking_lock:
                                self._queued_credentials.discard(path)
                            await self._queue_refresh(
                                path, force=True, needs_reauth=True
                            )
                        else:
                            await self._handle_refresh_failure(
                                path, force, f"HTTP {status_code}"
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
                f"Max retries ({self._refresh_max_retries}) reached for '{Path(path).name}'"
            )
            self._queue_retry_count.pop(path, None)
            async with self._queue_tracking_lock:
                self._queued_credentials.discard(path)
            return

        lib_logger.warning(
            f"Refresh failed for '{Path(path).name}' ({error}). "
            f"Retry {retry_count}/{self._refresh_max_retries}"
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

    async def _start_oauth_server(self, state: str, timeout: int = 300) -> Optional[str]:
        """
        Start local HTTP server on port 1455 to receive OAuth callback.

        Returns authorization code or None if timeout/error.
        """
        server_result = {"code": None, "ready": False}

        def run_server():
            nonlocal server_result
            try:
                server = HTTPServer(("127.0.0.1", 1455), CallbackHandler)
                server.auth_result = None
                server.timeout = 1

                # Check if we can bind to the port
                server_result["ready"] = True

                # Wait for callback
                start_time = time.time()
                while time.time() - start_time < timeout:
                    server.handle_request()
                    if hasattr(server, "auth_result") and server.auth_result:
                        code, returned_state = server.auth_result
                        if returned_state == state:
                            server_result["code"] = code
                        break
                    time.sleep(0.1)

            except OSError as e:
                if e.errno == 10048:  # Windows: Address already in use
                    lib_logger.warning("Port 1455 already in use")
                server_result["ready"] = False
            except Exception as e:
                lib_logger.error(f"OAuth server error: {e}")
                server_result["ready"] = False

        # Run server in background thread
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        time.sleep(0.5)  # Give server time to start

        if not server_result["ready"]:
            return None

        # Wait for result
        thread.join(timeout=timeout)

        return server_result["code"]

    async def _perform_interactive_oauth(
        self, path: str, creds: Dict[str, Any], display_name: str
    ) -> Dict[str, Any]:
        """
        Perform interactive OAuth authorization code flow with PKCE.
        """
        is_headless = is_headless_environment()

        # Generate PKCE pair
        code_verifier = (
            base64.urlsafe_b64encode(secrets.token_bytes(32))
            .decode("utf-8")
            .rstrip("=")
        )
        code_challenge = (
            base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode("utf-8")).digest()
            )
            .decode("utf-8")
            .rstrip("=")
        )

        # Generate state
        state = secrets.token_hex(16)

        # Build authorization URL
        from urllib.parse import urlencode

        auth_params = {
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPE,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
            "id_token_add_organizations": "true",
            "codex_simplified_flow": "true",
            "originator": "codex_rs",
        }

        auth_url = f"{AUTHORIZE_URL}?{urlencode(auth_params)}"

        # Display instructions
        if is_headless:
            auth_panel_text = Text.from_markup(
                "Running in headless environment.\n"
                "Please open the URL below in a browser to authorize:\n"
            )
        else:
            auth_panel_text = Text.from_markup(
                "Opening browser for authorization...\n"
                "If browser doesn't open, please visit the URL below manually:\n"
            )

        console.print(
            Panel(
                auth_panel_text,
                title=f"Codex CLI OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
                style="bold blue",
            )
        )

        escaped_url = rich_escape(auth_url)
        console.print(f"[bold]URL:[/bold] [link={auth_url}]{escaped_url}[/link]\n")

        # Try to open browser
        if not is_headless:
            try:
                webbrowser.open(auth_url)
                lib_logger.info("Browser opened for Codex OAuth flow")
            except Exception as e:
                lib_logger.warning(f"Failed to open browser: {e}")

        # Start local server and wait for callback
        console.status(
            "[bold green]Waiting for OAuth callback...[/bold green]",
            spinner="dots",
        )

        auth_code = await self._start_oauth_server(state, timeout=300)

        if not auth_code:
            # Fallback: ask user to paste code manually
            console.print("\n[yellow]Callback server failed or timed out.[/yellow]")
            console.print("Please complete authorization in the browser, then:")
            console.print("1. Copy the 'code' parameter from the callback URL")
            console.print("2. Paste it below\n")

            auth_code = Prompt.ask("[bold]Enter authorization code[/bold]")

        # Exchange code for tokens
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                TOKEN_URL,
                headers=headers,
                data={
                    "grant_type": "authorization_code",
                    "client_id": CLIENT_ID,
                    "code": auth_code,
                    "code_verifier": code_verifier,
                    "redirect_uri": REDIRECT_URI,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            token_data = response.json()

        # Extract account_id from JWT
        try:
            account_id = self._extract_account_id_from_jwt(token_data["access_token"])
        except Exception as e:
            lib_logger.warning(f"Failed to extract account_id: {e}")
            account_id = ""

        # Update credentials
        creds.update(
            {
                "access_token": token_data["access_token"],
                "refresh_token": token_data["refresh_token"],
                "expiry_date": (time.time() + token_data["expires_in"]) * 1000,
                "account_id": account_id,
            }
        )

        # Ensure we have a stable identifier without prompting for email
        metadata = creds.get("_proxy_metadata") or {}
        if not metadata.get("email"):
            if account_id:
                fallback_email = f"account-{account_id}"
                email_source = "account_id"
            else:
                fallback_email = "unknown"
                email_source = "fallback"
            metadata.update(
                {
                    "email": fallback_email,
                    "last_check_timestamp": time.time(),
                    "email_source": email_source,
                }
            )
            creds["_proxy_metadata"] = metadata

        # Save credentials
        if path:
            if not await self._save_credentials(path, creds):
                raise IOError(
                    f"Failed to save OAuth credentials to disk for '{display_name}'."
                )

        lib_logger.info(f"Codex OAuth initialized successfully for '{display_name}'.")
        return creds

    async def initialize_token(
        self,
        creds_or_path: Union[Dict[str, Any], str],
        force_interactive: bool = False,
    ) -> Dict[str, Any]:
        """
        Initialize OAuth token, triggering interactive flow if needed.
        """
        path = creds_or_path if isinstance(creds_or_path, str) else None

        if isinstance(creds_or_path, dict):
            display_name = creds_or_path.get("_proxy_metadata", {}).get(
                "display_name", "in-memory object"
            )
        else:
            display_name = Path(path).name if path else "in-memory object"

        lib_logger.debug(f"Initializing Codex token for '{display_name}'...")

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
                        return await self._refresh_token(path)
                    except Exception as e:
                        lib_logger.warning(
                            f"Automatic token refresh failed: {e}. Proceeding to interactive login."
                        )

                lib_logger.warning(
                    f"Codex OAuth token for '{display_name}' needs setup: {reason}."
                )

                coordinator = get_reauth_coordinator()

                async def _do_interactive_oauth():
                    return await self._perform_interactive_oauth(
                        path, creds, display_name
                    )

                return await coordinator.execute_reauth(
                    credential_path=path or display_name,
                    provider_name="CODEX",
                    reauth_func=_do_interactive_oauth,
                    timeout=300.0,
                )

            lib_logger.info(f"Codex OAuth token at '{display_name}' is valid.")
            return creds

        except Exception as e:
            raise ValueError(f"Failed to initialize Codex OAuth for '{path}': {e}")

    # =========================================================================
    # CREDENTIAL MANAGEMENT METHODS
    # =========================================================================

    def _get_provider_file_prefix(self) -> str:
        return "codex"

    def _get_oauth_base_dir(self) -> Path:
        from ..utils.paths import get_oauth_dir
        return get_oauth_dir()

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
            except (json.JSONDecodeError, IOError):
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
    ) -> CodexCredentialSetupResult:
        """Complete credential setup flow: OAuth -> save."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        base_dir.mkdir(exist_ok=True)

        try:
            temp_creds = {
                "_proxy_metadata": {"display_name": "new Codex CLI credential"}
            }
            new_creds = await self.initialize_token(temp_creds)

            email = new_creds.get("_proxy_metadata", {}).get("email")
            if not email:
                account_id = new_creds.get("account_id", "")
                email = f"account-{account_id}" if account_id else "unknown"
                new_creds.setdefault("_proxy_metadata", {})["email"] = email

            existing_path = self._find_existing_credential_by_email(email, base_dir)
            is_update = existing_path is not None

            if is_update:
                file_path = existing_path
            else:
                file_path = self._build_credential_path(base_dir)

            if not await self._save_credentials(str(file_path), new_creds):
                return CodexCredentialSetupResult(
                    success=False,
                    error=f"Failed to save credentials to disk",
                )

            return CodexCredentialSetupResult(
                success=True,
                file_path=str(file_path),
                email=email,
                is_update=is_update,
                credentials=new_creds,
            )

        except Exception as e:
            lib_logger.error(f"Credential setup failed: {e}")
            return CodexCredentialSetupResult(success=False, error=str(e))

    def build_env_lines(self, creds: Dict[str, Any], cred_number: int) -> List[str]:
        """Generate .env file lines for a Codex credential."""
        email = creds.get("_proxy_metadata", {}).get("email", "unknown")
        prefix = f"CODEX_{cred_number}"

        lines = [
            f"# CODEX Credential #{cred_number} for: {email}",
            f"# Exported from: codex_oauth_{cred_number}.json",
            f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"{prefix}_ACCESS_TOKEN={creds.get('access_token', '')}",
            f"{prefix}_REFRESH_TOKEN={creds.get('refresh_token', '')}",
            f"{prefix}_EXPIRY_DATE={creds.get('expiry_date', 0)}",
            f"{prefix}_ACCOUNT_ID={creds.get('account_id', '')}",
            f"{prefix}_EMAIL={email}",
        ]

        return lines

    def export_credential_to_env(
        self, credential_path: str, output_dir: Optional[Path] = None
    ) -> Optional[str]:
        """Export a credential file to .env format."""
        try:
            cred_path = Path(credential_path)

            with open(cred_path, "r") as f:
                creds = json.load(f)

            email = creds.get("_proxy_metadata", {}).get("email", "unknown")

            match = re.search(r"_oauth_(\d+)\.json$", cred_path.name)
            cred_number = int(match.group(1)) if match else 1

            if output_dir is None:
                output_dir = cred_path.parent

            safe_email = email.replace("@", "_at_").replace(".", "_")
            env_filename = f"codex_{cred_number}_{safe_email}.env"
            env_path = output_dir / env_filename

            env_lines = self.build_env_lines(creds, cred_number)
            with open(env_path, "w") as f:
                f.write("\n".join(env_lines))

            lib_logger.info(f"Exported credential to {env_path}")
            return str(env_path)

        except Exception as e:
            lib_logger.error(f"Failed to export credential: {e}")
            return None

    def list_credentials(self, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """List all Codex credential files."""
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

                credentials.append(
                    {
                        "file_path": cred_file,
                        "email": metadata.get("email", "unknown"),
                        "number": number,
                    }
                )
            except Exception as e:
                lib_logger.debug(f"Could not read credential file {cred_file}: {e}")
                continue

        return credentials

    def delete_credential(self, credential_path: str) -> bool:
        """Delete a credential file."""
        try:
            cred_path = Path(credential_path)

            prefix = self._get_provider_file_prefix()
            if not cred_path.name.startswith(f"{prefix}_oauth_"):
                lib_logger.error(
                    f"File {cred_path.name} does not appear to be a Codex CLI credential"
                )
                return False

            if not cred_path.exists():
                lib_logger.warning(f"Credential file does not exist: {credential_path}")
                return False

            self._credentials_cache.pop(credential_path, None)
            cred_path.unlink()
            lib_logger.info(f"Deleted credential file: {credential_path}")
            return True

        except Exception as e:
            lib_logger.error(f"Failed to delete credential: {e}")
            return False
