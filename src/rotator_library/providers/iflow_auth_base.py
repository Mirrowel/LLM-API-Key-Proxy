# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/iflow_auth_base.py

import secrets
import base64
import json
import time
import asyncio
import logging
import webbrowser
import socket
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from glob import glob
from typing import Dict, Any, Tuple, Union, Optional, List
from urllib.parse import urlencode, parse_qs, urlparse

import httpx
from aiohttp import web
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

IFLOW_OAUTH_AUTHORIZE_ENDPOINT = "https://iflow.cn/oauth"
IFLOW_OAUTH_TOKEN_ENDPOINT = "https://iflow.cn/oauth/token"
IFLOW_USER_INFO_ENDPOINT = "https://iflow.cn/api/oauth/getUserInfo"
IFLOW_SUCCESS_REDIRECT_URL = "https://iflow.cn/oauth/success"
IFLOW_ERROR_REDIRECT_URL = "https://iflow.cn/oauth/error"

# Client credentials provided by iFlow
IFLOW_CLIENT_ID = "10009311001"
IFLOW_CLIENT_SECRET = "4Z3YjXycVsQvyGF1etiNlIBB4RsqSDtW"

# Local callback server port
CALLBACK_PORT = 11451


@dataclass
class IFlowCredentialSetupResult:
    """
    Standardized result structure for iFlow credential setup operations.
    """

    success: bool
    file_path: Optional[str] = None
    email: Optional[str] = None
    is_update: bool = False
    error: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = field(default=None, repr=False)


def get_callback_port() -> int:
    """
    Get the OAuth callback port, checking environment variable first.

    Reads from IFLOW_OAUTH_PORT environment variable, falling back
    to the default CALLBACK_PORT if not set.
    """
    env_value = os.getenv("IFLOW_OAUTH_PORT")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            logging.getLogger("rotator_library").warning(
                f"Invalid IFLOW_OAUTH_PORT value: {env_value}, using default {CALLBACK_PORT}"
            )
    return CALLBACK_PORT


# Refresh tokens 24 hours before expiry
REFRESH_EXPIRY_BUFFER_SECONDS = 24 * 60 * 60

console = Console()


class OAuthCallbackServer:
    """
    Minimal HTTP server for handling iFlow OAuth callbacks.
    """

    def __init__(self, port: int = CALLBACK_PORT):
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.result_future: Optional[asyncio.Future] = None
        self.expected_state: Optional[str] = None

    def _is_port_available(self) -> bool:
        """Checks if the callback port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", self.port))
            sock.close()
            return True
        except OSError:
            return False

    async def start(self, expected_state: str):
        """Starts the OAuth callback server."""
        if not self._is_port_available():
            raise RuntimeError(f"Port {self.port} is already in use")

        self.expected_state = expected_state
        self.result_future = asyncio.Future()

        # Setup route
        self.app.router.add_get("/oauth2callback", self._handle_callback)

        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "localhost", self.port)
        await self.site.start()

        lib_logger.debug(f"iFlow OAuth callback server started on port {self.port}")

    async def stop(self):
        """Stops the OAuth callback server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        lib_logger.debug("iFlow OAuth callback server stopped")

    async def _handle_callback(self, request: web.Request) -> web.Response:
        """Handles the OAuth callback request."""
        query = request.query

        # Check for error parameter
        if "error" in query:
            error = query.get("error", "unknown_error")
            lib_logger.error(f"iFlow OAuth callback received error: {error}")
            if not self.result_future.done():
                self.result_future.set_exception(ValueError(f"OAuth error: {error}"))
            return web.Response(
                status=302, headers={"Location": IFLOW_ERROR_REDIRECT_URL}
            )

        # Check for authorization code
        code = query.get("code")
        if not code:
            lib_logger.error("iFlow OAuth callback missing authorization code")
            if not self.result_future.done():
                self.result_future.set_exception(
                    ValueError("Missing authorization code")
                )
            return web.Response(
                status=302, headers={"Location": IFLOW_ERROR_REDIRECT_URL}
            )

        # Validate state parameter
        state = query.get("state", "")
        if state != self.expected_state:
            lib_logger.error(
                f"iFlow OAuth state mismatch. Expected: {self.expected_state}, Got: {state}"
            )
            if not self.result_future.done():
                self.result_future.set_exception(ValueError("State parameter mismatch"))
            return web.Response(
                status=302, headers={"Location": IFLOW_ERROR_REDIRECT_URL}
            )

        # Success - set result and redirect to success page
        if not self.result_future.done():
            self.result_future.set_result(code)

        return web.Response(
            status=302, headers={"Location": IFLOW_SUCCESS_REDIRECT_URL}
        )

    async def wait_for_callback(self, timeout: float = 300.0) -> str:
        """Waits for the OAuth callback and returns the authorization code."""
        try:
            code = await asyncio.wait_for(self.result_future, timeout=timeout)
            return code
        except asyncio.TimeoutError:
            raise TimeoutError("Timeout waiting for OAuth callback")


class IFlowAuthBase:
    """
    iFlow OAuth authentication base class.
    Implements authorization code flow with local callback server.
    """

    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = (
            asyncio.Lock()
        )  # Protects the locks dict from race conditions
        # [BACKOFF TRACKING] Track consecutive failures per credential
        self._refresh_failures: Dict[
            str, int
        ] = {}  # Track consecutive failures per credential
        self._next_refresh_after: Dict[
            str, float
        ] = {}  # Track backoff timers (Unix timestamp)

        # [QUEUE SYSTEM] Sequential refresh processing with two separate queues
        # Normal refresh queue: for proactive token refresh (old token still valid)
        self._refresh_queue: asyncio.Queue = asyncio.Queue()
        self._queue_processor_task: Optional[asyncio.Task] = None

        # Re-auth queue: for invalid refresh tokens (requires user interaction)
        self._reauth_queue: asyncio.Queue = asyncio.Queue()
        self._reauth_processor_task: Optional[asyncio.Task] = None

        # Tracking sets/dicts
        self._queued_credentials: set = set()  # Track credentials in either queue
        # Only credentials in re-auth queue are marked unavailable (not normal refresh)
        # TTL cleanup is defense-in-depth for edge cases where re-auth processor crashes
        self._unavailable_credentials: Dict[
            str, float
        ] = {}  # Maps credential path -> timestamp when marked unavailable
        # TTL should exceed reauth timeout (300s) to avoid premature cleanup
        self._unavailable_ttl_seconds: int = 360  # 6 minutes TTL for stale entries
        self._queue_tracking_lock = asyncio.Lock()  # Protects queue sets

        # Retry tracking for normal refresh queue
        self._queue_retry_count: Dict[
            str, int
        ] = {}  # Track retry attempts per credential

        # Configuration constants
        self._refresh_timeout_seconds: int = 15  # Max time for single refresh
        self._refresh_interval_seconds: int = 30  # Delay between queue items
        self._refresh_max_retries: int = 3  # Attempts before kicked out
        self._reauth_timeout_seconds: int = 300  # Time for user to complete OAuth

    def _parse_env_credential_path(self, path: str) -> Optional[str]:
        """
        Parse a virtual env:// path and return the credential index.

        Supported formats:
        - "env://provider/0" - Legacy single credential (no index in env var names)
        - "env://provider/1" - First numbered credential (IFLOW_1_ACCESS_TOKEN)

        Returns:
            The credential index as string, or None if path is not an env:// path
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
        Load OAuth credentials from environment variables for stateless deployments.

        Supports two formats:
        1. Legacy (credential_index="0" or None): IFLOW_ACCESS_TOKEN
        2. Numbered (credential_index="1", "2", etc.): IFLOW_1_ACCESS_TOKEN, etc.

        Expected environment variables (for numbered format with index N):
        - IFLOW_{N}_ACCESS_TOKEN (required)
        - IFLOW_{N}_REFRESH_TOKEN (required)
        - IFLOW_{N}_API_KEY (required - critical for iFlow!)
        - IFLOW_{N}_EXPIRY_DATE (optional, defaults to empty string)
        - IFLOW_{N}_EMAIL (optional, defaults to "env-user-{N}")
        - IFLOW_{N}_TOKEN_TYPE (optional, defaults to "Bearer")
        - IFLOW_{N}_SCOPE (optional, defaults to "read write")

        Returns:
            Dict with credential structure if env vars present, None otherwise
        """
        # Determine the env var prefix based on credential index
        if credential_index and credential_index != "0":
            prefix = f"IFLOW_{credential_index}"
            default_email = f"env-user-{credential_index}"
        else:
            prefix = "IFLOW"
            default_email = "env-user"

        access_token = os.getenv(f"{prefix}_ACCESS_TOKEN")
        refresh_token = os.getenv(f"{prefix}_REFRESH_TOKEN")
        api_key = os.getenv(f"{prefix}_API_KEY")

        # All three are required for iFlow
        if not (access_token and refresh_token and api_key):
            return None

        lib_logger.debug(
            f"Loading iFlow credentials from environment variables (prefix: {prefix})"
        )

        # Parse expiry_date as string (ISO 8601 format)
        expiry_str = os.getenv(f"{prefix}_EXPIRY_DATE", "")

        creds = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "api_key": api_key,  # Critical for iFlow!
            "expiry_date": expiry_str,
            "email": os.getenv(f"{prefix}_EMAIL", default_email),
            "token_type": os.getenv(f"{prefix}_TOKEN_TYPE", "Bearer"),
            "scope": os.getenv(f"{prefix}_SCOPE", "read write"),
            "_proxy_metadata": {
                "email": os.getenv(f"{prefix}_EMAIL", default_email),
                "last_check_timestamp": time.time(),
                "loaded_from_env": True,
                "env_credential_index": credential_index or "0",
            },
        }

        return creds

    async def _read_creds_from_file(self, path: str) -> Dict[str, Any]:
        """Reads credentials from file and populates the cache. No locking."""
        try:
            lib_logger.debug(f"Reading iFlow credentials from file: {path}")
            with open(path, "r") as f:
                creds = json.load(f)
            self._credentials_cache[path] = creds
            return creds
        except FileNotFoundError:
            raise IOError(f"iFlow OAuth credential file not found at '{path}'")
        except Exception as e:
            raise IOError(f"Failed to load iFlow OAuth credentials from '{path}': {e}")

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
                        f"Using iFlow credentials from environment variables (index: {credential_index})"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                else:
                    raise IOError(
                        f"Environment variables for iFlow credential index {credential_index} not found"
                    )

            # Try file-based loading first (preferred for explicit file paths)
            try:
                return await self._read_creds_from_file(path)
            except IOError:
                # File not found - fall back to legacy env vars for backwards compatibility
                env_creds = self._load_from_env()
                if env_creds:
                    lib_logger.info(
                        f"File '{path}' not found, using iFlow credentials from environment variables"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                raise  # Re-raise the original file not found error

    async def _save_credentials(self, path: str, creds: Dict[str, Any]) -> bool:
        """Save credentials to disk, then update cache. Returns True only if disk write succeeded.

        For providers with rotating refresh tokens, disk persistence is CRITICAL.
        If we update the cache but fail to write to disk:
        - The old refresh_token on disk may become invalid (consumed by API)
        - On restart, we'd load the invalid token and require re-auth

        By writing to disk FIRST, we ensure:
        - Cache only updated after disk succeeds (guaranteed parity)
        - If disk fails, cache keeps old tokens, refresh is retried
        - No desync between cache and disk is possible
        """
        # Don't save to file if credentials were loaded from environment
        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            self._credentials_cache[path] = creds
            lib_logger.debug("Credentials loaded from env, skipping file save")
            return True

        # Write to disk FIRST - do NOT buffer on failure for rotating tokens
        # Buffering is dangerous because the refresh_token may be stale by retry time
        if not safe_write_json(
            path, creds, lib_logger, secure_permissions=True, buffer_on_failure=False
        ):
            lib_logger.error(
                f"Failed to write iFlow credentials to disk for '{Path(path).name}'. "
                f"Cache NOT updated to maintain parity with disk."
            )
            return False

        # Disk write succeeded - now update cache (guaranteed parity)
        self._credentials_cache[path] = creds
        lib_logger.debug(
            f"Saved updated iFlow OAuth credentials to '{Path(path).name}'."
        )
        return True

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        """Checks if the token is expired (with buffer for proactive refresh)."""
        # Try to parse expiry_date as ISO 8601 string
        expiry_str = creds.get("expiry_date")
        if not expiry_str:
            return True

        try:
            # Parse ISO 8601 format (e.g., "2025-01-17T12:00:00Z")
            from datetime import datetime

            expiry_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            expiry_timestamp = expiry_dt.timestamp()
        except (ValueError, AttributeError):
            # Fallback: treat as numeric timestamp
            try:
                expiry_timestamp = float(expiry_str)
            except (ValueError, TypeError):
                lib_logger.warning(f"Could not parse expiry_date: {expiry_str}")
                return True

        return expiry_timestamp < time.time() + REFRESH_EXPIRY_BUFFER_SECONDS

    def _is_token_truly_expired(self, creds: Dict[str, Any]) -> bool:
        """Check if token is TRULY expired (past actual expiry, not just threshold).

        This is different from _is_token_expired() which uses a buffer for proactive refresh.
        This method checks if the token is actually unusable.
        """
        expiry_str = creds.get("expiry_date")
        if not expiry_str:
            return True

        try:
            from datetime import datetime

            expiry_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
            expiry_timestamp = expiry_dt.timestamp()
        except (ValueError, AttributeError):
            try:
                expiry_timestamp = float(expiry_str)
            except (ValueError, TypeError):
                return True

        return expiry_timestamp < time.time()

    async def _fetch_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        Fetches user info (including API key) from iFlow API.
        This is critical: iFlow uses a separate API key for actual API calls.
        """
        if not access_token or not access_token.strip():
            raise ValueError("Access token is empty")

        url = f"{IFLOW_USER_INFO_ENDPOINT}?accessToken={access_token}"
        headers = {"Accept": "application/json"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()

        if not result.get("success"):
            raise ValueError("iFlow user info request not successful")

        data = result.get("data", {})
        api_key = data.get("apiKey", "").strip()
        if not api_key:
            raise ValueError("Missing API key in user info response")

        email = data.get("email", "").strip()
        if not email:
            email = data.get("phone", "").strip()
        if not email:
            raise ValueError("Missing email/phone in user info response")

        return {"api_key": api_key, "email": email}

    async def _exchange_code_for_tokens(
        self, code: str, redirect_uri: str
    ) -> Dict[str, Any]:
        """
        Exchanges authorization code for access and refresh tokens.
        Uses Basic Auth with client credentials.
        """
        # Create Basic Auth header
        auth_string = f"{IFLOW_CLIENT_ID}:{IFLOW_CLIENT_SECRET}"
        basic_auth = base64.b64encode(auth_string.encode()).decode()

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Authorization": f"Basic {basic_auth}",
        }

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": IFLOW_CLIENT_ID,
            "client_secret": IFLOW_CLIENT_SECRET,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                IFLOW_OAUTH_TOKEN_ENDPOINT, headers=headers, data=data
            )

            if response.status_code != 200:
                error_text = response.text
                lib_logger.error(
                    f"iFlow token exchange failed: {response.status_code} {error_text}"
                )
                raise ValueError(
                    f"Token exchange failed: {response.status_code} {error_text}"
                )

            token_data = response.json()

        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError("Missing access_token in token response")

        refresh_token = token_data.get("refresh_token", "")
        expires_in = token_data.get("expires_in", 3600)
        token_type = token_data.get("token_type", "Bearer")
        scope = token_data.get("scope", "")

        # Fetch user info to get API key
        user_info = await self._fetch_user_info(access_token)

        # Calculate expiry date
        from datetime import datetime, timedelta

        expiry_date = (
            datetime.utcnow() + timedelta(seconds=expires_in)
        ).isoformat() + "Z"

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "api_key": user_info["api_key"],
            "email": user_info["email"],
            "expiry_date": expiry_date,
            "token_type": token_type,
            "scope": scope,
        }

    async def _refresh_token(self, path: str, force: bool = False) -> Dict[str, Any]:
        """
        Refreshes the OAuth tokens and re-fetches the API key.
        CRITICAL: Must re-fetch user info to get potentially updated API key.
        """
        async with await self._get_lock(path):
            cached_creds = self._credentials_cache.get(path)
            if not force and cached_creds and not self._is_token_expired(cached_creds):
                return cached_creds

            # [ROTATING TOKEN FIX] Always read fresh from disk before refresh.
            # iFlow may use rotating refresh tokens - each refresh could invalidate the previous token.
            # If we use a stale cached token, refresh will fail.
            # Reading fresh from disk ensures we have the latest token.
            await self._read_creds_from_file(path)
            creds_from_file = self._credentials_cache[path]

            lib_logger.debug(f"Refreshing iFlow OAuth token for '{Path(path).name}'...")
            refresh_token = creds_from_file.get("refresh_token")
            if not refresh_token:
                raise ValueError("No refresh_token found in iFlow credentials file.")

            # [RETRY LOGIC] Implement exponential backoff for transient errors
            max_retries = 3
            new_token_data = None
            last_error = None

            # Create Basic Auth header
            auth_string = f"{IFLOW_CLIENT_ID}:{IFLOW_CLIENT_SECRET}"
            basic_auth = base64.b64encode(auth_string.encode()).decode()

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "Authorization": f"Basic {basic_auth}",
            }

            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": IFLOW_CLIENT_ID,
                "client_secret": IFLOW_CLIENT_SECRET,
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                for attempt in range(max_retries):
                    try:
                        response = await client.post(
                            IFLOW_OAUTH_TOKEN_ENDPOINT, headers=headers, data=data
                        )
                        response.raise_for_status()
                        new_token_data = response.json()

                        # [FIX] Handle wrapped response format: {success: bool, data: {...}}
                        # iFlow API may return tokens nested inside a 'data' key
                        if (
                            isinstance(new_token_data, dict)
                            and "data" in new_token_data
                        ):
                            lib_logger.debug(
                                f"iFlow refresh response wrapped in 'data' key, extracting..."
                            )
                            # Check for error in wrapped response
                            if not new_token_data.get("success", True):
                                error_msg = new_token_data.get(
                                    "message", "Unknown error"
                                )
                                raise ValueError(
                                    f"iFlow token refresh failed: {error_msg}"
                                )
                            new_token_data = new_token_data.get("data", {})

                        break  # Success

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        status_code = e.response.status_code
                        error_body = e.response.text

                        lib_logger.error(
                            f"[REFRESH HTTP ERROR] HTTP {status_code} for '{Path(path).name}': {error_body}"
                        )

                        # [STATUS CODE HANDLING]
                        # [INVALID GRANT HANDLING] Handle 400/401/403 by raising
                        # Queue for re-auth in background so credential gets fixed automatically
                        if status_code == 400:
                            # Check if this is an invalid refresh token error
                            try:
                                error_data = e.response.json()
                                error_type = error_data.get("error", "")
                                error_desc = error_data.get("error_description", "")
                                if not error_desc:
                                    error_desc = error_data.get("message", error_body)
                            except Exception:
                                error_type = ""
                                error_desc = error_body

                            if (
                                "invalid" in error_desc.lower()
                                or error_type == "invalid_request"
                            ):
                                lib_logger.info(
                                    f"Credential '{Path(path).name}' needs re-auth (HTTP 400: {error_desc}). "
                                    f"Queued for re-authentication, rotating to next credential."
                                )
                                # Queue for re-auth in background (non-blocking, fire-and-forget)
                                # This ensures credential gets fixed even if caller doesn't handle it
                                asyncio.create_task(
                                    self._queue_refresh(
                                        path, force=True, needs_reauth=True
                                    )
                                )
                                # Raise rotatable error instead of raw HTTPStatusError
                                raise CredentialNeedsReauthError(
                                    credential_path=path,
                                    message=f"Refresh token invalid for '{Path(path).name}'. Re-auth queued.",
                                )
                            else:
                                # Other 400 error - raise it
                                raise

                        elif status_code in (401, 403):
                            lib_logger.info(
                                f"Credential '{Path(path).name}' needs re-auth (HTTP {status_code}). "
                                f"Queued for re-authentication, rotating to next credential."
                            )
                            # Queue for re-auth in background (non-blocking, fire-and-forget)
                            asyncio.create_task(
                                self._queue_refresh(path, force=True, needs_reauth=True)
                            )
                            # Raise rotatable error instead of raw HTTPStatusError
                            raise CredentialNeedsReauthError(
                                credential_path=path,
                                message=f"Token invalid for '{Path(path).name}' (HTTP {status_code}). Re-auth queued.",
                            )

                        elif status_code == 429:
                            retry_after = int(e.response.headers.get("Retry-After", 60))
                            lib_logger.warning(
                                f"Rate limited (HTTP 429), retry after {retry_after}s"
                            )
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_after)
                                continue
                            raise

                        elif 500 <= status_code < 600:
                            if attempt < max_retries - 1:
                                wait_time = 2**attempt
                                lib_logger.warning(
                                    f"Server error (HTTP {status_code}), retry {attempt + 1}/{max_retries} in {wait_time}s"
                                )
                                await asyncio.sleep(wait_time)
                                continue
                            raise

                        else:
                            raise

                    except (httpx.RequestError, httpx.TimeoutException) as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            wait_time = 2**attempt
                            lib_logger.warning(
                                f"Network error during refresh: {e}, retry {attempt + 1}/{max_retries} in {wait_time}s"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        raise

            if new_token_data is None:
                # [BACKOFF TRACKING] Increment failure count and set backoff timer
                self._refresh_failures[path] = self._refresh_failures.get(path, 0) + 1
                backoff_seconds = min(
                    300, 30 * (2 ** self._refresh_failures[path])
                )  # Max 5 min backoff
                self._next_refresh_after[path] = time.time() + backoff_seconds
                lib_logger.debug(
                    f"Setting backoff for '{Path(path).name}': {backoff_seconds}s"
                )
                raise last_error or Exception("Token refresh failed after all retries")

            # Update tokens
            access_token = new_token_data.get("access_token")
            if not access_token:
                # Log response keys for debugging
                response_keys = (
                    list(new_token_data.keys())
                    if isinstance(new_token_data, dict)
                    else type(new_token_data).__name__
                )
                lib_logger.error(
                    f"Missing access_token in refresh response for '{Path(path).name}'. "
                    f"Response keys: {response_keys}"
                )
                raise ValueError("Missing access_token in refresh response")

            creds_from_file["access_token"] = access_token
            creds_from_file["refresh_token"] = new_token_data.get(
                "refresh_token", creds_from_file["refresh_token"]
            )

            expires_in = new_token_data.get("expires_in", 3600)
            from datetime import datetime, timedelta

            creds_from_file["expiry_date"] = (
                datetime.utcnow() + timedelta(seconds=expires_in)
            ).isoformat() + "Z"

            creds_from_file["token_type"] = new_token_data.get(
                "token_type", creds_from_file.get("token_type", "Bearer")
            )
            creds_from_file["scope"] = new_token_data.get(
                "scope", creds_from_file.get("scope", "")
            )

            # CRITICAL: Re-fetch user info to get potentially updated API key
            try:
                user_info = await self._fetch_user_info(access_token)
                if user_info.get("api_key"):
                    creds_from_file["api_key"] = user_info["api_key"]
                if user_info.get("email"):
                    creds_from_file["email"] = user_info["email"]
            except Exception as e:
                lib_logger.warning(
                    f"Failed to update API key during token refresh: {e}"
                )

            # Ensure _proxy_metadata exists and update timestamp
            if "_proxy_metadata" not in creds_from_file:
                creds_from_file["_proxy_metadata"] = {}
            creds_from_file["_proxy_metadata"]["last_check_timestamp"] = time.time()

            # [VALIDATION] Verify required fields exist after refresh
            required_fields = ["access_token", "refresh_token", "api_key"]
            missing_fields = [
                field for field in required_fields if not creds_from_file.get(field)
            ]
            if missing_fields:
                raise ValueError(
                    f"Refreshed credentials missing required fields: {missing_fields}"
                )

            # [BACKOFF TRACKING] Clear failure count on successful refresh
            self._refresh_failures.pop(path, None)
            self._next_refresh_after.pop(path, None)

            # Save credentials - MUST succeed for rotating token providers
            if not await self._save_credentials(path, creds_from_file):
                # CRITICAL: If we can't persist the new token, the old token may be
                # invalidated. This is a critical failure - raise so retry logic kicks in.
                raise IOError(
                    f"Failed to persist refreshed credentials for '{Path(path).name}'. "
                    f"Disk write failed - refresh will be retried."
                )

            lib_logger.debug(
                f"Successfully refreshed iFlow OAuth token for '{Path(path).name}'."
            )
            return self._credentials_cache[path]  # Return from cache (synced with disk)

    async def get_api_details(self, credential_identifier: str) -> Tuple[str, str]:
        """
        Returns the API base URL and API key (NOT access_token).
        CRITICAL: iFlow uses the api_key for API requests, not the OAuth access_token.

        Supports both credential types:
        - OAuth: credential_identifier is a file path to JSON credentials
        - API Key: credential_identifier is the API key string itself
        """
        # Detect credential type
        if os.path.isfile(credential_identifier):
            # OAuth credential: file path to JSON
            lib_logger.debug(
                f"Using OAuth credentials from file: {credential_identifier}"
            )
            creds = await self._load_credentials(credential_identifier)

            # Check if token needs refresh
            if self._is_token_expired(creds):
                creds = await self._refresh_token(credential_identifier)

            api_key = creds.get("api_key")
            if not api_key:
                raise ValueError("Missing api_key in iFlow OAuth credentials")
        else:
            # Direct API key: use as-is
            lib_logger.debug("Using direct API key for iFlow")
            api_key = credential_identifier

        base_url = "https://apis.iflow.cn/v1"
        return base_url, api_key

    async def proactively_refresh(self, credential_identifier: str):
        """
        Proactively refreshes tokens if they're close to expiry.
        Only applies to OAuth credentials (file paths or env:// paths). Direct API keys are skipped.
        """
        # lib_logger.debug(f"proactively_refresh called for: {credential_identifier}")

        # Try to load credentials - this will fail for direct API keys
        # and succeed for OAuth credentials (file paths or env:// paths)
        try:
            creds = await self._load_credentials(credential_identifier)
        except IOError as e:
            # Not a valid credential path (likely a direct API key string)
            # lib_logger.debug(
            #     f"Skipping refresh for '{credential_identifier}' - not an OAuth credential: {e}"
            # )
            return

        is_expired = self._is_token_expired(creds)
        # lib_logger.debug(
        #     f"Token expired check for '{Path(credential_identifier).name}': {is_expired}"
        # )

        if is_expired:
            # lib_logger.debug(
            #     f"Queueing refresh for '{Path(credential_identifier).name}'"
            # )
            # lib_logger.info(f"Proactive refresh triggered for '{Path(credential_identifier).name}'")
            await self._queue_refresh(
                credential_identifier, force=False, needs_reauth=False
            )

    async def _get_lock(self, path: str) -> asyncio.Lock:
        """Gets or creates a lock for the given credential path."""
        # [FIX RACE CONDITION] Protect lock creation with a master lock
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

    def is_credential_available(self, path: str) -> bool:
        """Check if a credential is available for rotation.

        Credentials are unavailable if:
        1. In re-auth queue (token is truly broken, requires user interaction)
        2. Token is TRULY expired (past actual expiry, not just threshold)

        Note: Credentials in normal refresh queue are still available because
        the old token is valid until actual expiry.

        TTL cleanup (defense-in-depth): If a credential has been in the re-auth
        queue longer than _unavailable_ttl_seconds without being processed, it's
        cleaned up. This should only happen if the re-auth processor crashes or
        is cancelled without proper cleanup.
        """
        # Check if in re-auth queue (truly unavailable)
        if path in self._unavailable_credentials:
            marked_time = self._unavailable_credentials.get(path)
            if marked_time is not None:
                now = time.time()
                if now - marked_time > self._unavailable_ttl_seconds:
                    # Entry is stale - clean it up and return available
                    # This is a defense-in-depth for edge cases where re-auth
                    # processor crashed or was cancelled without cleanup
                    lib_logger.warning(
                        f"Credential '{Path(path).name}' stuck in re-auth queue for "
                        f"{int(now - marked_time)}s (TTL: {self._unavailable_ttl_seconds}s). "
                        f"Re-auth processor may have crashed. Auto-cleaning stale entry."
                    )
                    # Clean up both tracking structures for consistency
                    self._unavailable_credentials.pop(path, None)
                    self._queued_credentials.discard(path)
                else:
                    return False  # Still in re-auth, not available

        # Check if token is TRULY expired (not just threshold-expired)
        creds = self._credentials_cache.get(path)
        if creds and self._is_token_truly_expired(creds):
            # Token is actually expired - should not be used
            # Queue for refresh if not already queued
            if path not in self._queued_credentials:
                # lib_logger.debug(
                #     f"Credential '{Path(path).name}' is truly expired, queueing for refresh"
                # )
                asyncio.create_task(
                    self._queue_refresh(path, force=True, needs_reauth=False)
                )
            return False

        return True

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

    async def _queue_refresh(
        self, path: str, force: bool = False, needs_reauth: bool = False
    ):
        """Add a credential to the appropriate refresh queue if not already queued.

        Args:
            path: Credential file path
            force: Force refresh even if not expired
            needs_reauth: True if full re-authentication needed (routes to re-auth queue)

        Queue routing:
        - needs_reauth=True: Goes to re-auth queue, marks as unavailable
        - needs_reauth=False: Goes to normal refresh queue, does NOT mark unavailable
          (old token is still valid until actual expiry)
        """
        # IMPORTANT: Only check backoff for simple automated refreshes
        # Re-authentication (interactive OAuth) should BYPASS backoff since it needs user input
        if not needs_reauth:
            now = time.time()
            if path in self._next_refresh_after:
                backoff_until = self._next_refresh_after[path]
                if now < backoff_until:
                    # Credential is in backoff for automated refresh, do not queue
                    # remaining = int(backoff_until - now)
                    # lib_logger.debug(
                    #     f"Skipping automated refresh for '{Path(path).name}' (in backoff for {remaining}s)"
                    # )
                    return

        async with self._queue_tracking_lock:
            if path not in self._queued_credentials:
                self._queued_credentials.add(path)

                if needs_reauth:
                    # Re-auth queue: mark as unavailable (token is truly broken)
                    self._unavailable_credentials[path] = time.time()
                    # lib_logger.debug(
                    #     f"Queued '{Path(path).name}' for RE-AUTH (marked unavailable). "
                    #     f"Total unavailable: {len(self._unavailable_credentials)}"
                    # )
                    await self._reauth_queue.put(path)
                    await self._ensure_reauth_processor_running()
                else:
                    # Normal refresh queue: do NOT mark unavailable (old token still valid)
                    # lib_logger.debug(
                    #     f"Queued '{Path(path).name}' for refresh (still available). "
                    #     f"Queue size: {self._refresh_queue.qsize() + 1}"
                    # )
                    await self._refresh_queue.put((path, force))
                    await self._ensure_queue_processor_running()

    async def _process_refresh_queue(self):
        """Background worker that processes normal refresh requests sequentially.

        Key behaviors:
        - 15s timeout per refresh operation
        - 30s delay between processing credentials (prevents thundering herd)
        - On failure: back of queue, max 3 retries before kicked
        - If 401/403 detected: routes to re-auth queue
        - Does NOT mark credentials unavailable (old token still valid)
        """
        # lib_logger.info("Refresh queue processor started")
        while True:
            path = None
            try:
                # Wait for an item with timeout to allow graceful shutdown
                try:
                    path, force = await asyncio.wait_for(
                        self._refresh_queue.get(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    # Queue is empty and idle for 60s - clean up and exit
                    async with self._queue_tracking_lock:
                        # Clear any stale retry counts
                        self._queue_retry_count.clear()
                    self._queue_processor_task = None
                    # lib_logger.debug("Refresh queue processor idle, shutting down")
                    return

                try:
                    # Quick check if still expired (optimization to avoid unnecessary refresh)
                    creds = self._credentials_cache.get(path)
                    if creds and not self._is_token_expired(creds):
                        # No longer expired, skip refresh
                        # lib_logger.debug(
                        #     f"Credential '{Path(path).name}' no longer expired, skipping refresh"
                        # )
                        # Clear retry count on skip (not a failure)
                        self._queue_retry_count.pop(path, None)
                        continue

                    # Perform refresh with timeout
                    try:
                        async with asyncio.timeout(self._refresh_timeout_seconds):
                            await self._refresh_token(path, force=force)

                        # SUCCESS: Clear retry count
                        self._queue_retry_count.pop(path, None)
                        # lib_logger.info(f"Refresh SUCCESS for '{Path(path).name}'")

                    except asyncio.TimeoutError:
                        lib_logger.warning(
                            f"Refresh timeout ({self._refresh_timeout_seconds}s) for '{Path(path).name}'"
                        )
                        await self._handle_refresh_failure(path, force, "timeout")

                    except httpx.HTTPStatusError as e:
                        status_code = e.response.status_code
                        # Check for invalid refresh token errors (400/401/403)
                        # These need to be routed to re-auth queue for interactive OAuth
                        needs_reauth = False

                        if status_code == 400:
                            # Check if this is an invalid refresh token error
                            try:
                                error_data = e.response.json()
                                error_type = error_data.get("error", "")
                                error_desc = error_data.get("error_description", "")
                                if not error_desc:
                                    error_desc = error_data.get("message", str(e))
                            except Exception:
                                error_type = ""
                                error_desc = str(e)

                            if (
                                "invalid" in error_desc.lower()
                                or error_type == "invalid_request"
                            ):
                                needs_reauth = True
                                lib_logger.info(
                                    f"Credential '{Path(path).name}' needs re-auth (HTTP 400: {error_desc}). "
                                    f"Routing to re-auth queue."
                                )
                        elif status_code in (401, 403):
                            needs_reauth = True
                            lib_logger.info(
                                f"Credential '{Path(path).name}' needs re-auth (HTTP {status_code}). "
                                f"Routing to re-auth queue."
                            )

                        if needs_reauth:
                            self._queue_retry_count.pop(path, None)  # Clear retry count
                            async with self._queue_tracking_lock:
                                self._queued_credentials.discard(
                                    path
                                )  # Remove from queued
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
                    # Remove from queued set (unless re-queued by failure handler)
                    async with self._queue_tracking_lock:
                        # Only discard if not re-queued (check if still in queue set from retry)
                        if (
                            path in self._queued_credentials
                            and self._queue_retry_count.get(path, 0) == 0
                        ):
                            self._queued_credentials.discard(path)
                    self._refresh_queue.task_done()

                # Wait between credentials to spread load
                await asyncio.sleep(self._refresh_interval_seconds)

            except asyncio.CancelledError:
                # lib_logger.debug("Refresh queue processor cancelled")
                break
            except Exception as e:
                lib_logger.error(f"Error in refresh queue processor: {e}")
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)

    async def _handle_refresh_failure(self, path: str, force: bool, error: str):
        """Handle a refresh failure with back-of-line retry logic.

        - Increments retry count
        - If under max retries: re-adds to END of queue
        - If at max retries: kicks credential out (retried next BackgroundRefresher cycle)
        """
        retry_count = self._queue_retry_count.get(path, 0) + 1
        self._queue_retry_count[path] = retry_count

        if retry_count >= self._refresh_max_retries:
            # Kicked out until next BackgroundRefresher cycle
            lib_logger.error(
                f"Max retries ({self._refresh_max_retries}) reached for '{Path(path).name}' "
                f"(last error: {error}). Will retry next refresh cycle."
            )
            self._queue_retry_count.pop(path, None)
            async with self._queue_tracking_lock:
                self._queued_credentials.discard(path)
            return

        # Re-add to END of queue for retry
        lib_logger.warning(
            f"Refresh failed for '{Path(path).name}' ({error}). "
            f"Retry {retry_count}/{self._refresh_max_retries}, back of queue."
        )
        # Keep in queued_credentials set, add back to queue
        await self._refresh_queue.put((path, force))

    async def _process_reauth_queue(self):
        """Background worker that processes re-auth requests.

        Key behaviors:
        - Credentials ARE marked unavailable (token is truly broken)
        - Uses ReauthCoordinator for interactive OAuth
        - No automatic retry (requires user action)
        - Cleans up unavailable status when done
        """
        # lib_logger.info("Re-auth queue processor started")
        while True:
            path = None
            try:
                # Wait for an item with timeout to allow graceful shutdown
                try:
                    path = await asyncio.wait_for(
                        self._reauth_queue.get(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    # Queue is empty and idle for 60s - exit
                    self._reauth_processor_task = None
                    # lib_logger.debug("Re-auth queue processor idle, shutting down")
                    return

                try:
                    lib_logger.info(f"Starting re-auth for '{Path(path).name}'...")
                    await self.initialize_token(path, force_interactive=True)
                    lib_logger.info(f"Re-auth SUCCESS for '{Path(path).name}'")

                except Exception as e:
                    lib_logger.error(f"Re-auth FAILED for '{Path(path).name}': {e}")
                    # No automatic retry for re-auth (requires user action)

                finally:
                    # Always clean up
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)
                        # lib_logger.debug(
                        #     f"Re-auth cleanup for '{Path(path).name}'. "
                        #     f"Remaining unavailable: {len(self._unavailable_credentials)}"
                        # )
                    self._reauth_queue.task_done()

            except asyncio.CancelledError:
                # Clean up current credential before breaking
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)
                # lib_logger.debug("Re-auth queue processor cancelled")
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
        Perform interactive OAuth authorization code flow (browser-based authentication).

        This method is called via the global ReauthCoordinator to ensure
        only one interactive OAuth flow runs at a time across all providers.

        Args:
            path: Credential file path
            creds: Current credentials dict (will be updated)
            display_name: Display name for logging/UI

        Returns:
            Updated credentials dict with new tokens
        """
        # [HEADLESS DETECTION] Check if running in headless environment
        is_headless = is_headless_environment()

        # Generate random state for CSRF protection
        state = secrets.token_urlsafe(32)

        # Build authorization URL
        callback_port = get_callback_port()
        redirect_uri = f"http://localhost:{callback_port}/oauth2callback"
        auth_params = {
            "loginMethod": "phone",
            "type": "phone",
            "redirect": redirect_uri,
            "state": state,
            "client_id": IFLOW_CLIENT_ID,
        }
        auth_url = f"{IFLOW_OAUTH_AUTHORIZE_ENDPOINT}?{urlencode(auth_params)}"

        # Start OAuth callback server
        callback_server = OAuthCallbackServer(port=callback_port)
        try:
            await callback_server.start(expected_state=state)

            # [HEADLESS SUPPORT] Display appropriate instructions
            if is_headless:
                auth_panel_text = Text.from_markup(
                    "Running in headless environment (no GUI detected).\n"
                    "Please open the URL below in a browser on another machine to authorize:\n"
                    "1. Visit the URL below to sign in with your phone number.\n"
                    "2. [bold]Authorize the application[/bold] to access your account.\n"
                    "3. You will be automatically redirected after authorization."
                )
            else:
                auth_panel_text = Text.from_markup(
                    "1. Visit the URL below to sign in with your phone number.\n"
                    "2. [bold]Authorize the application[/bold] to access your account.\n"
                    "3. You will be automatically redirected after authorization."
                )

            console.print(
                Panel(
                    auth_panel_text,
                    title=f"iFlow OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
                    style="bold blue",
                )
            )
            escaped_url = rich_escape(auth_url)
            console.print(f"[bold]URL:[/bold] [link={auth_url}]{escaped_url}[/link]\n")

            # [HEADLESS SUPPORT] Only attempt browser open if NOT headless
            if not is_headless:
                try:
                    webbrowser.open(auth_url)
                    lib_logger.info("Browser opened successfully for iFlow OAuth flow")
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to open browser automatically: {e}. Please open the URL manually."
                    )

            # Wait for callback
            with console.status(
                "[bold green]Waiting for authorization in the browser...[/bold green]",
                spinner="dots",
            ):
                # Note: The 300s timeout here is handled by the ReauthCoordinator
                # We use a slightly longer internal timeout to let the coordinator handle it
                code = await callback_server.wait_for_callback(timeout=310.0)

            lib_logger.info("Received authorization code, exchanging for tokens...")

            # Exchange code for tokens and API key
            token_data = await self._exchange_code_for_tokens(code, redirect_uri)

            # Update credentials
            creds.update(
                {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data["refresh_token"],
                    "api_key": token_data["api_key"],
                    "email": token_data["email"],
                    "expiry_date": token_data["expiry_date"],
                    "token_type": token_data["token_type"],
                    "scope": token_data["scope"],
                }
            )

            # Create metadata object
            if not creds.get("_proxy_metadata"):
                creds["_proxy_metadata"] = {
                    "email": token_data["email"],
                    "last_check_timestamp": time.time(),
                }

            if path:
                if not await self._save_credentials(path, creds):
                    raise IOError(
                        f"Failed to save OAuth credentials to disk for '{display_name}'. "
                        f"Please retry authentication."
                    )

            lib_logger.info(
                f"iFlow OAuth initialized successfully for '{display_name}'."
            )
            return creds

        finally:
            await callback_server.stop()

    async def initialize_token(
        self,
        creds_or_path: Union[Dict[str, Any], str],
        force_interactive: bool = False,
    ) -> Dict[str, Any]:
        """
        Initialize OAuth token, triggering interactive authorization flow if needed.

        If interactive OAuth is required (expired refresh token, missing credentials, etc.),
        the flow is coordinated globally via ReauthCoordinator to ensure only one
        interactive OAuth flow runs at a time across all providers.

        Args:
            creds_or_path: Either a credentials dict or path to credentials file.
            force_interactive: If True, skip expiry checks and force interactive OAuth.
                               Use this when the refresh token is known to be invalid
                               (e.g., after HTTP 400 from token endpoint).
        """
        path = creds_or_path if isinstance(creds_or_path, str) else None

        # Get display name from metadata if available, otherwise derive from path
        if isinstance(creds_or_path, dict):
            display_name = creds_or_path.get("_proxy_metadata", {}).get(
                "display_name", "in-memory object"
            )
        else:
            display_name = Path(path).name if path else "in-memory object"

        lib_logger.debug(f"Initializing iFlow token for '{display_name}'...")

        try:
            creds = (
                await self._load_credentials(creds_or_path) if path else creds_or_path
            )

            reason = ""
            if force_interactive:
                reason = (
                    "re-authentication was explicitly requested (refresh token invalid)"
                )
            elif not creds.get("refresh_token"):
                reason = "refresh token is missing"
            elif self._is_token_expired(creds):
                reason = "token is expired"

            if reason:
                # Try automatic refresh first if we have a refresh token
                if reason == "token is expired" and creds.get("refresh_token"):
                    try:
                        return await self._refresh_token(path)
                    except Exception as e:
                        lib_logger.warning(
                            f"Automatic token refresh for '{display_name}' failed: {e}. Proceeding to interactive login."
                        )

                # Interactive OAuth flow
                lib_logger.warning(
                    f"iFlow OAuth token for '{display_name}' needs setup: {reason}."
                )

                # [GLOBAL REAUTH COORDINATION] Use the global coordinator to ensure
                # only one interactive OAuth flow runs at a time across all providers
                coordinator = get_reauth_coordinator()

                # Define the interactive OAuth function to be executed by coordinator
                async def _do_interactive_oauth():
                    return await self._perform_interactive_oauth(
                        path, creds, display_name
                    )

                # Execute via global coordinator (ensures only one at a time)
                return await coordinator.execute_reauth(
                    credential_path=path or display_name,
                    provider_name="IFLOW",
                    reauth_func=_do_interactive_oauth,
                    timeout=300.0,  # 5 minute timeout for user to complete OAuth
                )

            lib_logger.info(f"iFlow OAuth token at '{display_name}' is valid.")
            return creds

        except Exception as e:
            raise ValueError(f"Failed to initialize iFlow OAuth for '{path}': {e}")

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        """
        Returns auth header with API key (NOT OAuth access_token).
        CRITICAL: iFlow API requests use the api_key, not the OAuth tokens.
        """
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_path)

        api_key = creds.get("api_key")
        if not api_key:
            raise ValueError("Missing api_key in iFlow credentials")

        return {"Authorization": f"Bearer {api_key}"}

    async def get_user_info(
        self, creds_or_path: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        """Retrieves user info from the _proxy_metadata in the credential file."""
        try:
            path = creds_or_path if isinstance(creds_or_path, str) else None
            creds = (
                await self._load_credentials(creds_or_path) if path else creds_or_path
            )

            # Ensure the token is valid
            if path:
                await self.initialize_token(path)
                creds = await self._load_credentials(path)

            email = creds.get("email") or creds.get("_proxy_metadata", {}).get("email")

            if not email:
                lib_logger.warning(
                    f"No email found in iFlow credentials for '{path or 'in-memory object'}'."
                )

            # Update timestamp in cache only (not disk) to avoid overwriting
            # potentially newer tokens that were saved by another process/refresh.
            # The timestamp is non-critical metadata - losing it on restart is fine.
            if path and "_proxy_metadata" in creds:
                creds["_proxy_metadata"]["last_check_timestamp"] = time.time()
                # Note: We intentionally don't save to disk here because:
                # 1. The cache may have older tokens than disk (if external refresh occurred)
                # 2. Saving would overwrite the newer disk tokens with stale cached ones
                # 3. The timestamp is non-critical and will be updated on next refresh

            return {"email": email}
        except Exception as e:
            lib_logger.error(f"Failed to get iFlow user info from credentials: {e}")
            return {"email": None}

    # =========================================================================
    # CREDENTIAL MANAGEMENT METHODS
    # =========================================================================

    def _get_provider_file_prefix(self) -> str:
        """Return the file prefix for iFlow credentials."""
        return "iflow"

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
                existing_email = creds.get("email") or creds.get(
                    "_proxy_metadata", {}
                ).get("email")
                if existing_email == email:
                    return Path(cred_file)
            except (json.JSONDecodeError, IOError) as e:
                lib_logger.debug(f"Could not read credential file {cred_file}: {e}")
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
    ) -> IFlowCredentialSetupResult:
        """
        Complete credential setup flow: OAuth -> save.

        This is the main entry point for setting up new credentials.
        """
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        # Ensure directory exists
        base_dir.mkdir(exist_ok=True)

        try:
            # Step 1: Perform OAuth authentication
            temp_creds = {"_proxy_metadata": {"display_name": "new iFlow credential"}}
            new_creds = await self.initialize_token(temp_creds)

            # Step 2: Get user info for deduplication
            email = new_creds.get("email") or new_creds.get("_proxy_metadata", {}).get(
                "email"
            )

            if not email:
                return IFlowCredentialSetupResult(
                    success=False, error="Could not retrieve email from OAuth response"
                )

            # Step 3: Check for existing credential with same email
            existing_path = self._find_existing_credential_by_email(email, base_dir)
            is_update = existing_path is not None

            if is_update:
                file_path = existing_path
                lib_logger.info(
                    f"Found existing credential for {email}, updating {file_path.name}"
                )
            else:
                file_path = self._build_credential_path(base_dir)
                lib_logger.info(
                    f"Creating new credential for {email} at {file_path.name}"
                )

            # Step 4: Save credentials to file
            if not await self._save_credentials(str(file_path), new_creds):
                return IFlowCredentialSetupResult(
                    success=False,
                    error=f"Failed to save credentials to disk at {file_path.name}",
                )

            return IFlowCredentialSetupResult(
                success=True,
                file_path=str(file_path),
                email=email,
                is_update=is_update,
                credentials=new_creds,
            )

        except Exception as e:
            lib_logger.error(f"Credential setup failed: {e}")
            return IFlowCredentialSetupResult(success=False, error=str(e))

    def build_env_lines(self, creds: Dict[str, Any], cred_number: int) -> List[str]:
        """Generate .env file lines for an iFlow credential."""
        email = creds.get("email") or creds.get("_proxy_metadata", {}).get(
            "email", "unknown"
        )
        prefix = f"IFLOW_{cred_number}"

        lines = [
            f"# IFLOW Credential #{cred_number} for: {email}",
            f"# Exported from: iflow_oauth_{cred_number}.json",
            f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "#",
            "# To combine multiple credentials into one .env file, copy these lines",
            "# and ensure each credential has a unique number (1, 2, 3, etc.)",
            "",
            f"{prefix}_ACCESS_TOKEN={creds.get('access_token', '')}",
            f"{prefix}_REFRESH_TOKEN={creds.get('refresh_token', '')}",
            f"{prefix}_API_KEY={creds.get('api_key', '')}",
            f"{prefix}_EXPIRY_DATE={creds.get('expiry_date', '')}",
            f"{prefix}_EMAIL={email}",
            f"{prefix}_TOKEN_TYPE={creds.get('token_type', 'Bearer')}",
            f"{prefix}_SCOPE={creds.get('scope', 'read write')}",
        ]

        return lines

    def export_credential_to_env(
        self, credential_path: str, output_dir: Optional[Path] = None
    ) -> Optional[str]:
        """Export a credential file to .env format."""
        try:
            cred_path = Path(credential_path)

            # Load credential
            with open(cred_path, "r") as f:
                creds = json.load(f)

            # Extract metadata
            email = creds.get("email") or creds.get("_proxy_metadata", {}).get(
                "email", "unknown"
            )

            # Get credential number from filename
            match = re.search(r"_oauth_(\d+)\.json$", cred_path.name)
            cred_number = int(match.group(1)) if match else 1

            # Build output path
            if output_dir is None:
                output_dir = cred_path.parent

            safe_email = email.replace("@", "_at_").replace(".", "_")
            env_filename = f"iflow_{cred_number}_{safe_email}.env"
            env_path = output_dir / env_filename

            # Build and write content
            env_lines = self.build_env_lines(creds, cred_number)
            with open(env_path, "w") as f:
                f.write("\n".join(env_lines))

            lib_logger.info(f"Exported credential to {env_path}")
            return str(env_path)

        except Exception as e:
            lib_logger.error(f"Failed to export credential: {e}")
            return None

    def list_credentials(self, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """List all iFlow credential files."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_oauth_*.json")

        credentials = []
        for cred_file in sorted(glob(pattern)):
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)

                email = creds.get("email") or creds.get("_proxy_metadata", {}).get(
                    "email", "unknown"
                )

                # Extract number from filename
                match = re.search(r"_oauth_(\d+)\.json$", cred_file)
                number = int(match.group(1)) if match else 0

                credentials.append(
                    {
                        "file_path": cred_file,
                        "email": email,
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

            # Validate that it's one of our credential files
            prefix = self._get_provider_file_prefix()
            if not cred_path.name.startswith(f"{prefix}_oauth_"):
                lib_logger.error(
                    f"File {cred_path.name} does not appear to be an iFlow credential"
                )
                return False

            if not cred_path.exists():
                lib_logger.warning(f"Credential file does not exist: {credential_path}")
                return False

            # Remove from cache if present
            self._credentials_cache.pop(credential_path, None)

            # Delete the file
            cred_path.unlink()
            lib_logger.info(f"Deleted credential file: {credential_path}")
            return True

        except Exception as e:
            lib_logger.error(f"Failed to delete credential: {e}")
            return False
