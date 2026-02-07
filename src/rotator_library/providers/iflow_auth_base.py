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

# OAuth endpoints
IFLOW_OAUTH_AUTHORIZE_ENDPOINT = "https://iflow.cn/oauth"
IFLOW_OAUTH_TOKEN_ENDPOINT = "https://iflow.cn/oauth/token"
IFLOW_USER_INFO_ENDPOINT = "https://iflow.cn/api/oauth/getUserInfo"
IFLOW_SUCCESS_REDIRECT_URL = "https://iflow.cn/oauth/success"
IFLOW_ERROR_REDIRECT_URL = "https://iflow.cn/oauth/error"

# Cookie-based authentication endpoint
IFLOW_API_KEY_ENDPOINT = "https://platform.iflow.cn/api/openapi/apikey"

# Client credentials provided by iFlow
IFLOW_CLIENT_ID = "10009311001"
IFLOW_CLIENT_SECRET = "4Z3YjXycVsQvyGF1etiNlIBB4RsqSDtW"

# Local callback server port
CALLBACK_PORT = 11451

# Cookie API key refresh buffer (48 hours before expiry)
COOKIE_REFRESH_BUFFER_HOURS = 48


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


def normalize_cookie(raw: str) -> str:
    """
    Normalize and validate a cookie string for iFlow authentication.

    Ensures the cookie contains the required BXAuth field and is properly formatted.

    Args:
        raw: Raw cookie string from user input

    Returns:
        Normalized cookie string ending with semicolon

    Raises:
        ValueError: If cookie is empty or missing BXAuth field
    """
    trimmed = raw.strip()
    if not trimmed:
        raise ValueError("Cookie cannot be empty")

    # Normalize whitespace
    combined = " ".join(trimmed.split())

    # Ensure ends with semicolon
    if not combined.endswith(";"):
        combined += ";"

    # Validate BXAuth field is present
    if "BXAuth=" not in combined:
        raise ValueError(
            "Cookie missing required 'BXAuth' field. "
            "Please copy the complete cookie including BXAuth."
        )

    return combined


def extract_bx_auth(cookie: str) -> Optional[str]:
    """
    Extract the BXAuth value from a cookie string.

    Args:
        cookie: Cookie string (e.g., "BXAuth=abc123; other=value;")

    Returns:
        The BXAuth value, or None if not found
    """
    parts = cookie.split(";")
    for part in parts:
        part = part.strip()
        if part.startswith("BXAuth="):
            return part[7:]  # Remove "BXAuth=" prefix
    return None


def should_refresh_cookie_api_key(expire_time: str) -> Tuple[bool, float]:
    """
    Check if a cookie-based API key needs refresh.

    Uses a 48-hour buffer to proactively refresh
    API keys before they expire.

    Args:
        expire_time: Expiry time string in format "YYYY-MM-DD HH:MM"

    Returns:
        Tuple of (needs_refresh, seconds_until_expiry)
        - needs_refresh: True if key expires within 48 hours
        - seconds_until_expiry: Time until expiry (negative if already expired)
    """
    if not expire_time or not expire_time.strip():
        return True, 0

    try:
        from datetime import datetime

        # Parse iFlow's expire time format: "YYYY-MM-DD HH:MM"
        expire_dt = datetime.strptime(expire_time.strip(), "%Y-%m-%d %H:%M")
        now = datetime.now()

        seconds_until_expiry = (expire_dt - now).total_seconds()
        buffer_seconds = COOKIE_REFRESH_BUFFER_HOURS * 3600

        needs_refresh = seconds_until_expiry < buffer_seconds
        return needs_refresh, seconds_until_expiry

    except (ValueError, AttributeError) as e:
        lib_logger.warning(f"Could not parse cookie expire_time '{expire_time}': {e}")
        return True, 0


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

        # [QUEUE SYSTEM] Sequential refresh processing
        # Normal refresh queue: for proactive token refresh (old token still valid)
        self._refresh_queue: asyncio.Queue = asyncio.Queue()
        self._queue_processor_task: Optional[asyncio.Task] = None

        # Tracking sets/dicts
        self._queued_credentials: set = set()  # Track credentials in refresh queue
        self._queue_tracking_lock = asyncio.Lock()  # Protects queue sets

        # [PERMANENTLY EXPIRED] Track credentials that have been permanently removed from rotation
        # These credentials have invalid/revoked refresh tokens and require manual re-authentication
        # via credential_tool.py. They will NOT be selected for rotation until proxy restart.
        self._permanently_expired_credentials: set = set()

        # Retry tracking for normal refresh queue
        self._queue_retry_count: Dict[
            str, int
        ] = {}  # Track retry attempts per credential

        # Configuration constants
        self._refresh_timeout_seconds: int = 15  # Max time for single refresh
        self._refresh_interval_seconds: int = 30  # Delay between queue items
        self._refresh_max_retries: int = 3  # Attempts before kicked out

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
                "credential_type": "oauth",
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

    async def _get_lock(self, path: str) -> asyncio.Lock:
        # [FIX RACE CONDITION] Protect lock creation with a master lock
        # This prevents TOCTOU bug where multiple coroutines check and create simultaneously
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

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

    def _mark_credential_expired(self, path: str, reason: str) -> None:
        """
        Permanently mark a credential as expired and remove it from rotation.

        This is called when a credential's refresh token is invalid or revoked,
        meaning normal token refresh cannot work. The credential is removed from
        rotation entirely and requires manual re-authentication via credential_tool.py.

        The proxy must be restarted after fixing the credential.

        Args:
            path: Credential file path or env:// path
            reason: Human-readable reason for expiration (e.g., "invalid_grant", "HTTP 401")
        """
        # Add to permanently expired set
        self._permanently_expired_credentials.add(path)

        # Clean up other tracking structures
        self._queued_credentials.discard(path)

        # Get display name
        if path.startswith("env://"):
            display_name = path
        else:
            display_name = Path(path).name

        # Rich-formatted output for high visibility
        console.print(
            Panel(
                f"[bold red]Credential:[/bold red] {display_name}\n"
                f"[bold red]Reason:[/bold red] {reason}\n\n"
                f"[yellow]This credential has been removed from rotation.[/yellow]\n"
                f"[yellow]To fix: Run 'python credential_tool.py' to re-authenticate,[/yellow]\n"
                f"[yellow]then restart the proxy.[/yellow]",
                title="[bold red]⚠ CREDENTIAL EXPIRED - REMOVED FROM ROTATION[/bold red]",
                border_style="red",
            )
        )

        # Also log at ERROR level for log files
        lib_logger.error(
            f"CREDENTIAL EXPIRED - REMOVED FROM ROTATION | "
            f"Credential: {display_name} | Reason: {reason} | "
            f"Action: Run 'credential_tool.py' to re-authenticate, then restart proxy"
        )

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

        data = result.get("data") or {}
        api_key = data.get("apiKey", "").strip()
        if not api_key:
            raise ValueError("Missing API key in user info response")

        email = data.get("email", "").strip()
        if not email:
            email = data.get("phone", "").strip()
        if not email:
            raise ValueError("Missing email/phone in user info response")

        return {"api_key": api_key, "email": email}

    # =========================================================================
    # COOKIE-BASED AUTHENTICATION METHODS
    # =========================================================================

    async def _fetch_api_key_info_with_cookie(self, cookie: str) -> Dict[str, Any]:
        """
        Fetch API key info using browser cookie (GET request).

        This retrieves the current API key information including name,
        masked key, and expiry time.

        Args:
            cookie: Cookie string containing BXAuth

        Returns:
            Dict with keys: name, apiKey, apiKeyMask, expireTime, hasExpired
        """
        headers = {
            "Cookie": cookie,
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(IFLOW_API_KEY_ENDPOINT, headers=headers)

            if response.status_code != 200:
                lib_logger.error(
                    f"iFlow cookie GET failed: {response.status_code} {response.text}"
                )
                raise ValueError(
                    f"Cookie authentication failed: HTTP {response.status_code}"
                )

            result = response.json()

        if not result.get("success"):
            error_msg = result.get("message", "Unknown error")
            raise ValueError(f"Cookie authentication failed: {error_msg}")

        data = result.get("data") or {}

        # Handle case where apiKey is masked - use apiKeyMask if apiKey is empty
        if not data.get("apiKey") and data.get("apiKeyMask"):
            data["apiKey"] = data["apiKeyMask"]

        return data

    async def _refresh_api_key_with_cookie(
        self, cookie: str, name: str
    ) -> Dict[str, Any]:
        """
        Refresh/regenerate API key using browser cookie (POST request).

        This requests a new API key from iFlow using the session cookie.

        Args:
            cookie: Cookie string containing BXAuth
            name: The API key name (obtained from GET request)

        Returns:
            Dict with keys: name, apiKey, expireTime, hasExpired
        """
        if not name or not name.strip():
            raise ValueError("API key name is required for refresh")

        headers = {
            "Cookie": cookie,
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Origin": "https://platform.iflow.cn",
            "Referer": "https://platform.iflow.cn/",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                IFLOW_API_KEY_ENDPOINT,
                headers=headers,
                json={"name": name},
            )

            if response.status_code != 200:
                lib_logger.error(
                    f"iFlow cookie POST failed: {response.status_code} {response.text}"
                )
                raise ValueError(
                    f"Cookie API key refresh failed: HTTP {response.status_code}"
                )

            result = response.json()

        if not result.get("success"):
            error_msg = result.get("message", "Unknown error")
            raise ValueError(f"Cookie API key refresh failed: {error_msg}")

        return result.get("data") or {}

    async def authenticate_with_cookie(self, cookie: str) -> Dict[str, Any]:
        """
        Authenticate using browser cookie and obtain API key.

        This performs the full cookie-based authentication flow:
        1. Validate and normalize the cookie
        2. GET request to fetch current API key info
        3. POST request to refresh/get full API key

        Args:
            cookie: Raw cookie string from browser (must contain BXAuth)

        Returns:
            Dict with credential data including:
            - cookie: Normalized cookie string (BXAuth only)
            - api_key: The API key for iFlow API calls
            - name: Account/key name
            - expire_time: When the API key expires
            - type: "iflow_cookie"
        """
        # Normalize and validate cookie
        try:
            normalized_cookie = normalize_cookie(cookie)
        except ValueError as e:
            raise ValueError(f"Invalid cookie: {e}")

        # Extract BXAuth value for storage (only store what's needed)
        bx_auth = extract_bx_auth(normalized_cookie)
        if not bx_auth:
            raise ValueError("Could not extract BXAuth from cookie")

        # Store only BXAuth for security (don't store other cookies)
        cookie_to_store = f"BXAuth={bx_auth};"

        lib_logger.debug("Fetching API key info with cookie...")

        # GET request to fetch current info
        key_info = await self._fetch_api_key_info_with_cookie(cookie_to_store)
        name = key_info.get("name", "")

        if not name:
            raise ValueError("Could not get API key name from iFlow")

        lib_logger.debug(f"Got API key info for '{name}', refreshing key...")

        # POST request to refresh/get full API key
        refreshed = await self._refresh_api_key_with_cookie(cookie_to_store, name)

        api_key = refreshed.get("apiKey", "")
        if not api_key:
            raise ValueError("Could not get API key from iFlow")

        expire_time = refreshed.get("expireTime", "")

        return {
            "cookie": cookie_to_store,
            "api_key": api_key,
            "name": name,
            "expire_time": expire_time,
            "_proxy_metadata": {
                "email": name,  # Use name as identifier
                "last_check_timestamp": time.time(),
                "credential_type": "cookie",
            },
        }

    async def _refresh_cookie_credential(self, path: str) -> Dict[str, Any]:
        """
        Refresh API key for a cookie-based credential.

        This is called when the API key is approaching expiry.
        Note: If the browser session cookie (BXAuth) expires, the user
        will need to re-authenticate manually.

        Args:
            path: Path to the credential file

        Returns:
            Updated credentials dict
        """
        async with await self._get_lock(path):
            # Read current credentials
            creds = await self._load_credentials(path)

            if not self._is_cookie_credential(creds):
                raise ValueError(f"Credential at '{path}' is not a cookie credential")

            cookie = creds.get("cookie", "")
            name = creds.get("name", "")

            if not cookie or not name:
                raise ValueError("Cookie credential missing cookie or name")

            # Check if refresh is actually needed
            expire_time = creds.get("expire_time", "")
            needs_refresh, seconds_until = should_refresh_cookie_api_key(expire_time)

            if not needs_refresh:
                lib_logger.debug(
                    f"Cookie API key for '{name}' not due for refresh "
                    f"({seconds_until / 3600:.1f}h until expiry)"
                )
                return creds

            lib_logger.info(f"Refreshing cookie API key for '{name}'...")

            try:
                # Refresh the API key
                refreshed = await self._refresh_api_key_with_cookie(cookie, name)

                # Update credentials
                creds["api_key"] = refreshed.get("apiKey", creds["api_key"])
                creds["expire_time"] = refreshed.get("expireTime", creds["expire_time"])
                creds["_proxy_metadata"]["last_check_timestamp"] = time.time()

                # Save to disk
                if not await self._save_credentials(path, creds):
                    raise IOError(f"Failed to save refreshed cookie credentials")

                lib_logger.info(
                    f"Successfully refreshed cookie API key for '{name}'. "
                    f"New expiry: {creds['expire_time']}"
                )
                return creds

            except Exception as e:
                # If refresh fails, the session cookie may be expired
                lib_logger.error(f"Failed to refresh cookie API key for '{name}': {e}")
                # Mark as expired if it's an auth error
                if (
                    "401" in str(e)
                    or "403" in str(e)
                    or "authentication" in str(e).lower()
                ):
                    self._mark_credential_expired(
                        path,
                        f"Cookie session expired. Please re-authenticate with a fresh cookie.",
                    )
                raise

    def _is_cookie_credential(self, creds: Dict[str, Any]) -> bool:
        """Check if credentials are cookie-based (vs OAuth-based)."""
        # Primary check: explicit credential_type in metadata
        cred_type = creds.get("_proxy_metadata", {}).get("credential_type")
        if cred_type:
            return cred_type == "cookie"

        # Fallback: infer from fields (for backwards compatibility)
        # Cookie creds have 'cookie' field but no 'refresh_token'
        return "cookie" in creds and "refresh_token" not in creds

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
                        # [INVALID GRANT HANDLING] Handle 400/401/403 by marking as expired
                        # These errors indicate the refresh token is invalid/revoked
                        # Mark as permanently expired - no interactive re-auth during proxy operation
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
                                self._mark_credential_expired(
                                    path,
                                    f"Refresh token invalid (HTTP 400: {error_desc})",
                                )
                                raise CredentialNeedsReauthError(
                                    credential_path=path,
                                    message=f"Refresh token invalid for '{Path(path).name}'. Credential removed from rotation.",
                                )
                            else:
                                # Other 400 error - raise it
                                raise

                        elif status_code in (401, 403):
                            self._mark_credential_expired(
                                path, f"Credential unauthorized (HTTP {status_code})"
                            )
                            raise CredentialNeedsReauthError(
                                credential_path=path,
                                message=f"Token invalid for '{Path(path).name}' (HTTP {status_code}). Credential removed from rotation.",
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

        Supports three credential types:
        - OAuth: credential_identifier is a file path to JSON credentials with refresh_token
        - Cookie: credential_identifier is a file path to JSON credentials with cookie
        - API Key: credential_identifier is the API key string itself
        """
        # Detect credential type
        if os.path.isfile(credential_identifier):
            creds = await self._load_credentials(credential_identifier)

            # Check if this is a cookie-based credential
            if self._is_cookie_credential(creds):
                lib_logger.debug(
                    f"Using cookie credentials from file: {credential_identifier}"
                )
                # Check if API key needs refresh
                expire_time = creds.get("expire_time", "")
                needs_refresh, _ = should_refresh_cookie_api_key(expire_time)
                if needs_refresh:
                    creds = await self._refresh_cookie_credential(credential_identifier)

                api_key = creds.get("api_key")
                if not api_key:
                    raise ValueError("Missing api_key in iFlow cookie credentials")
            else:
                # OAuth credential
                lib_logger.debug(
                    f"Using OAuth credentials from file: {credential_identifier}"
                )
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
        Proactively refreshes tokens/API keys if they're close to expiry.

        Handles both credential types:
        - OAuth credentials: Refresh access token using refresh_token
        - Cookie credentials: Refresh API key using browser session cookie

        Direct API keys are skipped.
        """
        # Try to load credentials - this will fail for direct API keys
        try:
            creds = await self._load_credentials(credential_identifier)
        except IOError as e:
            # Not a valid credential path (likely a direct API key string)
            return

        # Handle cookie-based credentials
        if self._is_cookie_credential(creds):
            expire_time = creds.get("expire_time", "")
            needs_refresh, seconds_until = should_refresh_cookie_api_key(expire_time)

            if needs_refresh:
                lib_logger.debug(
                    f"Proactive cookie API key refresh triggered for "
                    f"'{Path(credential_identifier).name}' "
                    f"({seconds_until / 3600:.1f}h until expiry)"
                )
                try:
                    await self._refresh_cookie_credential(credential_identifier)
                except Exception as e:
                    lib_logger.warning(
                        f"Proactive cookie refresh failed for "
                        f"'{Path(credential_identifier).name}': {e}"
                    )
            return

        # Handle OAuth credentials
        is_expired = self._is_token_expired(creds)

        if is_expired:
            await self._queue_refresh(credential_identifier, force=False)

    async def _queue_refresh(self, path: str, force: bool = False):
        """Add a credential to the refresh queue if not already queued.

        Args:
            path: Credential file path
            force: Force refresh even if not expired
        """
        # Check backoff for automated refreshes
        now = time.time()
        if path in self._next_refresh_after:
            backoff_until = self._next_refresh_after[path]
            if now < backoff_until:
                # Credential is in backoff, do not queue
                return

        async with self._queue_tracking_lock:
            if path not in self._queued_credentials:
                self._queued_credentials.add(path)
                await self._refresh_queue.put((path, force))
                await self._ensure_queue_processor_running()

    async def _ensure_queue_processor_running(self):
        """Lazily starts the queue processor if not already running."""
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(
                self._process_refresh_queue()
            )

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
                            # Mark credential as permanently expired (no auto-reauth)
                            self._mark_credential_expired(
                                path,
                                f"Refresh token invalid (HTTP {status_code}). Requires manual re-authentication.",
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
            # Always set credential_type for OAuth credentials
            creds["_proxy_metadata"]["credential_type"] = "oauth"

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

            # =========================================================
            # COOKIE CREDENTIAL HANDLING - check first before OAuth logic
            # =========================================================
            if self._is_cookie_credential(creds):
                # Validate required fields for cookie credentials
                if not creds.get("cookie") or not creds.get("api_key"):
                    error_msg = (
                        "Cookie credential missing required fields (cookie or api_key)"
                    )
                    if path:
                        self._mark_credential_expired(path, error_msg)
                        raise ValueError(
                            f"Credential '{display_name}' is invalid: {error_msg}. "
                            f"Run 'python credential_tool.py' to re-authenticate."
                        )
                    raise ValueError(error_msg)

                # Check if API key needs refresh (48-hour buffer)
                if path:
                    expire_time = creds.get("expire_time", "")
                    needs_refresh, seconds_until = should_refresh_cookie_api_key(
                        expire_time
                    )
                    if needs_refresh:
                        try:
                            lib_logger.info(
                                f"Cookie API key for '{display_name}' needs refresh "
                                f"({seconds_until / 3600:.1f}h until expiry)"
                            )
                            creds = await self._refresh_cookie_credential(path)
                        except Exception as e:
                            lib_logger.warning(
                                f"Cookie API key refresh for '{display_name}' failed: {e}"
                            )
                            # If API key is already expired (negative seconds), mark as expired
                            if seconds_until < 0:
                                self._mark_credential_expired(
                                    path,
                                    f"Cookie API key expired and refresh failed: {e}. "
                                    f"Please re-authenticate with a fresh cookie.",
                                )
                                raise ValueError(
                                    f"Credential '{display_name}' cookie API key expired. "
                                    f"Run 'python credential_tool.py' to re-authenticate."
                                )
                            # Otherwise continue with existing (still valid) API key

                lib_logger.info(f"Cookie credential at '{display_name}' is valid.")
                return creds

            # =========================================================
            # OAUTH CREDENTIAL HANDLING - existing logic
            # =========================================================
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
                            f"Automatic token refresh for '{display_name}' failed: {e}."
                        )
                        # Fall through to handle expired credential

                # Distinguish between proxy context (has path) and credential tool context (no path)
                # - Proxy context: mark as expired and fail (no interactive OAuth during proxy operation)
                # - Credential tool context: do interactive OAuth for new credential setup
                if path:
                    # [NO AUTO-REAUTH] Proxy context - mark as permanently expired
                    self._mark_credential_expired(
                        path,
                        f"{reason}. Manual re-authentication required via credential_tool.py",
                    )
                    raise ValueError(
                        f"Credential '{display_name}' is expired and requires manual re-authentication. "
                        f"Run 'python credential_tool.py' to fix, then restart the proxy."
                    )

                # Credential tool context - do interactive OAuth for new credential setup
                lib_logger.warning(
                    f"iFlow OAuth token for '{display_name}' needs setup: {reason}."
                )
                return await self._perform_interactive_oauth(path, creds, display_name)

            lib_logger.info(f"iFlow OAuth token at '{display_name}' is valid.")
            return creds

        except Exception as e:
            raise ValueError(f"Failed to initialize iFlow OAuth for '{path}': {e}")

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        """
        Returns auth header with API key (NOT OAuth access_token).
        CRITICAL: iFlow API requests use the api_key, not the OAuth tokens.

        Handles both OAuth and cookie-based credentials:
        - OAuth: checks token expiry and refreshes OAuth tokens if needed
        - Cookie: checks API key expiry and refreshes via cookie if needed
        """
        creds = await self._load_credentials(credential_path)

        # Handle credential refresh based on type
        if self._is_cookie_credential(creds):
            # Cookie credential: check API key expiry
            expire_time = creds.get("expire_time", "")
            needs_refresh, _ = should_refresh_cookie_api_key(expire_time)
            if needs_refresh:
                creds = await self._refresh_cookie_credential(credential_path)
        else:
            # OAuth credential: check token expiry
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
                # Check if existing credential is Cookie type (will be replaced)
                try:
                    with open(existing_path, "r") as f:
                        existing_creds = json.load(f)
                    if self._is_cookie_credential(existing_creds):
                        lib_logger.info(
                            f"Replacing existing Cookie credential for {email} with OAuth credential"
                        )
                except Exception:
                    pass
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

    async def setup_cookie_credential(
        self, base_dir: Optional[Path] = None
    ) -> IFlowCredentialSetupResult:
        """
        Complete cookie-based credential setup flow with manual paste.

        This guides the user through obtaining the BXAuth cookie from their
        browser and uses it to authenticate and get an API key.

        Args:
            base_dir: Directory to save credential file (defaults to oauth_creds/)

        Returns:
            IFlowCredentialSetupResult with success status and file path
        """
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        # Ensure directory exists
        base_dir.mkdir(exist_ok=True)

        try:
            # Display instructions for cookie extraction
            console.print(
                Panel(
                    Text.from_markup(
                        "[bold]To get your iFlow session cookie:[/bold]\n\n"
                        "1. Open [link=https://platform.iflow.cn]https://platform.iflow.cn[/link] in your browser\n"
                        "2. Make sure you are [bold]logged in[/bold]\n"
                        "3. Press [bold]F12[/bold] to open Developer Tools\n"
                        "4. Go to: [bold]Application[/bold] (tab) → [bold]Cookies[/bold] → [bold]platform.iflow.cn[/bold]\n"
                        "   [dim](In Firefox: Storage → Cookies)[/dim]\n"
                        "5. Find the row with Name = [bold cyan]'BXAuth'[/bold cyan]\n"
                        "6. Double-click the [bold]Value[/bold] cell and copy it (Ctrl+C)\n"
                        "7. Paste it below\n\n"
                        "[dim]Note: The cookie typically starts with 'eyJ' and is a long string.[/dim]"
                    ),
                    title="[bold blue]iFlow Cookie Setup[/bold blue]",
                    border_style="blue",
                )
            )

            # Prompt for cookie value
            while True:
                cookie_value = Prompt.ask(
                    "\n[bold]Paste your BXAuth cookie value[/bold] (or 'q' to quit)"
                )

                if cookie_value.lower() == "q":
                    return IFlowCredentialSetupResult(
                        success=False, error="Setup cancelled by user"
                    )

                if not cookie_value.strip():
                    console.print(
                        "[yellow]Cookie value cannot be empty. Please try again.[/yellow]"
                    )
                    continue

                # Clean up common paste issues
                cookie_value = cookie_value.strip()
                if cookie_value.startswith("BXAuth="):
                    cookie_value = cookie_value[7:]
                if cookie_value.endswith(";"):
                    cookie_value = cookie_value[:-1]

                if len(cookie_value) < 20:
                    console.print(
                        "[yellow]Cookie value seems too short. "
                        "Make sure you copied the complete BXAuth value.[/yellow]"
                    )
                    continue

                break

            # Build the full cookie string
            cookie_string = f"BXAuth={cookie_value};"

            console.print("\n[dim]Validating cookie...[/dim]")

            # Authenticate with the cookie
            try:
                new_creds = await self.authenticate_with_cookie(cookie_string)
            except ValueError as e:
                return IFlowCredentialSetupResult(
                    success=False, error=f"Cookie authentication failed: {e}"
                )

            # Get identifier for deduplication
            name = new_creds.get("name", "")
            if not name:
                return IFlowCredentialSetupResult(
                    success=False, error="Could not retrieve account name from cookie"
                )

            console.print(f"[green]✓ Cookie validated for account: {name}[/green]")

            # Check for existing credential with same name/email
            # Use name as the email identifier for deduplication
            existing_path = self._find_existing_credential_by_email(name, base_dir)
            is_update = existing_path is not None

            if is_update:
                file_path = existing_path
                # Check if existing credential is OAuth type (will be replaced)
                try:
                    with open(existing_path, "r") as f:
                        existing_creds = json.load(f)
                    if not self._is_cookie_credential(existing_creds):
                        console.print(
                            f"[yellow]Replacing existing OAuth credential for {name} with Cookie credential[/yellow]"
                        )
                except Exception:
                    pass
                console.print(
                    f"[yellow]Found existing credential for {name}, updating {file_path.name}[/yellow]"
                )
            else:
                file_path = self._build_credential_path(base_dir)
                console.print(
                    f"[green]Creating new credential for {name} at {file_path.name}[/green]"
                )

            # Set email field to name for consistency with OAuth credentials
            new_creds["email"] = name

            # Save credentials to file
            if not await self._save_credentials(str(file_path), new_creds):
                return IFlowCredentialSetupResult(
                    success=False,
                    error=f"Failed to save credentials to disk at {file_path.name}",
                )

            console.print(
                Panel(
                    f"[bold green]Cookie credential saved successfully![/bold green]\n\n"
                    f"Account: {name}\n"
                    f"API Key: {new_creds.get('api_key', '')[:20]}...\n"
                    f"Expires: {new_creds.get('expire_time', 'Unknown')}\n"
                    f"File: {file_path.name}",
                    title="[bold green]Success[/bold green]",
                    border_style="green",
                )
            )

            return IFlowCredentialSetupResult(
                success=True,
                file_path=str(file_path),
                email=name,
                is_update=is_update,
                credentials=new_creds,
            )

        except Exception as e:
            lib_logger.error(f"Cookie credential setup failed: {e}")
            return IFlowCredentialSetupResult(success=False, error=str(e))

    def _find_existing_cookie_credential_by_name(
        self, name: str, base_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """Find an existing cookie credential file for the given name."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_cookie_*.json")

        for cred_file in glob(pattern):
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)
                existing_name = creds.get("name", "")
                if existing_name == name:
                    return Path(cred_file)
            except (json.JSONDecodeError, IOError) as e:
                lib_logger.debug(f"Could not read credential file {cred_file}: {e}")
                continue

        return None

    def _get_next_cookie_credential_number(
        self, base_dir: Optional[Path] = None
    ) -> int:
        """Get the next available cookie credential number."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        pattern = str(base_dir / f"{prefix}_cookie_*.json")

        existing_numbers = []
        for cred_file in glob(pattern):
            match = re.search(r"_cookie_(\d+)\.json$", cred_file)
            if match:
                existing_numbers.append(int(match.group(1)))

        if not existing_numbers:
            return 1
        return max(existing_numbers) + 1

    def _build_cookie_credential_path(
        self, base_dir: Optional[Path] = None, number: Optional[int] = None
    ) -> Path:
        """Build a path for a new cookie credential file."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        if number is None:
            number = self._get_next_cookie_credential_number(base_dir)

        prefix = self._get_provider_file_prefix()
        filename = f"{prefix}_cookie_{number}.json"
        return base_dir / filename

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
        """List all iFlow credential files (both OAuth and cookie-based)."""
        if base_dir is None:
            base_dir = self._get_oauth_base_dir()

        prefix = self._get_provider_file_prefix()
        credentials = []

        # List all credentials (both OAuth and cookie are stored as *_oauth_*.json)
        oauth_pattern = str(base_dir / f"{prefix}_oauth_*.json")
        for cred_file in sorted(glob(oauth_pattern)):
            try:
                with open(cred_file, "r") as f:
                    creds = json.load(f)

                email = creds.get("email") or creds.get("_proxy_metadata", {}).get(
                    "email", "unknown"
                )

                # Determine credential type from _proxy_metadata
                cred_type = creds.get("_proxy_metadata", {}).get("credential_type")
                if not cred_type:
                    # Fallback: infer from fields
                    if "cookie" in creds and "refresh_token" not in creds:
                        cred_type = "cookie"
                    else:
                        cred_type = "oauth"

                # Extract number from filename
                match = re.search(r"_oauth_(\d+)\.json$", cred_file)
                number = int(match.group(1)) if match else 0

                cred_info = {
                    "file_path": cred_file,
                    "email": email,
                    "number": number,
                    "type": cred_type,
                }

                # Add expire_time for cookie credentials
                if cred_type == "cookie":
                    cred_info["expire_time"] = creds.get("expire_time", "")

                credentials.append(cred_info)
            except Exception as e:
                lib_logger.debug(f"Could not read credential file {cred_file}: {e}")
                continue

        return credentials

    def delete_credential(self, credential_path: str) -> bool:
        """Delete a credential file (OAuth or cookie-based)."""
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
