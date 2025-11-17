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
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional
from urllib.parse import urlencode, parse_qs, urlparse

import httpx
from aiohttp import web
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

lib_logger = logging.getLogger('rotator_library')

# OAuth endpoints and credentials from Go example
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

# Refresh tokens 24 hours before expiry (from Go example)
REFRESH_EXPIRY_BUFFER_SECONDS = 24 * 60 * 60

console = Console()


class OAuthCallbackServer:
    """
    Minimal HTTP server for handling iFlow OAuth callbacks.
    Based on the Go example's oauth_server.go implementation.
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
            sock.bind(('', self.port))
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
        self.app.router.add_get('/oauth2callback', self._handle_callback)

        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, 'localhost', self.port)
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
        if 'error' in query:
            error = query.get('error', 'unknown_error')
            lib_logger.error(f"iFlow OAuth callback received error: {error}")
            if not self.result_future.done():
                self.result_future.set_exception(ValueError(f"OAuth error: {error}"))
            return web.Response(status=302, headers={'Location': IFLOW_ERROR_REDIRECT_URL})

        # Check for authorization code
        code = query.get('code')
        if not code:
            lib_logger.error("iFlow OAuth callback missing authorization code")
            if not self.result_future.done():
                self.result_future.set_exception(ValueError("Missing authorization code"))
            return web.Response(status=302, headers={'Location': IFLOW_ERROR_REDIRECT_URL})

        # Validate state parameter
        state = query.get('state', '')
        if state != self.expected_state:
            lib_logger.error(f"iFlow OAuth state mismatch. Expected: {self.expected_state}, Got: {state}")
            if not self.result_future.done():
                self.result_future.set_exception(ValueError("State parameter mismatch"))
            return web.Response(status=302, headers={'Location': IFLOW_ERROR_REDIRECT_URL})

        # Success - set result and redirect to success page
        if not self.result_future.done():
            self.result_future.set_result(code)

        return web.Response(status=302, headers={'Location': IFLOW_SUCCESS_REDIRECT_URL})

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
    Based on the Go example implementation.
    """

    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}

    async def _read_creds_from_file(self, path: str) -> Dict[str, Any]:
        """Reads credentials from file and populates the cache. No locking."""
        try:
            lib_logger.debug(f"Reading iFlow credentials from file: {path}")
            with open(path, 'r') as f:
                creds = json.load(f)
            self._credentials_cache[path] = creds
            return creds
        except FileNotFoundError:
            raise IOError(f"iFlow OAuth credential file not found at '{path}'")
        except Exception as e:
            raise IOError(f"Failed to load iFlow OAuth credentials from '{path}': {e}")

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        """Loads credentials from cache or file."""
        if path in self._credentials_cache:
            return self._credentials_cache[path]

        async with self._get_lock(path):
            # Re-check cache after acquiring lock
            if path in self._credentials_cache:
                return self._credentials_cache[path]
            return await self._read_creds_from_file(path)

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
        """Saves credentials to cache and file."""
        self._credentials_cache[path] = creds
        try:
            with open(path, 'w') as f:
                json.dump(creds, f, indent=2)
            lib_logger.debug(f"Saved updated iFlow OAuth credentials to '{path}'.")
        except Exception as e:
            lib_logger.error(f"Failed to save updated iFlow OAuth credentials to '{path}': {e}")

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        """Checks if the token is expired (with buffer for proactive refresh)."""
        # Try to parse expiry_date as ISO 8601 string (from Go example)
        expiry_str = creds.get("expiry_date")
        if not expiry_str:
            return True

        try:
            # Parse ISO 8601 format (e.g., "2025-01-17T12:00:00Z")
            from datetime import datetime
            expiry_dt = datetime.fromisoformat(expiry_str.replace('Z', '+00:00'))
            expiry_timestamp = expiry_dt.timestamp()
        except (ValueError, AttributeError):
            # Fallback: treat as numeric timestamp
            try:
                expiry_timestamp = float(expiry_str)
            except (ValueError, TypeError):
                lib_logger.warning(f"Could not parse expiry_date: {expiry_str}")
                return True

        return expiry_timestamp < time.time() + REFRESH_EXPIRY_BUFFER_SECONDS

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

    async def _exchange_code_for_tokens(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """
        Exchanges authorization code for access and refresh tokens.
        Uses Basic Auth with client credentials (from Go example).
        """
        # Create Basic Auth header
        auth_string = f"{IFLOW_CLIENT_ID}:{IFLOW_CLIENT_SECRET}"
        basic_auth = base64.b64encode(auth_string.encode()).decode()

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Authorization": f"Basic {basic_auth}"
        }

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": IFLOW_CLIENT_ID,
            "client_secret": IFLOW_CLIENT_SECRET
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(IFLOW_OAUTH_TOKEN_ENDPOINT, headers=headers, data=data)

            if response.status_code != 200:
                error_text = response.text
                lib_logger.error(f"iFlow token exchange failed: {response.status_code} {error_text}")
                raise ValueError(f"Token exchange failed: {response.status_code} {error_text}")

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
        expiry_date = (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat() + 'Z'

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "api_key": user_info["api_key"],
            "email": user_info["email"],
            "expiry_date": expiry_date,
            "token_type": token_type,
            "scope": scope
        }

    async def _refresh_token(self, path: str, force: bool = False) -> Dict[str, Any]:
        """
        Refreshes the OAuth tokens and re-fetches the API key.
        CRITICAL: Must re-fetch user info to get potentially updated API key.
        """
        async with self._get_lock(path):
            cached_creds = self._credentials_cache.get(path)
            if not force and cached_creds and not self._is_token_expired(cached_creds):
                return cached_creds

            # If cache is empty, read from file
            if path not in self._credentials_cache:
                await self._read_creds_from_file(path)

            creds_from_file = self._credentials_cache[path]

            lib_logger.info(f"Refreshing iFlow OAuth token for '{Path(path).name}'...")
            refresh_token = creds_from_file.get("refresh_token")
            if not refresh_token:
                raise ValueError("No refresh_token found in iFlow credentials file.")

            # Create Basic Auth header
            auth_string = f"{IFLOW_CLIENT_ID}:{IFLOW_CLIENT_SECRET}"
            basic_auth = base64.b64encode(auth_string.encode()).decode()

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "Authorization": f"Basic {basic_auth}"
            }

            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": IFLOW_CLIENT_ID,
                "client_secret": IFLOW_CLIENT_SECRET
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(IFLOW_OAUTH_TOKEN_ENDPOINT, headers=headers, data=data)
                response.raise_for_status()
                new_token_data = response.json()

            # Update tokens
            access_token = new_token_data.get("access_token")
            if not access_token:
                raise ValueError("Missing access_token in refresh response")

            creds_from_file["access_token"] = access_token
            creds_from_file["refresh_token"] = new_token_data.get("refresh_token", creds_from_file["refresh_token"])

            expires_in = new_token_data.get("expires_in", 3600)
            from datetime import datetime, timedelta
            creds_from_file["expiry_date"] = (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat() + 'Z'

            creds_from_file["token_type"] = new_token_data.get("token_type", creds_from_file.get("token_type", "Bearer"))
            creds_from_file["scope"] = new_token_data.get("scope", creds_from_file.get("scope", ""))

            # CRITICAL: Re-fetch user info to get potentially updated API key
            try:
                user_info = await self._fetch_user_info(access_token)
                if user_info.get("api_key"):
                    creds_from_file["api_key"] = user_info["api_key"]
                if user_info.get("email"):
                    creds_from_file["email"] = user_info["email"]
            except Exception as e:
                lib_logger.warning(f"Failed to update API key during token refresh: {e}")

            # Update timestamp in metadata if it exists
            if creds_from_file.get("_proxy_metadata"):
                creds_from_file["_proxy_metadata"]["last_check_timestamp"] = time.time()

            await self._save_credentials(path, creds_from_file)
            lib_logger.info(f"Successfully refreshed iFlow OAuth token for '{Path(path).name}'.")
            return creds_from_file

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
            lib_logger.debug(f"Using OAuth credentials from file: {credential_identifier}")
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
        Only applies to OAuth credentials (file paths). Direct API keys are skipped.
        """
        # Only refresh if it's an OAuth credential (file path)
        if os.path.isfile(credential_identifier):
            creds = await self._load_credentials(credential_identifier)
            if self._is_token_expired(creds):
                await self._refresh_token(credential_identifier)
        # else: Direct API key, no refresh needed

    def _get_lock(self, path: str) -> asyncio.Lock:
        """Gets or creates a lock for the given credential path."""
        if path not in self._refresh_locks:
            self._refresh_locks[path] = asyncio.Lock()
        return self._refresh_locks[path]

    async def initialize_token(self, creds_or_path: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Initiates OAuth authorization code flow if tokens are missing or invalid.
        Uses local callback server to receive authorization code.
        """
        path = creds_or_path if isinstance(creds_or_path, str) else None
        file_name = Path(path).name if path else "in-memory object"
        lib_logger.debug(f"Initializing iFlow token for '{file_name}'...")

        try:
            creds = await self._load_credentials(creds_or_path) if path else creds_or_path

            reason = ""
            if not creds.get("refresh_token"):
                reason = "refresh token is missing"
            elif self._is_token_expired(creds):
                reason = "token is expired"

            if reason:
                # Try automatic refresh first if we have a refresh token
                if reason == "token is expired" and creds.get("refresh_token"):
                    try:
                        return await self._refresh_token(path)
                    except Exception as e:
                        lib_logger.warning(f"Automatic token refresh for '{file_name}' failed: {e}. Proceeding to interactive login.")

                # Interactive OAuth flow
                lib_logger.warning(f"iFlow OAuth token for '{file_name}' needs setup: {reason}.")

                # Generate random state for CSRF protection
                state = secrets.token_urlsafe(32)

                # Build authorization URL
                redirect_uri = f"http://localhost:{CALLBACK_PORT}/oauth2callback"
                auth_params = {
                    "loginMethod": "phone",
                    "type": "phone",
                    "redirect": redirect_uri,
                    "state": state,
                    "client_id": IFLOW_CLIENT_ID
                }
                auth_url = f"{IFLOW_OAUTH_AUTHORIZE_ENDPOINT}?{urlencode(auth_params)}"

                # Start OAuth callback server
                callback_server = OAuthCallbackServer(port=CALLBACK_PORT)
                try:
                    await callback_server.start(expected_state=state)

                    # Display instructions to user
                    auth_panel_text = Text.from_markup(
                        "1. Visit the URL below to sign in with your phone number.\n"
                        "2. [bold]Authorize the application[/bold] to access your account.\n"
                        "3. You will be automatically redirected after authorization."
                    )
                    console.print(Panel(auth_panel_text, title=f"iFlow OAuth Setup for [bold yellow]{file_name}[/bold yellow]", style="bold blue"))
                    console.print(f"[bold]URL:[/bold] [link={auth_url}]{auth_url}[/link]\n")

                    # Open browser
                    webbrowser.open(auth_url)

                    # Wait for callback
                    with console.status("[bold green]Waiting for authorization in the browser...[/bold green]", spinner="dots"):
                        code = await callback_server.wait_for_callback(timeout=300.0)

                    lib_logger.info("Received authorization code, exchanging for tokens...")

                    # Exchange code for tokens and API key
                    token_data = await self._exchange_code_for_tokens(code, redirect_uri)

                    # Update credentials
                    creds.update({
                        "access_token": token_data["access_token"],
                        "refresh_token": token_data["refresh_token"],
                        "api_key": token_data["api_key"],
                        "email": token_data["email"],
                        "expiry_date": token_data["expiry_date"],
                        "token_type": token_data["token_type"],
                        "scope": token_data["scope"]
                    })

                    # Create metadata object
                    if not creds.get("_proxy_metadata"):
                        creds["_proxy_metadata"] = {
                            "email": token_data["email"],
                            "last_check_timestamp": time.time()
                        }

                    if path:
                        await self._save_credentials(path, creds)

                    lib_logger.info(f"iFlow OAuth initialized successfully for '{file_name}'.")
                    return creds

                finally:
                    await callback_server.stop()

            lib_logger.info(f"iFlow OAuth token at '{file_name}' is valid.")
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

    async def get_user_info(self, creds_or_path: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """Retrieves user info from the _proxy_metadata in the credential file."""
        try:
            path = creds_or_path if isinstance(creds_or_path, str) else None
            creds = await self._load_credentials(creds_or_path) if path else creds_or_path

            # Ensure the token is valid
            if path:
                await self.initialize_token(path)
                creds = await self._load_credentials(path)

            email = creds.get("email") or creds.get("_proxy_metadata", {}).get("email")

            if not email:
                lib_logger.warning(f"No email found in iFlow credentials for '{path or 'in-memory object'}'.")

            # Update timestamp on check
            if path and "_proxy_metadata" in creds:
                creds["_proxy_metadata"]["last_check_timestamp"] = time.time()
                await self._save_credentials(path, creds)

            return {"email": email}
        except Exception as e:
            lib_logger.error(f"Failed to get iFlow user info from credentials: {e}")
            return {"email": None}
