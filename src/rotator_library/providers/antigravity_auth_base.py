# src/rotator_library/providers/antigravity_auth_base.py

import os
import webbrowser
from typing import Union, Optional
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from ..utils.headless_detection import is_headless_environment

lib_logger = logging.getLogger('rotator_library')

# Antigravity OAuth credentials from CLIProxyAPI
CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"
CLIENT_SECRET = "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"
TOKEN_URI = "https://oauth2.googleapis.com/token"
USER_INFO_URI = "https://www.googleapis.com/oauth2/v1/userinfo"
REFRESH_EXPIRY_BUFFER_SECONDS = 30 * 60  # 30 minutes buffer before expiry

# Antigravity requires additional scopes
OAUTH_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",  # Antigravity-specific
    "https://www.googleapis.com/auth/experimentsandconfigs"  # Antigravity-specific
]

console = Console()

class AntigravityAuthBase:
    """
    Base authentication class for Antigravity provider.
    Handles OAuth2 flow, token management, and refresh logic.
    
    Based on GeminiAuthBase but uses Antigravity-specific OAuth credentials and scopes.
    """
    
    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # Protects the locks dict from race conditions
        # [BACKOFF TRACKING] Track consecutive failures per credential
        self._refresh_failures: Dict[str, int] = {}  # Track consecutive failures per credential
        self._next_refresh_after: Dict[str, float] = {}  # Track backoff timers (Unix timestamp)
        
        # [QUEUE SYSTEM] Sequential refresh processing
        self._refresh_queue: asyncio.Queue = asyncio.Queue()
        self._queued_credentials: set = set()  # Track credentials already in queue
        self._unavailable_credentials: set = set()  # Mark credentials unavailable during re-auth
        self._queue_tracking_lock = asyncio.Lock()  # Protects queue sets
        self._queue_processor_task: Optional[asyncio.Task] = None  # Background worker task

    def _load_from_env(self) -> Optional[Dict[str, Any]]:
        """
        Load OAuth credentials from environment variables for stateless deployments.

        Expected environment variables:
        - ANTIGRAVITY_ACCESS_TOKEN (required)
        - ANTIGRAVITY_REFRESH_TOKEN (required)
        - ANTIGRAVITY_EXPIRY_DATE (optional, defaults to 0)
        - ANTIGRAVITY_CLIENT_ID (optional, uses default)
        - ANTIGRAVITY_CLIENT_SECRET (optional, uses default)
        - ANTIGRAVITY_TOKEN_URI (optional, uses default)
        - ANTIGRAVITY_UNIVERSE_DOMAIN (optional, defaults to googleapis.com)
        - ANTIGRAVITY_EMAIL (optional, defaults to "env-user")

        Returns:
            Dict with credential structure if env vars present, None otherwise
        """
        access_token = os.getenv("ANTIGRAVITY_ACCESS_TOKEN")
        refresh_token = os.getenv("ANTIGRAVITY_REFRESH_TOKEN")

        # Both access and refresh tokens are required
        if not (access_token and refresh_token):
            return None

        lib_logger.debug("Loading Antigravity credentials from environment variables")

        # Parse expiry_date as float, default to 0 if not present
        expiry_str = os.getenv("ANTIGRAVITY_EXPIRY_DATE", "0")
        try:
            expiry_date = float(expiry_str)
        except ValueError:
            lib_logger.warning(f"Invalid ANTIGRAVITY_EXPIRY_DATE value: {expiry_str}, using 0")
            expiry_date = 0

        creds = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": expiry_date,
            "client_id": os.getenv("ANTIGRAVITY_CLIENT_ID", CLIENT_ID),
            "client_secret": os.getenv("ANTIGRAVITY_CLIENT_SECRET", CLIENT_SECRET),
            "token_uri": os.getenv("ANTIGRAVITY_TOKEN_URI", TOKEN_URI),
            "universe_domain": os.getenv("ANTIGRAVITY_UNIVERSE_DOMAIN", "googleapis.com"),
            "_proxy_metadata": {
                "email": os.getenv("ANTIGRAVITY_EMAIL", "env-user"),
                "last_check_timestamp": time.time(),
                "loaded_from_env": True  # Flag to indicate env-based credentials
            }
        }

        return creds

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        """
        Load credentials from a file. First attempts file-based load,
        then falls back to environment variables if file not found.

        Args:
            path: File path to load credentials from

        Returns:
            Dict containing the credentials

        Raises:
            ValueError: If credentials cannot be loaded from either source
        """
        # If path is special marker "env", load from environment
        if path == "env":
            env_creds = self._load_from_env()
            if env_creds:
                lib_logger.debug("Using Antigravity credentials from environment variables")
                return env_creds
            raise ValueError("ANTIGRAVITY_ACCESS_TOKEN and ANTIGRAVITY_REFRESH_TOKEN environment variables not set")

        # Try loading from cache first
        if path in self._credentials_cache:
            cached_creds = self._credentials_cache[path]
            lib_logger.debug(f"Using cached Antigravity credentials for: {Path(path).name}")
            return cached_creds

        # Try loading from file
        try:
            with open(path, 'r') as f:
                creds = json.load(f)
            self._credentials_cache[path] = creds
            lib_logger.debug(f"Loaded Antigravity credentials from file: {Path(path).name}")
            return creds
        except FileNotFoundError:
            # Fall back to environment variables
            lib_logger.debug(f"Credential file not found: {path}, attempting environment variables")
            env_creds = self._load_from_env()
            if env_creds:
                lib_logger.debug("Using Antigravity credentials from environment variables as fallback")
                # Cache with special path marker
                self._credentials_cache[path] = env_creds
                return env_creds
            raise ValueError(f"Credential file not found: {path} and environment variables not set")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in credential file {path}: {e}")

    async def _save_credentials(self, path: str, creds: Dict[str, Any]) -> None:
        """
        Save credentials to a file. Skip if credentials were loaded from environment.

        Args:
            path: File path to save credentials to
            creds: Credentials dictionary to save
        """
        # Don't save environment-based credentials to file
        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            lib_logger.debug("Skipping credential save (loaded from environment)")
            return

        # Don't save if path is special marker
        if path == "env":
            return

        try:
            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write atomically using temp file + rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=Path(path).parent,
                prefix='.tmp_',
                suffix='.json'
            )
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(creds, f, indent=2)
                shutil.move(temp_path, path)
                lib_logger.debug(f"Saved Antigravity credentials to: {Path(path).name}")
            except Exception:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                raise
        except Exception as e:
            lib_logger.warning(f"Failed to save Antigravity credentials to {path}: {e}")

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        """
        Check if the access token is expired or close to expiry.

        Args:
            creds: Credentials dict with expiry_date field (in milliseconds)

        Returns:
            True if token is expired or within buffer time of expiry
        """
        if 'expiry_date' not in creds:
            return True

        # expiry_date is in milliseconds
        expiry_timestamp = creds['expiry_date'] / 1000.0
        current_time = time.time()
        
        # Consider expired if within buffer time
        return (expiry_timestamp - current_time) <= REFRESH_EXPIRY_BUFFER_SECONDS

    async def _refresh_token(self, path: str, creds: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refresh an expired OAuth token using the refresh token.

        Args:
            path: Credential file path (for saving updated credentials)
            creds: Current credentials dict with refresh_token

        Returns:
            Updated credentials dict with fresh access token

        Raises:
            ValueError: If refresh fails
        """
        if 'refresh_token' not in creds:
            raise ValueError("No refresh token available")

        lib_logger.debug(f"Refreshing Antigravity OAuth token for: {Path(path).name if path != 'env' else 'env'}")

        client_id = creds.get('client_id', CLIENT_ID)
        client_secret = creds.get('client_secret', CLIENT_SECRET)
        token_uri = creds.get('token_uri', TOKEN_URI)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    token_uri,
                    data={
                        'client_id': client_id,
                        'client_secret': client_secret,
                        'refresh_token': creds['refresh_token'],
                        'grant_type': 'refresh_token'
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                token_data = response.json()

                # Update credentials with new token
                creds['access_token'] = token_data['access_token']
                creds['expiry_date'] = (time.time() + token_data['expires_in']) * 1000

                # Update metadata
                if '_proxy_metadata' not in creds:
                    creds['_proxy_metadata'] = {}
                creds['_proxy_metadata']['last_check_timestamp'] = time.time()

                # Save updated credentials
                await self._save_credentials(path, creds)

                # Update cache
                self._credentials_cache[path] = creds

                # Reset failure count on success
                self._refresh_failures[path] = 0

                lib_logger.info(f"Successfully refreshed Antigravity OAuth token for: {Path(path).name if path != 'env' else 'env'}")
                return creds

            except httpx.HTTPStatusError as e:
                # Track failures for backoff
                self._refresh_failures[path] = self._refresh_failures.get(path, 0) + 1
                raise ValueError(f"Failed to refresh Antigravity token (HTTP {e.response.status_code}): {e.response.text}")
            except Exception as e:
                self._refresh_failures[path] = self._refresh_failures.get(path, 0) + 1
                raise ValueError(f"Failed to refresh Antigravity token: {e}")

    async def initialize_token(self, creds_or_path: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Initialize or refresh an OAuth token. Handles the complete OAuth flow if needed.

        Args:
            creds_or_path: Either a credentials dict or a file path string

        Returns:
            Valid credentials dict with fresh access token
        """
        path = creds_or_path if isinstance(creds_or_path, str) else None

        if isinstance(creds_or_path, dict):
            display_name = creds_or_path.get("_proxy_metadata", {}).get("display_name", "in-memory object")
        else:
            display_name = Path(path).name if path and path != "env" else "env"

        lib_logger.debug(f"Initializing Antigravity token for '{display_name}'...")
        
        try:
            creds = await self._load_credentials(creds_or_path) if path else creds_or_path
            reason = ""
            if not creds.get("refresh_token"):
                reason = "refresh token is missing"
            elif self._is_token_expired(creds):
                reason = "token is expired"

            if reason:
                if reason == "token is expired" and creds.get("refresh_token"):
                    try:
                        return await self._refresh_token(path, creds)
                    except Exception as e:
                        lib_logger.warning(f"Automatic token refresh for '{display_name}' failed: {e}. Proceeding to interactive login.")

                lib_logger.warning(f"Antigravity OAuth token for '{display_name}' needs setup: {reason}.")
                
                is_headless = is_headless_environment()
                
                auth_code_future = asyncio.get_event_loop().create_future()
                server = None

                async def handle_callback(reader, writer):
                    try:
                        request_line_bytes = await reader.readline()
                        if not request_line_bytes:
                            return
                        path_str = request_line_bytes.decode('utf-8').strip().split(' ')[1]
                        # Consume headers
                        while await reader.readline() != b'\r\n':
                            pass
                        
                        from urllib.parse import urlparse, parse_qs
                        query_params = parse_qs(urlparse(path_str).query)
                        
                        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n")
                        if 'code' in query_params:
                            if not auth_code_future.done():
                                auth_code_future.set_result(query_params['code'][0])
                            writer.write(b"<html><body><h1>Authentication successful!</h1><p>You can close this window.</p></body></html>")
                        else:
                            error = query_params.get('error', ['Unknown error'])[0]
                            if not auth_code_future.done():
                                auth_code_future.set_exception(Exception(f"OAuth failed: {error}"))
                            writer.write(f"<html><body><h1>Authentication Failed</h1><p>Error: {error}. Please try again.</p></body></html>".encode())
                        await writer.drain()
                    except Exception as e:
                        lib_logger.error(f"Error in OAuth callback handler: {e}")
                    finally:
                        writer.close()

                try:
                    server = await asyncio.start_server(handle_callback, '127.0.0.1', 8085)
                    
                    from urllib.parse import urlencode
                    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
                        "client_id": CLIENT_ID,
                        "redirect_uri": "http://localhost:8085/oauth2callback",
                        "scope": " ".join(OAUTH_SCOPES),
                        "access_type": "offline",
                        "response_type": "code",
                        "prompt": "consent"
                    })
                    
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
                    
                    console.print(Panel(auth_panel_text, title=f"Antigravity OAuth Setup for [bold yellow]{display_name}[/bold yellow]", style="bold blue"))
                    console.print(f"[bold]URL:[/bold] [link={auth_url}]{auth_url}[/link]\n")
                    
                    if not is_headless:
                        try:
                            webbrowser.open(auth_url)
                            lib_logger.info("Browser opened successfully for OAuth flow")
                        except Exception as e:
                            lib_logger.warning(f"Failed to open browser automatically: {e}. Please open the URL manually.")
                    
                    with console.status("[bold green]Waiting for you to complete authentication in the browser...[/bold green]", spinner="dots"):
                        auth_code = await asyncio.wait_for(auth_code_future, timeout=300)
                except asyncio.TimeoutError:
                    raise Exception("OAuth flow timed out. Please try again.")
                finally:
                    if server:
                        server.close()
                        await server.wait_closed()
                
                lib_logger.info(f"Attempting to exchange authorization code for tokens...")
                async with httpx.AsyncClient() as client:
                    response = await client.post(TOKEN_URI, data={
                        "code": auth_code.strip(),
                        "client_id": CLIENT_ID,
                        "client_secret": CLIENT_SECRET,
                        "redirect_uri": "http://localhost:8085/oauth2callback",
                        "grant_type": "authorization_code"
                    })
                    response.raise_for_status()
                    token_data = response.json()
                    
                    creds = token_data.copy()
                    creds["expiry_date"] = (time.time() + creds.pop("expires_in")) * 1000
                    creds["client_id"] = CLIENT_ID
                    creds["client_secret"] = CLIENT_SECRET
                    creds["token_uri"] = TOKEN_URI
                    creds["universe_domain"] = "googleapis.com"
                    
                    # Fetch user info
                    user_info_response = await client.get(
                        USER_INFO_URI,
                        headers={"Authorization": f"Bearer {creds['access_token']}"}
                    )
                    user_info_response.raise_for_status()
                    user_info = user_info_response.json()
                    
                    creds["_proxy_metadata"] = {
                        "email": user_info.get("email"),
                        "last_check_timestamp": time.time()
                    }

                    if path:
                        await self._save_credentials(path, creds)
                    
                    lib_logger.info(f"Antigravity OAuth initialized successfully for '{display_name}'.")
                    return creds

            lib_logger.info(f"Antigravity OAuth token at '{display_name}' is valid.")
            return creds
        except Exception as e:
            raise ValueError(f"Failed to initialize Antigravity OAuth for '{display_name}': {e}")

    async def get_valid_token(self, credential_path: str) -> str:
        """
        Get a valid access token, refreshing if necessary.

        Args:
            credential_path: Path to credential file or "env" for environment variables

        Returns:
            Valid access token string

        Raises:
            ValueError: If token cannot be obtained
        """
        try:
            creds = await self.initialize_token(credential_path)
            return creds['access_token']
        except Exception as e:
            raise ValueError(f"Failed to get valid Antigravity token: {e}")
