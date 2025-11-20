# src/rotator_library/providers/gemini_auth_base.py

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

lib_logger = logging.getLogger('rotator_library')

CLIENT_ID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com" #https://api.kilocode.ai/extension-config.json
CLIENT_SECRET = "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" #https://api.kilocode.ai/extension-config.json
TOKEN_URI = "https://oauth2.googleapis.com/token"
USER_INFO_URI = "https://www.googleapis.com/oauth2/v1/userinfo"
REFRESH_EXPIRY_BUFFER_SECONDS = 30 * 60  # 30 minutes buffer before expiry

console = Console()

class GeminiAuthBase:
    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # Protects the locks dict from race conditions
        # [BACKOFF TRACKING] Track consecutive failures per credential
        self._refresh_failures: Dict[str, int] = {}  # Track consecutive failures per credential
        self._next_refresh_after: Dict[str, float] = {}  # Track backoff timers (Unix timestamp)

    def _load_from_env(self) -> Optional[Dict[str, Any]]:
        """
        Load OAuth credentials from environment variables for stateless deployments.

        Expected environment variables:
        - GEMINI_CLI_ACCESS_TOKEN (required)
        - GEMINI_CLI_REFRESH_TOKEN (required)
        - GEMINI_CLI_EXPIRY_DATE (optional, defaults to 0)
        - GEMINI_CLI_CLIENT_ID (optional, uses default)
        - GEMINI_CLI_CLIENT_SECRET (optional, uses default)
        - GEMINI_CLI_TOKEN_URI (optional, uses default)
        - GEMINI_CLI_UNIVERSE_DOMAIN (optional, defaults to googleapis.com)
        - GEMINI_CLI_EMAIL (optional, defaults to "env-user")
        - GEMINI_CLI_PROJECT_ID (optional)

        Returns:
            Dict with credential structure if env vars present, None otherwise
        """
        access_token = os.getenv("GEMINI_CLI_ACCESS_TOKEN")
        refresh_token = os.getenv("GEMINI_CLI_REFRESH_TOKEN")

        # Both access and refresh tokens are required
        if not (access_token and refresh_token):
            return None

        lib_logger.debug("Loading Gemini CLI credentials from environment variables")

        # Parse expiry_date as float, default to 0 if not present
        expiry_str = os.getenv("GEMINI_CLI_EXPIRY_DATE", "0")
        try:
            expiry_date = float(expiry_str)
        except ValueError:
            lib_logger.warning(f"Invalid GEMINI_CLI_EXPIRY_DATE value: {expiry_str}, using 0")
            expiry_date = 0

        creds = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": expiry_date,
            "client_id": os.getenv("GEMINI_CLI_CLIENT_ID", CLIENT_ID),
            "client_secret": os.getenv("GEMINI_CLI_CLIENT_SECRET", CLIENT_SECRET),
            "token_uri": os.getenv("GEMINI_CLI_TOKEN_URI", TOKEN_URI),
            "universe_domain": os.getenv("GEMINI_CLI_UNIVERSE_DOMAIN", "googleapis.com"),
            "_proxy_metadata": {
                "email": os.getenv("GEMINI_CLI_EMAIL", "env-user"),
                "last_check_timestamp": time.time(),
                "loaded_from_env": True  # Flag to indicate env-based credentials
            }
        }

        # Add project_id if provided
        project_id = os.getenv("GEMINI_CLI_PROJECT_ID")
        if project_id:
            creds["_proxy_metadata"]["project_id"] = project_id

        return creds

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        if path in self._credentials_cache:
            return self._credentials_cache[path]

        async with await self._get_lock(path):
            if path in self._credentials_cache:
                return self._credentials_cache[path]

            # First, try loading from environment variables
            env_creds = self._load_from_env()
            if env_creds:
                lib_logger.info("Using Gemini CLI credentials from environment variables")
                # Cache env-based credentials using the path as key
                self._credentials_cache[path] = env_creds
                return env_creds

            # Fall back to file-based loading
            try:
                lib_logger.debug(f"Loading Gemini credentials from file: {path}")
                with open(path, 'r') as f:
                    creds = json.load(f)
                # Handle gcloud-style creds file which nest tokens under "credential"
                if "credential" in creds:
                    creds = creds["credential"]
                self._credentials_cache[path] = creds
                return creds
            except FileNotFoundError:
                raise IOError(f"Gemini OAuth credential file not found at '{path}'")
            except Exception as e:
                raise IOError(f"Failed to load Gemini OAuth credentials from '{path}': {e}")

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
        # Don't save to file if credentials were loaded from environment
        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            lib_logger.debug("Credentials loaded from env, skipping file save")
            # Still update cache for in-memory consistency
            self._credentials_cache[path] = creds
            return

        # [ATOMIC WRITE] Use tempfile + move pattern to ensure atomic writes
        # This prevents credential corruption if the process is interrupted during write
        parent_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent_dir, exist_ok=True)

        tmp_fd = None
        tmp_path = None
        try:
            # Create temp file in same directory as target (ensures same filesystem)
            tmp_fd, tmp_path = tempfile.mkstemp(dir=parent_dir, prefix='.tmp_', suffix='.json', text=True)

            # Write JSON to temp file
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(creds, f, indent=2)
                tmp_fd = None  # fdopen closes the fd

            # Set secure permissions (0600 = owner read/write only)
            try:
                os.chmod(tmp_path, 0o600)
            except (OSError, AttributeError):
                # Windows may not support chmod, ignore
                pass

            # Atomic move (overwrites target if it exists)
            shutil.move(tmp_path, path)
            tmp_path = None  # Successfully moved

            # Update cache AFTER successful file write (prevents cache/file inconsistency)
            self._credentials_cache[path] = creds
            lib_logger.debug(f"Saved updated Gemini OAuth credentials to '{path}' (atomic write).")

        except Exception as e:
            lib_logger.error(f"Failed to save updated Gemini OAuth credentials to '{path}': {e}")
            # Clean up temp file if it still exists
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except:
                    pass
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            raise

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        expiry = creds.get("token_expiry") # gcloud format
        if not expiry: # gemini-cli format
             expiry_timestamp = creds.get("expiry_date", 0) / 1000
        else:
            expiry_timestamp = time.mktime(time.strptime(expiry, "%Y-%m-%dT%H:%M:%SZ"))
        return expiry_timestamp < time.time() + REFRESH_EXPIRY_BUFFER_SECONDS

    async def _refresh_token(self, path: str, creds: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        async with await self._get_lock(path):
            # Skip the expiry check if a refresh is being forced
            if not force and not self._is_token_expired(self._credentials_cache.get(path, creds)):
                return self._credentials_cache.get(path, creds)

            lib_logger.debug(f"Refreshing Gemini OAuth token for '{Path(path).name}' (forced: {force})...")
            refresh_token = creds.get("refresh_token")
            if not refresh_token:
                raise ValueError("No refresh_token found in credentials file.")

            # [RETRY LOGIC] Implement exponential backoff for transient errors
            max_retries = 3
            new_token_data = None
            last_error = None

            async with httpx.AsyncClient() as client:
                for attempt in range(max_retries):
                    try:
                        response = await client.post(TOKEN_URI, data={
                            "client_id": creds.get("client_id", CLIENT_ID),
                            "client_secret": creds.get("client_secret", CLIENT_SECRET),
                            "refresh_token": refresh_token,
                            "grant_type": "refresh_token",
                        }, timeout=30.0)
                        response.raise_for_status()
                        new_token_data = response.json()
                        break  # Success, exit retry loop

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        status_code = e.response.status_code

                        # [STATUS CODE HANDLING] Handle per-status backoff strategy
                        if status_code == 401 or status_code == 403:
                            # Invalid credentials - don't retry, invalidate refresh token
                            lib_logger.error(f"Refresh token invalid (HTTP {status_code}), marking as revoked")
                            creds["refresh_token"] = None  # Invalidate refresh token
                            await self._save_credentials(path, creds)
                            raise ValueError(f"Refresh token revoked or invalid (HTTP {status_code}). Re-authentication required.")

                        elif status_code == 429:
                            # Rate limit - honor Retry-After header if present
                            retry_after = int(e.response.headers.get("Retry-After", 60))
                            lib_logger.warning(f"Rate limited (HTTP 429), retry after {retry_after}s")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_after)
                                continue
                            raise

                        elif status_code >= 500 and status_code < 600:
                            # Server error - retry with exponential backoff
                            if attempt < max_retries - 1:
                                wait_time = 2 ** attempt  # 1s, 2s, 4s
                                lib_logger.warning(f"Server error (HTTP {status_code}), retry {attempt + 1}/{max_retries} in {wait_time}s")
                                await asyncio.sleep(wait_time)
                                continue
                            raise  # Final attempt failed

                        else:
                            # Other errors - don't retry
                            raise

                    except (httpx.RequestError, httpx.TimeoutException) as e:
                        # Network errors - retry with backoff
                        last_error = e
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            lib_logger.warning(f"Network error during refresh: {e}, retry {attempt + 1}/{max_retries} in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        raise

            # If we exhausted retries without success
            if new_token_data is None:
                raise last_error or Exception("Token refresh failed after all retries")

            # [FIX 1] Update OAuth token fields from response
            creds["access_token"] = new_token_data["access_token"]
            expiry_timestamp = time.time() + new_token_data["expires_in"]
            creds["expiry_date"] = expiry_timestamp * 1000 # gemini-cli format

            # [FIX 2] Update refresh_token if server provided a new one (rare but possible with Google OAuth)
            if "refresh_token" in new_token_data:
                creds["refresh_token"] = new_token_data["refresh_token"]

            # [FIX 3] Ensure all required OAuth client fields are present (restore if missing)
            if "client_id" not in creds or not creds["client_id"]:
                creds["client_id"] = CLIENT_ID
            if "client_secret" not in creds or not creds["client_secret"]:
                creds["client_secret"] = CLIENT_SECRET
            if "token_uri" not in creds or not creds["token_uri"]:
                creds["token_uri"] = TOKEN_URI
            if "universe_domain" not in creds or not creds["universe_domain"]:
                creds["universe_domain"] = "googleapis.com"

            # [FIX 4] Add scopes array if missing
            if "scopes" not in creds:
                creds["scopes"] = [
                    "https://www.googleapis.com/auth/cloud-platform",
                    "https://www.googleapis.com/auth/userinfo.email",
                    "https://www.googleapis.com/auth/userinfo.profile",
                ]

            # [FIX 5] Ensure _proxy_metadata exists and update timestamp
            if "_proxy_metadata" not in creds:
                creds["_proxy_metadata"] = {}
            creds["_proxy_metadata"]["last_check_timestamp"] = time.time()

            # [VALIDATION] Verify refreshed credentials have all required fields
            required_fields = ["access_token", "refresh_token", "client_id", "client_secret", "token_uri"]
            missing_fields = [field for field in required_fields if not creds.get(field)]
            if missing_fields:
                raise ValueError(f"Refreshed credentials missing required fields: {missing_fields}")

            # [VALIDATION] Optional: Test that the refreshed token is actually usable
            try:
                async with httpx.AsyncClient() as client:
                    test_response = await client.get(
                        USER_INFO_URI,
                        headers={"Authorization": f"Bearer {creds['access_token']}"},
                        timeout=5.0
                    )
                    test_response.raise_for_status()
                    lib_logger.debug(f"Token validation successful for '{Path(path).name}'")
            except Exception as e:
                lib_logger.warning(f"Refreshed token validation failed for '{Path(path).name}': {e}")
                # Don't fail the refresh - the token might still work for other endpoints
                # But log it for debugging purposes

            await self._save_credentials(path, creds)
            lib_logger.debug(f"Successfully refreshed Gemini OAuth token for '{Path(path).name}'.")
            return creds

    async def proactively_refresh(self, credential_path: str):
        # [BACKOFF] Check if refresh is in backoff period (matches Go's refreshFailureBackoff)
        now = time.time()
        if credential_path in self._next_refresh_after:
            backoff_until = self._next_refresh_after[credential_path]
            if now < backoff_until:
                remaining = int(backoff_until - now)
                lib_logger.debug(f"Skipping refresh for '{Path(credential_path).name}' (in backoff for {remaining}s)")
                return

        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            try:
                await self._refresh_token(credential_path, creds)
                # [SUCCESS] Clear failure tracking on successful refresh
                self._refresh_failures.pop(credential_path, None)
                self._next_refresh_after.pop(credential_path, None)
                lib_logger.debug(f"Successfully refreshed '{Path(credential_path).name}', cleared failure tracking")
            except Exception as e:
                # [FAILURE] Increment failure count and set exponential backoff
                failures = self._refresh_failures.get(credential_path, 0) + 1
                self._refresh_failures[credential_path] = failures

                # Exponential backoff: 5min → 10min → 20min → 40min → max 1 hour
                backoff_seconds = min(300 * (2 ** (failures - 1)), 3600)
                self._next_refresh_after[credential_path] = now + backoff_seconds

                lib_logger.error(
                    f"Refresh failed for '{Path(credential_path).name}' "
                    f"(attempt {failures}). Next retry in {backoff_seconds}s. Error: {e}"
                )
                # Don't re-raise - let background refresher continue with other credentials

    async def _get_lock(self, path: str) -> asyncio.Lock:
        # [FIX RACE CONDITION] Protect lock creation with a master lock
        # This prevents TOCTOU bug where multiple coroutines check and create simultaneously
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

    async def initialize_token(self, creds_or_path: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        path = creds_or_path if isinstance(creds_or_path, str) else None

        # Get display name from metadata if available, otherwise derive from path
        if isinstance(creds_or_path, dict):
            display_name = creds_or_path.get("_proxy_metadata", {}).get("display_name", "in-memory object")
        else:
            display_name = Path(path).name if path else "in-memory object"

        lib_logger.debug(f"Initializing Gemini token for '{display_name}'...")
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

                lib_logger.warning(f"Gemini OAuth token for '{display_name}' needs setup: {reason}.")
                auth_code_future = asyncio.get_event_loop().create_future()
                server = None

                async def handle_callback(reader, writer):
                    try:
                        request_line_bytes = await reader.readline()
                        if not request_line_bytes: return
                        path = request_line_bytes.decode('utf-8').strip().split(' ')[1]
                        while await reader.readline() != b'\r\n': pass
                        from urllib.parse import urlparse, parse_qs
                        query_params = parse_qs(urlparse(path).query)
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
                        "scope": " ".join(["https://www.googleapis.com/auth/cloud-platform", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"]),
                        "access_type": "offline", "response_type": "code", "prompt": "consent"
                    })
                    auth_panel_text = Text.from_markup("1. Your browser will now open to log in and authorize the application.\n2. If it doesn't, please open the URL below manually.")
                    console.print(Panel(auth_panel_text, title=f"Gemini OAuth Setup for [bold yellow]{display_name}[/bold yellow]", style="bold blue"))
                    console.print(f"[bold]URL:[/bold] [link={auth_url}]{auth_url}[/link]\n")
                    webbrowser.open(auth_url)
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
                        "code": auth_code.strip(), "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
                        "redirect_uri": "http://localhost:8085/oauth2callback", "grant_type": "authorization_code"
                    })
                    response.raise_for_status()
                    token_data = response.json()
                    # Start with the full token data from the exchange
                    creds = token_data.copy()
                    
                    # Convert 'expires_in' to 'expiry_date' in milliseconds
                    creds["expiry_date"] = (time.time() + creds.pop("expires_in")) * 1000
                    
                    # Ensure client_id and client_secret are present
                    creds["client_id"] = CLIENT_ID
                    creds["client_secret"] = CLIENT_SECRET

                    creds["token_uri"] = TOKEN_URI
                    creds["universe_domain"] = "googleapis.com"
                    
                    # Fetch user info and add metadata
                    user_info_response = await client.get(USER_INFO_URI, headers={"Authorization": f"Bearer {creds['access_token']}"})
                    user_info_response.raise_for_status()
                    user_info = user_info_response.json()
                    creds["_proxy_metadata"] = {
                        "email": user_info.get("email"),
                        "last_check_timestamp": time.time()
                    }

                    if path:
                        await self._save_credentials(path, creds)
                    lib_logger.info(f"Gemini OAuth initialized successfully for '{display_name}'.")
                return creds

            lib_logger.info(f"Gemini OAuth token at '{display_name}' is valid.")
            return creds
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini OAuth for '{path}': {e}")

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_path, creds)
        return {"Authorization": f"Bearer {creds['access_token']}"}

    async def get_user_info(self, creds_or_path: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        path = creds_or_path if isinstance(creds_or_path, str) else None
        creds = await self._load_credentials(creds_or_path) if path else creds_or_path

        if path and self._is_token_expired(creds):
            creds = await self._refresh_token(path, creds)
        
        # Prefer locally stored metadata
        if creds.get("_proxy_metadata", {}).get("email"):
            if path:
                creds["_proxy_metadata"]["last_check_timestamp"] = time.time()
                await self._save_credentials(path, creds)
            return {"email": creds["_proxy_metadata"]["email"]}

        # Fallback to API call if metadata is missing
        headers = {"Authorization": f"Bearer {creds['access_token']}"}
        async with httpx.AsyncClient() as client:
            response = await client.get(USER_INFO_URI, headers=headers)
            response.raise_for_status()
            user_info = response.json()
            
            # Save the retrieved info for future use
            creds["_proxy_metadata"] = {
                "email": user_info.get("email"),
                "last_check_timestamp": time.time()
            }
            if path:
                await self._save_credentials(path, creds)
            return {"email": user_info.get("email")}