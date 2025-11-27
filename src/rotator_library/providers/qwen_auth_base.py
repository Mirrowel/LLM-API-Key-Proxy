# src/rotator_library/providers/qwen_auth_base.py

import secrets
import hashlib
import base64
import json
import time
import asyncio
import logging
import webbrowser
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional
import tempfile
import shutil

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from ..utils.headless_detection import is_headless_environment

lib_logger = logging.getLogger('rotator_library')

CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56" #https://api.kilocode.ai/extension-config.json
SCOPE = "openid profile email model.completion"
TOKEN_ENDPOINT = "https://chat.qwen.ai/api/v1/oauth2/token"
REFRESH_EXPIRY_BUFFER_SECONDS = 3 * 60 * 60  # 3 hours buffer before expiry

console = Console()

class QwenAuthBase:
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

    def _parse_env_credential_path(self, path: str) -> Optional[str]:
        """
        Parse a virtual env:// path and return the credential index.
        
        Supported formats:
        - "env://provider/0" - Legacy single credential (no index in env var names)
        - "env://provider/1" - First numbered credential (QWEN_CODE_1_ACCESS_TOKEN)
        
        Returns:
            The credential index as string, or None if path is not an env:// path
        """
        if not path.startswith("env://"):
            return None
        
        parts = path[6:].split("/")
        if len(parts) >= 2:
            return parts[1]
        return "0"

    def _load_from_env(self, credential_index: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load OAuth credentials from environment variables for stateless deployments.

        Supports two formats:
        1. Legacy (credential_index="0" or None): QWEN_CODE_ACCESS_TOKEN
        2. Numbered (credential_index="1", "2", etc.): QWEN_CODE_1_ACCESS_TOKEN, etc.

        Expected environment variables (for numbered format with index N):
        - QWEN_CODE_{N}_ACCESS_TOKEN (required)
        - QWEN_CODE_{N}_REFRESH_TOKEN (required)
        - QWEN_CODE_{N}_EXPIRY_DATE (optional, defaults to 0)
        - QWEN_CODE_{N}_RESOURCE_URL (optional, defaults to https://portal.qwen.ai/v1)
        - QWEN_CODE_{N}_EMAIL (optional, defaults to "env-user-{N}")

        Returns:
            Dict with credential structure if env vars present, None otherwise
        """
        # Determine the env var prefix based on credential index
        if credential_index and credential_index != "0":
            prefix = f"QWEN_CODE_{credential_index}"
            default_email = f"env-user-{credential_index}"
        else:
            prefix = "QWEN_CODE"
            default_email = "env-user"
        
        access_token = os.getenv(f"{prefix}_ACCESS_TOKEN")
        refresh_token = os.getenv(f"{prefix}_REFRESH_TOKEN")

        # Both access and refresh tokens are required
        if not (access_token and refresh_token):
            return None

        lib_logger.debug(f"Loading Qwen Code credentials from environment variables (prefix: {prefix})")

        # Parse expiry_date as float, default to 0 if not present
        expiry_str = os.getenv(f"{prefix}_EXPIRY_DATE", "0")
        try:
            expiry_date = float(expiry_str)
        except ValueError:
            lib_logger.warning(f"Invalid {prefix}_EXPIRY_DATE value: {expiry_str}, using 0")
            expiry_date = 0

        creds = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry_date": expiry_date,
            "resource_url": os.getenv(f"{prefix}_RESOURCE_URL", "https://portal.qwen.ai/v1"),
            "_proxy_metadata": {
                "email": os.getenv(f"{prefix}_EMAIL", default_email),
                "last_check_timestamp": time.time(),
                "loaded_from_env": True,
                "env_credential_index": credential_index or "0"
            }
        }

        return creds

    async def _read_creds_from_file(self, path: str) -> Dict[str, Any]:
        """Reads credentials from file and populates the cache. No locking."""
        try:
            lib_logger.debug(f"Reading Qwen credentials from file: {path}")
            with open(path, 'r') as f:
                creds = json.load(f)
            self._credentials_cache[path] = creds
            return creds
        except FileNotFoundError:
            raise IOError(f"Qwen OAuth credential file not found at '{path}'")
        except Exception as e:
            raise IOError(f"Failed to load Qwen OAuth credentials from '{path}': {e}")

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
                    lib_logger.info(f"Using Qwen Code credentials from environment variables (index: {credential_index})")
                    self._credentials_cache[path] = env_creds
                    return env_creds
                else:
                    raise IOError(f"Environment variables for Qwen Code credential index {credential_index} not found")

            # For file paths, try loading from legacy env vars first
            env_creds = self._load_from_env()
            if env_creds:
                lib_logger.info("Using Qwen Code credentials from environment variables")
                self._credentials_cache[path] = env_creds
                return env_creds

            # Fall back to file-based loading
            return await self._read_creds_from_file(path)

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
        # Don't save to file if credentials were loaded from environment
        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            lib_logger.debug("Credentials loaded from env, skipping file save")
            # Still update cache for in-memory consistency
            self._credentials_cache[path] = creds
            return

        # [ATOMIC WRITE] Use tempfile + move pattern to ensure atomic writes
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

            # Update cache AFTER successful file write
            self._credentials_cache[path] = creds
            lib_logger.debug(f"Saved updated Qwen OAuth credentials to '{path}' (atomic write).")

        except Exception as e:
            lib_logger.error(f"Failed to save updated Qwen OAuth credentials to '{path}': {e}")
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
        expiry_timestamp = creds.get("expiry_date", 0) / 1000
        return expiry_timestamp < time.time() + REFRESH_EXPIRY_BUFFER_SECONDS

    async def _refresh_token(self, path: str, force: bool = False) -> Dict[str, Any]:
        async with await self._get_lock(path):
            cached_creds = self._credentials_cache.get(path)
            if not force and cached_creds and not self._is_token_expired(cached_creds):
                return cached_creds

            # If cache is empty, read from file. This is safe because we hold the lock.
            if path not in self._credentials_cache:
                await self._read_creds_from_file(path)

            creds_from_file = self._credentials_cache[path]

            lib_logger.debug(f"Refreshing Qwen OAuth token for '{Path(path).name}'...")
            refresh_token = creds_from_file.get("refresh_token")
            if not refresh_token:
                lib_logger.error(f"No refresh_token found in '{Path(path).name}'")
                raise ValueError("No refresh_token found in Qwen credentials file.")

            # [RETRY LOGIC] Implement exponential backoff for transient errors
            max_retries = 3
            new_token_data = None
            last_error = None
            needs_reauth = False

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            async with httpx.AsyncClient() as client:
                for attempt in range(max_retries):
                    try:
                        response = await client.post(TOKEN_ENDPOINT, headers=headers, data={
                            "grant_type": "refresh_token",
                            "refresh_token": refresh_token,
                            "client_id": CLIENT_ID,
                        }, timeout=30.0)
                        response.raise_for_status()
                        new_token_data = response.json()
                        break  # Success

                    except httpx.HTTPStatusError as e:
                        last_error = e
                        status_code = e.response.status_code
                        error_body = e.response.text
                        lib_logger.error(f"HTTP {status_code} for '{Path(path).name}': {error_body}")

                        # [INVALID GRANT HANDLING] Handle 401/403 by triggering re-authentication
                        if status_code in (401, 403):
                            lib_logger.warning(
                                f"Refresh token invalid for '{Path(path).name}' (HTTP {status_code}). "
                                f"Token may have been revoked or expired. Starting re-authentication..."
                            )
                            needs_reauth = True
                            break  # Exit retry loop to trigger re-auth

                        elif status_code == 429:
                            retry_after = int(e.response.headers.get("Retry-After", 60))
                            lib_logger.warning(f"Rate limited (HTTP 429), retry after {retry_after}s")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_after)
                                continue
                            raise

                        elif 500 <= status_code < 600:
                            if attempt < max_retries - 1:
                                wait_time = 2 ** attempt
                                lib_logger.warning(f"Server error (HTTP {status_code}), retry {attempt + 1}/{max_retries} in {wait_time}s")
                                await asyncio.sleep(wait_time)
                                continue
                            raise

                        else:
                            raise

                    except (httpx.RequestError, httpx.TimeoutException) as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            lib_logger.warning(f"Network error during refresh: {e}, retry {attempt + 1}/{max_retries} in {wait_time}s")
                            await asyncio.sleep(wait_time)
                            continue
                        raise

            # [INVALID GRANT RE-AUTH] Trigger OAuth flow if refresh token is invalid
            if needs_reauth:
                lib_logger.info(f"Starting re-authentication for '{Path(path).name}'...")
                try:
                    # Call initialize_token to trigger OAuth flow
                    new_creds = await self.initialize_token(path)
                    return new_creds
                except Exception as reauth_error:
                    lib_logger.error(f"Re-authentication failed for '{Path(path).name}': {reauth_error}")
                    raise ValueError(f"Refresh token invalid and re-authentication failed: {reauth_error}")

            if new_token_data is None:
                raise last_error or Exception("Token refresh failed after all retries")

            creds_from_file["access_token"] = new_token_data["access_token"]
            creds_from_file["refresh_token"] = new_token_data.get("refresh_token", creds_from_file["refresh_token"])
            creds_from_file["expiry_date"] = (time.time() + new_token_data["expires_in"]) * 1000
            creds_from_file["resource_url"] = new_token_data.get("resource_url", creds_from_file.get("resource_url"))

            # Ensure _proxy_metadata exists and update timestamp
            if "_proxy_metadata" not in creds_from_file:
                creds_from_file["_proxy_metadata"] = {}
            creds_from_file["_proxy_metadata"]["last_check_timestamp"] = time.time()

            await self._save_credentials(path, creds_from_file)
            lib_logger.debug(f"Successfully refreshed Qwen OAuth token for '{Path(path).name}'.")
            return creds_from_file

    async def get_api_details(self, credential_identifier: str) -> Tuple[str, str]:
        """
        Returns the API base URL and access token.

        Supports both credential types:
        - OAuth: credential_identifier is a file path to JSON credentials
        - API Key: credential_identifier is the API key string itself
        """
        # Detect credential type
        if os.path.isfile(credential_identifier):
            # OAuth credential: file path to JSON
            lib_logger.debug(f"Using OAuth credentials from file: {credential_identifier}")
            creds = await self._load_credentials(credential_identifier)

            if self._is_token_expired(creds):
                creds = await self._refresh_token(credential_identifier)
                
            base_url = creds.get("resource_url", "https://portal.qwen.ai/v1")
            if not base_url.startswith("http"):
                base_url = f"https://{base_url}"
            access_token = creds["access_token"]
        else:
            # Direct API key: use as-is
            lib_logger.debug("Using direct API key for Qwen Code")
            base_url = "https://portal.qwen.ai/v1"
            access_token = credential_identifier

        return base_url, access_token

    async def proactively_refresh(self, credential_identifier: str):
        """
        Proactively refreshes tokens if they're close to expiry.
        Only applies to OAuth credentials (file paths). Direct API keys are skipped.
        """
        # Only refresh if it's an OAuth credential (file path)
        if not os.path.isfile(credential_identifier):
            return  # Direct API key, no refresh needed

        creds = await self._load_credentials(credential_identifier)
        if self._is_token_expired(creds):
            # Queue for refresh with needs_reauth=False (automated refresh)
            await self._queue_refresh(credential_identifier, force=False, needs_reauth=False)

    async def _get_lock(self, path: str) -> asyncio.Lock:
        # [FIX RACE CONDITION] Protect lock creation with a master lock
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

    def is_credential_available(self, path: str) -> bool:
        """Check if a credential is available for rotation (not queued/refreshing)."""
        return path not in self._unavailable_credentials

    async def _ensure_queue_processor_running(self):
        """Lazily starts the queue processor if not already running."""
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(self._process_refresh_queue())

    async def _queue_refresh(self, path: str, force: bool = False, needs_reauth: bool = False):
        """Add a credential to the refresh queue if not already queued.
        
        Args:
            path: Credential file path
            force: Force refresh even if not expired
            needs_reauth: True if full re-authentication needed (bypasses backoff)
        """
        # IMPORTANT: Only check backoff for simple automated refreshes
        # Re-authentication (interactive OAuth) should BYPASS backoff since it needs user input
        if not needs_reauth:
            now = time.time()
            if path in self._next_refresh_after:
                backoff_until = self._next_refresh_after[path]
                if now < backoff_until:
                    # Credential is in backoff for automated refresh, do not queue
                    remaining = int(backoff_until - now)
                    lib_logger.debug(f"Skipping automated refresh for '{Path(path).name}' (in backoff for {remaining}s)")
                    return
        
        async with self._queue_tracking_lock:
            if path not in self._queued_credentials:
                self._queued_credentials.add(path)
                self._unavailable_credentials.add(path)  # Mark as unavailable
                await self._refresh_queue.put((path, force, needs_reauth))
                await self._ensure_queue_processor_running()

    async def _process_refresh_queue(self):
        """Background worker that processes refresh requests sequentially."""
        while True:
            path = None
            try:
                # Wait for an item with timeout to allow graceful shutdown
                try:
                    path, force, needs_reauth = await asyncio.wait_for(
                        self._refresh_queue.get(), 
                        timeout=60.0
                    )
                except asyncio.TimeoutError:
                    # No items for 60s, exit to save resources
                    self._queue_processor_task = None
                    return
                
                try:
                    # Perform the actual refresh (still using per-credential lock)
                    async with await self._get_lock(path):
                        # Re-check if still expired (may have changed since queueing)
                        creds = self._credentials_cache.get(path)
                        if creds and not self._is_token_expired(creds):
                            # No longer expired, mark as available
                            async with self._queue_tracking_lock:
                                self._unavailable_credentials.discard(path)
                            continue
                        
                        # Perform refresh
                        if not creds:
                            creds = await self._load_credentials(path)
                        await self._refresh_token(path, force=force)
                        
                        # SUCCESS: Mark as available again
                        async with self._queue_tracking_lock:
                            self._unavailable_credentials.discard(path)
                        
                finally:
                    # Remove from queued set
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                    self._refresh_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                lib_logger.error(f"Error in queue processor: {e}")
                # Even on error, mark as available (backoff will prevent immediate retry)
                if path:
                    async with self._queue_tracking_lock:
                        self._unavailable_credentials.discard(path)

    async def initialize_token(self, creds_or_path: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """Initiates device flow if tokens are missing or invalid."""
        path = creds_or_path if isinstance(creds_or_path, str) else None

        # Get display name from metadata if available, otherwise derive from path
        if isinstance(creds_or_path, dict):
            display_name = creds_or_path.get("_proxy_metadata", {}).get("display_name", "in-memory object")
        else:
            display_name = Path(path).name if path else "in-memory object"

        lib_logger.debug(f"Initializing Qwen token for '{display_name}'...")
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
                        return await self._refresh_token(path)
                    except Exception as e:
                        lib_logger.warning(f"Automatic token refresh for '{display_name}' failed: {e}. Proceeding to interactive login.")

                lib_logger.warning(f"Qwen OAuth token for '{display_name}' needs setup: {reason}.")
                
                # [HEADLESS DETECTION] Check if running in headless environment
                is_headless = is_headless_environment()
                
                code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
                code_challenge = base64.urlsafe_b64encode(
                    hashlib.sha256(code_verifier.encode('utf-8')).digest()
                ).decode('utf-8').rstrip('=')
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json"
                }
                async with httpx.AsyncClient() as client:
                    request_data = {
                        "client_id": CLIENT_ID,
                        "scope": SCOPE,
                        "code_challenge": code_challenge,
                        "code_challenge_method": "S256"
                    }
                    lib_logger.debug(f"Qwen device code request data: {request_data}")
                    try:
                        dev_response = await client.post(
                            "https://chat.qwen.ai/api/v1/oauth2/device/code",
                            headers=headers,
                            data=request_data
                        )
                        dev_response.raise_for_status()
                        dev_data = dev_response.json()
                        lib_logger.debug(f"Qwen device auth response: {dev_data}")
                    except httpx.HTTPStatusError as e:
                        lib_logger.error(f"Qwen device code request failed with status {e.response.status_code}: {e.response.text}")
                        raise e
                    
                    # [HEADLESS SUPPORT] Display appropriate instructions
                    if is_headless:
                        auth_panel_text = Text.from_markup(
                            "Running in headless environment (no GUI detected).\n"
                            "Please open the URL below in a browser on another machine to authorize:\n"
                            "1. Visit the URL below to sign in.\n"
                            "2. [bold]Copy your email[/bold] or another unique identifier and authorize the application.\n"
                            "3. You will be prompted to enter your identifier after authorization."
                        )
                    else:
                        auth_panel_text = Text.from_markup(
                            "1. Visit the URL below to sign in.\n"
                            "2. [bold]Copy your email[/bold] or another unique identifier and authorize the application.\n"
                            "3. You will be prompted to enter your identifier after authorization."
                        )
                    
                    console.print(Panel(auth_panel_text, title=f"Qwen OAuth Setup for [bold yellow]{display_name}[/bold yellow]", style="bold blue"))
                    console.print(f"[bold]URL:[/bold] [link={dev_data['verification_uri_complete']}]{dev_data['verification_uri_complete']}[/link]\n")
                    
                    # [HEADLESS SUPPORT] Only attempt browser open if NOT headless
                    if not is_headless:
                        try:
                            webbrowser.open(dev_data['verification_uri_complete'])
                            lib_logger.info("Browser opened successfully for Qwen OAuth flow")
                        except Exception as e:
                            lib_logger.warning(f"Failed to open browser automatically: {e}. Please open the URL manually.")
                    
                    token_data = None
                    start_time = time.time()
                    interval = dev_data.get('interval', 5)

                    with console.status("[bold green]Polling for token, please complete authentication in the browser...[/bold green]", spinner="dots") as status:
                        while time.time() - start_time < dev_data['expires_in']:
                            poll_response = await client.post(
                                TOKEN_ENDPOINT,
                                headers=headers,
                                data={
                                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                                    "device_code": dev_data['device_code'],
                                    "client_id": CLIENT_ID,
                                    "code_verifier": code_verifier
                                }
                            )
                            if poll_response.status_code == 200:
                                token_data = poll_response.json()
                                lib_logger.info("Successfully received token.")
                                break
                            elif poll_response.status_code == 400:
                                poll_data = poll_response.json()
                                error_type = poll_data.get("error")
                                if error_type == "authorization_pending":
                                    lib_logger.debug(f"Polling status: {error_type}, waiting {interval}s")
                                elif error_type == "slow_down":
                                    interval = int(interval * 1.5)
                                    if interval > 10:
                                        interval = 10
                                    lib_logger.debug(f"Polling status: {error_type}, waiting {interval}s")
                                else:
                                    raise ValueError(f"Token polling failed: {poll_data.get('error_description', error_type)}")
                            else:
                                poll_response.raise_for_status()
                            
                            await asyncio.sleep(interval)
                    
                    if not token_data:
                        raise TimeoutError("Qwen device flow timed out.")
                    
                    creds.update({
                        "access_token": token_data["access_token"],
                        "refresh_token": token_data.get("refresh_token"),
                        "expiry_date": (time.time() + token_data["expires_in"]) * 1000,
                        "resource_url": token_data.get("resource_url")
                    })

                    # Prompt for user identifier and create metadata object if needed
                    if not creds.get("_proxy_metadata", {}).get("email"):
                        try:
                            prompt_text = Text.from_markup(f"\\n[bold]Please enter your email or a unique identifier for [yellow]'{display_name}'[/yellow][/bold]")
                            email = Prompt.ask(prompt_text)
                            creds["_proxy_metadata"] = {
                                "email": email.strip(),
                                "last_check_timestamp": time.time()
                            }
                        except (EOFError, KeyboardInterrupt):
                            console.print("\\n[bold yellow]No identifier provided. Deduplication will not be possible.[/bold yellow]")
                            creds["_proxy_metadata"] = {"email": None, "last_check_timestamp": time.time()}

                    if path:
                        await self._save_credentials(path, creds)
                    lib_logger.info(f"Qwen OAuth initialized successfully for '{display_name}'.")
                return creds

            lib_logger.info(f"Qwen OAuth token at '{display_name}' is valid.")
            return creds
        except Exception as e:
            raise ValueError(f"Failed to initialize Qwen OAuth for '{path}': {e}")

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_path)
        return {"Authorization": f"Bearer {creds['access_token']}"}

    async def get_user_info(self, creds_or_path: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Retrieves user info from the _proxy_metadata in the credential file.
        """
        try:
            path = creds_or_path if isinstance(creds_or_path, str) else None
            creds = await self._load_credentials(creds_or_path) if path else creds_or_path
            
            # This will ensure the token is valid and metadata exists if the flow was just run
            if path:
                await self.initialize_token(path)
                creds = await self._load_credentials(path) # Re-load after potential init

            metadata = creds.get("_proxy_metadata", {"email": None})
            email = metadata.get("email")

            if not email:
                lib_logger.warning(f"No email found in _proxy_metadata for '{path or 'in-memory object'}'.")

            # Update timestamp on check and save if it's a file-based credential
            if path and "_proxy_metadata" in creds:
                creds["_proxy_metadata"]["last_check_timestamp"] = time.time()
                await self._save_credentials(path, creds)

            return {"email": email}
        except Exception as e:
            lib_logger.error(f"Failed to get Qwen user info from credentials: {e}")
            return {"email": None}