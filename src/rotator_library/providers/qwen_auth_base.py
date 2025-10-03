# src/rotator_library/providers/qwen_auth_base.py

import secrets
import hashlib
import base64
import json
import time
import asyncio
import logging
import webbrowser
from pathlib import Path
from typing import Dict, Any, Tuple, Union

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

lib_logger = logging.getLogger('rotator_library')

CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56" #https://api.kilocode.ai/extension-config.json
SCOPE = "openid profile email model.completion"
TOKEN_ENDPOINT = "https://chat.qwen.ai/api/v1/oauth2/token"
REFRESH_EXPIRY_BUFFER_SECONDS = 300

console = Console()

class QwenAuthBase:
    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}

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
        """Loads credentials from cache or file."""
        if path in self._credentials_cache:
            return self._credentials_cache[path]
        
        async with self._get_lock(path):
            # Re-check cache after acquiring lock
            if path in self._credentials_cache:
                return self._credentials_cache[path]
            return await self._read_creds_from_file(path)

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
        self._credentials_cache[path] = creds
        try:
            with open(path, 'w') as f:
                json.dump(creds, f, indent=2)
            lib_logger.debug(f"Saved updated Qwen OAuth credentials to '{path}'.")
        except Exception as e:
            lib_logger.error(f"Failed to save updated Qwen OAuth credentials to '{path}': {e}")

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        expiry_timestamp = creds.get("expiry_date", 0) / 1000
        return expiry_timestamp < time.time() + REFRESH_EXPIRY_BUFFER_SECONDS

    async def _refresh_token(self, path: str, force: bool = False) -> Dict[str, Any]:
        async with self._get_lock(path):
            cached_creds = self._credentials_cache.get(path)
            if not force and cached_creds and not self._is_token_expired(cached_creds):
                return cached_creds

            # If cache is empty, read from file. This is safe because we hold the lock.
            if path not in self._credentials_cache:
                await self._read_creds_from_file(path)
            
            creds_from_file = self._credentials_cache[path]
            
            lib_logger.info(f"Refreshing Qwen OAuth token for '{Path(path).name}'...")
            refresh_token = creds_from_file.get("refresh_token")
            if not refresh_token:
                raise ValueError("No refresh_token found in Qwen credentials file.")

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(TOKEN_ENDPOINT, headers=headers, data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": CLIENT_ID,
                })
                response.raise_for_status()
                new_token_data = response.json()

            creds_from_file["access_token"] = new_token_data["access_token"]
            creds_from_file["refresh_token"] = new_token_data.get("refresh_token", creds_from_file["refresh_token"])
            creds_from_file["expiry_date"] = (time.time() + new_token_data["expires_in"]) * 1000
            
            # Update timestamp in metadata if it exists
            if creds_from_file.get("_proxy_metadata"):
                creds_from_file["_proxy_metadata"]["last_check_timestamp"] = time.time()

            await self._save_credentials(path, creds_from_file)
            lib_logger.info(f"Successfully refreshed Qwen OAuth token for '{Path(path).name}'.")
            return creds_from_file

    def get_api_details(self, credential_path: str) -> Tuple[str, str]:
        creds = self._credentials_cache[credential_path]
        base_url = creds.get("resource_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        if not base_url.startswith("http"):
            base_url = f"https://{base_url}"
        return base_url, creds["access_token"]

    async def proactively_refresh(self, credential_path: str):
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            await self._refresh_token(credential_path)
    
    def _get_lock(self, path: str) -> asyncio.Lock:
        if path not in self._refresh_locks:
            self._refresh_locks[path] = asyncio.Lock()
        return self._refresh_locks[path]

    async def initialize_token(self, creds_or_path: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """Initiates device flow if tokens are missing or invalid."""
        path = creds_or_path if isinstance(creds_or_path, str) else None
        file_name = Path(path).name if path else "in-memory object"
        lib_logger.debug(f"Initializing Qwen token for '{file_name}'...")
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
                        lib_logger.warning(f"Automatic token refresh for '{file_name}' failed: {e}. Proceeding to interactive login.")
                
                lib_logger.warning(f"Qwen OAuth token for '{file_name}' needs setup: {reason}.")
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
                    
                    auth_panel_text = Text.from_markup(
                        "1. Visit the URL below to sign in.\n"
                        "2. [bold]Copy your email[/bold] or another unique identifier and authorize the application.\n"
                        "3. You will be prompted to enter your identifier after authorization."
                    )
                    console.print(Panel(auth_panel_text, title=f"Qwen OAuth Setup for [bold yellow]{file_name}[/bold yellow]", style="bold blue"))
                    console.print(f"[bold]URL:[/bold] [link={dev_data['verification_uri_complete']}]{dev_data['verification_uri_complete']}[/link]\n")
                    webbrowser.open(dev_data['verification_uri_complete'])
                    
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
                            prompt_text = Text.from_markup(f"\n[bold]Please enter your email or a unique identifier for [yellow]'{file_name}'[/yellow][/bold]")
                            email = Prompt.ask(prompt_text)
                            creds["_proxy_metadata"] = {
                                "email": email.strip(),
                                "last_check_timestamp": time.time()
                            }
                        except (EOFError, KeyboardInterrupt):
                            console.print("\n[bold yellow]No identifier provided. Deduplication will not be possible.[/bold yellow]")
                            creds["_proxy_metadata"] = {"email": None, "last_check_timestamp": time.time()}
                    
                    if path:
                        await self._save_credentials(path, creds)
                    lib_logger.info(f"Qwen OAuth initialized successfully for '{file_name}'.")
                return creds
            
            lib_logger.info(f"Qwen OAuth token at '{file_name}' is valid.")
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