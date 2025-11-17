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

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

lib_logger = logging.getLogger('rotator_library')

CLIENT_ID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com" #https://api.kilocode.ai/extension-config.json
CLIENT_SECRET = "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" #https://api.kilocode.ai/extension-config.json
TOKEN_URI = "https://oauth2.googleapis.com/token"
USER_INFO_URI = "https://www.googleapis.com/oauth2/v1/userinfo"
REFRESH_EXPIRY_BUFFER_SECONDS = 300

console = Console()

class GeminiAuthBase:
    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}

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

        async with self._get_lock(path):
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
        self._credentials_cache[path] = creds

        # Don't save to file if credentials were loaded from environment
        if creds.get("_proxy_metadata", {}).get("loaded_from_env"):
            lib_logger.debug("Credentials loaded from env, skipping file save")
            return

        try:
            with open(path, 'w') as f:
                json.dump(creds, f, indent=2)
            lib_logger.debug(f"Saved updated Gemini OAuth credentials to '{path}'.")
        except Exception as e:
            lib_logger.error(f"Failed to save updated Gemini OAuth credentials to '{path}': {e}")

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        expiry = creds.get("token_expiry") # gcloud format
        if not expiry: # gemini-cli format
             expiry_timestamp = creds.get("expiry_date", 0) / 1000
        else:
            expiry_timestamp = time.mktime(time.strptime(expiry, "%Y-%m-%dT%H:%M:%SZ"))
        return expiry_timestamp < time.time() + REFRESH_EXPIRY_BUFFER_SECONDS

    async def _refresh_token(self, path: str, creds: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
        async with self._get_lock(path):
            # Skip the expiry check if a refresh is being forced
            if not force and not self._is_token_expired(self._credentials_cache.get(path, creds)):
                return self._credentials_cache.get(path, creds)

            lib_logger.info(f"Refreshing Gemini OAuth token for '{Path(path).name}' (forced: {force})...")
            refresh_token = creds.get("refresh_token")
            if not refresh_token:
                raise ValueError("No refresh_token found in credentials file.")

            async with httpx.AsyncClient() as client:
                response = await client.post(TOKEN_URI, data={
                    "client_id": creds.get("client_id", CLIENT_ID),
                    "client_secret": creds.get("client_secret", CLIENT_SECRET),
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                })
                response.raise_for_status()
                new_token_data = response.json()

            creds["access_token"] = new_token_data["access_token"]
            expiry_timestamp = time.time() + new_token_data["expires_in"]
            creds["expiry_date"] = expiry_timestamp * 1000 # gemini-cli format
            
            if creds.get("_proxy_metadata"):
                creds["_proxy_metadata"]["last_check_timestamp"] = time.time()

            await self._save_credentials(path, creds)
            lib_logger.info(f"Successfully refreshed Gemini OAuth token for '{Path(path).name}'.")
            return creds

    async def proactively_refresh(self, credential_path: str):
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            await self._refresh_token(credential_path, creds)

    def _get_lock(self, path: str) -> asyncio.Lock:
        if path not in self._refresh_locks:
            self._refresh_locks[path] = asyncio.Lock()
        return self._refresh_locks[path]

    async def initialize_token(self, creds_or_path: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        path = creds_or_path if isinstance(creds_or_path, str) else None
        file_name = Path(path).name if path else "in-memory object"
        lib_logger.debug(f"Initializing Gemini token for '{file_name}'...")
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
                        lib_logger.warning(f"Automatic token refresh for '{file_name}' failed: {e}. Proceeding to interactive login.")

                lib_logger.warning(f"Gemini OAuth token for '{file_name}' needs setup: {reason}.")
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
                    console.print(Panel(auth_panel_text, title=f"Gemini OAuth Setup for [bold yellow]{file_name}[/bold yellow]", style="bold blue"))
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
                    
                    # Add additional fields for compatibility with the Go implementation
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
                    lib_logger.info(f"Gemini OAuth initialized successfully for '{file_name}'.")
                return creds
            
            lib_logger.info(f"Gemini OAuth token at '{file_name}' is valid.")
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