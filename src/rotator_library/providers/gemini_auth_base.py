# src/rotator_library/providers/gemini_auth_base.py

import subprocess
import webbrowser
from typing import Optional
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

import httpx

lib_logger = logging.getLogger('rotator_library')

CLIENT_ID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com" ##https://api.kilocode.ai/extension-config.json
CLIENT_SECRET = "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" ##https://api.kilocode.ai/extension-config.json
TOKEN_URI = "https://oauth2.googleapis.com/token"
REFRESH_EXPIRY_BUFFER_SECONDS = 300

class GeminiAuthBase:
    def __init__(self):
        self._credentials_cache: Dict[str, Dict[str, Any]] = {}
        self._refresh_locks: Dict[str, asyncio.Lock] = {}

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        if path in self._credentials_cache:
            return self._credentials_cache[path]
        
        async with self._get_lock(path):
            if path in self._credentials_cache:
                return self._credentials_cache[path]
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
            creds["token_expiry"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expiry_timestamp)) # gcloud format
            
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

    # [NEW] Add init flow for invalid/expired tokens
    async def initialize_token(self, path: str) -> Dict[str, Any]:
        """Initiates OAuth flow if tokens are missing or invalid."""
        lib_logger.debug(f"Initializing Gemini token at '{path}'...")
        try:
            creds = await self._load_credentials(path)
            
            reason = ""
            if not creds.get("refresh_token"):
                reason = "refresh token is missing"
            elif self._is_token_expired(creds):
                reason = "token is expired"

            if reason:
                lib_logger.warning(f"Gemini OAuth token for '{Path(path).name}' needs setup: {reason}.")
                # Use subprocess to run gemini-cli setup or simulate web flow
                # Based on CLIProxyAPI-main/gemini/gemini_auth.go: Use web flow with local server
                # Automated web flow for OAuth
                auth_code_future = asyncio.get_event_loop().create_future()
                server = None

                async def handle_callback(reader, writer):
                    try:
                        request_line_bytes = await reader.readline()
                        if not request_line_bytes:
                            return
                        request_line = request_line_bytes.decode('utf-8').strip()
                        path = request_line.split(' ')[1]
                        
                        # Consume headers
                        while await reader.readline() != b'\r\n':
                            pass

                        from urllib.parse import urlparse, parse_qs
                        query_params = parse_qs(urlparse(path).query)
                        
                        writer.write(b"HTTP/1.1 200 OK\r\n")
                        writer.write(b"Content-Type: text/html\r\n\r\n")
                        
                        if 'code' in query_params:
                            auth_code = query_params['code'][0]
                            if not auth_code_future.done():
                                auth_code_future.set_result(auth_code)
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
                        "access_type": "offline",
                        "response_type": "code",
                        "prompt": "consent"
                    })

                    print(f"\n--- Gemini OAuth Setup Required for {Path(path).name} ---")
                    print("Your browser will now open for you to log in and authorize the application.")
                    print(f"If it doesn't, please open this URL manually:\n\n{auth_url}\n")
                    webbrowser.open(auth_url)

                    auth_code = await asyncio.wait_for(auth_code_future, timeout=300) # 5-minute timeout
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
                    creds = {
                        "access_token": token_data["access_token"],
                        "refresh_token": token_data["refresh_token"],
                        "expiry_date": (time.time() + token_data["expires_in"]) * 1000,
                        "client_id": CLIENT_ID,
                        "client_secret": CLIENT_SECRET
                    }
                    await self._save_credentials(path, creds)
                    lib_logger.info(f"Gemini OAuth initialized successfully for '{Path(path).name}'.")
                return creds
            
            lib_logger.info(f"Gemini OAuth token at '{Path(path).name}' is valid.")
            return creds
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini OAuth for '{path}': {e}")

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_path, creds)
        return {"Authorization": f"Bearer {creds['access_token']}"}

    async def get_user_info(self, credential_path: str) -> Dict[str, Any]:
        """Fetches user info from Google's userinfo endpoint."""
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_path, creds)
        
        headers = {"Authorization": f"Bearer {creds['access_token']}"}
        async with httpx.AsyncClient() as client:
            response = await client.get("https://www.googleapis.com/oauth2/v1/userinfo", headers=headers)
            response.raise_for_status()
            return response.json()