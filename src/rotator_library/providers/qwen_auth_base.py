# src/rotator_library/providers/qwen_auth_base.py

import secrets
import hashlib
import base64
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import httpx

lib_logger = logging.getLogger('rotator_library')

CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56" ##https://api.kilocode.ai/extension-config.json
TOKEN_ENDPOINT = "https://chat.qwen.ai/api/v1/oauth2/token"
REFRESH_EXPIRY_BUFFER_SECONDS = 300

class QwenAuthBase:
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
                with open(path, 'r') as f:
                    creds = json.load(f)
                self._credentials_cache[path] = creds
                return creds
            except Exception as e:
                raise IOError(f"Failed to load Qwen OAuth credentials from '{path}': {e}")

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
        self._credentials_cache[path] = creds
        try:
            with open(path, 'w') as f:
                json.dump(creds, f, indent=2)
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

            creds_from_file = await self._load_credentials(path)
            
            lib_logger.info(f"Refreshing Qwen OAuth token for '{Path(path).name}'...")
            refresh_token = creds_from_file.get("refresh_token")
            if not refresh_token:
                raise ValueError("No refresh_token found in Qwen credentials file.")

            async with httpx.AsyncClient() as client:
                response = await client.post(TOKEN_ENDPOINT, data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": CLIENT_ID,
                })
                response.raise_for_status()
                new_token_data = response.json()

            creds_from_file["access_token"] = new_token_data["access_token"]
            creds_from_file["refresh_token"] = new_token_data.get("refresh_token", creds_from_file["refresh_token"])
            creds_from_file["expiry_date"] = (time.time() + new_token_data["expires_in"]) * 1000
            
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

    # [NEW] Add init flow for invalid/expired tokens
    async def initialize_token(self, path: str) -> Dict[str, Any]:
        """Initiates device flow if tokens are missing or invalid."""
        try:
            creds = await self._load_credentials(path)
            if not creds.get("refresh_token") or self._is_token_expired(creds):
                lib_logger.warning(f"Invalid or missing Qwen OAuth tokens at '{path}'. Initiating device flow...")
                # Based on CLIProxyAPI-main/qwen/qwen_auth.go: Use device code with PKCE
                code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
                code_challenge = base64.urlsafe_b64encode(
                    hashlib.sha256(code_verifier.encode('utf-8')).digest()
                ).decode('utf-8').rstrip('=')
                
                async with httpx.AsyncClient() as client:
                    dev_response = await client.post(
                        "https://chat.qwen.ai/api/v1/oauth2/device/code",
                        data={
                            "client_id": CLIENT_ID,
                            "scope": "openid profile email model.completion",
                            "code_challenge": code_challenge,
                            "code_challenge_method": "S256"
                        }
                    )
                    dev_response.raise_for_status()
                    dev_data = dev_response.json()
                    
                    print(f"\n--- Qwen OAuth Setup Required for {Path(path).name} ---")
                    print(f"Please visit: {dev_data['verification_uri_complete']}")
                    print(f"And enter code: {dev_data['user_code']}\n")
                    
                    token_data = None
                    start_time = time.time()
                    while time.time() - start_time < dev_data['expires_in']:
                        poll_response = await client.post(
                            TOKEN_ENDPOINT,
                            data={
                                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                                "device_code": dev_data['device_code'],
                                "client_id": CLIENT_ID,
                                "code_verifier": code_verifier
                            }
                        )
                        if poll_response.status_code == 200:
                            token_data = poll_response.json()
                            break
                        await asyncio.sleep(dev_data['interval'])
                    
                    if not token_data:
                        raise TimeoutError("Qwen device flow timed out.")
                    
                    creds.update({
                        "access_token": token_data["access_token"],
                        "refresh_token": token_data.get("refresh_token"),
                        "expiry_date": (time.time() + token_data["expires_in"]) * 1000,
                        "resource_url": token_data.get("resource_url")
                    })
                    await self._save_credentials(path, creds)
                    lib_logger.info(f"Qwen OAuth initialized successfully for '{path}'.")
                return creds
            return creds
        except Exception as e:
            raise ValueError(f"Failed to initialize Qwen OAuth: {e}")

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_path)
        return {"Authorization": f"Bearer {creds['access_token']}"}