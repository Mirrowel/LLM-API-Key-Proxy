# src/rotator_library/providers/gemini_auth_base.py

import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

import httpx

lib_logger = logging.getLogger('rotator_library')

CLIENT_ID = "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
CLIENT_SECRET = "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"
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
                with open(path, 'r') as f:
                    creds = json.load(f)
                # Handle gcloud-style creds file which nest tokens under "credential"
                if "credential" in creds:
                    creds = creds["credential"]
                self._credentials_cache[path] = creds
                return creds
            except Exception as e:
                raise IOError(f"Failed to load Gemini OAuth credentials from '{path}': {e}")

    async def _save_credentials(self, path: str, creds: Dict[str, Any]):
        self._credentials_cache[path] = creds
        try:
            with open(path, 'w') as f:
                json.dump(creds, f, indent=2)
        except Exception as e:
            lib_logger.error(f"Failed to save updated Gemini OAuth credentials to '{path}': {e}")

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        expiry = creds.get("token_expiry") # gcloud format
        if not expiry: # gemini-cli format
             expiry_timestamp = creds.get("expiry_date", 0) / 1000
        else:
            expiry_timestamp = time.mktime(time.strptime(expiry, "%Y-%m-%dT%H:%M:%SZ"))

        return expiry_timestamp < time.time() + REFRESH_EXPIRY_BUFFER_SECONDS

    async def _refresh_token(self, path: str, creds: Dict[str, Any]) -> Dict[str, Any]:
        async with self._get_lock(path):
            if not self._is_token_expired(self._credentials_cache.get(path, creds)):
                return self._credentials_cache.get(path, creds)

            lib_logger.info(f"Refreshing Gemini OAuth token for '{Path(path).name}'...")
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

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            creds = await self._refresh_token(credential_path, creds)
        return {"Authorization": f"Bearer {creds['access_token']}"}

    async def proactively_refresh(self, credential_path: str):
        creds = await self._load_credentials(credential_path)
        if self._is_token_expired(creds):
            await self._refresh_token(credential_path, creds)

    def _get_lock(self, path: str) -> asyncio.Lock:
        if path not in self._refresh_locks:
            self._refresh_locks[path] = asyncio.Lock()
        return self._refresh_locks[path]