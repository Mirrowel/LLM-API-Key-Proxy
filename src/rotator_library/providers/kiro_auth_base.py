# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx

from .utilities.kiro_utils import (
    TOKEN_REFRESH_THRESHOLD,
    get_kiro_refresh_url,
    get_kiro_api_host,
    get_kiro_q_host,
    get_aws_sso_oidc_url,
    get_machine_fingerprint,
)


lib_logger = logging.getLogger("rotator_library")


SQLITE_TOKEN_KEYS = [
    "kirocli:social:token",
    "kirocli:odic:token",
    "codewhisperer:odic:token",
]

SQLITE_REGISTRATION_KEYS = [
    "kirocli:odic:device-registration",
    "codewhisperer:odic:device-registration",
]


class AuthType(Enum):
    KIRO_DESKTOP = "kiro_desktop"
    AWS_SSO_OIDC = "aws_sso_oidc"


class KiroAuthManager:
    def __init__(
        self,
        refresh_token: Optional[str] = None,
        profile_arn: Optional[str] = None,
        region: str = "us-east-1",
        sqlite_db: Optional[str] = None,
        json_creds_file: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> None:
        self._refresh_token = refresh_token
        self._profile_arn = profile_arn
        self._region = region
        self._sqlite_db = sqlite_db
        self._json_creds_file = json_creds_file
        self._client_id = client_id
        self._client_secret = client_secret
        self._sso_region: Optional[str] = None
        self._scopes: Optional[list] = None
        self._client_id_hash: Optional[str] = None

        self._sqlite_token_key: Optional[str] = None
        self._access_token: Optional[str] = None
        self._expires_at: Optional[datetime] = None
        self._lock = asyncio.Lock()

        self._auth_type: AuthType = AuthType.KIRO_DESKTOP
        self._refresh_url = get_kiro_refresh_url(region)
        self._api_host = get_kiro_api_host(region)
        self._q_host = get_kiro_q_host(region)
        self._fingerprint = get_machine_fingerprint()

        # Load credentials from JSON file first (if provided), then SQLite as fallback
        if json_creds_file:
            self._load_credentials_from_json_file(json_creds_file)
        elif sqlite_db:
            self._load_credentials_from_sqlite(sqlite_db)

        self._detect_auth_type()

    @property
    def auth_type(self) -> AuthType:
        return self._auth_type

    @property
    def profile_arn(self) -> Optional[str]:
        return self._profile_arn

    @property
    def api_host(self) -> str:
        return self._api_host

    @property
    def q_host(self) -> str:
        return self._q_host

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _detect_auth_type(self) -> None:
        if self._client_id and self._client_secret:
            self._auth_type = AuthType.AWS_SSO_OIDC
        else:
            self._auth_type = AuthType.KIRO_DESKTOP

    def _load_credentials_from_json_file(self, file_path: str) -> None:
        """Load credentials from a JSON file (e.g., kiro-auth-token.json)."""
        try:
            path = Path(file_path).expanduser()
            if not path.exists():
                lib_logger.warning(f"Credentials file not found: {file_path}")
                return

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load tokens
            if "refreshToken" in data:
                self._refresh_token = data["refreshToken"]
            if "accessToken" in data:
                self._access_token = data["accessToken"]
            if "profileArn" in data:
                self._profile_arn = data["profileArn"]
            if "region" in data:
                self._region = data["region"]
                self._sso_region = data["region"]
                # Update URLs for new region
                self._refresh_url = get_kiro_refresh_url(self._region)
                self._api_host = get_kiro_api_host(self._region)
                self._q_host = get_kiro_q_host(self._region)
                lib_logger.debug(f"Region from JSON file: {self._region}")

            # Load clientIdHash for Enterprise Kiro IDE
            if "clientIdHash" in data:
                self._client_id_hash = data["clientIdHash"]
                self._load_enterprise_device_registration(self._client_id_hash, path.parent)

            # Load AWS SSO OIDC fields if directly in credentials file
            if "clientId" in data:
                self._client_id = data["clientId"]
            if "clientSecret" in data:
                self._client_secret = data["clientSecret"]

            # Parse expiresAt
            if "expiresAt" in data:
                try:
                    expires_str = data["expiresAt"]
                    if expires_str.endswith("Z"):
                        self._expires_at = datetime.fromisoformat(
                            expires_str.replace("Z", "+00:00")
                        )
                    else:
                        self._expires_at = datetime.fromisoformat(expires_str)
                except Exception as exc:
                    lib_logger.warning(f"Failed to parse expiresAt: {exc}")

            lib_logger.info(f"Credentials loaded from JSON file: {file_path}")

        except json.JSONDecodeError as exc:
            lib_logger.error(f"JSON decode error in credentials file: {exc}")
        except Exception as exc:
            lib_logger.error(f"Error loading credentials from JSON file: {exc}")

    def _load_enterprise_device_registration(
        self, client_id_hash: str, cache_dir: Path
    ) -> None:
        """
        Load clientId and clientSecret from Enterprise Kiro IDE device registration file.
        Enterprise Kiro IDE uses AWS SSO OIDC. Device registration is stored at:
        ~/.aws/sso/cache/{clientIdHash}.json
        """
        try:
            registration_file = cache_dir / f"{client_id_hash}.json"
            if not registration_file.exists():
                lib_logger.debug(
                    f"Device registration file not found: {registration_file}"
                )
                return

            with open(registration_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "clientId" in data:
                self._client_id = data["clientId"]
            if "clientSecret" in data:
                self._client_secret = data["clientSecret"]
            if "region" in data and not self._sso_region:
                self._sso_region = data["region"]

            lib_logger.info(
                f"Loaded device registration from: {registration_file.name}"
            )
        except Exception as exc:
            lib_logger.warning(f"Failed to load device registration: {exc}")

    def _save_credentials_to_json_file(self) -> None:
        """Save updated credentials back to the JSON file."""
        if not self._json_creds_file:
            return

        try:
            path = Path(self._json_creds_file).expanduser()
            if not path.exists():
                lib_logger.warning(f"JSON file not found for writing: {self._json_creds_file}")
                return

            # Read existing data to preserve other fields
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Update token fields
            if self._access_token:
                data["accessToken"] = self._access_token
            if self._refresh_token:
                data["refreshToken"] = self._refresh_token
            if self._expires_at:
                data["expiresAt"] = self._expires_at.isoformat()

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            lib_logger.debug(f"Saved credentials to JSON file: {self._json_creds_file}")
        except Exception as exc:
            lib_logger.error(f"Error saving credentials to JSON file: {exc}")

    def _load_credentials_from_sqlite(self, db_path: str) -> None:
        try:
            path = Path(db_path).expanduser()
            if not path.exists():
                lib_logger.warning(f"SQLite database not found: {db_path}")
                return

            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()

            token_row = None
            for key in SQLITE_TOKEN_KEYS:
                cursor.execute("SELECT value FROM auth_kv WHERE key = ?", (key,))
                token_row = cursor.fetchone()
                if token_row:
                    self._sqlite_token_key = key
                    break

            if token_row:
                token_data = json.loads(token_row[0])
                if token_data:
                    self._access_token = token_data.get("access_token") or token_data.get(
                        "accessToken"
                    )
                    self._refresh_token = token_data.get(
                        "refresh_token"
                    ) or token_data.get("refreshToken")
                    self._profile_arn = token_data.get("profile_arn") or token_data.get(
                        "profileArn"
                    )
                    if "region" in token_data:
                        self._sso_region = token_data.get("region")
                    self._scopes = token_data.get("scopes")

                    expires_str = token_data.get("expires_at") or token_data.get(
                        "expiresAt"
                    )
                    if expires_str:
                        try:
                            if expires_str.endswith("Z"):
                                expires_str = expires_str.replace("Z", "+00:00")
                            self._expires_at = datetime.fromisoformat(expires_str)
                        except Exception as exc:
                            lib_logger.warning(
                                f"Failed to parse expires_at from SQLite: {exc}"
                            )

            registration_row = None
            for key in SQLITE_REGISTRATION_KEYS:
                cursor.execute("SELECT value FROM auth_kv WHERE key = ?", (key,))
                registration_row = cursor.fetchone()
                if registration_row:
                    break

            if registration_row:
                registration_data = json.loads(registration_row[0])
                if registration_data:
                    self._client_id = registration_data.get("client_id") or registration_data.get(
                        "clientId"
                    )
                    self._client_secret = registration_data.get(
                        "client_secret"
                    ) or registration_data.get("clientSecret")
                    if not self._sso_region:
                        self._sso_region = registration_data.get("region")

            conn.close()
        except sqlite3.Error as exc:
            lib_logger.error(f"SQLite error loading credentials: {exc}")
        except json.JSONDecodeError as exc:
            lib_logger.error(f"JSON decode error in SQLite data: {exc}")
        except Exception as exc:
            lib_logger.error(f"Error loading credentials from SQLite: {exc}")

    def _save_credentials_to_sqlite(self) -> None:
        if not self._sqlite_db:
            return

        try:
            path = Path(self._sqlite_db).expanduser()
            if not path.exists():
                lib_logger.warning(
                    f"SQLite database not found for writing: {self._sqlite_db}"
                )
                return

            conn = sqlite3.connect(str(path), timeout=5.0)
            cursor = conn.cursor()

            token_data = {
                "access_token": self._access_token,
                "refresh_token": self._refresh_token,
                "expires_at": self._expires_at.isoformat() if self._expires_at else None,
                "region": self._sso_region or self._region,
            }
            if self._scopes:
                token_data["scopes"] = self._scopes

            token_json = json.dumps(token_data)
            if self._sqlite_token_key:
                cursor.execute(
                    "UPDATE auth_kv SET value = ? WHERE key = ?",
                    (token_json, self._sqlite_token_key),
                )
                if cursor.rowcount > 0:
                    conn.commit()
                    conn.close()
                    return

            for key in SQLITE_TOKEN_KEYS:
                cursor.execute(
                    "UPDATE auth_kv SET value = ? WHERE key = ?",
                    (token_json, key),
                )
                if cursor.rowcount > 0:
                    conn.commit()
                    conn.close()
                    return

            conn.close()
            lib_logger.warning("Failed to save credentials to SQLite: no matching keys")
        except sqlite3.Error as exc:
            lib_logger.error(f"SQLite error saving credentials: {exc}")
        except Exception as exc:
            lib_logger.error(f"Error saving credentials to SQLite: {exc}")

    def is_token_expiring_soon(self) -> bool:
        if not self._expires_at:
            return True
        now = datetime.now(timezone.utc)
        threshold = now.timestamp() + TOKEN_REFRESH_THRESHOLD
        return self._expires_at.timestamp() <= threshold

    def is_token_expired(self) -> bool:
        if not self._expires_at:
            return True
        return datetime.now(timezone.utc) >= self._expires_at

    async def _refresh_token_kiro_desktop(self) -> None:
        if not self._refresh_token:
            raise ValueError("Refresh token is not set")

        payload = {"refreshToken": self._refresh_token}
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"KiroIDE-0.7.45-{self._fingerprint}",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(self._refresh_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        new_access_token = data.get("accessToken")
        new_refresh_token = data.get("refreshToken")
        expires_in = data.get("expiresIn", 3600)
        new_profile_arn = data.get("profileArn")

        if not new_access_token:
            raise ValueError(f"Refresh response missing accessToken: {data}")

        self._access_token = new_access_token
        if new_refresh_token:
            self._refresh_token = new_refresh_token
        if new_profile_arn:
            self._profile_arn = new_profile_arn

        self._expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=max(60, expires_in - 60)
        )
        self._save_credentials()

    def _save_credentials(self) -> None:
        """Save credentials to the appropriate storage (JSON file or SQLite)."""
        if self._json_creds_file:
            self._save_credentials_to_json_file()
        elif self._sqlite_db:
            self._save_credentials_to_sqlite()

    async def _do_aws_sso_oidc_refresh(self) -> None:
        if not self._refresh_token:
            raise ValueError("Refresh token is not set")
        if not self._client_id:
            raise ValueError("Client ID is not set (required for AWS SSO OIDC)")
        if not self._client_secret:
            raise ValueError("Client secret is not set (required for AWS SSO OIDC)")

        sso_region = self._sso_region or self._region
        url = get_aws_sso_oidc_url(sso_region)
        payload = {
            "grantType": "refresh_token",
            "clientId": self._client_id,
            "clientSecret": self._client_secret,
            "refreshToken": self._refresh_token,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                lib_logger.error(
                    f"AWS SSO OIDC refresh failed: status={response.status_code}, body={response.text}"
                )
                response.raise_for_status()
            result = response.json()

        new_access_token = result.get("accessToken")
        new_refresh_token = result.get("refreshToken")
        expires_in = result.get("expiresIn", 3600)
        if not new_access_token:
            raise ValueError(f"AWS SSO OIDC response missing accessToken: {result}")

        self._access_token = new_access_token
        if new_refresh_token:
            self._refresh_token = new_refresh_token
        self._expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=max(60, expires_in - 60)
        )

        self._save_credentials()

    async def _refresh_token_aws_sso_oidc(self) -> None:
        try:
            await self._do_aws_sso_oidc_refresh()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400:
                # Reload credentials and retry
                if self._json_creds_file:
                    lib_logger.warning(
                        "AWS SSO refresh failed with 400. Reloading JSON file and retrying."
                    )
                    self._load_credentials_from_json_file(self._json_creds_file)
                    await self._do_aws_sso_oidc_refresh()
                elif self._sqlite_db:
                    lib_logger.warning(
                        "AWS SSO refresh failed with 400. Reloading SQLite and retrying."
                    )
                    self._load_credentials_from_sqlite(self._sqlite_db)
                    await self._do_aws_sso_oidc_refresh()
                else:
                    raise
            else:
                raise

    async def _refresh_token_request(self) -> None:
        if self._auth_type == AuthType.AWS_SSO_OIDC:
            await self._refresh_token_aws_sso_oidc()
        else:
            await self._refresh_token_kiro_desktop()

    async def get_access_token(self) -> str:
        async with self._lock:
            if self._access_token and not self.is_token_expiring_soon():
                return self._access_token

            # Try reloading from file in case another process refreshed the token
            if self.is_token_expiring_soon():
                if self._json_creds_file:
                    self._load_credentials_from_json_file(self._json_creds_file)
                elif self._sqlite_db:
                    self._load_credentials_from_sqlite(self._sqlite_db)
                if self._access_token and not self.is_token_expiring_soon():
                    return self._access_token

            try:
                await self._refresh_token_request()
            except httpx.HTTPStatusError as exc:
                if self._access_token and not self.is_token_expired():
                    lib_logger.warning(
                        f"Refresh failed but access token still valid: {exc}"
                    )
                    return self._access_token
                raise

            if not self._access_token:
                raise ValueError("Failed to obtain access token")
            return self._access_token

    async def force_refresh(self) -> None:
        async with self._lock:
            await self._refresh_token_request()
