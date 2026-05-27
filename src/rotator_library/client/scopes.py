# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Classifier-scoped credential/provider resolution and registered state."""

import asyncio
import hashlib
import hmac
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional


class ScopeManager:
    """Manage registered classifier state and resolve per-request scopes."""

    def __init__(
        self,
        *,
        all_credentials: Dict[str, List[str]],
        usage_base_path: Path,
        fingerprint_key: bytes,
        model_list_cache: Dict[str, List[str]],
        ensure_scoped_usage_manager: Callable[
            [str, Optional[str], Optional[List[str]]], Awaitable[str]
        ],
    ):
        self._all_credentials = all_credentials
        self._usage_base_path = usage_base_path
        self._fingerprint_key = fingerprint_key
        self._model_list_cache = model_list_cache
        self._ensure_scoped_usage_manager = ensure_scoped_usage_manager
        self._lock = asyncio.Lock()
        self._registered_scopes: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def safe_scope_name(classifier: str) -> str:
        """Convert an external classifier label into a safe directory name."""
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", classifier.strip())[:80]
        if cleaned and cleaned == classifier:
            return cleaned
        digest = hashlib.sha256(classifier.encode("utf-8")).hexdigest()[:12]
        return f"{cleaned or 'scope'}_{digest}"

    @staticmethod
    def normalize_provider_map(
        providers: Optional[Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        if not providers:
            return {}
        return {
            str(provider).lower(): dict(config or {})
            for provider, config in providers.items()
        }

    @staticmethod
    def normalize_api_key_map(api_keys: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
        if not api_keys:
            return {}
        normalized: Dict[str, List[str]] = {}
        for provider, keys in api_keys.items():
            if keys is None:
                continue
            if isinstance(keys, str):
                values = [keys]
            else:
                values = [str(key) for key in keys if key]
            if values:
                normalized[str(provider).lower()] = values
        return normalized

    def fingerprint_credential(
        self,
        provider: str,
        classifier: Optional[str],
        secret: str,
    ) -> str:
        scope = classifier or "default"
        message = f"{scope}:{provider}:{secret}".encode("utf-8")
        digest = hmac.new(self._fingerprint_key, message, hashlib.sha256).hexdigest()
        return f"private:{digest[:32]}"

    def scope_usage_key(self, provider: str, classifier: Optional[str]) -> str:
        if classifier is None:
            return provider
        return f"classifier:{self.safe_scope_name(classifier)}:{provider}"

    def scope_usage_file(self, provider: str, classifier: Optional[str]) -> Path:
        if classifier is None:
            return self._usage_base_path / f"usage_{provider}.json"
        return (
            self._usage_base_path
            / "classifiers"
            / self.safe_scope_name(classifier)
            / f"usage_{provider}.json"
        )

    async def get_registered_scope(self, classifier: str) -> Dict[str, Any]:
        async with self._lock:
            scope = self._registered_scopes.get(classifier, {})
            return {
                "providers": {
                    provider: dict(config)
                    for provider, config in scope.get("providers", {}).items()
                },
                "credentials": {
                    provider: {
                        "keys": list(entry.get("keys", [])),
                        "private": bool(entry.get("private", True)),
                    }
                    for provider, entry in scope.get("credentials", {}).items()
                },
            }

    async def resolve_scope_for_provider(
        self,
        provider: str,
        classifier: Optional[str],
        request_api_keys: Optional[Dict[str, Any]],
        request_providers: Optional[Dict[str, Dict[str, Any]]],
        private: bool,
    ) -> Dict[str, Any]:
        api_key_map = self.normalize_api_key_map(request_api_keys)
        provider_map = self.normalize_provider_map(request_providers)
        isolated = classifier is not None or bool(api_key_map) or bool(provider_map)

        if not isolated:
            return {
                "usage_manager_key": provider,
                "credentials": self._all_credentials.get(provider, []),
                "credential_secrets": {},
                "provider_config": None,
                "classifier": None,
            }

        registered = await self.get_registered_scope(classifier) if classifier else {}

        provider_config: Dict[str, Any] = {}
        if provider in registered.get("providers", {}):
            provider_config.update(registered["providers"][provider])
        if provider in provider_map:
            provider_config.update(provider_map[provider])

        raw_credentials: List[str] = []
        credential_private = private
        if provider in api_key_map:
            raw_credentials = api_key_map[provider]
        elif provider in registered.get("credentials", {}):
            credential_entry = registered["credentials"][provider]
            raw_credentials = list(credential_entry.get("keys", []))
            credential_private = bool(credential_entry.get("private", True))

        scope_name = classifier or "default"
        credentials: List[str] = []
        credential_secrets: Dict[str, str] = {}
        if credential_private:
            for secret in raw_credentials:
                credential_id = self.fingerprint_credential(provider, scope_name, secret)
                credentials.append(credential_id)
                credential_secrets[credential_id] = secret
        else:
            credentials = list(raw_credentials)

        usage_manager_key = await self._ensure_scoped_usage_manager(
            provider, scope_name, raw_credentials
        )
        return {
            "usage_manager_key": usage_manager_key,
            "credentials": credentials,
            "credential_secrets": credential_secrets,
            "provider_config": provider_config or None,
            "classifier": scope_name,
        }

    async def register_scope(
        self,
        classifier: str,
        providers: Optional[Dict[str, Dict[str, Any]]] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        private: bool = True,
    ) -> Dict[str, Any]:
        """Create or replace registered provider/credential state for a classifier."""
        async with self._lock:
            self._registered_scopes[classifier] = {"providers": {}, "credentials": {}}
        await self.update_scope(
            classifier, providers=providers, api_keys=api_keys, private=private
        )
        return await self.get_scope(classifier)

    async def update_scope(
        self,
        classifier: str,
        providers: Optional[Dict[str, Dict[str, Any]]] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        private: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Merge provider configs and credentials into a registered classifier."""
        async with self._lock:
            scope = self._registered_scopes.setdefault(
                classifier, {"providers": {}, "credentials": {}}
            )
            scope["providers"].update(self.normalize_provider_map(providers))
            for provider, keys in self.normalize_api_key_map(api_keys).items():
                scope["credentials"][provider] = {
                    "keys": list(keys),
                    "private": True if private is None else bool(private),
                }
        return await self.get_scope(classifier)

    async def get_scope(
        self,
        classifier: str,
        include_secrets: bool = False,
    ) -> Dict[str, Any]:
        """Fetch registered state for a classifier."""
        scope = await self.get_registered_scope(classifier)
        credentials = {}
        for provider, entry in scope.get("credentials", {}).items():
            credentials[provider] = {
                "count": len(entry.get("keys", [])),
                "private": bool(entry.get("private", True)),
            }
            if include_secrets:
                credentials[provider]["keys"] = list(entry.get("keys", []))
        return {
            "classifier": classifier,
            "providers": scope.get("providers", {}),
            "credentials": credentials,
        }

    async def remove_scope(self, classifier: str) -> None:
        """Remove registered state for a classifier."""
        async with self._lock:
            self._registered_scopes.pop(classifier, None)
            prefix = f"classifier:{self.safe_scope_name(classifier)}:"
            stale_keys = [
                key for key in self._model_list_cache if key.startswith(prefix)
            ]
            for key in stale_keys:
                self._model_list_cache.pop(key, None)

    async def add_scope_provider(
        self,
        classifier: str,
        provider: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self.update_scope(classifier, providers={provider: config})

    async def update_scope_provider(
        self,
        classifier: str,
        provider: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self.add_scope_provider(classifier, provider, config)

    async def remove_scope_provider(self, classifier: str, provider: str) -> None:
        provider = provider.lower()
        async with self._lock:
            scope = self._registered_scopes.get(classifier)
            if scope:
                scope.get("providers", {}).pop(provider, None)

    async def list_scope_providers(self, classifier: str) -> Dict[str, Dict[str, Any]]:
        return (await self.get_registered_scope(classifier)).get("providers", {})

    async def add_scope_credentials(
        self,
        classifier: str,
        provider: str,
        keys: Any,
        private: bool = True,
    ) -> Dict[str, Any]:
        provider = provider.lower()
        new_keys = self.normalize_api_key_map({provider: keys}).get(provider, [])
        async with self._lock:
            scope = self._registered_scopes.setdefault(
                classifier, {"providers": {}, "credentials": {}}
            )
            entry = scope["credentials"].setdefault(
                provider, {"keys": [], "private": private}
            )
            entry["keys"].extend(new_keys)
            entry["private"] = private
        return await self.get_scope(classifier)

    async def set_scope_credentials(
        self,
        classifier: str,
        provider: str,
        keys: Any,
        private: bool = True,
    ) -> Dict[str, Any]:
        return await self.update_scope(
            classifier,
            api_keys={provider: keys},
            private=private,
        )

    async def remove_scope_credentials(
        self,
        classifier: str,
        provider: str,
        credential_ids: Optional[List[str]] = None,
    ) -> None:
        provider = provider.lower()
        async with self._lock:
            scope = self._registered_scopes.get(classifier)
            if not scope:
                return
            if credential_ids is None:
                scope.get("credentials", {}).pop(provider, None)
                return
            entry = scope.get("credentials", {}).get(provider)
            if not entry:
                return
            remove_ids = set(credential_ids)
            private = bool(entry.get("private", True))
            kept = []
            for key in entry.get("keys", []):
                credential_id = (
                    self.fingerprint_credential(provider, classifier, key)
                    if private
                    else key
                )
                if credential_id not in remove_ids:
                    kept.append(key)
            entry["keys"] = kept

    async def list_scope_credentials(
        self,
        classifier: str,
        provider: Optional[str] = None,
        include_secrets: bool = False,
    ) -> Dict[str, Any]:
        scope = await self.get_registered_scope(classifier)
        credentials = scope.get("credentials", {})
        if provider:
            credentials = {provider.lower(): credentials.get(provider.lower(), {})}
        result: Dict[str, Any] = {}
        for prov, entry in credentials.items():
            private = bool(entry.get("private", True))
            ids = [
                self.fingerprint_credential(prov, classifier, key) if private else key
                for key in entry.get("keys", [])
            ]
            result[prov] = {"credential_ids": ids, "private": private}
            if include_secrets:
                result[prov]["keys"] = list(entry.get("keys", []))
        return result
