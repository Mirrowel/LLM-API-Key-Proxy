# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Scoped and default provider model discovery."""

import asyncio
import fnmatch
import hashlib
import json
import logging
import random
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from ..core.errors import mask_credential

lib_logger = logging.getLogger("rotator_library")


class ModelDiscoveryService:
    """Fetch, filter, and cache available models for default and scoped pools."""

    def __init__(
        self,
        *,
        all_credentials: Dict[str, List[str]],
        model_list_cache: Dict[str, List[str]],
        normalize_provider_map: Callable[
            [Optional[Dict[str, Dict[str, Any]]]], Dict[str, Dict[str, Any]]
        ],
        normalize_api_key_map: Callable[[Optional[Dict[str, Any]]], Dict[str, List[str]]],
        scope_usage_key: Callable[[str, Optional[str]], str],
        get_registered_scope: Callable[[str], Awaitable[Dict[str, Any]]],
        resolve_scope_for_provider: Callable[
            [str, Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Dict[str, Any]]], bool],
            Awaitable[Dict[str, Any]],
        ],
        get_provider_instance: Callable[[str], Optional[Any]],
        get_http_client: Callable[[], Any],
        provider_config: Any,
        model_resolver: Any,
    ):
        self._all_credentials = all_credentials
        self._model_list_cache = model_list_cache
        self._normalize_provider_map = normalize_provider_map
        self._normalize_api_key_map = normalize_api_key_map
        self._scope_usage_key = scope_usage_key
        self._get_registered_scope = get_registered_scope
        self._resolve_scope_for_provider = resolve_scope_for_provider
        self._get_provider_instance = get_provider_instance
        self._get_http_client = get_http_client
        self._provider_config = provider_config
        self._model_resolver = model_resolver

    def model_cache_key(
        self,
        provider: str,
        classifier: Optional[str],
        provider_config: Optional[Dict[str, Any]],
        model_filters: Optional[Dict[str, Any]],
        credentials: Optional[List[str]] = None,
    ) -> str:
        if classifier is None and not provider_config and not model_filters:
            return provider
        credential_fingerprint = hashlib.sha256(
            json.dumps(sorted(credentials or []), sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        payload = json.dumps(
            {
                "provider_config": provider_config or {},
                "model_filters": model_filters or {},
                "credentials": credential_fingerprint,
            },
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"{self._scope_usage_key(provider, classifier)}:{digest}"

    @staticmethod
    def filter_values(
        model_filters: Optional[Dict[str, Any]],
        provider: str,
        names: tuple[str, ...],
    ) -> List[str]:
        if not model_filters:
            return []
        provider_filters = model_filters.get(provider, model_filters)
        if not isinstance(provider_filters, dict):
            return []
        for name in names:
            value = provider_filters.get(name)
            if value:
                return list(value) if not isinstance(value, str) else [value]
        return []

    def is_scoped_model_allowed(
        self,
        model: str,
        provider: str,
        classifier: Optional[str],
        model_filters: Optional[Dict[str, Any]],
    ) -> bool:
        if classifier is None and not model_filters:
            return self._model_resolver.is_model_allowed(model, provider)

        whitelist = self.filter_values(
            model_filters, provider, ("whitelist", "whitelist_models", "allow")
        )
        blacklist = self.filter_values(
            model_filters, provider, ("blacklist", "ignore", "ignore_models", "deny")
        )
        model_name = model.split("/", 1)[1] if "/" in model else model

        if whitelist:
            return any(
                fnmatch.fnmatch(model, pattern) or fnmatch.fnmatch(model_name, pattern)
                for pattern in whitelist
            )
        if blacklist:
            return not any(
                fnmatch.fnmatch(model, pattern) or fnmatch.fnmatch(model_name, pattern)
                for pattern in blacklist
            )
        return True

    async def get_openai_compatible_models(
        self,
        provider: str,
        api_key: str,
        provider_config: Optional[Dict[str, Any]],
    ) -> List[str]:
        api_base = None
        if provider_config:
            api_base = provider_config.get("base_url") or provider_config.get("api_base")
        if not api_base:
            api_base = self._provider_config.get_api_base(provider)
        if not api_base:
            return []

        response = await self._get_http_client().get(
            f"{str(api_base).rstrip('/')}/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        return [
            f"{provider}/{model['id']}"
            for model in response.json().get("data", [])
            if isinstance(model, dict) and model.get("id")
        ]

    async def get_available_models(
        self,
        provider: str,
        classifier: Optional[str] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        providers: Optional[Dict[str, Dict[str, Any]]] = None,
        private: bool = False,
        model_filters: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> List[str]:
        provider = provider.lower()
        if not provider:
            return []
        scope = await self._resolve_scope_for_provider(
            provider,
            classifier,
            api_keys,
            providers,
            private,
        )
        credentials = scope["credentials"]
        if not credentials:
            return []

        cache_key = self.model_cache_key(
            provider,
            scope["classifier"],
            scope["provider_config"],
            model_filters,
            credentials,
        )
        if not force_refresh and cache_key in self._model_list_cache:
            return self._model_list_cache[cache_key]

        shuffled = list(credentials)
        random.shuffle(shuffled)
        plugin = self._get_provider_instance(provider)

        for cred in shuffled:
            api_key = scope["credential_secrets"].get(cred, cred)
            try:
                has_base_override = bool(
                    scope["provider_config"]
                    and (
                        scope["provider_config"].get("base_url")
                        or scope["provider_config"].get("api_base")
                    )
                )
                if has_base_override or not plugin:
                    models = await self.get_openai_compatible_models(
                        provider, api_key, scope["provider_config"]
                    )
                else:
                    models = await plugin.get_models(api_key, self._get_http_client())

                final = [
                    model
                    for model in models
                    if self.is_scoped_model_allowed(
                        model, provider, scope["classifier"], model_filters
                    )
                ]
                self._model_list_cache[cache_key] = final
                return final

            except Exception as e:
                lib_logger.debug(
                    f"Failed to get models for {provider} with {mask_credential(cred)}: {e}"
                )
                continue

        return []

    async def get_all_available_models(
        self,
        grouped: bool = True,
        classifier: Optional[str] = None,
        api_keys: Optional[Dict[str, Any]] = None,
        providers: Optional[Dict[str, Dict[str, Any]]] = None,
        private: bool = False,
        model_filters: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> Union[Dict[str, List[str]], List[str]]:
        provider_configs = self._normalize_provider_map(providers)
        request_keys = self._normalize_api_key_map(api_keys)
        if classifier is None and not request_keys and not provider_configs:
            provider_names = list(self._all_credentials.keys())
        else:
            registered = await self._get_registered_scope(classifier) if classifier else {}
            provider_names = sorted(
                set(request_keys)
                | set(provider_configs)
                | set(registered.get("credentials", {}))
            )

        tasks = [
            self.get_available_models(
                provider,
                classifier=classifier,
                api_keys=api_keys,
                providers=providers,
                private=private,
                model_filters=model_filters,
                force_refresh=force_refresh,
            )
            for provider in provider_names
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_models: Dict[str, List[str]] = {}
        for provider, result in zip(provider_names, results):
            if isinstance(result, Exception):
                lib_logger.error(f"Failed to get models for {provider}: {result}")
                all_models[provider] = []
            else:
                all_models[provider] = result

        if grouped:
            return all_models
        flat = []
        for models in all_models.values():
            flat.extend(models)
        return flat
