# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""UsageManager lifecycle and factory helpers for RotatingClient."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..usage import UsageManager as NewUsageManager
from ..usage.config import WindowDefinition, load_provider_usage_config

lib_logger = logging.getLogger("rotator_library")


class UsageManagerRegistry:
    """Own global and classifier-scoped UsageManager instances."""

    def __init__(
        self,
        *,
        all_credentials: Dict[str, List[str]],
        usage_base_path: Path,
        provider_plugins: Dict[str, Any],
        max_concurrent_requests_per_key: Dict[str, int],
        max_concurrent_requests_per_key_by_mode: Dict[str, Dict[str, int]],
        optimal_concurrent_requests_per_key: Dict[str, int],
        optimal_concurrent_requests_per_key_by_mode: Dict[str, Dict[str, int]],
        rotation_tolerance: float,
        get_provider_instance: Callable[[str], Optional[Any]],
        scope_usage_key: Callable[[str, Optional[str]], str],
        scope_usage_file: Callable[[str, Optional[str]], Path],
    ):
        self._all_credentials = all_credentials
        self._usage_base_path = usage_base_path
        self._provider_plugins = provider_plugins
        self._max_concurrent_requests_per_key = max_concurrent_requests_per_key
        self._max_concurrent_requests_per_key_by_mode = (
            max_concurrent_requests_per_key_by_mode
        )
        self._optimal_concurrent_requests_per_key = optimal_concurrent_requests_per_key
        self._optimal_concurrent_requests_per_key_by_mode = (
            optimal_concurrent_requests_per_key_by_mode
        )
        self._rotation_tolerance = rotation_tolerance
        self._get_provider_instance = get_provider_instance
        self._scope_usage_key = scope_usage_key
        self._scope_usage_file = scope_usage_file

        self.managers: Dict[str, NewUsageManager] = {}
        self._usage_initialized = False
        self._usage_init_lock = asyncio.Lock()
        self._scoped_lock = asyncio.Lock()

    def create_global_managers(self) -> None:
        """Create one global/default UsageManager per configured provider."""
        for provider, credentials in self._all_credentials.items():
            config = load_provider_usage_config(provider, self._provider_plugins)
            config.rotation_tolerance = self._rotation_tolerance
            self.apply_usage_reset_config(provider, credentials, config)

            usage_file = self._usage_base_path / f"usage_{provider}.json"
            mode = config.rotation_mode.value
            max_concurrent, optimal_concurrent = self.get_concurrency_settings(
                provider, mode
            )

            self.managers[provider] = NewUsageManager(
                provider=provider,
                file_path=usage_file,
                provider_plugins=self._provider_plugins,
                config=config,
                max_concurrent_per_key=max_concurrent,
                optimal_concurrent_per_key=optimal_concurrent,
            )

    async def initialize_usage_managers(self) -> None:
        """Initialize global/default managers once before background jobs run."""
        if self._usage_initialized:
            return
        async with self._usage_init_lock:
            if self._usage_initialized:
                return
            for provider, manager in self.managers.items():
                if provider.startswith("classifier:"):
                    continue
                credentials = self._all_credentials.get(provider, [])
                priorities, tiers = self.get_credential_metadata(provider, credentials)
                await manager.initialize(
                    credentials, priorities=priorities, tiers=tiers
                )

            summaries = []
            for provider, manager in self.managers.items():
                if provider.startswith("classifier:"):
                    continue
                credentials = self._all_credentials.get(provider, [])
                status = (
                    f"loaded {manager.loaded_credentials}"
                    if manager.loaded_from_storage
                    else "fresh"
                )
                summaries.append(f"{provider}:{len(credentials)} ({status})")
            if summaries:
                lib_logger.info(
                    f"Usage managers initialized: {', '.join(sorted(summaries))}"
                )
            self._usage_initialized = True

    async def ensure_scoped_usage_manager(
        self,
        provider: str,
        classifier: Optional[str],
        credentials: Optional[List[str]] = None,
    ) -> str:
        """Create/get a classifier-scoped UsageManager."""
        usage_key = self._scope_usage_key(provider, classifier)
        if usage_key in self.managers:
            return usage_key

        async with self._scoped_lock:
            if usage_key in self.managers:
                return usage_key

            config = load_provider_usage_config(provider, self._provider_plugins)
            config.rotation_tolerance = self._rotation_tolerance
            self.apply_usage_reset_config(provider, credentials or [], config)
            mode = config.rotation_mode.value
            max_concurrent, optimal_concurrent = self.get_concurrency_settings(
                provider, mode
            )
            usage_file = self._scope_usage_file(provider, classifier)
            usage_file.parent.mkdir(parents=True, exist_ok=True)
            self.managers[usage_key] = NewUsageManager(
                provider=provider,
                file_path=usage_file,
                provider_plugins=self._provider_plugins,
                config=config,
                max_concurrent_per_key=max_concurrent,
                optimal_concurrent_per_key=optimal_concurrent,
            )
        return usage_key

    def get_concurrency_settings(
        self, provider: str, mode: str
    ) -> tuple[Optional[int], Optional[int]]:
        max_concurrent = self._max_concurrent_requests_per_key_by_mode.get(
            provider, {}
        ).get(mode)
        if max_concurrent is None:
            max_concurrent = self._max_concurrent_requests_per_key.get(provider)

        optimal_concurrent = self._optimal_concurrent_requests_per_key_by_mode.get(
            provider, {}
        ).get(mode)
        if optimal_concurrent is None:
            optimal_concurrent = self._optimal_concurrent_requests_per_key.get(provider)
        return max_concurrent, optimal_concurrent

    def get_credential_metadata(
        self,
        provider: str,
        credentials: List[str],
    ) -> tuple[Dict[str, int], Dict[str, str]]:
        """Resolve priority and tier metadata for credentials."""
        plugin = self._get_provider_instance(provider)
        priorities: Dict[str, int] = {}
        tiers: Dict[str, str] = {}

        if not plugin:
            return priorities, tiers

        for credential in credentials:
            if hasattr(plugin, "get_credential_priority"):
                priority = plugin.get_credential_priority(credential)
                if priority is not None:
                    priorities[credential] = priority
            if hasattr(plugin, "get_credential_tier_name"):
                tier_name = plugin.get_credential_tier_name(credential)
                if tier_name:
                    tiers[credential] = tier_name

        return priorities, tiers

    def apply_usage_reset_config(
        self,
        provider: str,
        credentials: List[str],
        config: Any,
    ) -> None:
        """Apply provider-specific usage reset config to window definitions."""
        if not credentials:
            return

        plugin = self._get_provider_instance(provider)
        if not plugin or not hasattr(plugin, "get_usage_reset_config"):
            return

        try:
            reset_config = plugin.get_usage_reset_config(credentials[0])
        except Exception as exc:
            lib_logger.debug(f"Failed to load usage reset config for {provider}: {exc}")
            return

        if not reset_config:
            return

        window_seconds = reset_config.get("window_seconds")
        if not window_seconds:
            return

        mode = reset_config.get("mode", "credential")
        applies_to = "credential" if mode == "credential" else "model"

        if window_seconds == 86400:
            window_name = "daily"
        elif window_seconds % 3600 == 0:
            window_name = f"{window_seconds // 3600}h"
        else:
            window_name = "window"

        config.windows = [
            WindowDefinition.rolling(
                name=window_name,
                duration_seconds=int(window_seconds),
                is_primary=True,
                applies_to=applies_to,
            ),
        ]

    def get_usage_manager(self, provider: str) -> Optional[NewUsageManager]:
        return self.managers.get(provider)
