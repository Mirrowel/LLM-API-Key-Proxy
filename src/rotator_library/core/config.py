# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Compatibility wrapper for legacy core configuration loading.

The active provider usage configuration lives in ``rotator_library.usage.config``.
Runtime code loads that module directly from ``UsageManager`` and
``client/usage_managers.py``. This module previously duplicated much of that
logic, including environment parsing for custom caps, which let the two loaders
drift and caused bugs such as numeric Gemini quota groups being parsed
differently.

Keep this file thin so older imports of ``rotator_library.core.config`` still
work while preserving one source of truth for config parsing.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from ..usage.config import (
    _parse_custom_cap_env_key,
    load_provider_usage_config,
)
from .types import (
    CustomCapConfig,
    FairCycleConfig,
    ProviderConfig,
    WindowConfig,
)


def _enum_value(value: Any) -> Any:
    """Return enum values as plain strings for legacy core dataclasses."""
    return value.value if hasattr(value, "value") else value


class ConfigLoader:
    """Legacy facade over the active usage configuration loader.

    This class is mostly unused internally; current runtime paths call
    ``load_provider_usage_config()`` directly. It remains for public imports and
    tests that expect the old ``ProviderConfig`` shape from ``core.types``.
    """

    def __init__(self, provider_plugins: Optional[Dict[str, type]] = None):
        """Initialize the wrapper with optional provider plugin classes."""
        self._plugins = provider_plugins or {}
        self._cache: Dict[str, ProviderConfig] = {}

    def load_provider_config(
        self,
        provider: str,
        force_reload: bool = False,
    ) -> ProviderConfig:
        """Load a provider config through the active usage config system."""
        if not force_reload and provider in self._cache:
            return self._cache[provider]

        usage_config = load_provider_usage_config(provider, self._plugins)
        config = self._to_core_config(usage_config)
        self._cache[provider] = config
        return config

    def load_all_provider_configs(
        self,
        providers: List[str],
    ) -> Dict[str, ProviderConfig]:
        """Load configurations for multiple providers."""
        return {provider: self.load_provider_config(provider) for provider in providers}

    def clear_cache(self, provider: Optional[str] = None) -> None:
        """Clear cached compatibility configs."""
        if provider:
            self._cache.pop(provider, None)
        else:
            self._cache.clear()

    def _parse_tier_model_from_env(
        self,
        remainder: str,
    ) -> Tuple[Optional[Union[int, Tuple[int, ...], str]], Optional[str]]:
        """Delegate legacy custom-cap env parsing to the active parser."""
        return _parse_custom_cap_env_key(remainder)

    def _to_core_config(self, usage_config: Any) -> ProviderConfig:
        """Convert ``ProviderUsageConfig`` into the legacy ``ProviderConfig``."""
        return ProviderConfig(
            rotation_mode=_enum_value(usage_config.rotation_mode),
            rotation_tolerance=usage_config.rotation_tolerance,
            priority_multipliers=dict(usage_config.priority_multipliers),
            priority_multipliers_by_mode={
                mode: dict(values)
                for mode, values in usage_config.priority_multipliers_by_mode.items()
            },
            sequential_fallback_multiplier=usage_config.sequential_fallback_multiplier,
            fair_cycle=FairCycleConfig(
                enabled=usage_config.fair_cycle.enabled,
                tracking_mode=_enum_value(usage_config.fair_cycle.tracking_mode),
                cross_tier=usage_config.fair_cycle.cross_tier,
                duration=usage_config.fair_cycle.duration,
            ),
            custom_caps=[self._to_core_custom_cap(cap) for cap in usage_config.custom_caps],
            exhaustion_cooldown_threshold=usage_config.exhaustion_cooldown_threshold,
            windows=[self._to_core_window(window) for window in usage_config.windows],
        )

    def _to_core_custom_cap(self, cap: Any) -> CustomCapConfig:
        """Convert an active custom cap into the legacy core dataclass."""
        return CustomCapConfig(
            tier_key=cap.tier_key,
            model_or_group=cap.model_or_group,
            max_requests=cap.max_requests,
            max_requests_mode=_enum_value(cap.max_requests_mode),
            cooldown_mode=_enum_value(cap.cooldown_mode),
            cooldown_value=cap.cooldown_value,
        )

    def _to_core_window(self, window: Any) -> WindowConfig:
        """Convert an active window definition into the legacy core dataclass."""
        return WindowConfig(
            name=window.name,
            duration_seconds=window.duration_seconds,
            reset_mode=_enum_value(window.reset_mode),
            applies_to=window.applies_to,
        )


_global_loader: Optional[ConfigLoader] = None


def get_config_loader(
    provider_plugins: Optional[Dict[str, type]] = None,
) -> ConfigLoader:
    """Get the global compatibility config loader."""
    global _global_loader

    if provider_plugins is not None:
        _global_loader = ConfigLoader(provider_plugins)
    elif _global_loader is None:
        _global_loader = ConfigLoader()

    return _global_loader


def load_provider_config(
    provider: str,
    provider_plugins: Optional[Dict[str, type]] = None,
) -> ProviderConfig:
    """Load a provider's legacy core config through the wrapper."""
    loader = get_config_loader(provider_plugins)
    return loader.load_provider_config(provider)
