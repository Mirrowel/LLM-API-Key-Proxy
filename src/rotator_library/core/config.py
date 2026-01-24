# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Centralized configuration loader for the rotator library.

This module provides a ConfigLoader class that handles all configuration
parsing from:
1. System defaults (from config/defaults.py)
2. Provider class attributes
3. Environment variables (ALWAYS override provider defaults)

The ConfigLoader ensures consistent configuration handling across
both the client and usage manager.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from .types import (
    ProviderConfig,
    FairCycleConfig,
    CustomCapConfig,
    WindowConfig,
)
from .constants import (
    # Defaults
    DEFAULT_ROTATION_MODE,
    DEFAULT_ROTATION_TOLERANCE,
    DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER,
    DEFAULT_FAIR_CYCLE_ENABLED,
    DEFAULT_FAIR_CYCLE_TRACKING_MODE,
    DEFAULT_FAIR_CYCLE_CROSS_TIER,
    DEFAULT_FAIR_CYCLE_DURATION,
    DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD,
    # Prefixes
    ENV_PREFIX_ROTATION_MODE,
    ENV_PREFIX_FAIR_CYCLE,
    ENV_PREFIX_FAIR_CYCLE_TRACKING,
    ENV_PREFIX_FAIR_CYCLE_CROSS_TIER,
    ENV_PREFIX_FAIR_CYCLE_DURATION,
    ENV_PREFIX_EXHAUSTION_THRESHOLD,
    ENV_PREFIX_CONCURRENCY_MULTIPLIER,
    ENV_PREFIX_CUSTOM_CAP,
    ENV_PREFIX_CUSTOM_CAP_COOLDOWN,
)

lib_logger = logging.getLogger("rotator_library")


class ConfigLoader:
    """
    Centralized configuration loader.

    Parses all configuration from:
    1. System defaults
    2. Provider class attributes
    3. Environment variables (ALWAYS override provider defaults)

    Usage:
        loader = ConfigLoader(provider_plugins)
        config = loader.load_provider_config("antigravity")
    """

    def __init__(self, provider_plugins: Optional[Dict[str, type]] = None):
        """
        Initialize the ConfigLoader.

        Args:
            provider_plugins: Dict mapping provider names to plugin classes.
                              If None, no provider-specific defaults are used.
        """
        self._plugins = provider_plugins or {}
        self._cache: Dict[str, ProviderConfig] = {}

    def load_provider_config(
        self,
        provider: str,
        force_reload: bool = False,
    ) -> ProviderConfig:
        """
        Load complete configuration for a provider.

        Configuration is loaded in this order (later overrides earlier):
        1. System defaults
        2. Provider class attributes
        3. Environment variables (ALWAYS win)

        Args:
            provider: Provider name (e.g., "antigravity", "gemini_cli")
            force_reload: If True, bypass cache and reload

        Returns:
            Complete ProviderConfig for the provider
        """
        if not force_reload and provider in self._cache:
            return self._cache[provider]

        # Start with system defaults
        config = self._get_system_defaults()

        # Apply provider class defaults
        plugin_class = self._plugins.get(provider)
        if plugin_class:
            config = self._apply_provider_defaults(config, plugin_class, provider)

        # Apply environment variable overrides (ALWAYS win)
        config = self._apply_env_overrides(config, provider)

        # Cache and return
        self._cache[provider] = config
        return config

    def load_all_provider_configs(
        self,
        providers: List[str],
    ) -> Dict[str, ProviderConfig]:
        """
        Load configurations for multiple providers.

        Args:
            providers: List of provider names

        Returns:
            Dict mapping provider names to their configs
        """
        return {p: self.load_provider_config(p) for p in providers}

    def clear_cache(self, provider: Optional[str] = None) -> None:
        """
        Clear cached configurations.

        Args:
            provider: If provided, only clear that provider's cache.
                     If None, clear all cached configs.
        """
        if provider:
            self._cache.pop(provider, None)
        else:
            self._cache.clear()

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _get_system_defaults(self) -> ProviderConfig:
        """Get a ProviderConfig with all system defaults."""
        return ProviderConfig(
            rotation_mode=DEFAULT_ROTATION_MODE,
            rotation_tolerance=DEFAULT_ROTATION_TOLERANCE,
            priority_multipliers={},
            priority_multipliers_by_mode={},
            sequential_fallback_multiplier=DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER,
            fair_cycle=FairCycleConfig(
                enabled=DEFAULT_FAIR_CYCLE_ENABLED,
                tracking_mode=DEFAULT_FAIR_CYCLE_TRACKING_MODE,
                cross_tier=DEFAULT_FAIR_CYCLE_CROSS_TIER,
                duration=DEFAULT_FAIR_CYCLE_DURATION,
            ),
            custom_caps=[],
            exhaustion_cooldown_threshold=DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD,
            windows=[],
        )

    def _apply_provider_defaults(
        self,
        config: ProviderConfig,
        plugin_class: type,
        provider: str,
    ) -> ProviderConfig:
        """
        Apply provider class default attributes to config.

        Args:
            config: Current configuration
            plugin_class: Provider plugin class
            provider: Provider name for logging

        Returns:
            Updated configuration
        """
        # Rotation mode
        if hasattr(plugin_class, "default_rotation_mode"):
            config.rotation_mode = plugin_class.default_rotation_mode

        # Priority multipliers
        if hasattr(plugin_class, "default_priority_multipliers"):
            multipliers = plugin_class.default_priority_multipliers
            if multipliers:
                config.priority_multipliers = dict(multipliers)

        # Sequential fallback multiplier
        if hasattr(plugin_class, "default_sequential_fallback_multiplier"):
            fallback = plugin_class.default_sequential_fallback_multiplier
            if fallback != DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER:
                config.sequential_fallback_multiplier = fallback

        # Fair cycle settings
        if hasattr(plugin_class, "default_fair_cycle_enabled"):
            val = plugin_class.default_fair_cycle_enabled
            if val is not None:
                config.fair_cycle.enabled = val

        if hasattr(plugin_class, "default_fair_cycle_tracking_mode"):
            config.fair_cycle.tracking_mode = (
                plugin_class.default_fair_cycle_tracking_mode
            )

        if hasattr(plugin_class, "default_fair_cycle_cross_tier"):
            config.fair_cycle.cross_tier = plugin_class.default_fair_cycle_cross_tier

        if hasattr(plugin_class, "default_fair_cycle_duration"):
            duration = plugin_class.default_fair_cycle_duration
            if duration != DEFAULT_FAIR_CYCLE_DURATION:
                config.fair_cycle.duration = duration

        # Exhaustion cooldown threshold
        if hasattr(plugin_class, "default_exhaustion_cooldown_threshold"):
            threshold = plugin_class.default_exhaustion_cooldown_threshold
            if threshold != DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD:
                config.exhaustion_cooldown_threshold = threshold

        # Custom caps
        if hasattr(plugin_class, "default_custom_caps"):
            caps = plugin_class.default_custom_caps
            if caps:
                config.custom_caps = self._parse_custom_caps_from_provider(caps)

        return config

    def _apply_env_overrides(
        self,
        config: ProviderConfig,
        provider: str,
    ) -> ProviderConfig:
        """
        Apply environment variable overrides to config.

        Environment variables ALWAYS override provider class defaults.

        Args:
            config: Current configuration
            provider: Provider name

        Returns:
            Updated configuration with env overrides applied
        """
        provider_upper = provider.upper()

        # Rotation mode: ROTATION_MODE_{PROVIDER}
        env_key = f"{ENV_PREFIX_ROTATION_MODE}{provider_upper}"
        env_val = os.getenv(env_key)
        if env_val:
            config.rotation_mode = env_val.lower()
            if config.rotation_mode not in ("balanced", "sequential"):
                lib_logger.warning(f"Invalid {env_key}='{env_val}'. Using 'balanced'.")
                config.rotation_mode = "balanced"

        # Fair cycle enabled: FAIR_CYCLE_{PROVIDER}
        env_key = f"{ENV_PREFIX_FAIR_CYCLE}{provider_upper}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            config.fair_cycle.enabled = env_val.lower() in ("true", "1", "yes")

        # Fair cycle tracking mode: FAIR_CYCLE_TRACKING_MODE_{PROVIDER}
        env_key = f"{ENV_PREFIX_FAIR_CYCLE_TRACKING}{provider_upper}"
        env_val = os.getenv(env_key)
        if env_val and env_val.lower() in ("model_group", "credential"):
            config.fair_cycle.tracking_mode = env_val.lower()

        # Fair cycle cross-tier: FAIR_CYCLE_CROSS_TIER_{PROVIDER}
        env_key = f"{ENV_PREFIX_FAIR_CYCLE_CROSS_TIER}{provider_upper}"
        env_val = os.getenv(env_key)
        if env_val is not None:
            config.fair_cycle.cross_tier = env_val.lower() in ("true", "1", "yes")

        # Fair cycle duration: FAIR_CYCLE_DURATION_{PROVIDER}
        env_key = f"{ENV_PREFIX_FAIR_CYCLE_DURATION}{provider_upper}"
        env_val = os.getenv(env_key)
        if env_val:
            try:
                config.fair_cycle.duration = int(env_val)
            except ValueError:
                lib_logger.warning(f"Invalid {env_key}='{env_val}'. Must be integer.")

        # Exhaustion cooldown threshold: EXHAUSTION_COOLDOWN_THRESHOLD_{PROVIDER}
        # Also check global: EXHAUSTION_COOLDOWN_THRESHOLD
        env_key = f"{ENV_PREFIX_EXHAUSTION_THRESHOLD}{provider_upper}"
        env_val = os.getenv(env_key) or os.getenv("EXHAUSTION_COOLDOWN_THRESHOLD")
        if env_val:
            try:
                config.exhaustion_cooldown_threshold = int(env_val)
            except ValueError:
                lib_logger.warning(f"Invalid exhaustion threshold='{env_val}'.")

        # Priority multipliers: CONCURRENCY_MULTIPLIER_{PROVIDER}_PRIORITY_{N}
        # Also supports mode-specific: CONCURRENCY_MULTIPLIER_{PROVIDER}_PRIORITY_{N}_{MODE}
        self._parse_priority_multiplier_env_vars(config, provider_upper)

        # Custom caps: CUSTOM_CAP_{PROVIDER}_T{TIER}_{MODEL}
        # Also: CUSTOM_CAP_COOLDOWN_{PROVIDER}_T{TIER}_{MODEL}
        self._parse_custom_cap_env_vars(config, provider_upper)

        return config

    def _parse_priority_multiplier_env_vars(
        self,
        config: ProviderConfig,
        provider_upper: str,
    ) -> None:
        """
        Parse CONCURRENCY_MULTIPLIER_* environment variables.

        Formats:
        - CONCURRENCY_MULTIPLIER_{PROVIDER}_PRIORITY_{N}=value
        - CONCURRENCY_MULTIPLIER_{PROVIDER}_PRIORITY_{N}_{MODE}=value
        """
        prefix = f"{ENV_PREFIX_CONCURRENCY_MULTIPLIER}{provider_upper}_PRIORITY_"

        for env_key, env_val in os.environ.items():
            if not env_key.startswith(prefix):
                continue

            remainder = env_key[len(prefix) :]
            try:
                multiplier = int(env_val)
                if multiplier < 1:
                    lib_logger.warning(f"Invalid {env_key}='{env_val}'. Must be >= 1.")
                    continue

                # Check for mode-specific suffix
                if "_" in remainder:
                    parts = remainder.rsplit("_", 1)
                    priority = int(parts[0])
                    mode = parts[1].lower()

                    if mode in ("sequential", "balanced"):
                        if mode not in config.priority_multipliers_by_mode:
                            config.priority_multipliers_by_mode[mode] = {}
                        config.priority_multipliers_by_mode[mode][priority] = multiplier
                    else:
                        lib_logger.warning(f"Unknown mode in {env_key}: {mode}")
                else:
                    # Universal priority multiplier
                    priority = int(remainder)
                    config.priority_multipliers[priority] = multiplier

            except ValueError:
                lib_logger.warning(f"Invalid {env_key}='{env_val}'. Could not parse.")

    def _parse_custom_cap_env_vars(
        self,
        config: ProviderConfig,
        provider_upper: str,
    ) -> None:
        """
        Parse CUSTOM_CAP_* environment variables.

        Formats:
        - CUSTOM_CAP_{PROVIDER}_T{TIER}_{MODEL}=value
        - CUSTOM_CAP_{PROVIDER}_TDEFAULT_{MODEL}=value
        - CUSTOM_CAP_COOLDOWN_{PROVIDER}_T{TIER}_{MODEL}=mode:value
        """
        cap_prefix = f"{ENV_PREFIX_CUSTOM_CAP}{provider_upper}_T"
        cooldown_prefix = f"{ENV_PREFIX_CUSTOM_CAP_COOLDOWN}{provider_upper}_T"

        # Collect caps by (tier_key, model_key) to merge cap and cooldown
        caps_dict: Dict[Tuple[Any, str], Dict[str, Any]] = {}

        for env_key, env_val in os.environ.items():
            if env_key.startswith(cooldown_prefix):
                remainder = env_key[len(cooldown_prefix) :]
                tier_key, model_key = self._parse_tier_model_from_env(remainder)
                if tier_key is None:
                    continue

                # Parse mode:value format
                if ":" in env_val:
                    mode, value_str = env_val.split(":", 1)
                    try:
                        value = int(value_str)
                    except ValueError:
                        lib_logger.warning(f"Invalid cooldown in {env_key}")
                        continue
                else:
                    mode = env_val
                    value = 0

                key = (tier_key, model_key)
                if key not in caps_dict:
                    caps_dict[key] = {}
                caps_dict[key]["cooldown_mode"] = mode
                caps_dict[key]["cooldown_value"] = value

            elif env_key.startswith(cap_prefix):
                remainder = env_key[len(cap_prefix) :]
                tier_key, model_key = self._parse_tier_model_from_env(remainder)
                if tier_key is None:
                    continue

                key = (tier_key, model_key)
                if key not in caps_dict:
                    caps_dict[key] = {}
                caps_dict[key]["max_requests"] = env_val

        # Convert to CustomCapConfig objects
        for (tier_key, model_key), cap_data in caps_dict.items():
            if "max_requests" not in cap_data:
                continue  # Need at least max_requests

            cap = CustomCapConfig(
                tier_key=tier_key,
                model_or_group=model_key,
                max_requests=cap_data["max_requests"],
                cooldown_mode=cap_data.get("cooldown_mode", "quota_reset"),
                cooldown_value=cap_data.get("cooldown_value", 0),
            )
            config.custom_caps.append(cap)

    def _parse_tier_model_from_env(
        self,
        remainder: str,
    ) -> Tuple[Optional[Union[int, Tuple[int, ...], str]], Optional[str]]:
        """
        Parse tier and model/group from env var remainder.

        Args:
            remainder: String after "CUSTOM_CAP_{PROVIDER}_T" prefix
                       e.g., "2_CLAUDE" or "2_3_CLAUDE" or "DEFAULT_CLAUDE"

        Returns:
            (tier_key, model_key) or (None, None) if parse fails
        """
        if not remainder:
            return None, None

        parts = remainder.split("_")
        if len(parts) < 2:
            return None, None

        tier_parts: List[int] = []
        tier_key: Union[int, Tuple[int, ...], str, None] = None
        model_key: Optional[str] = None

        for i, part in enumerate(parts):
            if part == "DEFAULT":
                tier_key = "default"
                model_key = "_".join(parts[i + 1 :])
                break
            elif part.isdigit():
                tier_parts.append(int(part))
            else:
                # First non-numeric part is start of model name
                if len(tier_parts) == 0:
                    return None, None
                elif len(tier_parts) == 1:
                    tier_key = tier_parts[0]
                else:
                    tier_key = tuple(tier_parts)
                model_key = "_".join(parts[i:])
                break
        else:
            # All parts were tier parts, no model
            return None, None

        if model_key:
            # Convert to lowercase with dashes (standard model name format)
            model_key = model_key.lower().replace("_", "-")

        return tier_key, model_key

    def _parse_custom_caps_from_provider(
        self,
        caps: Dict[Union[int, Tuple[int, ...], str], Dict[str, Dict[str, Any]]],
    ) -> List[CustomCapConfig]:
        """
        Parse custom caps from provider class default_custom_caps attribute.

        Args:
            caps: Provider's default_custom_caps dict

        Returns:
            List of CustomCapConfig objects
        """
        result = []

        for tier_key, models_config in caps.items():
            for model_key, cap_data in models_config.items():
                cap = CustomCapConfig(
                    tier_key=tier_key,
                    model_or_group=model_key,
                    max_requests=cap_data.get("max_requests", 0),
                    cooldown_mode=cap_data.get("cooldown_mode", "quota_reset"),
                    cooldown_value=cap_data.get("cooldown_value", 0),
                )
                result.append(cap)

        return result


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

# Global loader instance (initialized lazily)
_global_loader: Optional[ConfigLoader] = None


def get_config_loader(
    provider_plugins: Optional[Dict[str, type]] = None,
) -> ConfigLoader:
    """
    Get the global ConfigLoader instance.

    Creates a new instance if none exists or if provider_plugins is provided.

    Args:
        provider_plugins: Optional dict of provider plugins. If provided,
                         creates a new loader with these plugins.

    Returns:
        The global ConfigLoader instance
    """
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
    """
    Convenience function to load a provider's configuration.

    Args:
        provider: Provider name
        provider_plugins: Optional provider plugins dict

    Returns:
        ProviderConfig for the provider
    """
    loader = get_config_loader(provider_plugins)
    return loader.load_provider_config(provider)
