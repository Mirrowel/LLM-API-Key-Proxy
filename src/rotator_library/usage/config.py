# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Default configurations for the usage tracking package.

This module contains default values and configuration loading
for usage tracking, limits, and credential selection.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.constants import (
    DEFAULT_FAIR_CYCLE_DURATION,
    DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD,
    DEFAULT_ROTATION_TOLERANCE,
    DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER,
)
from .types import ResetMode, RotationMode, TrackingMode, CooldownMode


# =============================================================================
# WINDOW CONFIGURATION
# =============================================================================


@dataclass
class WindowDefinition:
    """
    Definition of a usage tracking window.

    Used to configure how usage is tracked and when it resets.
    """

    name: str  # e.g., "5h", "daily", "weekly"
    duration_seconds: Optional[int]  # None for infinite/total
    reset_mode: ResetMode
    is_primary: bool = False  # Primary window used for rotation decisions
    applies_to: str = "model"  # "credential", "model", "group"

    @classmethod
    def rolling(
        cls,
        name: str,
        duration_seconds: int,
        is_primary: bool = False,
        applies_to: str = "model",
    ) -> "WindowDefinition":
        """Create a rolling window definition."""
        return cls(
            name=name,
            duration_seconds=duration_seconds,
            reset_mode=ResetMode.ROLLING,
            is_primary=is_primary,
            applies_to=applies_to,
        )

    @classmethod
    def daily(
        cls,
        name: str = "daily",
        applies_to: str = "model",
    ) -> "WindowDefinition":
        """Create a daily fixed window definition."""
        return cls(
            name=name,
            duration_seconds=86400,
            reset_mode=ResetMode.FIXED_DAILY,
            applies_to=applies_to,
        )

    @classmethod
    def total(
        cls,
        name: str = "total",
        applies_to: str = "model",
    ) -> "WindowDefinition":
        """Create a total/infinite window definition."""
        return cls(
            name=name,
            duration_seconds=None,
            reset_mode=ResetMode.ROLLING,
            applies_to=applies_to,
        )


# =============================================================================
# FAIR CYCLE CONFIGURATION
# =============================================================================


@dataclass
class FairCycleConfig:
    """
    Fair cycle rotation configuration.

    Controls how credentials are cycled to ensure fair usage distribution.
    """

    enabled: Optional[bool] = (
        None  # None = derive from rotation mode (on for sequential)
    )
    tracking_mode: TrackingMode = TrackingMode.MODEL_GROUP
    cross_tier: bool = False  # Track across all tiers
    duration: int = DEFAULT_FAIR_CYCLE_DURATION  # Cycle duration in seconds


# =============================================================================
# CUSTOM CAP CONFIGURATION
# =============================================================================


@dataclass
class CustomCapConfig:
    """
    Custom cap configuration for a tier/model combination.

    Allows setting usage limits more restrictive than actual API limits.
    """

    tier_key: str  # Priority as string or "default"
    model_or_group: str  # Model name or quota group name
    max_requests: int  # Maximum requests allowed
    cooldown_mode: CooldownMode = CooldownMode.QUOTA_RESET
    cooldown_value: int = 0  # Seconds for offset/fixed modes

    @classmethod
    def from_dict(
        cls, tier_key: str, model_or_group: str, config: Dict[str, Any]
    ) -> "CustomCapConfig":
        """Create from dictionary config."""
        max_requests = config.get("max_requests", 0)

        # Handle percentage strings like "80%"
        if isinstance(max_requests, str) and max_requests.endswith("%"):
            # Store as negative to indicate percentage
            # Will be resolved later when actual limit is known
            max_requests = -int(max_requests.rstrip("%"))

        return cls(
            tier_key=tier_key,
            model_or_group=model_or_group,
            max_requests=max_requests,
            cooldown_mode=CooldownMode(config.get("cooldown_mode", "quota_reset")),
            cooldown_value=config.get("cooldown_value", 0),
        )


# =============================================================================
# PROVIDER USAGE CONFIG
# =============================================================================


@dataclass
class ProviderUsageConfig:
    """
    Complete usage configuration for a provider.

    Combines all settings needed for usage tracking and credential selection.
    """

    # Rotation settings
    rotation_mode: RotationMode = RotationMode.BALANCED
    rotation_tolerance: float = DEFAULT_ROTATION_TOLERANCE
    sequential_fallback_multiplier: int = DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER

    # Priority multipliers (priority -> max concurrent)
    priority_multipliers: Dict[int, int] = field(default_factory=dict)
    priority_multipliers_by_mode: Dict[str, Dict[int, int]] = field(
        default_factory=dict
    )

    # Fair cycle
    fair_cycle: FairCycleConfig = field(default_factory=FairCycleConfig)

    # Custom caps
    custom_caps: List[CustomCapConfig] = field(default_factory=list)

    # Exhaustion threshold (cooldown must exceed this to count as "exhausted")
    exhaustion_cooldown_threshold: int = DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD

    # Window definitions
    windows: List[WindowDefinition] = field(default_factory=list)

    def get_effective_multiplier(self, priority: int) -> int:
        """
        Get the effective multiplier for a priority level.

        Checks mode-specific overrides first, then universal multipliers,
        then falls back to sequential_fallback_multiplier.
        """
        mode_key = self.rotation_mode.value
        mode_multipliers = self.priority_multipliers_by_mode.get(mode_key, {})

        # Check mode-specific first
        if priority in mode_multipliers:
            return mode_multipliers[priority]

        # Check universal
        if priority in self.priority_multipliers:
            return self.priority_multipliers[priority]

        # Fall back
        return self.sequential_fallback_multiplier


# =============================================================================
# DEFAULT WINDOWS
# =============================================================================


def get_default_windows() -> List[WindowDefinition]:
    """
    Get default window definitions.

    Most providers use a 5-hour rolling window as primary.
    """
    return [
        WindowDefinition.rolling("5h", 18000, is_primary=True, applies_to="model"),
        WindowDefinition.daily("daily", applies_to="model"),
        WindowDefinition.total("total", applies_to="model"),
    ]


# =============================================================================
# CONFIG LOADER INTEGRATION
# =============================================================================


def load_provider_usage_config(
    provider: str,
    provider_plugins: Dict[str, Any],
) -> ProviderUsageConfig:
    """
    Load usage configuration for a provider.

    Merges:
    1. System defaults
    2. Provider class attributes
    3. Environment variables (always win)

    Args:
        provider: Provider name (e.g., "gemini", "openai")
        provider_plugins: Dict of provider plugin classes

    Returns:
        Complete configuration for the provider
    """
    import os

    config = ProviderUsageConfig()

    # Get plugin class
    plugin_class = provider_plugins.get(provider)

    # Apply provider defaults
    if plugin_class:
        # Rotation mode
        if hasattr(plugin_class, "default_rotation_mode"):
            config.rotation_mode = RotationMode(plugin_class.default_rotation_mode)

        # Priority multipliers
        if hasattr(plugin_class, "default_priority_multipliers"):
            config.priority_multipliers = dict(
                plugin_class.default_priority_multipliers
            )

        if hasattr(plugin_class, "default_priority_multipliers_by_mode"):
            config.priority_multipliers_by_mode = {
                k: dict(v)
                for k, v in plugin_class.default_priority_multipliers_by_mode.items()
            }

        # Sequential fallback multiplier
        if hasattr(plugin_class, "default_sequential_fallback_multiplier"):
            fallback = plugin_class.default_sequential_fallback_multiplier
            if fallback is not None:
                config.sequential_fallback_multiplier = fallback

        # Fair cycle
        if hasattr(plugin_class, "default_fair_cycle_config"):
            fc_config = plugin_class.default_fair_cycle_config
            config.fair_cycle = FairCycleConfig(
                enabled=fc_config.get("enabled"),
                tracking_mode=TrackingMode(
                    fc_config.get("tracking_mode", "model_group")
                ),
                cross_tier=fc_config.get("cross_tier", False),
                duration=fc_config.get("duration", DEFAULT_FAIR_CYCLE_DURATION),
            )
        else:
            if hasattr(plugin_class, "default_fair_cycle_enabled"):
                config.fair_cycle.enabled = plugin_class.default_fair_cycle_enabled
            if hasattr(plugin_class, "default_fair_cycle_tracking_mode"):
                config.fair_cycle.tracking_mode = TrackingMode(
                    plugin_class.default_fair_cycle_tracking_mode
                )
            if hasattr(plugin_class, "default_fair_cycle_cross_tier"):
                config.fair_cycle.cross_tier = (
                    plugin_class.default_fair_cycle_cross_tier
                )
            if hasattr(plugin_class, "default_fair_cycle_duration"):
                config.fair_cycle.duration = plugin_class.default_fair_cycle_duration

        # Custom caps
        if hasattr(plugin_class, "default_custom_caps"):
            for tier_key, models in plugin_class.default_custom_caps.items():
                tier_keys: Tuple[Union[int, str], ...]
                if isinstance(tier_key, tuple):
                    tier_keys = tuple(tier_key)
                else:
                    tier_keys = (tier_key,)
                for model_or_group, cap_config in models.items():
                    for resolved_tier in tier_keys:
                        config.custom_caps.append(
                            CustomCapConfig.from_dict(
                                str(resolved_tier), model_or_group, cap_config
                            )
                        )

        # Windows
        if hasattr(plugin_class, "usage_window_definitions"):
            config.windows = []
            for wdef in plugin_class.usage_window_definitions:
                config.windows.append(
                    WindowDefinition(
                        name=wdef.get("name", "default"),
                        duration_seconds=wdef.get("duration_seconds"),
                        reset_mode=ResetMode(wdef.get("reset_mode", "rolling")),
                        is_primary=wdef.get("is_primary", False),
                        applies_to=wdef.get("applies_to", "model"),
                    )
                )

    # Use default windows if none defined
    if not config.windows:
        config.windows = get_default_windows()

    # Apply environment variable overrides
    provider_upper = provider.upper()

    # Rotation mode from env
    env_mode = os.getenv(f"ROTATION_MODE_{provider_upper}")
    if env_mode:
        config.rotation_mode = RotationMode(env_mode.lower())

    # Sequential fallback multiplier
    env_fallback = os.getenv(f"SEQUENTIAL_FALLBACK_MULTIPLIER_{provider_upper}")
    if env_fallback:
        try:
            config.sequential_fallback_multiplier = int(env_fallback)
        except ValueError:
            pass

    # Fair cycle enabled from env
    env_fc = os.getenv(f"FAIR_CYCLE_{provider_upper}")
    if env_fc is None:
        env_fc = os.getenv(f"FAIR_CYCLE_ENABLED_{provider_upper}")
    if env_fc:
        config.fair_cycle.enabled = env_fc.lower() in ("true", "1", "yes")

    # Fair cycle tracking mode
    env_fc_mode = os.getenv(f"FAIR_CYCLE_TRACKING_MODE_{provider_upper}")
    if env_fc_mode:
        try:
            config.fair_cycle.tracking_mode = TrackingMode(env_fc_mode.lower())
        except ValueError:
            pass

    # Fair cycle cross-tier
    env_fc_cross = os.getenv(f"FAIR_CYCLE_CROSS_TIER_{provider_upper}")
    if env_fc_cross:
        config.fair_cycle.cross_tier = env_fc_cross.lower() in ("true", "1", "yes")

    # Fair cycle duration from env
    env_fc_duration = os.getenv(f"FAIR_CYCLE_DURATION_{provider_upper}")
    if env_fc_duration:
        try:
            config.fair_cycle.duration = int(env_fc_duration)
        except ValueError:
            pass

    # Exhaustion threshold from env
    env_threshold = os.getenv(f"EXHAUSTION_COOLDOWN_THRESHOLD_{provider_upper}")
    if env_threshold:
        try:
            config.exhaustion_cooldown_threshold = int(env_threshold)
        except ValueError:
            pass

    # Priority multipliers from env
    # Format: CONCURRENCY_MULTIPLIER_{PROVIDER}_PRIORITY_{N}=value
    # Format: CONCURRENCY_MULTIPLIER_{PROVIDER}_PRIORITY_{N}_{MODE}=value
    for key, value in os.environ.items():
        prefix = f"CONCURRENCY_MULTIPLIER_{provider_upper}_PRIORITY_"
        if key.startswith(prefix):
            try:
                remainder = key[len(prefix) :]
                multiplier = int(value)
                if multiplier < 1:
                    continue
                if "_" in remainder:
                    priority_str, mode = remainder.rsplit("_", 1)
                    priority = int(priority_str)
                    mode = mode.lower()
                    if mode in ("sequential", "balanced"):
                        config.priority_multipliers_by_mode.setdefault(mode, {})[
                            priority
                        ] = multiplier
                    else:
                        config.priority_multipliers[priority] = multiplier
                else:
                    priority = int(remainder)
                    config.priority_multipliers[priority] = multiplier
            except ValueError:
                pass

    # Custom caps from env
    if os.environ:
        cap_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for cap in config.custom_caps:
            cap_entry = cap_map.setdefault(str(cap.tier_key), {})
            cap_entry[cap.model_or_group] = {
                "max_requests": cap.max_requests,
                "cooldown_mode": cap.cooldown_mode.value,
                "cooldown_value": cap.cooldown_value,
            }

        cap_prefix = f"CUSTOM_CAP_{provider_upper}_T"
        cooldown_prefix = f"CUSTOM_CAP_COOLDOWN_{provider_upper}_T"
        for env_key, env_value in os.environ.items():
            if env_key.startswith(cap_prefix) and not env_key.startswith(
                cooldown_prefix
            ):
                remainder = env_key[len(cap_prefix) :]
                tier_key, model_key = _parse_custom_cap_env_key(remainder)
                if tier_key is None or not model_key:
                    continue
                cap_entry = cap_map.setdefault(str(tier_key), {})
                cap_entry.setdefault(model_key, {})["max_requests"] = env_value
            elif env_key.startswith(cooldown_prefix):
                remainder = env_key[len(cooldown_prefix) :]
                tier_key, model_key = _parse_custom_cap_env_key(remainder)
                if tier_key is None or not model_key:
                    continue
                if ":" in env_value:
                    mode, value_str = env_value.split(":", 1)
                    try:
                        value = int(value_str)
                    except ValueError:
                        continue
                else:
                    mode = env_value
                    value = 0
                cap_entry = cap_map.setdefault(str(tier_key), {})
                cap_entry.setdefault(model_key, {})["cooldown_mode"] = mode
                cap_entry.setdefault(model_key, {})["cooldown_value"] = value

        config.custom_caps = []
        for tier_key, models in cap_map.items():
            for model_or_group, cap_config in models.items():
                config.custom_caps.append(
                    CustomCapConfig.from_dict(tier_key, model_or_group, cap_config)
                )

    # Derive fair cycle enabled from rotation mode if not explicitly set
    if config.fair_cycle.enabled is None:
        config.fair_cycle.enabled = config.rotation_mode == RotationMode.SEQUENTIAL

    return config


def _parse_custom_cap_env_key(
    remainder: str,
) -> Tuple[Optional[Union[int, Tuple[int, ...], str]], Optional[str]]:
    """Parse the tier and model/group from a custom cap env var remainder."""
    if not remainder:
        return None, None

    remaining_parts = remainder.split("_")
    if len(remaining_parts) < 2:
        return None, None

    tier_key: Union[int, Tuple[int, ...], str, None] = None
    model_key: Optional[str] = None
    tier_parts: List[int] = []

    for i, part in enumerate(remaining_parts):
        if part == "DEFAULT":
            tier_key = "default"
            model_key = "_".join(remaining_parts[i + 1 :])
            break
        if part.isdigit():
            tier_parts.append(int(part))
            continue

        if not tier_parts:
            return None, None
        if len(tier_parts) == 1:
            tier_key = tier_parts[0]
        else:
            tier_key = tuple(tier_parts)
        model_key = "_".join(remaining_parts[i:])
        break
    else:
        return None, None

    if model_key:
        model_key = model_key.lower().replace("_", "-")

    return tier_key, model_key
