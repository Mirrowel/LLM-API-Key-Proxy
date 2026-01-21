# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Default configurations for the usage tracking package.

This module contains default values and configuration loading
for usage tracking, limits, and credential selection.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

    @classmethod
    def rolling(
        cls, name: str, duration_seconds: int, is_primary: bool = False
    ) -> "WindowDefinition":
        """Create a rolling window definition."""
        return cls(
            name=name,
            duration_seconds=duration_seconds,
            reset_mode=ResetMode.ROLLING,
            is_primary=is_primary,
        )

    @classmethod
    def daily(cls, name: str = "daily") -> "WindowDefinition":
        """Create a daily fixed window definition."""
        return cls(
            name=name,
            duration_seconds=86400,
            reset_mode=ResetMode.FIXED_DAILY,
        )

    @classmethod
    def total(cls, name: str = "total") -> "WindowDefinition":
        """Create a total/infinite window definition."""
        return cls(
            name=name,
            duration_seconds=None,
            reset_mode=ResetMode.ROLLING,
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
        WindowDefinition.rolling("5h", 18000, is_primary=True),  # 5 hours
        WindowDefinition.daily("daily"),
        WindowDefinition.total("total"),
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

        # Custom caps
        if hasattr(plugin_class, "default_custom_caps"):
            for tier_key, models in plugin_class.default_custom_caps.items():
                for model_or_group, cap_config in models.items():
                    config.custom_caps.append(
                        CustomCapConfig.from_dict(
                            str(tier_key), model_or_group, cap_config
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

    # Fair cycle enabled from env
    env_fc = os.getenv(f"FAIR_CYCLE_ENABLED_{provider_upper}")
    if env_fc:
        config.fair_cycle.enabled = env_fc.lower() in ("true", "1", "yes")

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
    for key, value in os.environ.items():
        prefix = f"CONCURRENCY_MULTIPLIER_{provider_upper}_PRIORITY_"
        if key.startswith(prefix):
            try:
                priority = int(key[len(prefix) :])
                config.priority_multipliers[priority] = int(value)
            except ValueError:
                pass

    # Derive fair cycle enabled from rotation mode if not explicitly set
    if config.fair_cycle.enabled is None:
        config.fair_cycle.enabled = config.rotation_mode == RotationMode.SEQUENTIAL

    return config
