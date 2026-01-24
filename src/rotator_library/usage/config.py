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
    DEFAULT_FAIR_CYCLE_QUOTA_THRESHOLD,
    DEFAULT_FAIR_CYCLE_RESET_COOLDOWN_THRESHOLD,
    DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD,
    DEFAULT_ROTATION_TOLERANCE,
    DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER,
)
from .types import ResetMode, RotationMode, TrackingMode, CooldownMode, CapMode


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
    quota_threshold: float = (
        DEFAULT_FAIR_CYCLE_QUOTA_THRESHOLD  # Multiplier of window limit for exhaustion
    )
    reset_cooldown_threshold: int = (
        DEFAULT_FAIR_CYCLE_RESET_COOLDOWN_THRESHOLD  # Min cooldown to count for reset
    )


# =============================================================================
# CUSTOM CAP CONFIGURATION
# =============================================================================


def _parse_duration_string(duration_str: str) -> Optional[int]:
    """
    Parse duration strings in various formats to total seconds.

    Handles:
    - Plain seconds (no unit): '300', '562476'
    - Simple durations: '3600s', '60m', '2h', '1d'
    - Compound durations: '2h30m', '1h30m45s', '2d1h30m'

    Args:
        duration_str: Duration string to parse

    Returns:
        Total seconds as integer, or None if parsing fails.
    """
    import re

    if not duration_str:
        return None

    remaining = duration_str.strip().lower()

    # Try parsing as plain number first (no units)
    try:
        return int(float(remaining))
    except ValueError:
        pass

    total_seconds = 0.0

    # Parse days component
    day_match = re.match(r"(\d+)d", remaining)
    if day_match:
        total_seconds += int(day_match.group(1)) * 86400
        remaining = remaining[day_match.end() :]

    # Parse hours component
    hour_match = re.match(r"(\d+)h", remaining)
    if hour_match:
        total_seconds += int(hour_match.group(1)) * 3600
        remaining = remaining[hour_match.end() :]

    # Parse minutes component - use negative lookahead to avoid matching 'ms'
    min_match = re.match(r"(\d+)m(?!s)", remaining)
    if min_match:
        total_seconds += int(min_match.group(1)) * 60
        remaining = remaining[min_match.end() :]

    # Parse seconds component (including decimals)
    sec_match = re.match(r"([\d.]+)s", remaining)
    if sec_match:
        total_seconds += float(sec_match.group(1))

    if total_seconds > 0:
        return int(total_seconds)
    return None


def _parse_cooldown_config(
    mode: Optional[str],
    value: Any,
) -> Tuple[CooldownMode, int]:
    """
    Parse cooldown configuration from config dict values.

    Supports comprehensive cooldown_value parsing:
    - Flat duration: 300, "300", "1h", "30m", "1h30m", "2d1h30m" → fixed seconds
    - Offset with sign: "+300", "+1h30m", "-300", "-5m" → offset from natural reset
    - Percentage: "+50%", "-20%" → percentage of window duration as offset
      (stored as negative value with special encoding: -1000 - percentage)
    - String "quota_reset" → use natural reset time

    The cooldown_mode is auto-detected from the value format if not explicitly set:
    - Starts with '+' or '-' → CooldownMode.OFFSET
    - Just a duration → CooldownMode.FIXED
    - "quota_reset" string → CooldownMode.QUOTA_RESET

    Args:
        mode: Explicit cooldown mode string, or None to auto-detect
        value: Cooldown value (int, str, or various formats)

    Returns:
        Tuple of (CooldownMode, cooldown_value in seconds)
    """
    # Handle explicit mode with simple value
    if mode is not None:
        try:
            cooldown_mode = CooldownMode(mode)
        except ValueError:
            cooldown_mode = CooldownMode.QUOTA_RESET

        # Parse value
        if isinstance(value, int):
            return cooldown_mode, value
        elif isinstance(value, str):
            parsed = _parse_duration_string(value.lstrip("+-"))
            return cooldown_mode, parsed or 0
        else:
            return cooldown_mode, 0

    # Auto-detect mode from value format
    if isinstance(value, int):
        if value == 0:
            return CooldownMode.QUOTA_RESET, 0
        return CooldownMode.FIXED, value

    if isinstance(value, str):
        value = value.strip()

        # Check for "quota_reset" string
        if value.lower() in ("quota_reset", "quota-reset", "quotareset"):
            return CooldownMode.QUOTA_RESET, 0

        # Check for percentage format: "+50%", "-20%"
        if value.endswith("%"):
            sign = 1
            val_str = value.rstrip("%")
            if val_str.startswith("+"):
                val_str = val_str[1:]
            elif val_str.startswith("-"):
                sign = -1
                val_str = val_str[1:]
            try:
                percentage = int(val_str)
                # Encode percentage as special value: -1000 - (sign * percentage)
                # This allows the custom_caps checker to detect and handle percentages
                # Range: -1001 to -1100 for +1% to +100%, -999 to -900 for -1% to -100%
                encoded = -1000 - (sign * percentage)
                return CooldownMode.OFFSET, encoded
            except ValueError:
                pass

        # Check for offset format: "+300", "+1h30m", "-5m"
        if value.startswith("+") or value.startswith("-"):
            sign = 1 if value.startswith("+") else -1
            duration_str = value[1:]
            parsed = _parse_duration_string(duration_str)
            if parsed is not None:
                return CooldownMode.OFFSET, sign * parsed
            return CooldownMode.OFFSET, 0

        # Plain duration: "300", "1h30m"
        parsed = _parse_duration_string(value)
        if parsed is not None:
            return CooldownMode.FIXED, parsed

    return CooldownMode.QUOTA_RESET, 0


import logging

_config_logger = logging.getLogger("rotator_library")


def _parse_max_requests(
    raw_value: Any, tier_key: str, model_or_group: str
) -> Optional[Tuple[int, "CapMode"]]:
    """
    Parse max_requests value and determine its mode.

    Formats supported:
    - 130 (int) → ABSOLUTE, 130
    - "130" → ABSOLUTE, 130
    - "-130" → OFFSET, -130 (means max - 130)
    - "+130" → OFFSET, +130 (means max + 130)
    - "80%" → PERCENTAGE, 80

    Returns:
        Tuple of (value, mode) or None if invalid (logs error).
    """
    # Handle None or missing
    if raw_value is None:
        _config_logger.error(
            f"Custom cap for tier={tier_key} model={model_or_group}: "
            "max_requests is None, skipping cap"
        )
        return None

    # Already an int
    if isinstance(raw_value, int):
        return (raw_value, CapMode.ABSOLUTE)

    # Float - convert to int
    if isinstance(raw_value, float):
        return (int(raw_value), CapMode.ABSOLUTE)

    # Must be a string from here
    if not isinstance(raw_value, str):
        _config_logger.error(
            f"Custom cap for tier={tier_key} model={model_or_group}: "
            f"max_requests has invalid type {type(raw_value).__name__}, skipping cap"
        )
        return None

    # Strip whitespace
    value_str = raw_value.strip()

    # Empty string is invalid
    if not value_str:
        _config_logger.error(
            f"Custom cap for tier={tier_key} model={model_or_group}: "
            "max_requests is empty string, skipping cap"
        )
        return None

    # Percentage format: "80%"
    if value_str.endswith("%"):
        try:
            percentage = int(value_str.rstrip("%"))
            if percentage < 0 or percentage > 100:
                _config_logger.error(
                    f"Custom cap for tier={tier_key} model={model_or_group}: "
                    f"percentage {percentage}% out of range (0-100), skipping cap"
                )
                return None
            return (percentage, CapMode.PERCENTAGE)
        except ValueError:
            _config_logger.error(
                f"Custom cap for tier={tier_key} model={model_or_group}: "
                f"invalid percentage '{value_str}', skipping cap"
            )
            return None

    # Offset format: "+130" or "-130"
    if value_str.startswith("+") or value_str.startswith("-"):
        try:
            offset = int(value_str)
            return (offset, CapMode.OFFSET)
        except ValueError:
            # Try float conversion
            try:
                offset = int(float(value_str))
                return (offset, CapMode.OFFSET)
            except ValueError:
                _config_logger.error(
                    f"Custom cap for tier={tier_key} model={model_or_group}: "
                    f"invalid offset '{value_str}', skipping cap"
                )
                return None

    # Absolute format: plain number string "130"
    try:
        value = int(value_str)
        return (value, CapMode.ABSOLUTE)
    except ValueError:
        # Try float conversion
        try:
            value = int(float(value_str))
            return (value, CapMode.ABSOLUTE)
        except ValueError:
            _config_logger.error(
                f"Custom cap for tier={tier_key} model={model_or_group}: "
                f"invalid value '{value_str}', skipping cap"
            )
            return None


@dataclass
class CustomCapConfig:
    """
    Custom cap configuration for a tier/model combination.

    Allows setting usage limits that can be absolute, offset from API limits,
    or percentage of API limits.
    """

    tier_key: str  # Priority as string or "default"
    model_or_group: str  # Model name or quota group name
    max_requests: int  # The numeric value
    max_requests_mode: CapMode = CapMode.ABSOLUTE  # How to interpret max_requests
    cooldown_mode: CooldownMode = CooldownMode.QUOTA_RESET
    cooldown_value: int = 0  # Seconds for offset/fixed modes

    @classmethod
    def from_dict(
        cls, tier_key: str, model_or_group: str, config: Dict[str, Any]
    ) -> Optional["CustomCapConfig"]:
        """
        Create from dictionary config.

        max_requests formats:
        - 130 or "130" → ABSOLUTE mode, exactly 130 requests
        - "-130" → OFFSET mode, max - 130 requests
        - "+130" → OFFSET mode, max + 130 requests
        - "80%" → PERCENTAGE mode, 80% of max requests

        cooldown_value formats:
        - Flat duration: 300, "300", "1h", "30m", "1h30m", "2d1h30m" → fixed seconds
        - Offset with sign: "+300", "+1h30m", "-300", "-5m" → offset from natural reset
        - String "quota_reset" → use natural reset time

        Returns:
            CustomCapConfig instance, or None if max_requests is invalid.
        """
        raw_max_requests = config.get("max_requests")

        # Check if mode is already explicitly provided (for round-trip serialization)
        explicit_mode = config.get("max_requests_mode")
        if explicit_mode is not None:
            # Mode was explicitly provided - use it directly
            try:
                if isinstance(explicit_mode, CapMode):
                    max_requests_mode = explicit_mode
                else:
                    max_requests_mode = CapMode(explicit_mode)
                # Still need to validate max_requests is a valid number
                if isinstance(raw_max_requests, int):
                    max_requests_value = raw_max_requests
                elif isinstance(raw_max_requests, float):
                    max_requests_value = int(raw_max_requests)
                elif isinstance(raw_max_requests, str):
                    try:
                        max_requests_value = int(
                            float(raw_max_requests.lstrip("+-").rstrip("%"))
                        )
                    except ValueError:
                        _config_logger.error(
                            f"Custom cap for tier={tier_key} model={model_or_group}: "
                            f"invalid max_requests value '{raw_max_requests}', skipping cap"
                        )
                        return None
                else:
                    max_requests_value = 0
            except ValueError:
                # Invalid mode string, fall through to parsing
                explicit_mode = None

        if explicit_mode is None:
            # Parse max_requests with mode detection
            parsed = _parse_max_requests(raw_max_requests, tier_key, model_or_group)
            if parsed is None:
                return None
            max_requests_value, max_requests_mode = parsed

        # Parse cooldown configuration
        cooldown_mode, cooldown_value = _parse_cooldown_config(
            config.get("cooldown_mode"),
            config.get("cooldown_value", 0),
        )

        return cls(
            tier_key=tier_key,
            model_or_group=model_or_group,
            max_requests=max_requests_value,
            max_requests_mode=max_requests_mode,
            cooldown_mode=cooldown_mode,
            cooldown_value=cooldown_value,
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

    # Window limits blocking (if True, block credentials when window quota exhausted locally)
    # Default False: only API errors (cooldowns) should block, not local tracking
    window_limits_enabled: bool = False

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

    Only used when provider doesn't define custom windows via
    usage_reset_configs or get_usage_reset_config().
    """
    return [
        WindowDefinition.rolling("daily", 86400, is_primary=True, applies_to="model"),
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
                quota_threshold=fc_config.get(
                    "quota_threshold", DEFAULT_FAIR_CYCLE_QUOTA_THRESHOLD
                ),
                reset_cooldown_threshold=fc_config.get(
                    "reset_cooldown_threshold",
                    DEFAULT_FAIR_CYCLE_RESET_COOLDOWN_THRESHOLD,
                ),
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
            if hasattr(plugin_class, "default_fair_cycle_quota_threshold"):
                config.fair_cycle.quota_threshold = (
                    plugin_class.default_fair_cycle_quota_threshold
                )

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
                        cap = CustomCapConfig.from_dict(
                            str(resolved_tier), model_or_group, cap_config
                        )
                        if cap is not None:
                            config.custom_caps.append(cap)
                        if cap is not None:
                            config.custom_caps.append(cap)

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

    # Fair cycle quota threshold from env
    env_fc_quota = os.getenv(f"FAIR_CYCLE_QUOTA_THRESHOLD_{provider_upper}")
    if env_fc_quota:
        try:
            config.fair_cycle.quota_threshold = float(env_fc_quota)
        except ValueError:
            pass

    # Fair cycle reset cooldown threshold from env
    env_fc_reset_cd = os.getenv(f"FAIR_CYCLE_RESET_COOLDOWN_THRESHOLD_{provider_upper}")
    if env_fc_reset_cd:
        try:
            config.fair_cycle.reset_cooldown_threshold = int(env_fc_reset_cd)
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
                "max_requests_mode": cap.max_requests_mode.value,
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
                cap = CustomCapConfig.from_dict(tier_key, model_or_group, cap_config)
                if cap is not None:
                    config.custom_caps.append(cap)

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
