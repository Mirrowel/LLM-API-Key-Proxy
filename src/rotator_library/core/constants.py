# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Constants and default values for the rotator library.

This module re-exports all constants from the config package and adds
any additional constants needed for the refactored architecture.

All tunable defaults are in config/defaults.py - this module provides
a unified import point and adds non-tunable constants.
"""

# Re-export all tunable defaults from config package
from ..config import (
    # Rotation & Selection
    DEFAULT_ROTATION_MODE,
    DEFAULT_ROTATION_TOLERANCE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_GLOBAL_TIMEOUT,
    # Tier & Priority
    DEFAULT_TIER_PRIORITY,
    DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER,
    # Fair Cycle Rotation
    DEFAULT_FAIR_CYCLE_ENABLED,
    DEFAULT_FAIR_CYCLE_TRACKING_MODE,
    DEFAULT_FAIR_CYCLE_CROSS_TIER,
    DEFAULT_FAIR_CYCLE_DURATION,
    DEFAULT_FAIR_CYCLE_QUOTA_THRESHOLD,
    DEFAULT_FAIR_CYCLE_RESET_COOLDOWN_THRESHOLD,
    DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD,
    # Custom Caps
    DEFAULT_CUSTOM_CAP_COOLDOWN_MODE,
    DEFAULT_CUSTOM_CAP_COOLDOWN_VALUE,
    # Cooldown & Backoff
    COOLDOWN_BACKOFF_TIERS,
    COOLDOWN_BACKOFF_MAX,
    COOLDOWN_AUTH_ERROR,
    COOLDOWN_TRANSIENT_ERROR,
    COOLDOWN_RATE_LIMIT_DEFAULT,
    # Small Cooldown Auto-Retry
    DEFAULT_SMALL_COOLDOWN_RETRY_THRESHOLD,
)

# =============================================================================
# ADDITIONAL CONSTANTS FOR REFACTORED ARCHITECTURE
# =============================================================================

# Environment variable prefixes for configuration
ENV_PREFIX_ROTATION_MODE = "ROTATION_MODE_"
ENV_PREFIX_FAIR_CYCLE = "FAIR_CYCLE_"
ENV_PREFIX_FAIR_CYCLE_TRACKING = "FAIR_CYCLE_TRACKING_MODE_"
ENV_PREFIX_FAIR_CYCLE_CROSS_TIER = "FAIR_CYCLE_CROSS_TIER_"
ENV_PREFIX_FAIR_CYCLE_DURATION = "FAIR_CYCLE_DURATION_"
ENV_PREFIX_EXHAUSTION_THRESHOLD = "EXHAUSTION_COOLDOWN_THRESHOLD_"
ENV_PREFIX_CONCURRENCY_MULTIPLIER = "CONCURRENCY_MULTIPLIER_"
ENV_PREFIX_CUSTOM_CAP = "CUSTOM_CAP_"
ENV_PREFIX_CUSTOM_CAP_COOLDOWN = "CUSTOM_CAP_COOLDOWN_"
ENV_PREFIX_QUOTA_GROUPS = "QUOTA_GROUPS_"

# Provider-specific providers that use request_count instead of success_count
# for credential selection (because failed requests also consume quota)
REQUEST_COUNT_PROVIDERS = frozenset({"antigravity", "gemini_cli", "chutes", "nanogpt"})

# Usage manager storage
USAGE_FILE_NAME = "usage.json"  # New format
LEGACY_USAGE_FILE_NAME = "key_usage.json"  # Old format
USAGE_SCHEMA_VERSION = 2

# Fair cycle tracking keys
FAIR_CYCLE_ALL_TIERS_KEY = "__all_tiers__"
FAIR_CYCLE_CREDENTIAL_KEY = "__credential__"
FAIR_CYCLE_STORAGE_KEY = "__fair_cycle__"

# Logging
LIB_LOGGER_NAME = "rotator_library"

__all__ = [
    # From config package
    "DEFAULT_ROTATION_MODE",
    "DEFAULT_ROTATION_TOLERANCE",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_GLOBAL_TIMEOUT",
    "DEFAULT_TIER_PRIORITY",
    "DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER",
    "DEFAULT_FAIR_CYCLE_ENABLED",
    "DEFAULT_FAIR_CYCLE_TRACKING_MODE",
    "DEFAULT_FAIR_CYCLE_CROSS_TIER",
    "DEFAULT_FAIR_CYCLE_DURATION",
    "DEFAULT_FAIR_CYCLE_QUOTA_THRESHOLD",
    "DEFAULT_FAIR_CYCLE_RESET_COOLDOWN_THRESHOLD",
    "DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD",
    "DEFAULT_CUSTOM_CAP_COOLDOWN_MODE",
    "DEFAULT_CUSTOM_CAP_COOLDOWN_VALUE",
    "COOLDOWN_BACKOFF_TIERS",
    "COOLDOWN_BACKOFF_MAX",
    "COOLDOWN_AUTH_ERROR",
    "COOLDOWN_TRANSIENT_ERROR",
    "COOLDOWN_RATE_LIMIT_DEFAULT",
    # Small Cooldown Auto-Retry
    "DEFAULT_SMALL_COOLDOWN_RETRY_THRESHOLD",
    # Environment variable prefixes
    "ENV_PREFIX_ROTATION_MODE",
    "ENV_PREFIX_FAIR_CYCLE",
    "ENV_PREFIX_FAIR_CYCLE_TRACKING",
    "ENV_PREFIX_FAIR_CYCLE_CROSS_TIER",
    "ENV_PREFIX_FAIR_CYCLE_DURATION",
    "ENV_PREFIX_EXHAUSTION_THRESHOLD",
    "ENV_PREFIX_CONCURRENCY_MULTIPLIER",
    "ENV_PREFIX_CUSTOM_CAP",
    "ENV_PREFIX_CUSTOM_CAP_COOLDOWN",
    "ENV_PREFIX_QUOTA_GROUPS",
    # Provider sets
    "REQUEST_COUNT_PROVIDERS",
    # Storage
    "USAGE_FILE_NAME",
    "LEGACY_USAGE_FILE_NAME",
    "USAGE_SCHEMA_VERSION",
    # Fair cycle keys
    "FAIR_CYCLE_ALL_TIERS_KEY",
    "FAIR_CYCLE_CREDENTIAL_KEY",
    "FAIR_CYCLE_STORAGE_KEY",
    # Logging
    "LIB_LOGGER_NAME",
]
