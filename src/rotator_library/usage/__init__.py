# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Usage tracking and credential selection package.

This package provides the UsageManager facade and associated components
for tracking API usage, enforcing limits, and selecting credentials.

Public API:
    UsageManager: Main facade for usage tracking and credential selection
    CredentialContext: Context manager for credential lifecycle

Components (for advanced usage):
    CredentialRegistry: Stable credential identity management
    TrackingEngine: Usage recording and window management
    LimitEngine: Limit checking and enforcement
    SelectionEngine: Credential selection with strategies
    UsageStorage: JSON file persistence
"""

# Types first (no dependencies on other modules)
from .types import (
    WindowStats,
    TotalStats,
    ModelStats,
    GroupStats,
    CredentialState,
    CooldownInfo,
    FairCycleState,
    UsageUpdate,
    SelectionContext,
    LimitCheckResult,
    RotationMode,
    ResetMode,
    LimitResult,
)

# Config
from .config import (
    ProviderUsageConfig,
    FairCycleConfig,
    CustomCapConfig,
    WindowDefinition,
    load_provider_usage_config,
)

# Components
from .identity.registry import CredentialRegistry
from .tracking.windows import WindowManager
from .tracking.engine import TrackingEngine
from .limits.engine import LimitEngine
from .selection.engine import SelectionEngine
from .persistence.storage import UsageStorage
from .integration.api import UsageAPI

# Main facade (imports components above)
from .manager import UsageManager, CredentialContext

__all__ = [
    # Main public API
    "UsageManager",
    "CredentialContext",
    # Types
    "WindowStats",
    "TotalStats",
    "ModelStats",
    "GroupStats",
    "CredentialState",
    "UsageUpdate",
    "CooldownInfo",
    "FairCycleState",
    "SelectionContext",
    "LimitCheckResult",
    "RotationMode",
    "ResetMode",
    "LimitResult",
    # Config
    "ProviderUsageConfig",
    "FairCycleConfig",
    "CustomCapConfig",
    "WindowDefinition",
    "load_provider_usage_config",
    # Engines
    "CredentialRegistry",
    "WindowManager",
    "TrackingEngine",
    "LimitEngine",
    "SelectionEngine",
    "UsageStorage",
    "UsageAPI",
]
