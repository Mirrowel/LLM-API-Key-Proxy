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

from typing import TYPE_CHECKING

# Types and config are dependency-light leaves (dataclasses + constants);
# importing them eagerly keeps `from rotator_library.usage import ...`
# ergonomic for the common type-only consumers. Every engine/facade
# import is deferred: components pull litellm (~8s) and must stay behind
# the proxy's loading screens, off the launcher fast path.
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
from .config import (
    ProviderUsageConfig,
    FairCycleConfig,
    CustomCapConfig,
    WindowDefinition,
    load_provider_usage_config,
)

if TYPE_CHECKING:
    from .accounting import UsageRecord, extract_usage_record
    from .costs import CostBreakdown, CostCalculator, ModelPricing
    from .identity.registry import CredentialRegistry
    from .integration.api import UsageAPI
    from .limits.engine import LimitEngine
    from .manager import UsageManager, CredentialContext
    from .persistence.storage import UsageStorage
    from .selection.engine import SelectionEngine
    from .tracking.engine import TrackingEngine
    from .tracking.windows import WindowManager

__all__ = [
    "WindowStats",
    "TotalStats",
    "ModelStats",
    "GroupStats",
    "CredentialState",
    "CooldownInfo",
    "FairCycleState",
    "UsageUpdate",
    "SelectionContext",
    "LimitCheckResult",
    "RotationMode",
    "ResetMode",
    "LimitResult",
    "ProviderUsageConfig",
    "FairCycleConfig",
    "CustomCapConfig",
    "WindowDefinition",
    "load_provider_usage_config",
    "UsageManager",
    "CredentialContext",
    "CredentialRegistry",
    "TrackingEngine",
    "WindowManager",
    "LimitEngine",
    "SelectionEngine",
    "UsageStorage",
    "UsageAPI",
    "UsageRecord",
    "extract_usage_record",
    "CostBreakdown",
    "CostCalculator",
    "ModelPricing",
]

_LAZY_EXPORTS = {
    "UsageRecord": ".accounting",
    "extract_usage_record": ".accounting",
    "CostBreakdown": ".costs",
    "CostCalculator": ".costs",
    "ModelPricing": ".costs",
    "CredentialRegistry": ".identity.registry",
    "UsageAPI": ".integration.api",
    "LimitEngine": ".limits.engine",
    "UsageManager": ".manager",
    "CredentialContext": ".manager",
    "UsageStorage": ".persistence.storage",
    "SelectionEngine": ".selection.engine",
    "TrackingEngine": ".tracking.engine",
    "WindowManager": ".tracking.windows",
}


def __getattr__(name):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path, __name__), name)
    globals()[name] = value
    return value
