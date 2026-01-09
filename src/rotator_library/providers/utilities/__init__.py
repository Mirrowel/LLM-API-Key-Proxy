# Utilities for provider implementations
from .base_quota_tracker import BaseQuotaTracker
from .antigravity_quota_tracker import AntigravityQuotaTracker
from .gemini_cli_quota_tracker import GeminiCliQuotaTracker

# Shared utilities for Gemini-based providers
from .gemini_shared_utils import (
    env_bool,
    env_int,
    inline_schema_refs,
    normalize_type_arrays,
    clean_gemini_schema,
    recursively_parse_json_strings,
    GEMINI3_TOOL_RENAMES,
    GEMINI3_TOOL_RENAMES_REVERSE,
    FINISH_REASON_MAP,
    DEFAULT_SAFETY_SETTINGS,
)
from .gemini_tool_handler import GeminiToolHandler
from .gemini_credential_manager import GeminiCredentialManager

# Re-export loggers from transaction_logger for backward compatibility
from ...transaction_logger import (
    ProviderLogger,
    AntigravityProviderLogger,
)

# Deprecated aliases for backward compatibility with external consumers
# These map old class names to their new equivalents
GeminiFileLogger = ProviderLogger
GeminiCliFileLogger = ProviderLogger
AntigravityFileLogger = AntigravityProviderLogger

__all__ = [
    # Quota trackers
    "BaseQuotaTracker",
    "AntigravityQuotaTracker",
    "GeminiCliQuotaTracker",
    # Shared utilities
    "env_bool",
    "env_int",
    "inline_schema_refs",
    "normalize_type_arrays",
    "clean_gemini_schema",
    "recursively_parse_json_strings",
    "GEMINI3_TOOL_RENAMES",
    "GEMINI3_TOOL_RENAMES_REVERSE",
    "FINISH_REASON_MAP",
    "DEFAULT_SAFETY_SETTINGS",
    # Loggers (from transaction_logger)
    "ProviderLogger",
    "AntigravityProviderLogger",
    # Deprecated logger aliases (for backward compatibility)
    "GeminiFileLogger",
    "GeminiCliFileLogger",
    "AntigravityFileLogger",
    # Mixins
    "GeminiToolHandler",
    "GeminiCredentialManager",
]
