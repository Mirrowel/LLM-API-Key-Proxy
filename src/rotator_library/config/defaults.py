"""
Centralized defaults for the rotator library.

This file contains all tunable default values for features like:
- Credential rotation and selection
- Fair Cycle Rotation
- Custom Caps
- Cooldown and backoff timing

Providers can override these by setting class attributes.
Environment variables can override at runtime.

See DOCUMENTATION.md for detailed descriptions of each setting.
"""

from typing import Dict, Optional

# =============================================================================
# ROTATION & SELECTION DEFAULTS
# =============================================================================

# Default credential rotation mode
# Options: "balanced" (distribute load) or "sequential" (use until exhausted)
# Override per-provider: ROTATION_MODE_{PROVIDER}=balanced/sequential
DEFAULT_ROTATION_MODE: str = "balanced"

# Weight tolerance for weighted random credential selection
# 0.0 = deterministic (always pick least-used)
# 2.0-4.0 = balanced randomness (recommended)
# 5.0+ = high randomness
DEFAULT_ROTATION_TOLERANCE: float = 3.0

# Maximum retries per credential before rotating
DEFAULT_MAX_RETRIES: int = 2

# Global request timeout in seconds
# This controls how long a request can wait for an available credential.
# If all credentials are on cooldown and the soonest one won't be available
# within this timeout, the request fails fast with a clear message.
# Override via environment variable: GLOBAL_TIMEOUT=<seconds>
DEFAULT_GLOBAL_TIMEOUT: int = 30

# =============================================================================
# TIER & PRIORITY DEFAULTS
# =============================================================================

# Default priority for tiers not in tier_priorities mapping (lower = higher priority)
DEFAULT_TIER_PRIORITY: int = 10

# Fallback concurrency multiplier for sequential mode
# Used when priority not in default_priority_multipliers
DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER: int = 1

# =============================================================================
# FAIR CYCLE ROTATION DEFAULTS
# =============================================================================
# Fair cycle ensures each credential exhausts at least once before reuse.

# Enable fair cycle rotation
# None = derive from rotation mode (enabled for sequential only)
# Override: FAIR_CYCLE_{PROVIDER}=true/false
DEFAULT_FAIR_CYCLE_ENABLED: Optional[bool] = None

# Tracking mode for fair cycle
# "model_group" = track per quota group (or per model if ungrouped)
# "credential" = track per credential globally (ignores model)
# Override: FAIR_CYCLE_TRACKING_MODE_{PROVIDER}=model_group/credential
DEFAULT_FAIR_CYCLE_TRACKING_MODE: str = "model_group"

# Cross-tier tracking
# False = each priority tier cycles independently
# True = ALL credentials must exhaust regardless of tier
# Override: FAIR_CYCLE_CROSS_TIER_{PROVIDER}=true/false
DEFAULT_FAIR_CYCLE_CROSS_TIER: bool = False

# Cycle duration in seconds (how long before cycle auto-resets)
# Override: FAIR_CYCLE_DURATION_{PROVIDER}=<seconds>
DEFAULT_FAIR_CYCLE_DURATION: int = 172800  # 48 hours

# Exhaustion cooldown threshold in seconds
# Cooldowns longer than this mark credential as "exhausted" for fair cycle
# Override: EXHAUSTION_COOLDOWN_THRESHOLD_{PROVIDER}=<seconds>
# Global fallback: EXHAUSTION_COOLDOWN_THRESHOLD=<seconds>
DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD: int = 300  # 5 minutes

# =============================================================================
# CUSTOM CAPS DEFAULTS
# =============================================================================
# Custom caps allow setting usage limits more restrictive than actual API limits.

# Default cooldown mode when custom cap is hit
# Options: "quota_reset" | "offset" | "fixed"
DEFAULT_CUSTOM_CAP_COOLDOWN_MODE: str = "quota_reset"

# Default cooldown value in seconds (for offset/fixed modes)
DEFAULT_CUSTOM_CAP_COOLDOWN_VALUE: int = 0

# =============================================================================
# COOLDOWN & BACKOFF DEFAULTS
# =============================================================================
# These control how long credentials are paused after errors.

# Escalating backoff tiers for consecutive failures (seconds)
# Key = failure count, Value = cooldown duration
COOLDOWN_BACKOFF_TIERS: Dict[int, int] = {
    1: 10,  # 1st failure: 10 seconds
    2: 30,  # 2nd failure: 30 seconds
    3: 60,  # 3rd failure: 1 minute
    4: 120,  # 4th failure: 2 minutes
}

# Maximum backoff for 5+ consecutive failures (seconds)
COOLDOWN_BACKOFF_MAX: int = 300  # 5 minutes

# Authentication error lockout duration (seconds)
# Applied when 401/403 received - credential assumed revoked
COOLDOWN_AUTH_ERROR: int = 300  # 5 minutes

# Transient/provider-level error cooldown (seconds)
# Applied for errors that don't count against credential health
COOLDOWN_TRANSIENT_ERROR: int = 30

# Default rate limit cooldown when retry_after not provided (seconds)
COOLDOWN_RATE_LIMIT_DEFAULT: int = 60
