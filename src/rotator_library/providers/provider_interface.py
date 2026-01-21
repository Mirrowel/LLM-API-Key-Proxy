# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    List,
    Dict,
    Any,
    Optional,
    AsyncGenerator,
    Union,
    FrozenSet,
    Tuple,
    TYPE_CHECKING,
)
import os
import httpx
import litellm

if TYPE_CHECKING:
    from ..usage_manager import UsageManager

from ..config import (
    DEFAULT_ROTATION_MODE,
    DEFAULT_TIER_PRIORITY,
    DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER,
    DEFAULT_FAIR_CYCLE_ENABLED,
    DEFAULT_FAIR_CYCLE_TRACKING_MODE,
    DEFAULT_FAIR_CYCLE_CROSS_TIER,
    DEFAULT_FAIR_CYCLE_DURATION,
    DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD,
)


# =============================================================================
# TIER & USAGE CONFIGURATION TYPES
# =============================================================================


@dataclass(frozen=True)
class UsageResetConfigDef:
    """
    Definition for usage reset configuration per tier type.

    Providers define these as class attributes to specify how usage stats
    should reset based on credential tier (paid vs free).

    Attributes:
        window_seconds: Duration of the usage tracking window in seconds.
        mode: Either "credential" (one window per credential) or "per_model"
              (separate window per model or model group).
        description: Human-readable description for logging.
        field_name: The key used in usage data JSON structure.
                    Typically "models" for per_model mode, "daily" for credential mode.
    """

    window_seconds: int
    mode: str  # "credential" or "per_model"
    description: str
    field_name: str = "daily"  # Default for backwards compatibility


# Type aliases for provider configuration
TierPriorityMap = Dict[str, int]  # tier_name -> priority
UsageConfigKey = Union[FrozenSet[int], str]  # frozenset of priorities OR "default"
UsageConfigMap = Dict[UsageConfigKey, UsageResetConfigDef]  # priority_set -> config
QuotaGroupMap = Dict[str, List[str]]  # group_name -> [models]


class ProviderInterface(ABC):
    """
    An interface for API provider-specific functionality, including model
    discovery and custom API call handling for non-standard providers.
    """

    skip_cost_calculation: bool = False

    # Default rotation mode for this provider ("balanced" or "sequential")
    # - "balanced": Rotate credentials to distribute load evenly
    # - "sequential": Use one credential until exhausted, then switch to next
    # See config/defaults.py for the global default value
    default_rotation_mode: str = DEFAULT_ROTATION_MODE

    # =========================================================================
    # TIER CONFIGURATION - Override in subclass
    # =========================================================================

    # Provider name for env var lookups (e.g., "antigravity", "gemini_cli")
    # Used for: QUOTA_GROUPS_{provider_env_name}_{GROUP}
    provider_env_name: str = ""

    # Tier name -> priority mapping (Single Source of Truth)
    # Lower numbers = higher priority (1 is highest)
    # Multiple tiers can map to the same priority
    # Unknown tiers fall back to default_tier_priority
    tier_priorities: TierPriorityMap = {}

    # Default priority for tiers not in tier_priorities mapping
    # See config/defaults.py for the global default value
    default_tier_priority: int = DEFAULT_TIER_PRIORITY

    # =========================================================================
    # USAGE RESET CONFIGURATION - Override in subclass
    # =========================================================================

    # Usage reset configurations keyed by priority sets
    # Keys: frozenset of priority values (e.g., frozenset({1, 2})) OR "default"
    # The "default" key is used for any priority not matched by a frozenset
    usage_reset_configs: UsageConfigMap = {}

    # =========================================================================
    # MODEL QUOTA GROUPS - Override in subclass
    # =========================================================================

    # Models that share quota/cooldown timing
    # Can be overridden via env: QUOTA_GROUPS_{PROVIDER}_{GROUP}="model1,model2"
    model_quota_groups: QuotaGroupMap = {}

    # Model usage weights for grouped usage calculation
    # When calculating combined usage for quota groups, each model's usage
    # is multiplied by its weight. This accounts for models that consume
    # more quota per request (e.g., Opus uses more than Sonnet).
    # Models not in the map default to weight 1.
    # Example: {"claude-opus-4-5": 2} means Opus usage counts 2x
    model_usage_weights: Dict[str, int] = {}

    # =========================================================================
    # PRIORITY CONCURRENCY MULTIPLIERS - Override in subclass
    # =========================================================================

    # Priority-based concurrency multipliers (universal, applies to all modes)
    # Maps priority level -> multiplier
    # Higher priority credentials (lower number) can have higher multipliers
    # to allow more concurrent requests
    # Example: {1: 5, 2: 3} means Priority 1 gets 5x, Priority 2 gets 3x
    default_priority_multipliers: Dict[int, int] = {}

    # Fallback multiplier for sequential mode when priority not in default_priority_multipliers
    # This is used for lower-priority tiers in sequential mode to maintain some stickiness
    # See config/defaults.py for the global default value
    default_sequential_fallback_multiplier: int = DEFAULT_SEQUENTIAL_FALLBACK_MULTIPLIER

    # =========================================================================
    # FAIR CYCLE ROTATION - Override in subclass
    # =========================================================================

    # Fair cycle ensures each credential is used at least once before reuse.
    # When a credential is "exhausted" (long cooldown > threshold), it's marked
    # and cannot be selected again until all credentials in its tier exhaust.

    # Enable fair cycle rotation for this provider
    # None = derive from rotation mode (enabled for sequential only, disabled for balanced)
    # Can be overridden via env: FAIR_CYCLE_{PROVIDER}=true/false
    default_fair_cycle_enabled: Optional[bool] = DEFAULT_FAIR_CYCLE_ENABLED

    # Tracking mode for fair cycle:
    # - "model_group": Track exhaustion per quota group (or per model if ungrouped)
    # - "credential": Track exhaustion per credential globally (ignores model)
    # Can be overridden via env: FAIR_CYCLE_TRACKING_MODE_{PROVIDER}=model_group/credential
    default_fair_cycle_tracking_mode: str = DEFAULT_FAIR_CYCLE_TRACKING_MODE

    # Cross-tier tracking:
    # - False: Each priority tier cycles independently
    # - True: ALL credentials must exhaust before any can reuse (ignores tier boundaries)
    # Can be overridden via env: FAIR_CYCLE_CROSS_TIER_{PROVIDER}=true/false
    default_fair_cycle_cross_tier: bool = DEFAULT_FAIR_CYCLE_CROSS_TIER

    # Cycle duration in seconds (how long before cycle resets from start)
    # Can be overridden via env: FAIR_CYCLE_DURATION_{PROVIDER}=<seconds>
    default_fair_cycle_duration: int = DEFAULT_FAIR_CYCLE_DURATION

    # Exhaustion cooldown threshold in seconds
    # A cooldown must exceed this duration to qualify as "exhausted" for fair cycle
    # Short rate limits (e.g., 60s) don't trigger exhaustion; only long quota cooldowns do
    # Can be overridden via env: EXHAUSTION_COOLDOWN_THRESHOLD_{PROVIDER}=<seconds>
    # Global fallback: EXHAUSTION_COOLDOWN_THRESHOLD=<seconds>
    default_exhaustion_cooldown_threshold: int = DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD

    # =========================================================================
    # CUSTOM CAPS - Override in subclass
    # =========================================================================

    # Custom request caps per tier, per model or quota group
    # Applies to ALL credentials of that tier for this provider
    #
    # Keys:
    #   - int: Single tier priority (e.g., 2 for standard-tier)
    #   - tuple of ints: Multiple tiers sharing same config (e.g., (2, 3))
    #   - "default": Fallback for tiers not explicitly configured
    #
    # Values: Dict mapping model/group name to config:
    #   {
    #       "max_requests": int | str,      # Absolute (100) or percentage ("80%")
    #       "cooldown_mode": str,           # "quota_reset" | "offset" | "fixed"
    #       "cooldown_value": int,          # Seconds for offset/fixed (default 0)
    #   }
    #
    # Resolution order: tier+model → tier+group → default+model → default+group
    #
    # Clamping (more restrictive only):
    #   - max_requests: min(custom, actual_max)
    #   - cooldown: max(calculated, natural_reset_ts)
    #
    # Env override format:
    #   CUSTOM_CAP_{PROVIDER}_T{TIER}_{MODEL_OR_GROUP}=<value>
    #   CUSTOM_CAP_COOLDOWN_{PROVIDER}_T{TIER}_{MODEL_OR_GROUP}=<mode>:<value>
    #
    # Name transformations for env vars:
    #   - Dashes (-) → Underscores (_)
    #   - Dots (.) → Underscores (_)
    #   - All uppercase
    default_custom_caps: Dict[
        Union[int, Tuple[int, ...], str], Dict[str, Dict[str, Any]]
    ] = {}

    @abstractmethod
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available model names from the provider's API.

        Args:
            api_key: The API key required for authentication.
            client: An httpx.AsyncClient instance for making requests.

        Returns:
            A list of model name strings.
        """
        pass

    # [NEW] Add methods for providers that need to bypass litellm
    def has_custom_logic(self) -> bool:
        """
        Returns True if the provider implements its own acompletion/aembedding logic,
        bypassing the standard litellm call.
        """
        return False

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handles the entire completion call for non-standard providers.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement custom acompletion."
        )

    async def aembedding(
        self, client: httpx.AsyncClient, **kwargs
    ) -> litellm.EmbeddingResponse:
        """Handles the entire embedding call for non-standard providers."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement custom aembedding."
        )

    def convert_safety_settings(
        self, settings: Dict[str, str]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Converts a generic safety settings dictionary to the provider-specific format.

        Args:
            settings: A dictionary with generic harm categories and thresholds.

        Returns:
            A list of provider-specific safety setting objects or None.
        """
        return None

    # [NEW] Add new methods for OAuth providers
    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        For OAuth providers, this method returns the Authorization header.
        For API key providers, this can be a no-op or raise NotImplementedError.
        """
        raise NotImplementedError("This provider does not support OAuth.")

    async def proactively_refresh(self, credential_path: str):
        """
        Proactively refreshes a token if it's nearing expiry.
        """
        pass

    # [NEW] Credential Prioritization System

    # =========================================================================
    # TIER RESOLUTION LOGIC (Centralized)
    # =========================================================================

    def _resolve_tier_priority(self, tier_name: Optional[str]) -> int:
        """
        Resolve priority for a tier name using provider's tier_priorities mapping.

        Args:
            tier_name: The tier name string (e.g., "free-tier", "standard-tier")

        Returns:
            Priority level from tier_priorities, or default_tier_priority if
            tier_name is None or not found in the mapping.
        """
        if tier_name is None:
            return self.default_tier_priority
        return self.tier_priorities.get(tier_name, self.default_tier_priority)

    def get_credential_priority(self, credential: str) -> Optional[int]:
        """
        Returns the priority level for a credential.
        Lower numbers = higher priority (1 is highest).
        Returns None if tier not yet discovered.

        Uses the provider's tier_priorities mapping to resolve priority from
        tier name. Unknown tiers fall back to default_tier_priority.

        Subclasses should:
        1. Define tier_priorities dict with all known tier names
        2. Override get_credential_tier_name() for tier lookup
        Do NOT override this method.

        Args:
            credential: The credential identifier (API key or path)

        Returns:
            Priority level (1-10) or None if tier not yet discovered
        """
        tier = self.get_credential_tier_name(credential)
        if tier is None:
            return None  # Tier not yet discovered
        return self._resolve_tier_priority(tier)

    def get_model_tier_requirement(self, model: str) -> Optional[int]:
        """
        Returns the minimum priority tier required for a model.
        If a model requires priority 1, only credentials with priority <= 1 can use it.

        This allows providers to restrict certain models to specific credential tiers.
        For example, Gemini 3 models require paid-tier credentials.

        Args:
            model: The model name (with or without provider prefix)

        Returns:
            Minimum required priority level or None if no restrictions

        Example:
            For Gemini CLI:
            - gemini-3-*: requires priority 1 (paid tier only)
            - gemini-2.5-*: no restriction (None)
        """
        return None

    async def initialize_credentials(self, credential_paths: List[str]) -> None:
        """
        Called at startup to initialize provider with all available credentials.

        Providers can override this to load cached tier data, discover priorities,
        or perform any other initialization needed before the first API request.

        This is called once during startup by the BackgroundRefresher before
        the main refresh loop begins.

        Args:
            credential_paths: List of credential file paths for this provider
        """
        pass

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """
        Returns the human-readable tier name for a credential.

        This is used for logging purposes to show which plan tier a credential belongs to.

        Args:
            credential: The credential identifier (API key or path)

        Returns:
            Tier name string (e.g., "free-tier", "paid-tier") or None if unknown
        """
        return None

    # =========================================================================
    # Sequential Rotation Support
    # =========================================================================

    @classmethod
    def get_rotation_mode(cls, provider_name: str) -> str:
        """
        Get the rotation mode for this provider.

        Checks ROTATION_MODE_{PROVIDER} environment variable first,
        then falls back to the class's default_rotation_mode.

        Args:
            provider_name: The provider name (e.g., "antigravity", "gemini_cli")

        Returns:
            "balanced" or "sequential"
        """
        env_key = f"ROTATION_MODE_{provider_name.upper()}"
        return os.getenv(env_key, cls.default_rotation_mode)

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a quota/rate-limit error and extract structured information.

        Providers should override this method to handle their specific error formats.
        This allows the error_handler to use provider-specific parsing when available,
        falling back to generic parsing otherwise.

        Args:
            error: The caught exception
            error_body: Optional raw response body string

        Returns:
            None if not a parseable quota error, otherwise:
            {
                "retry_after": int,  # seconds until quota resets
                "reason": str,       # e.g., "QUOTA_EXHAUSTED", "RATE_LIMITED"
                "reset_timestamp": str | None,  # ISO timestamp if available
                "quota_reset_timestamp": float | None,  # Unix timestamp for quota reset
            }
        """
        return None  # Default: no provider-specific parsing

    # =========================================================================
    # Per-Provider Usage Tracking Configuration
    # =========================================================================

    # =========================================================================
    # USAGE RESET CONFIG LOGIC (Centralized)
    # =========================================================================

    def _find_usage_config_for_priority(
        self, priority: int
    ) -> Optional[UsageResetConfigDef]:
        """
        Find usage config that applies to a priority value.

        Checks frozenset keys first (priority must be in the set),
        then falls back to "default" key if no match found.

        Args:
            priority: The credential priority level

        Returns:
            UsageResetConfigDef if found, None otherwise
        """
        # First, check frozenset keys for explicit priority match
        for key, config in self.usage_reset_configs.items():
            if isinstance(key, frozenset) and priority in key:
                return config

        # Fall back to "default" key
        return self.usage_reset_configs.get("default")

    def _build_usage_reset_config(
        self, tier_name: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Build usage reset configuration dict for a tier.

        Resolves tier to priority, then finds matching usage config.
        Returns None if provider doesn't define usage_reset_configs.

        Args:
            tier_name: The tier name string

        Returns:
            Usage config dict with window_seconds, mode, priority, description,
            field_name, or None if no config applies
        """
        if not self.usage_reset_configs:
            return None

        priority = self._resolve_tier_priority(tier_name)
        config = self._find_usage_config_for_priority(priority)

        if config is None:
            return None

        return {
            "window_seconds": config.window_seconds,
            "mode": config.mode,
            "priority": priority,
            "description": config.description,
            "field_name": config.field_name,
        }

    def get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Get provider-specific usage tracking configuration for a credential.

        Uses the provider's usage_reset_configs class attribute to build
        the configuration dict. Priority is auto-derived from tier.

        Subclasses should define usage_reset_configs as a class attribute
        instead of overriding this method. Only override get_credential_tier_name()
        to provide the tier lookup mechanism.

        The UsageManager will use this configuration to:
        1. Track usage per-model or per-credential based on mode
        2. Reset usage based on a rolling window OR quota exhausted timestamp
        3. Archive stats to "global" when the window/quota expires

        Args:
            credential: The credential identifier (API key or path)

        Returns:
            None to use default daily reset, otherwise a dict with:
            {
                "window_seconds": int,     # Duration in seconds (e.g., 18000 for 5h)
                "mode": str,               # "credential" or "per_model"
                "priority": int,           # Priority level (auto-derived from tier)
                "description": str,        # Human-readable description (for logging)
            }

        Modes:
            - "credential": One window per credential. Window starts from first
              request of ANY model. All models reset together when window expires.
            - "per_model": Separate window per model (or model group). Window starts
              from first request of THAT model. Models reset independently unless
              grouped. If a quota_exhausted error provides exact reset time, that
              becomes the authoritative reset time for the model.
        """
        tier = self.get_credential_tier_name(credential)
        return self._build_usage_reset_config(tier)

    def get_default_usage_field_name(self) -> str:
        """
        Get the default usage tracking field name for this provider.

        Providers can override this to use a custom field name for usage tracking
        when no credential-specific config is available.

        Returns:
            Field name string (default: "daily")
        """
        return "daily"

    # =========================================================================
    # Model Quota Grouping
    # =========================================================================

    # =========================================================================
    # QUOTA GROUPS LOGIC (Centralized)
    # =========================================================================

    def _get_effective_quota_groups(self) -> QuotaGroupMap:
        """
        Get quota groups with .env overrides applied.

        Env format: QUOTA_GROUPS_{PROVIDER}_{GROUP}="model1,model2"
        Set empty string to disable a default group.
        """
        if not self.provider_env_name or not self.model_quota_groups:
            return self.model_quota_groups

        result: QuotaGroupMap = {}

        for group_name, default_models in self.model_quota_groups.items():
            env_key = (
                f"QUOTA_GROUPS_{self.provider_env_name.upper()}_{group_name.upper()}"
            )
            env_value = os.getenv(env_key)

            if env_value is not None:
                # Env override present
                if env_value.strip():
                    # Parse comma-separated models
                    result[group_name] = [
                        m.strip() for m in env_value.split(",") if m.strip()
                    ]
                # Empty string = group disabled, don't add to result
            else:
                # Use default
                result[group_name] = list(default_models)

        return result

    def _find_model_quota_group(self, model: str) -> Optional[str]:
        """Find which quota group a model belongs to."""
        groups = self._get_effective_quota_groups()
        for group_name, models in groups.items():
            if model in models:
                return group_name
        return None

    def _get_quota_group_models(self, group: str) -> List[str]:
        """Get all models in a quota group."""
        groups = self._get_effective_quota_groups()
        return groups.get(group, [])

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """
        Returns the quota group name for a model, or None if not grouped.

        Uses the provider's model_quota_groups class attribute with .env overrides
        via QUOTA_GROUPS_{PROVIDER}_{GROUP}="model1,model2".

        Models in the same quota group share cooldown timing - when one model
        hits a quota exhausted error, all models in the group get the same
        reset timestamp. They also reset (archive stats) together.

        Subclasses should define model_quota_groups as a class attribute
        instead of overriding this method.

        Args:
            model: Model name (with or without provider prefix)

        Returns:
            Group name string (e.g., "claude") or None if model is not grouped
        """
        # Strip provider prefix if present
        clean_model = model.split("/")[-1] if "/" in model else model
        return self._find_model_quota_group(clean_model)

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """
        Returns all model names that belong to a quota group.

        Uses the provider's model_quota_groups class attribute with .env overrides.

        Args:
            group: Group name (e.g., "claude")

        Returns:
            List of model names (WITHOUT provider prefix) in the group.
            Empty list if group doesn't exist.
        """
        return self._get_quota_group_models(group)

    def get_model_usage_weight(self, model: str) -> int:
        """
        Returns the usage weight for a model when calculating grouped usage.

        Models with higher weights contribute more to the combined group usage.
        This accounts for models that consume more quota per request.

        Args:
            model: Model name (with or without provider prefix)

        Returns:
            Weight multiplier (default 1 if not configured)
        """
        # Strip provider prefix if present
        clean_model = model.split("/")[-1] if "/" in model else model
        return self.model_usage_weights.get(clean_model, 1)

    def normalize_model_for_tracking(self, model: str) -> str:
        """
        Normalize internal model names to public-facing names for usage tracking.

        Some providers use internal model variants (e.g., claude-sonnet-4-5-thinking)
        that should be tracked under their public name (e.g., claude-sonnet-4-5).
        This ensures key_usage.json only contains public-facing model names.

        Default implementation: returns model unchanged.
        Providers with internal variants should override this method.

        Args:
            model: Model name (with or without provider prefix)

        Returns:
            Normalized public-facing model name (preserves provider prefix if present)
        """
        return model

    # =========================================================================
    # BACKGROUND JOB INTERFACE - Override in subclass for periodic tasks
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Return configuration for provider-specific background job, or None if none.

        Providers that need periodic background tasks (e.g., quota refresh,
        cache cleanup) should override this method.

        The BackgroundRefresher will call run_background_job() at the specified
        interval for each provider that returns a config.

        Returns:
            None if no background job, otherwise:
            {
                "interval": 300,  # seconds between runs
                "name": "my_job",  # for logging (e.g., "quota_refresh")
                "run_on_start": True,  # whether to run immediately at startup
            }
        """
        return None

    async def run_background_job(
        self,
        usage_manager: "UsageManager",
        credentials: List[str],
    ) -> None:
        """
        Execute the provider's periodic background job.

        Called by BackgroundRefresher at the interval specified in
        get_background_job_config(). Override this method to implement
        provider-specific periodic tasks.

        Args:
            usage_manager: UsageManager instance for storing/reading usage data
            credentials: List of credential paths for this provider
        """
        pass
