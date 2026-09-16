# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import (
    List,
    Dict,
    Any,
    Optional,
    AsyncGenerator,
    Union,
    FrozenSet,
    Mapping,
    Tuple,
    TYPE_CHECKING,
)
import os
from fnmatch import fnmatchcase
from functools import lru_cache

import httpx
import litellm

if TYPE_CHECKING:
    from ..usage import UsageManager


# =============================================================================
# SINGLETON METACLASS FOR PROVIDERS
# =============================================================================


class SingletonABCMeta(ABCMeta):
    """
    Metaclass that combines ABC functionality with singleton pattern.

    All classes using this metaclass (including subclasses of ProviderInterface)
    will be singletons - only one instance per class exists.

    This prevents the bug where multiple provider instances are created
    by different components (RotatingClient, UsageManager, Hooks, etc.),
    each with their own caches and state.
    """

    _instances: Dict[type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in SingletonABCMeta._instances:
            SingletonABCMeta._instances[cls] = super().__call__(*args, **kwargs)
        return SingletonABCMeta._instances[cls]


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


def declared_endpoint_path(entry: Any, operation: str) -> Optional[str]:
    """Return an endpoint path for one operation from a declaration block.

    Plan 2.8 (D13): ``endpoint_paths`` is a map keyed by operation; the
    singular ``endpoint_path`` is the legacy fallback applied to any
    operation. Accepts a mapping (class/JSON profile entry) or the
    ``ProviderRuntimeConfig`` dataclass (provider-level JSON config).
    """

    if entry is None:
        return None
    if isinstance(entry, Mapping):
        paths = entry.get("endpoint_paths")
        singular = entry.get("endpoint_path")
    else:
        paths = getattr(entry, "endpoint_paths", None)
        singular = getattr(entry, "endpoint_path", None)
    if isinstance(paths, Mapping):
        candidate = paths.get(operation)
        if candidate:
            return str(candidate)
    return str(singular) if singular else None


def render_endpoint_path(path: str, *, model: str = "", operation: str = "", provider: str = "") -> str:
    """Render ``{model}``/``{operation}``/``{provider}`` placeholders."""

    return path.format(model=model, operation=operation, provider=provider)


def auth_header_pair(
    credential_identifier: str,
    auth_mode: Optional[str],
    auth_header_name: Optional[str],
    *,
    provider: str = "",
) -> Dict[str, str]:
    """Build the credential header for a normalized auth declaration."""

    if auth_mode == "none":
        return {}
    if auth_mode == "x-api-key":
        return {"x-api-key": credential_identifier}
    if auth_mode == "x-goog-api-key":
        return {"x-goog-api-key": credential_identifier}
    if auth_mode == "custom":
        if not auth_header_name:
            owner = f" for provider {provider!r}" if provider else ""
            raise ValueError(f"auth_header_name is required for custom auth{owner}")
        return {auth_header_name: credential_identifier}
    return {"Authorization": f"Bearer {credential_identifier}"}


def _hashable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(k), _hashable_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_value(item) for item in value)
    return value


def _accepts_parameter(method: Any, name: str) -> bool:
    """Whether a callable declares ``name`` (or ``**kwargs``).

    Signature inspection, not exception catching — the same convention the
    executor's profile-aware calls use: a pre-D13 override without the
    ``profile`` parameter keeps its single-argument call, and a TypeError
    raised INSIDE an implementation must propagate rather than be masked
    by a retry.
    """

    try:
        import inspect

        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    if name in parameters:
        return True
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


@lru_cache(maxsize=None)
def _resolved_speaks_profiles(speaks: Tuple[Any, ...]) -> Dict[str, Dict[str, Any]]:
    """Resolve + cache a speaks tuple (module-level so singleton provider
    classes share one resolution; entries normalized to hashable form)."""

    from ..protocols.defaults import resolve_speaks

    normalized = tuple(
        entry if isinstance(entry, str) else tuple(_hashable_value(part) for part in entry)
        for entry in speaks
    )
    # resolve_speaks understands (protocol, overrides-as-pairs)
    denormalized = tuple(
        entry if isinstance(entry, str) else tuple(_dehashable(part) for part in entry)
        for entry in normalized
    )
    return resolve_speaks(denormalized)


def _dehashable(value: Any) -> Any:
    if isinstance(value, tuple) and value and all(isinstance(item, tuple) and len(item) == 2 for item in value):
        return {str(k): _dehashable(v) for k, v in value}
    if isinstance(value, tuple):
        return tuple(_dehashable(item) for item in value)
    return value


class ProviderInterface(ABC, metaclass=SingletonABCMeta):
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

    # Default maximum concurrent requests per credential for this provider.
    # None means use the rotation-mode default. Values <= 0 mean unlimited.
    default_max_concurrent_per_key: Optional[int] = None

    # Mode-specific hard concurrency defaults. None means use the generic
    # provider default above, then the global mode default.
    default_max_concurrent_per_key_balanced: Optional[int] = None
    default_max_concurrent_per_key_sequential: Optional[int] = None

    # Default optimal concurrent requests per credential for this provider.
    # None means use the rotation-mode default. Values <= 0 mean no soft target.
    default_optimal_concurrent_per_key: Optional[int] = None

    # Mode-specific soft concurrency defaults. None means use the generic
    # provider default above, then the global mode default.
    default_optimal_concurrent_per_key_balanced: Optional[int] = None
    default_optimal_concurrent_per_key_sequential: Optional[int] = None

    # Priority multipliers to apply to optimal_concurrent. Kept separate from
    # legacy max multipliers so generic providers do not inherit hard-cap tuning
    # as a soft rotation preference unless they opt in.
    default_optimal_priority_multipliers: Dict[int, int] = {}

    # =========================================================================
    # TIER CONFIGURATION - Override in subclass
    # =========================================================================

    # Provider name for env var lookups (e.g., "openai")
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

    # Seconds to wait for a session-bound sequential credential that is blocked
    # only by concurrency. Providers with expensive context caching can increase
    # this; latency-sensitive providers can lower it or set it to 0.
    default_session_sticky_wait_seconds: Optional[float] = None
    default_session_sticky_entry_ttl_seconds: Optional[int] = None
    default_session_sticky_max_entries: Optional[int] = None

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

    # Native protocol/adapter declarations. W11 flips the default execution
    # mode: providers declaring ``protocol_name`` run native-by-default
    # (LiteLLM is an explicit, logged fallback); undeclared providers keep
    # the LiteLLM-backed path.
    protocol_name: Optional[str] = None

    # G8 envelope v2: declare what this provider speaks. Entries are
    # "protocol" | (protocol, overrides) | (name, protocol, overrides);
    # profile names default to protocol names, endpoints/auth/listing
    # inherit from the protocol registry, and the first entry is the
    # default face. See protocols/defaults.py.
    speaks: Tuple[Any, ...] = ()
    adapter_names: Tuple[str, ...] = ()
    # Optional fnmatch EXCLUSION patterns for model listing (G8 final): a
    # parsed listing id matching any pattern is dropped before the provider
    # prefix is added. Chutes declares ("default*", "*,*") to hide the
    # gateway's non-callable routing pseudo-models. ``listing_filters`` is
    # the declarable successor of a hand-rolled lister.
    listing_filters: Tuple[str, ...] = ()
    # Zero-credential providers (e.g. a local Ollama) declare this so routing
    # can mint the internal no-auth rotation slot when no secret is configured.
    default_auth_mode: Optional[str] = None
    # G2 hookable pipeline: class-level hook declarations (PipelineHook
    # classes, instances, or factories). JSON config ``hooks`` and globally
    # registered hook names add to this declaration (hooks/registry.py).
    hooks: Tuple[Any, ...] = ()
    field_cache_rules: Tuple[Any, ...] = ()
    native_streaming_supported: bool = False
    # G8 capability table: an ORDERED cascade of per-model rule rows
    # (top-to-bottom like CSS — later rows override conflicting keys,
    # non-conflicting keys inherit). Each row is
    # ``{"match": <fnmatch wildcard on the model id, case-insensitive>, ...}``
    # with the param_rules vocabulary inline (strip, clamp, map, rename,
    # strip_override), ``effort_accept`` (the accepted reasoning-effort
    # vocabulary the ladder normalizes into) and ``toggle`` (the OFF
    # control rides the chat wire's thinking toggle), and ``allow``/``deny``
    # protocol lists (per-model face limiting — a face outside allow, or in
    # deny, is refused with an error naming the row). ``*`` is the
    # provider-default row. JSON
    # runtime config ``model_rules`` rows append after these (config
    # overrides code). Supersedes ``model_param_rules`` (kept as a bridge).
    model_rules: Tuple[Dict[str, Any], ...] = ()
    # Default transport base for native execution; env ``{PROVIDER}_API_BASE``
    # overrides. Providers without a stable public base (dynamic/config
    # defined) leave this None.
    default_api_base: Optional[str] = None
    # Multi-profile providers (D13): ``{name: {"protocol": ..., "endpoint_path": ...}}``.
    # Profiles share one provider identity (credentials, quota, accounting);
    # addressing is ``provider:profile/model``. Single-protocol providers
    # leave this None and keep the plain ``protocol_name`` declaration.
    transport_profiles: Optional[Dict[str, Dict[str, Any]]] = None
    default_profile: Optional[str] = None
    # Declarative cache-and-replay (W13/D14): list of rule entries compiled
    # to FieldCacheRules at native-context build (see field_cache/replay.py).
    # Each entry declares its own ``source`` (request | response | stream_event
    # | unified_request | unified_response | unified_stream_event), so a
    # provider can force-cache response state by declaring a response-watching
    # rule with an ``inject.target`` of request/response/unified_*.
    cache_replay: Optional[List[Dict[str, Any]]] = None

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch available models through the shared listing cascade.

        The default is protocol-aware and shared by every provider: every
        declared face that carries a listing descriptor is collected in
        cascade order (the provider's ``listing_profile`` hint first, then
        the global protocol priority list, then remaining declaration
        order) and tried in that order. A failed face logs a maintainer
        warning naming the face — the first warning points at
        ``listing_profile`` as the pin — and the cascade falls through to
        the next face. When every face fails, or none can list at all, an
        ERROR is logged and the honest empty is returned: no hardcoded
        fallbacks, no silent failure.

        ``listing_filters`` (when declared) excludes parsed ids before the
        provider prefix is added. Response shapes parse per the face's
        protocol descriptor (openai-family ``data[].id``, gemini/ollama
        ``models[].name`` with prefix stripping). Providers with genuinely
        different listings override this method and win.
        """

        from ..protocols.defaults import listing_descriptor

        logger = self._provider_logger()
        faces = self._listing_faces()
        if not faces:
            logger.error(
                "no speakable face has a listing descriptor for %s; returning empty model list",
                self._provider_config_key(),
            )
            return []
        last_error: Optional[Exception] = None
        for position, (face_name, protocol) in enumerate(faces):
            descriptor = listing_descriptor(protocol)
            try:
                # Endpoint + auth construction is part of trying the face:
                # a face whose declaration cannot resolve falls through to
                # the next one instead of aborting the cascade.
                endpoint = self.get_native_endpoint(operation="models", profile=face_name or None)
                headers = await self._listing_headers(api_key, protocol, profile=face_name or None)
                payload = await self._fetch_listing(client, endpoint, headers, descriptor)
            except Exception as exc:
                last_error = exc
                if position == 0:
                    logger.warning(
                        "model listing failed on primary face %r for %s: %s; "
                        "falling back — declare listing_profile to pin the right face",
                        face_name,
                        self._provider_config_key(),
                        exc,
                    )
                else:
                    logger.warning(
                        "model listing failed on fallback face %r for %s: %s",
                        face_name,
                        self._provider_config_key(),
                        exc,
                    )
                continue
            ids = self._parse_listing_payload(payload, descriptor.get("shape"))
            strip_prefix = descriptor.get("strip")
            if strip_prefix:
                ids = [model_id[len(strip_prefix):] if model_id.startswith(strip_prefix) else model_id for model_id in ids]
            filters = tuple(getattr(self, "listing_filters", None) or ())
            if filters:
                ids = [model_id for model_id in ids if not any(fnmatchcase(model_id, pattern) for pattern in filters)]
            prefix = f"{self._provider_config_key()}/"
            return [f"{prefix}{model_id}" for model_id in ids]
        logger.error(
            "model listing failed on all speakable faces for %s: %s; returning empty model list",
            self._provider_config_key(),
            last_error,
        )
        return []

    def _listing_faces(self) -> Tuple[Tuple[str, str], ...]:
        """Speakable faces that can list models, in cascade order.

        The ``listing_profile`` hint (a PROFILE name; a bare protocol name
        survives the legacy path) is pinned first so the endpoint and auth
        resolve exactly as the pinned face addresses them. Every other
        declared face follows per the global protocol priority list, then
        any remaining face in declaration order. Legacy single-protocol
        providers contribute their one implicit face under the empty name;
        faces whose protocol carries no listing descriptor never enter the
        cascade (the caller reports the honest empty).
        """

        from ..protocols.defaults import PROTOCOL_PRIORITY, listing_descriptor

        profiles = self._speaks_profiles()
        if profiles:
            names = [name for name in profiles if not name.startswith("__")]
            protocol_by_name: Dict[str, str] = {
                name: str(profiles[name].get("protocol") or "") for name in names
            }
        else:
            names = [""]
            protocol_by_name = {"": self.get_protocol_name() or "openai_chat"}
        hint = getattr(self, "listing_profile", None)
        ordered: List[str] = []
        if hint:
            hint_protocol = protocol_by_name.get(hint, "")
            if not hint_protocol and isinstance(self.transport_profiles, Mapping):
                entry = self.transport_profiles.get(hint)
                if isinstance(entry, Mapping):
                    hint_protocol = str(entry.get("protocol") or entry.get("protocol_name") or "")
            if not hint_protocol:
                # A bare protocol name survives the legacy path.
                hint_protocol = str(hint)
            ordered.append(hint)
            protocol_by_name[hint] = hint_protocol
        priority = {protocol: index for index, protocol in enumerate(PROTOCOL_PRIORITY)}
        remaining = [name for name in names if name != hint]
        remaining.sort(key=lambda name: priority.get(protocol_by_name.get(name, ""), len(PROTOCOL_PRIORITY)))
        ordered.extend(remaining)
        faces: List[Tuple[str, str]] = []
        for name in ordered:
            protocol = protocol_by_name.get(name, "")
            if protocol and listing_descriptor(protocol):
                faces.append((name, protocol))
        return tuple(faces)

    @staticmethod
    def _provider_logger():
        import logging

        return logging.getLogger("rotator_library")

    async def _listing_headers(self, api_key: str, listing_protocol: str, profile: Optional[str] = None) -> Dict[str, str]:
        """Credential headers for one listing face.

        The provider's own header logic wins when it exists (an
        authenticated Ollama behind a proxy sends its Bearer; the gemini
        faces send x-goog or bearer per face) — the protocol-default
        pair is only the fallback for duck-typed providers. The face's
        profile rides through so per-face auth declarations actually
        reach the listing request.
        """

        if hasattr(self, "get_native_headers"):
            method = self.get_native_headers
            try:
                if profile and _accepts_parameter(method, "profile"):
                    return method(api_key, operation="models", profile=profile)
                # A pre-D13 override without the profile parameter keeps its
                # original call shape (never a masked TypeError retry).
                return method(api_key, operation="models")
            except Exception:
                pass
        from ..protocols.defaults import default_auth_mode

        mode = default_auth_mode(listing_protocol) or "bearer"
        return auth_header_pair(api_key, mode, None, provider=self._provider_config_key())

    async def _fetch_listing(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        headers: Dict[str, str],
        descriptor: Optional[Dict[str, Any]],
    ) -> Any:
        """Fetch the listing payload, following page tokens when the
        protocol paginates (gemini's ListModels defaults to 50 per page
        — a single bare GET silently truncates large catalogs).

        Caps at 20 pages as a runaway guard; a failure mid-pagination
        returns the pages gathered so far (honest partial beats empty).
        """

        paginated = bool((descriptor or {}).get("paginated"))
        params: Dict[str, str] = {"pageSize": "1000"} if paginated else {}
        first = await client.get(endpoint, headers=headers, params=params or None)
        first.raise_for_status()
        payload = first.json()
        if not paginated:
            return payload
        merged = payload
        for _ in range(20):
            token = None
            if isinstance(payload, dict):
                token = payload.get("nextPageToken")
            if not token:
                break
            response = await client.get(endpoint, headers=headers, params={"pageSize": "1000", "pageToken": str(token)})
            response.raise_for_status()
            payload = response.json()
            if isinstance(merged, dict) and isinstance(payload, dict):
                models = list(merged.get("models") or []) + list(payload.get("models") or [])
                merged = dict(payload)
                merged["models"] = models
        return merged

    @staticmethod
    def _parse_listing_payload(payload: Any, shape: Optional[str]) -> List[str]:
        if not isinstance(payload, dict) or not shape:
            return []
        if shape == "data_id":
            data = payload.get("data")
            # Entries explicitly flagged inactive (e.g. Groq's deprecated
            # models carrying active:false) stay out of the pool.
            return [
                entry.get("id")
                for entry in data
                if isinstance(entry, dict) and entry.get("id") and entry.get("active", True)
            ]
        if shape == "models_name":
            data = payload.get("models")
            return [entry.get("name") for entry in data if isinstance(entry, dict) and entry.get("name")]
        return []

    # [NEW] Add methods for providers that need to bypass litellm
    def has_custom_logic(self) -> bool:
        """
        Returns True if the provider implements its own acompletion/aembedding logic,
        bypassing the standard litellm call.
        """
        return False

    def get_session_tracking_hints(
        self,
        request_data: Dict[str, Any],
        *,
        model: str = "",
    ) -> Optional[Any]:
        """Return provider-specific evidence for core session tracking.

        Providers can expose stable native markers, custom request structure, or
        a preferred affinity key before credential selection. Native evidence is
        provider-qualified and may use ``SessionTrackingHints.session_scope`` for
        model/family partitioning without fragmenting the global logical session.
        The return value is intentionally evidence-only: providers must not select
        credentials or mutate sticky state. Core routing merges these hints with
        generic anchors and applies the same confidence policy for every provider.

        Expansion path: providers that know their upstream cache/session scope can
        return a richer ``SessionTrackingHints`` object from
        ``rotator_library.session_tracking``. Returning ``None`` keeps the generic
        OpenAI-compatible tracker behavior.
        """
        return None

    def get_protocol_name(self, model: str = "", profile: Optional[str] = None) -> Optional[str]:
        """Return the native protocol adapter name this provider prefers.

        ``speaks`` (G8 envelope v2) is the declaration surface: profile
        names default to the protocol names, endpoints/auth inherit from
        the protocol registry, and entries override only what differs.
        Legacy ``transport_profiles``/``protocol_name`` declarations keep
        working (speaks wins when both are present) while providers
        migrate to the envelope.

        Multi-profile providers resolve per profile; single-protocol
        providers ignore the profile argument. Returning ``None`` keeps
        the LiteLLM fallback execution behavior. An explicitly requested
        profile always decides the dialect — a runtime ``protocol_name``
        override applies only to profile-less requests (a silent override
        of an explicit ``provider:profile`` address is forbidden).

        G8 model_rules face limiting: when the resolved model's
        capability-table rows refuse the face (outside ``allow``, or in
        ``deny``), resolution raises ``ModelRulesFaceError`` naming the
        deciding row instead of returning a protocol the model rejects.
        """

        protocol = self._declared_protocol_name(model, profile)
        if protocol:
            self._enforce_model_rules_faces(model, protocol)
        return protocol

    def _declared_protocol_name(self, model: str = "", profile: Optional[str] = None) -> Optional[str]:
        profiles = self._speaks_profiles()
        if profiles:
            from ..routing.profiles import ModelReferenceError

            if profile:
                entry = profiles.get(profile)
                if entry is None:
                    raise ModelReferenceError(
                        f"{self.__class__.__name__} has no profile {profile!r}; "
                        f"known: {sorted(k for k in profiles if not k.startswith('__'))}"
                    )
                return str(entry["protocol"])
            default = profiles.get("__default__") or {}
            return str(default.get("protocol")) if default.get("protocol") else None
        if self.transport_profiles and profile:
            entry = self.transport_profiles.get(profile)
            if not isinstance(entry, dict):
                from ..routing.profiles import ModelReferenceError

                raise ModelReferenceError(
                    f"{self.__class__.__name__} has no profile {profile!r}; "
                    f"known: {sorted(self.transport_profiles)}"
                )
            protocol = entry.get("protocol") or entry.get("protocol_name") or self.protocol_name
            return str(protocol) if protocol else None
        configured = self._get_runtime_config(model).protocol_name
        if configured:
            return configured
        if self.transport_profiles:
            from ..routing.profiles import ModelReferenceError

            if self.default_profile and self.default_profile not in self.transport_profiles:
                raise ModelReferenceError(
                    f"{self.__class__.__name__} declares default profile "
                    f"{self.default_profile!r} but no such profile exists; "
                    f"known: {sorted(self.transport_profiles)}"
                )
            entry = self.transport_profiles.get(self.default_profile or "") or {}
            protocol = entry.get("protocol") or self.protocol_name
            return str(protocol) if protocol else None
        return self.protocol_name

    def _speaks_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Resolved ``speaks`` profile table (cached per class)."""

        speaks = getattr(self, "speaks", None)
        if not speaks:
            return {}
        normalized = tuple(
            entry if isinstance(entry, str) else tuple(_hashable_value(part) for part in entry)
            for entry in speaks
        )
        return _resolved_speaks_profiles(normalized)

    def get_declared_profiles(self) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
        """Unified profile view for routing: ``(profiles, default_name)``.

        ``speaks`` translates into the transport-profiles shape (name →
        protocol) with the first entry as the default; legacy
        ``transport_profiles``/``default_profile`` remain the fallback
        while providers migrate.
        """

        speaks_profiles = self._speaks_profiles()
        if speaks_profiles:
            names = [name for name in speaks_profiles if not name.startswith("__")]
            view = {
                name: {"protocol": speaks_profiles[name]["protocol"], **({"endpoint_paths": speaks_profiles[name]["endpoint_paths"]} if speaks_profiles[name].get("endpoint_paths") else {})}
                for name in names
            }
            return view, (names[0] if names else None)
        return self.transport_profiles, self.default_profile

    def _model_rules_rows(self, model: str = "") -> Tuple[Dict[str, Any], ...]:
        """Capability-table rows for a model: class declaration + JSON config."""

        rows = tuple(self.model_rules or ())
        configured = getattr(self._get_runtime_config(model), "model_rules", None) or ()
        return rows + tuple(configured)

    def _enforce_model_rules_faces(self, model: str, protocol: str) -> None:
        """Refuse a face the model's capability-table rows limit away."""

        if not model:
            return
        rows = self._model_rules_rows(model)
        if not rows:
            return
        from ..adapters.param_rules import enforce_model_rules_faces

        enforce_model_rules_faces(rows, model, protocol, provider=self._provider_config_key())

    def get_adapter_names(self, model: str = "") -> Tuple[str, ...]:
        """Return ordered adapter names for this provider/model.

        The order is significant and is preserved by the adapter chain
        runner. Providers can override this for model-specific quirks
        without mutating the global adapter registry.

        The param-rule engine is ALWAYS present: it is prepended unless a
        declared adapter already builds on it (an adapter whose class
        sets ``consumes_param_rules`` — e.g. the mistral wire adapter
        extends the engine for its think-chunk folding). Providers never
        declare it by name; with no resolved rules it is an identity
        no-op.
        """

        configured = self._get_runtime_config(model).adapter_names
        declared = tuple(configured) if configured is not None else tuple(self.adapter_names)
        if self._chain_has_param_consumer(declared):
            return declared
        if "param_rules" in declared:
            return declared
        return ("param_rules",) + declared

    def _chain_has_param_consumer(self, declared: Tuple[str, ...]) -> bool:
        """Whether any declared chain entry builds on the param engine."""

        try:
            from ..adapters.registry import get_adapter

            for name in declared:
                try:
                    adapter_class = get_adapter(name)
                except Exception:
                    continue
                if getattr(adapter_class, "consumes_param_rules", False):
                    return True
        except Exception:
            return False
        return False

    def get_adapter_config(self, model: str = "") -> Dict[str, Dict[str, Any]]:
        """Return adapter-specific config keyed by adapter name.

        Config is intentionally a plain dict so custom providers can define it in
        env/JSON later without importing adapter classes.

        Every chain entry whose adapter class declares
        ``consumes_param_rules`` receives the merged provider/model
        parameter-rule tables under its own key — matched by
        consumption, not by adapter name, so wire adapters that extend
        the engine (mistral) are fed exactly like the generic stage.
        """

        config = dict(self._get_runtime_config(model).adapter_config)
        adapter_names = self.get_adapter_names(model)
        from ..adapters.param_rules import declared_param_rules

        runtime_rows = getattr(self._get_runtime_config(model), "model_rules", None) or ()
        rules = declared_param_rules(
            self,
            model,
            {"model_rules": list(runtime_rows)} if runtime_rows else None,
        )
        if rules:
            for name in adapter_names:
                if name in config:
                    continue
                try:
                    from ..adapters.registry import get_adapter

                    adapter_class = get_adapter(name)
                except Exception:
                    continue
                if getattr(adapter_class, "consumes_param_rules", False):
                    # Resolved tables (provider+model merged) — the
                    # adapter's own resolution pass is idempotent over
                    # them.
                    config[name] = rules
        return config

    def get_hooks(self, model: str = "") -> Tuple[Any, ...]:
        """Return ordered hook declarations for this provider/model.

        Class-declared ``hooks`` are the base; JSON runtime config ``hooks``
        (validated at startup) are appended. Global registry names are resolved
        separately when the per-request run is minted. Order is significant —
        declaration order breaks priority ties (hooks/registry.py).
        """

        configured = self._get_runtime_config(model).hooks
        return tuple(self.hooks) + tuple(configured or ())

    def get_field_cache_rules(self, model: str = "") -> Tuple[Any, ...]:
        """Return field-cache rules for provider-specific protocol state.

        Rules preserve provider state such as reasoning content, thought
        signatures, prompt cache keys, provider session IDs, and response IDs.
        They are not a replacement for ``SessionTracker``; session tracking still
        decides continuity and credential affinity.
        """

        configured = self._get_runtime_config(model).field_cache_rules
        return tuple(self.field_cache_rules) + tuple(configured)

    def supports_native_streaming(self, model: str = "", operation: str = "chat") -> bool:
        """Return whether this provider explicitly supports native streaming.

        The default is intentionally false. Phase 8 keeps live streaming
        conservative: providers must opt in before routed streaming can use the
        native stream executor instead of current custom/LiteLLM behavior.
        """

        configured = self._get_runtime_config(model).native_streaming_supported
        if configured is None:
            return self.native_streaming_supported
        return bool(configured and self.supports_native_operation(model, operation))

    def _get_runtime_config(self, model: str = "") -> Any:
        """Return optional JSON runtime config for this provider.

        The helper keeps config loading lazy so provider imports do not depend on
        the experimental config layer during startup discovery.
        """

        from ..config.experimental import get_provider_runtime_config

        return get_provider_runtime_config(
            self._provider_config_key(),
            model,
            config=getattr(self, "_runtime_config_snapshot", None),
        )

    def bind_runtime_config(self, config: Any) -> None:
        """Bind the immutable process-start configuration used by this instance."""

        existing = getattr(self, "_runtime_config_snapshot", None)
        if existing is not None and existing != config:
            raise RuntimeError(
                f"Provider singleton {self._provider_config_key()!r} is already bound to a different startup configuration"
            )
        self._runtime_config_snapshot = config

    def _provider_config_key(self) -> str:
        """Return the JSON providers-section key for this provider."""

        if self.provider_env_name:
            key = self.provider_env_name.lower()
        else:
            name = self.__class__.__name__
            if name.endswith("Provider"):
                name = name[: -len("Provider")]
            key = name.lower()
        # Registry remap (nvidia registers as nvidia_nim): one identity per
        # provider across registry, credentials, and the JSON section.
        remap = getattr(self.__class__, "config_key_alias", None) or getattr(
            self, "config_key_alias", None
        )
        return remap if remap else key

    def supports_native_operation(self, model: str = "", operation: str = "chat", profile: Optional[str] = None) -> bool:
        """Return whether this provider supports a native operation.

        The check resolves the PROFILE's protocol (D13), not the default —
        an anthropic_messages profile must not be gated by chat vocabulary.
        """

        protocol_name = self.get_protocol_name(model, profile=profile) if profile else self.get_protocol_name(model)
        if not protocol_name:
            return False
        try:
            from ..protocols import get_protocol

            return get_protocol(protocol_name).supports_operation(operation)
        except Exception:
            return False

    def get_native_operation(self, model: str = "", request=None, stream: bool = False, profile: Optional[str] = None) -> str:
        """Return the provider-native operation for a request.

        Providers that expose native protocols often use operation names that
        are not simply ``chat``: Anthropic-compatible providers use ``messages``,
        Responses providers use ``responses``, and Gemini-style providers use a
        generate operation. The default maps the resolved profile protocol
        (D13) to its conventional operation; single-protocol providers keep
        ``chat`` unless they override.
        """

        try:
            protocol_name = self.get_protocol_name(model, profile=profile) if profile else self.get_protocol_name(model)
        except Exception:
            protocol_name = None
        if protocol_name == "anthropic_messages":
            return "messages"
        from ..protocols.canonical import family_wire_name

        wire = family_wire_name(protocol_name or "")
        if wire == "responses":
            return "responses"
        if wire == "gemini":
            return "stream_generate" if stream else "generate"
        if protocol_name == "ollama":
            # Both /api/chat and /api/generate stream via the body `stream`
            # flag, so the operation does not fork on the stream bit; chat is
            # the canonical shape for conversational requests.
            request = request if isinstance(request, dict) else {}
            if "prompt" in request and "messages" not in request:
                return "ollama_generate"
            return "ollama_chat"
        return "chat"

    def should_use_native_protocol(self, model: str = "", operation: str = "chat", *, stream: bool = False, execution: str = "auto") -> bool:
        """Return whether routing should use this provider's native protocol."""

        if stream and not self.supports_native_streaming(model, operation):
            return False
        return bool(self.get_protocol_name(model) and self.supports_native_operation(model, operation))

    def get_native_endpoint(self, model: str = "", operation: str = "chat", profile: Optional[str] = None) -> str:
        """Return the upstream endpoint for a native operation.

        The default derives from the transport base plus the selected
        profile's endpoint declarations (D13 / plan 2.8): the plural
        ``endpoint_paths`` map keyed by operation wins, the singular
        ``endpoint_path`` is the fallback, and ``{model}``/``{operation}``
        placeholders are rendered here. The profile's declarations win over
        provider-level JSON declarations; profiles without an explicit path
        get the per-protocol conventional path. Single-protocol providers
        that did not override fall through to the protocol's default path.
        """

        entry: Optional[Any] = None
        speaks_profiles = self._speaks_profiles()
        speaks_base: Optional[str] = None
        if speaks_profiles:
            resolved = speaks_profiles.get(profile or "") or speaks_profiles.get("__default__") or {}
            entry = {"protocol": resolved.get("protocol"), "endpoint_paths": resolved.get("endpoint_paths")}
            speaks_base = resolved.get("base")
        elif self.transport_profiles and profile:
            entry = self.transport_profiles.get(profile) or {}
        path = declared_endpoint_path(entry, operation)
        if not path:
            path = declared_endpoint_path(self._get_runtime_config(model), operation)
        if not path:
            protocol = ""
            if isinstance(entry, Mapping):
                protocol = str(entry.get("protocol") or entry.get("protocol_name") or "")
            if not protocol:
                protocol = self.get_protocol_name(model, profile=profile) if profile else self.get_protocol_name(model)
            from ..protocols.defaults import default_endpoint_paths

            inherited = default_endpoint_paths(protocol or "")
            path = inherited.get(operation) or self._default_endpoint_path(protocol or "", operation)
        base = speaks_base or self.get_provider_api_base()
        if not base:
            raise NotImplementedError(
                f"{self.__class__.__name__} has no transport base; "
                "set default_api_base or {PROVIDER}_API_BASE"
            )
        return f"{base}{render_endpoint_path(path, model=self.normalize_native_model(model), operation=operation, provider=self._provider_config_key())}"

    def _default_endpoint_path(self, protocol: str = "", operation: str = "chat") -> str:
        """Conventional per-protocol endpoint path (profiles without one).

        Honest by operation: a protocol only answers for the operations it
        declares (plus the listing pseudo-operation ``models``); anything
        else raises rather than silently returning the chat path.
        """

        from ..protocols.canonical import family_wire_name

        if operation == "models":
            return "/api/tags" if protocol == "ollama" else "/models"
        wire = family_wire_name(protocol or "")
        if protocol == "gemini" and operation not in ("embeddings", "embeddings_batch"):
            raise NotImplementedError(
                "Gemini endpoints are model-ridden; declare endpoint_paths for gemini profiles"
            )
        if protocol == "ollama":
            paths = {
                "chat": "/api/chat",
                "ollama_chat": "/api/chat",
                "ollama_generate": "/api/generate",
                "embeddings": "/api/embed",
            }
        elif protocol == "gemini":
            paths = {
                "embeddings": "/v1beta/models/{model}:embedContent",
                "embeddings_batch": "/v1beta/models/{model}:batchEmbedContents",
            }
        elif protocol == "anthropic_messages":
            paths = {
                "chat": "/v1/messages",
                "messages": "/v1/messages",
                "count_tokens": "/v1/messages/count_tokens",
            }
        elif wire == "responses":
            paths = {"chat": "/responses", "responses": "/responses"}
        else:
            # The openai-compatible convention: chat and embeddings on
            # their standard paths (G9 — embeddings are an operation on
            # the chat wire, not a separate transport).
            paths = {"chat": "/chat/completions", "embeddings": "/embeddings"}
        path = paths.get(operation)
        if path is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} has no default endpoint for "
                f"{protocol or 'unknown'!r} operation {operation!r}"
            )
        return path

    def get_native_headers(self, credential_identifier: str, model: str = "", operation: str = "chat", profile: Optional[str] = None) -> Dict[str, str]:
        """Return non-payload HTTP headers for native requests.

        The default covers bearer-token OpenAI-compatible providers
        (native-by-default, W11); providers with different auth schemes
        (e.g. Gemini's ``x-goog-api-key``) override this. Per-profile auth
        declarations (``auth_mode``/``auth_header_name``) override
        provider-level JSON declarations when the profile declares them.
        """

        auth_mode: Optional[str] = None
        auth_header_name: Optional[str] = None
        speaks_profiles = self._speaks_profiles()
        if speaks_profiles:
            resolved = speaks_profiles.get(profile or "") or speaks_profiles.get("__default__") or {}
            auth_mode = resolved.get("auth_mode")
            auth_header_name = resolved.get("auth_header_name")
        if auth_mode is None and self.transport_profiles and profile:
            entry = self.transport_profiles.get(profile)
            if isinstance(entry, Mapping):
                auth_mode = entry.get("auth_mode")
                auth_header_name = entry.get("auth_header_name")
        runtime = self._get_runtime_config(model)
        if auth_mode is None:
            auth_mode = getattr(runtime, "auth_mode", None)
        if auth_header_name is None:
            auth_header_name = getattr(runtime, "auth_header_name", None)
        return auth_header_pair(
            credential_identifier,
            auth_mode,
            auth_header_name,
            provider=self._provider_config_key(),
        )

    def get_provider_api_base(self) -> Optional[str]:
        """Return this provider's transport base URL.

        Resolution order: ``{PROVIDER}_API_BASE`` env override, then the
        class-declared ``default_api_base``. Provider transport identity is
        canonical in the provider class (never duplicated elsewhere).
        """

        if self.default_api_base is None:
            return None
        env_key = self._plugin_key()
        override = os.getenv(f"{env_key.upper()}_API_BASE") if env_key else None
        return (override or self.default_api_base).rstrip("/")

    def _plugin_key(self) -> Optional[str]:
        """Return this provider's registry key in PROVIDER_PLUGINS."""

        cached = getattr(self, "_plugin_key_cache", None)
        if cached is not None:
            return cached or None
        from . import PROVIDER_PLUGINS

        key = None
        for name, plugin_class in PROVIDER_PLUGINS.items():
            if type(self) is plugin_class:
                key = name
                break
        try:
            self._plugin_key_cache = key or ""
        except AttributeError:
            pass
        return key

    def normalize_native_model(self, model: str, profile: Optional[str] = None) -> str:
        """Return the upstream model name for native provider calls.

        The proxy-facing model commonly includes a provider prefix such as
        ``provider/model``. Native upstream APIs usually expect only ``model``.
        Providers may override this for aliases, but stripping the first prefix
        is the safe default for native execution.

        ``profile`` is the resolved transport face (None for single-face
        providers or bare addressing). Overrides that normalize per face
        (an id spelling only one face expects) declare the parameter;
        callers pass it only when the override accepts it, so pre-D13
        single-argument overrides keep working.
        """

        return model.split("/", 1)[1] if "/" in model else model

    def prepare_native_request(self, request: Dict[str, Any], model: str = "", operation: str = "") -> Dict[str, Any]:
        """Return a provider-adjusted native request payload.

        The declared provider protocol has already built a valid native payload
        before this hook runs. Implementations may add provider envelopes,
        aliases, or required defaults, but must never translate a client protocol.
        Credentials remain in ``get_native_headers()`` so payload traces never
        mix request data with secrets.
        """

        return dict(request)

    def get_model_pricing(self, model: str = "") -> Optional[Any]:
        """Return optional local pricing metadata for advisory cost tracking.

        Providers can return `usage.costs.ModelPricing` or a compatible dict.
        The default is `None`, which lets cost accounting safely fall back to
        LiteLLM model metadata or report pricing as unavailable.
        """

        return None

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[
        litellm.ModelResponse,
        AsyncGenerator[litellm.ModelResponseStream, None],
    ]:
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

    # convert_safety_settings() removed — see gemini_provider.py for details.

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
            A provider may restrict high-capability models to priority 1
            credentials while allowing lower-cost models for all priorities.
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
            provider_name: The provider name (e.g., "openai")

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
        configured_groups = self._get_runtime_config().model_quota_groups or {}
        base_groups: QuotaGroupMap = {group: list(models) for group, models in self.model_quota_groups.items()}
        for group, models in configured_groups.items():
            base_groups[group] = list(models)
        if not self.provider_env_name:
            return base_groups

        result: QuotaGroupMap = {}

        for group_name, default_models in base_groups.items():
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
