# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Reference provider template — the living documentation for adding a provider.

THIS MODULE IS EXCLUDED FROM AUTO-REGISTRATION.

Provider discovery imports every module in this package whose filename does
*not* start with an underscore (see ``providers/__init__.py`` ->
``_register_providers``). Because this file is ``_example_provider.py`` it is
never imported at runtime and its class is never inserted into
``PROVIDER_PLUGINS``. It exists purely as documentation: the code below is
valid, compiles, and would run if the file were renamed to
``example_provider.py`` — but nothing here executes in a normal deployment.
Copy it, rename it, and delete the parts you do not need.

===============================================================================
WHAT A PROVIDER IS
===============================================================================

A provider is an IDENTITY PLUS DECLARATIONS — never a translator.

Identity is the single name by which the proxy addresses the upstream
(``openrouter``, ``gemini``, ``myserver``). A request names a provider and a
model, optionally with a transport profile and an execution hint::

    provider/model                    e.g.  openai/gpt-5.1
    provider:profile/model            e.g.  gemini:openai/gemini-2.5-pro
    provider/model@execution          e.g.  codex/gpt-5.1-codex@native

The *profile* selects which transport face of a multi-face provider is used.
The *execution* suffix selects how the call is dispatched (``@custom``,
``@native``, ``@litellm_fallback``, ``@auto``). Neither changes identity:
usage pools, cooldowns, classifiers, session namespaces, cache provenance, and
quota accounting all key on the bare provider name. A profile only steers
transport (see ``routing/profiles.py``).

Declarations are the class attributes and method overrides that tell the
proxy what the provider can do: which native protocol it speaks, where its
endpoints live, how it authenticates, what provider state to cache and replay,
which adapters shape payloads, which hooks fire, how models group for quota,
and — optionally — how requests are counted. The framework does the rest.

A provider MUST NOT:

* translate between client and provider wire protocols — the protocol
  adapters in ``rotator_library.protocols`` own wire translation;
* put credentials into request payloads — credentials travel only through
  ``get_native_headers()`` so transaction traces never mix data with secrets;
* mutate global registries at request time — declarations are frozen and
  validated at startup (``validate_provider_hooks``).

The one hard rule: identity + declarations in, execution out. If you find
yourself writing request/response field mapping here, it belongs in a
protocol adapter or an adapter class instead.

===============================================================================
TWO WAYS TO CREATE A PROVIDER
===============================================================================

1. CODE PROVIDER (this file). Subclass ``ProviderInterface`` in a module named
   ``<name>_provider.py``. Discovery imports the module and registers the class
   under ``<name>`` (module name with ``_provider`` stripped). Set
   ``config_key_alias`` when the registry key must differ from the module name
   (``nvidia`` registers as ``nvidia_nim``). Code providers can implement fully
   custom execution via ``acompletion()``.

2. DYNAMIC PROVIDER (``providers/dynamic.py``). Created from configuration for
   any upstream that needs no custom code. ``<NAME>_API_BASE`` plus
   ``<NAME>_API_KEY`` is the minimal env form; the JSON ``providers`` section
   carries the full surface (``protocol_name``, ``endpoint_paths``,
   per-profile auth, ``cache_replay``, adapters, models, quota groups). The
   generic native runtime executes whatever protocol the declaration names.

The declaration surfaces below are shared: almost every class attribute has a
JSON-config equivalent, so a code provider can usually be reproduced as
configuration.

===============================================================================
REQUEST LIFECYCLE (WHO DOES WHAT)
===============================================================================

::

    client request        provider[:profile]/model[@execution]
          |
          v
    routing/profiles.py   resolve profile -> protocol + endpoint
          |
          v
    client/executor.py    resolve execution mode:
          |                 custom  -> provider.acompletion()  (this file)
          |                 native  -> protocols/<dialect> builds the wire body
          |                 litellm -> LiteLLM fallback (default if undeclared)
          v
    adapters/*            ordered payload adapters (model_override, ...)
          |
          v
    field_cache/*         inject cached provider state (reasoning, thought
          |                 signatures, prompt-cache keys, continuation ids)
          v
    hooks/*               declared pipeline stages (request/response/stream)
          |
          v
    HTTP send             endpoint + auth headers from the provider class
          |
          v
    response -> protocol parse -> client format
          |
          v
    usage/*               usage manager update; on_request_complete() hook

===============================================================================
DECLARATION SURFACES (QUICK MAP)
===============================================================================

Identity
    provider_env_name            env prefix and provider-config key ("EXAMPLE")
    config_key_alias             optional registry remap (one identity)

Transport / protocol
    protocol_name                native dialect, or None for LiteLLM fallback
    transport_profiles           multi-face providers: {profile: {protocol,...}}
    default_profile              profile used for bare ``provider/model``
    default_api_base             base URL; ``<PROVIDER>_API_BASE`` overrides
    native_streaming_supported   opt in to native streaming execution

Auth
    default_auth_mode            "none" for zero-credential (local) providers

Payload shaping
    adapter_names                ordered adapter list (order matters)
    get_adapter_names()          per-model override of the adapter chain
    get_adapter_config()         adapter config (JSON: provider.adapter_config)
    prepare_native_request()     last provider-owned payload adjustment

State preservation
    field_cache_rules            FieldCacheRule tuple (code declaration)
    cache_replay                 declarative rules (class attr; compiled)

Pipeline
    hooks                        PipelineHook declarations; JSON appends

Discovery
    get_models()                 REQUIRED: return provider-prefixed model names

Sessions
    get_session_tracking_hints() provider-specific conversation evidence

Quota / usage
    model_quota_groups, model_usage_weights, default_custom_caps,
    tier_priorities, usage_reset_configs, usage_window_definitions,
    default_rotation_mode, default_priority_multipliers, ...
    on_request_complete()        count / cooldown override hook
    get_background_job_config()  periodic quota-refresh job

===============================================================================
PRECEDENCE (WHO WINS)
===============================================================================

Per provider/model, configuration layers. Later layers win::

    code class attribute  <  JSON providers.<name>  <  environment variable

Environment overrides verified in source:

    <PROVIDER>_API_BASE       override default_api_base (code provider) or
                              supply api_base (dynamic provider)
    <PROVIDER>_API_KEY[_N]    credentials; numbered keys rotate independently
    <PROVIDER>_MODELS         model definitions (id aliases, default options)
    <PROVIDER>_CACHE_REPLAY   JSON list of cache_replay rules
    QUOTA_GROUPS_<PROVIDER>_<GROUP>   override a quota group's model list
    ROTATION_MODE_<PROVIDER>, FAIR_CYCLE_*, CUSTOM_CAP_*, ... usage tuning

The JSON config file is selected with ``LLM_PROXY_CONFIG_FILE`` or
``PROXY_CONFIG_FILE`` and must never contain credentials.

There are NO ``<PROVIDER>_PROTOCOL``, ``<PROVIDER>_AUTH_MODE``,
``<PROVIDER>_ENDPOINT_*``, or ``<PROVIDER>_CONFIG`` environment variables in
this codebase. For a dynamic provider, protocol, auth, endpoint paths,
profiles, adapters, and hooks are declared in the JSON ``providers`` section.

===============================================================================
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .provider_interface import ProviderInterface, QuotaGroupMap
from ..core.types import RequestCompleteResult
from ..hooks.types import HookContext, PipelineHook, StageInvocation
from ..session_tracking import SessionTrackingHints
from ..usage import UsageManager

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# INTERNAL RETRY COUNTING (ContextVar PATTERN)
# =============================================================================
#
# When your provider performs internal retries (transient errors, empty
# responses, rate limits), each retry is an upstream API call that should be
# counted for accurate usage tracking.
#
# Instance attributes (``self.count``) are unsafe here: a provider is a
# process-wide singleton (``SingletonABCMeta``), so concurrent requests would
# clobber each other's counters. ``ContextVar`` gives every async task its own
# isolated value.
#
# Pattern:
#   1. set(1) at the start of your retry loop;
#   2. set(get() + 1) before each retry;
#   3. read it in ``on_request_complete`` and return
#      ``RequestCompleteResult(count_override=...)``.
_example_attempt_count: ContextVar[int] = ContextVar(
    "example_provider_attempt_count", default=1
)


# =============================================================================
# A DECLARED PIPELINE HOOK
# =============================================================================
#
# ``hooks`` is the G2 pipeline extension point. The executor pauses at declared
# stages, hands over the live payload, and continues with whatever the hook
# returns. A hook can rewrite the payload, block the request, short-circuit a
# response, or drop/replace stream events. A callback (subclass
# ``PipelineCallback``) is a listener that runs after a slot settles and cannot
# change flow.
#
# Hooks are declared here on the class, in JSON ``providers.<name>.hooks``, or
# in the global registry. Order is significant: priority ascending, ties broken
# class -> config -> global. Every referenced name/stage is validated at
# startup (``providers/__init__.py`` -> ``validate_provider_hooks``), never per
# request.
#
# Danger note: a hook has FULL read/write power over the stage payload. Use a
# callback unless mutation is actually required.
class ExampleRequestObserver(PipelineHook):
    """Minimal observer hook: logs request entry, changes nothing.

    This is a real, import-safe hook class. If this module were registered it
    would be validated at startup and bound here for ``request_received``.
    Return ``None`` to continue with the payload unchanged.
    """

    name = "example_request_observer"
    stages: Tuple[str, ...] = ("request_received",)
    priority = 200  # late; smaller numbers fire first (default is 100)
    critical = False  # a failure here is contained (logged, not fatal)
    stateful = False  # per-request instance? False keeps one shared instance

    async def __call__(
        self, invocation: StageInvocation, context: HookContext
    ) -> None:
        lib_logger.debug(
            "example hook saw %s for %s/%s",
            invocation.stage,
            context.provider,
            context.model,
        )
        return None


# =============================================================================
# CODE PROVIDER
# =============================================================================


class ExampleProvider(ProviderInterface):
    """A fully-declared native provider, annotated as the reference template.

    Read top to bottom: identity, protocol/transport, payload shaping, state
    preservation, pipeline hooks, auth, discovery, sessions, and quota/usage.
    Every declaration is real code that the runtime would honor if this class
    were registered.
    """

    # =========================================================================
    # IDENTITY
    # =========================================================================
    #
    # The registry key is derived from the module filename at import time:
    # ``example_provider.py`` -> ``"example"``. That key is what model
    # references, the JSON ``providers`` section, usage files, and session
    # namespaces all use. Do not invent a second name.
    #
    # ``provider_env_name`` is the env-var prefix used by helpers such as
    # ``QUOTA_GROUPS_EXAMPLE_...`` and the provider-config lookup. The
    # interface convention is the uppercase registry key.
    provider_env_name = "EXAMPLE"

    # Optional: force the config/registry key when the module name differs.
    # NvidiaProvider does this because LiteLLM calls the provider
    # ``nvidia_nim`` while the module is ``nvidia_provider``. When set, one
    # identity is used across registry, credentials, and JSON config.
    # config_key_alias = "example_vendor"

    # Skip advisory cost accounting when the upstream does not report cost and
    # no reliable local pricing exists. Usage tokens are still tracked.
    skip_cost_calculation = True

    # =========================================================================
    # TRANSPORT / PROTOCOL DECLARATIONS
    # =========================================================================
    #
    # ``protocol_name`` names the NATIVE wire dialect the provider speaks:
    # ``openai_chat``, ``responses``, ``anthropic_messages``, ``gemini``, or
    # ``ollama`` (any registered protocol that declares a generative
    # operation). Declaring it flips execution to native-by-default; the
    # executor builds the wire body with the protocol adapter, not LiteLLM.
    #
    # ``None`` (the interface default) keeps the LiteLLM-backed path. A
    # dynamic provider with no declared protocol defaults to ``openai_chat``.
    protocol_name = "openai_chat"

    # ``transport_profiles`` declares multiple transport faces that share ONE
    # provider identity. Each profile has a ``protocol`` and may override
    # ``endpoint_paths`` and ``auth_mode``/``auth_header_name``. Addressing is
    # ``provider:profile/model``; a bare ``provider/model`` uses
    # ``default_profile`` (or the unique profile matching the client protocol).
    #
    # GeminiProvider is the canonical example: a native gemini face and a
    # Google OpenAI-compatibility face behind one identity::
    #
    #     default_profile = "native"
    #     transport_profiles = {
    #         "native": {"protocol": "gemini"},
    #         "openai": {
    #             "protocol": "openai_chat",
    #             "endpoint_paths": {
    #                 "chat": "/v1beta/openai/chat/completions",
    #                 "models": "/v1beta/openai/models",
    #             },
    #             "auth_mode": "bearer",
    #         },
    #     }
    #
    # Single-protocol providers leave this ``None`` and keep ``protocol_name``.
    transport_profiles: Optional[Dict[str, Dict[str, Any]]] = None
    default_profile: Optional[str] = None

    # Transport base. ``<PROVIDER>_API_BASE`` overrides it at runtime (the
    # registry key, not ``provider_env_name``, drives the env lookup). Code
    # providers that leave this ``None`` are not overridable this way; dynamic
    # providers read ``<NAME>_API_BASE`` directly.
    default_api_base = "https://api.example-vendor.example/v1"

    # Opt in before routed streaming can use the native stream executor.
    # The default is deliberately conservative (False).
    native_streaming_supported = True

    # When your endpoint shape is not the profile/JSON convention, override
    # ``get_native_endpoint`` (the base implementation raises loudly for
    # single-protocol providers with no declared path). This is exactly what
    # OpenAIProvider, NvidiaProvider, and friends do.
    def get_native_endpoint(
        self, model: str = "", operation: str = "chat", profile: Optional[str] = None
    ) -> str:
        """Return the upstream URL for a native operation.

        The base class resolves a profile's ``endpoint_paths`` (or JSON
        declarations) and otherwise raises ``NotImplementedError`` for a
        single-protocol provider. Override only when your path is fixed or
        computed. ``operation`` is dialect-specific: ``chat`` for
        openai_chat, ``responses`` for the Responses API, ``messages`` for
        Anthropic, ``generate``/``stream_generate`` for Gemini, and
        ``ollama_chat``/``ollama_generate`` for Ollama.
        """
        base = self.get_provider_api_base()
        return f"{base}/chat/completions"

    def get_native_headers(
        self,
        credential_identifier: str,
        model: str = "",
        operation: str = "chat",
        profile: Optional[str] = None,
    ) -> Dict[str, str]:
        """Return non-payload HTTP headers for native requests.

        The default covers bearer-token OpenAI-compatible providers. Override
        for other schemes (Gemini uses ``x-goog-api-key``; Anthropic uses
        ``x-api-key``). Per-profile auth declarations win over provider-level
        JSON declarations. Credentials live here so payload traces stay clean.
        """
        return {"Authorization": f"Bearer {credential_identifier}"}

    def normalize_native_model(self, model: str) -> str:
        """Map a proxy-facing model name to the upstream model id.

        The proxy commonly addresses ``provider/model``; upstream APIs usually
        want the bare ``model``. The base strips the first prefix. Override for
        aliases, but never for protocol translation.
        """
        return model.split("/", 1)[1] if "/" in model else model

    def prepare_native_request(
        self, request: Dict[str, Any], model: str = "", operation: str = ""
    ) -> Dict[str, Any]:
        """Final provider-owned adjustment to a native payload.

        The declared protocol has already produced a valid native body here.
        Add required defaults or provider envelopes — but never translate a
        client protocol. Credentials are not available in this payload by
        design; put them in ``get_native_headers``.
        """
        payload = dict(request)
        payload.setdefault("temperature", 1)
        return payload

    # =========================================================================
    # AUTH DECLARATIONS
    # =========================================================================
    #
    # Auth modes accepted by ``auth_header_pair`` / JSON validation:
    #   bearer          -> Authorization: Bearer <credential>   (default)
    #   x-api-key       -> x-api-key: <credential>
    #   x-goog-api-key  -> x-goog-api-key: <credential>
    #   custom          -> <auth_header_name>: <credential>
    #   none            -> no header at all
    #
    # ``default_auth_mode`` exists so a zero-credential provider (a local
    # Ollama) can declare it and have routing mint the internal no-auth
    # rotation slot. Dynamic providers mint that slot automatically when no
    # credential env and no explicit auth declaration exist.
    default_auth_mode: Optional[str] = None

    # =========================================================================
    # PAYLOAD SHAPING (ADAPTERS)
    # =========================================================================
    #
    # Adapters run in declared order between the protocol build and the send.
    # Built-ins (``adapters/builtin.py``): ``noop``, ``model_override``,
    # ``suppress_developer_role``, ``reasoning_content``, ``field_rename``,
    # ``antigravity_envelope``. Order is significant — envelope adapters must
    # be LAST because they wrap everything before them.
    #
    # Adapter *configuration* is not a class attribute today: it comes from
    # JSON ``providers.<name>.adapter_config`` (or a ``get_adapter_config``
    # override). See ``get_adapter_names`` / ``get_adapter_config`` below.
    adapter_names: Tuple[str, ...] = ("suppress_developer_role",)

    def get_adapter_names(self, model: str = "") -> Tuple[str, ...]:
        """Return the ordered adapter chain for this model.

        JSON ``adapter_names`` (per provider and per model) wins over the class
        attribute. Order is preserved by the adapter chain runner; override here
        for model-specific quirks without touching the global registry.
        """
        return super().get_adapter_names(model)

    def get_adapter_config(self, model: str = "") -> Dict[str, Dict[str, Any]]:
        """Return adapter config keyed by adapter name.

        The base implementation returns the JSON runtime config
        (``providers.<name>.adapter_config``). There is no class-level
        ``adapter_config`` attribute; override this method if you need to
        compute config in code. Example JSON::

            "adapter_config": {
                "suppress_developer_role": {"mode": "system"}
            }
        """
        return super().get_adapter_config(model)

    # =========================================================================
    # STATE PRESERVATION (FIELD CACHE / CACHE-AND-REPLAY)
    # =========================================================================
    #
    # Field-cache rules preserve provider state across turns: reasoning
    # content, thought signatures, prompt-cache keys, provider session ids,
    # response ids. They are NOT session affinity — session tracking still
    # decides continuity and credential stickiness.
    #
    # Two declaration surfaces:
    #
    #   field_cache_rules   tuple of ``FieldCacheRule`` objects (code only)
    #   cache_replay        list of declarative dicts (code class attr, and the
    #                       shape used by the ``<NAME>_CACHE_REPLAY`` env var)
    #
    # Both compile to the same engine. Extraction/injection run on the NATIVE
    # execution path only; a custom ``acompletion`` or the LiteLLM fallback
    # neither caches nor injects.
    #
    # ``cache_replay`` entry schema (``field_cache/replay.py``):
    #   name           unique, filesystem-safe rule name
    #   source         request | response | stream_event | unified_request |
    #                  unified_response | unified_stream_event (default response)
    #   path           extraction path (dotted, ``[n]`` indexes, ``*`` wildcard)
    #   keep           last | all | turn | turns:N | per_tool_call
    #   inject.path    restore path
    #   inject.if      auto (only when absent, default) | always (overwrite)
    #   inject.target  request | unified_request | metadata | response |
    #                  unified_response (default request)
    #   inject.insert / inject.as_list   list-tail insertion / always-list
    #   compatibility  bound (default) | portable
    #   transform      registered transform name (portable only)
    #   scope          scope dimensions; provider+model are always required
    #   critical       fail-closed escape hatch (default False: log+skip)
    #   ttl_seconds    retention window
    #   tool_call_id_path   required for ``keep: per_tool_call``
    #
    # Precedence per rule name:
    #   JSON ``field_cache`` > env ``<NAME>_CACHE_REPLAY`` >
    #   class ``cache_replay`` > provider-declared ``field_cache_rules``.
    #
    # NOTE: the JSON provider key ``cache_replay`` is accepted by validation
    # but is not consumed by the runtime-config loader; JSON users express
    # rules with ``field_cache`` (see docs/examples/README.md).
    field_cache_rules: Tuple[Any, ...] = ()

    cache_replay: List[Dict[str, Any]] = [
        {
            # Cache the upstream response id and replay it as the next
            # request's prompt-cache key.
            "name": "prompt_cache_key",
            "source": "response",
            "path": "id",
            "keep": "last",
            "inject": {
                "target": "request",
                "path": "prompt_cache_key",
                "if": "auto",
            },
            "scope": ["provider", "model", "credential", "session"],
            "ttl_seconds": 3600,
        },
        {
            # Preserve reasoning text across a tool-call round trip, keyed by
            # the tool-call id.
            "name": "reasoning_state",
            "source": "stream_event",
            "path": "choices[0].delta.reasoning_content",
            "keep": "per_tool_call",
            "tool_call_id_path": "choices[0].delta.tool_calls[0].id",
            "inject": {
                "target": "request",
                "path": "messages[-1].reasoning_content",
                "if": "auto",
            },
            "scope": ["provider", "model", "credential", "session"],
        },
    ]

    # =========================================================================
    # PIPELINE HOOKS
    # =========================================================================
    #
    # Class-declared hooks form the base; JSON ``providers.<name>.hooks``
    # append; globally registered hooks are resolved when the per-request run
    # is minted. Every declared name/stage is validated at startup. See
    # ``ExampleRequestObserver`` above for the hook contract.
    hooks: Tuple[Any, ...] = (ExampleRequestObserver,)

    # =========================================================================
    # EXECUTION MODES
    # =========================================================================
    #
    # Three dispatch modes, chosen per request by the executor:
    #
    #   custom  -> this class implements ``acompletion()``/``aembedding()``.
    #              Declare it with ``has_custom_logic() -> True``.
    #   native  -> ``protocol_name`` (or a profile's protocol) is declared and
    #              the protocol adapter builds the wire body. This is the
    #              default for a provider with a protocol declaration.
    #   litellm -> the fallback path; the default when no protocol is declared.
    #
    # The route suffix ``@custom``/``@native``/``@litellm_fallback``/``@auto``
    # selects explicitly; ``auto`` lets the executor decide.
    def has_custom_logic(self) -> bool:
        """Return True only if this provider implements its own execution.

        This template uses native protocol execution, so it returns False. A
        custom provider overrides this to True and implements
        ``acompletion()`` (and optionally ``aembedding()``), bypassing both the
        native protocol adapter and LiteLLM.
        """
        return False

    # If you flip ``has_custom_logic`` to True, implement the call surface:
    # (documented here; left unimplemented so the native path stays canonical)
    #
    # async def acompletion(self, client, **kwargs):
    #     """Handle the whole completion call for a non-standard provider."""
    #     ...
    # async def aembedding(self, client, **kwargs):
    #     """Handle the whole embedding call for a non-standard provider."""
    #     ...

    # =========================================================================
    # MODEL DISCOVERY
    # =========================================================================

    async def get_models(
        self, api_key: str, client: httpx.AsyncClient
    ) -> List[str]:
        """Fetch available model names from the provider's API. REQUIRED.

        Return provider-prefixed names (``example/model-id``); discovery
        rejects bare names. The base class declares this method abstract, so
        every concrete provider must implement it.

        Two discovery shapes exist in practice:

        * configured list — the dynamic provider returns the JSON
          ``providers.<name>.models`` list when present (prefixed);
        * listing endpoint — otherwise it calls the protocol's listing path
          (``/models`` for chat-family faces, ``/api/tags`` for Ollama) and
          reads ``data`` or ``models``.

        The ``<PROVIDER>_MODELS`` environment variable does NOT drive
        discovery for dynamic providers; it supplies model definitions (id
        aliases and default options) consumed by ``get_model_options`` and
        per-provider logic.
        """
        try:
            response = await client.get(
                f"{self.get_provider_api_base()}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            return [
                f"example/{entry.get('id', '')}"
                for entry in response.json().get("data", [])
                if isinstance(entry, dict) and entry.get("id")
            ]
        except httpx.RequestError as exc:
            lib_logger.error("Failed to fetch ExampleVendor models: %s", exc)
            return []

    # =========================================================================
    # SESSION TRACKING HINTS
    # =========================================================================

    def get_session_tracking_hints(
        self, request_data: Dict[str, Any], *, model: str = ""
    ) -> Optional[Any]:
        """Return provider-specific evidence for core session tracking.

        This is the seam for a provider that knows its own conversation/session
        marker and wants routing to keep related turns on one credential.
        Providers return EVIDENCE ONLY — they must never pick credentials or
        mutate sticky state; core routing merges hints with generic anchors
        and applies one confidence policy for every provider.

        Return ``None`` to keep the generic OpenAI-compatible tracker (the
        default). Otherwise return a ``SessionTrackingHints`` object::

            SessionTrackingHints(
                strong_anchors=["headers:x-conversation-id"],
                medium_anchors=["metadata.cache_key"],
                affinity_key="native-session-42",
                session_scope="thread",
            )

        ``session_scope`` partitions provider-native anchors without changing
        the global logical session identity.
        """
        return None

    def normalize_model_for_tracking(self, model: str) -> str:
        """Normalize internal model variants to public names for usage files.

        Some providers expose internal suffixes that should be accounted under
        one public name (for example, a thinking variant tracked as its base
        model). The default returns the model unchanged. Preserve any provider
        prefix if present.
        """
        has_prefix = "/" in model
        if has_prefix:
            provider, clean_model = model.split("/", 1)
        else:
            clean_model = model
        internal_to_public = {
            "example-frontier-v2-thinking": "example-frontier-v2",
        }
        normalized = internal_to_public.get(clean_model, clean_model)
        return f"{provider}/{normalized}" if has_prefix else normalized

    # =========================================================================
    # QUOTA / USAGE HOOKS
    # =========================================================================
    #
    # Usage accounting is per-provider. Declarative class attributes tune
    # rotation and quota; ``on_request_complete`` is the behavioral hook for
    # custom counting and cooldowns.
    #
    # Rotation mode: "sequential" (present in the config default) or
    # "balanced". Sequential keeps hitting one credential until it is
    # exhausted — ideal for per-credential quotas and cache affinity.
    default_rotation_mode = "sequential"

    # Models that share a quota pool. When one hits a quota-exhausted error,
    # all group members receive the same cooldown and reset together.
    # Env override: QUOTA_GROUPS_EXAMPLE_FRONTIER="frontier-v2,frontier-v2-mini"
    model_quota_groups: QuotaGroupMap = {
        "frontier": ["frontier-v2", "frontier-v2-mini"],
        "reasoning": ["reasoner-v1"],
    }

    # Relative quota cost per request, used when combining group usage.
    # Unlisted models default to weight 1.
    model_usage_weights: Dict[str, int] = {
        "reasoner-v1": 2,
    }

    # Priority-based concurrency multipliers: lower priority number = higher
    # tier. Applied to the provider's concurrency limits.
    default_priority_multipliers: Dict[int, int] = {
        1: 5,
        2: 3,
        3: 2,
    }
    default_sequential_fallback_multiplier = 2

    # Restrictive caps applied BEFORE the real API limit. Keys are a priority,
    # a tuple of priorities, or "default"; values map a model/group to a cap
    # config. ``max_requests`` may be an absolute number or a percentage;
    # ``cooldown_mode`` is "quota_reset" | "offset" | "fixed".
    default_custom_caps: Dict[Any, Dict[str, Dict[str, Any]]] = {
        3: {
            "frontier": {"max_requests": 50, "cooldown_mode": "quota_reset"},
        },
        (2, 3): {
            "reasoner-v1": {
                "max_requests": "80%",
                "cooldown_mode": "offset",
                "cooldown_value": 1800,
            },
        },
        "default": {
            "frontier": {
                "max_requests": 100,
                "cooldown_mode": "fixed",
                "cooldown_value": 3600,
            },
        },
    }

    # Tier name -> priority (lower is higher priority). This provider only
    # needs the mapping if credentials carry tier names.
    tier_priorities: Dict[str, int] = {
        "premium-tier": 1,
        "standard-tier": 2,
        "free-tier": 3,
    }

    # Custom usage windows are declared with ``usage_window_definitions``
    # (a list of dicts), which the usage-config loader reads. Each entry:
    #   name             window identifier ("5h", "daily", ...)
    #   duration_seconds window length (None only for a "total" window)
    #   reset_mode       rolling | fixed_daily | calendar_weekly |
    #                    calendar_monthly | api_authoritative
    #   is_primary       drives rotation decisions
    #   applies_to       credential | model | group
    #
    # NOTE: there is no ``default_windows`` class attribute; the loader reads
    # ``usage_window_definitions`` and otherwise falls back to a single daily
    # window. ``usage_reset_configs`` (below) is the per-tier reset surface.
    usage_window_definitions: List[Dict[str, Any]] = [
        {
            "name": "5h",
            "duration_seconds": 18000,
            "reset_mode": "rolling",
            "is_primary": True,
            "applies_to": "group",
        },
        {
            "name": "daily",
            "duration_seconds": 86400,
            "reset_mode": "rolling",
            "is_primary": False,
            "applies_to": "model",
        },
    ]

    def on_request_complete(
        self,
        credential: str,
        model: str,
        success: bool,
        response: Optional[Any],
        error: Optional[Any],
    ) -> Optional[RequestCompleteResult]:
        """Hook called after every request, success or failure.

        This is the primary behavioral extension point for customizing how
        requests are counted and how cooldowns are applied. Return ``None`` for
        default behavior, or a ``RequestCompleteResult``:

            count_override     0 = do not count; N = count as N requests
            cooldown_override  extra seconds to cool down this credential
            force_exhausted    mark for fair cycle even without a long cooldown

        Common patterns are shown below.
        """
        # Count internal retries accurately via the ContextVar.
        attempt_count = _example_attempt_count.get()
        _example_attempt_count.set(1)
        if attempt_count > 1:
            return RequestCompleteResult(count_override=attempt_count)

        if not success and error:
            error_type = getattr(error, "error_type", None)
            # Server errors are not the user's quota; do not count them.
            if error_type in ("server_error", "api_connection"):
                return RequestCompleteResult(count_override=0)
            # Rate limits: honor Retry-After, force exhaustion when long.
            if error_type == "rate_limit":
                retry_after = getattr(error, "retry_after", None)
                if retry_after and retry_after > 60:
                    return RequestCompleteResult(
                        cooldown_override=retry_after, force_exhausted=True
                    )
                if retry_after:
                    return RequestCompleteResult(cooldown_override=retry_after)
            # Quota exhaustion: block the credential for the fair cycle.
            if error_type == "quota_exceeded":
                return RequestCompleteResult(
                    force_exhausted=True, cooldown_override=3600.0
                )
        return None

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """Configure an optional periodic background task for this provider.

        Returns ``None`` for no job, otherwise a dict with ``interval``
        (seconds), ``name`` (for logging), and ``run_on_start`` (bool). The
        BackgroundRefresher calls ``run_background_job`` on that schedule.
        Typical uses: refresh quota baselines, clean caches, pre-refresh tokens.
        """
        return {
            "interval": 600,
            "name": "quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: UsageManager,
        credentials: List[str],
    ) -> None:
        """Periodic task body; ``get_background_job_config`` schedules it.

        Fetch provider quota and feed authoritative limits back into the usage
        manager so rotation decisions reflect the upstream truth.
        """
        for cred in credentials:
            try:
                quota_info = await self._fetch_quota_from_api(cred)
                if not quota_info:
                    continue
                for model, info in quota_info.items():
                    await usage_manager.update_quota_baseline(
                        accessor=cred,
                        model=model,
                        quota_max_requests=info.get("limit"),
                        quota_reset_ts=info.get("reset_ts"),
                        quota_used=info.get("used"),
                        quota_group=info.get("group"),
                    )
            except Exception as exc:  # pragma: no cover - provider-specific
                lib_logger.warning("Quota refresh failed for %s: %s", cred, exc)

    async def _fetch_quota_from_api(
        self, credential: str
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Return ``{model: {limit, used, reset_ts, group}}`` from the API.

        Placeholder — implement the real upstream call for your provider.
        """
        return None

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse a provider-specific quota/rate-limit error.

        Override when the upstream error format carries reset metadata. Return
        ``None`` (the default) to let the generic parser handle it, otherwise a
        dict with ``retry_after``, ``reason``, ``reset_timestamp``, and/or
        ``quota_reset_timestamp``.
        """
        return None

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """Return a human-readable tier name for a credential, or ``None``.

        Used for logging and for resolving priority via ``tier_priorities``.
        """
        return None

    def get_model_tier_requirement(self, model: str) -> Optional[int]:
        """Return the minimum priority a model requires, or ``None``.

        Restrict an expensive model to high-priority credentials by returning
        its minimum priority. The default allows every credential.
        """
        return None

    def get_model_pricing(self, model: str = "") -> Optional[Any]:
        """Return local pricing metadata for advisory cost tracking.

        Return a ``usage.costs.ModelPricing``-compatible object/dict, or
        ``None`` to fall back to LiteLLM metadata. This provider skips cost
        calculation, so it returns ``None``.
        """
        return None


# =============================================================================
# USAGE DATA ACCESS (FOR OPERATORS AND TOOLS)
# =============================================================================
#
# The per-provider usage manager exposes data for UI/monitoring:
#
#   stats = await usage_manager.get_availability_stats(model, quota_group)
#   stats = await usage_manager.get_stats_for_endpoint()
#   state = usage_manager.states.get(stable_id)
#   state.model_usage.get("frontier-v2")
#   state.group_usage.get("frontier")
#   cooldown = state.get_cooldown("frontier")
#   fc = state.fair_cycle.get("frontier")
#
# Authoritative quota can be pushed in from a background job:
#
#   await usage_manager.update_quota_baseline(
#       accessor=credential, model="frontier-v2",
#       quota_max_requests=500, quota_reset_ts=..., quota_used=123,
#       quota_group="frontier",
#   )
#
# =============================================================================


# =============================================================================
# REGISTERING YOUR PROVIDER
# =============================================================================
#
# 1. Rename this module to ``<name>_provider.py`` (no leading underscore). Put
#    the class in ``src/rotator_library/providers/``. Discovery does the rest.
#    If the registry key must differ from the module name, set
#    ``config_key_alias``.
#
# 2. Provide credentials with ``<NAME>_API_KEY`` (or numbered
#    ``<NAME>_API_KEY_1`` ...) in ``.env``. OAuth providers register discovery
#    in the client credential manager instead.
#
# 3. For a config-defined provider instead of code, use the JSON
#    ``providers`` section (``docs/examples/provider-config.example.json``):
#
#        {
#          "providers": {
#            "myserver": {
#              "api_base": "https://api.myserver.example/v1",
#              "protocol_name": "openai_chat",
#              "models": ["frontier-v2"],
#              "native_streaming_supported": true
#            }
#          }
#        }
#
#    and point ``LLM_PROXY_CONFIG_FILE`` at that file. Add
#    ``MYSERVER_API_KEY`` in the environment. There is no
#    ``MYSERVER_PROTOCOL`` env var — the protocol lives in this JSON.
#
# 4. Optional per-provider tuning (env):
#        EXAMPLE_API_BASE                 transport base override
#        EXAMPLE_MODELS                   model definitions (id/options)
#        EXAMPLE_CACHE_REPLAY             JSON cache_replay rules
#        QUOTA_GROUPS_EXAMPLE_FRONTIER    quota-group member override
#
# =============================================================================
