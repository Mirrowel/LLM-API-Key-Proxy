# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""G8 provider envelope — THE reference template for a provider file.

THIS MODULE IS EXCLUDED FROM AUTO-REGISTRATION.

Provider discovery imports every module in this package whose filename does
*not* start with an underscore (``providers/__init__.py`` ->
``_register_providers``). Because this file is ``_example_provider.py`` it is
never imported at runtime, its class is never inserted into
``PROVIDER_PLUGINS``, and nothing here executes in a deployment. It is a
teaching file: valid code, real surfaces, every declaration annotated with WHY
it exists and WHEN to use it. Copy it, rename it to ``<name>_provider.py``,
and delete whatever the upstream does not need.

===============================================================================
THE PHILOSOPHY: PICK FROM THE SET. INHERIT EVERYTHING. OVERRIDE ANYTHING.
===============================================================================

The envelope is a contract between the provider file and the shared runtime.
Three sentences carry all of it:

1. PICK FROM THE SET. Protocols, adapters, hooks, field locations, auth
   styles, listing shapes, and the reasoning-effort ladder are shared
   registries. A provider picks names from them; it never re-implements
   their content.
2. INHERIT EVERYTHING. A conventional value is written ZERO times. The
   provider says ``speaks = ("openai_chat",)`` and inherits the chat route,
   Bearer auth, the ``/models`` listing, the chat parameter defaults, and
   the chat effort base — because the protocol owns them.
3. OVERRIDE ANYTHING. A divergent value is written exactly ONCE, at the one
   surface that owns it: a ``speaks`` override for an endpoint or auth
   style, a ``model_rules`` row for a parameter capability, an explicit
   slot on a field rule. The override is a delta, never a copy of the
   inherited shape.

The failure mode this template exists to prevent is a provider file that
restates the shared set. Every restated endpoint, hardcoded vocabulary map,
and hand-rolled parameter strip is drift waiting to happen. If the runtime
adds a route or the ladder learns a word, the declared provider should
inherit it for free.

===============================================================================
WHAT A PROVIDER FILE IS — AND ISN'T
===============================================================================

A provider file IS:

* IDENTITY — ``provider_env_name`` and (rarely) ``config_key_alias``: the
  one name used by routing, credentials, usage files, session namespaces,
  and the JSON ``providers`` section.
* SPEAKS — the transport faces: which protocols the upstream accepts, what
  each face overrides (endpoints/auth/listing), which face is the default.
  This replaces all transport wiring.
* CAPABILITY TABLE — ``model_rules``: the ordered, CSS-like cascade that
  declares what each model family accepts — strip/clamp/map/rename, the
  reasoning-effort vocabulary, the off control's toggle, per-model face
  limits.
* FIELD RULES — ``field_cache_rules``: provider protocol state (reasoning,
  signatures, cache keys, continuation ids) to preserve across turns,
  addressed by FIELD so one rule serves every face whose protocol family
  the registry covers.
* OPTIONAL OVERRIDES — adapters named only for genuinely custom wire logic;
  hooks; listing overrides; session hints; quota/usage tuning; custom
  execution.

A provider file ISN'T:

* transport wiring — endpoints, auth headers, and listing routes inherit
  from the protocol registry (``protocols/defaults.py``);
* a vocabulary map — the effort system owns the ladder and the folds
  (``protocols/effort.py``); declarations state which rungs a model
  accepts, never what a word means;
* listing code — the shared interface implementation lists via the
  protocol's listing descriptor and returns an honest empty on failure;
* parameter hygiene code — the ``param_rules`` engine is always-on and
  consumes your declarations; providers never strip/clamp/rename in Python;
* a translator — client-vs-provider wire translation belongs to the
  protocol adapters and the neutral canonical model;
* a credential store — credentials travel only through
  ``get_native_headers()``, so payload traces never mix data with secrets.

The one hard rule: identity + declarations in, execution out. If you find
yourself writing request/response field mapping here, it belongs in a
protocol adapter (wire dialect) or a declared adapter (vendor quirk).

===============================================================================
TWO WAYS TO CREATE A PROVIDER
===============================================================================

1. CODE PROVIDER (this file). Subclass ``ProviderInterface`` in
   ``<name>_provider.py``; discovery registers the class under ``<name>``.
   Use this when the provider needs any Python: custom execution, a custom
   lister, a wire adapter, tier lookups.
2. DYNAMIC PROVIDER (``dynamic.py``). Created from the JSON ``providers``
   section for an upstream that needs none of the above.
   ``<NAME>_API_BASE`` + ``<NAME>_API_KEY`` is the minimal env form; the
   JSON section carries the full declaration surface. Almost every surface
   below has a JSON equivalent, so a code provider is usually reproducible
   as configuration.

===============================================================================
REQUEST LIFECYCLE (WHO DOES WHAT)
===============================================================================

::

    client request      provider[:profile]/model[@execution]
          |
          v
    routing/profiles.py speaks faces resolve: the client protocol selects a
          |             profile, or the default face answers bare requests
          v
    client/executor.py  execution mode: custom | native | litellm_fallback
          |             (native is the default for a provider with speaks)
          v
    protocols/*         the face's protocol builds the wire body
          |
          v
    native_provider/    effort emission: accepted vocabulary + ladder fold,
          |             off control -> thinking toggle (effort_emission.py)
          v
    adapters/*          param_rules ALWAYS first, then declared adapters in
          |             order (envelope adapters last)
          v
    field_cache/*       field-addressed rules extract provider state from
          |             the response/stream and inject it into later turns
          v
    hooks/*             declared pipeline stages (request/response/stream)
          |
          v
    HTTP send           endpoint + auth resolved from the face declaration
          |
          v
    response -> protocol parse -> adapters -> client dialect
          |
          v
    usage/*             usage accounting; on_request_complete() hook

===============================================================================
SURFACE MAP (WHAT LIVES WHERE)
===============================================================================

Identity       provider_env_name, config_key_alias (rare)
Transport      speaks, default_api_base, native_streaming_supported,
               default_auth_mode (only for zero-credential providers)
Capabilities   model_rules (ordered cascade; CSS inheritance)
State          field_cache_rules (field-addressed), legacy cache_replay
Adapters       adapter_names (escape hatch), get_adapter_names/config
Hooks          hooks
Execution      has_custom_logic(), acompletion()/aembedding()
Discovery      get_models() (shared; optional listing_profile hint)
Sessions       get_session_tracking_hints()
Quota/usage    model_quota_groups, model_usage_weights, default_*,
               usage_window_definitions, on_request_complete(),
               get_background_job_config()/run_background_job()

===============================================================================
PRECEDENCE (WHO WINS)
===============================================================================

Later layers win::

    code class attribute  <  JSON providers.<name>  <  environment variable

Surface-specific chains (verified in source):

* ``speaks`` transport: code-owned. JSON transport keys on a registered
  code provider are rejected at startup (``providers/__init__.py``).
* ``model_rules``: class rows apply top-to-bottom, then JSON rows append
  after them — config overrides code, per key.
* param rules: class ``param_rules`` -> JSON
  ``param_rules``/legacy ``model_param_rules`` -> matching ``model_rules``
  rows (class rows then JSON rows) -> resolved tables; ``strip_override``
  is terminal.
* reasoning effort: protocol base -> models.dev database seam (future) ->
  provider-level ``reasoning_effort_accept``/``_toggle`` -> ``model_rules``
  rows -> JSON config.
* field-cache rules merge by NAME: JSON ``field_cache`` > env
  ``<NAME>_CACHE_REPLAY`` > class ``cache_replay`` > declared
  ``field_cache_rules``; every same-name override must not weaken
  isolation/injection behavior or the request fails with a
  ``configuration_error``.

Environment variables that actually exist (do not invent others):

    <NAME>_API_BASE          transport base override (code + dynamic)
    <NAME>_API_KEY[_N]       credentials; numbered keys rotate
    <NAME>_MODELS            model definitions (id aliases/options)
    <NAME>_CACHE_REPLAY      JSON cache_replay rule list
    QUOTA_GROUPS_<PROVIDER>_<GROUP>    quota-group member override
    ROTATION_MODE_<PROVIDER>, FAIR_CYCLE_*, CUSTOM_CAP_*, MAX_CONCURRENT_*
    LLM_PROXY_CONFIG_FILE / PROXY_CONFIG_FILE    structured JSON config

There are NO ``<NAME>_PROTOCOL``, ``<NAME>_AUTH_MODE``,
``<NAME>_ENDPOINT_*``, or ``<NAME>_CONFIG`` variables. Protocol, auth,
endpoints, profiles, adapters, and hooks for a config-only upstream live in
the JSON ``providers`` section (see ``docs/examples/README.md``).
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..core.types import RequestCompleteResult
from ..field_cache import FieldCacheRule
from ..hooks.types import HookContext, PipelineHook, StageInvocation
from ..session_tracking import SessionTrackingHints
from ..usage import UsageManager
from .provider_interface import ProviderInterface, QuotaGroupMap

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# INTERNAL RETRY COUNTING (ContextVar PATTERN)
# =============================================================================
#
# A provider that retries internally (transient errors, empty responses, rate
# limits) performs several upstream calls that should all be counted. A
# provider instance is a process-wide singleton (``SingletonABCMeta``), so an
# instance attribute (``self.count``) would let concurrent requests clobber
# each other. A ``ContextVar`` gives every async task its own value:
#
#   1. set(1) at the start of the retry loop;
#   2. set(get() + 1) before each retry;
#   3. read it in ``on_request_complete`` and return
#      ``RequestCompleteResult(count_override=...)``.
#
# This template has no retry loop (execution is native), so the value stays 1
# and the hook below counts the request exactly once; the pattern is here for
# the day you implement custom execution.
_example_attempt_count: ContextVar[int] = ContextVar(
    "example_provider_attempt_count", default=1
)


# =============================================================================
# A DECLARED PIPELINE HOOK
# =============================================================================
#
# ``hooks`` is the pipeline extension point. The executor pauses at declared
# stages, hands over the live payload, and continues with whatever the hook
# returns. A hook may rewrite the payload, block the request, short-circuit a
# response, or drop/replace stream events. A callback (subclass
# ``PipelineCallback``) only observes settled snapshots and cannot change
# flow — prefer a callback unless mutation is required, because a hook has
# full read/write power over its stage payload.
#
# Class-declared hooks (below) are the base; JSON ``providers.<name>.hooks``
# append; globally registered hooks resolve when the per-request run is
# minted. Order is significant: priority ascending, ties broken
# class -> config -> global. Every referenced name/stage is validated at
# startup (``validate_provider_hooks``), never per request.
class ExampleRequestObserver(PipelineHook):
    """Minimal observer: logs request entry, changes nothing.

    This is a real, import-safe hook class. If this module were registered it
    would be validated at startup and bound for ``request_received``. Return
    ``None`` to continue with the payload unchanged.
    """

    name = "example_request_observer"
    stages: Tuple[str, ...] = ("request_received",)
    priority = 200  # late; smaller numbers fire first (default is 100)
    critical = False  # a failure here is contained (logged, not fatal)
    stateful = False  # shared instance; True would mint one per request

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
    """One fictional vendor, every G8 surface declared and annotated.

    If this module were registered, ``example`` would address it:
    ``example/example-frontier-v2``, ``example:responses/...``,
    ``example:anthropic/...``. Everything below is real code the runtime
    would honor.
    """

    # -------------------------------------------------------------------------
    # IDENTITY
    # -------------------------------------------------------------------------
    # The registry key is the module name with ``_provider`` stripped
    # (``example_provider.py`` -> ``"example"``). ``provider_env_name`` is the
    # env-name stem the shared helpers uppercase: quota-group lookups become
    # ``QUOTA_GROUPS_EXAMPLE_*``, and the JSON provider section key is this
    # name lowercased. Use the registry key; a second name is a bug waiting at
    # the routing boundary.
    provider_env_name = "example"

    # Optional registry remap for the rare case where the module name is NOT
    # the name the world calls the provider. NvidiaProvider sets
    # ``config_key_alias = "nvidia_nim"`` because LiteLLM calls it
    # ``nvidia_nim`` while the module is ``nvidia_provider.py``. When set, ONE
    # identity is used everywhere: registry, credentials, usage files, JSON
    # config, quota groups.
    # config_key_alias = "example_vendor"

    # Skip advisory cost accounting when the upstream reports no cost and no
    # reliable local pricing exists. Token usage is still tracked; only the
    # dollar estimate is suppressed.
    # NOTE(for-removal): dies with the cost phase.
    skip_cost_calculation = True

    # -------------------------------------------------------------------------
    # TRANSPORT: THE speaks ENVELOPE
    # -------------------------------------------------------------------------
    # A ``speaks`` entry declares ONE transport face. The grammar has exactly
    # three forms; use the cheapest form that tells the truth:
    #
    #   "protocol"                    the common case (~90%): profile name ==
    #                                 protocol name, and every endpoint /
    #                                 auth / listing default inherits from
    #                                 the protocol registry.
    #   (protocol, overrides)         same name, deltas only: somewhere the
    #                                 upstream diverges from the protocol
    #                                 convention. Overrides deep-merge over
    #                                 the protocol defaults.
    #   (name, protocol, overrides)   explicit profile identity: when the
    #                                 profile name must differ from the
    #                                 protocol (friendlier address) OR the
    #                                 same protocol is declared twice (a
    #                                 subscription/compat face — names must
    #                                 be unique).
    #
    # The FIRST entry is the default face: bare ``example/model`` routes
    # here. Other faces are addressed ``example:<name>/model`` and are
    # validated at routing time (an unknown profile is a loud error, never a
    # guess). Exactly one face serves a request; credentials, quota, sessions,
    # and cache provenance all stay keyed on the bare provider identity.
    speaks = (
        # (1) string form — the default chat face. Every default inherits:
        # /chat/completions, Bearer auth, the GET /models listing.
        "openai_chat",
        # (2) (protocol, overrides) — the Responses face. Responses accepts
        # /responses like the protocol default; only the native token-count
        # route diverges, so only that path is written down.
        (
            "responses",
            {"endpoint_paths": {"count_tokens": "/responses/input_tokens"}},
        ),
        # (3) (name, protocol, overrides) — an Anthropic-compatibility face
        # under the profile name ``anthropic`` (address:
        # example:anthropic/model). The name differs from the protocol on
        # purpose: it is the addressing word, not a wire fact. The paths
        # diverge (the compat surface is mounted under /anthropic/v1); auth
        # does NOT — x-api-key inherits from the anthropic_messages protocol
        # default.
        (
            "anthropic",
            "anthropic_messages",
            {
                "endpoint_paths": {
                    "messages": "/anthropic/v1/messages",
                    "count_tokens": "/anthropic/v1/count_tokens",
                }
            },
        ),
    )

    # Transport base for every face. Once this module is registered, env
    # ``EXAMPLE_API_BASE`` overrides it (the REGISTRY key drives the env
    # lookup, not provider_env_name). Leave None for a provider with no
    # stable public base (JSON-defined upstreams read ``api_base`` from
    # config).
    default_api_base = "https://api.example-vendor.example/v1"

    # Opt in before routed streaming may use the native stream executor. The
    # default is deliberately conservative (False); flip it once the
    # upstream's stream dialect is verified.
    native_streaming_supported = True

    # Zero-credential providers (a local Ollama) declare "none" here so
    # routing mints the internal no-auth rotation slot. Leave it None for
    # ordinary API-key providers; do NOT declare "bearer" — that value comes
    # from the protocol default, and restating it is drift.
    # default_auth_mode = "none"

    # -------------------------------------------------------------------------
    # CAPABILITY TABLE: model_rules
    # -------------------------------------------------------------------------
    # ``model_rules`` is an ORDERED tuple of rows. Rows matching the model id
    # (fnmatch, case-insensitive; a provider-prefixed id also matches its
    # stripped form) apply top-to-bottom like a CSS cascade: a later row
    # overrides the keys it declares and inherits everything it does not.
    # ``*`` is the provider-default row. JSON ``model_rules`` rows append
    # AFTER these, so config overrides code.
    #
    # Row vocabulary:
    #   strip           wire keys this model must not send
    #   clamp           numeric bounds ``{param: [lo, hi]}``
    #   map             value translation ``{param: {from: to}}`` (values
    #                   absent from the table pass through; declare an
    #                   exhaustive table only if the wire demands one)
    #   rename          key translation ``{old: new}``
    #   strip_override  TERMINAL: REPLACES the inherited strip list for this
    #                   model — no union. A model that re-admits a key must
    #                   not silently re-inherit a global strip added later.
    #   effort_accept   the reasoning-effort vocabulary this model accepts;
    #                   the ladder folds incoming words into it (nearest
    #                   accepted rung, searching upward first, ties round up)
    #   toggle          the OFF control rides the chat wire's thinking toggle
    #   allow / deny    per-model face limiting; the face limiter refuses a
    #                   disallowed face with an error naming the deciding row
    #
    # The accepted-effort chain (later wins): protocol base -> models.dev
    # database seam (future) -> provider-level
    # ``reasoning_effort_accept``/``reasoning_effort_toggle`` attribute ->
    # matching rows (this table) -> JSON config. Declare vocabularies ONLY as
    # accepted sets; never write a word-to-word map.
    model_rules = (
        # Row 1 — the provider default. Renames the modern token knob to this
        # vendor's spelling, clamps the documented temperature window, maps
        # the one tool_choice spelling difference, and strips keys the vendor
        # rejects: ``reasoning_effort`` lives here because ONLY the reasoning
        # families below accept it (their strip_override re-admits it).
        {
            "match": "*",
            "strip": ["reasoning_effort", "logit_bias", "logprobs", "top_logprobs"],
            "clamp": {"temperature": [0.0, 2.0]},
            "map": {"tool_choice": {"required": "any"}},
            "rename": {"max_completion_tokens": "max_tokens"},
        },
        # Row 2 — the reasoning family wildcard. ``strip_override`` is
        # TERMINAL: the family replaces row 1's strip list (so
        # reasoning_effort is legal here) and keeps only the three
        # always-rejected keys. ``effort_accept`` declares the family's spec
        # vocabulary; ``toggle`` says the OFF control rides the thinking
        # toggle on this wire.
        {
            "match": "example-reasoner-*",
            "strip_override": ["logit_bias", "logprobs", "top_logprobs"],
            "effort_accept": ["off", "low", "medium", "high"],
            "toggle": True,
        },
        # Row 3 — an EXACT model overriding its family. v2 accepts one more
        # rung (xhigh) than the family; the later row replaces the
        # effort_accept key and INHERITS row 2's toggle and strip_override
        # (non-conflicting keys). This is the cascade live: override what
        # differs, inherit the rest.
        {
            "match": "example-reasoner-v2",
            "effort_accept": ["off", "low", "medium", "high", "xhigh"],
        },
        # Want to pin a model to one face? The capability keys ``allow`` and
        # ``deny`` take protocol names and are enforced whenever a face
        # resolves (``get_protocol_name`` raises, naming the deciding row):
        #
        #     {"match": "example-reasoner-v2", "allow": ["openai_chat"],
        #      "deny": ["responses"]},
        #
        # The limiter matches by wire family, so a ``responses`` entry also
        # governs the sibling responses_* variants.
    )

    # -------------------------------------------------------------------------
    # STATE PRESERVATION: field_cache_rules
    # -------------------------------------------------------------------------
    # Field-cache rules preserve provider-generated state across turns —
    # reasoning content, thought signatures, prompt-cache keys,
    # response/continuation ids — so the upstream sees its own state again
    # instead of a client-echoed guess. They are NOT session affinity:
    # session tracking decides continuity and credential stickiness; the
    # cache only replays state.
    #
    # Declare rules by FIELD, not by path. The engine resolves the effective
    # extraction, injection, and occurrence-correlation locations from
    # ``protocols/defaults.FIELD_LOCATIONS`` for the EXECUTING face's
    # protocol family, so ONE rule serves every face the registry covers. Any
    # single slot can still be overridden explicitly (``path=``,
    # ``inject=...``, ``metadata=...``), and the explicit declaration wins
    # over the registry.
    #
    # ``sources=(a, b)`` is the multi-source convenience: the engine expands
    # one declaration into sibling rules sharing ONE cache key, so a value
    # extracted from a response can be restored after a streamed one (and
    # vice versa). An unset ``cache_key`` auto-derives as
    # ``{provider}:{field}``; an unset ``ttl_seconds`` lets the store's
    # 3-day inactivity default own retention (declare a TTL only when the
    # protocol demands one).
    #
    # ``inject="auto"`` restores ONLY where the field is absent (client
    # state stays authoritative); "always" overwrites. ``mode`` picks the
    # occurrence scope: "turn" (latest region, the default), "turns" (the
    # last ``turn_count`` regions), "all" (every occurrence). ``placeholder``
    # is injected for an occurrence that cannot be correlated — with a loud
    # warning, never silently — so declare one only when the upstream
    # hard-rejects a missing field.
    #
    # COVERAGE: the registry today covers the path-addressable families
    # (openai_chat, ollama). On a face whose family declares no locations
    # (the Responses / Anthropic faces above, until their structural
    # resolvers land) a field-addressed rule fails LOUDLY at derivation — a
    # rule that cannot see the field must not pretend. A provider shipping
    # such a face today either speaks only covered families or declares the
    # rule without ``field=`` and writes the paths explicitly (plain
    # path-addressed rules pass through untouched on every family).
    field_cache_rules = (
        # Rule 1 — reasoning, response + stream twins, all history. When
        # tools are in play the upstream demands reasoning on EVERY turn, so
        # mode="all", and an uncorrelated occurrence gets the documented
        # placeholder instead of fabricated silence.
        FieldCacheRule(
            name="reasoning",
            field="reasoning",
            sources=("response", "stream_event"),
            mode="all",
            placeholder="Reasoning content unavailable.",
            inject="auto",
        ),
        # Rule 2 — thought signatures, last two assistant regions. The
        # ``turns``/``turn_count`` variant scopes restoration to the most
        # recent N regions when replaying older signatures is pointless or
        # risky; no placeholder because this vendor does not 400 on a
        # missing signature.
        FieldCacheRule(
            name="signature",
            field="signature",
            source="response",
            mode="turns",
            turn_count=2,
            inject="auto",
        ),
    )

    # -------------------------------------------------------------------------
    # ADAPTERS: declare ONLY for genuinely custom wire logic
    # -------------------------------------------------------------------------
    # Adapters run on the raw provider payload between the protocol build and
    # the send (request, response, and stream stages). The ``param_rules``
    # engine is ALWAYS-ON and PREPENDED by the interface: never declare it,
    # and never write Python for anything the capability table can express.
    #
    # Justification criteria — declare a custom adapter ONLY when at least
    # one is true:
    #   1. NESTED targets: the change lands inside a sub-object (``seed`` ->
    #      ``extra_body.random_seed``); the flat table vocabulary cannot
    #      express it.
    #   2. STRUCTURAL reshaping: content chunks, tool calls, or whole
    #      envelopes change shape (think-chunk folding, wrapper envelopes).
    #   3. HISTORY surgery: fields must be added/dropped across every message
    #      rather than at one key.
    #   4. CONDITIONAL behavior: the change depends on other payload content
    #      (only strip when X is present).
    #
    # Even then, FIRST extend the param engine (subclass ParamRulesAdapter
    # and set ``consumes_param_rules = True``) so the declared tables and the
    # custom surgery run as ONE chain entry — ``adapters/mistral.py`` is the
    # worked example. Envelope adapters must be declared LAST (they wrap
    # everything before them).
    adapter_names: Tuple[str, ...] = ()
    #
    # With an adapter: reference it by name; it must exist in the adapter
    # registry or startup fails. Per-adapter config comes from JSON
    # ``providers.<name>.adapter_config`` (or a ``get_adapter_config``
    # override).
    #     adapter_names = ("example_wire",)

    # -------------------------------------------------------------------------
    # PIPELINE HOOKS
    # -------------------------------------------------------------------------
    # See ``ExampleRequestObserver`` above for the hook contract. Class hooks
    # are the base; JSON ``providers.<name>.hooks`` append; global hook names
    # resolve per request. Every name/stage is validated at startup
    # (``validate_provider_hooks``), never per request.
    hooks: Tuple[Any, ...] = (ExampleRequestObserver,)

    # -------------------------------------------------------------------------
    # EXECUTION MODES
    # -------------------------------------------------------------------------
    # Three dispatch modes; the executor picks per request from ``@custom`` /
    # ``@native`` / ``@litellm_fallback`` / ``@auto``:
    #   custom  - this class implements acompletion()/aembedding()
    #   native  - a speaks face exists; the protocol builds the wire body
    #             (the default for a provider with a declaration)
    #   litellm - the fallback path (the default when no face is declared)
    def has_custom_logic(self) -> bool:
        """False: this provider is pure declaration on the native path."""
        return False

    # Custom execution (only if has_custom_logic() returns True). Then every
    # wire detail is yours — and so is every native service: no protocol
    # build, no param engine, no field cache, no native pipeline stages.
    #     async def acompletion(self, client, **kwargs): ...
    #     async def aembedding(self, client, **kwargs): ...

    # -------------------------------------------------------------------------
    # OPTIONAL OVERRIDES — escape hatches, all inheritance-first
    # -------------------------------------------------------------------------
    # The base implementations are CORRECT for a protocol-conventional
    # provider. Override only what reality forces, and never translate
    # protocols here.
    #
    # get_native_endpoint(...)   override when a face's path is fixed or
    #   computed instead of declared (the base resolves the face's
    #   ``endpoint_paths``, then the protocol default). The operation
    #   vocabulary is dialect-specific: chat | responses | messages |
    #   generate | stream_generate | ollama_chat | ollama_generate | models.
    # get_native_headers(...)    override for a non-declared auth scheme;
    #   per-face auth declarations already win, and credentials belong HERE,
    #   never in the payload.
    # normalize_native_model(...) override for id aliases; the base strips
    #   the provider prefix.
    # prepare_native_request(...) LAST provider-owned payload adjustment
    #   before send; add required defaults/envelopes only.
    # get_adapter_names/config   per-model chain/config quirks.
    # get_models(...)            see DISCOVERY below.
    #
    # Example shapes (uncomment only with a reason):
    #
    #     def get_native_headers(self, credential_identifier, model="",
    #                            operation="chat", profile=None):
    #         return {"x-api-key": credential_identifier}
    #
    #     def prepare_native_request(self, request, model="", operation=""):
    #         payload = dict(request)
    #         payload.setdefault("temperature", 1)
    #         return payload

    # -------------------------------------------------------------------------
    # DISCOVERY
    # -------------------------------------------------------------------------
    # Model listing is INHERITED and shared: the interface resolves the
    # listing face (an optional ``listing_profile`` hint wins, else the
    # protocol priority list picks the first face with a listing descriptor),
    # fetches the descriptor's route with per-protocol auth, parses the
    # descriptor's shape (``data[].id`` for the openai family,
    # ``models[].name`` with prefix stripping for gemini/ollama), and
    # returns provider-prefixed ids. A failed listing is an HONEST EMPTY —
    # there are no hardcoded fallback lists.
    #
    # Override ONLY when the upstream's listing genuinely deviates (a
    # gateway that advertises uncallable pseudo-models — ``ChutesProvider``
    # is the worked example) or when a config-defined upstream must return
    # its configured model list. If you do, return ids prefixed with the
    # registry key, and keep failures empty:
    #
    #     async def get_models(self, api_key, client):
    #         response = await client.get(
    #             f"{self.get_provider_api_base()}/custom/models",
    #             headers=self.get_native_headers(api_key, operation="models"),
    #         )
    #         response.raise_for_status()
    #         try:
    #             return [
    #                 f"example/{entry['id']}"
    #                 for entry in response.json().get("result", [])
    #                 if isinstance(entry, dict) and entry.get("id")
    #             ]
    #         except (httpx.RequestError, ValueError):
    #             return []
    #
    # ``<NAME>_MODELS`` does NOT drive discovery: it supplies model
    # definitions (id aliases and default options) consumed elsewhere.

    # -------------------------------------------------------------------------
    # SESSION HINTS (the designed seam)
    # -------------------------------------------------------------------------
    def get_session_tracking_hints(
        self, request_data: Dict[str, Any], *, model: str = ""
    ) -> Optional[Any]:
        """Return provider-specific session evidence, or ``None``.

        Session tracking keeps related turns on one credential. Returning
        ``None`` keeps the generic OpenAI-compatible tracker, which is
        correct for almost every provider. A provider that knows its own
        conversation marker returns EVIDENCE ONLY — never credential
        choices, never sticky-state mutations; core routing merges hints
        with the generic anchors under ONE confidence policy::

            SessionTrackingHints(
                strong_anchors=["headers:x-conversation-id"],
                medium_anchors=["metadata.cache_key"],
                affinity_key="native-session-42",
                session_scope="thread",
            )

        ``session_scope`` partitions provider-native anchors without
        changing the global logical session identity.
        """
        return None

    # -------------------------------------------------------------------------
    # QUOTA / USAGE
    # -------------------------------------------------------------------------
    # Declarative surfaces first; behavioral hooks only when accounting needs
    # provider truth the generic engine cannot see.
    #
    # Rotation: ``sequential`` parks on one credential until it is exhausted —
    # ideal for per-credential quotas and cache affinity; ``balanced`` spreads
    # load. Overridable with ``ROTATION_MODE_<PROVIDER>``.
    default_rotation_mode = "sequential"

    # Models sharing a quota pool: one member's quota-exhausted error cools
    # the whole group down. Overridable with
    # ``QUOTA_GROUPS_EXAMPLE_FRONTIER="frontier-v2,frontier-v2-mini"``.
    model_quota_groups: QuotaGroupMap = {
        "frontier": ["example-frontier-v2", "example-frontier-v2-mini"],
    }

    # Relative quota cost per request when combining group usage; unlisted
    # models default to weight 1.
    model_usage_weights: Dict[str, int] = {"example-reasoner-v2": 2}

    # Restrictive caps applied BEFORE the upstream limit (clamping is
    # more-restrictive-only). Keys: a priority, a tuple of priorities, or
    # "default"; values map a model/group to a config:
    # ``{max_requests: int | "80%", cooldown_mode: quota_reset | offset |
    # fixed, cooldown_value: seconds}``.
    default_custom_caps: Dict[Any, Dict[str, Dict[str, Any]]] = {
        3: {"frontier": {"max_requests": 50, "cooldown_mode": "quota_reset"}},
    }

    # Custom usage windows. Each entry: ``name``, ``duration_seconds`` (None
    # only for a "total" window), ``reset_mode`` (rolling | fixed_daily |
    # calendar_weekly | calendar_monthly | api_authoritative), ``is_primary``
    # (drives rotation decisions), ``applies_to`` (credential | model |
    # group). Without a declaration the loader falls back to one daily
    # window.
    usage_window_definitions: List[Dict[str, Any]] = [
        {
            "name": "5h",
            "duration_seconds": 18000,
            "reset_mode": "rolling",
            "is_primary": True,
            "applies_to": "group",
        },
    ]

    # Tier name -> priority (lower = more valuable). Needed only when
    # credentials carry tier names; ``get_credential_tier_name`` supplies the
    # name.
    tier_priorities: Dict[str, int] = {
        "premium-tier": 1,
        "standard-tier": 2,
        "free-tier": 3,
    }

    def on_request_complete(
        self,
        credential: str,
        model: str,
        success: bool,
        response: Optional[Any],
        error: Optional[Any],
    ) -> Optional[RequestCompleteResult]:
        """Post-request accounting hook — return ``None`` for default behavior.

        This is THE behavioral extension point: count internal retries,
        exempt server errors, honor large Retry-After windows, force
        exhaustion on quota errors. Return a ``RequestCompleteResult``:

            count_override     0 = do not count; N = count as N requests
            cooldown_override  extra cooldown seconds for this credential
            force_exhausted    mark exhausted for the fair cycle

        Internal retries are counted through the ContextVar above: instance
        attributes would race across concurrent requests on this
        process-wide singleton.
        """
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
        """Optional periodic task; ``None`` means no job.

        Returns ``{"interval": seconds, "name": str, "run_on_start": bool}``.
        The BackgroundRefresher calls ``run_background_job`` on that
        schedule. Typical uses: refresh upstream quota baselines, clean
        caches, pre-refresh tokens.
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
        """Periodic body: push authoritative quota into the usage engine.

        Updating baselines keeps rotation decisions tied to upstream truth,
        not only local counts. Failures are logged and contained — a
        background job must never take the proxy down.
        """
        for credential in credentials:
            try:
                quota_info = await self._fetch_quota_from_api(credential)
                if not quota_info:
                    continue
                for model, info in quota_info.items():
                    await usage_manager.update_quota_baseline(
                        accessor=credential,
                        model=model,
                        quota_max_requests=info.get("limit"),
                        quota_reset_ts=info.get("reset_ts"),
                        quota_used=info.get("used"),
                        quota_group=info.get("group"),
                    )
            except Exception as exc:  # pragma: no cover - provider-specific
                lib_logger.warning("Quota refresh failed for %s: %s", credential, exc)

    async def _fetch_quota_from_api(
        self, credential: str
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Return ``{model: {limit, used, reset_ts, group}}`` or ``None``.

        Placeholder — implement the real upstream call for your provider.
        """
        return None

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Parse a vendor-specific quota/rate-limit error, or ``None``.

        Override when the upstream's error body carries reset metadata the
        generic parser cannot read; return ``retry_after``, ``reason``,
        ``reset_timestamp``, and/or ``quota_reset_timestamp``.
        """
        return None

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        """Human-readable tier for logging and ``tier_priorities`` lookup."""
        return None

    def get_model_tier_requirement(self, model: str) -> Optional[int]:
        """Minimum credential priority for a model, or ``None`` (open to all)."""
        return None

    def get_model_pricing(self, model: str = "") -> Optional[Any]:
        """Local pricing metadata for advisory cost tracking, or ``None``.

        This provider skips cost calculation, so ``None`` is the honest
        answer: the accounting layer reports pricing as unavailable.
        """
        return None

    def normalize_model_for_tracking(self, model: str) -> str:
        """Map internal variants to the public name usage files record.

        Keeps the provider prefix when present. Override when the upstream
        exposes suffixed variants that must not split usage accounting (a
        ``-thinking`` variant billed as its base model, for example).
        """
        return model


# =============================================================================
# REGISTERING YOUR PROVIDER — AND WHERE THE EXAMPLES LIVE
# =============================================================================
#
# 1. Copy this module to ``<name>_provider.py`` (no leading underscore) in
#    this package. Discovery imports it and registers the class under
#    ``<name>``. Delete the surfaces the upstream does not need; every
#    commented ``#`` example is optional.
#
# 2. Credentials: ``<NAME>_API_KEY`` (or numbered ``<NAME>_API_KEY_1`` ...)
#    in the environment. OAuth providers register discovery in the client
#    credential manager instead.
#
# 3. The JSON annotation layer is ``docs/examples/README.md``; the sample
#    file is ``docs/examples/provider-config.example.json``. A config-only
#    upstream needs no code at all: declare the provider in the JSON
#    ``providers`` section and point ``LLM_PROXY_CONFIG_FILE`` (or
#    ``PROXY_CONFIG_FILE``) at that file. Credentials never go in JSON.
#
# 4. Env knobs that actually exist for a provider (the full list is in the
#    module docstring above): ``<NAME>_API_BASE``, ``<NAME>_API_KEY[_N]``,
#    ``<NAME>_MODELS``, ``<NAME>_CACHE_REPLAY``,
#    ``QUOTA_GROUPS_<PROVIDER>_<GROUP>``, and the usage/rotation tuning
#    family (``ROTATION_MODE_*``, ``FAIR_CYCLE_*``, ``CUSTOM_CAP_*``).
#
# =============================================================================
