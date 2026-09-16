# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""RequestContext construction for RotatingClient public request methods."""

import inspect
import logging
import time
from copy import deepcopy
from dataclasses import replace
from typing import Any, Awaitable, Callable, Dict, Optional

from ..routing.profiles import PROFILE_SEPARATOR, ModelReferenceError

from ..core.types import RequestContext
from ..hooks.binding import make_pipeline_run
from ..hooks.runner import run_slot
from ..protocols import ProtocolContext, get_protocol
from ..protocols.validation import validate_embeddings_request
from ..routing import FallbackResolver, RoutingConfigError, load_routing_config_from_env
from ..routing.model_args import apply_model_args_to_unified, split_model_args
from ..routing.types import RouteTarget, RoutingDecision
from ..session_tracking import SessionTrackingHints
from ..transaction_logger import TransactionLogger
from .scopes import derive_session_isolation_key


class RequestContextBuilder:
    """Build scoped RequestContext objects for completion-like requests."""

    def __init__(
        self,
        *,
        resolve_scope_for_provider: Callable[
            [str, Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, Dict[str, Any]]], bool],
            Awaitable[Dict[str, Any]],
        ],
        model_resolver: Any,
        session_tracker: Any,
        get_global_timeout: Callable[[], int],
        get_enable_request_logging: Callable[[], bool],
        get_provider_instance: Optional[Callable[[str], Any]] = None,
        experimental_config: Optional[Any] = None,
    ):
        self._resolve_scope_for_provider = resolve_scope_for_provider
        self._model_resolver = model_resolver
        self._session_tracker = session_tracker
        self._get_global_timeout = get_global_timeout
        self._get_enable_request_logging = get_enable_request_logging
        self._get_provider_instance = get_provider_instance
        self._experimental_config = experimental_config

    @staticmethod
    def _pop_scope_kwargs(kwargs: Dict[str, Any]) -> tuple[Optional[str], Any, Any, bool, Any]:
        classifier = kwargs.pop("classifier", None)
        request_api_keys = kwargs.pop("api_keys", None)
        request_providers = kwargs.pop("providers", None)
        private = bool(kwargs.pop("private", False))
        session_tracking_hints = kwargs.pop("_session_tracking_hints", None)
        kwargs.pop("model_filters", None)
        return classifier, request_api_keys, request_providers, private, session_tracking_hints

    @staticmethod
    def _session_isolation_key(
        classifier: Optional[str],
        request_api_keys: Any,
        request_providers: Any,
        private: bool,
    ) -> str:
        """Return a provider-independent caller/credential isolation key.

        Ad hoc credentials and provider overrides are hashed as one bundle so
        secrets never enter logs or persistence. Named classifiers intentionally
        own one authoritative domain across their providers and credential changes.
        """

        return derive_session_isolation_key(
            classifier,
            request_api_keys,
            request_providers,
            private,
        )

    @staticmethod
    def _provider_from_model(model: str) -> str:
        return model.split("/")[0] if "/" in model else ""

    @staticmethod
    def _raise_no_provider(model: str) -> None:
        raise ValueError(f"Invalid model format or no credentials for provider: {model}")

    def _resolve_routing_decision(self, model: str) -> Optional[RoutingDecision]:
        """Resolve env-configured fallback routing, if any applies.

        Broken routing config never silently degrades: load and resolve
        errors surface (fail-loud), because an operator's fallback chain
        silently disappearing is the exact surprise this module exists to
        prevent.
        """

        config = load_routing_config_from_env()
        if not config.fallback_groups and not config.model_routes:
            return None
        decision = FallbackResolver(config).resolve(model)
        if decision.reason == "direct_provider_model" and model.lower() not in config.model_routes:
            return None
        return decision

    @staticmethod
    def _with_request_scope(target: RouteTarget, scope: Dict[str, Any]) -> RouteTarget:
        """Attach per-provider request scope to a route target without secrets in traces."""

        metadata = dict(target.metadata)
        metadata["request_scope"] = {
            "credentials": list(scope["credentials"]),
            "usage_manager_key": scope["usage_manager_key"],
            "provider_config": scope["provider_config"],
            "credential_secrets": dict(scope["credential_secrets"]),
        }
        # dataclasses.replace keeps every field (profile included) — the
        # historical bug here was a hand-written reconstruction that
        # dropped the transport profile (fix-pass G7).
        return replace(target, metadata=metadata)

    async def _get_session_hints(
        self,
        provider: str,
        model: str,
        kwargs: Dict[str, Any],
        *,
        unified_request: Any = None,
        input_protocol: str = "openai_chat",
    ) -> Any:
        """Ask the provider for optional session evidence before routing.

        Providers can understand native request shapes better than the generic
        OpenAI-compatible tracker, but they should only return evidence. The core
        tracker still decides whether that evidence is strong enough for sticky
        routing.
        """
        if not self._get_provider_instance:
            return None
        plugin = self._get_provider_instance(provider)
        hook = getattr(plugin, "get_session_tracking_hints", None) if plugin else None
        if not hook:
            return None
        try:
            provider_request = kwargs
            if unified_request is not None:
                provider_protocol_name = plugin.get_protocol_name(model) if hasattr(plugin, "get_protocol_name") else "openai_chat"
                provider_protocol = get_protocol(provider_protocol_name)
                native_model = plugin.normalize_native_model(model) if hasattr(plugin, "normalize_native_model") else model
                provider_view = deepcopy(unified_request)
                provider_view.model = native_model
                provider_request = provider_protocol.build_request(
                    provider_view,
                    ProtocolContext(
                        input_protocol=input_protocol,
                        provider_protocol=provider_protocol.name,
                        client_protocol=input_protocol,
                        source_protocol=input_protocol,
                        target_protocol=provider_protocol.name,
                        source_provider=None,
                        target_provider=provider,
                        provider=provider,
                        model=native_model,
                    ),
                )
                operation = plugin.get_native_operation(native_model, None, stream=bool(provider_view.stream)) if hasattr(plugin, "get_native_operation") else "generate"
                if hasattr(plugin, "prepare_native_request"):
                    provider_request = plugin.prepare_native_request(provider_request, native_model, operation)
            result = hook(provider_request, model=model)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as exc:
            # Hints are optional evidence. A provider bug here should not prevent
            # request construction or credential routing.
            import logging

            logging.getLogger("rotator_library").debug(
                "Provider session tracking hints failed for %s/%s: %s",
                provider,
                model,
                exc,
            )
            return None

    @staticmethod
    def _merge_session_hints(*hints: Any) -> Any:
        """Merge proxy-internal and provider session evidence.

        Internal hints are removed from request kwargs before provider execution.
        They let services such as Responses expose stable continuation IDs to the
        centralized tracker without adding provider-visible payload fields.
        """

        merged = SessionTrackingHints()
        seen = False
        for index, hint in enumerate(hints):
            if not hint:
                continue
            allow_global = index == 0 and isinstance(hint, SessionTrackingHints)
            if isinstance(hint, dict):
                hint = SessionTrackingHints(
                    strong_anchors=list(hint.get("strong_anchors") or []),
                    medium_anchors=list(hint.get("medium_anchors") or []),
                    weak_anchors=list(hint.get("weak_anchors") or []),
                    affinity_key=hint.get("affinity_key"),
                    session_scope=hint.get("session_scope"),
                )
            if not isinstance(hint, SessionTrackingHints):
                continue
            seen = True
            merged.strong_anchors.extend(hint.strong_anchors)
            merged.medium_anchors.extend(hint.medium_anchors)
            merged.weak_anchors.extend(hint.weak_anchors)
            if allow_global:
                merged.global_strong_anchors.extend(hint.global_strong_anchors)
                merged.global_medium_anchors.extend(hint.global_medium_anchors)
                merged.global_weak_anchors.extend(hint.global_weak_anchors)
            if not merged.affinity_key and hint.affinity_key:
                merged.affinity_key = hint.affinity_key
            if not merged.session_scope and hint.session_scope:
                merged.session_scope = hint.session_scope
        return merged if seen else None

    async def build_completion_context(
        self,
        request: Optional[Any],
        pre_request_callback: Optional[Callable],
        kwargs: Dict[str, Any],
    ) -> RequestContext:
        classifier, request_api_keys, request_providers, private, internal_session_hints = self._pop_scope_kwargs(
            kwargs
        )
        parent_log_dir = kwargs.pop("_parent_log_dir", None)
        disable_provider_continuation = bool(kwargs.pop("_disable_provider_continuation", False))
        requested_input_protocol = str(kwargs.pop("_input_protocol", "openai_chat") or "openai_chat")
        requested_operation = str(kwargs.pop("_requested_operation", "") or "")
        input_protocol = get_protocol(requested_input_protocol)
        # R1 client entry: the payload passes through the request_received
        # slot BEFORE any snapshot, normalization, or routing — hook edits are
        # authoritative (in-place), so the pristine protocol snapshot, the
        # cross-protocol projection, model resolution, routing, and session
        # inference all see the edited payload. A model rewrite here
        # legitimately changes routing. The run is minted bare (global and
        # config hooks only): the provider is not known at entry, so
        # provider-bound hooks cannot intercept this stage — they bind at the
        # enrichment below.
        pipeline_run = make_pipeline_run(None, operation="chat", config=self._experimental_config)
        entry_outcome = await run_slot(pipeline_run, "request_received", kwargs, direction="request", copy_payload=False)
        if entry_outcome.modified and isinstance(entry_outcome.payload, dict):
            edited = entry_outcome.payload
            changed_keys = sorted(
                key for key in set(edited) | set(kwargs)
                if key not in kwargs or key not in edited or edited.get(key) != kwargs.get(key)
            )[:20]
            kwargs.clear()
            kwargs.update(edited)
            pipeline_run.record_overlay(
                "request_received_edit",
                stage="request_received",
                direction="request",
                keys=changed_keys,
            )
        # D13 grammar FIRST: provider:profile/model normalizes to the bare
        # form before ANY snapshot or identity use, so protocol_request,
        # unified_request, raw fast-path payloads, and every identity sink
        # (usage pools, cooldowns, classifiers, sessions, cache scopes) see
        # only provider-level names.
        requested_profile = _normalize_profile_reference(kwargs)
        protocol_request = deepcopy(kwargs)
        protocol_context = ProtocolContext(
            source_protocol=input_protocol.name,
            target_protocol=input_protocol.name,
            input_protocol=input_protocol.name,
            client_protocol=input_protocol.name,
            metadata={"operation": requested_operation} if requested_operation else None,
        )
        unified_request = input_protocol.parse_request(protocol_request, protocol_context)
        if input_protocol.name != "openai_chat" and requested_operation != "embeddings":
            # Embeddings skip the chat rebuild: an embeddings payload from
            # a non-openai ingress (ollama /api/embed) must stay in ITS
            # wire shape — the native path builds from the protocol
            # snapshot, and a litellm fallback dispatches aembedding with
            # the ollama shape litellm's own handler expects. A chat
            # rebuild would hand both a {messages: []} body.
            kwargs = get_protocol("openai_chat").build_request(
                unified_request,
                ProtocolContext(
                    source_protocol=input_protocol.name,
                    target_protocol="openai_chat",
                    input_protocol=input_protocol.name,
                    provider_protocol="openai_chat",
                    client_protocol=input_protocol.name,
                ),
            )
        session_isolation_key = self._session_isolation_key(
            classifier,
            request_api_keys,
            request_providers,
            private,
        )
        model = kwargs.get("model", "")
        # Model-string arguments (``model:high``): split before routing so
        # aliases, groups, and anchors key on the clean id; the hint applies
        # to the canonical reasoning control only when the client set none.
        clean_model, model_args = split_model_args(model)
        if model_args:
            kwargs["model"] = clean_model
            if getattr(unified_request, "model", None):
                unified_request.model = clean_model
            apply_model_args_to_unified(unified_request, model_args)
        routing_decision = self._resolve_routing_decision(clean_model)
        routing_targets = routing_decision.targets if routing_decision else None
        provider = routing_targets[0].provider if routing_targets else self._provider_from_model(model)
        if not provider:
            self._raise_no_provider(model)

        scope = await self._resolve_scope_for_provider(
            provider,
            classifier,
            request_api_keys,
            request_providers,
            private,
        )
        if not scope["credentials"]:
            self._raise_no_provider(model)

        if routing_targets:
            # Skip-and-record (D17 breaker): a later target without
            # credentials is skipped, not fatal — the chain proceeds and
            # only a fully unserviceable decision raises.
            scoped_targets = []
            skipped: list[str] = []
            for index, target in enumerate(routing_targets):
                target_scope = scope if index == 0 else await self._resolve_scope_for_provider(
                    target.provider,
                    classifier,
                    request_api_keys,
                    request_providers,
                    private,
                )
                if not target_scope["credentials"]:
                    skipped.append(target.prefixed_model)
                    continue
                scoped_targets.append(self._with_request_scope(target, target_scope))
            if skipped:
                logging.getLogger("rotator_library").warning(
                    "Skipping %d fallback target(s) without credentials: %s",
                    len(skipped),
                    ", ".join(skipped),
                )
            if not scoped_targets:
                self._raise_no_provider(model)
            routing_targets = tuple(scoped_targets)

        resolved_model = self._model_resolver.resolve_model_id(routing_targets[0].prefixed_model if routing_targets else model, provider)
        kwargs["model"] = resolved_model

        transaction_logger = None
        if self._get_enable_request_logging():
            transaction_logger = TransactionLogger(
                provider=provider,
                model=resolved_model,
                enabled=True,
                parent_dir=parent_log_dir,
            )
            transaction_logger.log_request(kwargs)

        session = self._session_tracker.infer_session(
            kwargs,
            provider=provider,
            model=resolved_model,
            scope_key=session_isolation_key,
            hints=self._merge_session_hints(
                internal_session_hints,
                await self._get_session_hints(
                    provider,
                    resolved_model,
                    kwargs,
                    unified_request=unified_request,
                    input_protocol=input_protocol.name,
                ),
            ),
            _trusted_isolation_key=True,
        )
        if transaction_logger:
            transaction_logger.set_trace_context(
                session_id=session.session_id,
                scope_key=scope["usage_manager_key"],
                classifier=scope["classifier"],
            )

        # Provider enrichment: routing has resolved the provider — fill the
        # run's identity and extend its bindings with the provider's class
        # and config hooks. One request, one run: the same object the entry
        # stage fired on and the native executor will use.
        from ..hooks.binding import resolve_hook_declarations as _resolve_decls

        plugin = self._get_provider_instance(provider) if self._get_provider_instance else None
        provider_class_hooks, provider_config_hooks, _ = _resolve_decls(
            plugin, resolved_model, config=self._experimental_config, provider=provider
        )
        pipeline_run.enrich(
            provider=provider,
            model=resolved_model,
            session_id=getattr(session, "session_id", "") or "",
            scope_key=session_isolation_key or "",
            classifier=scope.get("classifier") or "",
            operation="chat",
            class_hooks=provider_class_hooks,
            config_hooks=provider_config_hooks,
        )
        # R2 routing decision is stamped; hooks observe the resolved targets.
        await run_slot(
            pipeline_run,
            "routing_resolved",
            {
                "targets": [
                    {
                        "provider": target.provider,
                        "model": target.prefixed_model,
                        "protocol": target.protocol,
                        "profile": target.profile,
                        "execution": target.execution,
                    }
                    for target in (routing_targets or ())
                ]
            },
            direction="request",
        )
        # R4 session evidence is settled; a session_id rewrite is applied
        # (cheap and safe — it feeds every identity sink downstream).
        session_outcome = await run_slot(
            pipeline_run,
            "session_resolved",
            {
                "session_id": session.session_id,
                "confidence": getattr(session, "confidence", None),
            },
            direction="request",
        )
        resolved_session_id = session.session_id
        if session_outcome.modified and isinstance(session_outcome.payload, dict):
            rewritten_session_id = session_outcome.payload.get("session_id")
            if rewritten_session_id:
                resolved_session_id = rewritten_session_id

        return RequestContext(
            model=resolved_model,
            provider=provider,
            execution_profile=requested_profile,
            kwargs=kwargs,
            streaming=kwargs.get("stream", False),
            credentials=scope["credentials"],
            deadline=time.time() + self._get_global_timeout(),
            session_id=resolved_session_id,
            session_affinity_key=session.affinity_key,
            session_tracker=self._session_tracker,
            session_tracking_namespace=session.tracking_namespace,
            session_isolation_key=session_isolation_key,
            request=request,
            pre_request_callback=pre_request_callback,
            transaction_logger=transaction_logger,
            usage_manager_key=scope["usage_manager_key"],
            provider_config=scope["provider_config"],
            credential_secrets=scope["credential_secrets"],
            classifier=scope["classifier"],
            routing_targets=routing_targets,
            routing_group_name=routing_decision.group_name if routing_decision else None,
            input_protocol_name=input_protocol.name,
            protocol_request=protocol_request,
            unified_request=unified_request,
            input_provider=provider,
            requested_operation=requested_operation,
            disable_provider_continuation=disable_provider_continuation,
            routing_group=routing_decision.group if routing_decision else None,
            pipeline_run=pipeline_run,
        )

    def _mint_pipeline_run(
        self,
        provider: str,
        model: str,
        session: Any,
        session_isolation_key: Optional[str],
        scope: Dict[str, Any],
    ) -> Any:
        """Mint the single per-request run from provider class/config hooks."""

        plugin = self._get_provider_instance(provider) if self._get_provider_instance else None
        return make_pipeline_run(
            plugin,
            provider=provider,
            model=model,
            session_id=getattr(session, "session_id", "") or "",
            scope_key=session_isolation_key or "",
            classifier=scope.get("classifier") or "",
            operation="chat",
            config=self._experimental_config,
        )

    async def build_embedding_context(
        self,
        request: Optional[Any],
        pre_request_callback: Optional[Callable],
        kwargs: Dict[str, Any],
    ) -> RequestContext:
        """Build an embeddings request context — the same treatment as
        completion (G9): the payload is parsed by the openai_embeddings
        protocol into canonical form, rides the full routing chain
        (fallback groups, scoped targets), gets its transaction logger,
        its pipeline run (operation="embeddings"), and the same hook
        stages. Targets without an embeddings surface are skipped with a
        warning — a mixed fallback group routes embeddings to whoever
        can serve them.
        """
        classifier, request_api_keys, request_providers, private, internal_session_hints = self._pop_scope_kwargs(
            kwargs
        )
        parent_log_dir = kwargs.pop("_parent_log_dir", None)
        input_protocol = get_protocol("openai_embeddings")
        # R1 client entry — identical contract to completions: hook edits
        # on the live payload are authoritative.
        pipeline_run = make_pipeline_run(None, operation="embeddings", config=self._experimental_config)
        entry_outcome = await run_slot(pipeline_run, "request_received", kwargs, direction="request", copy_payload=False)
        if entry_outcome.modified and isinstance(entry_outcome.payload, dict):
            edited = entry_outcome.payload
            kwargs.clear()
            kwargs.update(edited)
            pipeline_run.record_overlay("request_received_edit", stage="request_received", direction="request")
        # D13 grammar applies unchanged (profiles may serve embeddings).
        requested_profile = _normalize_profile_reference(kwargs)
        protocol_request = deepcopy(kwargs)
        protocol_context = ProtocolContext(
            source_protocol=input_protocol.name,
            target_protocol=input_protocol.name,
            input_protocol=input_protocol.name,
            client_protocol=input_protocol.name,
            metadata={"operation": "embeddings"},
        )
        unified_request = input_protocol.parse_request(protocol_request, protocol_context)
        # G9 contract validation BEFORE any routing or credential work: a
        # malformed embeddings payload is the client's 400 — it must never
        # burn rotation attempts looking for a credential that cannot fix
        # it. The route ladder renders the ProtocolError (a ValueError) as
        # the openai-shaped invalid_request envelope (400).
        validate_embeddings_request(unified_request, input_protocol.name)
        session_isolation_key = self._session_isolation_key(
            classifier,
            request_api_keys,
            request_providers,
            private,
        )
        model = kwargs.get("model", "")
        clean_model, model_args = split_model_args(model)
        if model_args:
            kwargs["model"] = clean_model
            if getattr(unified_request, "model", None):
                unified_request.model = clean_model
        routing_decision = self._resolve_routing_decision(clean_model)
        routing_targets = routing_decision.targets if routing_decision else None
        if not routing_targets:
            provider = self._provider_from_model(model)
            if not provider:
                self._raise_no_provider(model)
            scope = await self._resolve_scope_for_provider(
                provider,
                classifier,
                request_api_keys,
                request_providers,
                private,
            )
            if not scope["credentials"]:
                self._raise_no_provider(model)
        else:
            provider = routing_targets[0].provider
            scope = await self._resolve_scope_for_provider(
                provider,
                classifier,
                request_api_keys,
                request_providers,
                private,
            )

        if routing_targets:
            # Skip-and-record for BOTH no-credentials and no-embeddings-
            # surface targets (D17 breaker shape): a mixed group routes
            # embeddings to whoever can serve them. The FIRST target is
            # not special — a group whose head lacks credentials or an
            # embeddings surface falls through to the next target, and
            # identity (provider, scope, logger, session) binds to the
            # SURVIVING head, never the skipped one.
            scoped_targets = []
            skipped: list[str] = []
            for index, target in enumerate(routing_targets):
                target_scope = scope if index == 0 else await self._resolve_scope_for_provider(
                    target.provider,
                    classifier,
                    request_api_keys,
                    request_providers,
                    private,
                )
                reason = None
                if not target_scope["credentials"]:
                    reason = "no credentials"
                else:
                    target_plugin = self._get_provider_instance(target.provider) if self._get_provider_instance else None
                    if target_plugin is not None and not _target_serves_embeddings(target_plugin):
                        reason = "no embeddings surface"
                if reason:
                    skipped.append(f"{target.prefixed_model} ({reason})")
                    continue
                if index == 0 and target_scope is not scope:
                    scope = target_scope
                scoped_targets.append(self._with_request_scope(target, target_scope))
            if skipped:
                logging.getLogger("rotator_library").warning(
                    "Skipping %d fallback target(s) for embeddings: %s",
                    len(skipped),
                    ", ".join(skipped),
                )
            if not scoped_targets:
                raise ModelReferenceError(
                    "No fallback target can serve an embeddings request "
                    f"(model {model!r}): every target lacks credentials or an "
                    "embeddings surface"
                )
            routing_targets = tuple(scoped_targets)
            provider = routing_targets[0].provider
            scope = await self._resolve_scope_for_provider(
                provider,
                classifier,
                request_api_keys,
                request_providers,
                private,
            )

        resolved_model = self._model_resolver.resolve_model_id(routing_targets[0].prefixed_model if routing_targets else model, provider)
        kwargs["model"] = resolved_model

        transaction_logger = None
        if self._get_enable_request_logging():
            transaction_logger = TransactionLogger(
                provider=provider,
                model=resolved_model,
                enabled=True,
                parent_dir=parent_log_dir,
            )
            transaction_logger.log_request(kwargs)

        session = self._session_tracker.infer_session(
            kwargs,
            provider=provider,
            model=resolved_model,
            scope_key=session_isolation_key,
            hints=self._merge_session_hints(
                internal_session_hints,
                await self._get_session_hints(
                    provider,
                    resolved_model,
                    kwargs,
                    unified_request=unified_request,
                    input_protocol=input_protocol.name,
                ),
            ),
            _trusted_isolation_key=True,
        )
        if transaction_logger:
            transaction_logger.set_trace_context(
                session_id=session.session_id,
                scope_key=scope["usage_manager_key"],
                classifier=scope["classifier"],
            )

        from ..hooks.binding import resolve_hook_declarations as _resolve_decls

        plugin = self._get_provider_instance(provider) if self._get_provider_instance else None
        provider_class_hooks, provider_config_hooks, _ = _resolve_decls(
            plugin, resolved_model, config=self._experimental_config, provider=provider
        )
        pipeline_run.enrich(
            provider=provider,
            model=resolved_model,
            session_id=getattr(session, "session_id", "") or "",
            scope_key=session_isolation_key or "",
            classifier=scope.get("classifier") or "",
            operation="embeddings",
            class_hooks=provider_class_hooks,
            config_hooks=provider_config_hooks,
        )
        await run_slot(
            pipeline_run,
            "routing_resolved",
            {
                "targets": [
                    {
                        "provider": target.provider,
                        "model": target.prefixed_model,
                        "protocol": target.protocol,
                        "profile": target.profile,
                        "execution": target.execution,
                    }
                    for target in (routing_targets or ())
                ]
            },
            direction="request",
        )
        session_outcome = await run_slot(
            pipeline_run,
            "session_resolved",
            {
                "session_id": session.session_id,
                "confidence": getattr(session, "confidence", None),
            },
            direction="request",
        )
        resolved_session_id = session.session_id
        if session_outcome.modified and isinstance(session_outcome.payload, dict):
            rewritten_session_id = session_outcome.payload.get("session_id")
            if rewritten_session_id:
                resolved_session_id = rewritten_session_id

        return RequestContext(
            model=resolved_model,
            provider=provider,
            execution_profile=requested_profile,
            kwargs=kwargs,
            streaming=False,
            credentials=scope["credentials"],
            deadline=time.time() + self._get_global_timeout(),
            session_id=resolved_session_id,
            session_affinity_key=session.affinity_key,
            session_tracker=self._session_tracker,
            session_tracking_namespace=session.tracking_namespace,
            session_isolation_key=session_isolation_key,
            request=request,
            pre_request_callback=pre_request_callback,
            transaction_logger=transaction_logger,
            usage_manager_key=scope["usage_manager_key"],
            provider_config=scope["provider_config"],
            credential_secrets=scope["credential_secrets"],
            classifier=scope["classifier"],
            routing_targets=routing_targets,
            routing_group_name=routing_decision.group_name if routing_decision else None,
            input_protocol_name=input_protocol.name,
            protocol_request=protocol_request,
            unified_request=unified_request,
            input_provider=provider,
            requested_operation="embeddings",
            routing_group=routing_decision.group if routing_decision else None,
            pipeline_run=pipeline_run,
        )


def _target_serves_embeddings(plugin: Any) -> bool:
    """Whether ANY declared face of the provider natively serves embeddings.

    Multi-face providers (openai: responses-default + chat) serve
    embeddings on a NON-default face — probing the default face alone
    would wrongly skip them. Mirrors the executor's face-scan.
    """

    try:
        from .executor import _operation_declared_on_faces

        return _operation_declared_on_faces(plugin, "embeddings")
    except Exception:
        return False


def _normalize_profile_reference(kwargs: Dict[str, Any]) -> Optional[str]:
    """Normalize ``provider:profile/model`` in request kwargs (D13).

    The grammar applies only when the PROVIDER segment (before the first
    ``/``) contains the separator - model segments keep their colons
    (OpenRouter ``:free``, Ollama ``model:tag``). The value written back is
    always the bare ``provider/model`` form: the profile is stripped, the
    provider/profile segments are trimmed by the parser, and the model
    segment is preserved verbatim. Returns the requested profile, or None
    for plain references.
    """

    model = str(kwargs.get("model", "") or "")
    if not model or PROFILE_SEPARATOR not in model.split("/", 1)[0]:
        return None
    from ..routing.profiles import parse_model_reference

    reference = parse_model_reference(model)
    if reference.profile:
        # provider:profile/model -> provider/model: every identity sink
        # downstream (usage, cooldowns, sessions, cache scopes) sees only
        # the stripped provider-level form.
        kwargs["model"] = reference.bare
        return reference.profile
    return None
