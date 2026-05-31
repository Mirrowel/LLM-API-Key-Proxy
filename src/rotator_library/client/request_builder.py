# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""RequestContext construction for RotatingClient public request methods."""

import time
import inspect
from typing import Any, Awaitable, Callable, Dict, Optional

from ..core.types import RequestContext
from ..routing import FallbackResolver, RoutingConfigError, load_routing_config_from_env
from ..routing.types import RouteTarget, RoutingDecision
from ..session_tracking import SessionTrackingHints
from ..transaction_logger import TransactionLogger


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
    ):
        self._resolve_scope_for_provider = resolve_scope_for_provider
        self._model_resolver = model_resolver
        self._session_tracker = session_tracker
        self._get_global_timeout = get_global_timeout
        self._get_enable_request_logging = get_enable_request_logging
        self._get_provider_instance = get_provider_instance

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
    def _provider_from_model(model: str) -> str:
        return model.split("/")[0] if "/" in model else ""

    @staticmethod
    def _raise_no_provider(model: str) -> None:
        raise ValueError(f"Invalid model format or no credentials for provider: {model}")

    def _resolve_routing_decision(self, model: str) -> Optional[RoutingDecision]:
        """Resolve env-configured fallback routing, if any applies."""

        config = load_routing_config_from_env()
        if not config.fallback_groups and not config.model_routes:
            return None
        try:
            decision = FallbackResolver(config).resolve(model)
            if decision.reason == "direct_provider_model" and model.lower() not in config.model_routes:
                return None
            return decision
        except RoutingConfigError:
            if "/" in model:
                return None
            raise

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
        return RouteTarget(
            provider=target.provider,
            model=target.model,
            name=target.name,
            protocol=target.protocol,
            execution=target.execution,
            priority=target.priority,
            weight=target.weight,
            conditions=dict(target.conditions),
            metadata=metadata,
        )

    async def _get_session_hints(
        self,
        provider: str,
        model: str,
        kwargs: Dict[str, Any],
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
            result = hook(kwargs, model=model)
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
        for hint in hints:
            if not hint:
                continue
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
        model = kwargs.get("model", "")
        routing_decision = self._resolve_routing_decision(model)
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
            scoped_targets = []
            for index, target in enumerate(routing_targets):
                target_scope = scope if index == 0 else await self._resolve_scope_for_provider(
                    target.provider,
                    classifier,
                    request_api_keys,
                    request_providers,
                    private,
                )
                if not target_scope["credentials"]:
                    self._raise_no_provider(target.prefixed_model)
                scoped_targets.append(self._with_request_scope(target, target_scope))
            routing_targets = tuple(scoped_targets)

        parent_log_dir = kwargs.pop("_parent_log_dir", None)
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
            scope_key=scope["usage_manager_key"],
            hints=self._merge_session_hints(
                internal_session_hints,
                await self._get_session_hints(provider, resolved_model, kwargs),
            ),
        )
        if transaction_logger:
            transaction_logger.set_trace_context(
                session_id=session.session_id,
                scope_key=scope["usage_manager_key"],
                classifier=scope["classifier"],
            )

        return RequestContext(
            model=resolved_model,
            provider=provider,
            kwargs=kwargs,
            streaming=kwargs.get("stream", False),
            credentials=scope["credentials"],
            deadline=time.time() + self._get_global_timeout(),
            session_id=session.session_id,
            session_affinity_key=session.affinity_key,
            session_tracker=self._session_tracker,
            session_possible_compaction=session.possible_compaction,
            session_lineage_parent_id=session.lineage_parent_session_id,
            session_tracking_namespace=session.tracking_namespace,
            request=request,
            pre_request_callback=pre_request_callback,
            transaction_logger=transaction_logger,
            usage_manager_key=scope["usage_manager_key"],
            provider_config=scope["provider_config"],
            credential_secrets=scope["credential_secrets"],
            classifier=scope["classifier"],
            routing_targets=routing_targets,
            routing_group_name=routing_decision.group_name if routing_decision else None,
            routing_group=routing_decision.group if routing_decision else None,
        )

    async def build_embedding_context(
        self,
        request: Optional[Any],
        pre_request_callback: Optional[Callable],
        kwargs: Dict[str, Any],
    ) -> RequestContext:
        classifier, request_api_keys, request_providers, private, internal_session_hints = self._pop_scope_kwargs(
            kwargs
        )
        model = kwargs.get("model", "")
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

        session = self._session_tracker.infer_session(
            kwargs,
            provider=provider,
            model=model,
            scope_key=scope["usage_manager_key"],
            hints=self._merge_session_hints(
                internal_session_hints,
                await self._get_session_hints(provider, model, kwargs),
            ),
        )

        return RequestContext(
            model=model,
            provider=provider,
            kwargs=kwargs,
            streaming=False,
            credentials=scope["credentials"],
            deadline=time.time() + self._get_global_timeout(),
            session_id=session.session_id,
            session_affinity_key=session.affinity_key,
            session_tracker=self._session_tracker,
            session_possible_compaction=session.possible_compaction,
            session_lineage_parent_id=session.lineage_parent_session_id,
            session_tracking_namespace=session.tracking_namespace,
            request=request,
            pre_request_callback=pre_request_callback,
            usage_manager_key=scope["usage_manager_key"],
            provider_config=scope["provider_config"],
            credential_secrets=scope["credential_secrets"],
            classifier=scope["classifier"],
        )
