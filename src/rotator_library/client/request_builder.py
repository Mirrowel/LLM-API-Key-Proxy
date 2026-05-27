# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""RequestContext construction for RotatingClient public request methods."""

import time
from typing import Any, Awaitable, Callable, Dict, Optional

from ..core.types import RequestContext
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
    ):
        self._resolve_scope_for_provider = resolve_scope_for_provider
        self._model_resolver = model_resolver
        self._session_tracker = session_tracker
        self._get_global_timeout = get_global_timeout
        self._get_enable_request_logging = get_enable_request_logging

    @staticmethod
    def _pop_scope_kwargs(kwargs: Dict[str, Any]) -> tuple[Optional[str], Any, Any, bool]:
        classifier = kwargs.pop("classifier", None)
        request_api_keys = kwargs.pop("api_keys", None)
        request_providers = kwargs.pop("providers", None)
        private = bool(kwargs.pop("private", False))
        kwargs.pop("model_filters", None)
        return classifier, request_api_keys, request_providers, private

    @staticmethod
    def _provider_from_model(model: str) -> str:
        return model.split("/")[0] if "/" in model else ""

    @staticmethod
    def _raise_no_provider(model: str) -> None:
        raise ValueError(f"Invalid model format or no credentials for provider: {model}")

    async def build_completion_context(
        self,
        request: Optional[Any],
        pre_request_callback: Optional[Callable],
        kwargs: Dict[str, Any],
    ) -> RequestContext:
        classifier, request_api_keys, request_providers, private = self._pop_scope_kwargs(
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

        parent_log_dir = kwargs.pop("_parent_log_dir", None)
        resolved_model = self._model_resolver.resolve_model_id(model, provider)
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

        return RequestContext(
            model=resolved_model,
            provider=provider,
            kwargs=kwargs,
            streaming=kwargs.get("stream", False),
            credentials=scope["credentials"],
            deadline=time.time() + self._get_global_timeout(),
            session_id=self._session_tracker.infer_session_id(kwargs),
            request=request,
            pre_request_callback=pre_request_callback,
            transaction_logger=transaction_logger,
            usage_manager_key=scope["usage_manager_key"],
            provider_config=scope["provider_config"],
            credential_secrets=scope["credential_secrets"],
            classifier=scope["classifier"],
        )

    async def build_embedding_context(
        self,
        request: Optional[Any],
        pre_request_callback: Optional[Callable],
        kwargs: Dict[str, Any],
    ) -> RequestContext:
        classifier, request_api_keys, request_providers, private = self._pop_scope_kwargs(
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

        return RequestContext(
            model=model,
            provider=provider,
            kwargs=kwargs,
            streaming=False,
            credentials=scope["credentials"],
            deadline=time.time() + self._get_global_timeout(),
            request=request,
            pre_request_callback=pre_request_callback,
            usage_manager_key=scope["usage_manager_key"],
            provider_config=scope["provider_config"],
            credential_secrets=scope["credential_secrets"],
            classifier=scope["classifier"],
        )
