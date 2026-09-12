# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Adapters-as-hooks compatibility bridge (G2 migration seam).

``PayloadAdapter`` instances (or adapter names from the adapter registry)
become ``PipelineHook`` bindings without rewriting the adapter system. The
native executor still runs the raw adapter chain at its historical
positions; this bridge is the migration path proving adapter logic rides
the declared hook slots cleanly — it is NOT a replacement.

Stage mapping (adapter semantics preserved, payload types match):

    request        -> ``mutated``           (post adapter band, provider wire)
    response       -> ``response_received`` (raw provider wire, pre-parse)
    stream_event   -> ``stream_event``      (neutral parsed stream event)

Every adapter registered through ``adapters.registry.register_adapter`` is
mirrored into the global hooks registry under ``adapter:<name>`` (with
``adapter:<alias>`` hook aliases), so adapters are declarable from provider
class attributes, JSON config, and the global hook registry with the same
collision rules the adapter registry already enforces.

Intended use: incremental migration of provider payloads from
``adapter_names`` lists to hook declarations. Danger note: like every hook,
the bridge has FULL read/write power over the live stage payload — a
misbehaving adapter corrupts the request/response in flight. Bridge hooks
default to ``critical`` so adapter exceptions fail the request, matching
``run_adapter_chain`` fail-fast semantics instead of shipping an unadapted
payload silently.

Deferred imports of the adapters package keep this module importable from
``adapters.registry`` during auto-discovery without an import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from .types import DEFAULT_HOOK_PRIORITY, HookContext, PipelineHook, StageInvocation

if TYPE_CHECKING:  # pragma: no cover
    from ..adapters.base import AdapterContext, PayloadAdapter

__all__ = [
    "ADAPTER_STAGE_TO_HOOK_STAGE",
    "AdapterHookBridge",
    "adapters_compatible_hook",
]


#: Adapter stage semantics -> declared hook stage (the stable public mapping).
ADAPTER_STAGE_TO_HOOK_STAGE: Dict[str, str] = {
    "request": "mutated",
    "response": "response_received",
    "stream_event": "stream_event",
}

_HOOK_STAGE_TO_ADAPTER_STAGE = {v: k for k, v in ADAPTER_STAGE_TO_HOOK_STAGE.items()}


class AdapterHookBridge(PipelineHook):
    """Expose one ``PayloadAdapter`` as a ``PipelineHook`` binding.

    The bridge binds only at the hook stages mapped from the adapter's
    ``supported_stages`` and dispatches through ``PayloadAdapter.transform``
    so the adapter's own stage guard and dispatch semantics are preserved.
    Ordering among several adapter bridges follows hook declaration order
    (priority, then declaration sequence) — NOT the historical
    ``adapter_names`` list order; declaring layers own that ordering.
    """

    def __init__(
        self,
        adapter: "PayloadAdapter",
        *,
        protocol: Optional[str] = None,
        transport: str = "http",
        adapter_config: Optional[Dict[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = DEFAULT_HOOK_PRIORITY,
    ) -> None:
        self._adapter = adapter
        self.name = f"adapter:{adapter.name}"
        self.aliases = tuple(
            f"adapter:{alias}" for alias in getattr(adapter, "aliases", ()) or ()
        )
        self.stages = tuple(
            ADAPTER_STAGE_TO_HOOK_STAGE[stage]
            for stage in adapter.supported_stages
            if stage in ADAPTER_STAGE_TO_HOOK_STAGE
        )
        self.priority = priority
        self.critical = True
        self._protocol = protocol
        self._transport = transport
        self._adapter_config: Dict[str, Dict[str, Any]] = dict(adapter_config or {})
        self._metadata: Dict[str, Any] = dict(metadata or {})

    async def __call__(self, invocation: StageInvocation, context: HookContext):
        adapter_stage = _HOOK_STAGE_TO_ADAPTER_STAGE.get(invocation.stage)
        if adapter_stage is None:
            return None
        return await self._adapter.transform(
            adapter_stage,
            invocation.payload,
            self._adapter_context(context),
        )

    def _adapter_context(self, context: HookContext) -> "AdapterContext":
        """Build the adapter context field-by-field from the hook context.

        ``HookContext`` carries the identity fields (provider, model,
        credential, session, scope, classifier, operation); protocol,
        transport, and static adapter config are declaration-time concerns
        carried by the bridge. When no static config was declared, the
        documented ``adapter_config`` state-bag key is consulted so
        registry-declared bridges can still receive per-request config.
        """
        from ..adapters.base import AdapterContext

        config = dict(self._adapter_config)
        if not config:
            config = dict(context.state.get("adapter_config") or {})
        return AdapterContext(
            provider=context.provider or None,
            model=context.model or None,
            protocol=self._protocol,
            credential_id=context.credential_id or None,
            session_id=context.session_id or None,
            scope_key=context.scope_key or None,
            classifier=context.classifier or None,
            transport=self._transport,
            metadata={"operation": context.operation, **self._metadata},
            adapter_config=config,
            transaction_logger=None,
        )


def adapters_compatible_hook(
    name_or_adapter: "Union[str, PayloadAdapter]",
    *,
    protocol: Optional[str] = None,
    transport: str = "http",
    adapter_config: Optional[Dict[str, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    priority: int = DEFAULT_HOOK_PRIORITY,
) -> PipelineHook:
    """Wrap an adapter (instance, class, or registered name) as a hook.

    Names resolve through the adapter registry (aliases included) and share
    the registry's instance cache; instances and classes are used directly.
    The returned bridge registers as ``adapter:<name>`` with
    ``adapter:<alias>`` aliases when mirrored into the hooks registry by
    ``register_adapter``.
    """
    from ..adapters.base import PayloadAdapter

    adapter: "PayloadAdapter"
    if isinstance(name_or_adapter, PayloadAdapter):
        adapter = name_or_adapter
    elif isinstance(name_or_adapter, type) and issubclass(name_or_adapter, PayloadAdapter):
        adapter = name_or_adapter()
    else:
        from ..adapters.registry import get_adapter

        adapter = get_adapter(str(name_or_adapter))
    return AdapterHookBridge(
        adapter,
        protocol=protocol,
        transport=transport,
        adapter_config=adapter_config,
        metadata=metadata,
        priority=priority,
    )
