"""Per-request pipeline runs and slot execution.

``PipelineRun`` is the isolation unit: one per request, owning the state
bag, stateful hook instances, overlay records, and the boundary-event log.
Two concurrent requests to the same provider get two runs that cannot see
each other — the road-per-car model (user directive).

``run_slot`` executes one declared boundary: it resolves the ordered
bindings, runs the hook chain (take-modify-return), contains failures
(keep-payload + warn + trace, unless ``critical``), records a boundary
event for every hop (nothing is invisible), and then fires callbacks with
the settled snapshot. The transaction logger consumes boundary events
natively — it is not a participant and has no priority.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .registry import HookBinding, resolve_stage_bindings
from .types import (
    DEFAULT_HOOK_PRIORITY,
    HookAction,
    HookContext,
    HookResult,
    PipelineHook,
    StageEvent,
    StageInvocation,
    TransportView,
)

__all__ = ["PipelineRun", "run_slot", "BoundaryRecord", "SlotOutcome"]

logger = logging.getLogger("rotator_library.hooks")


@dataclass
class BoundaryRecord:
    """One observed hop at one boundary — the native trace emission."""

    stage: str
    direction: str
    kind: str  # "before" | "hop" | "after" | "contained" | "callback"
    source: str = ""      # binding source for hops
    name: str = ""        # hook/callback name
    changed: bool = False
    action: str = ""
    detail: str = ""
    event_index: Optional[int] = None


@dataclass
class SlotOutcome:
    """Result of running one slot."""

    payload: Any = None
    action: HookAction = HookAction.CONTINUE
    #: For BLOCK/RESPOND verdicts.
    message: Optional[str] = None
    error_type: Optional[str] = None
    #: True when the payload was rewritten by any hop.
    modified: bool = False


class PipelineRun:
    """Per-request isolation unit for hook execution."""

    def __init__(
        self,
        *,
        request_id: str = "",
        provider: str = "",
        model: str = "",
        credential_id: str = "",
        session_id: str = "",
        scope_key: str = "",
        classifier: str = "",
        operation: str = "",
        class_hooks: Sequence[Any] = (),
        config_hooks: Sequence[Any] = (),
        global_hooks: Optional[Sequence[str]] = None,
        boundary_recorder: Optional[Callable[[BoundaryRecord], None]] = None,
    ) -> None:
        self.context = HookContext(
            request_id=request_id,
            provider=provider,
            model=model,
            credential_id=credential_id,
            session_id=session_id,
            scope_key=scope_key,
            classifier=classifier,
            operation=operation,
        )
        self._class_hooks = tuple(class_hooks)
        self._config_hooks = tuple(config_hooks)
        self._global_hooks = tuple(global_hooks or ())
        self._boundary_recorder = boundary_recorder
        #: Per-request instances of stateful hooks (never shared).
        self._instances: Dict[int, Any] = {}
        self._binding_instances: Dict[int, Any] = {}
        #: Recorded rewrites (transport overlays etc.) for traces.
        self.overlays: List[Dict[str, Any]] = []
        #: Every boundary this run passed through.
        self.boundaries: List[BoundaryRecord] = []

    # -- boundary log ------------------------------------------------------

    def record(self, record: BoundaryRecord) -> None:
        self.boundaries.append(record)
        if self._boundary_recorder is not None:
            try:
                self._boundary_recorder(record)
            except Exception:  # recording must never fail a request
                logger.debug("boundary recorder failed", exc_info=True)

    def record_overlay(self, kind: str, **fields: Any) -> None:
        entry = {"kind": kind, **fields}
        self.overlays.append(entry)
        self.record(BoundaryRecord(stage=fields.get("stage", ""), direction=fields.get("direction", ""),
                                   kind="overlay", name=kind, changed=True,
                                   detail=str(fields)[:200]))

    def enrich(
        self,
        *,
        provider: str = "",
        model: str = "",
        session_id: str = "",
        scope_key: str = "",
        classifier: str = "",
        credential_id: str = "",
        operation: str = "",
        class_hooks: Sequence[Any] = (),
        config_hooks: Sequence[Any] = (),
    ) -> None:
        """Fill in identity and extend bindings once routing resolves them.

        A run may be minted bare at client entry (provider unknown, only
        global/config hooks bound) and enriched after provider resolution —
        stages fired before enrichment see only the early bindings, which is
        the honest semantic: ``request_received`` precedes provider
        selection, so provider-bound hooks cannot intercept it.
        """

        ctx = self.context
        if provider:
            ctx.provider = provider
        if model:
            ctx.model = model
        if session_id:
            ctx.session_id = session_id
        if scope_key:
            ctx.scope_key = scope_key
        if classifier:
            ctx.classifier = classifier
        if credential_id:
            ctx.credential_id = credential_id
        if operation:
            ctx.operation = operation
        if class_hooks:
            self._class_hooks = tuple(self._class_hooks) + tuple(class_hooks)
        if config_hooks:
            self._config_hooks = tuple(self._config_hooks) + tuple(config_hooks)

    # -- binding resolution -------------------------------------------------

    def _instance_for(self, binding: HookBinding) -> Any:
        """Resolve the invocable for a binding, honoring stateful isolation.

        - classes are instantiated (once per run for stateful, shared for
          stateless — safe because stateless hooks keep no per-request data)
        - instances pass through when stateless; stateful instances are
          re-minted via their zero-arg constructor per run
        - bare callables pass through; stateful callables are factories
          (called with no arguments per run to mint the instance)
        """
        if not binding.stateful and not inspect.isclass(binding.owner):
            return binding.call
        key = id(binding.owner)
        if not binding.stateful:
            # stateless class/factory: one shared invocable, cached on the run
            if key not in self._binding_instances:
                self._binding_instances[key] = self._mint(binding.owner)
            return self._binding_instances[key]
        if key not in self._binding_instances:
            self._binding_instances[key] = self._mint(binding.owner)
        return self._binding_instances[key]

    @staticmethod
    def _mint(owner: Any) -> Any:
        if inspect.isclass(owner):
            return owner()
        if isinstance(owner, (PipelineHook,)) or hasattr(owner, "__dict__") and callable(owner) and not inspect.isfunction(owner):
            # stateful instance: re-mint a fresh same-class instance
            return type(owner)()
        return owner()  # factory callable


async def _invoke(call: Any, *args: Any) -> Any:
    result = call(*args)
    if inspect.isawaitable(result):
        result = await result
    return result


async def run_slot(
    run: PipelineRun,
    stage: str,
    payload: Any,
    *,
    direction: str = "request",
    transport: Optional[TransportView] = None,
    is_terminal: bool = False,
    event_index: Optional[int] = None,
    copy_payload: bool = True,
) -> SlotOutcome:
    """Execute one declared boundary on ``run``.

    Containment: an exception in a hook keeps the original payload, logs a
    warning, and records a ``contained`` boundary — unless the binding is
    ``critical``. Payloads are deep-copied before the first mutating hop so
    a crashed hook cannot leave a half-edited object behind.
    """
    hooks, callbacks = resolve_stage_bindings(
        stage,
        class_hooks=run._class_hooks,
        config_hooks=run._config_hooks,
        global_hooks=run._global_hooks,
    )
    run.record(BoundaryRecord(stage=stage, direction=direction, kind="before",
                              event_index=event_index))
    if not hooks and not callbacks:
        return SlotOutcome(payload=payload, modified=False)

    working = copy.deepcopy(payload) if copy_payload else payload
    current = working
    modified = False
    hops: List[str] = []

    for binding in hooks:
        call = run._instance_for(binding)
        invocation = StageInvocation(
            stage=stage,
            payload=current,
            direction=direction,
            transport=transport,
            is_terminal=is_terminal,
            event_index=event_index,
        )
        before_repr = current if not isinstance(current, (dict, list)) else None
        try:
            result = await _invoke(call, invocation, run.context)
        except Exception as exc:
            if binding.critical:
                run.record(BoundaryRecord(stage=stage, direction=direction, kind="contained",
                                          source=binding.source, name=binding.name,
                                          detail=f"critical hook failed: {exc}", event_index=event_index))
                raise
            logger.warning(
                "hook '%s' (stage %s, source %s) failed and was contained: %s",
                binding.name, stage, binding.source, exc, exc_info=True,
            )
            run.record(BoundaryRecord(stage=stage, direction=direction, kind="contained",
                                      source=binding.source, name=binding.name,
                                      detail=str(exc)[:200], event_index=event_index))
            continue

        hop_changed = False
        if isinstance(result, HookResult):
            if result.action is HookAction.CONTINUE:
                if result.payload is not None and result.payload is not current:
                    current = result.payload
                    hop_changed = True
            elif result.action in (HookAction.BLOCK, HookAction.RESPOND):
                run.record(BoundaryRecord(stage=stage, direction=direction, kind="hop",
                                          source=binding.source, name=binding.name, changed=True,
                                          action=result.action.value, event_index=event_index))
                run.record(BoundaryRecord(stage=stage, direction=direction, kind="after",
                                          event_index=event_index))
                return SlotOutcome(
                    payload=result.payload,
                    action=result.action,
                    message=result.message,
                    error_type=result.error_type,
                    modified=True,
                )
            elif result.action is HookAction.DROP:
                run.record(BoundaryRecord(stage=stage, direction=direction, kind="hop",
                                          source=binding.source, name=binding.name, changed=True,
                                          action="drop", event_index=event_index))
                run.record(BoundaryRecord(stage=stage, direction=direction, kind="after",
                                          event_index=event_index))
                return SlotOutcome(payload=None, action=HookAction.DROP, modified=True)
            elif result.action is HookAction.REPLACE:
                current = result.payload
                hop_changed = True
                run.record(BoundaryRecord(stage=stage, direction=direction, kind="hop",
                                          source=binding.source, name=binding.name, changed=True,
                                          action="replace", event_index=event_index))
                continue
        elif result is not None and result is not current:
            current = result
            hop_changed = True

        if hop_changed:
            modified = True
            hops.append(f"{binding.source}:{binding.name}")
        run.record(BoundaryRecord(stage=stage, direction=direction, kind="hop",
                                  source=binding.source, name=binding.name,
                                  changed=hop_changed, event_index=event_index,
                                  action="continue"))

    # transport overlays: nothing invisible (user directive)
    if transport is not None and transport.changed:
        run.record_overlay(
            "transport_rewrite", stage=stage, direction=direction,
            endpoint=transport.endpoint, headers=dict(transport.headers or {}),
            timeout_seconds=transport.timeout_seconds,
        )

    run.record(BoundaryRecord(stage=stage, direction=direction, kind="after",
                              event_index=event_index))

    # Callbacks observe the settled snapshot, contained, after the chain.
    settled = current
    for binding in callbacks:
        call = run._instance_for(binding)
        event = StageEvent(stage=stage, payload=settled, direction=direction,
                           is_terminal=is_terminal, event_index=event_index,
                           hook_hops=tuple(hops))
        try:
            await _invoke(call, event, run.context)
        except Exception as exc:
            logger.warning(
                "callback '%s' (stage %s) failed and was contained: %s",
                binding.name, stage, exc, exc_info=True,
            )
            run.record(BoundaryRecord(stage=stage, direction=direction, kind="callback",
                                      source=binding.source, name=binding.name,
                                      detail=f"contained: {str(exc)[:150]}", event_index=event_index))
        else:
            run.record(BoundaryRecord(stage=stage, direction=direction, kind="callback",
                                      source=binding.source, name=binding.name,
                                      event_index=event_index))

    return SlotOutcome(payload=current, modified=modified)
