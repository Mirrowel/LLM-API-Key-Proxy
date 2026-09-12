"""Hook registration and binding resolution.

Three registration surfaces with a stable precedence order (ties within a
slot break by this order, after priority):

1. Provider class attribute ``hooks`` (declaration order preserved)
2. Provider JSON config ``hooks`` (list of names/references into the global
   registry, optionally with per-binding overrides)
3. Global registry (registration order)

Unknown names fail at startup, never per-request (audit ruling).

The registry stores declarations only — classes or factories. Live state
never lives here; per-request state lives on PipelineRun (runner.py).
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .types import (
    CallbackCallable,
    DEFAULT_HOOK_PRIORITY,
    HookCallable,
    HookContext,
    PipelineCallback,
    PipelineHook,
    StageEvent,
    StageInvocation,
    HOOK_STAGES,
)

__all__ = [
    "register_hook",
    "register_callback",
    "get_hook",
    "list_hooks",
    "resolve_stage_bindings",
    "HookBinding",
    "validate_declared_names",
]


class HookBinding:
    """A resolved, ordered binding of one hook/callback to one stage."""

    __slots__ = ("owner", "name", "call", "priority", "critical", "stateful",
                 "kind", "sequence", "source")

    def __init__(
        self,
        owner: Any,
        name: str,
        call: Callable,
        priority: int,
        critical: bool,
        stateful: bool,
        kind: str,
        sequence: int,
        source: str,
    ) -> None:
        self.owner = owner
        self.name = name
        self.call = call
        self.priority = priority
        self.critical = critical
        self.stateful = stateful
        self.kind = kind  # "hook" | "callback"
        self.sequence = sequence
        self.source = source  # "class" | "config" | "global"

    def order_key(self) -> Tuple[int, int]:
        return (self.priority, self.sequence)


_HOOK_PLUGINS: Dict[str, Any] = {}
_HOOK_ALIASES: Dict[str, str] = {}
_SEQUENCE = [0]


def _canonical_name(cls_or_factory: Any, declared: str) -> str:
    return declared or getattr(cls_or_factory, "name", "") or getattr(cls_or_factory, "__name__", "anonymous_hook")


def register_hook(obj: Any, *, replace: bool = False, name: str = "") -> None:
    """Register a hook class/factory/callable in the global registry.

    ``obj`` may be a PipelineHook subclass, a PipelineHook instance factory,
    or a bare async callable treated as a stateless hook bound to whatever
    ``stages`` it declares (callable: attach ``stages``/``priority`` attrs).
    """
    declared = _canonical_name(obj, name)
    if declared in _HOOK_PLUGINS and not replace:
        raise ValueError(f"hook '{declared}' is already registered; pass replace=True to override")
    if declared in _HOOK_ALIASES and _HOOK_ALIASES[declared] != declared and not replace:
        raise ValueError(f"hook name '{declared}' collides with an alias")
    if replace:
        for alias, target in list(_HOOK_ALIASES.items()):
            if target == declared or alias == declared:
                _HOOK_ALIASES.pop(alias, None)
        _HOOK_PLUGINS.pop(declared, None)
    _SEQUENCE[0] += 1
    _HOOK_PLUGINS[declared] = obj
    for alias in getattr(obj, "aliases", ()) or ():
        existing = _HOOK_ALIASES.get(alias)
        if existing and existing != declared and not replace:
            raise ValueError(f"hook alias '{alias}' already bound to '{existing}'")
        _HOOK_ALIASES[alias] = declared


def register_callback(obj: Any, *, replace: bool = False, name: str = "") -> None:
    """Register a callback (listener). Same mechanics as register_hook."""
    if not getattr(obj, "kind_marker", None):
        kind_marker = "callback"
        if isinstance(obj, type):
            obj.kind_marker = kind_marker
    register_hook(obj, replace=replace, name=name)


def get_hook(name: str) -> Any:
    resolved = _HOOK_ALIASES.get(name, name)
    if resolved not in _HOOK_PLUGINS:
        raise KeyError(f"unknown hook '{name}' — registered: {sorted(_HOOK_PLUGINS)}")
    return _HOOK_PLUGINS[resolved]


def list_hooks() -> List[str]:
    return sorted(_HOOK_PLUGINS)


def _binding_for(obj: Any, source: str, stage: str, overrides: Optional[dict]) -> Optional[HookBinding]:
    is_cb = isinstance(obj, PipelineCallback) or (
        isinstance(obj, type) and issubclass(obj, PipelineCallback)
    ) or getattr(obj, "kind_marker", "") == "callback"
    kind = "callback" if is_cb else "hook"
    base_cls = PipelineCallback if is_cb else PipelineHook
    if isinstance(obj, base_cls):
        instance = obj
        call = obj
        priority = getattr(obj, "priority", DEFAULT_HOOK_PRIORITY)
        critical = bool(getattr(obj, "critical", False))
        stateful = bool(getattr(obj, "stateful", False))
    else:
        call = obj
        priority = getattr(obj, "priority", DEFAULT_HOOK_PRIORITY)
        critical = bool(getattr(obj, "critical", False))
        stateful = bool(getattr(obj, "stateful", False))
    if overrides:
        priority = int(overrides.get("priority", priority))
        critical = bool(overrides.get("critical", critical))
    _SEQUENCE[0] += 1
    return HookBinding(
        owner=obj,
        name=getattr(obj, "name", "") or getattr(obj, "__name__", "hook"),
        call=call,
        priority=int(priority),
        critical=critical,
        stateful=stateful,
        kind=kind,
        sequence=_SEQUENCE[0],
        source=source,
    )


def resolve_stage_bindings(
    stage: str,
    *,
    class_hooks: Sequence[Any] = (),
    config_hooks: Sequence[Any] = (),
    global_hooks: Optional[Sequence[str]] = None,
) -> Tuple[List[HookBinding], List[HookBinding]]:
    """Resolve the ordered (hooks, callbacks) bound to one stage.

    Ordering: priority ascending, ties by registration order
    (class -> config -> global). Callbacks are separated out; the runner
    fires them after the hook chain settles.
    """
    if stage not in HOOK_STAGES:
        raise ValueError(f"unknown stage '{stage}'")
    hooks: List[HookBinding] = []
    callbacks: List[HookBinding] = []

    def _accept(obj: Any, source: str, overrides: Optional[dict]) -> None:
        stages = getattr(obj, "stages", None) or ()
        # config entries may carry inline stage targeting
        if overrides and "stages" in overrides:
            stages = overrides["stages"]
        if isinstance(stages, str):
            stages = (stages,)
        if stage not in stages:
            return
        binding = _binding_for(obj, source, stage, overrides)
        (callbacks if binding.kind == "callback" else hooks).append(binding)

    for entry in class_hooks:
        overrides = None
        obj = entry
        if isinstance(entry, dict):
            obj = get_hook(entry.get("name", ""))
            overrides = {k: v for k, v in entry.items() if k != "name"}
        _accept(obj, "class", overrides)
    for entry in config_hooks:
        overrides = None
        obj = entry
        if isinstance(entry, dict):
            obj = get_hook(entry.get("name", ""))
            overrides = {k: v for k, v in entry.items() if k != "name"}
        _accept(obj, "config", overrides)
    for name in global_hooks or ():
        obj = get_hook(name)
        _accept(obj, "global", None)

    hooks.sort(key=lambda b: b.order_key())
    callbacks.sort(key=lambda b: b.order_key())
    return hooks, callbacks


def validate_declared_names(
    *,
    class_hooks: Sequence[Any] = (),
    config_hooks: Sequence[Any] = (),
    global_hooks: Sequence[str] = (),
) -> None:
    """Startup validation: every referenced name must resolve (audit ruling).

    Raises KeyError naming the first unknown binding; called at provider
    registration / config load time so a typo fails the process at startup,
    never a request later.
    """
    for entry in list(class_hooks) + list(config_hooks):
        if isinstance(entry, dict):
            name = entry.get("name", "")
            stages = entry.get("stages") or ()
            if isinstance(stages, str):
                stages = (stages,)
            for stage in stages:
                if stage not in HOOK_STAGES:
                    raise ValueError(
                        f"hook '{name}' declares unknown stage '{stage}' "
                        f"(valid: {sorted(HOOK_STAGES)})"
                    )
            get_hook(name)
        else:
            stages = getattr(entry, "stages", None) or ()
            if isinstance(stages, str):
                stages = (stages,)
            for stage in stages:
                if stage not in HOOK_STAGES:
                    raise ValueError(
                        f"hook '{getattr(entry, 'name', entry)}' declares unknown stage '{stage}' "
                        f"(valid: {sorted(HOOK_STAGES)})"
                    )
    for name in global_hooks or ():
        get_hook(name)
