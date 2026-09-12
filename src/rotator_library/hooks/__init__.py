"""Hookable execution pipeline: hooks (interceptors) + callbacks (listeners).

Lay entry: heavy exports are lazy to keep the launcher fast (see lazy
import boundaries in project memory).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .adapter_compat import AdapterHookBridge, adapters_compatible_hook
    from .registry import HookBinding, resolve_stage_bindings, validate_declared_names
    from .binding import global_hook_names, make_pipeline_run, resolve_hook_declarations
    from .runner import BoundaryRecord, PipelineRun, SlotOutcome, run_slot
    from .types import (
        CRITICAL_HOOK_PRIORITY,
        DEFAULT_HOOK_PRIORITY,
        EARLY_HOOK_PRIORITY,
        HOOK_STAGES,
        HookAction,
        HookContext,
        HookResult,
        LATE_HOOK_PRIORITY,
        PipelineCallback,
        PipelineHook,
        StageEvent,
        StageInvocation,
        TransportView,
    )

__lazy = {
    "PipelineRun": ("runner", "PipelineRun"),
    "run_slot": ("runner", "run_slot"),
    "SlotOutcome": ("runner", "SlotOutcome"),
    "BoundaryRecord": ("runner", "BoundaryRecord"),
    "PipelineHook": ("types", "PipelineHook"),
    "PipelineCallback": ("types", "PipelineCallback"),
    "HookContext": ("types", "HookContext"),
    "HookResult": ("types", "HookResult"),
    "HookAction": ("types", "HookAction"),
    "StageInvocation": ("types", "StageInvocation"),
    "StageEvent": ("types", "StageEvent"),
    "TransportView": ("types", "TransportView"),
    "HOOK_STAGES": ("types", "HOOK_STAGES"),
    "DEFAULT_HOOK_PRIORITY": ("types", "DEFAULT_HOOK_PRIORITY"),
    "EARLY_HOOK_PRIORITY": ("types", "EARLY_HOOK_PRIORITY"),
    "CRITICAL_HOOK_PRIORITY": ("types", "CRITICAL_HOOK_PRIORITY"),
    "LATE_HOOK_PRIORITY": ("types", "LATE_HOOK_PRIORITY"),
    "register_hook": ("registry", "register_hook"),
    "register_callback": ("registry", "register_callback"),
    "get_hook": ("registry", "get_hook"),
    "list_hooks": ("registry", "list_hooks"),
    "resolve_stage_bindings": ("registry", "resolve_stage_bindings"),
    "validate_declared_names": ("registry", "validate_declared_names"),
    "HookBinding": ("registry", "HookBinding"),
    "make_pipeline_run": ("binding", "make_pipeline_run"),
    "resolve_hook_declarations": ("binding", "resolve_hook_declarations"),
    "global_hook_names": ("binding", "global_hook_names"),
    "AdapterHookBridge": ("adapter_compat", "AdapterHookBridge"),
    "adapters_compatible_hook": ("adapter_compat", "adapters_compatible_hook"),
}

__all__ = list(__lazy)


def __getattr__(name: str):
    try:
        module_name, attr = __lazy[name]
    except KeyError:
        raise AttributeError(name) from None
    import importlib

    module = importlib.import_module(f".{module_name}", __package__)
    return getattr(module, attr)
