"""Tri-source hook resolution and per-request PipelineRun minting.

One request owns exactly one ``PipelineRun`` (runner.py). The client entry
stages (request_received/routing_resolved/session_resolved) run on it in
RequestContextBuilder, the credential_selected/executor stages run on it in
the client executor, and the native executor reuses the very same object for
R5-R14/P1-P5/S1-S4. This module is the single place that turns a provider
plugin + runtime config into the declarations a run is built from.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from .runner import PipelineRun

__all__ = ["resolve_hook_declarations", "make_pipeline_run", "global_hook_names"]


def global_hook_names(*, config: Any = None, env: Any = None) -> Tuple[str, ...]:
    """Return configured global hook names, tolerating an unreadable config."""

    try:
        from ..config.experimental import get_global_hook_names

        return tuple(get_global_hook_names(config=config, env=env) or ())
    except Exception:
        return ()


def _provider_config_hooks(plugin: Any, model: str, *, config: Any, provider: str) -> Tuple[Any, ...]:
    """Return JSON ``providers.<name>.hooks`` entries for a plugin/model.

    Provider instances expose ``_get_runtime_config`` (bound process-start
    snapshot); bare test doubles may not, in which case the explicit config
    object is consulted. Absence is never an error here — startup validation
    owns unknown-name failures.
    """

    if plugin is None:
        return ()
    getter = getattr(plugin, "_get_runtime_config", None)
    if callable(getter):
        try:
            runtime = getter(model)
            return tuple(getattr(runtime, "hooks", None) or ())
        except Exception:
            return ()
    if config is None:
        return ()
    try:
        from ..config.experimental import get_provider_runtime_config

        runtime = get_provider_runtime_config(provider, model, config=config)
        return tuple(runtime.hooks or ())
    except Exception:
        return ()


def resolve_hook_declarations(
    plugin: Any,
    model: str = "",
    *,
    config: Any = None,
    provider: str = "",
) -> Tuple[Tuple[Any, ...], Tuple[Any, ...], Tuple[str, ...]]:
    """Resolve ``(class_hooks, config_hooks, global_hook_names)`` for a plugin."""

    class_hooks = tuple(getattr(plugin, "hooks", ()) or ()) if plugin is not None else ()
    config_hooks = _provider_config_hooks(
        plugin,
        model,
        config=config,
        provider=provider or str(getattr(plugin, "provider_env_name", "") or ""),
    )
    return class_hooks, config_hooks, global_hook_names(config=config)


def make_pipeline_run(
    plugin: Any,
    *,
    request_id: str = "",
    provider: str = "",
    model: str = "",
    credential_id: str = "",
    session_id: str = "",
    scope_key: str = "",
    classifier: str = "",
    operation: str = "",
    config: Any = None,
    class_hooks: Optional[Tuple[Any, ...]] = None,
    config_hooks: Optional[Tuple[Any, ...]] = None,
    global_hooks: Optional[Tuple[str, ...]] = None,
) -> PipelineRun:
    """Mint the single per-request run from tri-source declarations."""

    if class_hooks is None or config_hooks is None or global_hooks is None:
        resolved_class, resolved_config, resolved_global = resolve_hook_declarations(
            plugin, model, config=config, provider=provider
        )
        class_hooks = resolved_class if class_hooks is None else class_hooks
        config_hooks = resolved_config if config_hooks is None else config_hooks
        global_hooks = resolved_global if global_hooks is None else global_hooks
    return PipelineRun(
        request_id=request_id,
        provider=provider,
        model=model,
        credential_id=credential_id,
        session_id=session_id,
        scope_key=scope_key,
        classifier=classifier,
        operation=operation,
        class_hooks=class_hooks,
        config_hooks=config_hooks,
        global_hooks=global_hooks,
    )
