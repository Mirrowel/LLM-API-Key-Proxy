# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Proxy-owned tools reference demo (G2 maturity proof).

This module is a REFERENCE DEMO, not a production engine: no consumer in
the proxy is expected to mount these hooks. Its purpose is to prove the
declared hook system can express a proxy-owned tool plane at every seam:

- :class:`ToolStripperHook` removes designated tool definitions from the
  outbound request at BOTH ``parsed_canonical`` (``UnifiedRequest.tools``)
  and ``provider_built`` (wire payload, tolerant to dialect shapes) — the
  model never sees the tool.
- :class:`ToolInjectorHook` injects proxy-owned tool definitions at both
  levels (idempotent — re-injection never duplicates) and marks them
  proxy-owned in the session registry.
- :class:`ToolCallInterceptorHook` runs at ``response_parsed``: it detects
  calls to proxy-owned tools in the parsed ``UnifiedResponse``, records
  them, executes them locally through a caller-supplied async executor
  (default stub returns ``"proxy-tool-result:{name}"``), appends the
  assistant turn plus tool result to the conversation history, and sets
  the ``proxy_tools:reenter`` flag in the run state.

RE-ENTRY IS OUT OF SCOPE. When interception fires with a caller-supplied
executor, this demo stops at the recorded intent: the appended history and
the ``proxy_tools:reenter`` flag are the seams a future loop engine would
consume to re-enter the pipeline with the tool results. That engine is a
future group's work. Without a caller-supplied executor the demo answers
the client directly: a synthetic normal assistant response is returned via
``HookAction.RESPOND`` at ``response_parsed`` (openai_chat client shape
only — a documented dialect limitation: non-chat client protocols would
need their own formatter for the synthetic payload).

Session registry: cross-REQUEST persistence within a session uses a
process-level dict guarded by a lock, keyed by ``(scope_key, session_id)``,
with a simple FIFO eviction cap (``_MAX_SESSIONS``). No TTL, no disk
persistence — acceptable for a demo, documented as such. Per-REQUEST
observations (stripped/injected/intercepted/history/reenter) live in the
run's state bag and die with the request.

Danger note: like every hook here, these classes have FULL read/write
power over the live stage payloads. A bug in a stripper corrupts the
outbound request; a bug in the interceptor corrupts the client response.
Ordering between stripper and injector at a shared stage follows hook
declaration order (declare the stripper first).
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from ..types import DEFAULT_HOOK_PRIORITY, HookContext, HookAction, HookResult, PipelineHook, StageInvocation

__all__ = [
    "ToolStripperHook",
    "ToolInjectorHook",
    "ToolCallInterceptorHook",
    "proxy_tools_registry_snapshot",
    "reset_proxy_tools_registry",
]

#: State-bag keys (per request; die with the run).
STATE_STRIPPED = "proxy_tools:stripped"
STATE_INJECTED = "proxy_tools:injected"
STATE_INTERCEPTED = "proxy_tools:intercepted"
STATE_HISTORY = "proxy_tools:history"
STATE_REENTER = "proxy_tools:reenter"

#: Async executor contract: ``fn(tool_name, arguments) -> str``.
ToolExecutor = Callable[[str, Any], Awaitable[str]]

_REGISTRY_LOCK = threading.Lock()
_SESSIONS: Dict[Tuple[str, str], Dict[str, Any]] = {}
#: Demo-scoped FIFO eviction cap — no TTL, no persistence.
_MAX_SESSIONS = 1000


# -- session registry (process-level, demo-scoped) -------------------------


def _session_key(context: HookContext) -> Tuple[str, str]:
    return (context.scope_key or "", context.session_id or "")


def _session_entry(key: Tuple[str, str]) -> Dict[str, Any]:
    """Get or create the registry entry. Caller must hold ``_REGISTRY_LOCK``."""

    entry = _SESSIONS.get(key)
    if entry is None:
        if len(_SESSIONS) >= _MAX_SESSIONS:
            _SESSIONS.pop(next(iter(_SESSIONS)), None)
        entry = {"proxy_owned": {}, "stripped": [], "interceptions": [], "history": []}
        _SESSIONS[key] = entry
    return entry


def _record_stripped(context: HookContext, names: Sequence[str], stage: str) -> None:
    with _REGISTRY_LOCK:
        _session_entry(_session_key(context))["stripped"].append({"stage": stage, "names": list(names)})


def _mark_proxy_owned(context: HookContext, spec: Mapping[str, Any]) -> None:
    with _REGISTRY_LOCK:
        _session_entry(_session_key(context))["proxy_owned"][str(spec.get("name", ""))] = {
            "description": spec.get("description"),
            "parameters": dict(spec.get("parameters") or spec.get("input_schema") or {}),
        }


def _record_interceptions(context: HookContext, records: Sequence[Dict[str, Any]]) -> None:
    with _REGISTRY_LOCK:
        _session_entry(_session_key(context))["interceptions"].extend(dict(record) for record in records)


def _append_history(context: HookContext, turns: Sequence[Dict[str, Any]]) -> None:
    with _REGISTRY_LOCK:
        _session_entry(_session_key(context))["history"].extend(dict(turn) for turn in turns)


def _proxy_owned_names(context: HookContext) -> Set[str]:
    with _REGISTRY_LOCK:
        entry = _SESSIONS.get(_session_key(context))
        return set(entry["proxy_owned"]) if entry else set()


def proxy_tools_registry_snapshot(scope_key: str = "", session_id: str = "") -> Dict[str, Any]:
    """Deep-copied registry entry for one session (test/inspection helper)."""

    with _REGISTRY_LOCK:
        entry = _SESSIONS.get((scope_key or "", session_id or ""))
        return json.loads(json.dumps(entry)) if entry is not None else {
            "proxy_owned": {}, "stripped": [], "interceptions": [], "history": [],
        }


def reset_proxy_tools_registry() -> None:
    """Clear the process-level demo registry (test isolation / teardown)."""

    with _REGISTRY_LOCK:
        _SESSIONS.clear()


# -- wire-shape helpers (tolerant to dialect shapes) ------------------------


def _wire_tool_names(entry: Any) -> List[str]:
    """Names carried by one wire tool entry (openai nested, flat declarations)."""

    if not isinstance(entry, dict):
        return []
    names: List[str] = []
    if isinstance(entry.get("name"), str):
        names.append(entry["name"])
    function = entry.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        names.append(function["name"])
    return names


def _existing_wire_tool_names(entries: Sequence[Any]) -> Set[str]:
    """All tool names present in a wire ``tools`` list, across dialect shapes."""

    names: Set[str] = set()
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("functionDeclarations"), list):
            for declaration in entry["functionDeclarations"]:
                if isinstance(declaration, dict) and isinstance(declaration.get("name"), str):
                    names.add(declaration["name"])
        else:
            names.update(_wire_tool_names(entry))
    return names


def _strip_wire_tools(payload: Mapping[str, Any], names: Set[str]) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Remove matching tools from a wire payload, tolerant to dialect shapes.

    Returns ``(new_payload, removed_names)``; ``new_payload`` is ``None``
    when nothing matched (payload identity preserved — no-op strips never
    flip the D4 raw fast path or add trace noise).
    """

    tools = payload.get("tools")
    if not isinstance(tools, list):
        return None, []
    kept_tools: List[Any] = []
    removed: List[str] = []
    for entry in tools:
        if isinstance(entry, dict) and isinstance(entry.get("functionDeclarations"), list):
            declarations = entry["functionDeclarations"]
            kept = [
                declaration
                for declaration in declarations
                if not (isinstance(declaration, dict) and declaration.get("name") in names)
            ]
            if len(kept) == len(declarations):
                kept_tools.append(entry)
                continue
            removed.extend(
                declaration["name"]
                for declaration in declarations
                if isinstance(declaration, dict) and declaration.get("name") in names
            )
            if kept:
                kept_tools.append({**entry, "functionDeclarations": kept})
            continue
        hit = [name for name in _wire_tool_names(entry) if name in names]
        if hit:
            removed.extend(hit)
            continue
        kept_tools.append(entry)
    if not removed:
        return None, []
    updated = dict(payload)
    updated["tools"] = kept_tools
    return updated, removed


def _wire_dialect(entries: Sequence[Any]) -> str:
    """Sniff the dialect of an existing wire ``tools`` list (demo heuristic)."""

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("functionDeclarations"), list):
            return "gemini"
        if isinstance(entry.get("function"), dict):
            return "openai"
        if isinstance(entry.get("input_schema"), dict) and "name" in entry:
            return "anthropic"
        if entry.get("type") == "function" and "name" in entry and "parameters" in entry:
            return "responses"
    return "openai"


def _wire_tool_entry(dialect: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Build one wire tool entry in the given dialect from a canonical spec."""

    name = str(spec.get("name", ""))
    description = spec.get("description")
    parameters = dict(spec.get("parameters") or spec.get("input_schema") or {})
    entry: Dict[str, Any]
    if dialect == "anthropic":
        entry = {"name": name, "input_schema": parameters}
        if description is not None:
            entry["description"] = description
        return entry
    if dialect == "gemini":
        declaration: Dict[str, Any] = {"name": name, "parameters": parameters}
        if description is not None:
            declaration["description"] = description
        return {"functionDeclarations": [declaration]}
    if dialect == "responses":
        entry = {"type": "function", "name": name, "parameters": parameters}
        if description is not None:
            entry["description"] = description
        return entry
    function: Dict[str, Any] = {"name": name, "parameters": parameters}
    if description is not None:
        function["description"] = description
    return {"type": "function", "function": function}


def _serialize_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments)
    except (TypeError, ValueError):
        return str(arguments)


async def _stub_tool_executor(tool_name: str, arguments: Any) -> str:
    """Default demo executor: deterministic, side-effect free."""

    return f"proxy-tool-result:{tool_name}"


def _synthetic_chat_response(content: str) -> Dict[str, Any]:
    """Minimal client response for openai_chat (documented dialect limit)."""

    return {
        "id": "chatcmpl-proxy-tools-demo",
        "object": "chat.completion",
        "created": 0,
        "model": "proxy-tools-demo",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# -- hooks -------------------------------------------------------------------


class ToolStripperHook(PipelineHook):
    """Strip designated tool definitions from the outbound request.

    Binds at ``parsed_canonical`` (filter ``UnifiedRequest.tools`` by name)
    and ``provider_built`` (filter the wire payload ``tools``/``functions``
    by name, tolerant to openai/anthropic/gemini/responses shapes). What was
    stripped is recorded into the per-request state bag
    (``proxy_tools:stripped``) and the per-session registry. The model never
    sees the tool.

    A strip at ``parsed_canonical`` flips the D4 raw fast path to a
    canonical rebuild (traced overlay) — that is the intended, visible
    behavior; a ``provider_built``-only strip keeps the raw basis.

    Danger note: full read/write power over the request payload at both
    stages. Pass ``stages=("provider_built",)`` to leave the canonical
    request untouched.
    """

    def __init__(
        self,
        tool_names: Sequence[str],
        *,
        stages: Optional[Sequence[str]] = None,
        name: str = "proxy_tool_stripper",
        priority: int = DEFAULT_HOOK_PRIORITY,
    ) -> None:
        self.name = name
        self.tool_names = tuple(tool_names)
        self.stages = tuple(stages) if stages is not None else ("parsed_canonical", "provider_built")
        self.priority = priority

    async def __call__(self, invocation: StageInvocation, context: HookContext):
        names = set(self.tool_names)
        if invocation.stage == "parsed_canonical":
            request = invocation.payload
            tools = list(getattr(request, "tools", None) or [])
            kept = [tool for tool in tools if getattr(tool, "name", None) not in names]
            if len(kept) == len(tools):
                return None
            removed = [getattr(tool, "name") for tool in tools if getattr(tool, "name", None) in names]
            self._record(context, removed, "parsed_canonical")
            return replace(request, tools=kept)
        if invocation.stage == "provider_built":
            payload = invocation.payload
            if not isinstance(payload, dict):
                return None
            updated, removed = _strip_wire_tools(payload, names)
            if updated is None:
                return None
            self._record(context, removed, "provider_built")
            return updated
        return None

    def _record(self, context: HookContext, names: Sequence[str], stage: str) -> None:
        if not names:
            return
        context.state.setdefault(STATE_STRIPPED, []).extend(names)
        _record_stripped(context, names, stage)


class ToolInjectorHook(PipelineHook):
    """Inject proxy-owned tool definitions into the outbound request.

    Accepts canonical tool specs (``{"name", "description", "parameters"}``;
    ``input_schema`` accepted as an alias). Binds at ``parsed_canonical``
    (append ``ToolDefinition`` objects) and ``provider_built`` (append
    wire entries matching the dialect of the existing tools list;
    ``openai_chat`` shape when the list is empty — a documented demo
    default). Injection is idempotent at both levels: names already
    present are never duplicated, and a canonical injection renders the
    ``provider_built`` pass a no-op. Injected tools are marked proxy-owned
    in the session registry and recorded in ``proxy_tools:injected``.

    Danger note: full read/write power over the request payload at both
    stages; the injected schema must be valid for the target dialect.
    """

    def __init__(
        self,
        tools: Sequence[Mapping[str, Any]],
        *,
        stages: Optional[Sequence[str]] = None,
        name: str = "proxy_tool_injector",
        priority: int = DEFAULT_HOOK_PRIORITY,
    ) -> None:
        self.name = name
        self._tools: List[Dict[str, Any]] = [dict(tool) for tool in tools]
        self.stages = tuple(stages) if stages is not None else ("parsed_canonical", "provider_built")
        self.priority = priority

    async def __call__(self, invocation: StageInvocation, context: HookContext):
        if invocation.stage == "parsed_canonical":
            return self._inject_canonical(invocation, context)
        if invocation.stage == "provider_built":
            return self._inject_wire(invocation, context)
        return None

    def _inject_canonical(self, invocation: StageInvocation, context: HookContext):
        from ...protocols.types import ToolDefinition

        request = invocation.payload
        tools = list(getattr(request, "tools", None) or [])
        present = {getattr(tool, "name", None) for tool in tools}
        added = [spec for spec in self._tools if spec.get("name") not in present]
        if not added:
            return None
        injected = tools + [
            ToolDefinition(
                name=str(spec.get("name", "")),
                description=spec.get("description"),
                input_schema=dict(spec.get("parameters") or spec.get("input_schema") or {}),
                type=str(spec.get("type", "function")),
                extra=dict(spec.get("extra") or {}),
            )
            for spec in added
        ]
        self._record(context, added)
        return replace(request, tools=injected)

    def _inject_wire(self, invocation: StageInvocation, context: HookContext):
        payload = invocation.payload
        if not isinstance(payload, dict) or not isinstance(payload.get("tools"), list):
            return None
        tools = payload["tools"]
        present = _existing_wire_tool_names(tools)
        added = [spec for spec in self._tools if spec.get("name") not in present]
        if not added:
            return None
        dialect = _wire_dialect(tools)
        updated = dict(payload)
        updated["tools"] = list(tools) + [_wire_tool_entry(dialect, spec) for spec in added]
        self._record(context, added)
        return updated

    def _record(self, context: HookContext, added: Sequence[Mapping[str, Any]]) -> None:
        names = [str(spec.get("name", "")) for spec in added]
        context.state.setdefault(STATE_INJECTED, []).extend(names)
        for spec in added:
            _mark_proxy_owned(context, spec)


class ToolCallInterceptorHook(PipelineHook):
    """Intercept model calls to proxy-owned tools at ``response_parsed``.

    Detects ``tool_calls`` in the parsed ``UnifiedResponse`` whose names are
    proxy-owned (session registry marks from :class:`ToolInjectorHook`,
    plus any explicitly declared ``tool_names``). On interception the demo:

    1. records the intercepted call (state bag + session registry),
    2. executes the tool locally through the caller-supplied async executor
       ``fn(tool_name, arguments) -> str`` (default stub returns
       ``"proxy-tool-result:{name}"``),
    3. appends the assistant turn + tool result to the conversation
       history (``proxy_tools:history`` state bag + session registry),
    4. sets ``proxy_tools:reenter`` in the run state.

    Full re-entry into the pipeline is OUT OF SCOPE (see module docstring):
    with a caller-supplied executor the real response continues downstream
    and the re-enter intent lives in the recorded flag/history. WITHOUT an
    executor the demo answers the client itself via ``HookAction.RESPOND``
    with a synthetic openai_chat-shaped assistant message (dialect
    limitation documented in the module docstring).

    Danger note: full read/write power over the parsed response; a RESPOND
    verdict replaces the provider's answer entirely.
    """

    def __init__(
        self,
        *,
        tool_names: Sequence[str] = (),
        executor: Optional[ToolExecutor] = None,
        name: str = "proxy_tool_interceptor",
        priority: int = DEFAULT_HOOK_PRIORITY,
    ) -> None:
        self.name = name
        self.stages = ("response_parsed",)
        self.priority = priority
        self.tool_names = tuple(tool_names)
        self._executor = executor

    async def __call__(self, invocation: StageInvocation, context: HookContext):
        if invocation.stage != "response_parsed":
            return None
        response = invocation.payload
        owned = set(self.tool_names) | _proxy_owned_names(context)
        calls = [
            call
            for message in getattr(response, "messages", None) or []
            for call in getattr(message, "tool_calls", None) or []
            if getattr(call, "name", None) in owned
        ]
        if not calls:
            return None
        executor = self._executor or _stub_tool_executor
        records: List[Dict[str, Any]] = []
        for position, call in enumerate(calls):
            result = await executor(call.name, call.arguments)
            records.append({
                "name": call.name,
                "id": call.id or f"proxy_call_{position}",
                "arguments": call.arguments,
                "result": result,
            })
        # (i) record the intercepted calls
        context.state.setdefault(STATE_INTERCEPTED, []).extend(dict(record) for record in records)
        _record_interceptions(context, records)
        # (iii) append the assistant turn + tool results to the history
        assistant_turn = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": record["id"],
                    "type": "function",
                    "function": {"name": record["name"], "arguments": _serialize_arguments(record["arguments"])},
                }
                for record in records
            ],
        }
        tool_turns = [
            {"role": "tool", "tool_call_id": record["id"], "name": record["name"], "content": record["result"]}
            for record in records
        ]
        turns = [assistant_turn, *tool_turns]
        context.state.setdefault(STATE_HISTORY, []).extend(dict(turn) for turn in turns)
        _append_history(context, turns)
        # (iv) record the re-entry intent (loop engine is future work)
        context.state[STATE_REENTER] = True
        # Client-facing answer only when no executor was supplied
        if self._executor is None:
            content = "\n".join(f"proxy tool {record['name']} executed: {record['result']}" for record in records)
            return HookResult(action=HookAction.RESPOND, payload=_synthetic_chat_response(content))
        return None
