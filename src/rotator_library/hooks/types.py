"""Hook and callback contracts for the declared execution pipeline.

Two contracts share one registration surface:

- Hooks are interceptors: the pipeline pauses at a declared stage, hands the
  live payload over, waits, and continues with whatever the hook returns.
  Hooks can rewrite the payload, block the request with a protocol-shaped
  error, answer the request themselves, or drop/replace stream events.
- Callbacks are listeners: they fire after a slot settles, receive a snapshot
  of the settled state, cannot change the flow, and their errors are always
  contained.

Isolation: every request gets its own ``PipelineRun`` (see runner.py) that
owns the state bag, stateful hook instances, overlays, and boundary events.
Nothing mutable is shared between concurrent requests. The registries hold
declarations (classes/factories), never live per-request state.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Sequence, Union

__all__ = [
    "HOOK_STAGES",
    "STAGE_PAYLOAD_TYPES",
    "HookAction",
    "HookResult",
    "StageEvent",
    "StageInvocation",
    "HookContext",
    "PipelineHook",
    "PipelineCallback",
    "DEFAULT_HOOK_PRIORITY",
    "CRITICAL_HOOK_PRIORITY",
    "EARLY_HOOK_PRIORITY",
    "LATE_HOOK_PRIORITY",
]


#: Default priority: large enough that anything can comfortably register
#: before (smaller) or after (larger) default bindings (user directive).
DEFAULT_HOOK_PRIORITY = 100
#: Convenience bands around the default; hooks may use any integer.
EARLY_HOOK_PRIORITY = 0
CRITICAL_HOOK_PRIORITY = 50
LATE_HOOK_PRIORITY = 200


class _StageMeta(enum.Enum):
    """Payload kind carried at each declared stage boundary."""

    CLIENT_WIRE = "client_wire"          # raw client request/response dict
    CANONICAL = "canonical"              # UnifiedRequest / UnifiedResponse
    PROVIDER_WIRE = "provider_wire"      # provider-dialect dict
    TRANSPORT = "transport"              # endpoint/headers/timeout descriptor
    STREAM_EVENT = "stream_event"        # UnifiedStreamEvent
    USAGE = "usage"                      # usage/cost record
    ROUTING = "routing"                  # routing/session/credential info


# Declared stage boundaries. Names are the stable public contract; positions
# are documented in the runner. A/B pairs wrap internal field-cache passes so
# hooks can ride before and after cache work.
HOOK_STAGES: Dict[str, _StageMeta] = {
    # Request direction
    "request_received": _StageMeta.CLIENT_WIRE,        # R1 - client entry
    "routing_resolved": _StageMeta.ROUTING,             # R2
    "credential_selected": _StageMeta.ROUTING,          # R3
    "session_resolved": _StageMeta.ROUTING,             # R4
    "parsed_canonical": _StageMeta.CANONICAL,           # R5 - post-parse
    "canonical_state_inject_a": _StageMeta.CANONICAL,   # R6A - before cache inject
    "canonical_state_inject_b": _StageMeta.CANONICAL,   # R6B - after cache inject
    "transport_basis_selected": _StageMeta.CANONICAL,   # R7 - raw vs rebuild (read/flip)
    "provider_built": _StageMeta.PROVIDER_WIRE,         # R8 - post build_request
    "finalizer": _StageMeta.PROVIDER_WIRE,              # R9 - LAST pre-send edits
    "mutated": _StageMeta.PROVIDER_WIRE,                # R10 - adapter band
    "state_inject_a": _StageMeta.PROVIDER_WIRE,         # R11A
    "state_inject_b": _StageMeta.PROVIDER_WIRE,         # R11B
    "validated": _StageMeta.PROVIDER_WIRE,              # R12 - veto-capable
    "transport_ready": _StageMeta.TRANSPORT,            # R13 - endpoint/headers/timeout
    "sent": _StageMeta.PROVIDER_WIRE,                   # R14 - raw wire + status (observe)
    # Response direction
    "response_received": _StageMeta.PROVIDER_WIRE,      # P1
    "response_state_extract_a": _StageMeta.PROVIDER_WIRE,  # P2A
    "response_state_extract_b": _StageMeta.PROVIDER_WIRE,  # P2B
    "response_parsed": _StageMeta.CANONICAL,            # P3
    "response_formatted": _StageMeta.CLIENT_WIRE,       # P4
    "usage_recorded": _StageMeta.USAGE,                 # P5
    # Stream direction
    "stream_opened": _StageMeta.TRANSPORT,              # S1
    "stream_event": _StageMeta.STREAM_EVENT,            # S2 - every event incl. terminal
    "stream_assembled": _StageMeta.CANONICAL,           # S3 - assembled final response
    "stream_closed": _StageMeta.STREAM_EVENT,           # S4 - terminal outcome (observe)
}

STAGE_PAYLOAD_TYPES = {name: meta.value for name, meta in HOOK_STAGES.items()}


class HookAction(str, enum.Enum):
    """What the pipeline should do with a hook's verdict."""

    CONTINUE = "continue"      # keep (possibly rewritten) payload, resume
    BLOCK = "block"            # fail the request with a protocol-shaped error
    RESPOND = "respond"        # short-circuit with a synthetic payload
    DROP = "drop"              # stream only: omit this event downstream
    REPLACE = "replace"        # stream only: use the returned event instead


@dataclass
class HookResult:
    """A hook's explicit verdict.

    Returning ``None`` from a hook means CONTINUE with the payload unchanged;
    returning a bare payload means CONTINUE with that payload. ``HookResult``
    is only needed for the explicit actions.
    """

    action: HookAction = HookAction.CONTINUE
    payload: Any = None
    #: For BLOCK: error message (rendered in the client's dialect).
    message: Optional[str] = None
    #: For BLOCK: optional internal error class (defaults to invalid_request).
    error_type: Optional[str] = None


@dataclass
class StageInvocation:
    """What a hook receives: the live stage payload plus position state.

    Each stage exposes only what exists at that position. Everything on the
    invocation is readable; the ``payload`` (and the position-specific
    writable views) are the legal mutation surfaces. Anything not offered
    here is untouched and unread by the stage — each step takes only what it
    needs (user directive).
    """

    stage: str
    payload: Any
    #: Direction of travel: "request" | "response" | "stream".
    direction: str = "request"
    #: Position-specific writable views (e.g. transport endpoint/headers).
    transport: Optional["TransportView"] = None
    #: True when the payload is the terminal event of a stream.
    is_terminal: bool = False
    #: Index of the event in the stream (stream stages only).
    event_index: Optional[int] = None

    def result(self, payload: Any = None, **kwargs: Any) -> HookResult:
        """Convenience: CONTINUE with a (possibly new) payload."""
        if kwargs:
            return HookResult(action=HookAction.CONTINUE, payload=payload, **kwargs)
        return HookResult(action=HookAction.CONTINUE, payload=payload)


@dataclass
class TransportView:
    """Writable transport descriptor offered at transport-carrying stages.

    Rewrites are recorded as overlays on the run and surface in traces — no
    change is invisible (user directive).
    """

    endpoint: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    timeout_seconds: Optional[float] = None
    #: Set by the runner when rewrites happened, for trace overlays.
    changed: bool = False


@dataclass
class StageEvent:
    """What a callback receives: a settled snapshot of the boundary."""

    stage: str
    payload: Any
    direction: str
    is_terminal: bool = False
    event_index: Optional[int] = None
    #: How many hooks ran at this boundary and what they changed.
    hook_hops: Sequence[str] = field(default_factory=tuple)


class HookContext:
    """Per-request state bag shared by every hook invocation in one run.

    ``state`` is a free-form dict unique to the request; hooks may stash
    cross-stage scratch data there. It is never shared between requests.
    """

    __slots__ = ("request_id", "provider", "model", "credential_id", "session_id",
                 "scope_key", "classifier", "operation", "state")

    def __init__(
        self,
        request_id: str = "",
        provider: str = "",
        model: str = "",
        credential_id: str = "",
        session_id: str = "",
        scope_key: str = "",
        classifier: str = "",
        operation: str = "",
    ) -> None:
        self.request_id = request_id
        self.provider = provider
        self.model = model
        self.credential_id = credential_id
        self.session_id = session_id
        self.scope_key = scope_key
        self.classifier = classifier
        self.operation = operation
        self.state: Dict[str, Any] = {}


#: A hook callable: takes (invocation, context), returns None/payload/HookResult.
HookCallable = Callable[[StageInvocation, HookContext], Union[None, Any, HookResult, Awaitable[None | Any | HookResult]]]
#: A callback callable: takes (event, context), return value ignored.
CallbackCallable = Callable[[StageEvent, HookContext], Optional[Awaitable[None]]]


class PipelineHook:
    """Base class / declaration for hooks.

    Subclasses (or registered callables) declare where they bind. Keep hook
    code minimal: raw access to the data at the position, a pause, the work,
    a return — no interface ceremony beyond the binding declaration.

    Danger note: hooks at any stage have FULL read/write power over the
    stage payload. A misbehaving hook can corrupt requests, responses, and
    streams; ordering and overlays are traced, but correctness of the edit
    itself is the hook author's responsibility. Use callbacks when mutation
    is not needed.
    """

    name: str = ""
    aliases: Sequence[str] = ()
    #: Stages this hook binds to (any subset of HOOK_STAGES).
    stages: Sequence[str] = ()
    #: Smaller fires first; default 100 (user directive).
    priority: int = DEFAULT_HOOK_PRIORITY
    #: When True, an exception in this hook fails the request instead of
    #: being contained (keep-payload + warn).
    critical: bool = False
    #: When True, a fresh instance is created per request run; state then
    #: lives on the instance (isolated per request by construction).
    stateful: bool = False

    async def __call__(self, invocation: StageInvocation, context: HookContext):
        """Handle a stage invocation. Return None, a payload, or HookResult."""
        return None


class PipelineCallback:
    """Base class for callbacks (listeners).

    Callbacks fire after a slot settles, receive a snapshot, cannot change
    the flow, and errors are always contained (logged, never fatal).
    """

    name: str = ""
    aliases: Sequence[str] = ()
    stages: Sequence[str] = ()
    priority: int = DEFAULT_HOOK_PRIORITY

    async def __call__(self, event: StageEvent, context: HookContext) -> None:
        return None
