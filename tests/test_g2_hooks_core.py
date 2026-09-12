"""G2 hooks core: contracts, ordering, isolation, containment, decisions."""

from __future__ import annotations

import asyncio
import pytest

from rotator_library.hooks.registry import register_hook, resolve_stage_bindings, validate_declared_names
from rotator_library.hooks.runner import PipelineRun, run_slot, BoundaryRecord
from rotator_library.hooks.types import (
    DEFAULT_HOOK_PRIORITY,
    HookAction,
    HookResult,
    PipelineCallback,
    PipelineHook,
    StageEvent,
    TransportView,
)


class AppendHook(PipelineHook):
    """Appends a marker to a list payload — the canonical test hook."""

    def __init__(self, marker: str, priority: int = DEFAULT_HOOK_PRIORITY, stages=("parsed_canonical",)):
        self.name = f"append_{marker}"
        self.marker = marker
        self.priority = priority
        self.stages = tuple(stages)

    async def __call__(self, invocation, context):
        payload = invocation.payload
        if isinstance(payload, list):
            payload = list(payload) + [self.marker]
        return payload


@pytest.mark.asyncio
async def test_hooks_chain_in_priority_then_registration_order():
    class A(AppendHook):
        def __init__(self):
            super().__init__("a", priority=100)

    class B(AppendHook):
        def __init__(self):
            super().__init__("b", priority=10)

    class C(AppendHook):
        def __init__(self):
            super().__init__("c", priority=100)

    # class order A, B, C -> priority order: B(10), then A(100) before C(100)
    run = PipelineRun(class_hooks=[A(), B(), C()])
    outcome = await run_slot(run, "parsed_canonical", ["start"])
    assert outcome.payload == ["start", "b", "a", "c"]
    assert outcome.modified is True


@pytest.mark.asyncio
async def test_default_priority_is_100_with_room_both_ways():
    assert DEFAULT_HOOK_PRIORITY == 100
    run = PipelineRun(class_hooks=[AppendHook("default"), AppendHook("early", priority=0),
                                   AppendHook("late", priority=200)])
    outcome = await run_slot(run, "parsed_canonical", [])
    assert outcome.payload == ["early", "default", "late"]


@pytest.mark.asyncio
async def test_per_request_isolation_stateful_and_state_bag():
    shared_instances = []

    class Stateful(AppendHook):
        stateful = True

        def __init__(self):
            super().__init__("s")
            self.seen = 0
            shared_instances.append(self)

    hook = Stateful()
    shared_instances.clear()  # count only per-run mints from here on
    run1 = PipelineRun(class_hooks=[hook])
    run2 = PipelineRun(class_hooks=[hook])

    async def drive(run, marker):
        # stash cross-stage scratch in the per-request state bag
        run.context.state["marker"] = marker
        await run_slot(run, "parsed_canonical", [])

    await asyncio.gather(drive(run1, "one"), drive(run2, "two"))
    # two distinct instances were minted (per-request isolation by construction)
    assert len(shared_instances) == 2
    assert run1.context.state["marker"] == "one"
    assert run2.context.state["marker"] == "two"
    assert run1.context.state is not run2.context.state


@pytest.mark.asyncio
async def test_crashed_hook_is_contained_keeps_payload():
    class Boom(PipelineHook):
        name = "boom"
        stages = ("parsed_canonical",)

        async def __call__(self, invocation, context):
            raise RuntimeError("hook exploded")

    run = PipelineRun(class_hooks=[Boom(), AppendHook("after")])
    outcome = await run_slot(run, "parsed_canonical", ["orig"])
    # boom contained -> payload unchanged by it, later hooks still run
    assert outcome.payload == ["orig", "after"]
    contained = [r for r in run.boundaries if r.kind == "contained"]
    assert contained and contained[0].name == "boom"


@pytest.mark.asyncio
async def test_critical_hook_failure_raises():
    class CriticalBoom(PipelineHook):
        name = "critical_boom"
        stages = ("parsed_canonical",)
        critical = True

        async def __call__(self, invocation, context):
            raise RuntimeError("must fail the request")

    run = PipelineRun(class_hooks=[CriticalBoom()])
    with pytest.raises(RuntimeError):
        await run_slot(run, "parsed_canonical", {})


@pytest.mark.asyncio
async def test_block_and_respond_verdicts_escape_the_chain():
    class Blocker(PipelineHook):
        name = "blocker"
        stages = ("request_received",)

        async def __call__(self, invocation, context):
            return HookResult(action=HookAction.BLOCK, message="not allowed", error_type="invalid_request")

    run = PipelineRun(class_hooks=[Blocker(), AppendHook("never", stages=("request_received",))])
    outcome = await run_slot(run, "request_received", {"x": 1}, direction="request")
    assert outcome.action is HookAction.BLOCK
    assert outcome.message == "not allowed"
    assert outcome.error_type == "invalid_request"

    class Responder(PipelineHook):
        name = "responder"
        stages = ("response_parsed",)

        async def __call__(self, invocation, context):
            return HookResult(action=HookAction.RESPOND, payload={"synthetic": True})

    run2 = PipelineRun(class_hooks=[Responder()])
    out2 = await run_slot(run2, "response_parsed", {"real": False}, direction="response")
    assert out2.action is HookAction.RESPOND and out2.payload == {"synthetic": True}


@pytest.mark.asyncio
async def test_stream_drop_and_replace():
    class Dropper(PipelineHook):
        name = "dropper"
        stages = ("stream_event",)

        async def __call__(self, invocation, context):
            if isinstance(invocation.payload, str) and "secret" in invocation.payload:
                return HookResult(action=HookAction.DROP)
            return None

    class Replacer(PipelineHook):
        name = "replacer"
        stages = ("stream_event",)
        priority = 200

        async def __call__(self, invocation, context):
            if invocation.payload == "ok":
                return HookResult(action=HookAction.REPLACE, payload="rewritten")
            return None

    run = PipelineRun(class_hooks=[Dropper(), Replacer()])
    out1 = await run_slot(run, "stream_event", "hello", direction="stream", event_index=0)
    assert out1.payload == "hello" and out1.action is HookAction.CONTINUE
    out2 = await run_slot(run, "stream_event", "secret-token", direction="stream", event_index=1)
    assert out2.action is HookAction.DROP and out2.payload is None
    out3 = await run_slot(run, "stream_event", "ok", direction="stream", event_index=2)
    assert out3.payload == "rewritten"


@pytest.mark.asyncio
async def test_callbacks_observe_settled_state_and_are_contained():
    observed = []

    class Watcher(PipelineCallback):
        name = "watcher"
        stages = ("parsed_canonical",)

        async def __call__(self, event: StageEvent, context):
            observed.append((event.stage, list(event.payload), tuple(event.hook_hops)))

    class BadListener(PipelineCallback):
        name = "bad_listener"
        stages = ("parsed_canonical",)
        priority = 5  # fires first, still must not break anything

        async def __call__(self, event: StageEvent, context):
            raise RuntimeError("listener exploded")

    run = PipelineRun(class_hooks=[AppendHook("mut1"), BadListener(), Watcher()])
    outcome = await run_slot(run, "parsed_canonical", ["start"])
    assert outcome.payload == ["start", "mut1"]
    assert observed == [("parsed_canonical", ["start", "mut1"], ("class:append_mut1",))]


@pytest.mark.asyncio
async def test_every_hop_is_recorded_nothing_invisible():
    run = PipelineRun(class_hooks=[AppendHook("a"), AppendHook("b")])
    await run_slot(run, "parsed_canonical", [])
    kinds = [(r.kind, r.name, r.changed) for r in run.boundaries]
    assert kinds[0][0] == "before"
    hop_kinds = [k for k in kinds if k[0] == "hop"]
    assert len(hop_kinds) == 2
    assert kinds[-1][0] == "after"


@pytest.mark.asyncio
async def test_transport_rewrite_recorded_as_overlay():
    class UrlFlipper(PipelineHook):
        name = "url_flipper"
        stages = ("transport_ready",)

        async def __call__(self, invocation, context):
            view: TransportView = invocation.transport
            view.endpoint = "https://shadow.example/v1/chat"
            view.headers = {**(view.headers or {}), "x-shadow": "1"}
            view.changed = True
            return None

    view = TransportView(endpoint="https://real.example/v1/chat", headers={"Authorization": "Bearer x"})
    run = PipelineRun(class_hooks=[UrlFlipper()])
    await run_slot(run, "transport_ready", {"model": "m"}, direction="request", transport=view)
    assert view.endpoint == "https://shadow.example/v1/chat"
    assert any(o["kind"] == "transport_rewrite" and o["endpoint"] == "https://shadow.example/v1/chat"
               for o in run.overlays)


@pytest.mark.asyncio
async def test_registration_tri_source_precedence_and_startup_validation():
    class GlobalHook(AppendHook):
        stages = ("parsed_canonical",)

        def __init__(self):
            super().__init__("global")

    register_hook(GlobalHook, replace=True)
    validate_declared_names(global_hooks=["GlobalHook"])
    hooks, _ = resolve_stage_bindings("parsed_canonical", global_hooks=["GlobalHook"])
    assert hooks and hooks[0].source == "global"

    with pytest.raises(KeyError):
        validate_declared_names(global_hooks=["does_not_exist"])

    with pytest.raises(ValueError):
        validate_declared_names(config_hooks=[{"name": "GlobalHook", "stages": ("not_a_stage",)}])


@pytest.mark.asyncio
async def test_stateful_hook_factory_instantiation_per_run():
    minted = []

    def factory():
        instance = AppendHook("factory")
        minted.append(instance)
        return instance

    factory.name = "factory_hook"
    factory.stages = ("parsed_canonical",)
    factory.stateful = True
    register_hook(factory, replace=True, name="factory_hook")

    run1 = PipelineRun(global_hooks=["factory_hook"])
    run2 = PipelineRun(global_hooks=["factory_hook"])
    await asyncio.gather(
        run_slot(run1, "parsed_canonical", []),
        run_slot(run2, "parsed_canonical", []),
    )
    assert len(minted) == 2
