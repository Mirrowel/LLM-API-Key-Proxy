# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Base classes for configurable protocol/provider payload adapters."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional


@dataclass
class AdapterContext:
    """Context available to every adapter pass.

    Adapters are intentionally base implementations, not provider law. Provider
    code can subclass adapters, copy and mutate them, or bypass them when a
    protocol needs special behavior. The context mirrors transform trace fields
    so each adapter pass can be debugged by provider/model/session/scope.
    """

    provider: Optional[str] = None
    model: Optional[str] = None
    protocol: Optional[str] = None
    credential_id: Optional[str] = None
    session_id: Optional[str] = None
    scope_key: Optional[str] = None
    classifier: Optional[str] = None
    transport: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    adapter_config: dict[str, dict[str, Any]] = field(default_factory=dict)
    transaction_logger: Optional[Any] = None

    def config_for(self, adapter_name: str) -> dict[str, Any]:
        """Return adapter-specific config without mutating the original context."""

        return dict(self.adapter_config.get(adapter_name, {}))


class PayloadAdapter:
    """Override-friendly base adapter.

    The default adapter is a no-op. Concrete adapters override only the stages
    they support. Methods are async so future provider adapters can consult
    caches, metadata services, or remote capability probes without changing the
    chain runner API.
    """

    name: str = ""
    aliases: tuple[str, ...] = ()
    supported_stages: tuple[str, ...] = ("request", "response", "stream_event")

    async def transform_request(self, payload: Any, context: AdapterContext) -> Any:
        return payload

    async def transform_response(self, payload: Any, context: AdapterContext) -> Any:
        return payload

    async def transform_stream_event(self, payload: Any, context: AdapterContext) -> Any:
        return payload

    async def transform(self, stage: str, payload: Any, context: AdapterContext) -> Any:
        """Dispatch a stage-specific transform with a useful error for typos."""

        if stage not in self.supported_stages:
            return payload
        if stage == "request":
            return await self.transform_request(payload, context)
        if stage == "response":
            return await self.transform_response(payload, context)
        if stage == "stream_event":
            return await self.transform_stream_event(payload, context)
        raise ValueError(f"Unknown adapter stage: {stage}")


def _trace(context: AdapterContext, pass_name: str, payload: Any, *, stage: str, metadata: Mapping[str, Any]) -> None:
    logger = context.transaction_logger
    if not logger:
        return
    logger.log_transform_pass(
        pass_name,
        payload,
        direction="stream" if stage == "stream_event" else stage,
        stage="adapter",
        protocol=context.protocol,
        credential_id=context.credential_id,
        transport=context.transport,
        metadata=dict(metadata),
        snapshot=stage != "stream_event",
    )


async def run_adapter_chain(
    adapters: Iterable[PayloadAdapter],
    payload: Any,
    context: AdapterContext,
    *,
    stage: str,
    mutate: bool = False,
) -> Any:
    """Run adapters in order and emit trace entries around the chain.

    By default the payload is deep-copied before the first adapter. This keeps
    Phase 3 isolated and prevents surprise mutations until runtime integration
    explicitly chooses mutating behavior for performance.
    """

    current = payload if mutate else deepcopy(payload)
    tracing_enabled = context.transaction_logger is not None
    original = deepcopy(current) if tracing_enabled else None
    adapter_names = [adapter.name for adapter in adapters]
    _trace(context, "before_adapter_chain", current, stage=stage, metadata={"adapters": adapter_names})
    for adapter in adapters:
        before = deepcopy(current) if tracing_enabled else None
        try:
            current = await adapter.transform(stage, current, context)
        except Exception as exc:
            if context.transaction_logger:
                context.transaction_logger.log_transform_error(
                    f"adapter:{adapter.name}:{stage}",
                    exc,
                    payload=before if before is not None else current,
                    stage="adapter",
                    protocol=context.protocol,
                    transport=context.transport,
                    metadata={"adapter": adapter.name, "adapter_stage": stage},
                )
            raise
        _trace(
            context,
            "after_adapter",
            current,
            stage=stage,
            metadata={"adapter": adapter.name, "adapter_stage": stage, "changed": (current != before) if before is not None else None},
        )
    _trace(
        context,
        "after_adapter_chain",
        current,
        stage=stage,
        metadata={"adapters": adapter_names, "adapter_stage": stage, "adapter_count": len(adapter_names), "changed": (original != current) if original is not None else None},
    )
    return current
