# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Ordered fallback attempt runner."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .policy import FallbackPolicy, normalize_route_error_type
from .types import FallbackGroup, RouteAttemptResult, RouteTarget, RoutingDecision

AttemptCallback = Callable[[RouteTarget, int], Awaitable[Any]]


class FallbackExhaustedError(Exception):
    """Raised when every eligible fallback target fails."""

    def __init__(self, decision: RoutingDecision, attempts: tuple[RouteAttemptResult, ...]) -> None:
        self.decision = decision
        self.attempts = attempts
        summary = ", ".join(f"{attempt.target.name}:{attempt.error_type}" for attempt in attempts)
        super().__init__(f"fallback group exhausted for {decision.requested_model}: {summary}")


class FallbackAttemptRunner:
    """Run ordered route targets using an injected per-target attempt callback.

    The runner is independent from `RequestExecutor`. Phase 6 can unit-test
    fallback control flow here, then wire the callback to existing per-target
    credential retry logic without duplicating policy decisions.
    """

    def __init__(self, policy: FallbackPolicy | None = None) -> None:
        self.policy = policy or FallbackPolicy()

    async def run(self, decision: RoutingDecision, attempt: AttemptCallback, *, stream: bool = False) -> Any:
        """Try targets in order until one succeeds or fallback is exhausted."""

        return await self.run_group(decision, decision.group, attempt, stream=stream)

    async def run_group(
        self,
        decision: RoutingDecision,
        group: FallbackGroup | None,
        attempt: AttemptCallback,
        *,
        stream: bool = False,
    ) -> Any:
        """Try targets while honoring optional group-specific policy overrides."""

        attempts: list[RouteAttemptResult] = []
        for index, target in enumerate(decision.targets):
            try:
                return await attempt(target, index)
            except Exception as exc:
                error_type = _error_type(exc)
                emitted_output = bool(getattr(exc, "emitted_output", False))
                attempts.append(RouteAttemptResult(target=target, success=False, error_type=error_type, emitted_output=emitted_output))
                has_next = index < len(decision.targets) - 1
                if not has_next or (stream and getattr(group, "streaming_policy", "pre_output_only") == "never") or not self.policy.should_fallback(error_type, group=group, emitted_output=emitted_output, stream=stream):
                    raise FallbackExhaustedError(decision, tuple(attempts)) from exc
        raise FallbackExhaustedError(decision, tuple(attempts))


def _error_type(error: BaseException) -> str:
    return normalize_route_error_type(str(getattr(error, "error_type", error.__class__.__name__)))
