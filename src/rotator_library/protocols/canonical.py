# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Shared canonical semantics for cross-protocol generative conversion.

Protocol modules own wire parsing and formatting. This module owns only meanings
that must be identical across those modules: logical operations, completion
reasons, instruction placement, source-aware passthrough, and warning records.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterable, Optional

from .types import (
    ContentBlock,
    ConversionWarning,
    ProtocolContext,
    ReasoningBlock,
    ToolCall,
    ToolResult,
    UnifiedMessage,
    UnifiedRequest,
    serialize_value,
)


STOP_REASON_STOP = "stop"
STOP_REASON_MAX_TOKENS = "max_tokens"
STOP_REASON_TOOL_USE = "tool_use"
STOP_REASON_CONTENT_FILTER = "content_filter"
STOP_REASON_ERROR = "error"
STOP_REASON_INCOMPLETE = "incomplete"
STOP_REASON_UNKNOWN = "unknown"
STOP_REASON_PAUSE = "pause"


_STOP_REASON_ALIASES = {
    "stop": STOP_REASON_STOP,
    "end_turn": STOP_REASON_STOP,
    "stop_sequence": STOP_REASON_STOP,
    "completed": STOP_REASON_STOP,
    "length": STOP_REASON_MAX_TOKENS,
    "max_tokens": STOP_REASON_MAX_TOKENS,
    "max_output_tokens": STOP_REASON_MAX_TOKENS,
    "model_context_window_exceeded": STOP_REASON_MAX_TOKENS,
    "tool_calls": STOP_REASON_TOOL_USE,
    "function_call": STOP_REASON_TOOL_USE,
    "tool_use": STOP_REASON_TOOL_USE,
    "content_filter": STOP_REASON_CONTENT_FILTER,
    "safety": STOP_REASON_CONTENT_FILTER,
    "blocklist": STOP_REASON_CONTENT_FILTER,
    "prohibited_content": STOP_REASON_CONTENT_FILTER,
    "recitation": STOP_REASON_CONTENT_FILTER,
    "refusal": STOP_REASON_CONTENT_FILTER,
    "failed": STOP_REASON_ERROR,
    "error": STOP_REASON_ERROR,
    "incomplete": STOP_REASON_INCOMPLETE,
    "pause_turn": STOP_REASON_PAUSE,
    # Gemini payload-shape failures (LANGUAGE/SPII = malformed candidate
    # text; MALFORMED_FUNCTION_CALL/TOO_MANY_TOOL_CALLS = malformed tool
    # traffic): retryable signal, distinct from clean completion.
    "language": STOP_REASON_INCOMPLETE,
    "spii": STOP_REASON_INCOMPLETE,
    "malformed_function_call": STOP_REASON_INCOMPLETE,
    "too_many_tool_calls": STOP_REASON_INCOMPLETE,
    "malformed_response": STOP_REASON_INCOMPLETE,
    # Gemini image-generation refusals are content-filter outcomes.
    "image_safety": STOP_REASON_CONTENT_FILTER,
    "image_prohibited_content": STOP_REASON_CONTENT_FILTER,
    "image_other": STOP_REASON_CONTENT_FILTER,
    "model_armor": STOP_REASON_CONTENT_FILTER,
}


_TARGET_STOP_REASONS = {
    "openai_chat": {
        STOP_REASON_STOP: "stop",
        STOP_REASON_MAX_TOKENS: "length",
        STOP_REASON_TOOL_USE: "tool_calls",
        STOP_REASON_CONTENT_FILTER: "content_filter",
        STOP_REASON_ERROR: None,
        STOP_REASON_INCOMPLETE: "length",
        STOP_REASON_PAUSE: "stop",
        STOP_REASON_UNKNOWN: None,
    },
    "anthropic_messages": {
        STOP_REASON_STOP: "end_turn",
        STOP_REASON_MAX_TOKENS: "max_tokens",
        STOP_REASON_TOOL_USE: "tool_use",
        STOP_REASON_CONTENT_FILTER: "refusal",
        STOP_REASON_ERROR: None,
        STOP_REASON_INCOMPLETE: "max_tokens",
        # pause_turn drives the server-tool agent loop and must survive
        # round-trips exactly (Anthropic clients replay paused content).
        STOP_REASON_PAUSE: "pause_turn",
        STOP_REASON_UNKNOWN: None,
    },
    "gemini": {
        STOP_REASON_STOP: "STOP",
        STOP_REASON_MAX_TOKENS: "MAX_TOKENS",
        STOP_REASON_TOOL_USE: "STOP",
        STOP_REASON_CONTENT_FILTER: "SAFETY",
        STOP_REASON_ERROR: "OTHER",
        STOP_REASON_INCOMPLETE: "MAX_TOKENS",
        STOP_REASON_PAUSE: "STOP",
        STOP_REASON_UNKNOWN: "OTHER",
    },
    "responses": {
        STOP_REASON_STOP: "completed",
        STOP_REASON_MAX_TOKENS: "incomplete",
        STOP_REASON_TOOL_USE: "completed",
        STOP_REASON_CONTENT_FILTER: "incomplete",
        STOP_REASON_ERROR: "failed",
        STOP_REASON_INCOMPLETE: "incomplete",
        STOP_REASON_PAUSE: "incomplete",
        STOP_REASON_UNKNOWN: "incomplete",
    },
}


def canonical_stop_reason(value: Any) -> Optional[str]:
    """Normalize a provider/client completion reason into a stable meaning."""

    if value is None:
        return None
    normalized = str(value).strip().lower()
    return _STOP_REASON_ALIASES.get(normalized, STOP_REASON_UNKNOWN)


def format_stop_reason(value: Optional[str], target_protocol: str) -> Optional[str]:
    """Return the target protocol's public completion reason.

    Unknown values pass through verbatim by design (same-protocol native
    spellings the tables have not met yet); known alias spellings map
    defensively so a hand-built canonical value (e.g. ``pause_turn`` where
    the canonical is ``pause``) never emits an illegal wire value.
    """

    if value is None:
        return None
    table = _TARGET_STOP_REASONS.get(target_protocol, {})
    if value in table:
        return table[value]
    alias = canonical_stop_reason(value)
    if alias in table:
        return table[alias]
    return value


def is_same_protocol(
    context: ProtocolContext | None,
    protocol_name: str,
    source_protocol: str | None = None,
) -> bool:
    """Return whether source-owned raw fields are safe to replay.

    A missing context is not proof of ownership. Parsed unified objects carry
    their source protocol, so direct parse/build calls still preserve native
    fields without making manually constructed or foreign objects unsafe.
    """

    effective_source = context.source_protocol if context and context.source_protocol else source_protocol
    return effective_source == protocol_name


def source_extensions(
    extra: dict[str, Any],
    context: ProtocolContext | None,
    protocol_name: str,
    source_protocol: str | None = None,
) -> dict[str, Any]:
    """Return source extensions only for a same-protocol destination."""

    return deepcopy(extra) if is_same_protocol(context, protocol_name, source_protocol) else {}


def may_emit_opaque_provider_state(
    context: ProtocolContext | None,
    *,
    preserve_source: bool,
) -> bool:
    """Return whether opaque signatures may leave canonical/cache state.

    Opaque state is suppressed unless real execution explicitly proves provider
    compatibility through a compatible-domain flag or identical provider IDs.
    """

    if not preserve_source:
        return False
    if context is None:
        return False
    if context.provider_state_compatible:
        return True
    return bool(
        context.source_provider
        and context.target_provider
        and context.source_provider == context.target_provider
    )


def instruction_messages(request: UnifiedRequest) -> list[UnifiedMessage]:
    """Return ordered canonical system/developer instructions.

    Older parsers store a separate ``system`` block list. That field is
    promoted as a LEADING instruction turn — including when explicit system
    messages also exist (both sources are legal together; dropping either
    would silently lose Required instruction content).
    """

    instructions: list[UnifiedMessage] = []
    if request.system:
        instructions.append(UnifiedMessage(role="system", content=deepcopy(request.system)))
    instructions.extend(message for message in request.messages if message.role in {"system", "developer"})
    return instructions


def conversation_messages(request: UnifiedRequest) -> list[UnifiedMessage]:
    """Return messages excluding system/developer instruction turns."""

    return [message for message in request.messages if message.role not in {"system", "developer"}]


def instruction_blocks(request: UnifiedRequest) -> list[ContentBlock]:
    """Flatten ordered instruction turns for protocols with one system field."""

    blocks: list[ContentBlock] = []
    for message in instruction_messages(request):
        blocks.extend(deepcopy(message.content))
    return blocks


def instruction_layout(request: UnifiedRequest) -> tuple[bool, int]:
    """Describe the instruction placement of a canonical request.

    Returns ``(interleaved, count)``: ``interleaved`` is True when a
    system/developer message appears after conversation turns began (its
    position cannot survive a merge into a single instruction field).
    """

    instruction_count = 1 if request.system else 0
    interleaved = False
    seen_conversation = False
    for message in request.messages:
        if message.role in {"system", "developer"}:
            instruction_count += 1
            if seen_conversation:
                interleaved = True
        else:
            seen_conversation = True
    return interleaved, instruction_count


def record_instruction_merge(request: UnifiedRequest, target_protocol: str) -> bool:
    """Record a D7 level-5 instruction merge when one occurred.

    Returns True when instructions were merged or repositioned (any count > 1
    instruction turn, or an interleaved instruction whose position is lost).
    """

    interleaved, count = instruction_layout(request)
    if count > 1 or interleaved:
        detail = []
        if count > 1:
            detail.append(f"{count} instruction turns merged in order")
        if interleaved:
            detail.append("interleaved instruction repositioned (destination mandates a single instruction field)")
        add_conversion_warning(
            request,
            code="instructions_merged",
            message="; ".join(detail),
            field="system",
            target_protocol=target_protocol,
        )
        return True
    return False


def add_conversion_warning(
    request: UnifiedRequest,
    *,
    code: str,
    message: str,
    field: str | None,
    target_protocol: str,
) -> None:
    """Record a deliberate omission of an optional conversion hint.

    Deduplicated on (code, message, field, target): build_request may run
    more than once for one request (retry/rotation passes over the same
    canonical object).
    """

    for warning in request.warnings:
        if (
            warning.code == code
            and warning.message == message
            and warning.field == field
            and warning.target_protocol == target_protocol
        ):
            return
    request.warnings.append(
        ConversionWarning(
            code=code,
            message=message,
            field=field,
            source_protocol=request.source_protocol,
            target_protocol=target_protocol,
        )
    )


# Deterministic reasoning-effort <-> budget-tokens approximation table (D7
# level 3: deterministic inference). Chosen once, documented, warned on use.
_EFFORT_TO_BUDGET_TOKENS = {
    "minimal": 1024,
    "low": 4096,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
    "max": 65536,
}

# Protocols whose native reasoning vocabulary is the OpenAI effort scale:
# effort labels pass through verbatim between them (xhigh/max included);
# the approximation table only serves foreign-vocabulary targets.
_EFFORT_NATIVE_PROTOCOLS = {"openai_chat", "responses"}


def budget_tokens_from_effort(effort: str) -> int:
    """Map a reasoning-effort label to a budget-token approximation."""

    return _EFFORT_TO_BUDGET_TOKENS.get(str(effort).strip().lower(), _EFFORT_TO_BUDGET_TOKENS["medium"])


def effort_from_budget_tokens(budget: int) -> str:
    """Map a thinking budget to the nearest effort label."""

    try:
        value = int(budget)
    except (TypeError, ValueError):
        return "medium"
    if value < 2048:
        return "minimal"
    if value < 8192:
        return "low"
    if value < 16384:
        return "medium"
    return "high"


def format_reasoning_controls(
    reasoning: Any,
    target_protocol: str,
    request: UnifiedRequest,
) -> dict[str, Any]:
    """Map canonical reasoning controls to a target protocol (D7 level 3).

    Canonical shape: ``{"effort", "budget_tokens", "enabled",
    "include_thoughts", "summary"}``. Parse-side normalization
    (:func:`normalize_reasoning_controls`) folds ``effort:"none"``,
    Responses ``summary``, and Gemini ``thinkingBudget: 0`` into this shape.
    Every inference (effort <-> budget) is deterministic via the documented
    table and recorded as a warning; every drop is recorded — nothing silent
    (defect 7). Responses gets a normalized native dict — never foreign keys.
    """

    emissions: dict[str, Any] = {}
    if not isinstance(reasoning, dict) or not reasoning:
        return emissions

    normalized = normalize_reasoning_controls(reasoning)
    effort = normalized.get("effort")
    budget = normalized.get("budget_tokens")
    enabled = normalized.get("enabled")
    include_thoughts = normalized.get("include_thoughts")
    summary = normalized.get("summary")
    thinking_type = normalized.get("thinking_type")
    normalized_display = normalized.get("display")

    if normalized.get("dynamic") is True and target_protocol != "gemini":
        # Dynamic thinking (thinkingBudget: -1, the model decides) is
        # Gemini-exclusive. Anthropic's adaptive thinking is the closest
        # equivalent construct (the model steers its own budget) — disclosed
        # as an approximation; every other target drops the flag disclosed.
        add_conversion_warning(
            request,
            code="reasoning_control_approximated" if target_protocol == "anthropic_messages" else "reasoning_control_dropped",
            message=(
                "dynamic thinking approximated as Anthropic adaptive thinking (the model steers its own budget)"
                if target_protocol == "anthropic_messages"
                else "dynamic thinking has no representation outside Gemini; dropped (the model-decides flag does not carry)"
            ),
            field="reasoning.dynamic",
            target_protocol=target_protocol,
        )
    if normalized_display is not None and target_protocol != "anthropic_messages":
        add_conversion_warning(
            request,
            code="reasoning_control_dropped",
            message="thinking display control has no representation outside Anthropic; dropped",
            field="reasoning.display",
            target_protocol=target_protocol,
        )

    def _warn(code: str, message: str, field: str) -> None:
        add_conversion_warning(request, code=code, message=message, field=field, target_protocol=target_protocol)

    def _effort_or_approximation() -> tuple[Any, bool]:
        """Return (effort_value, approximated?). ``none`` and table levels are
        exact; same-vocabulary targets pass any label verbatim (xhigh/max);
        unknown efforts at foreign-vocabulary targets coerce to the table's
        medium with an explicit disclosure warning."""

        if effort is None:
            return None, False
        if effort == "none":
            return effort, False
        if effort not in _EFFORT_TO_BUDGET_TOKENS:
            if target_protocol in _EFFORT_NATIVE_PROTOCOLS:
                # Same vocabulary (current docs: none/minimal/low/medium/
                # high/xhigh/max — plus forward-compatible labels): verbatim.
                return effort, False
            _warn(
                "reasoning_effort_unknown",
                f"reasoning effort '{effort}' is not a known level; coerced to 'medium' (deterministic table)",
                "reasoning.effort",
            )
            return "medium", True
        if effort in {"xhigh", "max"} and target_protocol not in _EFFORT_NATIVE_PROTOCOLS:
            # Known OpenAI-only levels at foreign-vocabulary targets degrade
            # through the table with disclosure (not silent).
            _warn(
                "reasoning_effort_approximated",
                f"reasoning effort '{effort}' has no exact {target_protocol} mapping; approximated to 'high' (budget table)",
                "reasoning.effort",
            )
            return "high", True
        return effort, False

    def _warn_budget_discarded() -> None:
        if budget is not None and effort is not None:
            _warn(
                "reasoning_control_dropped",
                "effort takes precedence; budget_tokens discarded",
                "reasoning.budget_tokens",
            )

    def _anthropic_budget(candidate: Any, max_output_tokens: Any) -> Any:
        """Anthropic requires max_tokens > thinking.budget_tokens: clamp with
        disclosure, or omit when no headroom exists (never a guaranteed
        provider 400)."""

        if not isinstance(candidate, int) or max_output_tokens is None or not isinstance(max_output_tokens, int):
            return candidate
        if candidate < max_output_tokens:
            return candidate
        if max_output_tokens <= 1024:
            _warn(
                "reasoning_budget_invalid",
                f"max_tokens={max_output_tokens} leaves no room for thinking (budget must stay below it and at 1024+); thinking omitted",
                "reasoning.budget_tokens",
            )
            return None
        clamped = max(1024, max_output_tokens - 1)
        if clamped < candidate:
            _warn(
                "reasoning_budget_coerced",
                f"thinking budget_tokens={candidate} exceeds max_tokens={max_output_tokens}; clamped to {clamped}",
                "reasoning.budget_tokens",
            )
            return clamped
        return candidate

    if target_protocol == "openai_chat":
        if enabled is False:
            _warn(
                "reasoning_disabled_omitted",
                "reasoning disabled has no OpenAI Chat control; effort omitted",
                "reasoning.enabled",
            )
        else:
            if thinking_type == "adaptive" and effort is None and budget is None:
                # Adaptive-only sources carry no effort/budget lever to map —
                # the drop is recorded, never silent.
                _warn(
                    "reasoning_control_dropped",
                    "adaptive thinking has no Chat effort representation; reasoning_effort omitted",
                    "reasoning",
                )
            if effort is not None:
                # Chat accepts the full effort vocabulary verbatim, "none"
                # included — exact mapping, no inference.
                value, _ = _effort_or_approximation()
                emissions["reasoning_effort"] = value
            elif budget is not None:
                approximated = effort_from_budget_tokens(budget)
                emissions["reasoning_effort"] = approximated
                _warn(
                    "reasoning_budget_approximated",
                    f"reasoning budget_tokens={budget} approximated as effort '{approximated}' (deterministic table)",
                    "reasoning.budget_tokens",
                )
            _warn_budget_discarded()
        if include_thoughts is not None:
            _warn(
                "reasoning_control_dropped",
                "include_thoughts has no OpenAI Chat representation",
                "reasoning.include_thoughts",
            )
        if summary is not None:
            _warn(
                "reasoning_control_dropped",
                "reasoning summary preference has no OpenAI Chat representation",
                "reasoning.summary",
            )
    elif target_protocol == "anthropic_messages":
        # "none" maps exactly to Anthropic's disabled construct (D7 level 1).

        def _emit_thinking(config: dict[str, Any]) -> None:
            if normalized_display is not None:
                config["display"] = normalized_display
            emissions["thinking"] = config

        disabled = enabled is False or effort == "none"
        if disabled and (budget is not None or effort not in (None, "none")):
            _warn(
                "reasoning_control_dropped",
                "reasoning disabled wins; budget/effort control discarded",
                "reasoning",
            )
        if disabled:
            _emit_thinking({"type": "disabled"})
        elif thinking_type == "adaptive" or (thinking_type is None and budget is None and effort is None and enabled):
            # Adaptive: Anthropic steers the budget itself; a carried effort
            # stays in its native output_config lever (never discarded).
            _emit_thinking({"type": "adaptive"})
            if effort is not None:
                emissions["output_config"] = {"effort": effort}
        elif budget is not None:
            if isinstance(budget, int) and budget < 1024:
                _warn(
                    "reasoning_budget_invalid",
                    f"thinking budget_tokens={budget} is below Anthropic's 1024 minimum; thinking omitted",
                    "reasoning.budget_tokens",
                )
            else:
                clamped = _anthropic_budget(budget, (request.generation_params or {}).get("max_output_tokens"))
                if clamped is not None:
                    _emit_thinking({"type": "enabled", "budget_tokens": clamped})
        elif effort is not None:
            value, coerced = _effort_or_approximation()
            approximated = _anthropic_budget(budget_tokens_from_effort(value), (request.generation_params or {}).get("max_output_tokens"))
            if approximated is not None:
                _emit_thinking({"type": "enabled", "budget_tokens": approximated})
                _warn(
                    "reasoning_effort_model_dependent",
                    "effort mapped to thinking{enabled,budget_tokens}; current flagships (adaptive-only models) reject this shape — verify the target model accepts enabled thinking",
                    "reasoning.effort",
                )
            if not coerced:
                _warn(
                    "reasoning_effort_approximated",
                    f"reasoning effort '{effort}' approximated as budget_tokens={budget_tokens_from_effort(value)} (deterministic table)",
                    "reasoning.effort",
                )
        if include_thoughts is not None:
            _warn(
                "reasoning_control_dropped",
                "include_thoughts has no Anthropic Messages control (summaries follow the thinking setting)",
                "reasoning.include_thoughts",
            )
        if summary is not None:
            _warn(
                "reasoning_control_dropped",
                "reasoning summary preference has no Anthropic Messages representation",
                "reasoning.summary",
            )
    elif target_protocol == "responses":
        if enabled is False:
            _warn(
                "reasoning_disabled_omitted",
                "reasoning disabled has no Responses control; reasoning block omitted",
                "reasoning.enabled",
            )
        else:
            native: dict[str, Any] = {}
            if effort is not None:
                # Responses accepts the full effort vocabulary verbatim,
                # "none" included — exact mapping, no inference.
                value, _ = _effort_or_approximation()
                native["effort"] = value
            elif budget is not None:
                approximated = effort_from_budget_tokens(budget)
                native["effort"] = approximated
                _warn(
                    "reasoning_budget_approximated",
                    f"reasoning budget_tokens={budget} approximated as effort '{approximated}' (deterministic table)",
                    "reasoning.budget_tokens",
                )
            _warn_budget_discarded()
            if include_thoughts is not None:
                native["summary"] = "auto" if include_thoughts else "none"
            elif summary is not None:
                native["summary"] = summary
            if native:
                # Normalized native dict only — foreign spellings never leak
                # onto the Responses wire (no hybrid payloads).
                emissions["reasoning"] = native
    elif target_protocol == "gemini":
        thinking_config: dict[str, Any] = {}
        if enabled is False or effort == "none":
            # thinkingBudget: 0 is Gemini's off-switch — model-dependent
            # (thinking can only be disabled on some models), so the
            # emission is recorded, never silent. includeThoughts is
            # forced off: the API rejects it alongside disabled thinking.
            thinking_config["thinkingBudget"] = 0
            thinking_config["includeThoughts"] = False
            _warn(
                "reasoning_disabled_model_dependent",
                "thinkingBudget=0 disables thinking only on models that support disabling; verify the target model",
                "reasoning.enabled",
            )
        elif budget is not None and effort is None:
            thinking_config["thinkingBudget"] = budget
        elif effort is not None:
            value, coerced = _effort_or_approximation()
            approximated = budget_tokens_from_effort(value)
            thinking_config["thinkingBudget"] = approximated
            if not coerced:
                _warn(
                    "reasoning_effort_approximated",
                    f"reasoning effort '{effort}' approximated as thinkingBudget={approximated} (deterministic table)",
                    "reasoning.effort",
                )
        elif enabled is True and normalized.get("dynamic") is True:
            # Gemini dynamic thinking (thinkingBudget: -1): the model decides.
            thinking_config["thinkingBudget"] = -1
        elif enabled is True and include_thoughts is None and summary is None:
            _warn(
                "reasoning_control_dropped",
                "reasoning enabled without a budget/effort has no Gemini representation",
                "reasoning.enabled",
            )
        _warn_budget_discarded()
        if not (enabled is False or effort == "none"):
            if include_thoughts is not None:
                thinking_config["includeThoughts"] = bool(include_thoughts)
            elif summary is not None:
                thinking_config["includeThoughts"] = summary != "none"
        if thinking_config:
            emissions["generation_config"] = {"thinkingConfig": thinking_config}
    return emissions


def normalize_reasoning_controls(reasoning: Any) -> dict[str, Any]:
    """Fold provider spellings into the canonical reasoning shape.

    - Responses ``summary`` -> ``include_thoughts`` (summary preserved verbatim
      for same-protocol rebuild)
    - ``effort: "none"`` stays a first-class effort value: Chat and Responses
      accept it natively; Anthropic/Gemini targets map it to their disabled
      constructs at format time
    """

    if not isinstance(reasoning, dict):
        return {}
    normalized = dict(reasoning)
    summary = normalized.get("summary")
    if isinstance(summary, str) and "include_thoughts" not in normalized:
        normalized["include_thoughts"] = summary != "none"
    return {key: value for key, value in normalized.items() if value is not None or key in {"enabled", "include_thoughts"}}


def disclose_response_drops(unified_response: Any, target_protocol: str) -> None:
    """Disclose response-side conversions that cannot carry a warning at
    their emission point (stop-reason approximations, usage detail drops).

    Called once per format_response pass; appends to the response warnings
    (deduplicated — repeated format passes on one object never duplicate).
    """

    from .types import ConversionWarning  # local import: avoid cycles

    def _disclose(code: str, message: str, field: str) -> None:
        if not any(w.code == code and w.message == message for w in unified_response.warnings):
            unified_response.warnings.append(
                ConversionWarning(
                    code=code,
                    message=message,
                    field=field,
                    source_protocol=getattr(unified_response, "source_protocol", None),
                    target_protocol=target_protocol,
                )
            )

    stop_reason = getattr(unified_response, "stop_reason", None)
    if stop_reason == "pause" and target_protocol != "anthropic_messages":
        _disclose(
            "stop_reason_approximated",
            "pause_turn has no representation outside Anthropic; approximated (the server-tool loop semantics do not carry)",
            "stop_reason",
        )
    metadata = getattr(unified_response, "metadata", None) or {}
    extra = getattr(unified_response, "extra", None) or {}
    matched_sequence = (
        metadata.get("stop_sequence")
        if isinstance(metadata, dict) and metadata.get("stop_sequence") is not None
        else (extra.get("stop_sequence") if isinstance(extra, dict) else None)
    )
    if matched_sequence is not None and target_protocol != "anthropic_messages":
        _disclose(
            "stop_sequence_dropped",
            "stop_sequence (the matched sequence text) has no representation outside Anthropic; only the stop reason survives",
            "stop_reason",
        )
    usage = getattr(unified_response, "usage", None)
    usage_extra = getattr(usage, "extra", None) if usage is not None else None
    if isinstance(usage_extra, dict) and usage_extra:
        anthropic_native = {"cache_creation", "server_tool_use", "service_tier"}
        gemini_native = {"tool_use_prompt_tokens"}
        keys = [
            key
            for key in sorted(usage_extra)
            if not (target_protocol == "anthropic_messages" and key in anthropic_native)
            and not (target_protocol == "gemini" and key in gemini_native)
        ]
        if keys:
            _disclose(
                "usage_detail_dropped",
                f"usage detail buckets have no {target_protocol} spelling; dropped: {', '.join(keys)}",
                "usage",
            )


def attach_conversion_summary(payload: dict[str, Any], unified_response: Any) -> dict[str, Any]:
    """Attach the recorded conversion summary to a client response payload.

    The summary renders recorded warnings (deliberate omissions, merges,
    approximations) under the ``x-proxy-conversion`` extension key — present
    only when warnings exist. Raw-passthrough responses never carry it (they
    produce no warnings); D7's "recorded summary" becomes client-visible
    instead of internal-only.
    """

    summary = conversion_summary(getattr(unified_response, "warnings", None) or [])
    if summary is not None:
        payload["x-proxy-conversion"] = summary
    return payload


def conversion_summary(
    warnings: list[Any],
) -> dict[str, Any] | None:
    """Render recorded conversion warnings as a wire-safe summary block.

    Returned as the value of the response's ``x-proxy-conversion`` extension
    key: present only when warnings exist, never fabricated, and never on
    raw-passthrough (same-protocol fast) responses, which produce no warnings.
    """

    entries = [w for w in warnings if getattr(w, "code", None)]
    if not entries:
        return None
    rendered: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for warning in entries:
        key = (getattr(warning, "code", None), getattr(warning, "message", None), getattr(warning, "field", None), getattr(warning, "target_protocol", None))
        if key in seen:
            continue
        seen.add(key)
        rendered.append(
            {
                "code": warning.code,
                "message": warning.message,
                "field": warning.field,
                "target": getattr(warning, "target_protocol", None),
            }
        )
    return {"warnings": rendered}


def retain_supported_generation_params(
    request: UnifiedRequest,
    params: dict[str, Any],
    *,
    supported: set[str],
    target_protocol: str,
) -> dict[str, Any]:
    """Return supported optional controls and record every deliberate omission."""

    kept: dict[str, Any] = {}
    for key, value in params.items():
        if key in supported:
            kept[key] = value
            continue
        add_conversion_warning(
            request,
            code="unsupported_optional_control",
            message=f"{target_protocol} has no safe mapping for optional control '{key}'",
            field=key,
            target_protocol=target_protocol,
        )
    return kept


def canonical_tool_choice(value: Any, source_protocol: str) -> dict[str, Any] | None:
    """Normalize tool-choice spellings used by the four generative APIs."""

    if value is None:
        return None
    if isinstance(value, str):
        mode = value.lower()
        if mode in {"auto", "none", "required"}:
            return {"mode": mode}
        if mode in {"any", "any_required"}:
            return {"mode": "required"}
        return {"mode": "named", "name": value}
    if not isinstance(value, dict):
        return {"mode": "auto"}

    value_type = str(value.get("type") or value.get("mode") or "auto").lower()
    if source_protocol == "gemini":
        # Gemini normalization normally happens in its protocol module because
        # the value is nested in toolConfig; accept the canonical intermediate.
        if "allowed_names" in value:
            return deepcopy(value)
    if value_type in {"auto", "none"}:
        result = {"mode": value_type}
        if isinstance(value.get("allowed_names"), list) and value["allowed_names"]:
            result["allowed_names"] = deepcopy(value["allowed_names"])
        if value.get("disable_parallel_tool_use") is True:
            result["disable_parallel_tool_use"] = True
        return result
    if value_type in {"required", "any"}:
        result = {"mode": "required", "allowed_names": deepcopy(value.get("allowed_names") or [])}
        if value.get("disable_parallel_tool_use") is True:
            result["disable_parallel_tool_use"] = True
        return result
    if value_type not in {"function", "tool", "named", "allowed_tools"} and isinstance(value.get("type"), str):
        # Namespaced tool-choice variants (mcp{server_label,name}, custom,
        # local_shell, apply_patch, allowed_types...): preserve the raw shape
        # for native round-trips; foreign targets warn instead of silently
        # widening to auto.
        return {"mode": "auto", "namespaced": deepcopy(value)}
    if value_type == "allowed_tools":
        # Chat allowed-tools constraint. Variant spellings: {"type":"allowed_tools",
        # "allowed_tools": {"mode":auto|required, "tools":[...]}} or a bare list.
        raw_allowed = value.get("allowed_tools")
        if isinstance(raw_allowed, dict):
            allowed = raw_allowed
        elif isinstance(raw_allowed, list):
            allowed = {"mode": "auto", "tools": raw_allowed}
        else:
            allowed = {}
        mode = str(allowed.get("mode") or "auto").lower()
        if mode not in {"auto", "required"}:
            mode = "auto"
        result: dict[str, Any] = {"mode": mode, "allowed_names": deepcopy(allowed.get("tools") or [])}
        if isinstance(raw_allowed, dict):
            result["allowed_tools"] = deepcopy(raw_allowed)
        elif isinstance(raw_allowed, list):
            result["allowed_tools"] = {"mode": mode, "tools": deepcopy(raw_allowed)}
        return result
    if value_type in {"function", "tool", "named"}:
        function = value.get("function") if isinstance(value.get("function"), dict) else {}
        name = value.get("name") or function.get("name")
        result = {"mode": "named", "name": name}
        if value.get("disable_parallel_tool_use") is True:
            # The parallelism constraint carries in named mode too — the
            # chat target maps it to its parallel_tool_calls sibling.
            result["disable_parallel_tool_use"] = True
        return result
    return {"mode": "auto"}


def format_tool_choice(value: Any, target_protocol: str) -> Any:
    """Format canonical tool choice for a destination protocol."""

    choice = value if isinstance(value, dict) and "mode" in value else canonical_tool_choice(value, target_protocol)
    if not choice:
        return None
    mode = choice.get("mode", "auto")
    name = choice.get("name")
    allowed_names = deepcopy(choice.get("allowed_names") or [])
    no_parallel = choice.get("disable_parallel_tool_use") is True
    if target_protocol != "responses" and isinstance(choice.get("namespaced"), dict):
        # Namespaced variants have no representation outside Responses —
        # callers narrow to the mode; the warning lands there via the
        # canonical record (mode stays "auto").
        pass
    if target_protocol == "openai_chat":
        # disable_parallel_tool_use has an exact native sibling here:
        # parallel_tool_calls:false — the caller merges it into the payload.
        if mode == "named":
            return {"type": "function", "function": {"name": name or ""}}
        if allowed_names and choice.get("allowed_tools") is None:
            # Constraint without the native variant shape (e.g. from Gemini
            # allowedFunctionNames): the plain modes cannot express an
            # allowlist — closest legal narrowing is required.
            return "required" if mode == "required" else mode
        if choice.get("allowed_tools") is not None:
            return {"type": "allowed_tools", "allowed_tools": deepcopy(choice["allowed_tools"])}
        return "required" if mode == "required" else mode
    if target_protocol == "anthropic_messages":
        if mode == "none":
            # Anthropic has no {"type":"none"}: tools are disabled by
            # omitting the tools array (callers handle that + warn).
            return None
        if mode == "named":
            payload: dict[str, Any] = {"type": "tool", "name": name or ""}
            if no_parallel:
                payload["disable_parallel_tool_use"] = True
            return payload
        payload = {"type": "any" if mode == "required" else "auto"}
        if no_parallel:
            payload["disable_parallel_tool_use"] = True
        return payload
    if target_protocol == "responses":
        if mode == "named":
            return {"type": "function", "name": name or ""}
        if isinstance(choice.get("namespaced"), dict):
            # Namespaced variants are Responses-native: round-trip verbatim.
            return deepcopy(choice["namespaced"])
        return "required" if mode == "required" else mode
    if target_protocol == "gemini":
        if mode == "none":
            config: dict[str, Any] = {"mode": "NONE"}
        elif mode == "required":
            config = {"mode": "ANY"}
            if allowed_names:
                config["allowedFunctionNames"] = allowed_names
        elif mode == "named":
            config = {"mode": "ANY", "allowedFunctionNames": [name] if name else []}
        else:
            config = {"mode": "AUTO"}
            if allowed_names:
                # AUTO within an allowlist is a legal Gemini shape.
                config["allowedFunctionNames"] = allowed_names
        return {"functionCallingConfig": config}
    return deepcopy(value)


def canonical_structured_output(value: Any, source_protocol: str) -> dict[str, Any] | None:
    """Normalize JSON/object schema controls without source wrappers."""

    if not isinstance(value, dict):
        return None
    if source_protocol == "openai_chat":
        output_type = value.get("type")
        if output_type == "json_schema" and isinstance(value.get("json_schema"), dict):
            schema = value["json_schema"]
            return {
                "type": "json_schema",
                "name": schema.get("name"),
                "schema": deepcopy(schema.get("schema")),
                "strict": schema.get("strict"),
            }
        if output_type == "json_object":
            return {"type": "json_object"}
        return deepcopy(value)
    if source_protocol == "responses":
        return {
            "type": value.get("type") or "json_schema",
            "name": value.get("name"),
            "schema": deepcopy(value.get("schema")),
            "strict": value.get("strict"),
        }
    if source_protocol == "anthropic_messages":
        format_value = value.get("format") if isinstance(value.get("format"), dict) else value
        return {
            "type": format_value.get("type") or "json_schema",
            "name": format_value.get("name"),
            "schema": deepcopy(format_value.get("schema")),
            "strict": format_value.get("strict"),
        }
    if source_protocol == "gemini":
        normalized = deepcopy(value)
        if normalized.get("type") == "json_schema" and normalized.get("schema") is not None:
            normalized.setdefault("strict", True)
        return normalized
    return deepcopy(value)


def format_structured_output(value: Any, target_protocol: str) -> Any:
    """Format a canonical structured-output requirement for a destination."""

    if not isinstance(value, dict):
        return None
    output_type = value.get("type") or "json_schema"
    if target_protocol == "openai_chat":
        if output_type == "json_object":
            return {"type": "json_object"}
        if output_type == "text":
            # Text output is the absence of a format constraint.
            return None
        if output_type != "json_schema":
            # Unknown/custom formats (e.g. grammar) have no Chat
            # representation — callers drop with a recorded warning rather
            # than fabricating an empty json_schema.
            return None
        return {
            "type": "json_schema",
            "json_schema": {
                key: deepcopy(item)
                for key, item in {
                    "name": value.get("name") or "response",
                    "schema": value.get("schema") or {},
                    "strict": value.get("strict"),
                }.items()
                if item is not None
            },
        }
    if target_protocol == "responses":
        if output_type == "json_object":
            return {"type": "json_object"}
        if output_type == "text":
            # Text is the documented default format — an explicit constraint,
            # never a fabricated json_schema.
            return {"type": "text"}
        if output_type != "json_schema":
            return None
        return {
            key: deepcopy(item)
            for key, item in {
                "type": "json_schema",
                "name": value.get("name") or "response",
                "schema": value.get("schema") or {},
                "strict": value.get("strict"),
            }.items()
            if item is not None
        }
    if target_protocol == "anthropic_messages":
        if output_type == "json_object":
            return {"format": {"type": "json_schema", "schema": {"type": "object"}}}
        return {
            "format": {
                key: deepcopy(item)
                for key, item in {
                    "type": "json_schema",
                    "name": value.get("name"),
                    "schema": value.get("schema") or {},
                    "strict": value.get("strict"),
                }.items()
                if item is not None
            }
        }
    if target_protocol == "gemini":
        if output_type == "text":
            # Text is Gemini's default output — an explicit text constraint
            # is the absence of a response schema, never application/json.
            return None
        if output_type not in {"json_schema", "json_object"}:
            return None
        result: dict[str, Any] = {"responseMimeType": "application/json"}
        if output_type != "json_object":
            result["responseJsonSchema"] = deepcopy(value.get("schema"))
        return result
    return deepcopy(value)


def message_tool_calls(message: UnifiedMessage) -> list[ToolCall]:
    """Return de-duplicated calls from message and content representations."""

    calls: list[ToolCall] = []
    seen: set[tuple[Optional[str], Optional[str], str]] = set()
    for call in [*message.tool_calls, *[block.tool_call for block in message.content if block.tool_call]]:
        if call is None:
            continue
        key = (call.id, call.name, json.dumps(serialize_value(call.arguments), sort_keys=True, separators=(",", ":")))
        if key not in seen:
            calls.append(call)
            seen.add(key)
    return calls


def message_reasoning(message: UnifiedMessage) -> list[ReasoningBlock]:
    """Return de-duplicated reasoning from message and content representations."""

    blocks: list[ReasoningBlock] = []
    seen: set[tuple[Optional[str], Optional[str], bool]] = set()
    for reasoning in [*message.reasoning, *[block.reasoning for block in message.content if block.reasoning]]:
        if reasoning is None:
            continue
        key = (reasoning.text, reasoning.signature, reasoning.redacted)
        if key not in seen:
            blocks.append(reasoning)
            seen.add(key)
    return blocks


def message_tool_results(message: UnifiedMessage) -> list[ToolResult]:
    """Return tool results embedded in a canonical message."""

    return [block.tool_result for block in message.content if block.tool_result is not None]


def resolve_tool_result_names(messages: Iterable[UnifiedMessage]) -> list[UnifiedMessage]:
    """Enrich result records with function names from preceding calls.

    Chat and Anthropic identify results by call ID, while Gemini requires the
    function name on its response part. Keeping both in the canonical record
    allows either direction without provider-specific history lookups.
    """

    message_list = list(messages)
    names: dict[str, str] = {}
    ids_by_name: dict[str, list[str]] = {}
    result_index_by_name: dict[str, int] = {}
    call_index = 0
    for message in message_list:
        for call in message_tool_calls(message):
            if not call.id:
                call.id = f"call_{call_index}"
                call.extra["synthetic_id"] = True
            call_index += 1
            if call.id and call.name:
                names[call.id] = call.name
                ids_by_name.setdefault(call.name, []).append(call.id)
        for result in message_tool_results(message):
            if not result.name and result.tool_call_id in names:
                result.name = names[result.tool_call_id]
            if result.name and (not result.tool_call_id or result.tool_call_id == result.name):
                candidates = ids_by_name.get(result.name, [])
                result_index = result_index_by_name.get(result.name, 0)
                if result_index < len(candidates):
                    result.tool_call_id = candidates[result_index]
                    result.extra["synthetic_tool_call_id"] = True
                    result_index_by_name[result.name] = result_index + 1
    return message_list


def normalize_tool_result_messages(messages: Iterable[UnifiedMessage]) -> list[UnifiedMessage]:
    """Split embedded provider result blocks into canonical tool-role turns."""

    normalized: list[UnifiedMessage] = []
    for message in messages:
        if not any(block.tool_result for block in message.content):
            normalized.append(message)
            continue
        groups: list[tuple[bool, list[ContentBlock]]] = []
        for block in message.content:
            is_result = block.tool_result is not None
            if groups and groups[-1][0] == is_result:
                groups[-1][1].append(block)
            else:
                groups.append((is_result, [block]))
        for group_index, (is_result, blocks) in enumerate(groups):
            split = deepcopy(message)
            split.content = blocks
            split.role = "tool" if is_result else message.role
            split.tool_calls = [block.tool_call for block in blocks if block.tool_call]
            split.reasoning = [block.reasoning for block in blocks if block.reasoning]
            split.tool_call_id = blocks[0].tool_result.tool_call_id if is_result and blocks[0].tool_result else None
            if group_index:
                split.raw = None
                split.extra = {}
            normalized.append(split)
    return normalized


def ordered_message_blocks(message: UnifiedMessage) -> list[ContentBlock]:
    """Return one ordered block sequence without duplicate promoted fields."""

    blocks = deepcopy(message.content)
    reasoning_keys = {
        (block.reasoning.text, block.reasoning.signature, block.reasoning.redacted)
        for block in blocks
        if block.reasoning
    }
    call_keys = {
        (block.tool_call.id, block.tool_call.name, json.dumps(serialize_value(block.tool_call.arguments), sort_keys=True))
        for block in blocks
        if block.tool_call
    }
    missing_reasoning: list[ContentBlock] = []
    for reasoning in message.reasoning:
        key = (reasoning.text, reasoning.signature, reasoning.redacted)
        if key not in reasoning_keys:
            missing_reasoning.append(ContentBlock(type="reasoning", reasoning=deepcopy(reasoning)))
            reasoning_keys.add(key)
    if missing_reasoning:
        blocks = [*missing_reasoning, *blocks]
    for call in message.tool_calls:
        key = (call.id, call.name, json.dumps(serialize_value(call.arguments), sort_keys=True))
        if key not in call_keys:
            blocks.append(ContentBlock(type="tool_call", tool_call=deepcopy(call)))
            call_keys.add(key)
    return blocks


def coalesce_assistant_message(messages: Iterable[UnifiedMessage]) -> UnifiedMessage:
    """Collapse item-oriented provider output into one assistant message."""

    merged = UnifiedMessage(role="assistant")
    for message in messages:
        if message.role not in {"assistant", "model"}:
            continue
        merged.content.extend(ordered_message_blocks(message))
    merged.reasoning = message_reasoning(merged)
    merged.tool_calls = message_tool_calls(merged)
    return merged


def canonical_tool_arguments(value: Any) -> Any:
    """Decode JSON tool arguments while retaining invalid source text."""

    if not isinstance(value, str):
        return deepcopy(value)
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def tool_arguments_object(value: Any) -> dict[str, Any]:
    """Return object arguments required by Anthropic and Gemini."""

    normalized = canonical_tool_arguments(value)
    if isinstance(normalized, dict):
        return normalized
    raise ValueError("tool arguments must be a JSON object")


def tool_arguments_text(value: Any) -> str:
    """Return compact JSON arguments required by Chat and Responses."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(serialize_value(value), separators=(",", ":"))


def tool_result_text(value: Any) -> str:
    """Return the string result representation required by Chat/Responses."""

    if isinstance(value, str):
        return value
    return json.dumps(serialize_value(value), separators=(",", ":"))


def tool_result_object(value: Any) -> dict[str, Any]:
    """Return the object result representation required by Gemini."""

    normalized = canonical_tool_arguments(value)
    if isinstance(normalized, dict):
        return normalized
    return {"result": serialize_value(normalized)}
