# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Canonical reasoning-effort vocabulary and normalization (G8).

One ordered ladder owns the math; capability declarations state what a
model on a provider ACCEPTS; the normalizer maps any incoming word to
the nearest accepted rung. Providers stop writing vocabulary maps.

Rules (locked):

- ``off`` is a collapsed group: none/off/disable/disabled all mean OFF.
- Nearest accepted rung wins, searching UPWARD first, then downward —
  ties round up (``medium`` on {low, high} lands on ``high``, the
  official DeepSeek fold).
- An ON word never normalizes to OFF: only explicit OFF words produce
  OFF. If no ON rung is accepted at all, the effort control drops with
  a disclosure note.
- Every normalization returns a note for the conversion-warning channel
  — nothing silent.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

OFF_WORDS: FrozenSet[str] = frozenset({"none", "off", "disable", "disabled"})

# The ordered ladder. Position defines "nearest".
EFFORT_LADDER: Tuple[str, ...] = (
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "ultra",
    "max",
)

_LADDER_INDEX: Dict[str, int] = {rung: index for index, rung in enumerate(EFFORT_LADDER)}

# Protocol base vocabularies: the no-declaration default per family.
CHAT_BASE_ACCEPTED: Tuple[str, ...] = ("off", "low", "medium", "high")


def canonical_effort_word(value: Any) -> Optional[str]:
    """Fold any incoming spelling into the canonical vocabulary.

    Returns ``"off"`` for the off-group, a ladder rung, or ``None`` for
    words the system does not know (callers drop those with a note).
    """

    word = str(value or "").strip().lower()
    if not word:
        return None
    if word in OFF_WORDS:
        return "off"
    return word if word in _LADDER_INDEX else None


def normalize_effort(word: str, accepted: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
    """Normalize ``word`` against the accepted set.

    Returns ``(normalized, note)`` — ``normalized`` is an accepted rung,
    ``"off"`` when the word was an off-word and off is accepted, or
    ``None`` when the control must drop. ``note`` is set on every
    transformation (never silent).
    """

    canonical = canonical_effort_word(word)
    accepted_set = {str(rung).strip().lower() for rung in accepted or ()}
    if canonical is None:
        return None, f"effort {word!r} is not a known reasoning-effort word; dropped"
    if canonical in accepted_set:
        return canonical, None
    if canonical == "off":
        if "off" in accepted_set:
            return "off", None
        # OFF asked, off not accepted: drop the control rather than
        # fabricating an on-rung (the model cannot be disabled).
        return None, "thinking-off requested but this model does not accept an off control; dropped"

    # ON word: nearest accepted rung by ladder distance, ties round UP
    # (medium on {low, high} lands on high — the official fold) — never
    # into "off".
    index = _LADDER_INDEX[canonical]
    best: Optional[Tuple[int, str]] = None
    for candidate_index, rung in enumerate(EFFORT_LADDER):
        if rung in accepted_set:
            distance = abs(candidate_index - index)
            # Strictly nearer wins; equal distance keeps the LATER rung
            # (round up).
            if best is None or distance <= best[0]:
                best = (distance, rung)
    if best is not None:
        return best[1], f"effort {canonical} normalized to {best[1]} (accepted: {sorted(accepted_set)})"
    if "off" in accepted_set:
        # No on-rung accepted at all — dropping the control leaves the
        # model at its default rather than fabricating reasoning.
        return None, f"model accepts no on-reasoning rung (accepted: {sorted(accepted_set)}); effort dropped"
    return None, f"effort {canonical} has no accepted rung on this model; dropped"


# ---------------------------------------------------------------------------
# Accepted-set resolution chain (locked order, later wins)
# ---------------------------------------------------------------------------

# protocol base → model/provider database (the seam that makes providers
# declare nothing) → provider code (an override of the DB) → model rows in
# provider code → config. The database seam is an empty resolver today.
_DB_RESOLVER: Any = None


def register_effort_database_resolver(resolver: Any) -> None:
    """Register the models.dev-sourced capability resolver (future phase).

    Signature: ``resolver(provider: str, model: str) -> Optional[set[str]]``
    returning the accepted effort vocabulary or None when the database
    has no row.
    """

    global _DB_RESOLVER
    _DB_RESOLVER = resolver


def protocol_base_accepted(protocol_family: str) -> Tuple[str, ...]:
    """The no-declaration vocabulary for a protocol family."""

    return CHAT_BASE_ACCEPTED


def _model_rule_rows(provider_plugin: Any, model: str) -> Tuple[Any, ...]:
    """The provider's effective model_rules rows (class + JSON runtime).

    ``ProviderInterface._model_rules_rows`` is the one merged view (JSON
    config appends after class rows); a bare plugin object without the
    helper falls back to its class declaration.
    """

    getter = getattr(provider_plugin, "_model_rules_rows", None)
    if callable(getter):
        try:
            rows = getter(model)
        except Exception:
            rows = None
        if rows is not None:
            return tuple(rows)
    return tuple(getattr(provider_plugin, "model_rules", None) or ())


def resolve_accepted_effort(
    provider_plugin: Any = None,
    model: str = "",
    *,
    runtime_config: Any = None,
    protocol_family: str = "",
) -> Tuple[Tuple[str, ...], str]:
    """Resolve the accepted effort vocabulary through the chain.

    Returns ``(accepted, source)`` where source names the deciding layer
    (for disclosure notes and tests).
    """

    accepted: Tuple[str, ...] = ()
    source = "protocol_base"
    base = protocol_base_accepted(protocol_family or "openai_chat")
    if base:
        accepted, source = tuple(base), "protocol_base"

    if _DB_RESOLVER is not None and provider_plugin is not None:
        try:
            provider_name = getattr(provider_plugin, "provider_env_name", "") or ""
            db_value = _DB_RESOLVER(provider_name, model)
            if db_value:
                accepted, source = tuple(sorted(str(r) for r in db_value)), "model_database"
        except Exception:
            pass

    provider_level = getattr(provider_plugin, "reasoning_effort_accept", None) if provider_plugin is not None else None
    if provider_level:
        accepted, source = tuple(str(r) for r in provider_level), "provider_code"

    if provider_plugin is not None:
        from ..adapters.param_rules import _model_match_candidates, _row_matches

        provider_name = getattr(provider_plugin, "provider_env_name", "") or ""
        candidates = _model_match_candidates(model, provider_name)
        for row in _model_rule_rows(provider_plugin, model):
            if not isinstance(row, Mapping) or not _row_matches(row, candidates):
                continue
            row_accept = row.get("effort_accept")
            if row_accept:
                accepted, source = tuple(str(r) for r in row_accept), f"model_rules:{row.get('match', '*')}"

    if isinstance(runtime_config, dict):
        config_accept = runtime_config.get("reasoning_effort_accept")
        if config_accept:
            accepted, source = tuple(str(r) for r in config_accept), "config"

    return accepted, source


def resolve_effort_toggle(
    provider_plugin: Any = None,
    model: str = "",
    *,
    runtime_config: Any = None,
) -> bool:
    """Whether the OFF control rides the provider's thinking toggle.

    The base is the provider-level ``reasoning_effort_toggle`` attribute;
    matching ``model_rules`` rows declaring ``toggle`` override it in
    cascade order (later rows win), and a flat runtime-config
    ``reasoning_effort_toggle`` decides last.
    """

    toggle = bool(getattr(provider_plugin, "reasoning_effort_toggle", False)) if provider_plugin is not None else False

    if provider_plugin is not None:
        from ..adapters.param_rules import _model_match_candidates, _row_matches

        provider_name = getattr(provider_plugin, "provider_env_name", "") or ""
        candidates = _model_match_candidates(model, provider_name)
        for row in _model_rule_rows(provider_plugin, model):
            if isinstance(row, Mapping) and _row_matches(row, candidates) and "toggle" in row:
                toggle = bool(row["toggle"])

    if isinstance(runtime_config, dict) and "reasoning_effort_toggle" in runtime_config:
        toggle = bool(runtime_config["reasoning_effort_toggle"])

    return toggle


def effort_accepts_toggle(accepted: Iterable[str]) -> bool:
    """Whether the OFF control rides a thinking toggle on this wire.

    Providers/models declare ``toggle`` in their model_rules row; this
    helper exists for emission sites that check the resolved
    declarations.
    """

    return "off" in {str(rung).strip().lower() for rung in accepted or ()}
