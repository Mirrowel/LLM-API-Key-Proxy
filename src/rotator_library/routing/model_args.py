# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Model-string arguments (G8): ``provider/model:arg`` request hints.

Clients that cannot set request parameters directly (fixed SDKs, chat
front-ends without reasoning controls) can carry the intent in the model
string itself: ``deepseek/deepseek-flash:high`` behaves as if the client
had sent ``reasoning_effort: high``.

Design contract:

- Only the **model segment** carries arguments — the transport-profile
  colon lives in the provider segment (``provider:profile/model``), so
  the two grammars never collide.
- A trailing colon segment is an argument **only when it matches the
  registered vocabulary**; anything else rides verbatim to the provider
  (OpenRouter ``:free``/``:nitro`` variants, Ollama ``model:tag``).
- An explicit request parameter always beats the model-string hint —
  the string is a *default for clients that cannot set it*, never an
  override of clients that can.
- The vocabulary is a registry: new argument words are one entry each.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Vocabulary registry
# ---------------------------------------------------------------------------

_LOCK = threading.RLock()

# word -> request parameter it sets (launch vocabulary: reasoning effort)
_EFFORT_WORDS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

_REGISTRY: Dict[str, Tuple[str, str]] = {word: ("reasoning_effort", word) for word in _EFFORT_WORDS}


def register_model_arg(word: str, param: str, value: str) -> None:
    """Register (or replace) one model-string argument word."""

    word = str(word).strip().lower()
    if not word:
        raise ValueError("model-string argument word must be non-empty")
    with _LOCK:
        _REGISTRY[word] = (str(param), str(value))


def known_model_args() -> Tuple[str, ...]:
    with _LOCK:
        return tuple(sorted(_REGISTRY))


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_model_args(model: str) -> Tuple[str, Dict[str, str]]:
    """Split trailing argument words off a model segment.

    Consumes colon-separated segments from the END while they match the
    registered vocabulary; the first non-vocabulary segment stops
    consumption so mixed ids (``llama3:8b``, ``kimi:free``) ride whole.
    Returns ``(clean_model, args)`` where ``args`` maps request parameter
    to value.
    """

    text = str(model or "")
    if ":" not in text:
        return text, {}
    segments = text.split(":")
    consumed: Dict[str, str] = {}
    index = len(segments) - 1
    while index > 0:
        word = segments[index].strip().lower()
        with _LOCK:
            entry = _REGISTRY.get(word)
        if entry is None or not word:
            break
        param, value = entry
        # First registration of a param wins: the leftmost argument word
        # is the most specific intent.
        consumed.setdefault(param, value)
        index -= 1
    if not consumed or index == len(segments) - 1:
        return text, {}
    clean = ":".join(segments[: index + 1])
    return clean, consumed


def apply_model_args_to_unified(unified_request, args: Dict[str, str]) -> bool:
    """Apply split arguments to a parsed canonical request.

    Returns whether anything changed. Only fills absent controls — an
    explicit client-sent control always wins.
    """

    if not args:
        return False
    changed = False
    effort = args.get("reasoning_effort")
    if effort:
        params = getattr(unified_request, "generation_params", None)
        if params is None:
            params = {}
            unified_request.generation_params = params
        reasoning = params.get("reasoning")
        if reasoning is None:
            params["reasoning"] = {"effort": effort}
            changed = True
        elif isinstance(reasoning, dict) and not reasoning.get("effort"):
            reasoning["effort"] = effort
            changed = True
    return changed
