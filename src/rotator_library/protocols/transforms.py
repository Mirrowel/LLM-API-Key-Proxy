# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Cross-format transforms for cached provider state (W13/D14).

Cached values are stored in the format they were captured in (extraction
always runs on the finalized, assembled response). When the restore target
speaks a different wire format, a registered transform converts the stored
shape into the target's native representation — plain reasoning text
captured from a chat-completions response becomes an Anthropic thinking
block on injection, and vice versa. Transforms are registered by name and
referenced from field-cache rule metadata (``{"transform": "<name>"}``).
"""

from __future__ import annotations

from typing import Any, Callable, Dict

TransformFn = Callable[[Any], Any]

_REGISTRY: Dict[str, TransformFn] = {}


def register_transform(name: str) -> Callable[[TransformFn], TransformFn]:
    def decorator(fn: TransformFn) -> TransformFn:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_transform(name: str) -> TransformFn:
    fn = _REGISTRY.get(name)
    if fn is None:
        raise KeyError(f"Unknown field-cache transform: {name!r}; known: {sorted(_REGISTRY)}")
    return fn


def apply_transform(name: str, value: Any) -> Any:
    """Apply a named transform; transform failures raise (misconfigured
    rules should fail loudly at first use, not silently skip restores)."""

    return get_transform(name)(value)


# --- Seeded transforms (chat <-> anthropic reasoning shapes) ---


@register_transform("chat_reasoning_to_anthropic_thinking")
def _chat_reasoning_to_anthropic_thinking(value: Any) -> Any:
    """Chat ``reasoning_content`` string -> Anthropic thinking block."""

    if isinstance(value, dict) and "thinking" in value:
        return value
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        text = value["text"]
    else:
        text = str(value) if value is not None else ""
    if not text:
        return None
    return {"type": "thinking", "thinking": text, "signature": None}


@register_transform("anthropic_thinking_to_chat_reasoning")
def _anthropic_thinking_to_chat_reasoning(value: Any) -> Any:
    """Anthropic thinking block -> chat ``reasoning_content`` string."""

    if isinstance(value, dict):
        return value.get("thinking") or value.get("text") or ""
    return str(value or "")


@register_transform("identity")
def _identity(value: Any) -> Any:
    return value
