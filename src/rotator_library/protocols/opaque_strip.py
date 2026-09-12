"""Foreign opaque-state stripping on the raw wire (G3).

Per-provider opaque state (thinking signatures, thought signatures,
encrypted reasoning) rides the same-protocol raw fast path verbatim —
correct when it returns to the provider that minted it, a guaranteed
rejection when the executing provider switched (failover or a conversation
that moved). This module strips those carriers from RAW wire payloads
in place, per protocol, and reports exactly what was removed so the
executor can trace the edit as an overlay (the operator's rule: the fast
path is conditional on no UNDISCLOSED modification — a strip is always a
disclosed edit).

Isolation philosophy (operator ruling, G3 D4): the cache key ADDRESSES
(loosely, provider+model by default); the compatibility class ISOLATES
(bound fields restore only to the exact provider+model that produced
them). Stripping never touches the field cache — stripped state stays
stored under its origin keys, so a conversation returning home restores
it for free.
"""

from __future__ import annotations

from typing import Any

# Opaque carriers per wire protocol. Keys are raw payload field names —
# these walk the UNPARSED client wire, so they speak protocol dialect,
# not canonical vocabulary.
_ANTHROPIC_OPAQUE_BLOCKS = frozenset({"thinking", "redacted_thinking"})
_GEMINI_SIGNATURE_KEYS = frozenset({"thoughtSignature", "thought_signature"})


def payload_carries_opaque_state(payload: Any, protocol_name: str) -> bool:
    """Cheap check: does this raw payload carry any opaque carrier?"""

    return strip_foreign_opaque_state(payload, protocol_name, mutate=False) is not None


def strip_foreign_opaque_state(payload: Any, protocol_name: str, *, mutate: bool = True) -> list[str] | None:
    """Strip foreign opaque state from a raw wire payload.

    Returns the list of removed field descriptors (protocol-shaped
    strings for the trace overlay), or ``None`` when nothing was carried.
    With ``mutate=False`` this only detects (for the reactive retry path).
    """

    if not isinstance(payload, dict):
        return None
    if protocol_name == "anthropic_messages":
        return _strip_anthropic(payload, mutate=mutate)
    if protocol_name == "gemini":
        return _strip_gemini(payload, mutate=mutate)
    if protocol_name == "openai_chat":
        return _strip_chat(payload, mutate=mutate)
    if protocol_name == "responses":
        return _strip_responses(payload, mutate=mutate)
    return None


def _strip_anthropic(payload: dict, *, mutate: bool) -> list[str] | None:
    stripped: list[str] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for position, message in enumerate(messages):
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            kept = []
            for block in content:
                if isinstance(block, dict) and block.get("type") in _ANTHROPIC_OPAQUE_BLOCKS:
                    # Unsigned thinking is invalid toward Anthropic: the
                    # whole block drops, never an empty thinking husk.
                    stripped.append(f"messages[{position}].{block.get('type')}")
                    continue
                kept.append(block)
            if mutate:
                message["content"] = kept
    return stripped or None


def _strip_gemini(payload: dict, *, mutate: bool) -> list[str] | None:
    stripped: list[str] = []
    contents = payload.get("contents")
    if isinstance(contents, list):
        for c_position, content in enumerate(contents):
            parts = content.get("parts") if isinstance(content, dict) else None
            if not isinstance(parts, list):
                continue
            for p_position, part in enumerate(parts):
                if not isinstance(part, dict):
                    continue
                for key in _GEMINI_SIGNATURE_KEYS:
                    if key in part:
                        # The any-part rule means the signature key can ride
                        # any part; the part itself (text/functionCall/...) is
                        # legal without it.
                        stripped.append(f"contents[{c_position}].parts[{p_position}].{key}")
                        if mutate:
                            part.pop(key, None)
    return stripped or None


def _strip_chat(payload: dict, *, mutate: bool) -> list[str] | None:
    stripped: list[str] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for position, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            extra_content = message.get("extra_content")
            if isinstance(extra_content, dict) and isinstance(extra_content.get("google"), dict):
                google = extra_content["google"]
                if "thought_signature" in google:
                    # Bound vendor-keyed signature (the compat-surface
                    # spelling); plaintext reasoning_content is PORTABLE and
                    # deliberately never stripped here.
                    stripped.append(f"messages[{position}].extra_content.google.thought_signature")
                    if mutate:
                        google.pop("thought_signature", None)
                        if not google:
                            extra_content.pop("google", None)
                            if not extra_content:
                                message.pop("extra_content", None)
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                call_extra = call.get("extra_content")
                if isinstance(call_extra, dict) and isinstance(call_extra.get("google"), dict):
                    google = call_extra["google"]
                    if "thought_signature" in google:
                        stripped.append(f"messages[{position}].tool_calls[{call.get('id') or call.get('index')}].extra_content.google.thought_signature")
                        if mutate:
                            google.pop("thought_signature", None)
                            if not google:
                                call_extra.pop("google", None)
                                if not call_extra:
                                    call.pop("extra_content", None)
    return stripped or None


def _strip_responses(payload: dict, *, mutate: bool) -> list[str] | None:
    stripped: list[str] = []
    items = payload.get("input")
    if isinstance(items, list):
        for position, item in enumerate(items):
            if isinstance(item, dict) and item.get("type") == "reasoning" and "encrypted_content" in item:
                # The reasoning item itself (summary text) is portable; the
                # encrypted blob is provider-bound and drops.
                stripped.append(f"input[{position}].reasoning.encrypted_content")
                if mutate:
                    item.pop("encrypted_content", None)
    return stripped or None


def signature_rejection_message(error: BaseException) -> bool:
    """Does this error look like a provider rejecting foreign signatures?

    The reactive safety net (strip-and-retry once, same target — the
    plexus pattern) fires only on errors that NAME the opaque field
    family. Narrow by construction: phrase-level matching on the
    message, never bare substrings.
    """

    message = str(error).lower()
    if "signature" not in message:
        return False
    markers = (
        "thinking",
        "thought",
        "encrypted",
        "reasoning",
    )
    return any(marker in message for marker in markers)
