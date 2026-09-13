# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Protocol-owned defaults (G8 envelope v2): the SDK surface providers
pick from instead of wiring by hand.

Three tables live here, all keyed by protocol (family names resolve their
variants — bare ``responses`` means its stateless default):

- ``PROTOCOL_DEFAULTS`` — endpoint routes per operation, the conventional
  auth style, and the listing descriptor (how models are discovered on
  that protocol). A provider's ``speaks`` entry inherits all of it and
  overrides only what differs.
- ``FIELD_LOCATIONS`` — where protocol-state fields (reasoning,
  signatures, …) live on each protocol's wire shapes: extraction from
  responses, extraction from stream events, injection into requests, and
  occurrence correlation. Field-addressed cache rules resolve through
  this table so one rule serves every face a provider speaks.
- ``resolve_speaks_entry`` / ``validate_speaks`` — the ``speaks`` tuple
  grammar: ``"protocol"`` (name = protocol, all defaults),
  ``(protocol, overrides)`` (same name, diffs only), or
  ``(name, protocol, overrides)`` (explicit identity — the
  same-protocol-twice case, e.g. a subscription face).

Scope honesty: the location table fully covers the path-addressable
shapes (openai_chat family, ollama). The structural protocols
(anthropic thinking blocks, gemini thought parts, responses reasoning
items) keep their declared explicit paths — the registry slots exist and
gain structural resolvers when those protocols need field rules of
their own.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

# ---------------------------------------------------------------------------
# Protocol defaults: endpoints, auth, listing
# ---------------------------------------------------------------------------

# Auth styles: the conventional credential header family for a protocol.
# Providers override per face when reality differs.
_AUTH_BEARER = {"auth_mode": "bearer"}
_AUTH_XAPIKEY = {"auth_mode": "x-api-key"}
_AUTH_GOOG = {"auth_mode": "x-goog-api-key"}
_AUTH_NONE = {"auth_mode": "none"}

# Listing descriptors: how each protocol discovers models.
#   path:    the listing route on the transport base
#   shape:   response shape — data_id (openai family), models_name
#            (gemini/ollama), none (no listing on this protocol)
#   strip:   prefix stripped from listed ids (gemini "models/")
_LISTING_OPENAI = {"path": "/models", "shape": "data_id"}
_LISTING_GEMINI = {"path": "/models", "shape": "models_name", "strip": "models/"}
_LISTING_OLLAMA = {"path": "/api/tags", "shape": "models_name"}
_LISTING_NONE = {"path": None, "shape": None}

PROTOCOL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "openai_chat": {
        "family": "openai_chat",
        "endpoint_paths": {
            "chat": "/chat/completions",
            "models": "/models",
        },
        "auth": _AUTH_BEARER,
        "listing": _LISTING_OPENAI,
    },
    "responses": {
        "family": "responses",
        "endpoint_paths": {
            "responses": "/responses",
        },
        "auth": _AUTH_BEARER,
        # The responses family lists through the openai-style route; a
        # chat+responses provider resolves its listing face via the
        # protocol priority list (openai_chat first).
        "listing": _LISTING_OPENAI,
    },
    "responses_stateful": {
        "family": "responses",
        "endpoint_paths": {
            "responses": "/responses",
        },
        "auth": _AUTH_BEARER,
        "listing": _LISTING_OPENAI,
    },
    "responses_websocket": {
        "family": "responses",
        "endpoint_paths": {
            "responses": "/responses",
        },
        "auth": _AUTH_BEARER,
        "listing": _LISTING_NONE,
    },
    "anthropic_messages": {
        "family": "anthropic_messages",
        "endpoint_paths": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "models": "/v1/models",
        },
        "auth": _AUTH_XAPIKEY,
        "listing": _LISTING_OPENAI,
    },
    "gemini": {
        "family": "gemini",
        "endpoint_paths": {
            "generate": "/models/{model}:generateContent",
            "stream_generate": "/models/{model}:streamGenerateContent?alt=sse",
            "count_tokens": "/models/{model}:countTokens",
            "models": "/models",
        },
        "auth": _AUTH_GOOG,
        "listing": _LISTING_GEMINI,
    },
    "ollama": {
        "family": "ollama",
        "endpoint_paths": {
            "ollama_chat": "/api/chat",
            "ollama_generate": "/api/generate",
            "embeddings": "/api/embed",
            "models": "/api/tags",
        },
        "auth": _AUTH_NONE,
        "listing": _LISTING_OLLAMA,
    },
}

# The priority list used when a face must be chosen without an explicit
# declaration (model listing on a multi-face provider, zero-match
# conversion). Mirrors the routing priority list.
PROTOCOL_PRIORITY: Tuple[str, ...] = (
    "openai_chat",
    "responses",
    "anthropic_messages",
    "gemini",
)


def default_endpoint_paths(protocol: str) -> Dict[str, str]:
    entry = PROTOCOL_DEFAULTS.get(str(protocol))
    return dict(entry["endpoint_paths"]) if entry else {}


def default_auth_mode(protocol: str) -> Optional[str]:
    entry = PROTOCOL_DEFAULTS.get(str(protocol))
    return entry["auth"].get("auth_mode") if entry else None


def listing_descriptor(protocol: str) -> Optional[Dict[str, Any]]:
    entry = PROTOCOL_DEFAULTS.get(str(protocol))
    listing = entry.get("listing") if entry else None
    if not listing or not listing.get("path"):
        return None
    return dict(listing)


def resolve_listing_protocol(available: Tuple[str, ...]) -> Optional[str]:
    """Pick the listing face: the provider's explicit choice wins; else the
    first protocol in priority order that both is available and has a
    listing descriptor."""

    for candidate in PROTOCOL_PRIORITY:
        if candidate in available and listing_descriptor(candidate):
            return candidate
    for candidate in available:
        if listing_descriptor(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Field locations: protocol-state fields across wire shapes
# ---------------------------------------------------------------------------

# Structural protocols (anthropic thinking blocks, gemini thought parts,
# responses reasoning items) are not path-addressable; their slots are
# reserved and rules for them keep explicit paths until structural
# resolvers exist.
FIELD_LOCATIONS: Dict[str, Dict[str, Dict[str, str]]] = {
    "reasoning": {
        "openai_chat": {
            "response_path": "choices.*.message.reasoning_content",
            "stream_path": "raw.choices.*.delta.reasoning_content",
            "inject_path": "messages.*.reasoning_content",
            "tool_call_id_path": "tool_calls.*.id",
        },
        "ollama": {
            "response_path": "message.thinking",
            "stream_path": "raw.message.thinking",
            "inject_path": "messages.*.thinking",
            "tool_call_id_path": "tool_calls.*.id",
        },
    },
    "signature": {
        "openai_chat": {
            "response_path": "choices.*.message.tool_calls.*.extra_content.google.thought_signature",
            "stream_path": "raw.choices.*.delta.tool_calls.*.extra_content.google.thought_signature",
            "inject_path": "messages.*.tool_calls.*.extra_content.google.thought_signature",
            "tool_call_id_path": "tool_calls.*.id",
        },
    },
}


def field_locations(field: str, protocol: str) -> Optional[Dict[str, str]]:
    family = PROTOCOL_DEFAULTS.get(str(protocol), {}).get("family", protocol)
    return FIELD_LOCATIONS.get(str(field), {}).get(family)


# ---------------------------------------------------------------------------
# speaks grammar
# ---------------------------------------------------------------------------

def _normalize_protocol_name(name: str) -> str:
    text = str(name).strip()
    from .registry import resolve_protocol_name

    try:
        return resolve_protocol_name(text)
    except KeyError:
        return text


def resolve_speaks_entry(entry: Any) -> Dict[str, Any]:
    """Resolve one ``speaks`` entry to ``{name, protocol, overrides}``.

    Forms: ``"protocol"`` | ``(protocol, overrides)`` |
    ``(name, protocol, overrides)``. The protocol must exist in the
    registry (variants resolve to their family defaults unless they
    declare their own).
    """

    if isinstance(entry, str):
        protocol = _normalize_protocol_name(entry)
        return {"name": protocol, "protocol": protocol, "overrides": {}}
    if isinstance(entry, (tuple, list)):
        if len(entry) == 2:
            protocol, overrides = entry
            protocol = _normalize_protocol_name(protocol)
            return {"name": protocol, "protocol": protocol, "overrides": dict(overrides or {})}
        if len(entry) == 3:
            name, protocol, overrides = entry
            protocol = _normalize_protocol_name(protocol)
            return {"name": str(name).strip(), "protocol": protocol, "overrides": dict(overrides or {})}
    raise ValueError(
        f"Invalid speaks entry {entry!r}: expected 'protocol', (protocol, overrides), "
        "or (name, protocol, overrides)"
    )


def resolve_speaks(speaks) -> Dict[str, Dict[str, Any]]:
    """Resolve a full ``speaks`` tuple into a profile table keyed by name.

    First entry is the provider's default face. Duplicate profile names
    are rejected; unknown protocols are rejected with the legal names.
    """

    profiles: Dict[str, Dict[str, Any]] = {}
    order: list[str] = []
    for raw in speaks or ():
        resolved = resolve_speaks_entry(raw)
        name = resolved["name"]
        if not name:
            raise ValueError("speaks profile name must be non-empty")
        if name in profiles:
            raise ValueError(f"Duplicate speaks profile name {name!r}")
        defaults = PROTOCOL_DEFAULTS.get(resolved["protocol"], {})
        profiles[name] = {
            "protocol": resolved["protocol"],
            "endpoint_paths": {**defaults.get("endpoint_paths", {}), **resolved["overrides"].get("endpoint_paths", {})},
            "auth_mode": resolved["overrides"].get("auth_mode") or defaults.get("auth", {}).get("auth_mode"),
            "auth_header_name": resolved["overrides"].get("auth_header_name"),
            "base": resolved["overrides"].get("base"),
        }
        order.append(name)
    if order:
        profiles["__default__"] = profiles[order[0]]
    return profiles


def validate_speaks(speaks) -> Tuple[str, ...]:
    """Validate at import/registration time; returns legal names on failure."""

    from .registry import list_protocols

    legal = tuple(sorted(list_protocols()))
    for raw in speaks or ():
        protocol = resolve_speaks_entry(raw)["protocol"]
        if protocol not in legal:
            raise ValueError(
                f"Unknown protocol {protocol!r} in speaks declaration; legal names: {', '.join(legal)}"
            )
    return legal
