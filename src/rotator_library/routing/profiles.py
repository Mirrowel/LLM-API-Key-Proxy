"""Provider:profile/model grammar (D13, revised per fix-pass G7).

A model reference may address a specific transport profile of a provider:

``provider/model``            bare name — the profile whose protocol matches
                              the client's request protocol; when nothing
                              speaks it, auto-convert through the default
                              protocol priority list with a terminal
                              warning (never a silent surprise)
``provider:profile/model``    explicit profile — that protocol, converting
                              as asked; fails loudly when unknown

Ambiguity rule: when several profiles speak the client's protocol and none
of them is the default, the proxy cannot guess an endpoint — the error
lists the candidates and suggests declaring a default.

Provider and profile names never contain ``:``; the separator is therefore
unambiguous even though model segments themselves may contain slashes
(``openrouter:fast/openai/gpt-4``). Identity stays provider-level: usage
pools, cooldowns, classifiers, session namespaces, and cache provenance all
key on the bare provider name — the profile only steers transport.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

PROFILE_SEPARATOR = ":"

# Provider segment grammar: the provider registry is case-sensitive, so case
# is PRESERVED (never silently lowercased) and the shape is normalized-or-
# rejected: anything outside this alphabet is an addressing error.
_PROVIDER_NAME_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_-]*"
_PROVIDER_NAME_RE = re.compile(rf"^{_PROVIDER_NAME_PATTERN}$")
_PROVIDER_FIRST_ALLOWED = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
_PROVIDER_REST_ALLOWED = _PROVIDER_FIRST_ALLOWED | {"-", "_"}

# D13 revision: when a bare name's protocol is unavailable, convert through
# this priority order — first protocol the provider actually offers wins.
DEFAULT_PROTOCOL_PRIORITY: Tuple[str, ...] = (
    "openai_chat",
    "responses",
    "anthropic_messages",
    "gemini",
)

lib_logger = logging.getLogger("rotator_library.routing")

# One terminal warning per distinct substitution per process: conversion
# notices must be visible without spamming every retry in a failover chain.
_warned_substitutions: set = set()


class ModelReferenceError(ValueError):
    """Raised for malformed provider:profile/model references."""


@dataclass(frozen=True)
class ModelReference:
    provider: str
    profile: Optional[str]
    model: str

    @property
    def bare(self) -> str:
        """Provider-level model reference (identity form)."""

        return f"{self.provider}/{self.model}" if self.model else self.provider

    @property
    def addressed(self) -> str:
        """Original addressing form including the profile, if any."""

        if self.profile:
            return f"{self.provider}{PROFILE_SEPARATOR}{self.profile}/{self.model}"
        return self.bare


def valid_profile_name(name: str) -> bool:
    """Profile and provider names must be non-empty and separator-free."""

    return bool(name) and PROFILE_SEPARATOR not in name and "/" not in name


def _offending_provider_character(name: str) -> str:
    """Return the first character that makes ``name`` an invalid provider."""

    if not name:
        return ""
    if name[0] not in _PROVIDER_FIRST_ALLOWED:
        return name[0]
    for character in name[1:]:
        if character not in _PROVIDER_REST_ALLOWED:
            return character
    return ""


def validate_provider_name(name: str, *, reference: Optional[str] = None) -> None:
    """Normalize-or-reject the provider segment: reject, never lowercase.

    Provider names are case-sensitive registry keys, so the grammar keeps
    case exactly as addressed. The allowed shape is
    ``[A-Za-z0-9][A-Za-z0-9_-]*``; a violation raises ``ModelReferenceError``
    naming the offending character.
    """

    if _PROVIDER_NAME_RE.match(name):
        return
    offending = _offending_provider_character(name)
    where = f" in model reference {reference!r}" if reference is not None else ""
    why = (
        f"invalid character {offending!r}"
        if offending
        else "empty name"
    )
    raise ModelReferenceError(
        f"Invalid provider name {name!r}{where}: {why} "
        f"(provider names must match {_PROVIDER_NAME_PATTERN})"
    )


def parse_model_reference(model: str) -> ModelReference:
    """Parse a model reference into provider, optional profile, and model.

    The grammar applies only to the provider segment (before the first
    ``/``): ``provider:profile/model``. Model segments may contain anything,
    including colons (OpenRouter ``:free`` variants, Ollama ``model:tag``).

    Whitespace is stripped per segment around the provider and profile only;
    leading/trailing whitespace inside the model segment is part of the
    model id and is preserved. Provider names are validated (not lowercased):
    see ``validate_provider_name``.

    Raises ``ModelReferenceError`` for empty or malformed references
    (``:profile/model`` without a provider, ``provider:/model`` without a
    profile name, separator-only segments).
    """

    text = model or ""
    if not text.strip():
        raise ModelReferenceError("Empty model reference")
    provider_segment, sep, remainder = text.partition("/")
    if not sep:
        remainder = ""
    profile: Optional[str] = None
    provider = provider_segment.strip()
    if PROFILE_SEPARATOR in provider_segment:
        raw_provider, _, profile_segment = provider_segment.partition(PROFILE_SEPARATOR)
        provider = raw_provider.strip()
        profile = profile_segment.strip()
        if not provider:
            raise ModelReferenceError(f"Model reference missing provider: {model!r}")
        if not profile or not valid_profile_name(profile):
            raise ModelReferenceError(f"Invalid profile name in model reference: {model!r}")
        if not remainder:
            raise ModelReferenceError(f"Model reference missing model after profile: {model!r}")
    validate_provider_name(provider, reference=model)
    if not remainder:
        # A bare provider with no model is not a usable reference.
        raise ModelReferenceError(f"Model reference missing model: {model!r}")
    return ModelReference(provider=provider, profile=profile, model=remainder)


def resolve_profile(
    *,
    declared_profiles: Optional[dict],
    default_profile: Optional[str],
    protocol_name: Optional[str],
    client_protocol: Optional[str],
    requested_profile: Optional[str],
    provider: str,
) -> Optional[str]:
    """Resolve which transport profile serves a request.

    Explicit ``provider:profile`` wins and must exist. Bare names pick the
    default profile, or — when the client speaks a different protocol than
    the default — the unique profile matching the client protocol; no match
    is a fast-path error (the client asked for an endpoint that does not
    exist), never a silent conversion.
    """

    if declared_profiles is None:
        # Single-protocol provider: no profile machinery.
        if requested_profile:
            raise ModelReferenceError(
                f"Provider {provider} has no profiles; unknown profile {requested_profile!r}"
            )
        return None
    names = {str(name) for name in declared_profiles}
    if requested_profile:
        if requested_profile not in names:
            raise ModelReferenceError(
                f"Provider {provider} has no profile {requested_profile!r}; known: {sorted(names)}"
            )
        return requested_profile
    # Explicit addressing never depends on default hygiene: validate the
    # default only for bare-name resolution.
    if default_profile and default_profile not in names:
        raise ModelReferenceError(
            f"Provider {provider} declares default profile {default_profile!r} "
            f"but no such profile exists; known: {sorted(names)}"
        )
    if default_profile and default_profile in names:
        default_protocol = _profile_protocol(declared_profiles, default_profile, protocol_name)
        if not client_protocol or default_protocol == client_protocol:
            return default_profile
    # Bare name with a non-default client protocol: the unique matching
    # profile, else convert through the priority list with a warning.
    matches = [
        str(name)
        for name in names
        if _profile_protocol(declared_profiles, str(name), protocol_name) == client_protocol
    ]
    if len(matches) == 1:
        return matches[0]
    known = {str(name): _profile_protocol(declared_profiles, str(name), protocol_name) for name in names}
    if len(matches) > 1:
        # Ambiguity about WHICH endpoint speaks the protocol is not a
        # conversion case: the operator must pick, or declare a default.
        raise ModelReferenceError(
            f"Provider {provider} has multiple profiles for client protocol "
            f"{client_protocol!r}: {sorted(matches)} — request it as "
            f"provider:profile/model or declare one of them the default profile"
        )
    for candidate_protocol in DEFAULT_PROTOCOL_PRIORITY:
        if candidate_protocol == client_protocol:
            continue
        offering = sorted(
            str(name)
            for name in names
            if _profile_protocol(declared_profiles, str(name), protocol_name) == candidate_protocol
        )
        if not offering:
            continue
        chosen = default_profile if default_profile in offering else offering[0]
        key = (provider, client_protocol, chosen, candidate_protocol)
        if key not in _warned_substitutions:
            _warned_substitutions.add(key)
            lib_logger.warning(
                "Provider %s has no endpoint for client protocol %r — serving "
                "converted via profile %r (%s) per the default protocol "
                "priority; request %s:%s/model to pin it explicitly",
                provider,
                client_protocol,
                chosen,
                candidate_protocol,
                provider,
                chosen,
            )
        return chosen
    raise ModelReferenceError(
        f"Provider {provider} has no endpoint for client protocol {client_protocol!r} "
        f"(known profiles: {known}); request it as provider:profile/model"
    )


def _profile_protocol(declared: dict, name: str, fallback: Optional[str]) -> Optional[str]:
    entry = declared.get(name)
    if isinstance(entry, dict):
        protocol = entry.get("protocol") or entry.get("protocol_name")
        return str(protocol) if protocol else fallback
    return fallback
