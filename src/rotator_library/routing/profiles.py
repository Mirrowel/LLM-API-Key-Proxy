"""Provider:profile/model grammar (D13).

A model reference may address a specific transport profile of a provider:

``provider/model``            bare name — the profile whose protocol matches
                              the client's request protocol, error when no
                              profile speaks it (never silent conversion)
``provider:profile/model``    explicit profile — that protocol, converting
                              as asked

Provider and profile names never contain ``:``; the separator is therefore
unambiguous even though model segments themselves may contain slashes
(``openrouter:fast/openai/gpt-4``). Identity stays provider-level: usage
pools, cooldowns, classifiers, session namespaces, and cache provenance all
key on the bare provider name — the profile only steers transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

PROFILE_SEPARATOR = ":"


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


def parse_model_reference(model: str) -> ModelReference:
    """Parse a model reference into provider, optional profile, and model.

    The grammar applies only to the provider segment (before the first
    ``/``): ``provider:profile/model``. Model segments may contain anything,
    including colons (OpenRouter ``:free`` variants, Ollama ``model:tag``).

    Raises ``ModelReferenceError`` for empty or malformed references
    (``:profile/model`` without a provider, ``provider:/model`` without a
    profile name, separator-only segments).
    """

    text = (model or "").strip()
    if not text:
        raise ModelReferenceError("Empty model reference")
    provider_segment, sep, remainder = text.partition("/")
    if not sep:
        remainder = ""
    profile: Optional[str] = None
    provider = provider_segment
    if PROFILE_SEPARATOR in provider_segment:
        provider, _, profile_segment = provider_segment.partition(PROFILE_SEPARATOR)
        profile = profile_segment.strip()
        if not provider:
            raise ModelReferenceError(f"Model reference missing provider: {model!r}")
        if not profile or not valid_profile_name(profile):
            raise ModelReferenceError(f"Invalid profile name in model reference: {model!r}")
        if not remainder:
            raise ModelReferenceError(f"Model reference missing model after profile: {model!r}")
    if not provider or not valid_profile_name(provider):
        raise ModelReferenceError(f"Invalid provider name in model reference: {model!r}")
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
    if default_profile and default_profile not in names:
        raise ModelReferenceError(
            f"Provider {provider} declares default profile {default_profile!r} "
            f"but no such profile exists; known: {sorted(names)}"
        )
    if requested_profile:
        if requested_profile not in names:
            raise ModelReferenceError(
                f"Provider {provider} has no profile {requested_profile!r}; known: {sorted(names)}"
            )
        return requested_profile
    if default_profile and default_profile in names:
        default_protocol = _profile_protocol(declared_profiles, default_profile, protocol_name)
        if not client_protocol or default_protocol == client_protocol:
            return default_profile
    # Bare name with a non-default client protocol: the unique matching
    # profile, else error (fast-path-or-error).
    matches = [
        str(name)
        for name in names
        if _profile_protocol(declared_profiles, str(name), protocol_name) == client_protocol
    ]
    if len(matches) == 1:
        return matches[0]
    known = {str(name): _profile_protocol(declared_profiles, str(name), protocol_name) for name in names}
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


def split_profile_from_provider(provider: str) -> Tuple[str, Optional[str]]:
    """Split a routing-level ``provider:profile`` identity into parts."""

    if PROFILE_SEPARATOR in provider:
        provider_name, _, profile = provider.partition(PROFILE_SEPARATOR)
        return provider_name, (profile or None)
    return provider, None
