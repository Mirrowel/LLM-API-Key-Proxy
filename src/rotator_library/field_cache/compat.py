# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Compatibility classes for cached provider state (D12).

Bound fields (encrypted reasoning, thinking signatures) are provider-locked
opaque state: they restore only to the exact provider+model that produced
them — cross-provider transport would be rejected upstream anyway. Portable
fields (plaintext reasoning) inherit within declared compatibility groups
(``model:<name>`` identity, curated global groups, user config groups);
unknown pairs are denied by default. The operator is the trust boundary:
explicit group configuration is never second-guessed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional

COMPATIBILITY_ENV_VAR = "FIELD_CACHE_COMPAT_GROUPS"


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"


def parse_model_ref(text: str) -> Optional[ModelRef]:
    if not isinstance(text, str) or "/" not in text:
        return None
    provider, _, model = text.partition("/")
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        return None
    return ModelRef(provider=provider, model=model)


class CompatibilityRegistry:
    """Group membership and inheritance policy for cached fields."""

    def __init__(self, groups: Optional[dict[str, list[str]]] = None) -> None:
        # group name -> ordered member "provider/model" refs
        self._groups: dict[str, list[ModelRef]] = {}
        # member key -> group names (a model may live in several groups)
        self._membership: dict[str, set[str]] = {}
        for name, members in (groups or {}).items():
            self.add_group(name, members)

    def add_group(self, name: str, members: Iterable[str]) -> None:
        refs: list[ModelRef] = []
        for member in members:
            ref = parse_model_ref(str(member))
            if ref is None:
                raise ValueError(f"Compatibility group {name!r} member is not provider/model: {member!r}")
            refs.append(ref)
        self._groups[str(name)] = refs
        for ref in refs:
            self._membership.setdefault(ref.key, set()).add(str(name))

    def groups_for(self, ref: ModelRef) -> set[str]:
        """All groups the model belongs to, including its identity group."""

        groups = set(self._membership.get(ref.key, set()))
        groups.add(f"model:{ref.model}")
        return groups

    def can_inherit(
        self,
        source: ModelRef,
        target: ModelRef,
        *,
        field_class: str,
    ) -> bool:
        """Whether cached state from ``source`` may restore into ``target``.

        ``field_class``: ``bound`` (provider-locked opaque state) or
        ``portable`` (plaintext, group-inheritable).
        """

        if source.key == target.key:
            return True
        if field_class == "bound":
            # Cross-provider transport of opaque provider state would be
            # rejected upstream; identity match only (D8/D12).
            return False
        if field_class != "portable":
            return False
        return bool(self.groups_for(source) & self.groups_for(target))

    def siblings(self, ref: ModelRef, *, field_class: str) -> list[ModelRef]:
        """Inheritable sibling models for a reference (excluding itself)."""

        if field_class != "portable":
            return []
        groups = self.groups_for(ref)
        seen: dict[str, ModelRef] = {}
        for group in groups:
            declared = self._groups.get(group)
            if declared:
                for member in declared:
                    seen.setdefault(member.key, member)
        identity_group = f"model:{ref.model}"
        if identity_group in groups:
            # Identity group: any provider exposing the same model name.
            for member in self._declared_refs():
                if member.model == ref.model:
                    seen.setdefault(member.key, member)
        seen.pop(ref.key, None)
        return list(seen.values())

    def _declared_refs(self) -> list[ModelRef]:
        refs: list[ModelRef] = []
        for members in self._groups.values():
            refs.extend(members)
        return refs


_REGISTRY: Optional[CompatibilityRegistry] = None


def reset_compatibility_registry() -> None:
    global _REGISTRY
    _REGISTRY = None


def get_compatibility_registry() -> CompatibilityRegistry:
    """Process-wide registry: env-configured groups (single-user proxy —
    the operator configures groups explicitly; there is no multi-user
    split yet)."""

    global _REGISTRY
    if _REGISTRY is None:
        groups: dict[str, list[str]] = {}
        raw = os.getenv(COMPATIBILITY_ENV_VAR, "")
        if raw.strip():
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError(f"{COMPATIBILITY_ENV_VAR} must be a JSON object of group -> [provider/model]")
            groups = {str(name): list(members) for name, members in parsed.items()}
        _REGISTRY = CompatibilityRegistry(groups)
    return _REGISTRY
