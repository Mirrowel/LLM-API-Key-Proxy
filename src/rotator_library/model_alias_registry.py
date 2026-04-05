# SPDX-License-Identifier: LGPL-3.0-only

"""
Model Alias Registry for cross-provider model routing.

Parses MODEL_ALIAS_* environment variables to map canonical model names
to provider-specific model names. Enables a single request to fail over
across multiple providers transparently.

Env config format:
    MODEL_ALIAS_<CANONICAL>=provider1:model1,provider2:model2[|retry_mode]

Examples:
    MODEL_ALIAS_DEEPSEEK_V3="chutes:deepseek-v3,nanogpt:deepseek-chat"
    MODEL_ALIAS_GLM_5="chutes:glm-5,nanogpt:glm-5:thinking|exhaust"
"""

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

lib_logger = logging.getLogger("rotator_library")

DEFAULT_RETRY_MODE = "round_robin"
VALID_RETRY_MODES = {"round_robin", "exhaust"}


@dataclass
class AliasTarget:
    """A single provider+model target within an alias."""

    provider: str  # e.g., "chutes"
    model_name: str  # e.g., "deepseek-v3" (provider-specific name)

    @property
    def full_model(self) -> str:
        """Return provider/model format for the existing executor."""
        return f"{self.provider}/{self.model_name}"


@dataclass
class ModelAlias:
    """A canonical model alias with its provider targets and retry config."""

    canonical: str  # e.g., "deepseek-v3"
    targets: List[AliasTarget]
    retry_mode: str = DEFAULT_RETRY_MODE  # "round_robin" or "exhaust"


class ModelAliasRegistry:
    """
    Registry that maps canonical model names to cross-provider targets.

    Parses MODEL_ALIAS_* environment variables at construction time.
    Thread-safe for reads after initialization.
    """

    def __init__(self) -> None:
        self._aliases: Dict[str, ModelAlias] = {}
        # Lookup table: maps normalized names to canonical keys
        self._lookup: Dict[str, str] = {}
        self._load_from_env()

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize a model name for lookup (lowercase, periods→hyphens)."""
        return name.lower().replace(".", "-")

    def _register_alias(self, canonical: str, alias: ModelAlias) -> None:
        """Register an alias with lookup variants."""
        self._aliases[canonical] = alias
        # Register the canonical name itself
        self._lookup[self._normalize(canonical)] = canonical
        # Also register with periods restored (kimi-k2-5 → kimi-k2.5)
        # so clients can use either form
        self._lookup[canonical] = canonical

    def _load_from_env(self) -> None:
        """Load all MODEL_ALIAS_* environment variables."""
        for key, value in os.environ.items():
            if not key.startswith("MODEL_ALIAS_"):
                continue

            # Extract canonical name: MODEL_ALIAS_DEEPSEEK_V3 → deepseek-v3
            canonical = key[len("MODEL_ALIAS_"):].lower().replace("_", "-")

            try:
                alias = self._parse_alias_value(canonical, value)
                if alias and alias.targets:
                    self._register_alias(canonical, alias)
                    target_summary = ", ".join(
                        f"{t.provider}:{t.model_name}" for t in alias.targets
                    )
                    lib_logger.info(
                        f"Registered model alias: {canonical} → [{target_summary}] "
                        f"(retry: {alias.retry_mode})"
                    )
            except Exception as e:
                lib_logger.warning(
                    f"Failed to parse {key}: {e}"
                )

    def _parse_alias_value(self, canonical: str, value: str) -> Optional[ModelAlias]:
        """
        Parse an alias env value.

        Format: provider1:model1,provider2:model2[|retry_mode]

        The retry mode suffix is optional, separated by |.
        Model names can contain colons (e.g., glm-5:thinking).
        """
        value = value.strip()
        if not value:
            return None

        # Split off retry mode suffix (last | in the string)
        retry_mode = DEFAULT_RETRY_MODE
        if "|" in value:
            parts = value.rsplit("|", 1)
            candidate_mode = parts[1].strip().lower()
            if candidate_mode in VALID_RETRY_MODES:
                retry_mode = candidate_mode
                value = parts[0].strip()
            # If not a valid mode, treat | as part of the value

        # Parse comma-separated provider:model pairs
        targets: List[AliasTarget] = []
        for entry in value.split(","):
            entry = entry.strip()
            if not entry:
                continue

            # Split on first colon only — model name can contain colons
            if ":" not in entry:
                lib_logger.warning(
                    f"Invalid alias target '{entry}' for '{canonical}': "
                    f"expected 'provider:model' format"
                )
                continue

            provider, model_name = entry.split(":", 1)
            provider = provider.strip().lower()
            model_name = model_name.strip()

            if not provider or not model_name:
                lib_logger.warning(
                    f"Invalid alias target '{entry}' for '{canonical}': "
                    f"empty provider or model name"
                )
                continue

            targets.append(AliasTarget(provider=provider, model_name=model_name))

        if not targets:
            return None

        return ModelAlias(
            canonical=canonical,
            targets=targets,
            retry_mode=retry_mode,
        )

    def _resolve_key(self, model: str) -> Optional[str]:
        """Resolve a model name to its canonical key via lookup table."""
        # Try exact match first, then normalized
        key = self._lookup.get(model.lower())
        if key:
            return key
        return self._lookup.get(self._normalize(model))

    def resolve(self, model: str) -> Optional[List[AliasTarget]]:
        """
        Resolve a model name to its provider targets.

        Handles period/hyphen variations (e.g., kimi-k2.5 and kimi-k2-5
        both resolve to the same alias).

        Args:
            model: Model name (without provider prefix)

        Returns:
            List of AliasTarget in priority order, or None if not an alias
        """
        key = self._resolve_key(model)
        if key:
            alias = self._aliases.get(key)
            if alias:
                return list(alias.targets)
        return None

    def get_retry_mode(self, model: str) -> str:
        """
        Get the retry mode for a canonical model.

        Args:
            model: Canonical model name

        Returns:
            "round_robin" or "exhaust"
        """
        key = self._resolve_key(model)
        if key:
            alias = self._aliases.get(key)
            if alias:
                return alias.retry_mode
        return DEFAULT_RETRY_MODE

    def is_alias(self, model: str) -> bool:
        """Check if a model name is a registered alias."""
        return self._resolve_key(model) is not None

    def get_canonical_models(self) -> List[str]:
        """
        Get all registered canonical model names.

        Used to add alias entries to the /v1/models endpoint.
        """
        return list(self._aliases.keys())

    def get_all_aliases(self) -> Dict[str, ModelAlias]:
        """Get the full alias registry (for debugging/admin endpoints)."""
        return dict(self._aliases)
