# SPDX-License-Identifier: LGPL-3.0-only

"""
Model Fallback Registry for per-model provider spillover.

When a prefixed request (e.g., ``chutes/gemma-4-31b-it``) exhausts all
credentials on the primary provider, the fallback registry provides an
ordered list of alternative providers to try.

Unlike MODEL_ALIAS (which intercepts *unprefixed* requests at entry),
MODEL_FALLBACK only activates *after* the primary provider fails —
preserving the user's preferred provider as the strong first choice.

Env config format:
    MODEL_FALLBACK_<MODEL>=provider1[:model1],provider2[:model2][|retry_mode]

When only a provider name is given (no ``:model``), the original model
name from the failed request is used.

Examples:
    MODEL_FALLBACK_GEMMA_4_31B_IT="google,nvidia_nim,ollama_cloud"
    MODEL_FALLBACK_GEMMA_4_31B_IT="google:gemma-4-31b-it,nvidia_nim:google/gemma-4-31b-it|exhaust"
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .model_alias_registry import AliasTarget

lib_logger = logging.getLogger("rotator_library")

DEFAULT_FALLBACK_RETRY_MODE = "exhaust"
VALID_RETRY_MODES = {"round_robin", "exhaust"}


@dataclass
class ModelFallback:
    """A fallback configuration for a specific model."""

    model_name: str  # Canonical model name (e.g., "gemma-4-31b-it")
    targets: List[AliasTarget] = field(default_factory=list)
    retry_mode: str = DEFAULT_FALLBACK_RETRY_MODE  # "exhaust" or "round_robin"


class ModelFallbackRegistry:
    """
    Registry that maps model names to fallback provider chains.

    Parses MODEL_FALLBACK_* environment variables at construction time.
    Thread-safe for reads after initialization.
    """

    def __init__(self) -> None:
        self._fallbacks: Dict[str, ModelFallback] = {}
        # Lookup table: normalized name → canonical key
        self._lookup: Dict[str, str] = {}
        self._load_from_env()

    @staticmethod
    def _normalize(name: str) -> str:
        """Normalize a model name for lookup (lowercase, periods→hyphens)."""
        return name.lower().replace(".", "-")

    def _register_fallback(self, canonical: str, fb: ModelFallback) -> None:
        """Register a fallback with lookup variants."""
        self._fallbacks[canonical] = fb
        self._lookup[self._normalize(canonical)] = canonical
        self._lookup[canonical] = canonical

    def _load_from_env(self) -> None:
        """Load all MODEL_FALLBACK_* environment variables."""
        for key, value in os.environ.items():
            if not key.startswith("MODEL_FALLBACK_"):
                continue

            # Extract model name: MODEL_FALLBACK_GEMMA_4_31B_IT → gemma-4-31b-it
            canonical = key[len("MODEL_FALLBACK_"):].lower().replace("_", "-")

            try:
                fb = self._parse_fallback_value(canonical, value)
                if fb and fb.targets:
                    self._register_fallback(canonical, fb)
                    target_summary = ", ".join(
                        f"{t.provider}:{t.model_name}" for t in fb.targets
                    )
                    lib_logger.info(
                        f"Registered model fallback: {canonical} → [{target_summary}] "
                        f"(retry: {fb.retry_mode})"
                    )
            except Exception as e:
                lib_logger.warning(
                    f"Failed to parse {key}: {e}"
                )

    def _parse_fallback_value(
        self, canonical: str, value: str
    ) -> Optional[ModelFallback]:
        """
        Parse a fallback env value.

        Format: provider1[:model1],provider2[:model2][|retry_mode]

        The retry mode suffix is optional, separated by |.
        If no :model is given after a provider, the canonical model name is used.
        """
        value = value.strip()
        if not value:
            return None

        # Split off retry mode suffix (last | in the string)
        retry_mode = DEFAULT_FALLBACK_RETRY_MODE
        if "|" in value:
            parts = value.rsplit("|", 1)
            candidate_mode = parts[1].strip().lower()
            if candidate_mode in VALID_RETRY_MODES:
                retry_mode = candidate_mode
                value = parts[0].strip()
            # If not a valid mode, treat | as part of the value

        # Parse comma-separated entries
        targets: List[AliasTarget] = []
        for entry in value.split(","):
            entry = entry.strip()
            if not entry:
                continue

            if ":" in entry:
                # Explicit provider:model format
                provider, model_name = entry.split(":", 1)
                provider = provider.strip().lower()
                model_name = model_name.strip()
            else:
                # Provider-only: use canonical model name
                provider = entry.strip().lower()
                model_name = canonical

            if not provider:
                lib_logger.warning(
                    f"Invalid fallback target '{entry}' for '{canonical}': "
                    f"empty provider name"
                )
                continue

            targets.append(AliasTarget(provider=provider, model_name=model_name))

        if not targets:
            return None

        return ModelFallback(
            model_name=canonical,
            targets=targets,
            retry_mode=retry_mode,
        )

    def _resolve_key(self, model: str) -> Optional[str]:
        """Resolve a model name to its canonical key via lookup table."""
        key = self._lookup.get(model.lower())
        if key:
            return key
        return self._lookup.get(self._normalize(model))

    def resolve(self, model: str) -> Optional[List[AliasTarget]]:
        """
        Resolve a model name to its fallback provider targets.

        Args:
            model: Model name (without provider prefix)

        Returns:
            List of AliasTarget in priority order, or None if no fallback configured
        """
        key = self._resolve_key(model)
        if key:
            fb = self._fallbacks.get(key)
            if fb:
                return list(fb.targets)
        return None

    def get_retry_mode(self, model: str) -> str:
        """
        Get the retry mode for a model's fallback chain.

        Args:
            model: Model name

        Returns:
            "exhaust" or "round_robin"
        """
        key = self._resolve_key(model)
        if key:
            fb = self._fallbacks.get(key)
            if fb:
                return fb.retry_mode
        return DEFAULT_FALLBACK_RETRY_MODE

    def has_fallback(self, model: str) -> bool:
        """Check if a model has fallback providers configured."""
        return self._resolve_key(model) is not None

    def get_all_fallbacks(self) -> Dict[str, ModelFallback]:
        """Get the full fallback registry (for debugging/admin endpoints)."""
        return dict(self._fallbacks)
