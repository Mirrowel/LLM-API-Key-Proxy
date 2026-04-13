# SPDX-License-Identifier: LGPL-3.0-only

"""
Smart Latest Model Alias Registry.

Provides stable endpoint names (e.g., ``nanogpt/glm-latest``) that
automatically resolve to the newest matching model version at request
time, using glob patterns and semantic version sorting.

Configuration via environment variables::

    MODEL_LATEST_<ALIAS_NAME>=<provider>:<glob_pattern>[:<options>]

Options (colon-separated key=value pairs after the glob pattern):
    exclude=<glob>,<glob>   Exclude matching models
    prefer=<suffix>         When same version has multiple variants, prefer this suffix
    tiebreak=<mode>         Tiebreaker for same-version candidates:
                             cheapest (default), expensive, stripped

Global configuration::

    MODEL_LATEST_STRIP_SUFFIXES=-TEE,-FP8,-original

Examples::

    MODEL_LATEST_STRIP_SUFFIXES=-TEE,-FP8,-original
    MODEL_LATEST_GLM_LATEST=nanogpt:glm-[0-9]*:exclude=*:thinking,*v*
    MODEL_LATEST_GLM_TURBO=chutes:GLM-*-Turbo
    MODEL_LATEST_DEEPSEEK_V=chutes:DeepSeek-V*:prefer=-TEE
"""

import fnmatch
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

lib_logger = logging.getLogger("rotator_library")

# Valid tiebreak modes
VALID_TIEBREAK_MODES = {"cheapest", "expensive", "stripped"}
DEFAULT_TIEBREAK = "cheapest"


@dataclass
class LatestRule:
    """A single 'latest' resolution rule."""

    alias_name: str  # e.g., "glm-latest" (virtual endpoint name)
    provider: str  # e.g., "nanogpt"
    glob_pattern: str  # e.g., "glm-[0-9]*"
    exclude_patterns: List[str] = field(default_factory=list)
    prefer_suffix: Optional[str] = None  # e.g., "-TEE"
    strip_suffixes: List[str] = field(default_factory=list)
    tiebreak: str = DEFAULT_TIEBREAK  # "cheapest", "expensive", "stripped"

    @property
    def virtual_model(self) -> str:
        """Full virtual model ID: provider/alias-name."""
        return f"{self.provider}/{self.alias_name}"


@dataclass
class _VersionedCandidate:
    """Internal: a model candidate with extracted version info."""

    original_model_id: str  # Full original ID from model list (e.g., "zai-org/GLM-5-TEE")
    bare_name: str  # After stripping org prefix (e.g., "GLM-5-TEE")
    stripped_name: str  # After stripping infra suffixes (e.g., "GLM-5")
    version: Tuple[int, ...]  # Extracted version tuple (e.g., (5,))
    was_stripped: bool  # Whether an infra suffix was actually removed


class ModelLatestRegistry:
    """
    Registry for smart 'latest' model aliases with version-aware resolution.

    Parses ``MODEL_LATEST_*`` environment variables at construction time.
    Thread-safe for reads after initialization.
    """

    # How long to cache resolved alias results (seconds)
    RESOLUTION_CACHE_TTL = 6 * 60 * 60  # 6 hours

    def __init__(self) -> None:
        self._rules: Dict[str, LatestRule] = {}  # "provider/alias" → rule
        self._global_strip_suffixes: List[str] = []
        self._pricing_resolver: Optional[Callable[[str, str], Optional[float]]] = None
        self._resolution_cache: Dict[str, Tuple[str, float]] = {}
        self._load_from_env()

    def set_pricing_resolver(
        self, resolver: Callable[[str, str], Optional[float]]
    ) -> None:
        """
        Inject a pricing resolver callback for cost-based tiebreaking.

        The callback signature is:
            resolver(provider: str, model_id: str) -> Optional[float]
        where the return value is the input cost per token, or None.
        """
        self._pricing_resolver = resolver

    # =========================================================================
    # ENV LOADING
    # =========================================================================

    def _load_from_env(self) -> None:
        """Load all MODEL_LATEST_* environment variables."""
        # Load global strip suffixes first
        strip_raw = os.environ.get("MODEL_LATEST_STRIP_SUFFIXES", "")
        if strip_raw:
            self._global_strip_suffixes = [
                s.strip() for s in strip_raw.split(",") if s.strip()
            ]
            if self._global_strip_suffixes:
                lib_logger.info(
                    f"Latest aliases: global strip suffixes: "
                    f"{self._global_strip_suffixes}"
                )

        for key, value in os.environ.items():
            if not key.startswith("MODEL_LATEST_"):
                continue
            # Skip the global config key
            if key == "MODEL_LATEST_STRIP_SUFFIXES":
                continue

            # Extract alias name: MODEL_LATEST_GLM_LATEST → glm-latest
            alias_name = key[len("MODEL_LATEST_"):].lower().replace("_", "-")

            try:
                rule = self._parse_rule(alias_name, value)
                if rule:
                    # Auto-strip redundant provider prefix from alias name
                    # e.g., MODEL_LATEST_CHUTES_GLM_LATEST with provider=chutes
                    # produces alias "chutes-glm-latest" → strip to "glm-latest"
                    # so virtual model is "chutes/glm-latest" not "chutes/chutes-glm-latest"
                    provider_prefix = f"{rule.provider}-"
                    if rule.alias_name.startswith(provider_prefix):
                        rule.alias_name = rule.alias_name[len(provider_prefix):]
                    lookup_key = rule.virtual_model
                    self._rules[lookup_key] = rule
                    lib_logger.info(
                        f"Registered latest alias: {lookup_key} → "
                        f"{rule.provider}:{rule.glob_pattern} "
                        f"(tiebreak={rule.tiebreak}"
                        f"{', prefer=' + rule.prefer_suffix if rule.prefer_suffix else ''}"
                        f"{', exclude=' + str(rule.exclude_patterns) if rule.exclude_patterns else ''}"
                        f")"
                    )
            except Exception as e:
                lib_logger.warning(f"Failed to parse {key}: {e}")

    def _parse_rule(self, alias_name: str, value: str) -> Optional[LatestRule]:
        """
        Parse a MODEL_LATEST_* value.

        Format: provider:glob_pattern[:option1=val1[:option2=val2]]

        Options:
            exclude=<glob>,<glob>
            prefer=<suffix>
            tiebreak=cheapest|expensive|stripped

        Note: Colons may appear inside option values (e.g., exclude=*:thinking).
        We split provider on the first colon, then use regex to find option
        boundaries by looking for ':keyword=' patterns.
        """
        value = value.strip()
        if not value:
            return None

        # Split on first colon to get provider
        first_colon = value.find(":")
        if first_colon == -1:
            lib_logger.warning(
                f"Invalid latest alias '{alias_name}': "
                f"expected 'provider:pattern' format, got '{value}'"
            )
            return None

        provider = value[:first_colon].strip().lower()
        remainder = value[first_colon + 1:]

        if not provider or not remainder.strip():
            lib_logger.warning(
                f"Invalid latest alias '{alias_name}': "
                f"empty provider or pattern"
            )
            return None

        # Find the first option boundary: :exclude=, :prefer=, or :tiebreak=
        # This avoids misinterpreting colons inside glob patterns like *:thinking
        first_opt_pos = len(remainder)
        for kw in ("exclude=", "prefer=", "tiebreak="):
            pos = remainder.lower().find(f":{kw}")
            if pos != -1 and pos < first_opt_pos:
                first_opt_pos = pos

        glob_pattern = remainder[:first_opt_pos].strip()
        options_str = remainder[first_opt_pos:].strip()

        if not glob_pattern:
            lib_logger.warning(
                f"Invalid latest alias '{alias_name}': empty pattern"
            )
            return None

        # Parse options: split on ":keyword=" boundaries using regex
        # This correctly handles colons inside values like exclude=*:thinking
        exclude_patterns: List[str] = []
        prefer_suffix: Optional[str] = None
        tiebreak = DEFAULT_TIEBREAK

        if options_str:
            opt_parts = re.split(
                r":(?=(?:exclude|prefer|tiebreak)=)", options_str
            )
            for opt_part in opt_parts:
                opt_part = opt_part.strip()
                if not opt_part:
                    continue

                if opt_part.startswith("exclude="):
                    raw_excludes = opt_part[len("exclude="):]
                    exclude_patterns = [
                        e.strip() for e in raw_excludes.split(",") if e.strip()
                    ]
                elif opt_part.startswith("prefer="):
                    prefer_suffix = opt_part[len("prefer="):].strip()
                elif opt_part.startswith("tiebreak="):
                    mode = opt_part[len("tiebreak="):].strip().lower()
                    if mode in VALID_TIEBREAK_MODES:
                        tiebreak = mode
                    else:
                        lib_logger.warning(
                            f"Invalid tiebreak mode '{mode}' for '{alias_name}', "
                            f"using '{DEFAULT_TIEBREAK}'"
                        )

        # If prefer= is set, that overrides tiebreak mode
        if prefer_suffix:
            tiebreak = "prefer"

        return LatestRule(
            alias_name=alias_name,
            provider=provider,
            glob_pattern=glob_pattern,
            exclude_patterns=exclude_patterns,
            prefer_suffix=prefer_suffix,
            strip_suffixes=list(self._global_strip_suffixes),
            tiebreak=tiebreak,
        )

    # =========================================================================
    # RESOLUTION
    # =========================================================================

    def resolve(
        self,
        model: str,
        available_models: Dict[str, List[str]],
    ) -> Optional[str]:
        """
        Resolve a latest-alias to the actual latest model.

        Args:
            model: Full model string e.g., "nanogpt/glm-latest"
            available_models: Dict of provider → list of model names
                (from RotatingClient._model_list_cache)

        Returns:
            Resolved model string e.g., "nanogpt/zai-org/glm-5.1", or None
        """
        # Check resolution cache first
        now = time.time()
        cached = self._resolution_cache.get(model.lower())
        if cached:
            resolved_model, timestamp = cached
            if now - timestamp < self.RESOLUTION_CACHE_TTL:
                return resolved_model

        # Lookup by exact virtual model key
        rule = self._rules.get(model.lower())
        if not rule:
            return None

        # Get provider's cached model list
        provider_models = available_models.get(rule.provider, [])
        if not provider_models:
            lib_logger.debug(
                f"Latest alias '{model}': no cached models for "
                f"provider '{rule.provider}'"
            )
            return None

        # Build versioned candidates
        candidates = self._match_and_sort(rule, provider_models)

        if not candidates:
            lib_logger.debug(
                f"Latest alias '{model}': no models matched "
                f"pattern '{rule.glob_pattern}'"
            )
            return None

        # Take the highest version group
        best_version = candidates[0].version
        top_candidates = [c for c in candidates if c.version == best_version]

        # Apply tiebreaker
        winner = self._apply_tiebreaker(rule, top_candidates)

        resolved = f"{rule.provider}/{winner.original_model_id}"
        self._resolution_cache[model.lower()] = (resolved, now)
        lib_logger.info(
            f"Latest alias resolved: {model} → {resolved} "
            f"(version={best_version}, candidates={len(top_candidates)})"
        )
        return resolved

    def _match_and_sort(
        self,
        rule: LatestRule,
        provider_models: List[str],
    ) -> List[_VersionedCandidate]:
        """
        Match models against rule pattern and sort by version descending.

        Steps:
        1. Strip provider prefix from each model
        2. Strip org prefix for matching
        3. Case-insensitive glob match
        4. Apply exclude patterns
        5. Extract versions and sort descending
        """
        candidates: List[_VersionedCandidate] = []

        for full_model in provider_models:
            # Strip provider prefix (e.g., "chutes/zai-org/GLM-5-TEE" → "zai-org/GLM-5-TEE")
            model_id = full_model
            if "/" in full_model:
                # The model list entries from get_models() are often prefixed
                # with "provider/" already — strip that layer
                parts = full_model.split("/", 1)
                if parts[0].lower() == rule.provider.lower():
                    model_id = parts[1]

            # Strip org prefix for matching (e.g., "zai-org/GLM-5-TEE" → "GLM-5-TEE")
            bare_name = self._strip_org_prefix(model_id)

            # Case-insensitive glob match
            if not fnmatch.fnmatch(bare_name.lower(), rule.glob_pattern.lower()):
                continue

            # Apply exclude patterns
            excluded = False
            for exc_pattern in rule.exclude_patterns:
                if fnmatch.fnmatch(bare_name.lower(), exc_pattern.lower()):
                    excluded = True
                    break
            if excluded:
                continue

            # Strip infra suffixes and extract version
            stripped_name, was_stripped = self._strip_infra_suffixes(
                bare_name, rule.strip_suffixes
            )
            version = self._extract_version(stripped_name)

            candidates.append(
                _VersionedCandidate(
                    original_model_id=model_id,
                    bare_name=bare_name,
                    stripped_name=stripped_name,
                    version=version,
                    was_stripped=was_stripped,
                )
            )

        # Sort by version descending (highest first)
        candidates.sort(key=lambda c: c.version, reverse=True)

        return candidates

    def _apply_tiebreaker(
        self,
        rule: LatestRule,
        candidates: List[_VersionedCandidate],
    ) -> _VersionedCandidate:
        """
        Break ties between candidates with the same version.

        Tiebreak modes:
        - prefer: pick candidate matching prefer_suffix
        - cheapest: pick lowest input cost (via pricing resolver)
        - expensive: pick highest input cost
        - stripped: pick candidate whose infra suffix was stripped
        """
        if len(candidates) == 1:
            return candidates[0]

        # prefer= suffix match
        if rule.tiebreak == "prefer" and rule.prefer_suffix:
            for c in candidates:
                if c.bare_name.lower().endswith(rule.prefer_suffix.lower()):
                    return c
            # Suffix not found — fall through to cost-based

        # Cost-based tiebreaker
        if rule.tiebreak in ("cheapest", "expensive") or (
            rule.tiebreak == "prefer" and rule.prefer_suffix
        ):
            winner = self._cost_tiebreak(
                rule.provider, candidates, prefer_cheap=(rule.tiebreak != "expensive")
            )
            if winner:
                return winner

        # stripped: prefer the candidate that had its suffix removed
        if rule.tiebreak == "stripped":
            for c in candidates:
                if c.was_stripped:
                    return c

        # Final fallback: stripped > alphabetical
        stripped_candidates = [c for c in candidates if c.was_stripped]
        if stripped_candidates:
            return stripped_candidates[0]
        return candidates[0]

    def _cost_tiebreak(
        self,
        provider: str,
        candidates: List[_VersionedCandidate],
        prefer_cheap: bool = True,
    ) -> Optional[_VersionedCandidate]:
        """
        Break ties using pricing data.

        Returns None if pricing is unavailable for all candidates.
        """
        if not self._pricing_resolver:
            return None

        priced: List[Tuple[float, _VersionedCandidate]] = []
        for c in candidates:
            cost = self._pricing_resolver(provider, c.original_model_id)
            if cost is not None:
                priced.append((cost, c))

        if not priced:
            lib_logger.debug(
                f"Cost tiebreak: no pricing data for {len(candidates)} candidates"
            )
            return None

        # Sort by cost
        priced.sort(key=lambda x: x[0], reverse=not prefer_cheap)
        winner_cost, winner = priced[0]
        runner_up = priced[1] if len(priced) > 1 else None

        lib_logger.debug(
            f"Cost tiebreak ({'cheapest' if prefer_cheap else 'expensive'}): "
            f"picked {winner.bare_name} (${winner_cost:.6f}/tok)"
            f"{f' over {runner_up[1].bare_name} (${runner_up[0]:.6f}/tok)' if runner_up else ''}"
        )
        return winner

    # =========================================================================
    # UTILITIES
    # =========================================================================

    @staticmethod
    def _strip_org_prefix(model_name: str) -> str:
        """
        Strip org prefix for glob matching.

        'zai-org/GLM-5-TEE' → 'GLM-5-TEE'
        'GLM-5' → 'GLM-5'
        """
        return model_name.rsplit("/", 1)[-1] if "/" in model_name else model_name

    @staticmethod
    def _strip_infra_suffixes(
        name: str, suffixes: List[str]
    ) -> Tuple[str, bool]:
        """
        Strip infrastructure suffixes for version comparison.

        Returns (stripped_name, was_stripped).
        Only the first matching suffix is removed.
        """
        for suffix in suffixes:
            if name.lower().endswith(suffix.lower()):
                return name[: -len(suffix)], True
        return name, False

    @staticmethod
    def _extract_version(name: str) -> Tuple[int, ...]:
        """
        Extract a sortable version tuple from a model name.

        Examples:
            'GLM-5'      → (5,)
            'GLM-5.1'    → (5, 1)
            'GLM-4.7'    → (4, 7)
            'DeepSeek-V3.2' → (3, 2)
            'Qwen3.5'    → (3, 5)

        Non-numeric parts are ignored. Returns (0,) if no numbers found.
        """
        # Find all numeric segments (integers and decimals)
        numbers = re.findall(r"(\d+)", name)
        return tuple(int(n) for n in numbers) if numbers else (0,)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_virtual_models(self) -> List[str]:
        """
        Return all virtual model names for the /v1/models endpoint.

        These are the stable endpoint names that clients can target.
        """
        return [rule.virtual_model for rule in self._rules.values()]

    def get_all_rules(self) -> Dict[str, LatestRule]:
        """Get the full rule registry (for debugging/admin endpoints)."""
        return dict(self._rules)

    def get_diagnostics(
        self,
        available_models: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """
        Return debug info: each alias, its matches, and resolved target.

        Used by the admin endpoint.
        """
        result: Dict[str, Any] = {
            "aliases": {},
            "global_strip_suffixes": list(self._global_strip_suffixes),
        }

        for key, rule in self._rules.items():
            provider_models = available_models.get(rule.provider, [])
            candidates = self._match_and_sort(rule, provider_models)

            # Resolve the winner
            resolved_to: Optional[str] = None
            if candidates:
                best_version = candidates[0].version
                top = [c for c in candidates if c.version == best_version]
                winner = self._apply_tiebreaker(rule, top)
                resolved_to = f"{rule.provider}/{winner.original_model_id}"

            result["aliases"][key] = {
                "rule": f"{rule.provider}:{rule.glob_pattern}",
                "exclude": rule.exclude_patterns,
                "prefer": rule.prefer_suffix,
                "tiebreak": rule.tiebreak,
                "resolved_to": resolved_to,
                "all_matches": [
                    {
                        "name": c.bare_name,
                        "version": list(c.version),
                        "full_id": c.original_model_id,
                        "was_stripped": c.was_stripped,
                    }
                    for c in candidates
                ],
            }

        return result

    def is_latest_alias(self, model: str) -> bool:
        """Check if a model name is a registered latest alias."""
        return model.lower() in self._rules

    def has_rules(self) -> bool:
        """Check if any latest alias rules are configured."""
        return bool(self._rules)

    def clear_resolution_cache(self) -> None:
        """Clear the resolution cache.

        Call this when the underlying model lists are refreshed so that
        aliases are re-evaluated against the new catalog.
        """
        self._resolution_cache.clear()
