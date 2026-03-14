from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set


class RoutingPolicyError(ValueError):
    """Raised when routing override configuration or resolution is invalid."""


@dataclass(frozen=True)
class RouteDecision:
    requested_model: str
    clean_model: str
    selected_provider: Optional[str]
    rewritten_model: Optional[str]
    strategy: str
    selection_source: str
    override_applied: bool
    candidate_providers: list[str]
    strict: bool
    allow_global_fallback: bool
    reason: Optional[str] = None


class RoutingPolicy:
    """Resolve weighted-router models into concrete provider-prefixed models.

    v1 intentionally supports only strict single-provider overrides. It rewrites
    abstract `weighted-router/<model>` requests before provider lock-in so the
    existing retry and credential machinery can continue unchanged.
    """

    def __init__(
        self,
        model_overrides: Dict[str, Any],
        available_providers: Iterable[str],
        provider_models: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        if not isinstance(model_overrides, dict):
            raise RoutingPolicyError("MODEL_ROUTING_OVERRIDES must decode to an object")

        self.model_overrides = model_overrides
        self.available_providers = set(available_providers)
        self.provider_models = provider_models or {}
        self._validate()

    def _validate(self) -> None:
        for clean_model, override in self.model_overrides.items():
            if not isinstance(clean_model, str) or not clean_model:
                raise RoutingPolicyError("routing override keys must be non-empty model names")
            if not isinstance(override, dict):
                raise RoutingPolicyError(f"routing override for '{clean_model}' must be an object")

            strategy = override.get("strategy")
            if strategy != "single":
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' must use strategy 'single' in v1"
                )

            primary = override.get("primary")
            if not isinstance(primary, str) or not primary:
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' requires a non-empty 'primary' provider"
                )
            if primary not in self.available_providers:
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' references unknown provider '{primary}'"
                )

            allowed_providers = override.get("allowed_providers", [primary])
            if not isinstance(allowed_providers, list) or not all(
                isinstance(provider, str) and provider for provider in allowed_providers
            ):
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' must use a string list for 'allowed_providers'"
                )
            if allowed_providers != [primary]:
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' must restrict 'allowed_providers' to ['{primary}'] in v1"
                )

            fallback_providers = override.get("fallback_providers", [])
            if fallback_providers not in (None, []):
                raise RoutingPolicyError(
                    f"routing override for '{clean_model}' cannot define 'fallback_providers' in v1"
                )

            provider_models = self.provider_models.get(primary)
            if provider_models and clean_model not in provider_models:
                raise RoutingPolicyError(
                    f"provider '{primary}' does not expose model '{clean_model}' in configured model definitions"
                )

    def resolve(self, model: str) -> RouteDecision:
        if "/" not in model:
            return RouteDecision(
                requested_model=model,
                clean_model=model,
                selected_provider=None,
                rewritten_model=model,
                strategy="passthrough",
                selection_source="passthrough",
                override_applied=False,
                candidate_providers=[],
                strict=False,
                allow_global_fallback=True,
            )

        provider, clean_model = model.split("/", 1)
        if provider != "weighted-router":
            return RouteDecision(
                requested_model=model,
                clean_model=clean_model,
                selected_provider=provider,
                rewritten_model=model,
                strategy="passthrough",
                selection_source="passthrough",
                override_applied=False,
                candidate_providers=[provider],
                strict=False,
                allow_global_fallback=True,
            )

        override = self.model_overrides.get(clean_model)
        if override is None:
            raise RoutingPolicyError(
                f"No routing override configured for weighted-router model '{clean_model}'"
            )

        selected_provider = override["primary"]
        return RouteDecision(
            requested_model=model,
            clean_model=clean_model,
            selected_provider=selected_provider,
            rewritten_model=f"{selected_provider}/{clean_model}",
            strategy="single",
            selection_source="model_override",
            override_applied=True,
            candidate_providers=[selected_provider],
            strict=bool(override.get("strict", True)),
            allow_global_fallback=bool(override.get("allow_global_fallback", False)),
            reason=override.get("reason"),
        )
