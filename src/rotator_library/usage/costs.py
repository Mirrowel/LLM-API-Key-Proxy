# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Advisory cost calculation for normalized usage records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import litellm

from ..protocols import serialize_value
from .accounting import UsageRecord


@dataclass(frozen=True)
class ModelPricing:
    """Per-token pricing for one provider/model.

    Prices are advisory and local-only; this module never calls a network pricing
    endpoint. Providers can return this object from `get_model_pricing()` later.
    """

    input_cost_per_token: float = 0.0
    cache_read_cost_per_token: float = 0.0
    cache_write_cost_per_token: float = 0.0
    output_cost_per_token: float = 0.0
    reasoning_cost_per_token: float = 0.0
    currency: str = "USD"
    source: str = "explicit"


@dataclass(frozen=True)
class CostBreakdown:
    """Advisory request cost split by normalized usage bucket."""

    input_cost: float = 0.0
    cache_read_cost: float = 0.0
    cache_write_cost: float = 0.0
    output_cost: float = 0.0
    reasoning_cost: float = 0.0
    provider_reported_cost: float = 0.0
    currency: str = "USD"
    pricing_source: str = "unavailable"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        if self.provider_reported_cost:
            return self.provider_reported_cost
        return self.input_cost + self.cache_read_cost + self.cache_write_cost + self.output_cost + self.reasoning_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_cost": self.input_cost,
            "cache_read_cost": self.cache_read_cost,
            "cache_write_cost": self.cache_write_cost,
            "output_cost": self.output_cost,
            "reasoning_cost": self.reasoning_cost,
            "provider_reported_cost": self.provider_reported_cost,
            "total_cost": self.total_cost,
            "currency": self.currency,
            "pricing_source": self.pricing_source,
            "metadata": serialize_value(self.metadata),
        }


class CostCalculator:
    """Calculate advisory costs without replacing usage tracking."""

    def __init__(self, *, provider_plugin: Any = None, use_litellm_fallback: bool = True, config: Any = None, env: Any = None) -> None:
        self.provider_plugin = provider_plugin
        self.use_litellm_fallback = use_litellm_fallback
        self.config = config
        self.env = env

    def calculate(self, usage: UsageRecord, *, model: str, response: Any = None, provider: str | None = None) -> CostBreakdown:
        """Return an advisory cost breakdown for a normalized usage record."""

        if self.provider_plugin and getattr(self.provider_plugin, "skip_cost_calculation", False):
            return CostBreakdown(pricing_source="skipped", metadata={"reason": "provider_skip_cost_calculation"})
        if usage.provider_reported_cost is not None:
            return CostBreakdown(
                provider_reported_cost=usage.provider_reported_cost,
                currency=usage.cost_currency,
                pricing_source=usage.cost_source or "provider_reported",
                metadata={"actual_provider_reported": True},
            )
        pricing = self._provider_pricing(model)
        if pricing:
            return _calculate_from_pricing(usage, pricing)
        pricing = self._configured_pricing(provider or usage.provider or _provider_from_model(model), model)
        if pricing:
            return _calculate_from_pricing(usage, pricing)
        if self.use_litellm_fallback:
            lite = self._litellm_cost(usage, model=model, response=response)
            if lite:
                return lite
        return CostBreakdown(pricing_source="unavailable")

    def _provider_pricing(self, model: str) -> Optional[ModelPricing]:
        if not self.provider_plugin:
            return None
        method = getattr(self.provider_plugin, "get_model_pricing", None)
        if not method:
            return None
        pricing = method(model)
        if isinstance(pricing, ModelPricing):
            return pricing
        if isinstance(pricing, dict):
            return ModelPricing(**pricing)
        return None

    def _configured_pricing(self, provider: str | None, model: str) -> Optional[ModelPricing]:
        if not provider:
            return None
        from ..config.experimental import get_configured_model_pricing

        return get_configured_model_pricing(provider, _model_without_provider(provider, model), config=self.config, env=self.env)

    @staticmethod
    def _litellm_cost(usage: UsageRecord, *, model: str, response: Any = None) -> Optional[CostBreakdown]:
        if response is not None:
            try:
                cost = litellm.completion_cost(completion_response=response, model=model)
                if cost is not None:
                    return CostBreakdown(output_cost=float(cost), pricing_source="litellm_completion_cost")
            except Exception:
                pass
        try:
            model_info = litellm.get_model_info(model)
        except Exception:
            return None
        input_price = float(model_info.get("input_cost_per_token") or 0.0)
        output_price = float(model_info.get("output_cost_per_token") or 0.0)
        if input_price == 0.0 and output_price == 0.0:
            return None
        pricing = ModelPricing(
            input_cost_per_token=input_price,
            cache_read_cost_per_token=float(model_info.get("cache_read_input_token_cost") or input_price),
            cache_write_cost_per_token=float(model_info.get("cache_creation_input_token_cost") or input_price),
            output_cost_per_token=output_price,
            reasoning_cost_per_token=output_price,
            source="litellm_model_info",
        )
        return _calculate_from_pricing(usage, pricing)


def _calculate_from_pricing(usage: UsageRecord, pricing: ModelPricing) -> CostBreakdown:
    return CostBreakdown(
        input_cost=usage.input_tokens * pricing.input_cost_per_token,
        cache_read_cost=usage.cache_read_tokens * pricing.cache_read_cost_per_token,
        cache_write_cost=usage.cache_write_tokens * pricing.cache_write_cost_per_token,
        output_cost=usage.completion_tokens * pricing.output_cost_per_token,
        reasoning_cost=usage.reasoning_tokens * (pricing.reasoning_cost_per_token or pricing.output_cost_per_token),
        currency=pricing.currency,
        pricing_source=pricing.source,
    )


def _provider_from_model(model: str) -> Optional[str]:
    return model.split("/", 1)[0] if "/" in model else None


def _model_without_provider(provider: str, model: str) -> str:
    prefix = f"{provider}/"
    return model[len(prefix) :] if model.startswith(prefix) else model
