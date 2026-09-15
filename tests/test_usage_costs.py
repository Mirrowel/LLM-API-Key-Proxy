from __future__ import annotations

from rotator_library.usage.accounting import UsageRecord
from rotator_library.usage.costs import CostCalculator, ModelPricing


class PricingProvider:
    def get_model_pricing(self, model: str):
        return ModelPricing(
            input_cost_per_token=0.001,
            cache_read_cost_per_token=0.0001,
            cache_write_cost_per_token=0.0005,
            output_cost_per_token=0.002,
            reasoning_cost_per_token=0.003,
            source="provider_test",
        )


class SkipCostProvider:
    skip_cost_calculation = True


def test_explicit_pricing_calculates_all_usage_buckets() -> None:
    usage = UsageRecord(input_tokens=10, cache_read_tokens=20, cache_write_tokens=5, completion_tokens=7, reasoning_tokens=3)

    cost = CostCalculator(provider_plugin=PricingProvider(), use_litellm_fallback=False).calculate(usage, model="test")

    assert cost.input_cost == 0.01
    assert cost.cache_read_cost == 0.002
    assert cost.cache_write_cost == 0.0025
    assert cost.output_cost == 0.014
    assert cost.reasoning_cost == 0.009000000000000001
    assert cost.pricing_source == "provider_test"


def test_skip_cost_provider_returns_zero_skipped_breakdown() -> None:
    cost = CostCalculator(provider_plugin=SkipCostProvider()).calculate(UsageRecord(input_tokens=100), model="test")

    assert cost.total_cost == 0.0
    assert cost.pricing_source == "skipped"


def test_provider_reported_cost_wins_over_advisory_pricing() -> None:
    usage = UsageRecord(input_tokens=10, completion_tokens=10, provider_reported_cost=0.123, cost_currency="EUR", cost_source="provider_actual")

    cost = CostCalculator(provider_plugin=PricingProvider(), use_litellm_fallback=False).calculate(usage, model="test")

    assert cost.total_cost == 0.123
    assert cost.provider_reported_cost == 0.123
    assert cost.currency == "EUR"
    assert cost.pricing_source == "provider_actual"
    assert cost.input_cost == 0.0


def test_skip_cost_still_wins_over_provider_reported_cost() -> None:
    usage = UsageRecord(provider_reported_cost=0.123)

    cost = CostCalculator(provider_plugin=SkipCostProvider()).calculate(usage, model="test")

    assert cost.total_cost == 0.0
    assert cost.pricing_source == "skipped"


def test_missing_pricing_returns_unavailable_zero() -> None:
    cost = CostCalculator(use_litellm_fallback=False).calculate(UsageRecord(input_tokens=100), model="unknown-model")

    assert cost.total_cost == 0.0
    assert cost.pricing_source == "unavailable"


def test_litellm_model_info_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "rotator_library.usage.costs.litellm.get_model_info",
        lambda model: {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002},
    )

    cost = CostCalculator().calculate(UsageRecord(input_tokens=2, completion_tokens=3), model="gpt-test")

    assert cost.total_cost == 0.008
    assert cost.pricing_source == "litellm_model_info"
