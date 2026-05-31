from __future__ import annotations

from rotator_library.config.experimental import env_price_key, load_config_from_mapping
from rotator_library.usage.accounting import UsageRecord
from rotator_library.usage.costs import CostCalculator, ModelPricing


class ExplicitPricingProvider:
    def get_model_pricing(self, model: str) -> ModelPricing:
        return ModelPricing(input_cost_per_token=9.0, source="provider")


class SkipCostProvider:
    skip_cost_calculation = True


def _usage() -> UsageRecord:
    return UsageRecord(input_tokens=10, completion_tokens=5, reasoning_tokens=2, cache_read_tokens=3, cache_write_tokens=4, provider="openai", model="gpt-test")


def test_json_pricing_calculates_all_buckets() -> None:
    config = load_config_from_mapping(
        {"pricing": {"openai": {"gpt-test": {"input": 1.0, "output": 2.0, "reasoning": 3.0, "cache_read": 0.5, "cache_write": 0.75}}}}
    )

    cost = CostCalculator(config=config, use_litellm_fallback=False).calculate(_usage(), model="openai/gpt-test")

    assert cost.pricing_source == "json_config"
    assert cost.input_cost == 10.0
    assert cost.output_cost == 10.0
    assert cost.reasoning_cost == 6.0
    assert cost.cache_read_cost == 1.5
    assert cost.cache_write_cost == 3.0


def test_env_pricing_overrides_json_pricing() -> None:
    config = load_config_from_mapping({"pricing": {"openai": {"gpt-test": {"input": 1.0}}}})
    env = {env_price_key("openai", "gpt-test", "input"): "4.0"}

    cost = CostCalculator(config=config, env=env, use_litellm_fallback=False).calculate(_usage(), model="openai/gpt-test")

    assert cost.pricing_source == "env"
    assert cost.input_cost == 40.0


def test_provider_pricing_beats_env_and_json_pricing() -> None:
    config = load_config_from_mapping({"pricing": {"openai": {"gpt-test": {"input": 1.0}}}})
    env = {env_price_key("openai", "gpt-test", "input"): "4.0"}

    cost = CostCalculator(provider_plugin=ExplicitPricingProvider(), config=config, env=env, use_litellm_fallback=False).calculate(_usage(), model="openai/gpt-test")

    assert cost.pricing_source == "provider"
    assert cost.input_cost == 90.0


def test_skip_cost_provider_beats_all_config_pricing() -> None:
    config = load_config_from_mapping({"pricing": {"openai": {"gpt-test": {"input": 1.0}}}})

    cost = CostCalculator(provider_plugin=SkipCostProvider(), config=config, use_litellm_fallback=False).calculate(_usage(), model="openai/gpt-test")

    assert cost.pricing_source == "skipped"
    assert cost.total_cost == 0.0


def test_missing_pricing_remains_unavailable() -> None:
    cost = CostCalculator(use_litellm_fallback=False).calculate(_usage(), model="openai/gpt-test")

    assert cost.pricing_source == "unavailable"
