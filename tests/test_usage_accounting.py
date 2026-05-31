from __future__ import annotations

from types import SimpleNamespace

from rotator_library.protocols.types import CostDetails, Usage
from rotator_library.usage.accounting import UsageRecord, extract_usage_record


def test_openai_dict_usage_extracts_cache_and_reasoning_without_double_counting() -> None:
    record = extract_usage_record(
        {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "total_tokens": 130,
                "prompt_tokens_details": {"cached_tokens": 40, "cache_creation_tokens": 5},
                "completion_tokens_details": {"reasoning_tokens": 10},
            }
        },
        provider="openai",
        model="gpt-test",
    )

    assert record.input_tokens == 55
    assert record.cache_read_tokens == 40
    assert record.cache_write_tokens == 5
    assert record.completion_tokens == 20
    assert record.reasoning_tokens == 10
    assert record.output_tokens == 30
    assert record.raw_total_tokens == 130
    assert record.total_tokens == 130


def test_openai_usage_extracts_provider_reported_cost_details() -> None:
    record = extract_usage_record(
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cost_details": {"total_cost": 0.0123, "currency": "EUR", "source": "provider_usage"},
            }
        }
    )

    assert record.provider_reported_cost == 0.0123
    assert record.cost_currency == "EUR"
    assert record.cost_source == "provider_usage"


def test_top_level_cost_is_preserved_when_usage_exists() -> None:
    record = extract_usage_record(
        {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "total_cost": 0.03,
            "currency": "EUR",
        }
    )

    assert record.provider_reported_cost == 0.03
    assert record.cost_currency == "EUR"


def test_reference_request_cost_usd_is_preserved() -> None:
    record = extract_usage_record({"usage": {"prompt_tokens": 1, "completion_tokens": 1}, "request_cost_usd": 0.019})

    assert record.provider_reported_cost == 0.019


def test_top_level_estimated_cost_is_preserved() -> None:
    record = extract_usage_record({"usage": {"prompt_tokens": 1, "completion_tokens": 1}, "estimated_cost": 0.027})

    assert record.provider_reported_cost == 0.027


def test_structured_cost_breakdown_without_total_is_summed() -> None:
    record = extract_usage_record(
        {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cost_details": {"cached_input_cost": 0.01, "upstream_inference_cost": 0.02, "web_search_cost": 0.003},
            }
        }
    )

    assert record.provider_reported_cost == 0.033
    assert record.cost_source == "provider_reported_breakdown"


def test_reference_extended_cost_breakdown_aliases_are_summed() -> None:
    record = extract_usage_record(
        {
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cost_details": {
                    "upstream_inference_input_cost": 0.01,
                    "upstream_inference_output_cost": 0.02,
                    "image_input_cost": 0.003,
                    "audio_input_cost": 0.004,
                    "data_storage_cost": 0.005,
                    "estimated_cost": 0.006,
                    "cost_in_usd_ticks": 70_000_000,
                },
            }
        }
    )

    assert record.provider_reported_cost == 0.055


def test_upstream_inference_total_wins_over_split_fields() -> None:
    record = extract_usage_record(
        {
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cost_details": {
                    "upstream_inference_cost": 0.04,
                    "upstream_inference_prompt_cost": 0.01,
                    "upstream_inference_completions_cost": 0.03,
                },
            }
        }
    )

    assert record.provider_reported_cost == 0.04


def test_openai_object_usage_extracts_attributes() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=7,
            prompt_tokens_details=SimpleNamespace(cached_tokens=2, cache_creation_tokens=1),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
        )
    )

    record = extract_usage_record(response)

    assert record.input_tokens == 9
    assert record.cache_read_tokens == 2
    assert record.cache_write_tokens == 1
    assert record.completion_tokens == 4
    assert record.reasoning_tokens == 3


def test_anthropic_usage_extracts_cache_buckets() -> None:
    record = extract_usage_record(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_creation_input_tokens": 8,
            "cache_read_input_tokens": 12,
        }
    )

    assert record.input_tokens == 80
    assert record.cache_read_tokens == 12
    assert record.cache_write_tokens == 8
    assert record.completion_tokens == 20
    assert record.metadata["shape"] == "anthropic"


def test_gemini_usage_metadata_extracts_thought_and_cached_tokens() -> None:
    record = extract_usage_record(
        {
            "usageMetadata": {
                "promptTokenCount": 80,
                "candidatesTokenCount": 25,
                "thoughtsTokenCount": 5,
                "cachedContentTokenCount": 30,
                "totalTokenCount": 110,
            }
        }
    )

    assert record.input_tokens == 50
    assert record.cache_read_tokens == 30
    assert record.completion_tokens == 20
    assert record.reasoning_tokens == 5
    assert record.metadata["shape"] == "gemini"


def test_gemini_usage_extracts_cost_metadata() -> None:
    record = extract_usage_record(
        {
            "usageMetadata": {
                "promptTokenCount": 1,
                "totalTokenCount": 1,
                "costMetadata": {"cost": "0.004", "currency": "USD", "source": "gemini_usage"},
            }
        }
    )

    assert record.provider_reported_cost == 0.004
    assert record.cost_source == "gemini_usage"


def test_protocol_usage_cost_details_are_preserved() -> None:
    record = extract_usage_record(
        Usage(input_tokens=2, output_tokens=3, cost=CostDetails(provider_reported_cost=0.02, currency="GBP", source="protocol_usage"))
    )

    assert record.input_tokens == 2
    assert record.completion_tokens == 3
    assert record.provider_reported_cost == 0.02
    assert record.cost_currency == "GBP"
    assert record.cost_source == "protocol_usage"


def test_responses_usage_extracts_nested_details() -> None:
    record = extract_usage_record(
        {
            "input_tokens": 42,
            "output_tokens": 13,
            "input_tokens_details": {"cached_tokens": 10},
            "output_tokens_details": {"reasoning_tokens": 4},
        },
        source="responses",
    )

    assert record.input_tokens == 32
    assert record.cache_read_tokens == 10
    assert record.completion_tokens == 9
    assert record.reasoning_tokens == 4


def test_unknown_usage_shape_returns_empty_record() -> None:
    record = extract_usage_record({"not_usage": True}, provider="x", model="y")

    assert record == UsageRecord(provider="x", model="y", source="response", metadata={"shape": "openai_like"})
