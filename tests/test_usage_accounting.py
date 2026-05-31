from __future__ import annotations

from types import SimpleNamespace

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

    assert record.input_tokens == 60
    assert record.cache_read_tokens == 40
    assert record.cache_write_tokens == 5
    assert record.completion_tokens == 20
    assert record.reasoning_tokens == 10
    assert record.output_tokens == 30
    assert record.raw_total_tokens == 130


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

    assert record.input_tokens == 10
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
