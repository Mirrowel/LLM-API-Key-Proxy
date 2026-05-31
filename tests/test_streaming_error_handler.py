from __future__ import annotations

from litellm import RateLimitError

from rotator_library.streaming.errors import decide_streaming_error_action


def _rate_limit(retry_after: int | None = None) -> Exception:
    error = RateLimitError("rate limited", llm_provider="openai", model="gpt-test")
    if retry_after is not None:
        error.retry_after = retry_after
    return error


def test_streaming_error_decision_retries_same_key_for_small_retry_after() -> None:
    decision = decide_streaming_error_action(
        _rate_limit(3),
        provider="openai",
        last_streamed_chunk=None,
        attempt=0,
        max_retries=2,
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        provider_cooldown_default_seconds=30,
    )

    assert decision.action == "retry_same"
    assert decision.start_provider_cooldown is False


def test_streaming_error_decision_starts_cooldown_before_visible_output() -> None:
    decision = decide_streaming_error_action(
        _rate_limit(60),
        provider="openai",
        last_streamed_chunk=None,
        attempt=1,
        max_retries=2,
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        provider_cooldown_default_seconds=30,
    )

    assert decision.action == "rotate"
    assert decision.start_provider_cooldown is True
    assert decision.provider_cooldown_duration == 60
    assert decision.provider_cooldown_scope == "provider"


def test_streaming_error_decision_blocks_after_visible_output_and_skips_cooldown() -> None:
    decision = decide_streaming_error_action(
        _rate_limit(60),
        provider="openai",
        last_streamed_chunk='data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
        attempt=0,
        max_retries=2,
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        provider_cooldown_default_seconds=30,
    )

    assert decision.action == "fallback_blocked_after_output"
    assert decision.start_provider_cooldown is False


def test_streaming_error_decision_allows_reasoning_only_retry_when_enabled() -> None:
    decision = decide_streaming_error_action(
        _rate_limit(3),
        provider="openai",
        last_streamed_chunk='data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n',
        attempt=0,
        max_retries=2,
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        provider_cooldown_default_seconds=30,
        allow_reasoning_only_retry=True,
    )

    assert decision.action == "retry_same"


def test_streaming_error_decision_reports_model_cooldown_scope() -> None:
    decision = decide_streaming_error_action(
        Exception("MODEL_CAPACITY_EXHAUSTED"),
        provider="openai",
        model="gpt-5",
        last_streamed_chunk=None,
        attempt=1,
        max_retries=2,
        small_cooldown_threshold=10,
        provider_cooldown_min_seconds=10,
        provider_cooldown_default_seconds=30,
    )

    assert decision.start_provider_cooldown is True
    assert decision.provider_cooldown_scope == "model"
    assert decision.provider_cooldown_model == "gpt-5"
