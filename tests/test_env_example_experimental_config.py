from __future__ import annotations

from pathlib import Path


def test_env_example_documents_experimental_config_knobs() -> None:
    text = Path(".env.example").read_text(encoding="utf-8")

    for key in (
        "LLM_PROXY_CONFIG_FILE",
        "FALLBACK_GROUPS",
        "FALLBACK_GROUP_CODE_CHAIN",
        "MODEL_ROUTE_CODE",
        "PROVIDER_COOLDOWN_MIN_SECONDS",
        "PROVIDER_COOLDOWN_DEFAULT_SECONDS",
        "PROVIDER_COOLDOWN_ON_QUOTA",
        "PROVIDER_BACKOFF_WINDOW_SECONDS",
        "PROVIDER_BACKOFF_THRESHOLD",
        "PROVIDER_BACKOFF_BASE_SECONDS",
        "PROVIDER_BACKOFF_MAX_SECONDS",
        "FAILURE_HISTORY_MAX_ENTRIES",
        "RESPONSES_STORE_TTL_SECONDS",
        "RESPONSES_STORE_MAX_ITEMS",
        "RESPONSES_STORE_FAILED",
        "RESPONSES_STORE_IN_PROGRESS",
        "STREAM_TRACE_METRICS",
        "STREAM_TTFB_TIMEOUT_SECONDS",
        "STREAM_STALL_TIMEOUT_SECONDS",
        "STREAM_HEARTBEAT_INTERVAL_SECONDS",
        "STREAM_HEARTBEAT_SECONDS",
        "STREAM_CANCEL_UPSTREAM_ON_DISCONNECT",
        "MODEL_PRICE_OPENAI_GPT_5_1_INPUT",
        "MODEL_PRICE_OPENAI_GPT_5_1_REASONING",
    ):
        assert key in text

    assert "Do not put API keys" in text
    assert "PROVIDER_COOLDOWN_DEFAULT_SECONDS=30" in text
