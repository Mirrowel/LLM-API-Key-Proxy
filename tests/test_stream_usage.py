from proxy_app.stream_usage import StreamUsageTracker


def test_stream_usage_tracker_keeps_only_minimal_metadata() -> None:
    tracker = StreamUsageTracker(model="openai/gpt-4o-mini")

    for _ in range(5000):
        tracker.ingest_chunk(
            {
                "id": "chatcmpl-123",
                "model": "openai/gpt-4o-mini",
                "created": 1700000000,
                "choices": [
                    {
                        "delta": {
                            "content": "x" * 200,
                        }
                    }
                ],
            }
        )

    tracker.ingest_chunk(
        {
            "id": "chatcmpl-123",
            "model": "openai/gpt-4o-mini",
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }
    )

    payload = tracker.build_logging_payload()
    assert payload["id"] == "chatcmpl-123"
    assert payload["model"] == "openai/gpt-4o-mini"
    assert payload["usage"]["total_tokens"] == 20
    assert payload["choices"] == []
    assert not hasattr(tracker, "response_chunks")
