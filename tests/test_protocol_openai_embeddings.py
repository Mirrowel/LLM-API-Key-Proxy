from __future__ import annotations

from rotator_library.protocols import OPERATION_EMBEDDINGS, get_protocol, list_protocols


def test_openai_embeddings_protocol_round_trip_and_usage() -> None:
    adapter = get_protocol("openai_embeddings")
    raw = {"model": "text-embedding-test", "input": ["one", "two"], "dimensions": 128, "custom": True}

    unified = adapter.parse_request(raw)
    rebuilt = adapter.build_request(unified)
    response = adapter.parse_response(
        {
            "model": "text-embedding-test",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
    )

    assert "openai_embeddings" in list_protocols()
    assert get_protocol("embeddings") is adapter
    assert adapter.supports_operation(OPERATION_EMBEDDINGS)
    assert unified.operation == OPERATION_EMBEDDINGS
    assert unified.input == ["one", "two"]
    assert rebuilt == raw
    assert response.data[0]["embedding"] == [0.1, 0.2]
    assert response.usage is not None
    assert response.usage.input_tokens == 4
    assert adapter.format_response(response)["data"] == response.data
    assert adapter.format_response(response)["usage"]["total_tokens"] == 4
