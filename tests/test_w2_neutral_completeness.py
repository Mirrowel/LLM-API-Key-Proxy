# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""W2 acceptance fixtures: neutral model completeness.

Locks the canonical capabilities added in W2 (final plan §7):

- candidate identity + per-candidate stop (D9 direct mapping / first-wins)
- refusal blocks (chat message.refusal, responses refusal parts)
- annotations/citations (chat, responses, anthropic citations, gemini grounding)
- built-in (provider-executed) tool records: never dropped silently; reject
  when the destination cannot represent them and nothing else remains
- output modality identity
"""

from __future__ import annotations

import pytest

from rotator_library.protocols import get_protocol
from rotator_library.protocols.types import ProtocolContext, ProtocolError


def _ctx(source: str, target: str) -> ProtocolContext:
    return ProtocolContext(
        provider="synthetic",
        model="model-a",
        source_protocol=source,
        target_protocol=target,
        input_protocol=source,
        client_protocol=target,
    )


def test_chat_n_choices_keep_candidate_identity_cross_protocol() -> None:
    chat = get_protocol("openai_chat")
    parsed = chat.parse_response(
        {
            "id": "chatcmpl-1",
            "model": "model-a",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "first"}, "finish_reason": "stop"},
                {"index": 1, "message": {"role": "assistant", "content": "second"}, "finish_reason": "length"},
            ],
        }
    )
    assert [m.index for m in parsed.messages] == [0, 1]
    assert [m.stop_reason for m in parsed.messages] == ["stop", "max_tokens"]
    assert parsed.modalities == ["text"]

    # Direct mapping: chat and gemini both carry alternatives natively.
    chat_out = chat.format_response(parsed, _ctx("openai_chat", "openai_chat"))
    assert [(c["index"], c["message"]["content"], c["finish_reason"]) for c in chat_out["choices"]] == [
        (0, "first", "stop"),
        (1, "second", "length"),
    ]

    gemini = get_protocol("gemini")
    gemini_out = gemini.format_response(parsed, _ctx("openai_chat", "gemini"))
    assert [(c["index"], c["finishReason"]) for c in gemini_out["candidates"]] == [
        (0, "STOP"),
        (1, "MAX_TOKENS"),
    ]
    texts = [p["text"] for c in gemini_out["candidates"] for p in c["content"]["parts"]]
    assert texts == ["first", "second"]


def test_multiple_candidates_first_wins_for_anthropic_with_warning() -> None:
    chat = get_protocol("openai_chat")
    anthropic = get_protocol("anthropic_messages")
    parsed = chat.parse_response(
        {
            "id": "chatcmpl-1",
            "model": "model-a",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "first"}, "finish_reason": "stop"},
                {"index": 1, "message": {"role": "assistant", "content": "second"}, "finish_reason": "stop"},
            ],
        }
    )

    out = anthropic.format_response(parsed, _ctx("openai_chat", "anthropic_messages"))

    assert out["content"][0]["text"] == "first"
    assert "second" not in str(out["content"])
    assert any(w.code == "candidates_first_wins" for w in parsed.warnings)


def test_chat_refusal_maps_to_anthropic_refusal_stop() -> None:
    chat = get_protocol("openai_chat")
    anthropic = get_protocol("anthropic_messages")
    parsed = chat.parse_response(
        {
            "id": "chatcmpl-1",
            "model": "model-a",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": None, "refusal": "cannot help"}, "finish_reason": "content_filter"},
            ],
        }
    )
    assert any(block.type == "refusal" for block in parsed.messages[0].content)

    out = anthropic.format_response(parsed, _ctx("openai_chat", "anthropic_messages"))

    assert out["stop_reason"] == "refusal"
    assert any(block.get("text") == "cannot help" for block in out["content"])


def test_responses_refusal_round_trips_and_maps_to_chat() -> None:
    responses = get_protocol("responses")
    chat = get_protocol("openai_chat")
    parsed = responses.parse_response(
        {
            "id": "resp_1",
            "object": "response",
            "model": "model-a",
            "status": "completed",
            "output": [
                {
                    "id": "msg_0",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "refusal", "refusal": "no can do"}],
                }
            ],
        }
    )
    assert any(block.type == "refusal" for block in parsed.messages[0].content)

    same = responses.format_response(parsed, _ctx("responses", "responses"))
    assert same["output"][0]["content"][0]["type"] == "refusal"

    chat_out = chat.format_response(parsed, _ctx("responses", "openai_chat"))
    assert chat_out["choices"][0]["message"]["refusal"] == "no can do"


def test_annotations_flow_between_chat_responses_anthropic_gemini() -> None:
    responses = get_protocol("responses")
    chat = get_protocol("openai_chat")
    parsed = responses.parse_response(
        {
            "id": "resp_1",
            "object": "response",
            "model": "model-a",
            "status": "completed",
            "output": [
                {
                    "id": "msg_0",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "see this",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url_citation": {"url": "https://example.test", "title": "Example", "start_index": 0, "end_index": 3},
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )
    block = parsed.messages[0].content[0]
    assert block.annotations[0].url == "https://example.test"

    chat_out = chat.format_response(parsed, _ctx("responses", "openai_chat"))
    assert chat_out["choices"][0]["message"]["annotations"][0]["url_citation"]["url"] == "https://example.test"

    same = responses.format_response(parsed, _ctx("responses", "responses"))
    assert same["output"][0]["content"][0]["annotations"][0]["url_citation"]["url"] == "https://example.test"

    anthropic = get_protocol("anthropic_messages")
    anth_out = anthropic.format_response(parsed, _ctx("responses", "anthropic_messages"))
    assert anth_out["content"][0]["text"] == "see this"
    assert any(w.code == "annotations_dropped" for w in parsed.warnings)


def test_gemini_grounding_becomes_annotations() -> None:
    gemini = get_protocol("gemini")
    chat = get_protocol("openai_chat")
    parsed = gemini.parse_response(
        {
            "responseId": "resp_g",
            "modelVersion": "model-a",
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "grounded answer"}]},
                    "finishReason": "STOP",
                    "groundingMetadata": {
                        "groundingChunks": [{"web": {"uri": "https://ground.test", "title": "Ground"}}],
                        "webSearchQueries": ["test"],
                    },
                }
            ],
        }
    )
    block = parsed.messages[0].content[0]
    assert block.annotations[0].url == "https://ground.test"
    assert parsed.messages[0].index == 0
    assert parsed.messages[0].stop_reason == "stop"

    chat_out = chat.format_response(parsed, _ctx("gemini", "openai_chat"))
    assert chat_out["choices"][0]["message"]["annotations"][0]["url_citation"]["url"] == "https://ground.test"


def test_anthropic_citations_parse_to_annotations() -> None:
    anthropic = get_protocol("anthropic_messages")
    parsed = anthropic.parse_response(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "model-a",
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": "cited",
                    "citations": [{"type": "web_search_result_location", "url": "https://cite.test", "title": "Cite", "cited_text": "quoted"}],
                }
            ],
        }
    )
    assert parsed.messages[0].content[0].annotations[0].url == "https://cite.test"


def test_builtin_tool_round_trips_and_rejects_empty_successes() -> None:
    responses = get_protocol("responses")
    parsed = responses.parse_response(
        {
            "id": "resp_1",
            "object": "response",
            "model": "model-a",
            "status": "completed",
            "output": [
                {"type": "web_search_call", "id": "ws_1", "call_id": "ws_1", "status": "completed", "results": [{"title": "hit"}]},
                {
                    "id": "msg_0",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "found it"}],
                },
            ],
        }
    )

    # Same-protocol: the native item round-trips.
    same = responses.format_response(parsed, _ctx("responses", "responses"))
    assert [item["type"] for item in same["output"]] == ["web_search_call", "message"]

    # Cross-protocol WITH text: the answer survives, the record degrades with a warning.
    chat = get_protocol("openai_chat")
    chat_out = chat.format_response(parsed, _ctx("responses", "openai_chat"))
    content = chat_out["choices"][0]["message"]["content"]
    text = content[0]["text"] if isinstance(content, list) else content
    assert text == "found it"

    # Record-only response: honest rejection, never an empty success.
    record_only = responses.parse_response(
        {
            "id": "resp_2",
            "object": "response",
            "model": "model-a",
            "status": "completed",
            "output": [{"type": "web_search_call", "id": "ws_1", "call_id": "ws_1", "status": "completed"}],
        }
    )
    with pytest.raises(ProtocolError):
        chat.format_response(record_only, _ctx("responses", "openai_chat"))
    anthropic = get_protocol("anthropic_messages")
    with pytest.raises(ProtocolError):
        anthropic.format_response(record_only, _ctx("responses", "anthropic_messages"))


def test_responses_cross_protocol_output_rebuilds_builtin_item() -> None:
    chat = get_protocol("openai_chat")
    responses = get_protocol("responses")
    parsed = chat.parse_response(
        {
            "id": "chatcmpl-1",
            "model": "model-a",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"},
            ],
        }
    )
    # Attach a builtin record to the canonical message (provider-side origin).
    from rotator_library.protocols.types import BuiltinToolCall, ContentBlock, UnifiedMessage

    parsed.messages.append(
        UnifiedMessage(
            role="assistant",
            content=[
                ContentBlock(
                    type="builtin_tool",
                    builtin_tool=BuiltinToolCall(kind="web_search", call_id="ws_9", status="completed", output=[{"title": "found"}]),
                )
            ],
        )
    )

    out = responses.format_response(parsed, _ctx("openai_chat", "responses"))
    types = [item["type"] for item in out["output"]]
    assert "web_search_call" in types
    assert "message" in types


def test_gemini_target_refusal_text_and_builtin_rejection() -> None:
    gemini = get_protocol("gemini")
    responses = get_protocol("responses")
    chat = get_protocol("openai_chat")
    from rotator_library.protocols.types import ProtocolError

    refusal = chat.parse_response(
        {
            "id": "chatcmpl-1",
            "model": "model-a",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "refusal": "no"}, "finish_reason": "content_filter"}],
        }
    )
    out = gemini.format_response(refusal, _ctx("openai_chat", "gemini"))
    parts = out["candidates"][0]["content"]["parts"]
    assert parts == [{"text": "no"}]

    record_only = responses.parse_response(
        {
            "id": "resp_2",
            "object": "response",
            "model": "model-a",
            "status": "completed",
            "output": [{"type": "web_search_call", "id": "ws_1", "call_id": "ws_1", "status": "completed"}],
        }
    )
    with pytest.raises(ProtocolError):
        gemini.format_response(record_only, _ctx("responses", "gemini"))

    record_with_text = responses.parse_response(
        {
            "id": "resp_3",
            "object": "response",
            "model": "model-a",
            "status": "completed",
            "output": [
                {"type": "web_search_call", "id": "ws_1", "call_id": "ws_1", "status": "completed"},
                {"id": "msg_0", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "kept"}]},
            ],
        }
    )
    out = gemini.format_response(record_with_text, _ctx("responses", "gemini"))
    assert out["candidates"][0]["content"]["parts"] == [{"text": "kept"}]
    assert any(w.code == "builtin_tool_dropped" for w in record_with_text.warnings)


def test_gemini_target_annotations_dropped_with_warning() -> None:
    gemini = get_protocol("gemini")
    responses = get_protocol("responses")
    parsed = responses.parse_response(
        {
            "id": "resp_1",
            "object": "response",
            "model": "model-a",
            "status": "completed",
            "output": [
                {
                    "id": "msg_0",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "cited",
                            "annotations": [{"type": "url_citation", "url_citation": {"url": "https://x.test"}}],
                        }
                    ],
                }
            ],
        }
    )
    out = gemini.format_response(parsed, _ctx("responses", "gemini"))
    assert out["candidates"][0]["content"]["parts"] == [{"text": "cited"}]
    assert any(w.code == "annotations_dropped" for w in parsed.warnings)


def test_responses_target_candidates_first_wins_not_concatenated() -> None:
    chat = get_protocol("openai_chat")
    responses = get_protocol("responses")
    parsed = chat.parse_response(
        {
            "id": "chatcmpl-1",
            "model": "model-a",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "first"}, "finish_reason": "stop"},
                {"index": 1, "message": {"role": "assistant", "content": "second"}, "finish_reason": "stop"},
            ],
        }
    )
    out = responses.format_response(parsed, _ctx("openai_chat", "responses"))
    texts = [
        block.get("text")
        for item in out["output"]
        if item.get("type") == "message"
        for block in item.get("content", [])
        if isinstance(block, dict)
    ]
    assert texts == ["first"]
    assert any(w.code == "candidates_first_wins" for w in parsed.warnings)


def test_responses_item_oriented_output_still_coalesces_without_warning() -> None:
    responses = get_protocol("responses")
    anthropic = get_protocol("anthropic_messages")
    parsed = responses.parse_response(
        {
            "id": "resp_1",
            "object": "response",
            "model": "model-a",
            "status": "completed",
            "output": [
                {"id": "rs_0", "type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
                {"id": "msg_0", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "answer"}]},
                {"id": "fc_0", "type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"},
            ],
        }
    )
    out = anthropic.format_response(parsed, _ctx("responses", "anthropic_messages"))
    block_types = [block["type"] for block in out["content"]]
    assert block_types == ["thinking", "text", "tool_use"]
    assert not any(w.code == "candidates_first_wins" for w in parsed.warnings)


def test_anthropic_citations_round_trip_same_protocol() -> None:
    anthropic = get_protocol("anthropic_messages")
    parsed = anthropic.parse_response(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "model-a",
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": "cited",
                    "citations": [{"type": "web_search_result_location", "url": "https://cite.test", "title": "Cite", "cited_text": "quoted"}],
                }
            ],
        }
    )
    out = anthropic.format_response(parsed, _ctx("anthropic_messages", "anthropic_messages"))
    assert out["content"][0]["citations"][0]["url"] == "https://cite.test"


def test_gemini_n_choices_round_trip_to_chat() -> None:
    gemini = get_protocol("gemini")
    chat = get_protocol("openai_chat")
    parsed = gemini.parse_response(
        {
            "responseId": "resp_g",
            "modelVersion": "model-a",
            "candidates": [
                {"index": 0, "content": {"role": "model", "parts": [{"text": "alpha"}]}, "finishReason": "STOP"},
                {"index": 1, "content": {"role": "model", "parts": [{"text": "beta"}]}, "finishReason": "MAX_TOKENS"},
            ],
        }
    )
    out = chat.format_response(parsed, _ctx("gemini", "openai_chat"))
    assert [(c["index"], c["message"]["content"], c["finish_reason"]) for c in out["choices"]] == [
        (0, "alpha", "stop"),
        (1, "beta", "length"),
    ]


def test_new_fields_serialize_round_trip() -> None:
    from rotator_library.protocols.types import serialize_value

    gemini = get_protocol("gemini")
    parsed = gemini.parse_response(
        {
            "responseId": "resp_g",
            "modelVersion": "model-a",
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "grounded"}]},
                    "finishReason": "STOP",
                    "groundingMetadata": {"groundingChunks": [{"web": {"uri": "https://g.test", "title": "G"}}]},
                }
            ],
        }
    )
    serialized = serialize_value(parsed)
    message = serialized["messages"][0]
    assert message["index"] == 0
    assert message["stop_reason"] == "stop"
    assert message["content"][0]["annotations"][0]["url"] == "https://g.test"
    assert serialized["modalities"] == ["text"]

    responses = get_protocol("responses")
    builtin = responses.parse_response(
        {
            "id": "resp_1",
            "object": "response",
            "model": "model-a",
            "status": "completed",
            "output": [{"type": "web_search_call", "id": "ws_1", "call_id": "ws_1", "status": "completed", "results": [{"title": "t"}]}],
        }
    )
    serialized_builtin = serialize_value(builtin)
    block = serialized_builtin["messages"][0]["content"][0]
    assert block["type"] == "builtin_tool"
    assert block["builtin_tool"]["kind"] == "web_search"
    assert block["index"] == 0 or block["index"] is None


def test_chat_stream_refusal_delta_parses() -> None:
    chat = get_protocol("openai_chat")
    event = chat.parse_stream_event({"choices": [{"index": 0, "delta": {"refusal": "no way"}}]})
    assert event.delta is not None
    assert any(block.type == "refusal" for block in event.delta.content)
