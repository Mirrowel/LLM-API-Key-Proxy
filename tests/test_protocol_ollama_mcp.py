from __future__ import annotations

from rotator_library.protocols import OPERATION_EMBEDDINGS, OPERATION_MCP, OPERATION_OLLAMA_CHAT, OPERATION_OLLAMA_GENERATE, get_protocol


def test_ollama_chat_generate_and_stream_shapes() -> None:
    adapter = get_protocol("ollama")
    chat = adapter.parse_request({"model": "llama3", "messages": [{"role": "user", "content": "hi"}], "stream": True})
    generate = adapter.parse_request({"model": "llama3", "prompt": "write", "options": {"temperature": 0.1}})
    embeddings = adapter.parse_request({"model": "llama3", "operation": OPERATION_EMBEDDINGS, "prompt": "embed this"})
    final_chat = adapter.parse_response({"model": "llama3", "message": {"role": "assistant", "content": "hello"}, "done": True})
    chunk = adapter.parse_stream_event('{"model":"llama3","response":"he","done":false,"eval_count":2}')

    assert chat.operation == OPERATION_OLLAMA_CHAT
    assert adapter.build_request(chat)["messages"][0]["content"] == "hi"
    assert generate.operation == OPERATION_OLLAMA_GENERATE
    assert adapter.build_request(generate)["prompt"] == "write"
    assert embeddings.operation == OPERATION_EMBEDDINGS
    assert adapter.build_request(embeddings)["prompt"] == "embed this"
    assert final_chat.messages[0].content[0].text == "hello"
    assert chunk.delta is not None
    assert chunk.delta.content[0].text == "he"
    assert chunk.usage is not None
    assert chunk.usage.output_tokens == 2


def test_mcp_jsonrpc_round_trip_and_error_preservation() -> None:
    adapter = get_protocol("mcp")
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "lookup"}}
    unified = adapter.parse_request(request)
    rebuilt = adapter.build_request(unified)
    response = adapter.parse_response({"jsonrpc": "2.0", "id": 1, "result": {"content": []}})
    error = adapter.parse_response({"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "failed"}})

    assert unified.operation == OPERATION_MCP
    assert rebuilt == request
    assert adapter.format_response(response)["result"] == {"content": []}
    assert adapter.format_response(error)["error"]["message"] == "failed"


def test_mcp_preserves_notifications_and_falsey_params() -> None:
    adapter = get_protocol("mcp")
    notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    false_params = {"jsonrpc": "2.0", "id": 0, "method": "tools/call", "params": False}

    assert adapter.build_request(adapter.parse_request(notification)) == notification
    assert adapter.build_request(adapter.parse_request(false_params)) == false_params
