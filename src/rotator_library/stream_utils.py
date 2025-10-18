import time
import logging
from typing import Any, Dict, List, Optional

import litellm

lib_logger = logging.getLogger('rotator_library')


def _to_dict(chunk: Any) -> Dict[str, Any]:
    """Best-effort conversion of a litellm.ModelResponse (or similar) to a dict."""
    try:
        if isinstance(chunk, dict):
            return chunk
        # pydantic v2
        if hasattr(chunk, "model_dump"):
            return chunk.model_dump()
        # pydantic v1
        if hasattr(chunk, "dict"):
            return chunk.dict()
        # Fallback to JSON then parse
        if hasattr(chunk, "model_dump_json"):
            import json
            return json.loads(chunk.model_dump_json())
    except Exception as e:
        lib_logger.warning(f"Failed to coerce chunk to dict: {e}")
    # Last resort
    return {}


def assemble_stream_chunks_to_response(
    chunks: List[Any],
    default_model: Optional[str] = None,
) -> litellm.ModelResponse:
    """
    Assemble a list of streaming chunks (OpenAI-chat-completion-like) into a single
    non-streaming completion response compatible with litellm.ModelResponse.

    - Concatenates delta.content parts
    - Concatenates delta.reasoning_content parts into message.reasoning_content
    - Merges tool_calls (best-effort)
    - Picks the last non-null finish_reason
    - Uses the last provided usage block
    """
    full_text: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None

    # Merge tool call deltas by index
    def ensure_tool_call_slot(idx: int) -> Dict[str, Any]:
        while len(tool_calls) <= idx:
            tool_calls.append({
                "index": len(tool_calls),
                "id": None,
                "type": "function",
                "function": {"name": None, "arguments": ""}
            })
        return tool_calls[idx]

    for raw in chunks:
        data = _to_dict(raw)
        if not data:
            continue

        # Track model if present
        if data.get("model"):
            model = data["model"]

        # Track usage (last occurrence wins; usually cumulative on final chunk)
        if "usage" in data and isinstance(data["usage"], dict):
            usage = data["usage"]

        # Merge choices/deltas
        for choice in data.get("choices", []) or []:
            delta = choice.get("delta", {}) or {}
            # Aggregate textual content
            text_piece = delta.get("content")
            if isinstance(text_piece, str) and text_piece:
                full_text.append(text_piece)

            # Aggregate reasoning content if present
            reasoning_piece = delta.get("reasoning_content")
            if isinstance(reasoning_piece, str) and reasoning_piece:
                reasoning_parts.append(reasoning_piece)

            # Merge tool_calls (best-effort OpenAI-compatible structure)
            if "tool_calls" in delta and isinstance(delta["tool_calls"], list):
                for idx, call_delta in enumerate(delta["tool_calls"]):
                    slot = ensure_tool_call_slot(idx)
                    if call_delta.get("id"):
                        slot["id"] = call_delta["id"]
                    # only function type supported here
                    fn = call_delta.get("function", {}) or {}
                    if fn.get("name"):
                        slot.setdefault("function", {})["name"] = fn["name"]
                    if fn.get("arguments"):
                        # Append content for streaming partial args
                        slot.setdefault("function", {}).setdefault("arguments", "")
                        slot["function"]["arguments"] += fn["arguments"]

            # Track finish reason
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(full_text),
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        # Remove empty names/ids if never set
        for t in tool_calls:
            if t.get("id") is None:
                t.pop("id", None)
            fn = t.get("function", {})
            if fn.get("name") is None:
                fn.pop("name", None)
        message["tool_calls"] = tool_calls

    response_payload: Dict[str, Any] = {
        "id": f"chatcmpl-assembled-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or (default_model or "unknown-model"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or "stop",
            }
        ],
    }
    if usage:
        response_payload["usage"] = usage

    return litellm.ModelResponse(**response_payload)
