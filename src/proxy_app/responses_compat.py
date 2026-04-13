"""
OpenAI Responses API compatibility layer.

Converts between the Responses API format (/v1/responses) and the
Chat Completions format (/v1/chat/completions) used internally by
the proxy pipeline.

The Responses API is OpenAI's newer primitive used by codex-cli,
the OpenAI Python SDK, and other clients. It uses typed input/output
items instead of the flat messages array.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def convert_responses_input_to_messages(
    input_data: Any,
    instructions: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert Responses API input + instructions into Chat Completions messages.

    Handles three input modes:
      - Plain string: "Hello!" -> single user message
      - Easy message array: [{role, content}] -> direct mapping
      - Typed item array: [{type: "message", role, content: [...]}] -> full conversion
    """
    messages: List[Dict[str, Any]] = []

    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
        return messages

    if not isinstance(input_data, list):
        return messages

    for item in input_data:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type is None and "role" in item:
            # Easy message format: {role, content} — pass through directly
            role = item.get("role", "user")
            content = item.get("content", "")
            if role in ("system", "developer"):
                messages.append({"role": "system", "content": _flatten_content(content)})
            elif role == "user":
                messages.append({"role": "user", "content": _convert_input_content(content)})
            elif role == "assistant":
                msg = {"role": "assistant", "content": _flatten_output_content(content)}
                if item.get("tool_calls"):
                    msg["tool_calls"] = item["tool_calls"]
                messages.append(msg)
            elif role == "tool":
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("tool_call_id", ""),
                    "content": _flatten_content(content),
                })
            else:
                messages.append({"role": role, "content": _flatten_content(content)})
            continue

        if item_type == "message":
            role = item.get("role", "user")
            content = item.get("content", [])

            if role in ("system", "developer"):
                messages.append({"role": "system", "content": _flatten_content(content)})
            elif role == "user":
                messages.append({"role": "user", "content": _convert_input_content(content)})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": _flatten_output_content(content)})

        elif item_type == "function_call":
            # Responses function_call -> assistant message with tool_calls
            call_id = item.get("call_id", "")
            name = item.get("name", "")
            arguments = item.get("arguments", "{}")

            if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls") is not None:
                messages[-1]["tool_calls"].append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }],
                })

        elif item_type == "function_call_output":
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            if isinstance(output, dict) or isinstance(output, list):
                output = json.dumps(output)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": output,
            })

        # Reasoning items are informational — skip them for the Chat Completions pipeline

    return messages


def convert_tools_from_responses_format(
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Convert Responses API tool definitions to Chat Completions format.

    Responses format (flat):  {type:"function", name, description, parameters, strict}
    Chat Completions format (nested):  {type:"function", function:{name, description, parameters}}
    """
    if not tools:
        return None

    cc_tools = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type", "function")
        if tool_type == "function":
            name = tool.get("name", "")
            if not name:
                func = tool.get("function", {})
                if func:
                    cc_tools.append({"type": "function", "function": func})
                    continue
                continue
            cc_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            })
        elif tool_type in ("web_search", "web_search_preview", "code_interpreter"):
            pass  # Built-in tools not supported in Chat Completions passthrough
        else:
            # Unknown tool type — try passing through as-is if it has a function wrapper
            if "function" in tool:
                cc_tools.append(tool)

    return cc_tools if cc_tools else None


def convert_responses_request_to_chat(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a full Responses API request body into a Chat Completions request body.
    """
    messages = convert_responses_input_to_messages(
        request_data.get("input", ""),
        request_data.get("instructions"),
    )

    cc_request: Dict[str, Any] = {
        "model": request_data.get("model", ""),
        "messages": messages,
        "stream": request_data.get("stream", False),
    }

    tools = convert_tools_from_responses_format(request_data.get("tools"))
    if tools:
        cc_request["tools"] = tools

    tool_choice = request_data.get("tool_choice")
    if tool_choice is not None:
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            cc_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice.get("name", "")},
            }
        elif isinstance(tool_choice, str):
            cc_request["tool_choice"] = tool_choice

    if request_data.get("parallel_tool_calls") is not None:
        cc_request["parallel_tool_calls"] = request_data["parallel_tool_calls"]

    if request_data.get("max_output_tokens") is not None:
        cc_request["max_completion_tokens"] = request_data["max_output_tokens"]
    elif request_data.get("max_tokens") is not None:
        cc_request["max_completion_tokens"] = request_data["max_tokens"]

    if request_data.get("temperature") is not None:
        cc_request["temperature"] = request_data["temperature"]

    if request_data.get("top_p") is not None:
        cc_request["top_p"] = request_data["top_p"]

    # Map reasoning params
    reasoning = request_data.get("reasoning")
    if isinstance(reasoning, dict):
        if "effort" in reasoning:
            cc_request["reasoning_effort"] = reasoning["effort"]

    # Map text.format -> response_format
    text_config = request_data.get("text")
    if isinstance(text_config, dict):
        fmt = text_config.get("format")
        if isinstance(fmt, dict):
            fmt_type = fmt.get("type")
            if fmt_type == "json_schema":
                cc_request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": fmt.get("name", "response"),
                        "strict": fmt.get("strict", False),
                        "schema": fmt.get("schema", {}),
                    },
                }
            elif fmt_type == "json_object":
                cc_request["response_format"] = {"type": "json_object"}
            elif fmt_type == "text":
                cc_request["response_format"] = {"type": "text"}

    # Pass through service_tier and user if present
    if request_data.get("service_tier"):
        cc_request["service_tier"] = request_data["service_tier"]
    if request_data.get("user"):
        cc_request["user"] = request_data["user"]

    return cc_request


def build_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def build_item_id(prefix: str = "msg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def convert_chat_response_to_responses(
    cc_response: Any,
    response_id: str,
    request_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert a Chat Completions response object into a Responses API response.
    """
    resp_dict = cc_response.model_dump() if hasattr(cc_response, "model_dump") else dict(cc_response)

    output_items: List[Dict[str, Any]] = []
    status = "completed"
    finish_reason_raw = None

    choices = resp_dict.get("choices", [])
    if choices:
        choice = choices[0]
        message = choice.get("message", {})
        finish_reason_raw = choice.get("finish_reason", "stop")

        # Extract tool calls as separate function_call output items
        tool_calls = message.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                output_items.append({
                    "type": "function_call",
                    "id": build_item_id("fc"),
                    "call_id": tc.get("id", build_item_id("call")),
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                    "status": "completed",
                })

        # Extract text content as a message output item
        content = message.get("content")
        if content:
            output_items.append({
                "type": "message",
                "id": build_item_id("msg"),
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content}],
            })

        if finish_reason_raw == "length":
            status = "incomplete"

    # Build usage in Responses API format
    usage_raw = resp_dict.get("usage", {})
    usage = {
        "input_tokens": usage_raw.get("prompt_tokens", 0),
        "output_tokens": usage_raw.get("completion_tokens", 0),
        "total_tokens": usage_raw.get("total_tokens", 0),
    }

    return {
        "id": response_id,
        "object": "response",
        "created_at": resp_dict.get("created", int(time.time())),
        "model": resp_dict.get("model", request_data.get("model", "")),
        "status": status,
        "output": output_items,
        "usage": usage,
        "metadata": request_data.get("metadata", {}),
    }


class ResponsesStreamConverter:
    """
    Converts Chat Completions SSE stream chunks into Responses API SSE events.

    Chat Completions yields: data: {"choices":[{"delta":{"content":"..."}}]}
    Responses API yields: event: response.output_text.delta\ndata: {"type":"...","delta":"..."}
    """

    def __init__(self, response_id: str, model: str):
        self.response_id = response_id
        self.model = model
        self.created_at = int(time.time())
        self.started = False
        self.content_started = False
        self.message_item_id = build_item_id("msg")
        self.tool_calls: Dict[int, Dict[str, Any]] = {}
        self.tool_item_ids: Dict[int, str] = {}
        self.accumulated_content = ""
        self.accumulated_reasoning = ""
        self.usage: Optional[Dict[str, Any]] = None
        self.finish_reason: Optional[str] = None
        self.output_index = 0

    def _sse(self, event_type: str, data: Dict[str, Any]) -> str:
        data["type"] = event_type
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def _build_response_shell(self, status: str = "in_progress") -> Dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "model": self.model,
            "status": status,
            "output": [],
        }

    def convert_chunk(self, chunk_str: str) -> str:
        """Convert a single SSE chunk string from Chat Completions format to Responses API events."""
        events = ""

        if not chunk_str.strip() or not chunk_str.startswith("data:"):
            return ""

        content = chunk_str[len("data:"):].strip()
        if content == "[DONE]":
            return self._finalize()

        try:
            chunk = json.loads(content)
        except json.JSONDecodeError:
            return ""

        if not self.started:
            self.started = True
            if chunk.get("model"):
                self.model = chunk["model"]
            events += self._sse("response.created", {"response": self._build_response_shell()})
            events += self._sse("response.in_progress", {"response": self._build_response_shell()})

        choices = chunk.get("choices", [])
        if not choices:
            if chunk.get("usage"):
                self.usage = chunk["usage"]
            return events

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        if finish_reason:
            self.finish_reason = finish_reason

        # Handle reasoning_content delta
        reasoning = delta.get("reasoning_content")
        if reasoning:
            self.accumulated_reasoning += reasoning
            events += self._sse("response.reasoning_summary_text.delta", {
                "item_id": build_item_id("rs"),
                "output_index": 0,
                "summary_index": 0,
                "delta": reasoning,
            })

        # Handle text content delta
        text_content = delta.get("content")
        if text_content:
            if not self.content_started:
                self.content_started = True
                self.output_index = len(self.tool_calls)
                events += self._sse("response.output_item.added", {
                    "output_index": self.output_index,
                    "item": {
                        "type": "message",
                        "id": self.message_item_id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                })
                events += self._sse("response.content_part.added", {
                    "item_id": self.message_item_id,
                    "output_index": self.output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": ""},
                })
            self.accumulated_content += text_content
            events += self._sse("response.output_text.delta", {
                "item_id": self.message_item_id,
                "output_index": self.output_index,
                "content_index": 0,
                "delta": text_content,
            })

        # Handle tool call deltas
        tc_deltas = delta.get("tool_calls", [])
        for tc in tc_deltas:
            idx = tc.get("index", 0)

            if idx not in self.tool_calls:
                item_id = build_item_id("fc")
                self.tool_item_ids[idx] = item_id
                self.tool_calls[idx] = {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": "",
                }
                events += self._sse("response.output_item.added", {
                    "output_index": idx,
                    "item": {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": "",
                        "status": "in_progress",
                    },
                })

            func = tc.get("function", {})
            if func.get("name"):
                self.tool_calls[idx]["name"] = func["name"]
            if func.get("arguments"):
                self.tool_calls[idx]["arguments"] += func["arguments"]
                events += self._sse("response.function_call_arguments.delta", {
                    "item_id": self.tool_item_ids[idx],
                    "output_index": idx,
                    "delta": func["arguments"],
                })

            if tc.get("id"):
                self.tool_calls[idx]["id"] = tc["id"]

        # Track usage from stream_options
        if chunk.get("usage"):
            self.usage = chunk["usage"]

        return events

    def _finalize(self) -> str:
        """Emit final events: output_item.done, content_part.done, response.completed."""
        events = ""

        # Finalize tool call items
        for idx, tc in sorted(self.tool_calls.items()):
            item_id = self.tool_item_ids.get(idx, build_item_id("fc"))
            events += self._sse("response.function_call_arguments.done", {
                "item_id": item_id,
                "output_index": idx,
                "arguments": tc["arguments"],
            })
            events += self._sse("response.output_item.done", {
                "output_index": idx,
                "item": {
                    "type": "function_call",
                    "id": item_id,
                    "call_id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "status": "completed",
                },
            })

        # Finalize message content
        if self.content_started:
            events += self._sse("response.output_text.done", {
                "item_id": self.message_item_id,
                "output_index": self.output_index,
                "content_index": 0,
                "text": self.accumulated_content,
            })
            events += self._sse("response.content_part.done", {
                "item_id": self.message_item_id,
                "output_index": self.output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": self.accumulated_content},
            })
            events += self._sse("response.output_item.done", {
                "output_index": self.output_index,
                "item": {
                    "type": "message",
                    "id": self.message_item_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": self.accumulated_content}],
                },
            })

        # Build final output items for the completed response
        output_items = []
        for idx, tc in sorted(self.tool_calls.items()):
            item_id = self.tool_item_ids.get(idx, build_item_id("fc"))
            output_items.append({
                "type": "function_call",
                "id": item_id,
                "call_id": tc["id"],
                "name": tc["name"],
                "arguments": tc["arguments"],
                "status": "completed",
            })
        if self.content_started:
            output_items.append({
                "type": "message",
                "id": self.message_item_id,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": self.accumulated_content}],
            })

        # Determine status
        status = "completed"
        if self.finish_reason == "length":
            status = "incomplete"

        # Build usage
        usage = None
        if self.usage:
            usage = {
                "input_tokens": self.usage.get("prompt_tokens", 0),
                "output_tokens": self.usage.get("completion_tokens", 0),
                "total_tokens": self.usage.get("total_tokens", 0),
            }

        final_response = self._build_response_shell(status)
        final_response["output"] = output_items
        if usage:
            final_response["usage"] = usage

        event_type = "response.completed" if status == "completed" else "response.incomplete"
        events += self._sse(event_type, {"response": final_response})

        return events


# --- Content helpers ---

def _flatten_content(content: Any) -> str:
    """Flatten content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)
    return str(content) if content else ""


def _convert_input_content(content: Any) -> Any:
    """
    Convert Responses API input content parts to Chat Completions format.
    Returns either a string or a list of content parts for multimodal.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content else ""

    has_images = any(
        isinstance(p, dict) and p.get("type") in ("input_image", "image_url")
        for p in content
    )

    if not has_images:
        texts = []
        for part in content:
            if isinstance(part, dict):
                texts.append(part.get("text", ""))
            elif isinstance(part, str):
                texts.append(part)
        return "\n".join(texts) if texts else ""

    # Multimodal content with images
    cc_parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype in ("input_text", "text"):
            cc_parts.append({"type": "text", "text": part.get("text", "")})
        elif ptype in ("input_image", "image_url"):
            url = part.get("image_url", "")
            if isinstance(url, dict):
                url = url.get("url", "")
            cc_parts.append({"type": "image_url", "image_url": {"url": url}})
    return cc_parts


def _flatten_output_content(content: Any) -> str:
    """Flatten Responses API output content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type", "")
                if ptype in ("output_text", "text"):
                    parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content) if content else ""
