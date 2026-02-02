# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from .kiro_utils import (
    FAKE_REASONING_ENABLED,
    FAKE_REASONING_HANDLING,
    FAKE_REASONING_INITIAL_BUFFER_SIZE,
    FAKE_REASONING_OPEN_TAGS,
    FIRST_TOKEN_MAX_RETRIES,
    FIRST_TOKEN_TIMEOUT,
    generate_completion_id,
    generate_tool_call_id,
)


lib_logger = logging.getLogger("rotator_library")


def find_matching_brace(text: str, start_pos: int) -> int:
    if start_pos >= len(text) or text[start_pos] != "{":
        return -1
    brace_count = 0
    in_string = False
    escape_next = False
    for i in range(start_pos, len(text)):
        char = text[i]
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    return i
    return -1


def parse_bracket_tool_calls(response_text: str) -> List[Dict[str, Any]]:
    if not response_text or "[Called" not in response_text:
        return []
    tool_calls = []
    pattern = r"\[Called\s+(\w+)\s+with\s+args:\s*"
    import re

    for match in re.finditer(pattern, response_text, re.IGNORECASE):
        func_name = match.group(1)
        args_start = match.end()
        json_start = response_text.find("{", args_start)
        if json_start == -1:
            continue
        json_end = find_matching_brace(response_text, json_start)
        if json_end == -1:
            continue
        json_str = response_text[json_start : json_end + 1]
        try:
            args = json.loads(json_str)
            tool_calls.append(
                {
                    "id": generate_tool_call_id(),
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args),
                    },
                }
            )
        except json.JSONDecodeError:
            lib_logger.warning(
                f"Failed to parse tool call arguments: {json_str[:100]}"
            )
    return tool_calls


def deduplicate_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for tc in tool_calls:
        tc_id = tc.get("id", "")
        if not tc_id:
            continue
        existing = by_id.get(tc_id)
        if existing is None:
            by_id[tc_id] = tc
        else:
            existing_args = existing.get("function", {}).get("arguments", "{}")
            current_args = tc.get("function", {}).get("arguments", "{}")
            if current_args != "{}" and (
                existing_args == "{}" or len(current_args) > len(existing_args)
            ):
                by_id[tc_id] = tc

    result_with_id = list(by_id.values())
    result_without_id = [tc for tc in tool_calls if not tc.get("id")]

    seen = set()
    unique = []
    for tc in result_with_id + result_without_id:
        func = tc.get("function") or {}
        func_name = func.get("name") or ""
        func_args = func.get("arguments") or "{}"
        key = f"{func_name}-{func_args}"
        if key not in seen:
            seen.add(key)
            unique.append(tc)
    return unique


class AwsEventStreamParser:
    EVENT_PATTERNS = [
        ('{"content":', "content"),
        ('{"name":', "tool_start"),
        ('{"input":', "tool_input"),
        ('{"stop":', "tool_stop"),
        ('{"usage":', "usage"),
        ('{"contextUsagePercentage":', "context_usage"),
    ]

    def __init__(self) -> None:
        self.buffer = ""
        self.last_content: Optional[str] = None
        self.current_tool_call: Optional[Dict[str, Any]] = None
        self.tool_calls: List[Dict[str, Any]] = []

    def feed(self, chunk: bytes) -> List[Dict[str, Any]]:
        try:
            self.buffer += chunk.decode("utf-8", errors="ignore")
        except Exception:
            return []

        events = []
        while True:
            earliest_pos = -1
            earliest_type = None
            for pattern, event_type in self.EVENT_PATTERNS:
                pos = self.buffer.find(pattern)
                if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
                    earliest_pos = pos
                    earliest_type = event_type
            if earliest_pos == -1:
                break

            json_end = find_matching_brace(self.buffer, earliest_pos)
            if json_end == -1:
                break

            json_str = self.buffer[earliest_pos : json_end + 1]
            self.buffer = self.buffer[json_end + 1 :]
            try:
                data = json.loads(json_str)
                event = self._process_event(data, earliest_type)
                if event:
                    events.append(event)
            except json.JSONDecodeError:
                lib_logger.warning(f"Failed to parse JSON: {json_str[:100]}")

        return events

    def _process_event(self, data: dict, event_type: str) -> Optional[Dict[str, Any]]:
        if event_type == "content":
            return self._process_content_event(data)
        if event_type == "tool_start":
            return self._process_tool_start_event(data)
        if event_type == "tool_input":
            return self._process_tool_input_event(data)
        if event_type == "tool_stop":
            return self._process_tool_stop_event(data)
        if event_type == "usage":
            return {"type": "usage", "data": data.get("usage", 0)}
        if event_type == "context_usage":
            return {"type": "context_usage", "data": data.get("contextUsagePercentage", 0)}
        return None

    def _process_content_event(self, data: dict) -> Optional[Dict[str, Any]]:
        content = data.get("content", "")
        if data.get("followupPrompt"):
            return None
        if content == self.last_content:
            return None
        self.last_content = content
        return {"type": "content", "data": content}

    def _process_tool_start_event(self, data: dict) -> Optional[Dict[str, Any]]:
        if self.current_tool_call:
            self._finalize_tool_call()

        input_data = data.get("input", "")
        if isinstance(input_data, dict):
            input_str = json.dumps(input_data)
        else:
            input_str = str(input_data) if input_data else ""

        self.current_tool_call = {
            "id": data.get("toolUseId", generate_tool_call_id()),
            "type": "function",
            "function": {
                "name": data.get("name", ""),
                "arguments": input_str,
            },
        }

        if data.get("stop"):
            self._finalize_tool_call()
        return None

    def _process_tool_input_event(self, data: dict) -> Optional[Dict[str, Any]]:
        if self.current_tool_call:
            input_data = data.get("input", "")
            if isinstance(input_data, dict):
                input_str = json.dumps(input_data)
            else:
                input_str = str(input_data) if input_data else ""
            self.current_tool_call["function"]["arguments"] += input_str
        return None

    def _process_tool_stop_event(self, data: dict) -> Optional[Dict[str, Any]]:
        if self.current_tool_call and data.get("stop"):
            self._finalize_tool_call()
        return None

    def _finalize_tool_call(self) -> None:
        if not self.current_tool_call:
            return
        args = self.current_tool_call["function"].get("arguments", "")
        if isinstance(args, str):
            if args.strip():
                try:
                    parsed = json.loads(args)
                    self.current_tool_call["function"]["arguments"] = json.dumps(parsed)
                except json.JSONDecodeError:
                    self.current_tool_call["function"]["arguments"] = "{}"
            else:
                self.current_tool_call["function"]["arguments"] = "{}"
        elif isinstance(args, dict):
            self.current_tool_call["function"]["arguments"] = json.dumps(args)
        else:
            self.current_tool_call["function"]["arguments"] = "{}"
        self.tool_calls.append(self.current_tool_call)
        self.current_tool_call = None

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        if self.current_tool_call:
            self._finalize_tool_call()
        return deduplicate_tool_calls(self.tool_calls)


class ParserState(IntEnum):
    PRE_CONTENT = 0
    IN_THINKING = 1
    STREAMING = 2


@dataclass
class ThinkingParseResult:
    thinking_content: Optional[str] = None
    regular_content: Optional[str] = None
    is_first_thinking_chunk: bool = False
    is_last_thinking_chunk: bool = False
    state_changed: bool = False


class ThinkingParser:
    def __init__(
        self,
        handling_mode: Optional[str] = None,
        open_tags: Optional[List[str]] = None,
        initial_buffer_size: int = FAKE_REASONING_INITIAL_BUFFER_SIZE,
    ) -> None:
        self.handling_mode = handling_mode or FAKE_REASONING_HANDLING
        self.open_tags = open_tags or FAKE_REASONING_OPEN_TAGS
        self.initial_buffer_size = initial_buffer_size
        self.max_tag_length = max(len(tag) for tag in self.open_tags) * 2

        self.state = ParserState.PRE_CONTENT
        self.initial_buffer = ""
        self.thinking_buffer = ""
        self.open_tag: Optional[str] = None
        self.close_tag: Optional[str] = None
        self.is_first_thinking_chunk = True
        self._thinking_block_found = False

    def feed(self, content: str) -> ThinkingParseResult:
        result = ThinkingParseResult()
        if not content:
            return result

        if self.state == ParserState.PRE_CONTENT:
            result = self._handle_pre_content(content)

        if self.state == ParserState.IN_THINKING and not result.state_changed:
            result = self._handle_in_thinking(content)

        if self.state == ParserState.STREAMING and not result.state_changed:
            result.regular_content = content

        return result

    def _handle_pre_content(self, content: str) -> ThinkingParseResult:
        result = ThinkingParseResult()
        self.initial_buffer += content
        stripped = self.initial_buffer.lstrip()

        for tag in self.open_tags:
            if stripped.startswith(tag):
                self.state = ParserState.IN_THINKING
                self.open_tag = tag
                self.close_tag = f"</{tag[1:]}"
                self._thinking_block_found = True
                result.state_changed = True

                content_after_tag = stripped[len(tag) :]
                self.thinking_buffer = content_after_tag
                self.initial_buffer = ""

                thinking_result = self._process_thinking_buffer()
                if thinking_result.thinking_content:
                    result.thinking_content = thinking_result.thinking_content
                    result.is_first_thinking_chunk = (
                        thinking_result.is_first_thinking_chunk
                    )
                if thinking_result.is_last_thinking_chunk:
                    result.is_last_thinking_chunk = True
                if thinking_result.regular_content:
                    result.regular_content = thinking_result.regular_content
                return result

        for tag in self.open_tags:
            if tag.startswith(stripped) and len(stripped) < len(tag):
                return result

        if len(self.initial_buffer) > self.initial_buffer_size:
            self.state = ParserState.STREAMING
            result.state_changed = True
            result.regular_content = self.initial_buffer
            self.initial_buffer = ""
        return result

    def _handle_in_thinking(self, content: str) -> ThinkingParseResult:
        self.thinking_buffer += content
        return self._process_thinking_buffer()

    def _process_thinking_buffer(self) -> ThinkingParseResult:
        result = ThinkingParseResult()
        if not self.close_tag:
            return result

        if self.close_tag in self.thinking_buffer:
            idx = self.thinking_buffer.find(self.close_tag)
            thinking_content = self.thinking_buffer[:idx]
            after_tag = self.thinking_buffer[idx + len(self.close_tag) :]
            if thinking_content:
                result.thinking_content = thinking_content
                result.is_first_thinking_chunk = self.is_first_thinking_chunk
                self.is_first_thinking_chunk = False
            result.is_last_thinking_chunk = True
            self.state = ParserState.STREAMING
            result.state_changed = True
            self.thinking_buffer = ""
            if after_tag:
                stripped_after = after_tag.lstrip()
                if stripped_after:
                    result.regular_content = stripped_after
            return result

        if len(self.thinking_buffer) > self.max_tag_length:
            send_part = self.thinking_buffer[: -self.max_tag_length]
            self.thinking_buffer = self.thinking_buffer[-self.max_tag_length :]
            result.thinking_content = send_part
            result.is_first_thinking_chunk = self.is_first_thinking_chunk
            self.is_first_thinking_chunk = False
        return result

    def finalize(self) -> ThinkingParseResult:
        result = ThinkingParseResult()
        if self.thinking_buffer:
            if self.state == ParserState.IN_THINKING:
                result.thinking_content = self.thinking_buffer
                result.is_first_thinking_chunk = self.is_first_thinking_chunk
                result.is_last_thinking_chunk = True
            else:
                result.regular_content = self.thinking_buffer
            self.thinking_buffer = ""
        if self.initial_buffer:
            result.regular_content = (result.regular_content or "") + self.initial_buffer
            self.initial_buffer = ""
        return result

    @property
    def found_thinking_block(self) -> bool:
        return self._thinking_block_found

    def process_for_output(
        self, thinking_content: Optional[str], is_first: bool, is_last: bool
    ) -> Optional[str]:
        if not thinking_content:
            return None
        if self.handling_mode == "remove":
            return None
        if self.handling_mode == "pass":
            prefix = self.open_tag if is_first and self.open_tag else ""
            suffix = self.close_tag if is_last and self.close_tag else ""
            return f"{prefix}{thinking_content}{suffix}"
        if self.handling_mode == "strip_tags":
            return thinking_content
        return thinking_content


@dataclass
class KiroEvent:
    type: str
    content: Optional[str] = None
    thinking_content: Optional[str] = None
    tool_use: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None
    context_usage_percentage: Optional[float] = None
    is_first_thinking_chunk: bool = False
    is_last_thinking_chunk: bool = False


@dataclass
class StreamResult:
    content: str = ""
    thinking_content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None
    context_usage_percentage: Optional[float] = None


class FirstTokenTimeoutError(Exception):
    pass


async def parse_kiro_stream(
    response: httpx.Response,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    enable_thinking_parser: bool = True,
) -> AsyncGenerator[KiroEvent, None]:
    parser = AwsEventStreamParser()
    thinking_parser: Optional[ThinkingParser] = None
    if FAKE_REASONING_ENABLED and enable_thinking_parser:
        thinking_parser = ThinkingParser(handling_mode=FAKE_REASONING_HANDLING)

    byte_iterator = response.aiter_bytes()
    try:
        first_byte_chunk = await asyncio.wait_for(
            byte_iterator.__anext__(), timeout=first_token_timeout
        )
    except asyncio.TimeoutError:
        raise FirstTokenTimeoutError(
            f"No response within {first_token_timeout} seconds"
        )
    except StopAsyncIteration:
        return

    async for event in _process_chunk(parser, first_byte_chunk, thinking_parser):
        yield event

    async for chunk in byte_iterator:
        async for event in _process_chunk(parser, chunk, thinking_parser):
            yield event

    if thinking_parser:
        final_result = thinking_parser.finalize()
        if final_result.thinking_content:
            processed_thinking = thinking_parser.process_for_output(
                final_result.thinking_content,
                final_result.is_first_thinking_chunk,
                final_result.is_last_thinking_chunk,
            )
            if processed_thinking:
                yield KiroEvent(
                    type="thinking",
                    thinking_content=processed_thinking,
                    is_first_thinking_chunk=final_result.is_first_thinking_chunk,
                    is_last_thinking_chunk=final_result.is_last_thinking_chunk,
                )
        if final_result.regular_content:
            yield KiroEvent(type="content", content=final_result.regular_content)

    for tc in parser.get_tool_calls():
        yield KiroEvent(type="tool_use", tool_use=tc)


async def _process_chunk(
    parser: AwsEventStreamParser,
    chunk: bytes,
    thinking_parser: Optional[ThinkingParser],
) -> AsyncGenerator[KiroEvent, None]:
    events = parser.feed(chunk)
    for event in events:
        if event["type"] == "content":
            content = event["data"]
            if thinking_parser:
                parse_result = thinking_parser.feed(content)
                if parse_result.thinking_content:
                    processed_thinking = thinking_parser.process_for_output(
                        parse_result.thinking_content,
                        parse_result.is_first_thinking_chunk,
                        parse_result.is_last_thinking_chunk,
                    )
                    if processed_thinking:
                        yield KiroEvent(
                            type="thinking",
                            thinking_content=processed_thinking,
                            is_first_thinking_chunk=parse_result.is_first_thinking_chunk,
                            is_last_thinking_chunk=parse_result.is_last_thinking_chunk,
                        )
                if parse_result.regular_content:
                    yield KiroEvent(type="content", content=parse_result.regular_content)
            else:
                yield KiroEvent(type="content", content=content)
        elif event["type"] == "usage":
            yield KiroEvent(type="usage", usage=event["data"])
        elif event["type"] == "context_usage":
            yield KiroEvent(
                type="context_usage", context_usage_percentage=event["data"]
            )


async def collect_stream_to_result(
    response: httpx.Response,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    enable_thinking_parser: bool = True,
) -> StreamResult:
    result = StreamResult()
    full_content_for_bracket_tools = ""
    async for event in parse_kiro_stream(
        response, first_token_timeout, enable_thinking_parser
    ):
        if event.type == "content" and event.content:
            result.content += event.content
            full_content_for_bracket_tools += event.content
        elif event.type == "thinking" and event.thinking_content:
            result.thinking_content += event.thinking_content
            full_content_for_bracket_tools += event.thinking_content
        elif event.type == "tool_use" and event.tool_use:
            result.tool_calls.append(event.tool_use)
        elif event.type == "usage" and event.usage:
            result.usage = event.usage
        elif event.type == "context_usage" and event.context_usage_percentage is not None:
            result.context_usage_percentage = event.context_usage_percentage

    bracket_tool_calls = parse_bracket_tool_calls(full_content_for_bracket_tools)
    if bracket_tool_calls:
        result.tool_calls = deduplicate_tool_calls(
            result.tool_calls + bracket_tool_calls
        )
    return result


async def stream_kiro_to_openai_chunks(
    response: httpx.Response,
    model: str,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
) -> AsyncGenerator[Dict[str, Any], None]:
    completion_id = generate_completion_id()
    created_time = int(time.time())
    first_chunk = True
    full_content = ""
    full_thinking_content = ""
    tool_calls_from_stream: List[Dict[str, Any]] = []
    try:
        async for event in parse_kiro_stream(response, first_token_timeout):
            if event.type == "content" and event.content:
                full_content += event.content
                delta = {"content": event.content}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
            elif event.type == "thinking" and event.thinking_content:
                full_thinking_content += event.thinking_content
                if FAKE_REASONING_HANDLING == "as_reasoning_content":
                    delta = {"reasoning_content": event.thinking_content}
                elif FAKE_REASONING_HANDLING == "remove":
                    continue
                else:
                    delta = {"content": event.thinking_content}
                if first_chunk:
                    delta["role"] = "assistant"
                    first_chunk = False
                yield {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
            elif event.type == "tool_use" and event.tool_use:
                tool_calls_from_stream.append(event.tool_use)
    finally:
        try:
            await response.aclose()
        except Exception:
            pass

    bracket_tool_calls = parse_bracket_tool_calls(full_content)
    all_tool_calls = deduplicate_tool_calls(tool_calls_from_stream + bracket_tool_calls)
    if all_tool_calls:
        indexed_tool_calls = []
        for idx, tc in enumerate(all_tool_calls):
            func = tc.get("function") or {}
            indexed_tool_calls.append(
                {
                    "index": idx,
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": func.get("name") or "",
                        "arguments": func.get("arguments") or "{}",
                    },
                }
            )
        yield {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": indexed_tool_calls},
                    "finish_reason": None,
                }
            ],
        }

    finish_reason = "tool_calls" if all_tool_calls else "stop"
    yield {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 1,
            "total_tokens": 1,
        },
    }


async def collect_stream_response(
    response: httpx.Response,
    model: str,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
) -> Dict[str, Any]:
    result = await collect_stream_to_result(
        response, first_token_timeout=first_token_timeout
    )
    message: Dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.thinking_content and FAKE_REASONING_HANDLING == "as_reasoning_content":
        message["reasoning_content"] = result.thinking_content
    if result.tool_calls:
        cleaned_tool_calls = []
        for tc in result.tool_calls:
            func = tc.get("function") or {}
            cleaned_tool_calls.append(
                {
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", "{}"),
                    },
                }
            )
        message["tool_calls"] = cleaned_tool_calls

    finish_reason = "tool_calls" if result.tool_calls else "stop"
    return {
        "id": generate_completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 1,
            "total_tokens": 1,
        },
    }


async def stream_with_first_token_retry(
    make_request,
    stream_processor,
    max_retries: int = FIRST_TOKEN_MAX_RETRIES,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
):
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        response: Optional[httpx.Response] = None
        try:
            response = await make_request()
            if response.status_code != 200:
                error_text = await response.aread()
                raise RuntimeError(
                    f"Upstream API error ({response.status_code}): {error_text!r}"
                )
            async for chunk in stream_processor(response):
                yield chunk
            return
        except FirstTokenTimeoutError as exc:
            last_error = exc
            if response:
                try:
                    await response.aclose()
                except Exception:
                    pass
            continue
    if last_error:
        raise last_error
    raise RuntimeError(
        f"Model did not respond within {first_token_timeout}s after {max_retries} attempts."
    )
