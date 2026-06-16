"""
Detailed logger for AI Assistant requests and responses.

Logs comprehensive details of each AI Assistant transaction to help debug
tool calling issues and understand the full request/response flow.
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from rotator_library.utils.resilient_io import (
    safe_write_json,
    safe_log_write,
    safe_mkdir,
)

LOGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "logs"
ASSISTANT_LOGS_DIR = LOGS_DIR / "assistant_logs"

logger = logging.getLogger(__name__)


class AssistantLogger:
    """
    Logs comprehensive details of each AI Assistant conversation turn.

    Creates a directory per conversation turn containing:
    - request.json: The full messages array and tools sent to the LLM
    - streaming_chunks.jsonl: Each streaming chunk received
    - tool_calls.json: Parsed tool calls from the response
    - tool_results.json: Results from tool execution
    - final_response.json: Accumulated response data
    - metadata.json: Summary of the turn
    """

    def __init__(self, session_id: str):
        """
        Initialize the logger for a conversation turn.

        Args:
            session_id: The chat session ID
        """
        self.session_id = session_id
        self.turn_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = ASSISTANT_LOGS_DIR / session_id / f"{timestamp}_{self.turn_id}"
        self._dir_available = safe_mkdir(self.log_dir, logger)

        # Accumulate data for final summary
        self._chunks_count = 0
        self._content_accumulated = ""
        self._reasoning_accumulated = ""
        self._raw_tool_calls: Dict[int, Dict[str, Any]] = {}
        self._model = ""
        self._error: Optional[str] = None

    def _write_json(self, filename: str, data: Dict[str, Any]) -> None:
        """Helper to write data to a JSON file in the log directory."""
        if not self._dir_available:
            self._dir_available = safe_mkdir(self.log_dir, logger)
            if not self._dir_available:
                return

        safe_write_json(
            self.log_dir / filename,
            data,
            logger,
            atomic=False,
            indent=2,
            ensure_ascii=False,
        )

    def log_request(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: str,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        """
        Log the request being sent to the LLM.

        Args:
            messages: The messages array in OpenAI format
            tools: The tools array in OpenAI format
            model: The model being used
            reasoning_effort: Optional reasoning effort level
        """
        self._model = model

        request_data = {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "model": model,
            "reasoning_effort": reasoning_effort,
            "messages_count": len(messages),
            "tools_count": len(tools) if tools else 0,
            "messages": messages,
            "tools": tools,
        }
        self._write_json("request.json", request_data)

        # Also log a summary to the main logger
        logger.info(
            f"[AssistantLogger:{self.turn_id}] Request: model={model}, "
            f"messages={len(messages)}, tools={len(tools) if tools else 0}"
        )

    def log_chunk(
        self,
        chunk: Any,
        parsed_content: Optional[str] = None,
        parsed_reasoning: Optional[str] = None,
        parsed_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Log a streaming chunk.

        Args:
            chunk: The raw chunk (can be string or object)
            parsed_content: Extracted content from the chunk
            parsed_reasoning: Extracted reasoning content from the chunk
            parsed_tool_calls: Extracted tool calls from the chunk
        """
        if not self._dir_available:
            return

        self._chunks_count += 1

        # Accumulate for summary
        if parsed_content:
            self._content_accumulated += parsed_content
        if parsed_reasoning:
            self._reasoning_accumulated += parsed_reasoning

        # Accumulate tool calls
        if parsed_tool_calls:
            for tc in parsed_tool_calls:
                index = tc.get("index", 0)
                if index not in self._raw_tool_calls:
                    self._raw_tool_calls[index] = {
                        "id": "",
                        "name": "",
                        "arguments": "",
                        "chunks": [],
                    }

                # Record this chunk's contribution
                self._raw_tool_calls[index]["chunks"].append(tc)

                # Accumulate
                if tc.get("id"):
                    self._raw_tool_calls[index]["id"] = tc["id"]
                func = tc.get("function", {})
                if func.get("name"):
                    self._raw_tool_calls[index]["name"] = func["name"]
                if func.get("arguments"):
                    self._raw_tool_calls[index]["arguments"] += func["arguments"]

        # Convert chunk to serializable format
        if hasattr(chunk, "model_dump"):
            chunk_data = chunk.model_dump()
        elif hasattr(chunk, "__dict__"):
            chunk_data = str(chunk)
        else:
            chunk_data = chunk

        log_entry = {
            "chunk_number": self._chunks_count,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "raw_chunk": chunk_data,
            "parsed": {
                "content": parsed_content,
                "reasoning": parsed_reasoning,
                "tool_calls": parsed_tool_calls,
            },
        }

        content = json.dumps(log_entry, ensure_ascii=False, default=str) + "\n"
        safe_log_write(self.log_dir / "streaming_chunks.jsonl", content, logger)

    def log_tool_calls_parsed(self, tool_calls: List[Any]) -> None:
        """
        Log the final parsed tool calls.

        Args:
            tool_calls: List of ToolCall objects
        """
        tool_calls_data = {
            "turn_id": self.turn_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "tool_calls_count": len(tool_calls),
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "id_empty": not tc.id,
                    "name_empty": not tc.name,
                }
                for tc in tool_calls
            ],
            "raw_accumulated": {
                str(k): {
                    "id": v["id"],
                    "name": v["name"],
                    "arguments": v["arguments"],
                    "chunk_count": len(v.get("chunks", [])),
                }
                for k, v in self._raw_tool_calls.items()
            },
        }
        self._write_json("tool_calls.json", tool_calls_data)

        # Log summary
        for tc in tool_calls:
            status = []
            if not tc.id:
                status.append("EMPTY_ID")
            if not tc.name:
                status.append("EMPTY_NAME")
            status_str = f" [{', '.join(status)}]" if status else ""
            logger.info(
                f"[AssistantLogger:{self.turn_id}] Tool call: {tc.name or '(empty)'}"
                f"({json.dumps(tc.arguments)[:100]}){status_str}"
            )

    def log_tool_execution(
        self,
        tool_call_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        result_success: bool,
        result_message: str,
        result_data: Optional[Dict[str, Any]] = None,
        error_code: Optional[str] = None,
    ) -> None:
        """
        Log a tool execution result.

        Args:
            tool_call_id: The tool call ID
            tool_name: The tool name
            arguments: The arguments passed to the tool
            result_success: Whether the tool succeeded
            result_message: The result message
            result_data: Optional result data
            error_code: Optional error code if failed
        """
        if not self._dir_available:
            return

        log_entry = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "result": {
                "success": result_success,
                "message": result_message,
                "data": result_data,
                "error_code": error_code,
            },
        }

        content = json.dumps(log_entry, ensure_ascii=False, default=str) + "\n"
        safe_log_write(self.log_dir / "tool_results.jsonl", content, logger)

        # Log summary
        status = "SUCCESS" if result_success else f"FAILED ({error_code})"
        logger.info(
            f"[AssistantLogger:{self.turn_id}] Tool result: {tool_name} -> {status}: {result_message[:100]}"
        )

    def log_error(self, error: str) -> None:
        """Log an error that occurred during the turn."""
        self._error = error
        logger.error(f"[AssistantLogger:{self.turn_id}] Error: {error}")

        error_data = {
            "turn_id": self.turn_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "error": error,
        }
        self._write_json("error.json", error_data)

    def log_completion(self, finish_reason: Optional[str] = None) -> None:
        """
        Log the completion of this turn and write final summary.

        Args:
            finish_reason: The finish reason from the LLM (if available)
        """
        end_time = time.time()
        duration_ms = (end_time - self.start_time) * 1000

        # Parse accumulated tool call arguments for summary
        parsed_tool_calls = []
        for index in sorted(self._raw_tool_calls.keys()):
            tc_data = self._raw_tool_calls[index]
            try:
                args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                args = {"_parse_error": tc_data["arguments"]}

            parsed_tool_calls.append(
                {
                    "id": tc_data["id"],
                    "name": tc_data["name"],
                    "arguments": args,
                }
            )

        final_response = {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "duration_ms": round(duration_ms),
            "model": self._model,
            "chunks_received": self._chunks_count,
            "finish_reason": finish_reason,
            "content": self._content_accumulated,
            "reasoning_content": self._reasoning_accumulated
            if self._reasoning_accumulated
            else None,
            "tool_calls": parsed_tool_calls if parsed_tool_calls else None,
            "error": self._error,
        }
        self._write_json("final_response.json", final_response)

        # Metadata summary
        metadata = {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "duration_ms": round(duration_ms),
            "model": self._model,
            "chunks_received": self._chunks_count,
            "content_length": len(self._content_accumulated),
            "reasoning_length": len(self._reasoning_accumulated),
            "tool_calls_count": len(parsed_tool_calls),
            "tool_calls_summary": [
                {"name": tc["name"], "id_present": bool(tc["id"])}
                for tc in parsed_tool_calls
            ],
            "finish_reason": finish_reason,
            "had_error": self._error is not None,
        }
        self._write_json("metadata.json", metadata)

        logger.info(
            f"[AssistantLogger:{self.turn_id}] Completed: "
            f"duration={round(duration_ms)}ms, chunks={self._chunks_count}, "
            f"content_len={len(self._content_accumulated)}, "
            f"tool_calls={len(parsed_tool_calls)}, "
            f"finish_reason={finish_reason}"
        )

    def log_messages_sent(self, messages: List[Dict[str, Any]]) -> None:
        """
        Log the messages being sent in a continuation request.

        Useful for debugging the agentic loop.

        Args:
            messages: The full messages array being sent
        """
        messages_data = {
            "turn_id": self.turn_id,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "context": "agentic_loop_continuation",
            "messages_count": len(messages),
            "messages": messages,
        }

        # Write to a separate file to track continuations
        content = json.dumps(messages_data, ensure_ascii=False, default=str) + "\n"
        safe_log_write(self.log_dir / "continuation_requests.jsonl", content, logger)

        logger.info(
            f"[AssistantLogger:{self.turn_id}] Continuation request: "
            f"messages={len(messages)}"
        )
